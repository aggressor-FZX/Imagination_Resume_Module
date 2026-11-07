"""Offline coverage for the OpenRouter streaming transport contract."""
import json
import socket

import pytest
import requests

from openrouter_safe_client import OpenRouterSafeClient, _iter_sse_lines_utf8


MESSAGES = [{"role": "user", "content": "test"}]
JSON_FORMAT = {"type": "json_object"}


def event(choice=None, **fields):
    return "data: " + json.dumps({"choices": [] if choice is None else [choice], **fields})


def content(text, **extra):
    return event({"delta": {"content": text, **extra}, "finish_reason": None})


def finish(reason="stop"):
    return event({"delta": {}, "finish_reason": reason})


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.delenv("OPENROUTER_API_KEY_1", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_2", raising=False)


@pytest.fixture
def transport(monkeypatch):
    client = OpenRouterSafeClient("test-key")
    monkeypatch.setattr(client, "_is_model_available", lambda model: True)
    responses = []
    calls = []

    class Response:
        status_code = 200

        def __init__(self, lines):
            self.lines = lines

        def iter_lines(self, decode_unicode=False):
            assert decode_unicode is False
            yield from self.lines

    def post(*args, **kwargs):
        calls.append(kwargs["json"])
        return Response(responses.pop(0))

    monkeypatch.setattr("openrouter_safe_client.requests.post", post)

    def run(first, second=None, callback=None, response_format=JSON_FORMAT):
        responses.append(first)
        if second is not None:
            responses.append(second)
        result = client.call_model(MESSAGES, model="first", fallback_models=["second"],
                                   stream=True, on_token=callback,
                                   response_format=response_format)
        return result, calls

    return run


def test_stop_preserves_finish_usage_and_callback_order(transport):
    seen = []

    def lines():
        yield content('{"name":"caf')
        assert seen == ['{"name":"caf']  # callback before reading the next event
        yield 'data: {"choices":[{"delta":{"content":"é"},"finish_reason":null}]}'.encode()
        yield content('"}')
        yield finish()
        yield event(usage={"total_tokens": 7})
        yield "data: [DONE]"
        raise AssertionError("must stop reading after DONE")

    result, calls = transport(lines(), callback=seen.append)
    assert result["success"] and result["content"] == '{"name":"café"}'
    assert seen == ['{"name":"caf', 'é', '"}']
    assert result["data"]["choices"][0]["finish_reason"] == "stop"
    assert result["data"]["usage"] == {"total_tokens": 7}
    assert calls[0]["response_format"] == JSON_FORMAT


@pytest.mark.parametrize("lines", [
    [content('{"ok":true}'), finish("length")],
    [content('{"ok":true}'), "data: [DONE]"],
    [content('{"ok":'), finish()],
    [content('[1,2]'), finish()],
    [content('{"n":NaN}'), finish()],
    [content('{"ok":true}'), 'data: {broken', finish()],
    [content('{"ok":true}'), event(error={"message": "PRIVATE"})],
    [event({"delta": {"reasoning": '{"ok":true}'}, "finish_reason": "stop"})],
])
def test_invalid_stream_fails_without_callback_and_can_retry(transport, lines):
    result, calls = transport(lines, [content('{"fallback":true}'), finish()])
    assert result["success"] and result["content"] == '{"fallback":true}'
    assert [call["model"] for call in calls] == ["first", "second"]

@pytest.mark.parametrize("tail,error", [
    ([finish("length")], "Incomplete"),
    (["data: [DONE]"], "Missing"),
    ([finish()], "Invalid stream JSON"),
    (['data: {broken'], "Malformed"),
    ([event(error={"message": "PRIVATE"})], "Provider stream error"),
])
def test_visible_tokens_prevent_fallback(transport, tail, error):
    seen = []
    text = '{"incomplete":' if error == "Invalid stream JSON" else '{"ok":true}'
    result, calls = transport([content(text), *tail], [content('{"fallback":true}'), finish()],
                              callback=seen.append)
    assert not result["success"] and error in result["error"]
    assert "PRIVATE" not in result["error"]
    assert seen == [text] and [call["model"] for call in calls] == ["first"]


def test_callback_exception_may_have_published_and_blocks_retry(transport):
    seen = []

    def callback(token):
        seen.append(token)
        raise RuntimeError("callback failure")

    result, calls = transport([content("partial"), finish("length")],
                              [content("replacement"), finish()], callback=callback,
                              response_format=None)
    assert not result["success"] and seen == ["partial"]
    assert len(calls) == 1


def test_callback_exception_does_not_break_valid_stream(transport):
    def callback(token):
        raise RuntimeError("callback failure")

    result, calls = transport([content('{"ok":true}'), finish()], callback=callback)
    assert result["success"] and len(calls) == 1


@pytest.mark.parametrize("after_token", [False, True])
def test_stream_read_exception_retries_only_before_callback(transport, after_token):
    seen = []

    def lines():
        if after_token:
            yield content("partial")
        raise requests.exceptions.Timeout("PRIVATE")

    result, calls = transport(lines(), [content("fallback"), finish()], callback=seen.append,
                              response_format=None)
    if after_token:
        assert not result["success"] and "timed out" in result["error"]
        assert seen == ["partial"] and len(calls) == 1
    else:
        assert result["success"] and seen == ["fallback"] and len(calls) == 2


def test_reasoning_not_emitted_as_content(transport):
    seen = []
    result, calls = transport([event({"delta": {"reasoning": "private thought"}}),
                               content("answer", reasoning="more thought"), finish()],
                              callback=seen.append, response_format=None)
    assert result["success"] and result["content"] == "answer" and seen == ["answer"]


def test_all_attempts_invalid_returns_explicit_failure(transport):
    invalid = [content('{"partial":'), finish()]
    result, calls = transport(invalid, invalid)
    assert not result["success"] and result["model_used"] is None
    assert "Invalid stream JSON object" in result["error"] and len(calls) == 2


def test_invalid_utf8_cannot_be_accepted_as_json(transport):
    lines = [b'data: {"choices":[{"delta":{"content":"\xff"}}]}', finish()]
    result, calls = transport(lines, [content('{"fallback":true}'), finish()])
    assert result["success"] and result["content"] == '{"fallback":true}'
    assert len(calls) == 2


def test_utf8_character_split_across_raw_chunks():
    class Response:
        iter_lines = requests.Response.iter_lines

        def iter_content(self, chunk_size, decode_unicode):
            yield b'data: caf\xc3'
            yield b'\xa9\n\ndata: [DONE]\n'

    assert list(_iter_sse_lines_utf8(Response())) == ["data: café", "", "data: [DONE]"]


@pytest.mark.parametrize("lines,error", [
    ([content('{"ok":true}'), finish("length")], "Incomplete"),
    ([content('{"ok":true}'), "data: [DONE]"], "Missing"),
    ([event({"delta": {"reasoning": "private"}, "finish_reason": "stop"})], "Empty"),
    ([content('{"ok":true}'), event(error={"message": "PRIVATE"})], "Provider stream error"),
])
def test_terminal_failure_is_explicit(transport, lines, error):
    result, calls = transport(lines, lines)
    assert not result["success"] and error in result["error"]
    assert "PRIVATE" not in result["error"] and len(calls) == 2