"""Offline sentinel coverage for untrusted model IDs and response statistics."""
import asyncio
import json
import logging
import socket

import pytest
import requests

from llm_client_adapter import LLMClientAdapter
from openrouter_safe_client import OpenRouterSafeClient


REQUESTED = "SYNTHETIC_REQUESTED_PRIVATE_MODEL"
FALLBACK = "SYNTHETIC_FALLBACK_PRIVATE_MODEL"
USED = "SYNTHETIC_PROVIDER_PRIVATE_MODEL"
STAT = "SYNTHETIC_PROVIDER_PRIVATE_STAT"
STAT_NUMBER = 9182736451


@pytest.fixture(autouse=True)
def offline(monkeypatch, caplog):
    def blocked(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.delenv("OPENROUTER_API_KEY_1", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_2", raising=False)
    caplog.set_level(logging.DEBUG)


def assert_private(capsys, caplog):
    captured = capsys.readouterr()
    logs = captured.out + captured.err + caplog.text
    for sentinel in (REQUESTED, FALLBACK, USED, STAT, str(STAT_NUMBER)):
        assert sentinel not in logs
    assert not any(record.exc_info or record.stack_info for record in caplog.records)


@pytest.mark.parametrize("stream", [False, True])
def test_availability_fallback_and_success_are_private(monkeypatch, capsys, caplog, stream):
    class ModelResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": FALLBACK}]}

    class CompletionResponse:
        status_code = 200

        def json(self):
            return {"model": USED, "choices": [{"message": {"content": "safe response"}}]}

        def iter_lines(self, decode_unicode=True):
            yield 'data: {"choices":[{"delta":{"content":"safe response"}}]}'
            yield "data: [DONE]"

    monkeypatch.setattr("openrouter_safe_client.requests.get", lambda *a, **kw: ModelResponse())
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs["json"]["model"])
        return CompletionResponse()

    monkeypatch.setattr("openrouter_safe_client.requests.post", post)
    client = OpenRouterSafeClient("synthetic-key")
    tokens = []
    result = client.call_model([{"role": "user", "content": "hello"}], model=REQUESTED,
                               fallback_models=[FALLBACK], stream=stream, on_token=tokens.append)
    assert result["success"] and result["model_used"] == FALLBACK
    assert result["content"] == "safe response"
    assert calls == [FALLBACK]
    assert tokens == (["safe response"] if stream else [])
    assert_private(capsys, caplog)


def test_timeout_does_not_log_or_return_model(monkeypatch, capsys, caplog):
    client = OpenRouterSafeClient("synthetic-key")
    monkeypatch.setattr(client, "_get_available_models", lambda **kw: {REQUESTED: {}})

    def timeout(*args, **kwargs):
        raise requests.exceptions.Timeout(REQUESTED)

    monkeypatch.setattr("openrouter_safe_client.requests.post", timeout)
    result = client.call_model([{"role": "user", "content": "hello"}], model=REQUESTED,
                               fallback_models=[])
    assert not result["success"] and "timed out" in result["error"]
    assert REQUESTED not in result["error"]
    assert_private(capsys, caplog)


def test_adapter_success_keeps_provider_statistics_and_model_ids_out_of_logs(monkeypatch, capsys, caplog):
    adapter = LLMClientAdapter("synthetic-key")

    def call(**kwargs):
        assert kwargs["model"] == REQUESTED and kwargs["fallback_models"] == [FALLBACK]
        return {"success": True, "model_used": USED, "data": {
            "choices": [{"message": {"content": "safe response"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": STAT_NUMBER},
        }}

    monkeypatch.setattr(adapter.client, "call_model", call)
    response = asyncio.run(adapter.call_llm_async("system", "user", REQUESTED,
                                                   fallback_models=[FALLBACK]))
    assert response == "safe response"
    stats = adapter.get_usage_stats()
    assert stats["calls"][0]["model"] == REQUESTED
    assert stats["calls"][0]["model_used"] == USED
    assert stats["total_tokens"] == STAT_NUMBER
    assert_private(capsys, caplog)


def test_adapter_failure_with_untrusted_requested_and_used_models(monkeypatch, capsys, caplog):
    adapter = LLMClientAdapter("synthetic-key")
    monkeypatch.setattr(adapter.client, "call_model", lambda **kw: {
        "success": False, "model_used": USED, "error": STAT})
    result = asyncio.run(adapter.call_llm_async("system", "user", REQUESTED,
                         fallback_models=[FALLBACK], response_format={"type": "json_object"}))
    assert json.loads(result)["error"] == "llm_call_failed"
    assert_private(capsys, caplog)