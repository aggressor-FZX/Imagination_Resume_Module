"""Provider reasoning is not an authoritative final answer."""
import asyncio
import json

import pytest

from llm_client_adapter import LLMClientAdapter
from openrouter_safe_client import OpenRouterSafeClient
from orchestrator import research_used_fallback
from stages.researcher import Researcher


class FakeClient:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason

    def call_model(self, **kwargs):
        return {"success": True, "data": {"choices": [{"message": self.message,
                "finish_reason": self.finish_reason}], "usage": {}}}


def adapter_call(message, finish_reason="stop", response_format=None):
    adapter = LLMClientAdapter.__new__(LLMClientAdapter)
    adapter.client = FakeClient(message, finish_reason)
    adapter.reset_usage_stats()
    return asyncio.run(adapter.call_llm_async(
        system_prompt="s", user_prompt="u", model="test",
        response_format=response_format))


@pytest.mark.parametrize("reasoning", ["prose", '{"skill_bridges": []}'])
def test_reasoning_is_never_promoted_to_answer(reasoning):
    out = adapter_call({"content": "", "reasoning": reasoning},
                       response_format={"type": "json_object"})
    assert json.loads(out)["error"] == "llm_call_failed"
    assert research_used_fallback(Researcher(None)._parse_response(out))


def test_reasoning_is_not_promoted_for_prose_either():
    assert adapter_call({"content": "", "reasoning": "thinking"}) == ""


def test_length_is_failure_even_with_content():
    out = adapter_call({"content": '{"skill_bridges": []}'}, "length",
                       {"type": "json_object"})
    assert json.loads(out)["fallback"] is True


def test_valid_zero_findings_are_not_failure():
    out = adapter_call({"content": '{"skill_bridges": [], "domain_vocab": []}'},
                       response_format={"type": "json_object"})
    assert not research_used_fallback(Researcher(None)._parse_response(out))


def test_provider_failure_and_unrelated_object_are_not_success():
    for text in ('{"error": "llm_call_failed", "fallback": true}', '{"unexpected": 1}',
                 '{"skill_bridges": "not a list"}'):
        assert research_used_fallback(Researcher(None)._parse_response(text))


@pytest.mark.parametrize("content,reason", [('', 'length'), ('', 'stop'),
    ('{"skill_bridges":', 'stop'), ('{"skill_bridges": []}', 'length')])
def test_transport_retries_unusable_provider_response(monkeypatch, content, reason):
    client = OpenRouterSafeClient("test-key")
    monkeypatch.setattr(client, "_is_model_available", lambda _: True)
    calls = []
    class Response:
        status_code = 200
        def json(self):
            return {"choices": [{"finish_reason": reason,
                "message": {"content": content, "reasoning": '{"skill_bridges": []}'}}]}
    def post(*args, **kwargs):
        calls.append(kwargs["json"]["model"])
        return Response()
    monkeypatch.setattr("openrouter_safe_client.requests.post", post)
    result = client.call_model([{"role": "user", "content": "test"}], model="first",
                               fallback_models=["second"], response_format={"type": "json_object"})
    assert not result["success"]
    assert calls == ["first", "second"]
