"""Synthetic error-path tests; all provider traffic is replaced by local doubles."""
import asyncio
import json

import pytest

from llm_client_adapter import LLMClientAdapter
from openrouter_safe_client import OpenRouterSafeClient
from stages.drafter import Drafter
from stages.researcher import Researcher


MARKER = "SYNTHETIC_PRIVATE_CAREER_MARKER"


@pytest.mark.parametrize("stream,status", [(False, 400), (False, 404), (False, 500),
                                           (True, 400), (True, 500)])
def test_http_failure_does_not_return_or_print_provider_body(monkeypatch, capsys, stream, status):
    class Response:
        status_code = status
        text = MARKER

        def json(self):
            return {"error": {"message": MARKER}}

    monkeypatch.setattr("openrouter_safe_client.requests.post", lambda *a, **kw: Response())
    client = OpenRouterSafeClient("synthetic-key")
    monkeypatch.setattr(client, "_is_model_available", lambda *a, **kw: True)
    result = client.call_model([{"role": "user", "content": MARKER}], model="test/model",
                               fallback_models=[], stream=stream)
    assert not result["success"]
    assert MARKER not in repr(result) + capsys.readouterr().out
    assert f"HTTP {status}" in result["error"]


def test_malformed_envelope_and_exception_do_not_leak(monkeypatch, capsys):
    class Response:
        status_code = 200

        def json(self):
            return {"detail": MARKER}

    client = OpenRouterSafeClient("synthetic-key")
    monkeypatch.setattr(client, "_is_model_available", lambda *a, **kw: True)
    monkeypatch.setattr("openrouter_safe_client.requests.post", lambda *a, **kw: Response())
    messages = [{"role": "user", "content": MARKER}]
    malformed = client.call_model(messages, model="test/model", fallback_models=[])

    def fail(*args, **kwargs):
        raise RuntimeError(MARKER)

    monkeypatch.setattr("openrouter_safe_client.requests.post", fail)
    failure = client.call_model(messages, model="test/model", fallback_models=[])
    assert not malformed["success"] and not failure["success"]
    assert MARKER not in repr((malformed, failure)) + capsys.readouterr().out


@pytest.mark.parametrize("throws", [False, True])
def test_adapter_sanitizes_untrusted_client_errors(monkeypatch, caplog, throws):
    adapter = LLMClientAdapter("synthetic-key")

    def call(**kwargs):
        if throws:
            raise RuntimeError(MARKER)
        return {"success": False, "error": MARKER}

    monkeypatch.setattr(adapter.client, "call_model", call)
    response = asyncio.run(adapter.call_llm_async("system", MARKER, "test/model",
                                                  response_format={"type": "json_object"}))
    assert json.loads(response) == {"error": "llm_call_failed", "fallback": True}
    assert MARKER not in response + caplog.text


@pytest.mark.parametrize("stage", ["researcher", "drafter"])
def test_stage_parse_failures_do_not_log_raw_response(caplog, stage):
    if stage == "researcher":
        result = Researcher(None)._parse_response(MARKER)
    else:
        result = Drafter(None)._parse_response(MARKER, [])
    assert result["fallback_used"] is True
    assert MARKER not in caplog.text