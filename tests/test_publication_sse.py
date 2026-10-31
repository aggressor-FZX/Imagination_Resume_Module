"""Offline /analyze SSE containment: final approval never approves old tokens."""

import asyncio
import importlib
import json
import socket

import pytest
from fastapi.testclient import TestClient


APPROVED_MARKDOWN = (
    "## Professional Experience\n"
    "**Synthetic Workshop**\n"
    "- Maintained the workshop inventory and documented repair procedures.\n"
)
RAW_TOKENS = (
    ("draft", "Increased throughput by "),
    ("draft", "87%"),
    ("rewrite", "UNAPPROVED_REWRITE"),
    ("reasoning", "PRIVATE_REASONING"),
)


@pytest.fixture
def sse_client(monkeypatch):
    # Import the real route without reading dotenv/secret files or starting
    # lifespan-managed model clients. All HTTP below is in-process ASGI.
    import dotenv
    from pydantic_settings.sources import DotEnvSettingsSource

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(DotEnvSettingsSource, "_read_env_files", lambda self: {})

    def no_network(*args, **kwargs):
        raise AssertionError("Publication SSE tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)

    app_module = importlib.import_module("app")
    monkeypatch.setitem(
        app_module.app.dependency_overrides,
        app_module.get_api_key,
        lambda: "synthetic-test-auth",
    )
    client = TestClient(app_module.app)
    yield app_module, client
    client.close()


def _stub_analysis(monkeypatch, app_module, result):
    from stream_context import get_token_callback

    async def analyze_stub(payload, request, api_key):
        callback = get_token_callback()
        if callback is not None:
            # Match worker-thread token delivery, not just a synchronous mock.
            def emit_intermediate_tokens():
                for field, delta in RAW_TOKENS:
                    callback(field, delta)

            await asyncio.to_thread(emit_intermediate_tokens)
        return result

    monkeypatch.setattr(app_module, "_analyze_resume_impl", analyze_stub)


def _post_sse(client):
    response = client.post(
        "/analyze",
        json={
            "resume_text": APPROVED_MARKDOWN,
            "job_ad": "Maintain workshop inventory and repair documentation.",
        },
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = []
    for block in response.text.split("\n\n"):
        if not block.strip():
            continue
        lines = block.splitlines()
        name = next(line[6:].strip() for line in lines if line.startswith("event:"))
        data = next(line[5:].strip() for line in lines if line.startswith("data:"))
        events.append((name, json.loads(data)))
    assert events[-1][0] == "result"
    return response, events


@pytest.mark.parametrize("response_kind", ["dict", "model"])
def test_approved_final_never_releases_unsafe_intermediate_tokens(
    sse_client, monkeypatch, response_kind
):
    app_module, client = sse_client
    final = {
        "processing_status": "completed",
        "requires_review": False,
        "metrics_support": {
            "verdict": "PASS (tier A only)",
            "checked": True,
            "unsupported_metrics": [],
        },
        "final_written_section_markdown": APPROVED_MARKDOWN,
        "final_written_section": APPROVED_MARKDOWN,
        "rewritten_resume": APPROVED_MARKDOWN,
        "processing_time_seconds": 0.0,
    }
    result = app_module.AnalysisResponse(**final) if response_kind == "model" else final
    _stub_analysis(monkeypatch, app_module, result)

    response, events = _post_sse(client)

    assert "87%" not in response.text, "Final approval leaked an invented draft metric"
    assert "UNAPPROVED_REWRITE" not in response.text
    assert "PRIVATE_REASONING" not in response.text
    tokens = [data for name, data in events if name == "token"]
    assert tokens == [{"field": "rewrite", "delta": APPROVED_MARKDOWN}]
    assert not any(name == "withheld" for name, _ in events)
    published = events[-1][1]
    assert published["final_written_section_markdown"] == APPROVED_MARKDOWN
    assert published["processing_status"] == "completed"
    assert published["requires_review"] is False


@pytest.mark.parametrize("response_kind", ["dict", "model"])
def test_blocked_final_emits_zero_tokens(sse_client, monkeypatch, response_kind):
    app_module, client = sse_client
    final = {
        "processing_status": "needs_review",
        "requires_review": True,
        "review_reasons": ["unsupported_claims"],
        "review_message": "Generated content requires review.",
        "final_written_section_markdown": None,
        "final_written_section": None,
        "rewritten_resume": None,
        "processing_time_seconds": 0.0,
    }
    result = app_module.AnalysisResponse(**final) if response_kind == "model" else final
    _stub_analysis(monkeypatch, app_module, result)

    response, events = _post_sse(client)

    assert not any(name == "token" for name, _ in events)
    assert "87%" not in response.text
    assert "UNAPPROVED_REWRITE" not in response.text
    assert "PRIVATE_REASONING" not in response.text
    withheld = [data for name, data in events if name == "withheld"]
    assert withheld == [{
        "reasons": ["unsupported_claims"],
        "message": "Generated content requires review.",
    }]
    blocked = events[-1][1]
    assert blocked["processing_status"] == "needs_review"
    assert blocked["requires_review"] is True
    for field in ("final_written_section_markdown", "final_written_section", "rewritten_resume"):
        assert blocked[field] is None