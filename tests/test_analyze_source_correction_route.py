"""Offline normal-route integration; these synthetic reviews do not prove semantic accuracy."""
import asyncio
import copy
import json
import socket
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import app
import correction_provider
from approved_resume import export_payload
from source_correction import CorrectionResult
from stream_context import get_token_callback
from tests.test_correction_integration_offline import (
    DRAFT, FRAGMENT, SECOND_FRAGMENT, SOURCE, FakeSession,
)


@pytest.fixture
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("external network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(app.settings, "openrouter_api_key_1", "offline-key")
    monkeypatch.setattr(app.settings, "openrouter_api_key_2", None)
    monkeypatch.setenv("PUBLICATION_RECEIPT_SECRET", "offline-route-secret-1234567890")
    monkeypatch.setitem(app.app.dependency_overrides, app.get_api_key, lambda: "offline")
    session = FakeSession()
    monkeypatch.setattr(app.app.state, "http_session", session, raising=False)
    monkeypatch.setattr(app.imaginator_flow_module, "run_generation_async", forbidden)
    monkeypatch.setattr(app, "run_criticism_async", forbidden)
    monkeypatch.setattr(app, "run_synthesis_async", forbidden)
    with TestClient(app.app) as client:
        # Startup creates a real session; replace it only after it has been closed.
        client.portal.call(app.app.state.http_session.close)
        app.app.state.http_session = session
        yield client, session
    app.app.dependency_overrides.clear()


@pytest.fixture
def synthetic_disclosure_bypass(offline, monkeypatch):
    """Mock the privacy boundary for fake sources and sessions, not production policy.

    This is opt-in; the shared offline fixture retains the real privacy default.
    """
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available", lambda: True)


def pipeline(monkeypatch, *, draft=DRAFT, status="completed", extra=None):
    async def run(**kwargs):
        callback = get_token_callback()
        if callback:
            callback("rewrite", "PRIVATE_RAW_STREAM_TOKEN")
        return {
            "pipeline_status": status,
            "final_written_section_markdown": draft,
            "final_written_section": "PRIVATE_STALE_DRAFT",
            "rewritten_resume": "PRIVATE_STALE_DRAFT",
            "experiences": [{"snippet": "PRIVATE_STALE_DRAFT"}],
            "editorial_notes": "PRIVATE_EDITOR_NOTES",
            "correction_commentary": "PRIVATE_PROVIDER_COMMENTARY",
            "final_written_section_provenance": [{"claim": "PRIVATE_CLAIM"}],
            "metrics_support": {"verdict": "FAIL", "checked": True},
            "run_metrics": {"calls": [{"prompt_tokens": 3, "completion_tokens": 2,
                                       "cost_usd": 0.01, "raw": "PRIVATE_RAW"}],
                            "total_prompt_tokens": 3, "total_completion_tokens": 2,
                            "total_tokens": 5, "estimated_cost_usd": 0.01,
                            "failures": []},
            **(extra or {}),
        }
    monkeypatch.setattr(app, "run_new_pipeline_async", run)


def post(client, *, source=SOURCE, sse=False, docs=None):
    headers = {"X-Cogito-Owner-Id": "offline-owner"}
    if sse:
        headers["Accept"] = "text/event-stream"
    body = {"resume_text": source, "job_ad": "Synthetic equipment maintenance role"}
    if docs is not None:
        body["source_documents"] = docs
    return client.post("/analyze", headers=headers, json=body)


def result_from_sse(response):
    blocks = [block for block in response.text.split("\n\n") if block]
    result = next(block for block in blocks if block.startswith("event: result\n"))
    return blocks, json.loads(result.split("data: ", 1)[1])


def test_corrected_final_is_only_authority_for_aliases_receipt_and_export(offline, monkeypatch, synthetic_disclosure_bypass):
    client, session = offline
    pipeline(monkeypatch)
    result = post(client).json()
    assert result["processing_status"] == "completed", result
    assert result["correction_status"] == "approved"
    assert "revised 1 span(s) flagged by the reviewer" in result["correction_commentary"]
    assert "independent review is required" in result["correction_commentary"]
    for field in ("final_written_section_markdown", "final_written_section", "rewritten_resume"):
        assert result[field] == SOURCE
    assert FRAGMENT not in result["final_written_section_markdown"] and "30%" in SOURCE
    assert result["approved_export_payload"] == export_payload(SOURCE)
    assert result["experiences"][0]["title_line"] == result["approved_export_payload"]["experience"][0]["title"]
    assert result["final_written_section_provenance"] == []
    assert "PRIVATE_" not in json.dumps(result)
    assert result["metrics_support"]["verdict"] == "PASS (tier A only)"
    assert [call["phase"] for call in result["run_metrics"]["calls"] if call["stage"] == "source_correction"] == [
        "source_inventory", "initial_review", "repair", "post_patch_review"]
    assert result["run_metrics"]["total_tokens"] == 61
    assert len(session.requests) == 4
    assert all(req["json"]["provider"] == {"allow_fallbacks": False} and not req["allow_redirects"]
               for req in session.requests)

    renderer = Mock(return_value={"filename": "synthetic", "pdf_base64": "AA", "docx_base64": "AA"})
    monkeypatch.setattr("resume_builder.build_resume_from_export.build_resume_from_payload_base64", renderer)
    body = {**result["approved_export_payload"], "final_written_section_markdown": SOURCE,
            "publication_receipt": result["publication_receipt"]}
    headers = {"X-Cogito-Owner-Id": "offline-owner"}
    assert client.post("/resume/export", json=body, headers=headers).status_code == 200
    renderer.reset_mock()
    altered = copy.deepcopy(body)
    altered["experience"][0]["bullets"][0] += " invented"
    assert client.post("/resume/export", json=altered, headers=headers).status_code == 409
    renderer.assert_not_called()
    altered = {**body, "final_written_section_markdown": SOURCE + " invented"}
    assert client.post("/resume/export", json=altered, headers=headers).status_code == 409


@pytest.mark.parametrize("sse", [False, True])
@pytest.mark.parametrize("mode", ["timeout", "malformed", "finish"])
def test_provider_failures_withhold_without_retry_or_private_leak(offline, monkeypatch, synthetic_disclosure_bypass, sse, mode):
    client, _ = offline
    session = FakeSession(timeout=mode == "timeout", malformed=mode == "malformed",
                          finish="length" if mode == "finish" else "stop")
    app.app.state.http_session = session
    pipeline(monkeypatch)
    response = post(client, sse=sse)
    if sse:
        blocks, result = result_from_sse(response)
        assert not any(block.startswith("event: token") for block in blocks)
        assert any(block.startswith("event: withheld") for block in blocks)
    else:
        result = response.json()
    assert result["requires_review"] and result["publication_receipt"] is None
    assert result["final_written_section_markdown"] is None
    assert "PRIVATE_" not in response.text and FRAGMENT not in response.text
    assert "private provider error" not in response.text
    assert len(session.requests) == 1
    failure = result["run_metrics"]["failures"][-1]
    assert failure["usage_unknown"] is (mode == "timeout")
    assert result["run_metrics"]["total_tokens"] == (5 if mode == "timeout" else 19)


def test_sse_only_releases_final_approved_markdown(offline, monkeypatch, synthetic_disclosure_bypass):
    client, _ = offline
    pipeline(monkeypatch)
    response = post(client, sse=True)
    blocks, result = result_from_sse(response)
    tokens = [json.loads(block.split("data: ", 1)[1]) for block in blocks if block.startswith("event: token\n")]
    assert tokens == [{"field": "rewrite", "delta": SOURCE}]
    assert result["final_written_section_markdown"] == SOURCE
    assert "PRIVATE_" not in response.text and FRAGMENT not in response.text


def test_two_repairs_are_bounded_to_six_provider_calls(offline, monkeypatch, synthetic_disclosure_bypass):
    client, _ = offline
    session = FakeSession(repairs=[SECOND_FRAGMENT, ""])
    app.app.state.http_session = session
    pipeline(monkeypatch)
    result = post(client).json()
    assert result["final_written_section_markdown"] == SOURCE
    calls = [c["phase"] for c in result["run_metrics"]["calls"] if c["stage"] == "source_correction"]
    assert calls == ["source_inventory", "initial_review", "repair", "post_patch_review", "repair", "post_patch_review"]
    assert len(session.requests) == 6 and result["run_metrics"]["total_tokens"] == 89


@pytest.mark.parametrize("status,extra,draft", [
    ("failed", {}, DRAFT), ("completed_with_fallback", {}, DRAFT),
    ("unknown", {}, DRAFT), ("completed", {"fallback_used": True}, DRAFT),
    ("completed", {"finish_reason": "length"}, DRAFT),
    ("completed", {}, SOURCE), ("completed", {}, "## Experience\nshort"),
    ("completed", {}, None),
])
def test_failed_fallback_or_invalid_pipeline_never_calls_correction(offline, monkeypatch, status, extra, draft):
    client, session = offline
    pipeline(monkeypatch, status=status, extra=extra, draft=draft)
    result = post(client).json()
    assert result["requires_review"] and result["final_written_section_markdown"] is None
    assert result["publication_receipt"] is None and not session.requests
    assert "PRIVATE_" not in json.dumps(result)


def test_mismatched_sources_never_call_pipeline(offline, monkeypatch):
    client, session = offline
    pipeline(monkeypatch)
    result = post(client, docs=[{"source_id": "a", "text": "different", "kind": "candidate_document"}]).json()
    assert result["review_reason"] == "source_documents_mismatch"
    assert not session.requests


def test_explicit_source_document_is_used_for_inventory(offline, monkeypatch, synthetic_disclosure_bypass):
    client, session = offline
    pipeline(monkeypatch)
    docs = [{"source_id": "candidate-one", "text": SOURCE, "kind": "candidate_document"}]
    result = post(client, docs=docs).json()
    assert result["final_written_section_markdown"] == SOURCE
    inventory = json.loads(session.requests[0]["json"]["messages"][1]["content"])
    # The designated document is what gets inventoried, under its own id, and the
    # segments reconstruct its text exactly rather than a re-derived copy.
    assert {segment["source_id"] for segment in inventory["segments"]} == {"candidate-one"}
    assert "".join(segment["text"] for segment in inventory["segments"]) == SOURCE


def test_missing_source_representation_withholds_without_repair(offline, monkeypatch, synthetic_disclosure_bypass):
    client, _ = offline
    session = FakeSession(missing_representation=True)
    app.app.state.http_session = session
    pipeline(monkeypatch)
    result = post(client).json()
    assert result["correction_reason"] == "source_facts_missing"
    assert result["final_written_section"] is None and len(session.requests) == 2
    assert "PRIVATE_" not in json.dumps(result)


def test_pipeline_exception_never_triggers_provider_and_usage_is_unknown(offline, monkeypatch):
    client, session = offline
    async def broken(**kwargs):
        raise RuntimeError("PRIVATE_PIPELINE_EXCEPTION")
    monkeypatch.setattr(app, "run_new_pipeline_async", broken)
    response = post(client)
    result = response.json()
    assert result["requires_review"] and not session.requests
    assert result["run_metrics"]["failures"][-1]["usage_unknown"] is True
    assert "PRIVATE_" not in response.text


def test_overall_deadline_cancels_inflight_provider_without_retry(offline, monkeypatch, synthetic_disclosure_bypass):
    client, _ = offline
    pipeline(monkeypatch)
    original_correct = app.correct
    class HangingSession:
        calls = 0
        cancelled = False
        async def close(self):
            pass
        def post(self, *args, **kwargs):
            self.calls += 1
            session = self
            class Response:
                async def __aenter__(self):
                    try:
                        await asyncio.Event().wait()
                    finally:
                        session.cancelled = True
                async def __aexit__(self, *args):
                    pass
            return Response()
    session = HangingSession()
    app.app.state.http_session = session
    async def short_deadline(markdown, sources, provider, **kwargs):
        kwargs.pop("timeout_seconds", None)
        return await original_correct(markdown, sources, provider, timeout_seconds=0.02, **kwargs)
    monkeypatch.setattr(app, "correct", short_deadline)
    result = post(client).json()
    assert result["correction_reason"] == "timeout_source_inventory" and result["requires_review"]
    assert session.calls == 1 and session.cancelled
    assert result["run_metrics"]["failures"][-1]["usage_unknown"] is True


@pytest.mark.parametrize("mutation", ["numeric", "entity", "structure"])
def test_real_final_verifiers_veto_changed_correction_result(offline, monkeypatch, mutation):
    client, _ = offline
    pipeline(monkeypatch)
    async def artificial_approval(markdown, sources, provider, **kwargs):
        if mutation == "numeric":
            final = SOURCE.replace("30% using", "30% in 6 months using")
        elif mutation == "entity":
            final = SOURCE.replace("Synthetic Workshop", "Invented Workshop")
        else:
            final = SOURCE.replace("## Education", "## Unknown Section")
        return CorrectionResult("approved", final, None, "PRIVATE_COMMENTARY", None, 1, 0)
    monkeypatch.setattr(app, "correct", artificial_approval)
    response = post(client)
    result = response.json()
    assert result["requires_review"] and result["final_written_section_markdown"] is None
    assert result["publication_receipt"] is None and "PRIVATE_" not in response.text
    assert {"numeric": "unsupported_claims", "entity": "required_facts_missing",
            "structure": "missing_section"}[mutation] in result["review_reasons"]


@pytest.mark.parametrize("failure", ["metrics", "entity", "receipt"])
def test_post_correction_checks_and_receipt_fail_closed(offline, monkeypatch, synthetic_disclosure_bypass, failure):
    client, _ = offline
    pipeline(monkeypatch)
    if failure == "metrics":
        monkeypatch.setattr(app.tier_a_verifier, "verify_metrics", lambda md, source: {"verdict": "FAIL", "checked": True})
    elif failure == "entity":
        monkeypatch.setattr(app, "unsupported_entities", lambda md, source: ["PRIVATE_ENTITY"])
    else:
        monkeypatch.delenv("PUBLICATION_RECEIPT_SECRET")
    response = post(client)
    result = response.json()
    assert result["requires_review"] and result["publication_receipt"] is None
    assert result["final_written_section"] is None and result["approved_export_payload"] is None
    assert "PRIVATE_" not in response.text