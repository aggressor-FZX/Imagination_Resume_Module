"""Offline-only privacy boundary tests; no model or external service is used."""
import asyncio
import json

import pytest

import app
import correction_provider
from correction_provider import CorrectionProvider, PrivacyReviewRequired
from source_correction import (
    MAX_DRAFT_CHARS, MAX_SOURCE_CHARS, MAX_SOURCE_DOCUMENTS,
    MAX_TIMEOUT_SECONDS, SourceDocument, correct,
)
from tests.test_correction_integration_offline import DRAFT, SOURCE, FakeSession
from tests.test_analyze_source_correction_route import offline, pipeline, post, result_from_sse  # noqa: F401


@pytest.mark.parametrize("private", [
    "Avery Example | avery@example.org | +44 20 7946 0958",
    "37 Rue des Écoles, Paris; portfolio.example.org/user; github.com/user",
    "李小明; avery [at] example [dot] org; ١٢٣ شارع السلام",
    "Alice supervised Bob; Bob supervised Alice",
    "[PERSON] and [PERSON] are different individuals",
    "No obvious contact details in this text",
])
def test_provider_never_sends_any_source_or_draft_without_policy(private):
    session = FakeSession()
    provider = CorrectionProvider("fake", session)
    source = SourceDocument("candidate", private + "\n" + SOURCE)
    result = asyncio.run(correct(private + "\n" + DRAFT, (source,), provider))
    assert (result.status, result.reason, result.markdown) == (
        "withheld", "privacy_review_required", None)
    assert not session.requests and provider.calls == 0
    assert result.review_calls == result.repair_calls == 0
    assert private not in repr(result)


@pytest.mark.parametrize("phase", ["source_inventory", "initial_review", "repair", "post_patch_review"])
def test_provider_boundary_withholds_every_phase_even_when_called_directly(phase):
    session = FakeSession()
    provider = CorrectionProvider("fake", session)
    with pytest.raises(PrivacyReviewRequired):
        asyncio.run(provider(phase, {"contract": "test", "text": "Nadia Private"}))
    assert not session.requests and provider.calls == 0


def test_route_withholds_before_external_correction_and_emits_no_private_aliases(offline, monkeypatch):
    client, session = offline
    pipeline(monkeypatch)
    result = post(client).json()
    assert result["correction_reason"] == "privacy_review_required"
    assert result["requires_review"] and result["publication_receipt"] is None
    assert result["final_written_section_markdown"] is None
    assert result["approved_export_payload"] is None
    assert not session.requests
    assert "PRIVATE_" not in json.dumps(result)


def test_sse_withholds_without_streaming_draft(offline, monkeypatch):
    client, session = offline
    pipeline(monkeypatch)
    response = post(client, sse=True)
    blocks, result = result_from_sse(response)
    assert result["correction_reason"] == "privacy_review_required"
    assert not any(block.startswith("event: token") for block in blocks)
    assert any(block.startswith("event: withheld") for block in blocks)
    assert "PRIVATE_" not in response.text and not session.requests


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf"),
                                      MAX_TIMEOUT_SECONDS + 1, 10 ** 1000, "30"])
def test_invalid_deadlines_never_invoke_callback(timeout):
    async def forbidden(*args):
        raise AssertionError("must not invoke")
    result = asyncio.run(correct("draft", (SourceDocument("id", "source"),), forbidden,
                                 timeout_seconds=timeout))
    assert result.reason == "invalid_input"


@pytest.mark.parametrize("draft,sources", [
    ("x" * (MAX_DRAFT_CHARS + 1), (SourceDocument("a", "source"),)),
    ("draft", (SourceDocument("a", "x" * (MAX_SOURCE_CHARS + 1)),)),
    ("draft", (SourceDocument("a", "x" * MAX_SOURCE_CHARS), SourceDocument("b", "x"))),
    ("draft", tuple(SourceDocument(str(n), "x") for n in range(MAX_SOURCE_DOCUMENTS + 1))),
])
def test_input_bounds_reject_before_callback(draft, sources):
    async def forbidden(*args):
        raise AssertionError("must not invoke")
    result = asyncio.run(correct(draft, sources, forbidden))
    assert result.reason == "invalid_input"


def test_complete_json_byte_bound_counts_unicode_escaping_before_transport(monkeypatch):
    session = FakeSession()
    provider = CorrectionProvider("fake", session)
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available", lambda: True)
    with pytest.raises(ValueError, match="request too large"):
        asyncio.run(provider("source_inventory", {"contract": "x", "data": "😀" * 10_000}))
    assert not session.requests and provider.calls == 0


def test_synthetic_local_transport_preserves_offsets_output_and_evidence(monkeypatch):
    # This synthetic-only override is not a disclosure policy and is never used
    # in the application. It exercises existing offset and repair semantics.
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available", lambda: True)
    session = FakeSession()
    provider = CorrectionProvider("fake", session)
    source = SourceDocument("original-evidence", SOURCE)
    result = asyncio.run(correct(DRAFT, (source,), provider))
    assert result.status == "approved" and result.markdown == SOURCE
    payload = json.loads(session.requests[0]["json"]["messages"][1]["content"])
    # Offsets are the application's, so the transmitted segments must reconstruct
    # the source exactly: concatenating them yields the source text unchanged.
    sent = payload["segments"]
    assert {segment["source_id"] for segment in sent} == {source.source_id}
    assert all(segment["start"] == position
               for position, segment in zip([0] + [s["end"] for s in sent], sent))
    assert "".join(segment["text"] for segment in sent) == SOURCE
    assert source.text == SOURCE
    # The timeout is derived from the remaining budget now, so assert the bounds
    # rather than a constant that no longer exists.
    assert all(correction_provider.PROVIDER_TIMEOUT_SECONDS
               <= req["timeout"].total
               <= correction_provider.PROVIDER_TIMEOUT_CEILING_SECONDS
               for req in session.requests)


def test_oversized_response_is_rejected_without_retry(monkeypatch):
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available", lambda: True)
    class Session:
        calls = 0
        def post(self, *args, **kwargs):
            self.calls += 1
            class Response:
                status = 200
                content = self
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *args):
                    pass
                async def iter_chunked(self, size):
                    yield b"x" * (correction_provider.MAX_RESPONSE_BYTES + 1)
            return Response()
    session = Session()
    provider = CorrectionProvider("fake", session)
    result = asyncio.run(correct(DRAFT, (SourceDocument("resume", SOURCE),), provider))
    assert result.reason == "provider_error" and session.calls == 1