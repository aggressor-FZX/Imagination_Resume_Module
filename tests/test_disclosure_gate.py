"""Synthetic tests for the correction disclosure gate and redaction policy.

These are local, offline tests. They do not call a provider and do not prove
that arbitrary text is non-identifying; they verify the implemented contract.
"""
import asyncio
import json

import pytest

import correction_provider
from correction_provider import CorrectionProvider
from disclosure import (
    POLICY_ID,
    contains_pii,
    redact_sources,
    redact_text,
    verify_disclosure,
)
from source_correction import SourceDocument, correct


SOURCE = (
    "Alex Example\n"
    "alex.example@example.test | 555-010-0199\n"
    "1234 Example Street\n"
    "linkedin.com/in/alex-example\n"
    "Engineer at Alpha Works 2019-2021\n"
    "Improved diagnostic accuracy by 30% over 6 months\n"
)


class FakeResponse:
    status = 200

    def __init__(self, body):
        self.content = self
        self.body = body

    async def iter_chunked(self, size):
        for index in range(0, len(self.body), size):
            yield self.body[index:index + size]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class RecordingSession:
    def __init__(self, body):
        self.requests = []
        self.body = body

    def post(self, url, **kwargs):
        self.requests.append(kwargs)
        return FakeResponse(self.body)


def _inventory(payload):
    assert set(payload) == {"contract", "sources"}
    items = []
    for source in payload["sources"]:
        text, sid = source["text"], source["source_id"]
        start = position = 0
        for part in text.splitlines(keepends=True):
            position += len(part)
            if not part.strip():
                continue
            items.append({"id": f"{sid}:{start}:{position}", "source_id": sid,
                          "start": start, "end": position,
                          "quote": text[start:position], "classification": "required"})
            start = position
        if position > start:
            last = items[-1]
            last.update(end=position, quote=text[last["start"]:position],
                        id=f"{sid}:{last['start']}:{position}")
    return {"source_claims": items}


def _review(payload):
    claims, spans = [], {item["id"]: [] for item in payload["source_claims"]}
    for line in payload["lines"]:
        text, line_id = line["text"], line["line_id"]
        item = next((item for item in payload["source_claims"]
                     if item["quote"].strip().splitlines()[-1] == text), None)
        if item is None:
            claims.append({"id": f"{line_id}:0:{len(text)}", "line_id": line_id,
                           "start": 0, "end": len(text), "verdict": "unsupported",
                           "citations": []})
            continue
        citation = {key: item[key] for key in ("source_id", "start", "end", "quote")}
        claims.append({"id": f"{line_id}:0:{len(text)}", "line_id": line_id,
                       "start": 0, "end": len(text), "verdict": "supported",
                       "citations": [citation]})
        spans[item["id"]].append({"line_id": line_id, "start": 0, "end": len(text)})
    mappings = [{"source_claim_id": item["id"],
                 "status": "represented" if spans[item["id"]] else "missing",
                 "spans": spans[item["id"]]} for item in payload["source_claims"]]
    return {"claims": claims, "representations": mappings}


def _provider(session):
    def post(_payload):
        body = {"claims": [], "representations": []}
        return json.dumps({
            "model": "qwen/qwen3-coder",
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            "choices": [{"finish_reason": "stop",
                         "message": {"content": json.dumps(body)}}],
        }).encode()
    session.body_factory = post
    return session


def test_redaction_preserves_every_character_offset():
    cleaned, count = redact_text(SOURCE)
    assert len(cleaned) == len(SOURCE)
    assert count >= 4
    assert "alex.example@example.test" not in cleaned
    assert "555-010-0199" not in cleaned
    assert "1234 Example Street" not in cleaned
    assert "linkedin.com/in/alex-example" not in cleaned
    # Resume content the reviewer needs is retained.
    assert "Engineer at Alpha Works" in cleaned
    assert "30% over 6 months" in cleaned
    for original_line, cleaned_line in zip(SOURCE.splitlines(), cleaned.splitlines()):
        assert len(original_line) == len(cleaned_line)


def test_redaction_is_idempotent_and_pii_free():
    cleaned, _ = redact_text(SOURCE)
    again, second = redact_text(cleaned)
    assert again == cleaned and second == 0
    assert contains_pii(cleaned) is False
    assert contains_pii(SOURCE) is True


@pytest.mark.parametrize("text", [
    "person@example.test",
    "555-010-0199",
    "123-45-6789",
    "1234 Example Street",
    "linkedin.com/in/someone",
])
def test_each_covered_category_is_detected_and_removed(text):
    cleaned, count = redact_text(text)
    assert count == 1
    assert len(cleaned) == len(text)
    assert contains_pii(cleaned) is False


def test_unexpected_redaction_shape_withholds_instead_of_sending():
    # Correspondence cannot be proven when a source is not a string.
    documents, attestation = redact_sources([("a", "text"), ("b", None)])
    assert documents == () and attestation is None


def test_gate_is_closed_by_default(monkeypatch):
    monkeypatch.delenv(correction_provider.DISCLOSURE_SETTING, raising=False)
    assert correction_provider.approved_disclosure_policy() is None
    session = _provider(RecordingSession(b""))
    provider = CorrectionProvider("synthetic-key", session)
    with pytest.raises(Exception):
        asyncio.run(provider("source_inventory",
                             {"contract": "c", "sources": [{"source_id": "a", "text": "x"}]}))
    assert session.requests == []


def test_gate_blocks_when_sources_still_contain_pii(monkeypatch):
    monkeypatch.setenv(correction_provider.DISCLOSURE_SETTING, "1")
    session = _provider(RecordingSession(b""))
    provider = CorrectionProvider("synthetic-key", session)
    raw = [{"source_id": "a", "text": "contact me at a@example.test"}]
    with pytest.raises(Exception):
        asyncio.run(provider("source_inventory", {"contract": "c", "sources": raw}))
    assert session.requests == []


def test_verify_disclosure_requires_approved_policy_and_clean_sources():
    clean = [{"source_id": "a", "text": "Engineer at Alpha Works"}]
    assert verify_disclosure(clean, approved_policy=POLICY_ID) is True
    assert verify_disclosure(clean, approved_policy=None) is False
    assert verify_disclosure([], approved_policy=POLICY_ID) is False
    assert verify_disclosure([{"source_id": "a", "text": "a@example.test"}],
                             approved_policy=POLICY_ID) is False
    assert verify_disclosure([{"source_id": "a", "text": "x", "extra": 1}],
                             approved_policy=POLICY_ID) is False


def test_correction_deadline_exceeds_a_single_provider_call():
    """Regression: a live run timed out because the whole-operation deadline was
    not larger than one provider call's own timeout."""
    import correction_provider
    import source_correction

    assert source_correction.MAX_TIMEOUT_SECONDS > correction_provider.PROVIDER_TIMEOUT_SECONDS
    # A clean pass is 2 calls and one repair is 4; the deadline must cover two
    # calls even in the worst case of both consuming their full timeout.
    assert source_correction.MAX_TIMEOUT_SECONDS >= 2 * correction_provider.PROVIDER_TIMEOUT_SECONDS
    import inspect
    default = inspect.signature(source_correction.correct).parameters["timeout_seconds"].default
    assert default > correction_provider.PROVIDER_TIMEOUT_SECONDS
    assert source_correction.MAX_TIMEOUT_SECONDS >= default


def test_real_adapter_with_redacted_sources_can_inventory(monkeypatch):
    """End-to-end through the real adapter: only redacted text reaches transport."""
    monkeypatch.setenv(correction_provider.DISCLOSURE_SETTING, "1")
    session = RecordingSession(b"")

    class Transport:
        def __init__(self, inner):
            self.inner = inner
            self.requests = inner.requests

        def post(self, url, **kwargs):
            payload = json.loads(kwargs["json"]["messages"][1]["content"])
            body = _inventory(payload) if "lines" not in payload else _review(payload)
            self.inner.requests.append(kwargs)
            return FakeResponse(json.dumps({
                "model": "qwen/qwen3-coder",
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                "choices": [{"finish_reason": "stop",
                             "message": {"content": json.dumps(body)}}],
            }).encode())

    transport = Transport(session)
    redacted, attestation = redact_sources([("candidate", SOURCE)])
    assert attestation is not None
    sources = tuple(SourceDocument(sid, text) for sid, text in redacted)
    # This fixture drafts every required source line, so the review can account
    # for the complete redacted inventory rather than a single claim.
    draft = "\n".join(line for line in redacted[0][1].splitlines()
                      if line.strip() and not contains_pii(line))
    result = asyncio.run(correct(draft, sources, CorrectionProvider("k", transport)))
    assert result.status == "approved", result
    assert transport.requests
    sent = transport.requests[0]["json"]["messages"][1]["content"]
    for secret in ("alex.example@example.test", "555-010-0199",
                   "1234 Example Street", "linkedin.com/in/alex-example"):
        assert secret not in sent
    assert "Engineer at Alpha Works" in sent
