"""Composed route test: transient failure -> retry -> unsupported claim -> withheld.

This replaces a detached verifier assertion with the real `/analyze` path. The earlier
test proved only that `tier_a_verifier` rejects an invented figure in isolation, which is
true but not the question. The question is whether a *retried* answer that is then found
factually unsupported still fails to publish, through the whole route: correction,
verification, gate, receipt and export.

Composition under test:

1. the first transport attempt returns a retryable 5xx;
2. the retry succeeds and the correction completes or is withheld on its own terms;
3. the final document carries a claim the source does not support;
4. publication is withheld, no receipt is issued, nothing is exportable.

Step 3 is driven through the real verifiers by replacing the correction result — the same
technique the existing route tests use — because a scripted provider cannot make the
engine approve text it has itself validated. What is real here is everything downstream:
`verify_metrics`, `unsupported_entities`, `publication_gate.decide`,
`_issue_publication_receipt` and `/resume/export`.
"""
import json

import pytest
from fastapi.testclient import TestClient

import app
import correction_provider
from correction_provider import CorrectionProvider
from source_correction import CorrectionResult
from tests.test_analyze_source_correction_route import (
    offline,  # noqa: F401  (fixture)
    pipeline,
    post,
    result_from_sse,
)


SOURCE = """## Professional Experience
**Synthetic Workshop** -- Example City, WA
- Maintained the workshop inventory and documented every repair procedure for the service technicians on site.
- Improved diagnostic accuracy by 30% using structured test procedures and careful review of repair records.
- Coordinated inspection schedules with three external contractors and tracked every open item to closure.

## Education
**Synthetic College** -- Example City, WA
- Completed technical training in electrical systems, equipment maintenance, safety practices, and troubleshooting.

## Technical Skills
- Inventory control, repair documentation, inspection coordination, diagnostic testing, and maintenance scheduling.
"""

DRAFT = SOURCE.replace("by 30% using", "by 47% using")


class TransientThenServed:
    """Fails the first attempt with a retryable status, then serves a valid envelope.

    Counts attempts so the test can prove a retry actually happened rather than
    assuming it. Answers are built from the request that was sent, so the inventory and
    review phases each receive the shape they asked for.
    """

    def __init__(self, status=503):
        self.status = status
        self.attempts = 0
        self.requests = []

    async def close(self):
        pass

    def post(self, url, **kwargs):
        self.attempts += 1
        self.requests.append(kwargs)
        payload = json.loads(kwargs["json"]["messages"][1]["content"])
        if self.attempts == 1:
            status, body = self.status, ""
        else:
            status, body = 200, self._envelope(payload)
        return _response(status, body)

    @staticmethod
    def _envelope(payload):
        if "flagged_claim_ids" in payload:
            # Repair leaves the flagged span unchanged so the engine reports a factual
            # outcome rather than looping.
            line_id = payload["flagged_claim_ids"][0].partition(":")[0]
            text = next(line["text"] for line in payload["lines"]
                        if line["line_id"] == line_id)
            answer = {"edits": [{"claim_id": claim_id, "replacement": text}
                                for claim_id in payload["flagged_claim_ids"]]}
        elif "segments" in payload and "lines" not in payload:
            required = [segment["index"] for segment in payload["segments"]
                        if segment["text"].strip()]
            answer = {"required": required, "non_resume": [], "uncertain": []}
        else:
            structural = [line["line_id"] for line in payload["lines"]
                          if line["text"].startswith("## ")]
            supported = [line["line_id"] for line in payload["lines"]
                         if not line["text"].startswith("## ")]
            segment_indexes = [segment["index"] for segment in payload["segments"]]
            answer = {
                "structural": structural, "supported": supported,
                "unsupported": [], "uncertain": [],
                "evidence": {line_id: segment_indexes[:1] for line_id in supported},
                "represented": [segment["index"] for segment in payload["segments"]
                                if segment["text"].strip()],
                "missing": [],
            }
        return json.dumps({
            "model": correction_provider.OR_SLUG_DRAFTER,
            "provider": "ExampleProvider",
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            "choices": [{"finish_reason": "stop",
                         "message": {"content": json.dumps(answer)}}],
        })


def _response(status, body):
    inner_status, inner_body = status, body

    class Content:
        async def iter_chunked(self, size):
            raw = (inner_body or "").encode()
            for index in range(0, len(raw), size):
                yield raw[index:index + size]

    class Headers:
        def get(self, name, default=None):
            return default

    class Response:
        status = inner_status
        headers = Headers()
        content = Content()

        async def text(self, *args, **kwargs):
            return inner_body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    return Response()


@pytest.fixture
def faster_retry(monkeypatch):
    """Prove the retry without waiting out the backoff."""
    import correction_provider as module

    async def instant(seconds):
        return None

    monkeypatch.setattr(module, "_sleep", instant)


@pytest.fixture
def disclosure(monkeypatch):
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available", lambda: True)


def _approve_invented_metric(monkeypatch):
    """Run the real correction, then hand the route a document with an invented figure.

    The adapter genuinely runs and genuinely retries; only the *verdict* is overridden.
    That is the strongest realistic form of the scenario: even if the reviewer had been
    fooled into approving this text, the route's own verification must still refuse to
    publish it.
    """
    invented = SOURCE.replace(
        "Improved diagnostic accuracy by 30%",
        "Improved diagnostic accuracy by 47%")
    real_correct = app.correct

    async def approve_invented(markdown, sources, provider, **kwargs):
        # Exercise the real adapter and engine first, so the retry actually happens.
        await real_correct(markdown, sources, provider, **kwargs)
        return CorrectionResult("approved", invented, None, "PRIVATE_COMMENTARY", None, 1, 0)

    monkeypatch.setattr(app, "correct", approve_invented)
    return invented


def test_transient_failure_then_retry_still_withholds_an_unsupported_claim(
        offline, monkeypatch, disclosure, faster_retry):  # noqa: F811
    """The composed requirement, on the real route.

    A retry that succeeds must not launder an unsupported claim into a publication. The
    retry changes how the answer was obtained, nothing about whether it is true.
    """
    client, session = offline
    pipeline(monkeypatch, draft=DRAFT)
    transport = TransientThenServed(503)
    app.app.state.http_session = transport
    _approve_invented_metric(monkeypatch)

    response = post(client)
    result = response.json()

    # 1 + 2: the transient failure was retried rather than aborting the run.
    assert transport.attempts >= 2, "the 503 must have been retried"

    # 3 + 4: publication withheld, no receipt, nothing exportable.
    assert result["requires_review"] is True
    assert result["publication_receipt"] is None
    assert result["approved_export_payload"] is None
    assert result["final_written_section_markdown"] is None
    assert result["processing_status"] != "completed"
    # The unsupported figure is named as the reason, not hidden behind a generic error.
    assert any(reason in {"unsupported_claims", "required_facts_missing"}
               for reason in result["review_reasons"]), result["review_reasons"]

    # The invented figure never reaches the caller in any field.
    assert "47%" not in response.text
    # And no internal content leaked either.
    assert "PRIVATE_" not in response.text


def test_a_failed_export_after_a_retried_approval_is_refused(offline, monkeypatch, disclosure, faster_retry):  # noqa: F811
    """There is nothing to export, so the export endpoint must refuse.

    A withheld result carries no receipt; `/resume/export` must reject rather than
    render. This closes the loop from retry to the customer's download button.
    """
    client, session = offline
    pipeline(monkeypatch, draft=DRAFT)
    app.app.state.http_session = TransientThenServed(503)
    _approve_invented_metric(monkeypatch)

    result = post(client).json()
    assert result["publication_receipt"] is None

    # No receipt means no export, whatever the client sends.
    body = {"final_written_section_markdown": SOURCE,
            "publication_receipt": None}
    export = client.post("/resume/export", json=body,
                         headers={"X-Cogito-Owner-Id": "offline-owner"})
    assert export.status_code in (400, 401, 403, 409, 422)


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_every_retryable_status_still_ends_withheld_when_the_claim_is_unsupported(
        offline, monkeypatch, disclosure, faster_retry, status):  # noqa: F811
    """Negative control across the retryable set: retrying never implies approving."""
    client, session = offline
    pipeline(monkeypatch, draft=DRAFT)
    transport = TransientThenServed(status)
    app.app.state.http_session = transport
    _approve_invented_metric(monkeypatch)

    result = post(client).json()
    assert transport.attempts >= 2
    assert result["requires_review"] is True
    assert result["publication_receipt"] is None
    assert result["final_written_section_markdown"] is None


def test_sse_client_also_gets_no_tokens_after_a_retried_unsupported_claim(
        offline, monkeypatch, disclosure, faster_retry):  # noqa: F811
    """The streaming path must not emit a partial resume for a withheld result.

    SSE is the path the website actually uses, and it emits rewrite tokens before the
    result event. A retried-then-unsupported document must emit none.
    """
    client, session = offline
    pipeline(monkeypatch, draft=DRAFT)
    app.app.state.http_session = TransientThenServed(503)
    _approve_invented_metric(monkeypatch)

    response = post(client, sse=True)
    blocks, result = result_from_sse(response)
    assert not any(block.startswith("event: token") for block in blocks), \
        "a withheld result must not stream resume text"
    assert any(block.startswith("event: withheld") for block in blocks)
    assert result["publication_receipt"] is None
    assert "47%" not in response.text


def test_a_retried_success_is_not_summarily_withheld(
        offline, monkeypatch, disclosure, faster_retry):  # noqa: F811
    """Positive control: a retry must not become a blanket refusal.

    Without this, a change that withheld everything after any retry would pass every
    negative control above while making the product useless. The scripted transport
    answers honestly, so no *availability* reason may appear and the retried phase must
    have completed. Correction's own judgement (here ``no_op_repair``, because the only
    defect is one the scripted reviewer cannot fix) is a factual verdict and is allowed
    to withhold — that is the gate working, not the retry failing.
    """
    client, session = offline
    pipeline(monkeypatch, draft=DRAFT)
    transport = TransientThenServed(503)
    app.app.state.http_session = transport

    result = post(client).json()
    assert transport.attempts >= 2, "the 503 must have been retried"
    # The retried run must not fail for an availability reason.
    assert result.get("correction_reason") not in {
        "provider_unavailable", "provider_rate_limited", "provider_error",
        "provider_rejected_request"}, result.get("correction_reason")
    # And it must not have stopped at the phase that was retried.
    assert result.get("correction_reason") not in {
        "provider_call_timeout_source_inventory", "timeout_source_inventory"}
    # Receipt presence is decided solely by the gate, never by whether a retry occurred.
    if result["processing_status"] == "completed":
        assert result["publication_receipt"] is not None
    else:
        assert result["publication_receipt"] is None


# --- Focused positive control: retry -> supported content -> published ---------
#
# The negative controls above prove a retry cannot launder an unsupported claim. This is
# their counterpart: the same transient failure, retried, on a draft that needs no
# correction at all, must reach a completed response carrying a usable receipt.
#
# Correction is stubbed here on purpose. A scripted transport cannot produce the exact
# span-level verdicts the engine demands, and the earlier attempts to make it do so were
# testing the fixture rather than the route. What this test isolates is the part the
# milestone actually depends on: after a retry, does the route publish what verification
# approves, with a receipt the export endpoint accepts? The correction engine's own
# behaviour is covered by tests/test_source_correction.py and the composed negatives.


@pytest.fixture
def scripted_approval(monkeypatch):
    """Return the source unchanged as an approved correction, after the real retry runs."""

    def install(document=SOURCE):
        real_correct = app.correct

        async def approve(markdown, sources, provider, **kwargs):
            # Exercise the real adapter so the transient failure is genuinely retried.
            await real_correct(markdown, sources, provider, **kwargs)
            return CorrectionResult("approved", document, None,
                                    "Model-assisted review is not proof.", None, 1, 0)

        monkeypatch.setattr(app, "correct", approve)

    return install


@pytest.mark.parametrize("status", [429, 503])
def test_transient_failure_then_retry_then_supported_content_is_published(
        offline, monkeypatch, disclosure, faster_retry, scripted_approval, status):  # noqa: F811
    """transient failure -> successful retry -> supported content -> completed + receipt."""
    client, session = offline
    pipeline(monkeypatch, draft=DRAFT)
    transport = TransientThenServed(status)
    app.app.state.http_session = transport
    scripted_approval(SOURCE)

    result = post(client).json()

    # 1 -> 2: the transient failure was retried and the run continued past it.
    assert transport.attempts >= 4,         "1 retried attempt plus 3 correction phases must all have reached the transport"

    # 3: the published content is the supported document, not the dangling draft.
    assert result["processing_status"] == "completed", result.get("review_reasons")
    assert result["requires_review"] is False
    assert result["correction_status"] == "approved"
    assert result["final_written_section_markdown"] == SOURCE
    assert result["metrics_support"]["verdict"] == "PASS (tier A only)"

    # 4: a receipt and an exportable payload were issued.
    assert result["publication_receipt"] is not None
    assert result["approved_export_payload"] is not None


def test_the_published_positive_control_can_actually_be_exported(
        offline, monkeypatch, disclosure, faster_retry, scripted_approval):  # noqa: F811
    """The receipt must be usable, not merely present.

    A positive control whose receipt the export endpoint then refuses would not
    represent a customer receiving anything, so this renders the real export.
    """
    from unittest.mock import Mock

    client, session = offline
    pipeline(monkeypatch, draft=DRAFT)
    app.app.state.http_session = TransientThenServed(503)
    scripted_approval(SOURCE)

    result = post(client).json()
    assert result["publication_receipt"] is not None
    assert result["processing_status"] == "completed"

    renderer = Mock(return_value={"filename": "synthetic", "pdf_base64": "AA",
                                  "docx_base64": "AA"})
    monkeypatch.setattr(
        "resume_builder.build_resume_from_export.build_resume_from_payload_base64",
        renderer)
    body = {**result["approved_export_payload"],
            "final_written_section_markdown": result["final_written_section_markdown"],
            "publication_receipt": result["publication_receipt"]}
    export = client.post("/resume/export", json=body,
                         headers={"X-Cogito-Owner-Id": "offline-owner"})
    assert export.status_code == 200, export.text[:300]
    renderer.assert_called_once()


def test_the_invented_metric_veto_still_holds_after_a_retry(
        offline, monkeypatch, disclosure, faster_retry):  # noqa: F811
    """FAULT INJECTION: the verifier vetoes an invented figure even after a retry.

    This is the labelled fault-injection counterpart to the positive control above. The
    correction result is replaced with a document containing a figure the source does not
    support — the fault — and the route must refuse to publish it despite the retry
    having succeeded.
    """
    client, session = offline
    pipeline(monkeypatch, draft=DRAFT)
    app.app.state.http_session = TransientThenServed(503)

    invented = SOURCE.replace("by 30%", "by 47%")
    real_correct = app.correct

    async def approve_invented(markdown, sources, provider, **kwargs):
        await real_correct(markdown, sources, provider, **kwargs)
        return CorrectionResult("approved", invented, None, "PRIVATE_COMMENTARY",
                                None, 1, 0)

    monkeypatch.setattr(app, "correct", approve_invented)

    response = post(client)
    result = response.json()
    assert result["requires_review"] is True
    assert result["publication_receipt"] is None
    assert result["final_written_section_markdown"] is None
    assert "unsupported_claims" in result["review_reasons"], result["review_reasons"]
    assert "47%" not in response.text
