"""Offline regressions for the three behaviours the calibration depends on.

These exist because each one was, at some point, quietly wrong in a way that made
the numbers look BETTER than the run deserved:

1. semantic agreement  — `cannot_assess` and structurally invalid answers were
                         counted as agreement, because the check compared booleans
                         instead of decisions;
2. corrected-document  — approval was read off gate release rather than off the
    approval              re-review's own decision, so a scaffold property was
                         reported as a reviewer verdict;
3. request accounting  — only one unit per logical call was counted, so retries
                         were invisible and a "$1.00 ceiling" was not provable.

Each test below fails if the corresponding bug is reintroduced. No provider calls.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import decision_contract as dc
from experiment import gate_axes
from experiment.gate_axes import (
    AGREE, CANNOT_ASSESS, DISAGREE, INVALID_RESPONSE, OPERATIONAL_FAILURE,
    semantic_agreement,
)
from experiment.provider_v2 import RequestMeter, make_asker
from experiment.run_calibration import agreement, summarise

ROOT = Path(__file__).resolve().parents[1]


# ══ 1. Semantic agreement ════════════════════════════════════════════════════

@pytest.mark.parametrize("decision,ok,expected,want", [
    # A valid exact match is the ONLY agreement.
    ("revise", True, "revise", AGREE),
    ("approve", True, "approve", AGREE),
    # A valid mismatch is a disagreement, not an absence.
    ("approve", True, "revise", DISAGREE),
    ("revise", True, "approve", DISAGREE),
    # cannt_assess is NEVER agreement. The old boolean comparison scored
    # ("cannot_assess" == "approve") == ("revise" == "approve") as True.
    ("cannot_assess", True, "revise", CANNOT_ASSESS),
    ("cannot_assess", True, "approve", CANNOT_ASSESS),
    # A structurally invalid answer is not a verdict, even if the word matches.
    ("revise", False, "revise", INVALID_RESPONSE),
    ("approve", False, "approve", INVALID_RESPONSE),
    # No decision at all is an operational failure.
    (None, False, "revise", OPERATIONAL_FAILURE),
])
def test_semantic_agreement_requires_a_valid_exact_match(decision, ok, expected,
                                                         want):
    assert semantic_agreement(decision, ok, expected) == want


def test_cannot_assess_is_never_counted_as_agreement():
    """The specific bug, stated as a test.

    A `cannot_assess` answer on a `revise` case must not appear in the agreement
    count. Withholding is not a revision request.
    """
    assert semantic_agreement("cannot_assess", True, "revise") != AGREE
    assert semantic_agreement("cannot_assess", True, "approve") != AGREE
    # ...and it is not a disagreement either: it is its own class, so it cannot
    # inflate or deflate the rate the wrong way.
    assert semantic_agreement("cannot_assess", True, "revise") != DISAGREE


def test_the_calibration_runner_uses_the_same_agreement_rule():
    """Two modules, one rule — they must not drift apart."""
    base = {"first_decision": "cannot_assess", "first_valid": True,
            "expected": "revise", "first_error": ""}
    assert agreement(base) == "cannot_assess"
    assert agreement({**base, "first_valid": False}) == "invalid_response"
    assert agreement({**base, "first_error": "TimeoutError"}) == "operational_failure"
    assert agreement({**base, "first_decision": "revise"}) == "agree"
    assert agreement({**base, "first_decision": "approve"}) == "disagree"


def test_the_rate_denominator_excludes_non_verdicts():
    """A success/failure rate must not have non-verdicts in its denominator."""
    rows = [
        {"group": "unfamiliar", "case_id": "a", "expected": "revise",
         "first_decision": "revise", "first_valid": True, "first_error": "",
         "repair_status": "not_attempted", "proposal_drawn": False,
         "release": False, "release_reason": "revision_unresolved",
         "first_cost_usd": 0.0, "re_review_valid": None,
         "first_requests": 1, "re_review_requests": 0,
         "first_usage_missing": False, "re_review_usage_missing": None},
        {"group": "unfamiliar", "case_id": "b", "expected": "revise",
         "first_decision": "cannot_assess", "first_valid": True, "first_error": "",
         "repair_status": "not_attempted", "proposal_drawn": False,
         "release": False, "release_reason": "llm_cannot_assess",
         "first_cost_usd": 0.0, "re_review_valid": None,
         "first_requests": 1, "re_review_requests": 0,
         "first_usage_missing": False, "re_review_usage_missing": None},
        {"group": "unfamiliar", "case_id": "c", "expected": "revise",
         "first_decision": "revise", "first_valid": False, "first_error": "",
         "repair_status": "not_attempted", "proposal_drawn": False,
         "release": False, "release_reason": "initial_review_unverifiable",
         "first_cost_usd": 0.0, "re_review_valid": None,
         "first_requests": 1, "re_review_requests": 0,
         "first_usage_missing": False, "re_review_usage_missing": None},
        {"group": "unfamiliar", "case_id": "d", "expected": "revise",
         "first_decision": None, "first_valid": None, "first_error": "TimeoutError",
         "repair_status": "not_attempted", "proposal_drawn": False,
         "release": False, "release_reason": "operational:TimeoutError",
         "first_cost_usd": 0.0, "re_review_valid": None,
         "first_requests": 1, "re_review_requests": 0,
         "first_usage_missing": False, "re_review_usage_missing": None},
    ]
    summary = summarise(rows)["unfamiliar"]
    assert summary["agreement_classes"] == {
        "agree": 1, "disagree": 0, "cannot_assess": 1,
        "invalid_response": 1, "operational_failure": 1}
    # Exactly one cell reached a verdict, so the rate denominator is 1, not 4.
    assert summary["rate_denominator"] == 1
    assert summary["agrees_with_expectation"] == 1


# ══ 2. Corrected-document approval ═══════════════════════════════════════════

def test_retained_run_reports_approval_from_the_re_review_decision():
    """Item 2: approval needs a VALID re-review that says `approve`.

    The retained data has two valid re-reviews and neither approved, so the count
    must be 0 — and crucially the one cell that DID see the exact proposal must be
    decided by its own answer, not by gate release.
    """
    summary = gate_axes.build()["axis_3_corrected_document_approval"]
    # Zero approvals in this run: the two valid re-reviews both said `revise`.
    assert summary["status"].get("approved", 0) == 0
    # The identity gate still holds: only 2 valid re-reviews, 5 different document.
    assert summary["re_review_document_identity"]["same_document"] == 2
    assert summary["re_review_decision"]["revise"] == 2


def test_an_invalid_re_review_is_reported_separately_from_a_refusal():
    """An unreadable re-review is neither an approval nor a refusal."""
    rows = [r for r in gate_axes.build()["rows"]
            if r["axis_corrected_document_approval"] == "invalid_re_review"]
    assert rows, "the retained run contains an invalid re-review to test against"
    for row in rows:
        assert row["axis_re_review_decision"] == "revise"
        assert row["axis_re_review_valid"] is False
        assert "not a usable verdict" in row["axis_corrected_document_note"]


def test_approval_requires_valid_and_approve_not_merely_release():
    """A release with a non-approving re-review must not be read as approval."""
    rows = [r for r in gate_axes.build()["rows"]
            if r["in_candidate_denominator"]]
    for row in rows:
        if row["axis_corrected_document_approval"] == "approved":
            assert row["axis_re_review_valid"] is True
            assert row["axis_re_review_decision"] == "approve"
    # The retained run has no such row, which is exactly why the positive path is
    # covered by unit control instead of by this data.
    assert not any(r["axis_corrected_document_approval"] == "approved"
                   for r in rows)


# ══ 3. Request accounting ════════════════════════════════════════════════════

def test_the_meter_counts_every_attempt_including_failures():
    meter = RequestMeter(max_requests=10, max_cost_usd=1.0)
    for status, cost, reported in [
        ("http_503", None, False),
        ("http_503", None, False),
        ("ok", 0.001, True),
    ]:
        meter.record_attempt(label="x:first", attempt=1, status=status,
                             cost=cost, usage_reported=reported)
    assert meter.requests_sent == 3, "failed attempts are still requests"
    assert meter.billable_requests == 1
    assert meter.cost_usd == pytest.approx(0.001)


def test_the_request_ceiling_is_checked_before_each_attempt():
    meter = RequestMeter(max_requests=2, max_cost_usd=1.0)
    assert meter.can_send()[0] is True
    meter.record_attempt(label="a", attempt=1, status="http_500", cost=None,
                         usage_reported=False)
    assert meter.can_send()[0] is True
    meter.record_attempt(label="a", attempt=2, status="http_500", cost=None,
                         usage_reported=False)
    allowed, why = meter.can_send()
    assert allowed is False and "request ceiling" in why


def test_missing_usage_is_not_zero_cost():
    """A response without usage is UNKNOWN spend, not free."""
    meter = RequestMeter(max_requests=10, max_cost_usd=1.0)
    meter.record_attempt(label="a", attempt=1, status="ok_no_usage", cost=0.0,
                         usage_reported=False)
    assert meter.cost_is_known is False
    assert meter.unmetered_requests == 1
    assert "true spend is unknown" in meter.to_dict()["unmetered_caveat"]


def test_a_clean_meter_declares_cost_known():
    meter = RequestMeter(max_requests=10, max_cost_usd=1.0)
    meter.record_attempt(label="a", attempt=1, status="ok", cost=0.002,
                         usage_reported=True)
    assert meter.cost_is_known is True
    assert meter.to_dict()["unmetered_caveat"] is None


class _FakeResp:
    def __init__(self, status, payload=None):
        self.status = status
        self._payload = payload or {}

    async def json(self):
        return self._payload

    async def text(self):
        return "error"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Yields a scripted sequence of responses, one per request."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, *_a, **_k):
        self.requests += 1
        return self.script.pop(0)

    def timeout(self, *_a, **_k):  # pragma: no cover - only for shape parity
        return None


def _patch_aiohttp(monkeypatch, script):
    import sys
    import types
    session = _FakeSession(script)
    module = types.SimpleNamespace(
        ClientSession=lambda **_: session,
        ClientTimeout=lambda **_: None,
    )
    monkeypatch.setitem(sys.modules, "aiohttp", module)
    return session


SUCCESS = {"choices": [{"message": {"content": "Decision: approve\n"},
                        "finish_reason": "stop"}],
           "usage": {"cost": 0.004, "prompt_tokens": 10, "completion_tokens": 5,
                     "total_tokens": 15}}


def test_a_retried_call_reports_every_request_it_sent(monkeypatch):
    """The retry path is billed; the meter and `requests_sent` must both say so."""
    session = _patch_aiohttp(monkeypatch, [
        _FakeResp(503), _FakeResp(503), _FakeResp(200, SUCCESS)])
    meter = RequestMeter(max_requests=10, max_cost_usd=1.0)
    ask = make_asker(api_key="k", meter=meter, sleep_between_retries=False)
    result = asyncio.run(ask("prompt", label="case:first"))

    assert result.requests_sent == 3
    assert meter.requests_sent == 3
    assert meter.logical_calls == 1
    assert session.requests == 3
    assert result.cost_usd == pytest.approx(0.004)
    # The two 503s were sent but reported no usage; they are still requests.
    assert [a["status"] for a in meter.attempts] == ["http_503", "http_503", "ok"]


def test_a_transport_error_is_metered_as_an_attempt(monkeypatch):
    """A timeout was sent and may be billed; it must not vanish from the count."""
    class _BoomSession(_FakeSession):
        def post(self, *_a, **_k):
            self.requests += 1
            raise TimeoutError("read timeout")

    import sys
    import types
    session = _BoomSession([])
    monkeypatch.setitem(sys.modules, "aiohttp", types.SimpleNamespace(
        ClientSession=lambda **_: session, ClientTimeout=lambda **_: None))

    meter = RequestMeter(max_requests=10, max_cost_usd=1.0)
    ask = make_asker(api_key="k", meter=meter, sleep_between_retries=False,
                     max_attempts=2)
    with pytest.raises(RuntimeError):
        asyncio.run(ask("prompt", label="case:first"))
    assert meter.requests_sent == 2, "each timed-out attempt is a request"
    assert meter.cost_is_known is False, "a timeout's cost is unknown, not zero"


def test_the_meter_refuses_to_send_beyond_the_ceiling(monkeypatch):
    """The ceiling must bind the request that would cross it."""
    _patch_aiohttp(monkeypatch, [_FakeResp(503), _FakeResp(503), _FakeResp(503)])
    meter = RequestMeter(max_requests=2, max_cost_usd=1.0)
    ask = make_asker(api_key="k", meter=meter, sleep_between_retries=False,
                     max_attempts=3)
    with pytest.raises(RuntimeError) as err:
        asyncio.run(ask("prompt", label="case:first"))
    assert "meter refused request" in str(err.value)
    assert meter.requests_sent == 2, "no third request may be sent"


# ══ The frozen plan must declare a request ceiling, not only a call ceiling ══

def test_the_frozen_plan_states_a_hard_request_ceiling():
    from experiment.run_calibration import freeze
    frozen = freeze()
    assert frozen["max_requests"] == frozen["max_logical_calls"] * \
        frozen["attempts_per_call"]
    assert "HARD ceiling" in frozen["ceiling_statement"]
    assert "not a proof of maximum spend" in frozen["ceiling_statement"]
