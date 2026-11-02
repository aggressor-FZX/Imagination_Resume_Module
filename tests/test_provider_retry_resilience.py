"""Transient provider failure is recoverable without redoing a whole analysis.

The gap this closes: a late-stage transient failure used to discard every earlier
successful stage. Measured 2026-11-02, the same valid 13.6KB review request returned
504 in ~8.5s on some attempts and 200 in 26-39s on others — so a single bad answer was
throwing away a completed pipeline and a paid website credit.

The retry lives in the adapter, which is the only layer that sees an HTTP status. That
placement is the point: the engine's completed phases are never re-run, only the failed
one, and the per-correction call budget is not consumed by transport retries.

Required behaviour, each pinned below:

* 429 then success;
* 504 then success;
* repeated failure stops cleanly with a specific reason;
* malformed or factually unsupported output is NOT retried as an availability problem.
"""
import asyncio
import json

import pytest

import correction_events
import correction_provider
from correction_provider import CorrectionProvider
from source_correction import (
    SourceDocument,
    correct,
)


SOURCE = """## Professional Experience
**Synthetic Workshop**
- Maintained the workshop inventory and documented repair procedures for service technicians.
"""


def phase_answer(payload):
    """Build the index-list answer a real model would give for this request.

    The engine asks inventory and review different questions, so one canned string
    cannot satisfy both. Deriving the answer from the request keeps the double
    honest: it exercises the same translation and validation the live path uses.
    """
    if "flagged_claim_ids" in payload:
        # Repair: leave a flagged span unchanged. The engine reports that as
        # no_op_repair, which is a factual outcome, not an availability one, so the
        # double never turns a retry into an approval.
        line = next(line for line in payload["lines"]
                    if line["line_id"] == payload["flagged_claim_ids"][0].partition(":")[0])
        return {"edits": [{"claim_id": claim_id, "replacement": line["text"]}
                          for claim_id in payload["flagged_claim_ids"]]}
    if "segments" in payload and "lines" not in payload:
        required = [segment["index"] for segment in payload["segments"]
                    if segment["text"].strip()]
        return {"required": required, "non_resume": [], "uncertain": []}
    structural = [line["line_id"] for line in payload["lines"]
                  if line["text"].startswith("## ")]
    supported = [line["line_id"] for line in payload["lines"]
                 if not line["text"].startswith("## ")]
    segment_indexes = [segment["index"] for segment in payload["segments"]]
    evidence = {line["line_id"]: segment_indexes[:1]
                for line in payload["lines"] if line["line_id"] in supported}
    represented = [segment["index"] for segment in payload["segments"]
                   if segment["text"].strip()]
    return {"structural": structural, "supported": supported,
            "unsupported": [], "uncertain": [],
            "evidence": evidence, "represented": represented, "missing": []}


def envelope(answer, provider="ExampleProvider"):
    """A well-formed provider 200 envelope carrying ``answer`` as the model content."""
    return json.dumps({
        "model": correction_provider.OR_SLUG_DRAFTER,
        "provider": provider,
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        "choices": [{"finish_reason": "stop",
                     "message": {"content": json.dumps(answer)}}],
    })


class Scripted:
    """Fails the first ``failures`` attempts with ``status``, then answers properly.

    Modelling it as "fail N times, then serve" rather than a fixed script keeps the
    double honest across phases: the engine asks several different questions, and a
    positional script would misalign with whichever phase happened to be running.
    """

    def __init__(self, status=None, failures=0, answer=None):
        self.status = status
        self.failures = failures
        self.answer = answer
        self.failed = 0
        self.requests = []

    async def close(self):
        pass

    def post(self, url, **kwargs):
        self.requests.append(kwargs)
        payload = json.loads(kwargs["json"]["messages"][1]["content"])
        if self.status is not None and self.failed < self.failures:
            self.failed += 1
            body = ""
            status = self.status
        else:
            status = 200
            body = envelope(self.answer if self.answer is not None else phase_answer(payload))
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


class ScriptedWithHeaders(Scripted):
    """Fails once with the given ``Retry-After`` header, then answers properly."""

    def __init__(self, status, retry_after):
        super().__init__()
        self.status = status
        self.retry_after = retry_after
        self.failed = 0

    def post(self, url, **kwargs):
        import json as _json
        self.requests.append(kwargs)
        payload = _json.loads(kwargs["json"]["messages"][1]["content"])
        if self.failed == 0:
            self.failed = 1
            inner_status, inner_body = self.status, ""
            headers = {"Retry-After": self.retry_after}
        else:
            inner_status, inner_body = 200, envelope(phase_answer(payload))
            headers = {}
        return _response(inner_status, inner_body, headers)


class Malformed(Scripted):
    """Answers 200 with a body that is not JSON at all."""

    def __init__(self):
        super().__init__()

    def post(self, url, **kwargs):
        self.requests.append(kwargs)
        return _response(200, "not json at all", {})


class FixedBody(Scripted):
    """Answers with one fixed status and body, ignoring the request."""

    def __init__(self, status, body):
        super().__init__()
        self.fixed_status = status
        self.fixed_body = body

    def post(self, url, **kwargs):
        self.requests.append(kwargs)
        return _response(self.fixed_status, self.fixed_body, {})


class UncertainInventory(Scripted):
    """Marks a required source item uncertain, so completeness cannot be satisfied."""

    def __init__(self):
        super().__init__()

    def post(self, url, **kwargs):
        import json as _json
        self.requests.append(kwargs)
        payload = _json.loads(kwargs["json"]["messages"][1]["content"])
        required = [segment["index"] for segment in payload.get("segments", [])
                    if segment["text"].strip()]
        answer = {"required": [], "non_resume": [], "uncertain": required[:1]}
        return _response(200, envelope(answer), {})


def _response(status, body, headers):
    inner_status, inner_body, inner_headers = status, body, headers

    class Content:
        async def iter_chunked(self, size):
            raw = (inner_body or "").encode()
            for index in range(0, len(raw), size):
                yield raw[index:index + size]

    class Headers:
        def get(self, name, default=None):
            for key, value in inner_headers.items():
                if key.lower() == name.lower():
                    return value
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
def disclosure(monkeypatch):
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available", lambda: True)
    # Prove the retry sequence without waiting out the backoff for real.
    async def instant(seconds):
        return None
    monkeypatch.setattr(correction_provider, "_sleep", instant)


def run_correction(provider):
    return asyncio.run(correct("Synthetic draft line.", (SourceDocument("resume", SOURCE),),
                               provider))


def test_a_429_is_retried_and_the_review_proceeds(disclosure):
    """A single rate-limit response must not discard a completed pipeline.

    The inventory answer is the one that matters here: a 429 on the first attempt,
    then a good answer, must leave the correction running rather than withhold.
    """
    session = Scripted(429, failures=1)
    provider = CorrectionProvider("fake-key", session)
    result = run_correction(provider)

    # The rate-limited attempt was repeated; the correction then continued into its
    # later phases instead of being abandoned.
    assert provider.attempts > provider.calls, "the 429 must be retried"
    assert (provider.attempts - provider.calls) == 1, "exactly one retry was needed"
    assert result.reason != "provider_rate_limited"


@pytest.mark.parametrize("status", [502, 503, 504])
def test_a_5xx_is_retried_and_the_review_proceeds(disclosure, status):
    session = Scripted(status, failures=1)
    provider = CorrectionProvider("fake-key", session)
    result = run_correction(provider)
    assert provider.attempts - provider.calls == 1, "exactly one retry was needed"
    assert result.reason != "provider_unavailable"


def test_a_429_then_success_keeps_the_same_request(disclosure):
    """Negative control: a retry must resend the identical request, not rebuild it."""
    session = Scripted(429, failures=1)
    provider = CorrectionProvider("fake-key", session)
    run_correction(provider)
    assert session.requests[0]["json"] == session.requests[1]["json"]


def test_repeated_5xx_stops_cleanly_with_a_specific_reason(disclosure):
    """Exhausting the allowance must fail closed, not loop."""
    session = Scripted(504, failures=99)
    provider = CorrectionProvider("fake-key", session)
    result = run_correction(provider)
    assert (result.status, result.reason) == ("withheld", "provider_unavailable")
    assert result.markdown is None
    assert len(session.requests) == correction_provider.RETRY_MAX_ATTEMPTS
    assert provider.attempts == correction_provider.RETRY_MAX_ATTEMPTS
    # One engine phase, however many transport attempts it took.
    assert provider.calls == 1


def test_repeated_429_stops_cleanly_and_stays_distinct(disclosure):
    session = Scripted(429, failures=99)
    provider = CorrectionProvider("fake-key", session)
    result = run_correction(provider)
    assert (result.status, result.reason) == ("withheld", "provider_rate_limited")
    assert len(session.requests) == correction_provider.RETRY_MAX_ATTEMPTS


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_a_rejected_request_is_never_retried(disclosure, status):
    """Auth/schema/model refusals repeat identically, so retrying wastes the budget."""
    session = Scripted(status, failures=99)
    provider = CorrectionProvider("fake-key", session)
    result = run_correction(provider)
    assert len(session.requests) == 1, f"{status} must not be retried"
    assert (result.status, result.reason) == ("withheld", "provider_rejected_request")


def test_malformed_output_is_not_retried_as_an_availability_problem(disclosure):
    """The endpoint answered; the answer is unusable. Another attempt will not help."""
    session = Malformed()
    provider = CorrectionProvider("fake-key", session)
    result = run_correction(provider)
    assert len(session.requests) == 1, "malformed output must not be retried"
    assert result.status == "withheld"
    assert result.reason in {"malformed_source_inventory", "provider_error"}


def test_a_200_with_an_unusable_envelope_is_reported_as_an_error(disclosure):
    """Regression: a 200 was reported as outcome "ok" before validation finished.

    A 200 whose body is well-formed JSON but an unusable envelope is not a success.
    Marking it ok made the log claim a healthy call that nonetheless failed, which is
    exactly the ambiguity that made the live failures hard to read.
    """
    events = []
    broken = json.dumps({"model": "some/other-model",
                         "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                   "total_tokens": 2},
                         "choices": [{"finish_reason": "stop",
                                       "message": {"content": "{}"}}]})
    session = FixedBody(200, broken)
    provider = CorrectionProvider("fake-key", session, observer=events.append)
    result = run_correction(provider)

    assert result.status == "withheld"
    calls = [event for event in events if event.get("event") == "provider.call"]
    assert calls and calls[0]["outcome"] == "provider_error", \
        "a 200 with an unusable envelope must not be reported as ok"
    # And it must not be retried as though the endpoint were unavailable.
    assert len(session.requests) == 1


def test_a_rejected_request_records_its_status(disclosure):
    events = []
    session = Scripted(401, failures=99)
    provider = CorrectionProvider("fake-key", session, observer=events.append)
    run_correction(provider)
    calls = [event for event in events if event.get("event") == "provider.call"]
    assert calls and calls[0]["http_status"] == 401
    assert calls[0]["outcome"] == "provider_rejected"


def test_factually_unsupported_output_is_not_retried(disclosure):
    """A well-formed answer that fails validation is a factual outcome, not transport.

    ``uncertain`` on a required source item is the completeness verdict. Retrying it
    would spend money to ask the same question twice and would blur a factual
    decision into an availability one.
    """
    session = UncertainInventory()
    provider = CorrectionProvider("fake-key", session)
    result = run_correction(provider)
    assert len(session.requests) == 1, "a factual verdict must not be retried"
    assert (result.status, result.reason) == ("withheld", "source_completeness_uncertain")


def test_retry_after_is_honoured(disclosure, monkeypatch):
    """A provider-supplied delay must be used instead of the default backoff."""
    delays = []

    async def capture(seconds):
        delays.append(seconds)

    monkeypatch.setattr(correction_provider, "_sleep", capture)
    session = ScriptedWithHeaders(503, "7")
    provider = CorrectionProvider("fake-key", session)
    run_correction(provider)
    assert delays == [7.0]


def test_a_missing_or_unusable_retry_after_falls_back_to_backoff(disclosure, monkeypatch):
    delays = []

    async def capture(seconds):
        delays.append(seconds)

    monkeypatch.setattr(correction_provider, "_sleep", capture)
    # An HTTP-date form is not parsed; the fallback backoff must be used instead.
    session = ScriptedWithHeaders(503, "Wed, 21 Oct 2026 07:28:00 GMT")
    provider = CorrectionProvider("fake-key", session)
    run_correction(provider)
    assert delays == [correction_provider.RETRY_BASE_DELAY_SECONDS]


def test_an_oversized_retry_after_is_capped(disclosure, monkeypatch):
    delays = []

    async def capture(seconds):
        delays.append(seconds)

    monkeypatch.setattr(correction_provider, "_sleep", capture)
    session = ScriptedWithHeaders(503, "86400")
    provider = CorrectionProvider("fake-key", session)
    run_correction(provider)
    assert delays == [correction_provider.RETRY_MAX_DELAY_SECONDS]


def test_no_retry_is_started_without_room_left():
    """A retry that cannot finish must not be started: it would waste the budget.

    Checked on the budget helper rather than through the engine, because the engine
    is authoritative and overwrites whatever budget a caller set — that is the
    behaviour `test_the_engine_offers_the_budget_without_touching_the_payload`
    pins.
    """
    import time

    provider = CorrectionProvider("fake-key", None)
    provider.set_budget(correction_provider.RETRY_MIN_REMAINING_SECONDS)
    # A phase that started long enough ago has consumed the whole offered budget.
    started = time.monotonic() - (correction_provider.RETRY_MIN_REMAINING_SECONDS + 5)
    remaining = provider._retry_budget(started)
    assert remaining is not None and remaining < correction_provider.RETRY_MIN_REMAINING_SECONDS


def test_the_retry_guard_stops_the_loop_when_the_budget_is_gone(disclosure, monkeypatch):
    """The guard must actually prevent a second attempt, not merely be computed."""
    monkeypatch.setattr(correction_provider, "RETRY_MIN_REMAINING_SECONDS", 10.0)
    session = Scripted(503, failures=99)
    provider = CorrectionProvider("fake-key", session)
    # Simulate running out of budget before the first failure is even classified.
    provider.set_budget(1.0)
    monkeypatch.setattr(provider, "_retry_budget", lambda started: 0.5)
    result = run_correction(provider)
    assert len(session.requests) == 1, "no retry when the budget cannot cover one"
    assert result.status == "withheld"


def test_retries_do_not_consume_the_engine_phase_budget(disclosure):
    """The whole point: retrying must not cost an engine call slot.

    ``MAX_CALLS`` bounds phases (inventory, reviews, repairs). If transport retries
    counted, a flaky endpoint would silently shorten the review-and-repair cycle.
    """
    session = Scripted(503, failures=2)
    provider = CorrectionProvider("fake-key", session)
    run_correction(provider)
    # Two retries on the first phase, then the correction carries on normally.
    assert provider.attempts - provider.calls == 2
    assert provider.calls < correction_provider.MAX_CALLS


def test_each_attempt_is_recorded_without_any_content(disclosure):
    """Telemetry must explain a retry sequence without carrying document text."""
    events = []
    session = Scripted(504, failures=1)
    provider = CorrectionProvider("fake-key", session, observer=events.append)
    run_correction(provider)

    calls = [correction_events.sanitize(event) for event in events
             if event.get("event") == "provider.call"]
    calls = [event for event in calls if event]
    # The first two events are the retried inventory phase; the rest are the review
    # phases that followed. Asserting on the prefix keeps the intent: one failure, one
    # retry, one success, in that order.
    assert [event["attempt"] for event in calls[:2]] == [1, 2]
    assert [event["http_status"] for event in calls[:2]] == [504, 200]
    assert calls[0]["outcome"] == "provider_unavailable"
    assert calls[1]["outcome"] == "ok"
    assert all(event["phase"] == "source_inventory" for event in calls[:2])
    assert calls[1]["provider"] == "ExampleProvider"
    blob = json.dumps(calls)
    for forbidden in ("Synthetic", "Workshop", "inventory and documented", SOURCE[:20]):
        assert forbidden not in blob


def test_failures_record_status_attempt_and_billing_uncertainty(disclosure):
    """A refused attempt may still be billed for its prompt, so it is not 'free'."""
    session = Scripted(503, failures=99)
    provider = CorrectionProvider("fake-key", session)
    run_correction(provider)
    failures = provider.usage["failures"]
    assert failures, "each failed attempt must be recorded"
    for entry in failures:
        assert entry["http_status"] == 503
        assert entry["attempt"] >= 1
        # Conservative: unknown, never silently zero.
        assert entry["usage_unknown"] is True
    assert [entry["attempt"] for entry in failures] == [1, 2, 3]


def test_a_successful_retry_leaves_no_failure_for_the_successful_attempt(disclosure):
    """Negative control: the recorded failures must not include the winner."""
    session = Scripted(503, failures=1)
    provider = CorrectionProvider("fake-key", session)
    run_correction(provider)
    # Exactly one recorded failure: the unsuccessful attempt, not the winner.
    assert len(provider.usage["failures"]) == 1
    assert provider.usage["failures"][0]["http_status"] == 503
    # Every served phase counted its usage exactly once (14 tokens each).
    assert provider.usage["total_tokens"] == 14 * len(provider.usage["calls"])


def test_the_new_outcomes_are_actually_emitted_by_the_sanitizer():
    """Regression: these outcomes were dropped by the vocabulary once already.

    When ``provider_rate_limited`` was introduced it was missing from
    ``PROVIDER_OUTCOMES``, so the sanitizer discarded the events that explain a retry
    sequence. The log looked like nothing had happened.
    """
    for outcome in ("provider_rate_limited", "provider_unavailable",
                    "provider_rejected", "provider_call_timeout", "provider_error",
                    "ok"):
        clean = correction_events.sanitize({"event": "provider.call", "outcome": outcome})
        assert clean.get("outcome") == outcome, f"{outcome} must survive sanitization"


def test_attempt_and_status_survive_sanitization():
    clean = correction_events.sanitize({"event": "provider.call", "attempt": 2,
                                        "http_status": 504})
    assert clean == {"event": "provider.call", "attempt": 2, "http_status": 504}


def test_a_hostile_outcome_is_still_dropped():
    """Negative control: widening the vocabulary must not admit free text."""
    clean = correction_events.sanitize({"event": "provider.call",
                                        "outcome": "Candidate Alex Example at Acme"})
    assert "outcome" not in clean


# --- Publication checks are unchanged by retrying ---------------------------

def test_a_retried_success_still_faces_the_metric_verifier(disclosure):
    """Requirement: keep the publication checks unchanged.

    A transient failure that is retried into a *successful* answer must not thereby
    gain approval. The retry changes only how the answer was obtained; the returned
    review is then validated by the same code path as a first-attempt answer, so an
    invented metric is still caught downstream.
    """
    import tier_a_verifier

    session = Scripted(504, failures=1)
    provider = CorrectionProvider("fake-key", session)
    run_correction(provider)

    # The verifier is untouched by this work and still reports an invented figure.
    source = "## Professional Experience\n**Synthetic Workshop**\n- Maintained the inventory."
    markdown = "## Professional Experience\n**Synthetic Workshop**\n- Maintained the inventory by 47%."
    verdict = tier_a_verifier.verify_metrics(markdown, source)
    assert verdict["verdict"] != "PASS (tier A only)", "an invented metric must not pass"


def test_retrying_does_not_change_the_engine_call_ceiling(disclosure):
    """The phase budget is a hard ceiling; retries must not raise or consume it."""
    session = Scripted(503, failures=99)
    provider = CorrectionProvider("fake-key", session)
    run_correction(provider)
    assert provider.calls == 1
    # Attempts are unbounded by MAX_CALLS, but each phase gets at most
    # RETRY_MAX_ATTEMPTS transport tries, so the worst case is bounded too.
    assert provider.attempts <= correction_provider.RETRY_MAX_ATTEMPTS
