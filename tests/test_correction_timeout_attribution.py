"""Phase attribution, budget propagation, sanitized events and output allowance.

Offline and synthetic. Drives the real /analyze route, correction engine and
CorrectionProvider. Only the transport (scripted session) and the pipeline
(fixed draft, optional delay) are replaced.
"""
import asyncio
import json
import time
from unittest.mock import MagicMock

import pytest

import app
import correction_events
from correction_provider import MAX_OUTPUT_TOKENS, PROVIDER_TIMEOUT_SECONDS
from pipeline_config import OR_SLUG_DRAFTER
from source_correction import MAX_SOURCE_CHARS
from tests.test_analyze_source_correction_route import (  # noqa: F401
    offline, pipeline, post, synthetic_disclosure_bypass,
)
from tests.test_correction_integration_offline import (
    DRAFT, FRAGMENT, SOURCE, Response, inventory, review,
)

PHASES = ["source_inventory", "initial_review", "repair", "post_patch_review"]
SENTINEL = "PRIVATE_SENTINEL_7731"


class HangingResponse:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        try:
            await asyncio.Event().wait()
        finally:
            self.session.cancelled = True

    async def __aexit__(self, *args):
        return False


class ScriptedSession:
    """One scripted behaviour per provider request, in order.

    "ok" answers from the synthetic inventory/review fixtures; "provider_timeout"
    raises the timeout a single aiohttp request produces; "hang" never answers.
    After the script is exhausted, requests succeed.
    """

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.cancelled = False

    async def close(self):
        pass

    def post(self, url, **kwargs):
        self.requests.append(kwargs)
        behaviour = self.script.pop(0) if self.script else "ok"
        if behaviour == "provider_timeout":
            raise asyncio.TimeoutError()
        if behaviour == "hang":
            return HangingResponse(self)
        payload = json.loads(kwargs["json"]["messages"][1]["content"])
        if "flagged_claim_ids" in payload:
            body = {"edits": [{"claim_id": cid, "replacement": ""}
                              for cid in payload["flagged_claim_ids"]]}
        elif "lines" not in payload:
            body = inventory(payload)
        else:
            body = review(payload)
        return Response(json.dumps({
            "model": OR_SLUG_DRAFTER,
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(body)}}],
        }).encode())


@pytest.fixture
def logger(monkeypatch):
    recorder = MagicMock()
    monkeypatch.setattr(app, "logger", recorder)
    return recorder


def correction_lines(logger):
    """Parsed sanitized correction events emitted through the app logger."""
    lines = []
    for call in logger.info.call_args_list:
        if call.args and call.args[0] == "correction_event %s":
            lines.append(json.loads(call.args[1]))
    return lines


def named(lines, event, **match):
    return [line for line in lines if line.get("event") == event
            and all(line.get(key) == value for key, value in match.items())]


def delay_pipeline(monkeypatch, delay):
    pipeline(monkeypatch)
    inner = app.run_new_pipeline_async

    async def slow(**kwargs):
        await asyncio.sleep(delay)
        return await inner(**kwargs)

    monkeypatch.setattr(app, "run_new_pipeline_async", slow)


def scale_upstream(monkeypatch):
    """Same arithmetic as production, scaled down; margins kept for slow CI."""
    monkeypatch.setattr(app, "UPSTREAM_CLIENT_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(app, "UPSTREAM_RESPONSE_RESERVE_SECONDS", 0.4)
    monkeypatch.setattr(app, "MAX_CORRECTION_DEADLINE_SECONDS", 5.0)
    monkeypatch.setattr(app, "MIN_CORRECTION_BUDGET_SECONDS", 0.05)


# --- 1. Provider-call timeout versus overall deadline, per phase --------------

@pytest.mark.parametrize("index,phase", list(enumerate(PHASES)), ids=PHASES)
def test_provider_call_timeout_is_reported_for_the_phase_that_failed(
        offline, monkeypatch, synthetic_disclosure_bypass, logger, index, phase):
    client, _ = offline
    pipeline(monkeypatch)
    session = ScriptedSession(["ok"] * index + ["provider_timeout"])
    app.app.state.http_session = session

    result = post(client).json()

    assert result["correction_status"] == "withheld" and result["publication_receipt"] is None
    assert result["correction_reason"] == f"provider_call_timeout_{phase}"
    assert len(session.requests) == index + 1, "no retry and no later phase after a timeout"
    lines = correction_lines(logger)
    assert named(lines, "correction.phase_end", phase=phase, outcome="provider_call_timeout")
    assert named(lines, "correction.end", reason=f"provider_call_timeout_{phase}", status="withheld")


@pytest.mark.parametrize("index,phase", list(enumerate(PHASES)), ids=PHASES)
def test_overall_deadline_is_reported_for_the_phase_in_flight(
        offline, monkeypatch, synthetic_disclosure_bypass, logger, index, phase):
    client, _ = offline
    pipeline(monkeypatch)
    monkeypatch.setattr(app, "MAX_CORRECTION_DEADLINE_SECONDS", 0.5)
    monkeypatch.setattr(app, "MIN_CORRECTION_BUDGET_SECONDS", 0.05)
    session = ScriptedSession(["ok"] * index + ["hang"])
    app.app.state.http_session = session

    started = time.monotonic()
    result = post(client).json()
    elapsed = time.monotonic() - started

    assert result["correction_reason"] == f"timeout_{phase}"
    assert result["requires_review"] and result["publication_receipt"] is None
    assert len(session.requests) == index + 1 and session.cancelled
    # Terminates at the deadline, not at the 60s per-call timeout.
    assert elapsed < PROVIDER_TIMEOUT_SECONDS / 4
    lines = correction_lines(logger)
    assert named(lines, "correction.phase_end", phase=phase, outcome="deadline")
    assert named(lines, "correction.end", reason=f"timeout_{phase}", status="withheld")


def test_the_two_failure_modes_are_distinct_reasons_for_the_same_phase(
        offline, monkeypatch, synthetic_disclosure_bypass, logger):
    client, _ = offline
    pipeline(monkeypatch)
    app.app.state.http_session = ScriptedSession(["provider_timeout"])
    provider_reason = post(client).json()["correction_reason"]
    monkeypatch.setattr(app, "MAX_CORRECTION_DEADLINE_SECONDS", 0.3)
    monkeypatch.setattr(app, "MIN_CORRECTION_BUDGET_SECONDS", 0.05)
    app.app.state.http_session = ScriptedSession(["hang"])
    deadline_reason = post(client).json()["correction_reason"]
    assert provider_reason == "provider_call_timeout_source_inventory"
    assert deadline_reason == "timeout_source_inventory"
    assert provider_reason != deadline_reason


# --- 2. Budget is what remains after the pipeline; termination inside upstream -

def test_pipeline_time_is_subtracted_and_the_request_ends_inside_the_upstream_timeout(
        offline, monkeypatch, synthetic_disclosure_bypass, logger):
    client, _ = offline
    scale_upstream(monkeypatch)
    delay_pipeline(monkeypatch, 0.3)
    session = ScriptedSession(["hang"])
    app.app.state.http_session = session

    started = time.monotonic()
    result = post(client).json()
    wall = time.monotonic() - started

    lines = correction_lines(logger)
    budget = named(lines, "correction.budget")[0]
    assert budget["pipeline_elapsed_ms"] >= 300
    # budget = min(5.0, 2.0 - 0.4 - elapsed) -> about 1300 ms, not the 5s maximum.
    assert 1200 <= budget["budget_ms"] <= 1300
    # The sum the orchestrator cares about: pipeline + correction + reserve <= upstream.
    assert (budget["pipeline_elapsed_ms"] + budget["budget_ms"] + budget["reserve_ms"]
            <= budget["upstream_timeout_ms"])
    assert result["correction_reason"] == "timeout_source_inventory"
    assert wall < 2.0, f"request took {wall:.3f}s against a 2.0s upstream timeout"
    start = named(lines, "correction.phase_start", phase="source_inventory")[0]
    # Phase budget is the remaining deadline at call time; rounding may differ by 1 ms.
    assert abs(start["budget_ms"] - budget["budget_ms"]) <= 2


def test_no_provider_call_is_made_when_the_pipeline_leaves_too_little_time(
        offline, monkeypatch, synthetic_disclosure_bypass, logger):
    client, _ = offline
    scale_upstream(monkeypatch)
    delay_pipeline(monkeypatch, 1.8)
    session = ScriptedSession([])
    app.app.state.http_session = session

    result = post(client).json()

    assert result["correction_reason"] == "pipeline_budget_exhausted"
    assert result["requires_review"] and result["publication_receipt"] is None
    assert session.requests == []
    budget = named(correction_lines(logger), "correction.budget")[0]
    assert budget["budget_ms"] == 0 and budget["pipeline_elapsed_ms"] >= 1800


@pytest.mark.parametrize("elapsed", [0, 16, 37, 63, 100, 140, 149.9])
def test_budget_never_lets_pipeline_plus_correction_exceed_the_upstream_timeout(elapsed):
    budget = app.correction_budget_seconds(elapsed)
    if budget is None:
        return
    assert 0 < budget <= app.MAX_CORRECTION_DEADLINE_SECONDS
    assert elapsed + budget + app.UPSTREAM_RESPONSE_RESERVE_SECONDS <= app.UPSTREAM_CLIENT_TIMEOUT_SECONDS + 1e-9


def test_budget_is_monotonic_and_withheld_below_the_minimum():
    previous = None
    withheld = False
    for tenth in range(0, 1801):
        budget = app.correction_budget_seconds(tenth / 10)
        if budget is None:
            withheld = True
            continue
        assert not withheld, "once withheld, a later pipeline time cannot regain budget"
        if previous is not None:
            assert budget <= previous + 1e-9
        previous = budget
    assert app.correction_budget_seconds(16) == pytest.approx(120.0)
    # The 63s pipeline time observed in an earlier live run: the old fixed 120s
    # deadline would end at 203s, past the 180s orchestrator timeout.
    assert 63 + 120 + app.UPSTREAM_RESPONSE_RESERVE_SECONDS > app.UPSTREAM_CLIENT_TIMEOUT_SECONDS
    assert app.correction_budget_seconds(63) == pytest.approx(97.0)
    assert app.correction_budget_seconds(150) is None
    assert app.correction_budget_seconds(float("nan")) is None
    assert app.correction_budget_seconds(True) is None


# --- 3. Phase events: useful metadata, no document, prompt or response text ----

def test_events_carry_phase_timing_and_token_metadata_for_a_completed_run(
        offline, monkeypatch, synthetic_disclosure_bypass, logger):
    client, _ = offline
    pipeline(monkeypatch)
    app.app.state.http_session = ScriptedSession([])

    result = post(client).json()

    assert result["correction_status"] == "approved"
    lines = correction_lines(logger)
    phases = [line["phase"] for line in named(lines, "correction.phase_start")]
    assert phases == PHASES
    for line in named(lines, "provider.call"):
        assert line["outcome"] == "ok" and line["finish_reason"] == "stop"
        assert line["prompt_tokens"] == 10 and line["completion_tokens"] == 4
        assert line["max_tokens"] == MAX_OUTPUT_TOKENS
        assert line["call_timeout_ms"] == int(PROVIDER_TIMEOUT_SECONDS * 1000)
    end = named(lines, "correction.end")[0]
    assert end["status"] == "approved" and end["reviews"] == 2 and end["repairs"] == 1
    correlation = {line["correlation_id"] for line in lines}
    assert len(correlation) == 1


@pytest.mark.parametrize("script", [["ok"] * 4, ["provider_timeout"]], ids=["approved", "timeout"])
def test_logged_events_never_contain_document_draft_or_model_text(
        offline, monkeypatch, synthetic_disclosure_bypass, logger, script):
    client, _ = offline
    source = SOURCE.replace("Synthetic Workshop", f"Synthetic Workshop {SENTINEL}")
    draft = DRAFT.replace("Synthetic Workshop", f"Synthetic Workshop {SENTINEL}")
    pipeline(monkeypatch, draft=draft)
    app.app.state.http_session = ScriptedSession(script)

    post(client, source=source)

    lines = correction_lines(logger)
    assert lines, "phase events should be emitted"
    rendered = json.dumps(lines)
    for forbidden in (SENTINEL, "Synthetic", FRAGMENT.strip(), "Maintained", "Improved"):
        assert forbidden not in rendered
    for call in logger.info.call_args_list:
        if call.args and call.args[0] == "correction_event %s":
            assert SENTINEL not in call.args[1]
    for line in lines:
        assert line["event"] in correction_events.EVENTS
        assert set(line) <= correction_events._FIELDS | {"event"}


def test_sanitizer_drops_free_text_unknown_keys_and_unknown_events():
    assert correction_events.sanitize({"event": "not.an.event", "reason": "x"}) == {}
    clean = correction_events.sanitize({
        "event": "correction.end",
        "status": "withheld",
        "reason": "Model said: the candidate is Alex Example",
        "elapsed_ms": 12,
        "document": "Alex Example at Acme",
        "phase": "not_a_phase",
        "outcome": "free text with spaces",
        "cost_usd": float("nan"),
        "correlation_id": "bad id with spaces",
    })
    assert clean == {"event": "correction.end", "status": "withheld", "elapsed_ms": 12}
    assert correction_events.sanitize({"event": "provider.call", "model": "qwen/qwen3-coder",
                                       "prompt_tokens": True}) == {
        "event": "provider.call", "model": "qwen/qwen3-coder"}


def test_events_join_the_orchestrator_correlation_id_when_supplied(
        offline, monkeypatch, synthetic_disclosure_bypass, logger):
    client, _ = offline
    pipeline(monkeypatch)
    app.app.state.http_session = ScriptedSession(["provider_timeout"])
    response = client.post("/analyze", headers={"X-Cogito-Owner-Id": "offline-owner",
                                                "x-correlation-id": "235c04c6-7113-426b-8253-f680be6f547a"},
                           json={"resume_text": SOURCE, "job_ad": "Synthetic equipment maintenance role"})
    assert response.status_code == 200
    lines = correction_lines(logger)
    assert {line["correlation_id"] for line in lines} == {"235c04c6-7113-426b-8253-f680be6f547a"}

    # Unsafe or oversized identifiers are not logged; a generated ID is used instead.
    response = client.post("/analyze", headers={"X-Cogito-Owner-Id": "offline-owner",
                                                "x-correlation-id": "bad id with spaces"},
                           json={"resume_text": SOURCE, "job_ad": "Synthetic equipment maintenance role"})
    assert response.status_code == 200
    recent = correction_lines(logger)[len(lines):]
    assert recent and all("bad id" not in line["correlation_id"] for line in recent)


def test_observer_failures_never_change_the_correction_outcome(
        offline, monkeypatch, synthetic_disclosure_bypass):
    client, _ = offline
    pipeline(monkeypatch)
    app.app.state.http_session = ScriptedSession([])

    def broken_logger(*args, **kwargs):
        if args and args[0] == "correction_event %s":
            raise RuntimeError("logging backend unavailable")

    recorder = MagicMock()
    recorder.info.side_effect = broken_logger
    monkeypatch.setattr(app, "logger", recorder)
    result = post(client).json()
    assert result["correction_status"] == "approved"


# --- 4. Output allowance: why inventory and review need their output size --------

def _inventory_output_chars(source_text):
    payload = {"contract": "", "sources": [{"source_id": "resume", "text": source_text}]}
    return len(json.dumps(inventory(payload), ensure_ascii=False))


def test_output_allowance_covers_small_inputs_but_not_the_input_ceiling():
    """Lower-bound estimate from the fixture's minimal inventory shape.

    The inventory must return every span with an id, offsets, a copied quote
    and a classification, so output grows with input. A real model's output is
    at least this long. Pessimistic 3 chars/token for the small case and
    optimistic 4 chars/token for the maximum case keep both assertions robust.
    """
    small_chars = _inventory_output_chars(SOURCE)
    repeats = MAX_SOURCE_CHARS // len(SOURCE)
    maximum_source = (SOURCE * repeats)[:MAX_SOURCE_CHARS]
    assert len(maximum_source) <= MAX_SOURCE_CHARS
    maximum_chars = _inventory_output_chars(maximum_source)

    assert small_chars / 3 < MAX_OUTPUT_TOKENS, "small sources fit the allowance"
    assert maximum_chars / 4 > MAX_OUTPUT_TOKENS, (
        "the maximum accepted source cannot fit the allowance; a truncated reply is withheld")
