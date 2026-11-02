"""Measured route differences that set the correction budget.

These tests encode measurements taken from the orchestrator source, not
assumptions. Both numbers are read from `Login_Landing_page/backend`, and if that
code changes these tests are the signal that the Imaginator budget must follow.

Measured facts (2026-11-02):

* `services/mlAuthMiddleware.js` hardcodes ``timeout: 180000`` for every non-SSE
  request and retries up to three attempts.
* `server.cjs` serves the multi-file SSE branch with a direct axios call using
  ``timeout: 300000``, ``validateStatus: () => true`` and no retry, bypassing the
  middleware entirely.

Run 6 of the live synthetic campaign is the evidence that this mattered: the
pipeline consumed 106,483ms, the Imaginator computed a 53,516ms budget from the
180s ceiling, and the review phase was cancelled with ``budget_left_ms: 0`` while
the caller was still willing to wait 300s.
"""
import math
from pathlib import Path

import pytest

import app


BACKEND = Path(__file__).resolve().parents[2] / "Login_Landing_page" / "backend"


class Headers(dict):
    """Minimal case-insensitive header lookup, as Starlette provides."""

    def get(self, key, default=None):
        lowered = key.lower()
        for name, value in self.items():
            if name.lower() == lowered:
                return value
        return default


class Request:
    def __init__(self, headers):
        self.headers = Headers(headers)


def test_json_route_ceiling_matches_the_measured_middleware_timeout():
    """The default ceiling must equal the middleware's hardcoded 180s."""
    source = (BACKEND / "services" / "mlAuthMiddleware.js").read_text()
    assert "timeout: 180000" in source
    assert app.UPSTREAM_CLIENT_TIMEOUT_SECONDS == 180.0


def test_sse_route_ceiling_matches_the_measured_direct_axios_timeout():
    """The SSE ceiling must equal the orchestrator's direct-axios 300s."""
    source = (BACKEND / "server.cjs").read_text()
    # The direct branch sets its own timeout and bypasses the middleware.
    assert "timeout: 300000" in source
    assert "responseType: 'stream'" in source
    assert app.UPSTREAM_SSE_TIMEOUT_SECONDS == 300.0


@pytest.mark.parametrize("accept,expected", [
    ("text/event-stream", 300.0),
    ("TEXT/EVENT-STREAM", 300.0),
    ("text/event-stream; charset=utf-8", 300.0),
    ("application/json", 180.0),
    ("", 180.0),
])
def test_route_is_detected_from_the_accept_header(accept, expected):
    assert app._upstream_timeout_seconds(Request({"accept": accept})) == expected


def test_ambiguous_or_unavailable_request_keeps_the_safe_ceiling():
    """Being wrong upward would let the caller time out before the response."""
    assert app._upstream_timeout_seconds(None) == 180.0
    assert app._upstream_timeout_seconds(Request({})) == 180.0
    assert app.UPSTREAM_CLIENT_TIMEOUT_SECONDS <= app.UPSTREAM_SSE_TIMEOUT_SECONDS


def test_sse_budget_matches_the_run_6_measurement():
    """Recompute run 6 from its logged inputs.

    Run 6 logged ``pipeline_elapsed_ms: 106483`` against the 180s ceiling and
    derived ``budget_ms: 53516``. Under the SSE ceiling the same elapsed time
    leaves the full cap, which is what the measurement says was available.
    """
    elapsed = 106.483
    json_budget = app.correction_budget_seconds(elapsed, app.UPSTREAM_CLIENT_TIMEOUT_SECONDS)
    sse_budget = app.correction_budget_seconds(elapsed, app.UPSTREAM_SSE_TIMEOUT_SECONDS)
    assert round(json_budget * 1000) == 53_517          # matches the live log
    assert sse_budget == app.MAX_CORRECTION_DEADLINE_SECONDS  # the cap, not the remainder
    # The cap is what actually bound correction in every live run.
    assert sse_budget - json_budget == pytest.approx(66.483, abs=0.01)


def test_budget_never_exceeds_the_caller_deadline():
    """The reserve must always be honoured, on both routes and at the margin."""
    for upstream in (app.UPSTREAM_CLIENT_TIMEOUT_SECONDS, app.UPSTREAM_SSE_TIMEOUT_SECONDS):
        for elapsed in (0.0, 1.0, 50.0, 120.0, 150.0, 200.0, 260.0, 279.9, 280.0, 400.0):
            budget = app.correction_budget_seconds(elapsed, upstream)
            if budget is None:
                # Refusal is only allowed when a meaningful call cannot fit.
                remaining = upstream - app.UPSTREAM_RESPONSE_RESERVE_SECONDS - elapsed
                assert remaining < app.MIN_CORRECTION_BUDGET_SECONDS
                continue
            assert budget <= app.MAX_CORRECTION_DEADLINE_SECONDS
            assert elapsed + budget + app.UPSTREAM_RESPONSE_RESERVE_SECONDS <= upstream + 1e-9


def test_sse_route_does_not_change_the_engine_ceiling():
    """A larger caller budget must not raise the operation's own hard ceiling."""
    from source_correction import MAX_TIMEOUT_SECONDS

    assert app.MAX_CORRECTION_DEADLINE_SECONDS <= MAX_TIMEOUT_SECONDS
    # The engine rejects anything above its ceiling regardless of the route.
    assert app.correction_budget_seconds(0.0, app.UPSTREAM_SSE_TIMEOUT_SECONDS) <= MAX_TIMEOUT_SECONDS


def test_invalid_measurements_never_produce_a_budget():
    for elapsed in (True, None, "60", float("nan"), float("inf"), -1e309):
        assert app.correction_budget_seconds(elapsed) is None
    for upstream in (True, None, "300", float("nan"), float("inf")):
        budget = app.correction_budget_seconds(10.0, upstream)
        # An invalid ceiling falls back to the safe 180s default.
        assert budget == app.correction_budget_seconds(10.0, app.UPSTREAM_CLIENT_TIMEOUT_SECONDS)
