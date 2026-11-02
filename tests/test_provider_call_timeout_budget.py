"""Per-call timeout must be sized from the budget that actually exists.

Measured live on 2026-11-02: run 3 granted itself 120s, spent ~48s on inventory, then
abandoned the initial review at a fixed 60s per-call ceiling with ~12s of its own
budget unspent. Replaying the exact captured review request returned 200 in 38.3s, so
the request was never too slow — the fixed ceiling was simply smaller than the work.

The engine now offers each call the remaining budget out-of-band, and the adapter
takes the larger of its own floor and that budget, bounded by a ceiling so one slow
phase cannot consume the whole correction.
"""
import asyncio
import json

import pytest

import correction_provider
from correction_provider import (
    CorrectionProvider,
    PROVIDER_TIMEOUT_CEILING_SECONDS,
    PROVIDER_TIMEOUT_SECONDS,
)
from source_correction import SourceDocument, correct


SOURCE = """## Professional Experience
**Synthetic Workshop**
- Maintained the workshop inventory and documented repair procedures for service technicians.
"""


class Fixed:
    """Records the timeout the adapter asked for, then fails closed."""

    def __init__(self):
        self.requests = []

    async def close(self):
        pass

    def post(self, url, **kwargs):
        self.requests.append(kwargs)
        raise RuntimeError("stop before the network")


@pytest.fixture
def disclosure(monkeypatch):
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available", lambda: True)


def _run(provider):
    return asyncio.run(correct("Synthetic draft line.", (SourceDocument("resume", SOURCE),),
                               provider))


def test_a_large_remaining_budget_raises_the_call_timeout(disclosure):
    session = Fixed()
    provider = CorrectionProvider("fake-key", session)
    _run(provider)
    asked = session.requests[0]["timeout"].total
    # The engine granted 100s, which is more than the old fixed 60s ceiling.
    assert asked == pytest.approx(PROVIDER_TIMEOUT_CEILING_SECONDS)
    assert asked > PROVIDER_TIMEOUT_SECONDS


def test_the_call_timeout_is_capped_by_the_ceiling(disclosure):
    """One slow phase must not be able to consume the whole correction."""
    session = Fixed()
    provider = CorrectionProvider("fake-key", session)
    provider.set_budget(10_000.0)
    _run(provider)
    assert session.requests[0]["timeout"].total == pytest.approx(PROVIDER_TIMEOUT_CEILING_SECONDS)


def test_a_small_budget_never_lowers_the_floor():
    """Negative control: a nearly-exhausted budget must not shrink a call below 60s.

    The engine already refuses to start a phase with no room left, so shrinking the
    transport timeout here would only convert a viable call into an instant failure.
    This is checked on the helper directly, because the engine overwrites whatever
    budget a caller set.
    """
    provider = CorrectionProvider("fake-key", None)
    provider.set_budget(1.0)
    assert provider._call_timeout() == pytest.approx(PROVIDER_TIMEOUT_SECONDS)


def test_a_caller_that_never_sets_a_budget_sees_the_old_value():
    """Negative control: the floor still applies when no budget was offered."""
    provider = CorrectionProvider("fake-key", None)
    assert provider._remaining_seconds is None
    assert provider._call_timeout() == pytest.approx(PROVIDER_TIMEOUT_SECONDS)


def test_an_invalid_budget_is_ignored():
    provider = CorrectionProvider("fake-key", None)
    for bad in (0, -5, None, "80", float("nan"), float("inf")):
        provider.set_budget(bad)
        assert provider._remaining_seconds is None
    assert provider._call_timeout() == PROVIDER_TIMEOUT_SECONDS


def test_the_engine_offers_the_budget_without_touching_the_payload():
    """The payload key set is a pinned contract; the hint must stay out-of-band."""
    offered = []
    seen = []

    class Adapter:
        def set_budget(self, seconds):
            offered.append(seconds)

        async def __call__(self, phase, payload):
            seen.append(sorted(payload))
            raise RuntimeError("stop")

    corpus = (SourceDocument("resume", SOURCE),)
    asyncio.run(correct("Synthetic draft line.", corpus, Adapter()))
    assert offered and offered[0] > 0
    # Only the documented keys, and no underscore-prefixed hint.
    assert seen == [["contract", "sources"]]


def test_the_observed_event_reports_the_applied_timeout(disclosure):
    """The sanitized event must carry the value actually used, not a constant.

    The engine is authoritative: it offers its own remaining budget before each
    phase, so a value set by hand is deliberately overwritten.
    """
    events = []
    session = Fixed()
    provider = CorrectionProvider("fake-key", session, observer=events.append)
    _run(provider)
    calls = [event for event in events if event.get("event") == "provider.call"]
    assert calls
    # The engine granted ~100s, so the ceiling applied rather than the 60s floor.
    # Compared with a tolerance because the engine subtracts elapsed wall time.
    assert calls[0]["call_timeout_ms"] == pytest.approx(PROVIDER_TIMEOUT_CEILING_SECONDS * 1000,
                                                        abs=1000)
    assert calls[0]["call_timeout_ms"] > PROVIDER_TIMEOUT_SECONDS * 1000
