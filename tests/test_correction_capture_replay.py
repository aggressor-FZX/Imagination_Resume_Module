"""Replay of captured real-model synthetic cases through the real adapter.

These tests need a capture on disk, which is deliberately git-ignored because it
contains complete resume-shaped text. When no capture is present the tests skip
rather than fail, so a fresh clone is green without the (synthetic) data; the
capture is regenerated with ``tools/capture_correction_case.py``.

What these prove:

* the recorded provider bytes are consumed by the real ``CorrectionProvider`` and
  the real ``source_correction.correct`` validator, so the decision is recomputed;
* replay is deterministic and makes no network call;
* a failure that production reported as ``malformed_output`` is now attributed to
  the model leaving lines unjudged, and no repair call is wasted on it.
"""
import json
import socket

import pytest

import correction_capture as cc


CASES = ("live-rewrite-01", "live-rewrite-02")


def load(case_id):
    try:
        return cc.load_capture(case_id)
    except FileNotFoundError:
        pytest.skip(f"capture {case_id} not present; see tools/capture_correction_case.py")


@pytest.fixture
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("replay must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


def test_capture_is_marked_synthetic_and_owner_only():
    capture = load("live-rewrite-01")
    assert cc.SYNTHETIC_MARKER in capture.source_text
    assert cc.SYNTHETIC_MARKER in capture.draft_text
    from pathlib import Path
    import stat
    path = Path("captures") / f"{capture.case_id}.json"
    assert path.stat().st_mode & (stat.S_IRGRP | stat.S_IROTH) == 0


def test_capture_refuses_unmarked_text():
    """A capture of a non-synthetic document is refused, not written."""
    with pytest.raises(cc.CaptureRefused):
        cc.assert_synthetic("case", "Alex Example, real resume", "draft text")
    with pytest.raises(cc.CaptureRefused):
        cc.assert_synthetic("case", f"{cc.SYNTHETIC_MARKER} source", "no marker here")


def test_capture_refuses_wrong_schema_or_tampered_text():
    capture = load("live-rewrite-01")
    payload = capture.as_dict()
    with pytest.raises(cc.CaptureRefused):
        cc.Capture.from_dict({**payload, "schema": 99})
    with pytest.raises(cc.CaptureRefused):
        cc.Capture.from_dict({**payload, "source_text": payload["source_text"] + "x"})


@pytest.mark.parametrize("case_id", CASES)
def test_captured_case_replays_offline_and_deterministically(no_network, case_id):
    capture = load(case_id)
    first, provider, session = cc.replay(capture)
    second, _, session2 = cc.replay(capture)

    # Deterministic: the same bytes produce the same decision.
    assert (first.status, first.reason) == (second.status, second.reason)
    assert (first.review_calls, first.repair_calls) == (second.review_calls, second.repair_calls)
    # Replay never consumes more exchanges than were recorded, so it cannot invent
    # a phase production did not perform.
    assert session.calls == session2.calls
    assert session.calls <= len(capture.calls)
    # Usage is recomputed from the recorded bodies, not trusted from the file.
    assert provider.usage["total_tokens"] > 0


def test_replay_stops_before_a_repair_that_production_wasted(no_network):
    """Case 02 consumed 3 provider calls live; only 2 were justified.

    Production sent the unjudged draft to repair and the model returned it
    unchanged, so the third exchange produced nothing. Replay performs the
    justified work only, which is why it consumes fewer calls than were recorded.
    """
    capture = load("live-rewrite-02")
    _, _, session = cc.replay(capture)
    assert len(capture.calls) == 3
    assert session.calls == 2
    # The recorded repair exchange is still on disk for audit; it is simply not
    # needed to reproduce the corrected decision.
    assert "flagged_claim_ids" in json.dumps(capture.calls[2]["request"])


def test_paraphrased_capture_is_approved(no_network):
    """The realistic paraphrase with a repair is approved by the real validator."""
    capture = load("live-rewrite-01")
    result, provider, session = cc.replay(capture)
    assert (result.status, result.reason) == ("approved", None)
    assert result.markdown is not None
    assert [call["phase"] for call in provider.usage["calls"]] == [
        "source_inventory", "initial_review", "repair", "post_patch_review"]


def test_unjudged_line_capture_is_attributed_without_a_wasted_repair(no_network):
    """The live failure was reported as malformed_output and burned a repair call.

    The captured answer parses but leaves two lines out of every verdict list. That
    is an incomplete review, not a broken protocol, and no repair may be attempted
    because a repair can only replace an already-flagged span.
    """
    capture = load("live-rewrite-02")
    result, provider, session = cc.replay(capture)
    assert result.status == "withheld"
    assert result.reason == "incomplete_review"
    assert result.repair_calls == 0
    # Two calls: inventory + the one review whose answer was incomplete.
    assert session.calls == 2
    assert [call["phase"] for call in provider.usage["calls"]] == [
        "source_inventory", "initial_review"]


def test_replay_never_reports_a_reason_outside_the_public_set(no_network):
    """Every replay outcome must be a reason the route is allowed to forward."""
    import app
    import inspect
    source = inspect.getsource(app._analyze_resume_impl)
    allowed = {
        "invalid_input", "malformed_source_inventory", "source_completeness_uncertain",
        "no_required_source_claims", "provider_error", "no_supported_claims",
        "source_facts_missing", "unresolved_claims", "malformed_output",
        "privacy_review_required", "privacy_redaction_failed", "attribution_conflict",
        "incomplete_review", "no_op_repair",
    }
    for case_id in CASES:
        result, _, _ = cc.replay(load(case_id))
        assert result.reason is None or result.reason in allowed
        if result.reason:
            assert f'"{result.reason}"' in source, (
                f"{result.reason} is not forwarded by the route")
