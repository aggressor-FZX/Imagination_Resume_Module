"""Replay of real-provider probe recordings through the real adapter.

**Evidence scope.** The recordings these tests load are probes: a real provider was
called with a *constructed* synthetic draft, driven by
``tools/capture_correction_case.py`` against the correction engine directly. They are
not captures of website ``/analyze`` runs, which were never recorded. A test here
proves how the engine handles the recorded provider bytes; it proves nothing about
what the website did.

These tests need a recording on disk, which is deliberately git-ignored because it
holds complete resume-shaped text. When none is present the tests skip rather than
fail, so a fresh clone is green without the (synthetic) data; regenerate with
``tools/capture_correction_case.py``.

What these do prove:

* the recorded provider bytes are consumed by the real ``CorrectionProvider`` and
  the real ``source_correction.correct`` validator, so the decision is recomputed;
* replay is deterministic and makes no network call;
* an answer that leaves lines unjudged yields ``incomplete_review`` and no repair
  call, rather than a wasted repair followed by ``malformed_output``.
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


def test_probe_02_recorded_an_unjudged_answer_then_a_wasted_repair(no_network):
    """Inspect what probe 02 actually recorded, without replaying past its end.

    The recording holds 3 exchanges: inventory, the review whose answer omitted two
    lines, and a repair the model answered by echoing the lines unchanged. Those are
    facts about this recording and are asserted directly.
    """
    capture = load("live-rewrite-02")
    assert len(capture.calls) == 3
    review_request = capture.calls[1]["request"]
    assert "lines" in review_request
    judged: set[str] = set()
    answer = json.loads(json.loads(capture.calls[1]["response_body"])
                        ["choices"][0]["message"]["content"])
    for key in ("supported", "unsupported", "uncertain", "structural"):
        judged.update(answer.get(key, []))
    omitted = [line["line_id"] for line in review_request["lines"]
               if line["line_id"] not in judged]
    assert omitted, "this probe is only meaningful if lines were omitted"
    # The third exchange is a repair of the very lines that were never judged.
    assert "flagged_claim_ids" in capture.calls[2]["request"]


def test_paraphrased_capture_is_approved(no_network):
    """The realistic paraphrase with a repair is approved by the real validator."""
    capture = load("live-rewrite-01")
    result, provider, session = cc.replay(capture)
    assert (result.status, result.reason) == ("approved", None)
    assert result.markdown is not None
    assert [call["phase"] for call in provider.usage["calls"]] == [
        "source_inventory", "initial_review", "repair", "post_patch_review"]


def test_probe_02_cannot_validate_recovery(no_network):
    """Stated limitation: probe 02 was recorded *before* bounded recovery existed.

    The engine now attempts one completion re-review when an answer omits lines. That
    request does not exist in the recording, so replay cannot answer it and the
    correction ends withheld rather than approved. This is the honest outcome: the
    probe cannot validate the recovery path, and it must not be read as doing so.
    Recovery is covered by scripted tests (``test_correction_recovery.py``); only a
    new live run can validate it against a real model.
    """
    capture = load("live-rewrite-02")
    result, provider, session = cc.replay(capture)
    assert result.status == "withheld"
    # Not ``incomplete_review`` any more: recovery is attempted, and replay has no
    # exchange to answer it with, so the honest outcome is a provider-side failure.
    assert result.reason in {"provider_error", "incomplete_review"}
    assert result.markdown is None
    # The recording's third (wasted repair) exchange is not reproduced as a repair:
    # no repair is performed on a draft with unjudged lines.
    assert result.repair_calls == 0
    assert session.calls <= len(capture.calls)


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
