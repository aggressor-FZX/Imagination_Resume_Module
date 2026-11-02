"""Access controls and completeness of the live-path capture.

The capture exists so a live website failure can be replayed instead of re-bought.
That makes its controls load-bearing: it must be off by default, refuse non-synthetic
material, refuse to write into Git, write owner-only, and never put text in a log.
"""
import json
import os
import stat
from pathlib import Path

import pytest

import live_capture
from live_capture import (
    LiveCapture,
    LiveCaptureRefused,
    SETTING_DIR,
    SETTING_ENABLED,
    SYNTHETIC_MARKER,
    assert_synthetic,
    assert_writable_directory,
    summarise,
)


SOURCE = f"## Professional Experience\n**{SYNTHETIC_MARKER} Example Works**\n- Did synthetic work.\n"
DRAFT = f"## Professional Experience\n**{SYNTHETIC_MARKER} Example Works**\n- Delivered synthetic work.\n"


@pytest.fixture
def capture_dir(tmp_path):
    """A directory outside the repository, as production must use."""
    target = tmp_path / "captures"
    target.mkdir()
    return target


@pytest.fixture
def enabled(monkeypatch, capture_dir):
    monkeypatch.setenv(SETTING_ENABLED, "1")
    monkeypatch.setenv(SETTING_DIR, str(capture_dir))
    return capture_dir


# --- default off ------------------------------------------------------------

def test_capture_is_off_by_default(monkeypatch):
    monkeypatch.delenv(SETTING_ENABLED, raising=False)
    monkeypatch.delenv(SETTING_DIR, raising=False)
    assert live_capture.enabled() is False
    assert live_capture.start("abc123") is None


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_only_explicit_values_enable_capture(monkeypatch, value):
    monkeypatch.setenv(SETTING_ENABLED, value)
    assert live_capture.enabled() is False
    assert live_capture.start("abc123") is None


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_explicit_values_enable_capture(monkeypatch, value, capture_dir):
    monkeypatch.setenv(SETTING_ENABLED, value)
    monkeypatch.setenv(SETTING_DIR, str(capture_dir))
    assert live_capture.enabled() is True
    assert isinstance(live_capture.start("abc123"), LiveCapture)


def test_start_needs_a_safe_correlation_id(enabled):
    assert live_capture.start("") is None
    assert live_capture.start("../../etc/passwd") is None
    assert live_capture.start("a" * 200) is None
    assert isinstance(live_capture.start("550e8400-e29b-41d4-a716-446655440000"), LiveCapture)


# --- synthetic only ---------------------------------------------------------

def test_unmarked_material_is_refused():
    with pytest.raises(LiveCaptureRefused):
        assert_synthetic(["Alex Example, a real resume"])
    with pytest.raises(LiveCaptureRefused):
        assert_synthetic([])
    assert_synthetic([SOURCE, DRAFT])


def test_write_refuses_a_non_synthetic_source(enabled):
    capture = LiveCapture("case-1")
    capture.record_sources([("resume", "Alex Example, a real resume")])
    capture.record_draft(DRAFT)
    with pytest.raises(LiveCaptureRefused):
        capture.write()


# --- never in Git -----------------------------------------------------------

def test_repository_location_is_refused_unless_ignored():
    repo_captures = Path(live_capture._REPO_ROOT) / "some_unignored_dir"
    with pytest.raises(LiveCaptureRefused):
        assert_writable_directory(repo_captures)


def test_git_ignored_repository_location_is_allowed():
    """`captures/` is git-ignored, so writing there is permitted by the guard."""
    assert_writable_directory(Path(live_capture._REPO_ROOT) / "captures")


def test_capture_dir_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv(SETTING_DIR, raising=False)
    assert live_capture.capture_dir() is None


# --- owner-only, single file ------------------------------------------------

def test_capture_is_written_owner_only_and_readable(enabled):
    capture = LiveCapture("case-2")
    capture.record_sources([("resume", SOURCE)])
    capture.record_draft(DRAFT)
    capture.record_request("source_inventory", {"contract": "c", "segments": []})
    capture.record_response(0, 200, '{"required":[]}')
    path = capture.write()

    assert path is not None and path.exists()
    assert path.stat().st_mode & (stat.S_IRGRP | stat.S_IROTH) == 0
    assert path.parent.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO) == 0
    stored = json.loads(path.read_text())
    assert stored["correlation_id"] == "case-2"
    assert stored["synthetic"] == SYNTHETIC_MARKER
    assert stored["sources"][0]["text"] == SOURCE
    assert stored["draft"] == DRAFT
    assert stored["exchanges"][0]["response_body"] == '{"required":[]}'


def test_write_is_a_noop_when_capture_is_off(monkeypatch, capture_dir):
    monkeypatch.delenv(SETTING_ENABLED, raising=False)
    capture = LiveCapture("case-3")
    capture.record_sources([("resume", SOURCE)])
    capture.record_draft(DRAFT)
    assert capture.write(capture_dir) is None
    assert list(capture_dir.iterdir()) == []


def test_response_recording_ignores_an_unknown_index():
    capture = LiveCapture("case-4")
    capture.record_response(None, 200, "body")
    capture.record_response(99, 200, "body")
    assert capture.data["exchanges"] == []


# --- nothing textual in summaries ------------------------------------------

def test_summary_contains_no_document_text(enabled):
    capture = LiveCapture("case-5")
    capture.record_sources([("resume", SOURCE)])
    capture.record_draft(DRAFT)
    summary = summarise(capture)
    blob = json.dumps(summary)
    assert "Example Works" not in blob
    assert "synthetic work" not in blob
    assert summary == {"live_capture": "on", "exchanges": 0, "source_documents": 1,
                       "draft_chars": len(DRAFT), "responses_recorded": 0}


def test_gate_summary_keeps_only_identifiers_and_numbers():
    capture = LiveCapture("case-6")
    capture.record_gate(["unsupported_claims", 3, {"free text": "PRIVATE_DRAFT_TEXT"}])
    blob = json.dumps(capture.data["gate"])
    assert "PRIVATE_DRAFT_TEXT" not in blob
    assert "unsupported_claims" in blob


def test_timing_records_numbers_and_ids_only():
    capture = LiveCapture("case-7")
    capture.record_timing(budget_ms=1000, reason="approved", note="PRIVATE_TEXT")
    blob = json.dumps(capture.data["timing"])
    assert "PRIVATE_TEXT" not in blob
    assert capture.data["timing"]["budget_ms"] == 1000
    assert capture.data["timing"]["correlation_id"] == "case-7"


def test_summary_of_absent_capture_is_explicit():
    assert summarise(None) == {"live_capture": "off"}
