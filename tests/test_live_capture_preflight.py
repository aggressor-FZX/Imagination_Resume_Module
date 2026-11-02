"""The preflight must catch a misconfigured capture before a paid run.

A capture that silently fails to write wastes the run it was meant to preserve, and the
failure is invisible because capture never breaks a request. Preflight is the guard, so
its outcomes are pinned here.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

import live_capture
from live_capture import SETTING_DIR, SETTING_ENABLED


TOOL = Path(__file__).resolve().parents[1] / "tools" / "live_capture_preflight.py"
ROOT = Path(__file__).resolve().parents[1]


def run_preflight(**env):
    base = {key: value for key, value in os.environ.items()
            if key not in {SETTING_ENABLED, SETTING_DIR}}
    base.update(env)
    base["PYTHONPATH"] = str(ROOT)
    return subprocess.run([sys.executable, str(TOOL)], env=base,
                          capture_output=True, text=True, timeout=60)


def test_preflight_reports_disabled_when_unset():
    result = run_preflight()
    assert result.returncode == 3
    assert "DISABLED" in result.stdout


def test_preflight_reports_missing_directory():
    result = run_preflight(**{SETTING_ENABLED: "1"})
    assert result.returncode == 2
    assert "NOT READY" in result.stdout


def test_preflight_refuses_an_unsafe_directory():
    result = run_preflight(**{SETTING_ENABLED: "1", SETTING_DIR: str(ROOT)})
    assert result.returncode == 2
    assert "git-ignored" in result.stdout


def test_preflight_succeeds_for_a_safe_directory(tmp_path):
    target = tmp_path / "captures"
    result = run_preflight(**{SETTING_ENABLED: "1", SETTING_DIR: str(target)})
    assert result.returncode == 0, result.stdout
    assert "READY" in result.stdout
    # It must clean up after itself rather than leaving a probe file behind.
    assert list(target.iterdir()) == []


def test_preflight_prints_no_document_text(tmp_path):
    target = tmp_path / "captures"
    result = run_preflight(**{SETTING_ENABLED: "1", SETTING_DIR: str(target)})
    # The probe text it writes is synthetic; nothing of it should be echoed.
    assert live_capture.SYNTHETIC_MARKER not in result.stdout
    assert "preflight draft" not in result.stdout
