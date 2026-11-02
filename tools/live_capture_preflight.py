#!/usr/bin/env python3
"""Preflight: verify live capture is actually ready before spending a run.

A capture that silently fails to write wastes the very run it was meant to preserve.
This checks every condition up front and exits non-zero on the first problem, so it can
be run as a one-off task on the deployed service before an analysis.

It prints no document text: at most the resolved path, sizes and the failure reason.

Usage::

    PYTHONPATH=. python tools/live_capture_preflight.py

Exit codes: 0 ready, 2 not ready, 3 capture disabled.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import live_capture


def main() -> int:
    problems: list[str] = []

    if not live_capture.enabled():
        print(f"DISABLED: set {live_capture.SETTING_ENABLED}=1 to enable capture")
        return 3

    directory = live_capture.capture_dir()
    if directory is None:
        print(f"NOT READY: {live_capture.SETTING_DIR} is unset")
        return 2
    print(f"directory: {directory}")

    try:
        live_capture.assert_writable_directory(directory)
    except live_capture.LiveCaptureRefused as exc:
        print(f"NOT READY: {exc}")
        return 2

    # Actually create it and prove a file can be written with owner-only permissions.
    try:
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
    except OSError as exc:
        print(f"NOT READY: cannot create {directory}: {exc.strerror}")
        return 2

    probe = live_capture.LiveCapture("preflight-0000")
    probe.record_sources([("probe", f"{live_capture.SYNTHETIC_MARKER} preflight")])
    probe.record_draft(f"{live_capture.SYNTHETIC_MARKER} preflight draft")
    try:
        written = probe.write(directory)
    except Exception as exc:
        print(f"NOT READY: write failed: {type(exc).__name__}")
        return 2
    if written is None:
        print("NOT READY: write returned no path")
        return 2

    mode = written.stat().st_mode & 0o777
    if mode != 0o600:
        problems.append(f"capture file mode is {mode:o}, expected 600")
    try:
        written.unlink()
    except OSError:
        problems.append("preflight file could not be removed")

    free = _free_bytes(directory)
    if free is not None and free < 1_000_000:
        problems.append(f"less than 1 MB free at {directory} ({free} bytes)")

    if problems:
        for problem in problems:
            print(f"NOT READY: {problem}")
        return 2

    print("READY: capture will write 0600 files in 0700 directory")
    print(f"free_bytes: {free if free is not None else 'unknown'}")
    return 0


def _free_bytes(directory: Path) -> int | None:
    try:
        stats = os.statvfs(directory)
        return stats.f_bavail * stats.f_frsize
    except (AttributeError, OSError):
        return None


if __name__ == "__main__":
    sys.exit(main())
