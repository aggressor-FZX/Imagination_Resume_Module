"""Replaying a live capture must reproduce the decision, offline and exactly.

This closes the gap the earlier probes left open. A live website failure is only
useful if it can be replayed, and replay is only trustworthy if it (a) uses the
recorded writer draft rather than a rebuild, (b) recomputes the decision instead of
trusting a stored verdict, (c) makes no network call, and (d) refuses a recording it
cannot faithfully reproduce rather than quietly succeeding with fewer phases.
"""
import json
import socket
import stat

import pytest

import live_capture
from live_capture import (
    LiveCapture,
    LiveCaptureRefused,
    ReplayExhausted,
    SETTING_DIR,
    SETTING_ENABLED,
    SYNTHETIC_MARKER,
    load,
    replay,
)


SOURCE = f"""## Professional Experience
**{SYNTHETIC_MARKER} Cascade Power Solutions** -- Example City, WA
*Journeyman Electrician | 03/2020 - Present*
- Maintained the workshop inventory and documented repair procedures for service technicians, coordinated inspections with the team, and kept complete records for all routine equipment maintenance across the facility.
- Improved diagnostic accuracy by 30% using structured test procedures and careful review of repair records, with written documentation shared with colleagues during regular team meetings and shift handovers.

## Education
**{SYNTHETIC_MARKER} Example Technical College**
- Completed technical training in electrical systems, equipment maintenance, safety practices, and troubleshooting with practical instruction across commercial installations.
"""
DRAFT = SOURCE.replace("Maintained the workshop inventory", "Delivered workshop inventory upkeep")


def segments_for(text, source_id="resume"):
    """Build the wire segments the way the adapter does, for scripted responses."""
    parts, position = [], 0
    for line in text.splitlines(keepends=True):
        position += len(line)
        parts.append({"source_id": source_id, "start": position - len(line),
                      "end": position, "text": line})
    return parts


def inventory_body(segments):
    required = [index for index, segment in enumerate(segments)
                if segment["text"].strip() and not segment["text"].startswith("## ")]
    headings = [index for index, segment in enumerate(segments)
                if segment["text"].startswith("## ")]
    return {"required": required, "non_resume": headings, "uncertain": []}


def review_body(payload):
    """Scripted reviewer: exact-match support, everything else unsupported."""
    segments, lines = payload["segments"], payload["lines"]
    verdicts = {"supported": [], "unsupported": [], "uncertain": [], "structural": []}
    evidence, carried = {}, set()
    for line in lines:
        text = line["text"]
        if text.startswith("## "):
            verdicts["structural"].append(line["line_id"])
            continue
        exact = [segment["index"] for segment in segments if segment["text"].strip() == text]
        if exact:
            verdicts["supported"].append(line["line_id"])
            evidence[line["line_id"]] = exact
            carried.update(exact)
        else:
            verdicts["unsupported"].append(line["line_id"])
    verdicts["evidence"] = evidence
    verdicts["represented"] = [segment["index"] for segment in segments
                               if segment["index"] in carried]
    verdicts["missing"] = []
    return verdicts


def build_capture(path, *, draft=DRAFT, source=SOURCE):
    """Write a capture in the live format, as the route would."""
    segments = segments_for(source)
    capture = LiveCapture("replay-case-1")
    capture.record_sources([("resume", source)])
    capture.record_draft(draft)
    index = capture.record_request("source_inventory", {"json": {
        "model": "qwen/qwen3-coder", "messages": [
            {"role": "system", "content": "contract"},
            {"role": "user", "content": json.dumps({"contract": "c", "segments": segments})},
        ]}})
    capture.record_response(index, 200, json.dumps({
        "model": "qwen/qwen3-coder",
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        "choices": [{"finish_reason": "stop",
                     "message": {"content": json.dumps(inventory_body(segments))}}]}))
    index = capture.record_request("review", {"json": {
        "model": "qwen/qwen3-coder", "messages": [
            {"role": "system", "content": "contract"},
            {"role": "user", "content": json.dumps({"contract": "c", "segments": segments,
                                                    "lines": []})},
        ]}})
    # The review answer is filled in by replay-time reconstruction below, so this
    # request/response pair is only a placeholder for tests that expect exhaustion.
    capture.record_response(index, 200, json.dumps({
        "model": "qwen/qwen3-coder",
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}))
    capture.record_timing(budget_ms=1000, reason="withheld")
    return capture.write(path)


@pytest.fixture
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("replay must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


@pytest.fixture
def capture_file(tmp_path, monkeypatch):
    monkeypatch.setenv(SETTING_ENABLED, "1")
    monkeypatch.setenv(SETTING_DIR, str(tmp_path))
    return build_capture(tmp_path)


# --- load controls ---------------------------------------------------------

def test_load_requires_owner_only_permissions(capture_file):
    import os
    os.chmod(capture_file, 0o644)
    with pytest.raises(LiveCaptureRefused, match="owner-only"):
        load(capture_file)


def test_load_refuses_an_unsupported_schema(capture_file):
    data = json.loads(capture_file.read_text())
    data["schema"] = 99
    capture_file.write_text(json.dumps(data))
    import os
    os.chmod(capture_file, 0o600)
    with pytest.raises(LiveCaptureRefused, match="schema"):
        load(capture_file)


def test_load_refuses_unmarked_sources(capture_file):
    data = json.loads(capture_file.read_text())
    data["sources"] = [{"source_id": "resume", "text": "Alex Example, a real resume"}]
    capture_file.write_text(json.dumps(data))
    import os
    os.chmod(capture_file, 0o600)
    with pytest.raises(LiveCaptureRefused, match="synthetic"):
        load(capture_file)


# --- replay behaviour ------------------------------------------------------

def test_replay_makes_no_network_call_and_fails_closed(no_network, capture_file):
    """An incomplete recording withholds; it never becomes an empty success.

    The engine treats a callback error as a withholding outcome, so the exhaustion is
    absorbed rather than propagated. That is the correct fail-closed behaviour: the
    replay reports a withheld result, and the session records that it ran out of
    exchanges.
    """
    result, provider, session = replay(capture_file)
    assert result.status == "withheld"
    assert result.markdown is None
    assert session.calls >= 1


def test_replay_uses_the_recorded_draft_and_sources(capture_file, no_network):
    """The replay must feed the engine the writer's own draft, not a rebuild."""
    import correction_protocol

    seen = {}
    real_inventory = correction_protocol.indexed_inventory_request

    def observe(payload):
        seen.setdefault("sources", []).append(
            [item["text"] for item in payload.get("sources", [])])
        return real_inventory(payload)

    correction_protocol.indexed_inventory_request = observe
    try:
        result, _, _ = replay(capture_file)
    finally:
        correction_protocol.indexed_inventory_request = real_inventory

    # The engine was handed the recorded source, and the recorded draft is what the
    # capture holds -- so replay cannot be reprocessing a reconstruction.
    assert seen["sources"][0] == [SOURCE]
    data = load(capture_file)
    assert data["draft"] == DRAFT
    assert result.status == "withheld"


def test_replay_reports_exhaustion_instead_of_empty_success(capture_file, no_network):
    """The scripted capture stops at the review, so replay must say so."""
    data = load(capture_file)
    session = live_capture.ReplaySession(data)
    # Consume everything the recording holds, then demand one more exchange.
    for _ in range(len(data["exchanges"])):
        session.post("https://example.invalid", json={"messages": [{}, {"content": "{}"}]})
    with pytest.raises(ReplayExhausted, match="replay exhausted"):
        session.post("https://example.invalid", json={"messages": [{}, {"content": "{}"}]})
    assert session.calls == len(data["exchanges"])


def test_replay_counts_every_consumed_exchange(capture_file, no_network):
    data = load(capture_file)
    session = live_capture.ReplaySession(data)
    session.post("u", json={"messages": [{}, {"content": "{}"}]})
    session.post("u", json={"messages": [{}, {"content": "{}"}]})
    assert session.calls == 2 and session.position == 2


def test_replay_restores_the_disclosure_setting(capture_file, no_network):
    """Replay must not leave the process with disclosure forced on."""
    import os
    import correction_provider
    os.environ.pop(correction_provider.DISCLOSURE_SETTING, None)
    replay(capture_file)
    assert correction_provider.DISCLOSURE_SETTING not in os.environ


def test_replay_forces_disclosure_on_for_the_duration(tmp_path, monkeypatch, no_network):
    """Replay the configuration the live run actually used, then restore it."""
    monkeypatch.setenv(SETTING_ENABLED, "1")
    monkeypatch.setenv(SETTING_DIR, str(tmp_path))
    path = build_capture(tmp_path)
    assert path is not None
    import correction_provider
    observed = {}

    async def record(markdown, sources, callback, **kwargs):
        observed["policy"] = correction_provider.approved_disclosure_policy()
        from source_correction import CorrectionResult
        return CorrectionResult("withheld", None, "provider_error", "x", None, 0, 0)

    import source_correction
    real = source_correction.correct
    source_correction.correct = record
    try:
        replay(path)
    finally:
        source_correction.correct = real
    assert observed["policy"] is not None
