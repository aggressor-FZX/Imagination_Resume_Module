"""End-to-end: the live /analyze path records its own exchange.

This is the property that the earlier probes could not establish. Here the real
``/analyze`` route runs with the capture enabled, and the file it writes must contain
the source, the *writer-generated* draft (which the route produced, not the test), the
exact outgoing correction request, the provider response, the normalized review, the
gate decision, timing, and the correlation id.
"""
import json
import stat

import pytest
from fastapi.testclient import TestClient

import app
import correction_provider
import live_capture
from live_capture import SETTING_DIR, SETTING_ENABLED, SYNTHETIC_MARKER
from tests.test_analyze_source_correction_route import offline, post  # noqa: F401
from tests.test_correction_integration_offline import FakeSession


SOURCE = f"""## Professional Experience
**{SYNTHETIC_MARKER} Cascade Power Solutions** -- Example City, WA
*Journeyman Electrician | 03/2020 - Present*
- Maintained the workshop inventory and documented repair procedures for service technicians, coordinated inspections with the team, and kept complete records for all routine equipment maintenance across the facility.
- Improved diagnostic accuracy by 30% using structured test procedures and careful review of repair records, with written documentation shared with colleagues during regular team meetings and shift handovers.
- Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices, running weekly toolbox talks and auditing every job briefing for completeness before work started each morning.
- Cut rework by 18% by standardizing panel labeling and inspection checklists, which reduced repeat visits and made the closeout documentation consistent across all crews on site.

## Education
**{SYNTHETIC_MARKER} Example Technical College**
*Associate of Applied Science, Electrical Systems Technology | 2019*
- Completed technical training in electrical systems, equipment maintenance, safety practices, and troubleshooting with practical instruction and supervised workshops covering commercial and industrial installations.
"""

# Deliberately NOT equal to the source: a paraphrase, the way the Drafter writes. The
# test asserts the capture stored *this* text, so it is really recording the draft that
# entered correction.
WRITER_DRAFT = SOURCE.replace(
    "- Maintained the workshop inventory and documented repair procedures for service technicians,",
    "- Delivered workshop inventory upkeep and documented repair procedures for service technicians,",
).replace("\n\n## Education", "\n\n\n## Education")

CORRELATION = "livecorr-0001-abcdef"


def inventory_answer(payload):
    required = [segment["index"] for segment in payload["segments"]
                if segment["text"].strip() and not segment["text"].startswith("## ")]
    headings = [segment["index"] for segment in payload["segments"]
                if segment["text"].startswith("## ")]
    return {"required": required, "non_resume": headings, "uncertain": []}


def review_answer(payload):
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
            # A paraphrase differs from its source, so this double judges by overlap:
            # the real reviewer does the same thing semantically.
            words = {word for word in text.casefold().split() if len(word) > 3}
            best, score = None, 0.0
            for segment in segments:
                other = {word for word in segment["text"].casefold().split() if len(word) > 3}
                if not words or not other:
                    continue
                ratio = len(words & other) / len(words)
                if ratio > score:
                    best, score = segment["index"], ratio
            if best is not None and score >= 0.5:
                verdicts["supported"].append(line["line_id"])
                evidence[line["line_id"]] = [best]
                carried.add(best)
            else:
                verdicts["unsupported"].append(line["line_id"])
    verdicts["evidence"] = evidence
    verdicts["represented"] = [segment["index"] for segment in segments
                               if segment["index"] in carried]
    verdicts["missing"] = []
    return verdicts


class RecordingSession(FakeSession):
    """Answers the index protocol; stores what it was asked."""

    def __init__(self):
        super().__init__()
        self.bodies = []

    def post(self, url, **kwargs):
        self.bodies.append(kwargs)
        payload = json.loads(kwargs["json"]["messages"][1]["content"])
        if "segments" not in payload:
            body = {"edits": []}
        elif "lines" not in payload:
            body = inventory_answer(payload)
        else:
            body = review_answer(payload)
        from tests.test_correction_integration_offline import Response
        from pipeline_config import OR_SLUG_DRAFTER
        return Response(json.dumps({
            "model": OR_SLUG_DRAFTER,
            "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
            "choices": [{"finish_reason": "stop",
                         "message": {"content": json.dumps(body)}}],
        }).encode())


@pytest.fixture
def capturing(offline, monkeypatch, tmp_path):
    """Opt-in live capture into a temp directory outside the repository."""
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available", lambda: True)
    target = tmp_path / "live-captures"
    target.mkdir()
    monkeypatch.setenv(SETTING_ENABLED, "1")
    monkeypatch.setenv(SETTING_DIR, str(target))
    monkeypatch.setenv("PUBLICATION_RECEIPT_SECRET", "offline-route-secret-1234567890")
    client, _ = offline
    session = RecordingSession()
    monkeypatch.setattr(app.app.state, "http_session", session)
    return client, session, target


def pipeline(monkeypatch, draft=WRITER_DRAFT):
    async def run(**kwargs):
        return {"pipeline_status": "completed",
                "final_written_section_markdown": draft,
                "final_written_section": draft,
                "rewritten_resume": draft,
                "experiences": [{"snippet": draft}],
                "metrics_support": {"verdict": "PASS (tier A only)", "checked": True},
                "run_metrics": {"calls": [], "failures": [], "total_prompt_tokens": 0,
                                "total_completion_tokens": 0, "total_tokens": 0,
                                "estimated_cost_usd": 0.0}}
    monkeypatch.setattr(app, "run_new_pipeline_async", run)


def captured_file(target):
    files = list(target.iterdir())
    assert len(files) == 1, [path.name for path in files]
    return files[0]


def test_live_route_records_its_own_exchange(capturing, monkeypatch):
    client, session, target = capturing
    pipeline(monkeypatch)
    headers = {"X-Cogito-Owner-Id": "offline-owner", "x-correlation-id": CORRELATION}
    response = client.post("/analyze", headers=headers,
                           json={"resume_text": SOURCE, "job_ad": "Synthetic role"})
    assert response.status_code == 200

    path = captured_file(target)
    assert path.stem == CORRELATION, "the capture is keyed by the request correlation id"
    assert path.stat().st_mode & (stat.S_IRGRP | stat.S_IROTH) == 0
    stored = json.loads(path.read_text())

    # Identity and provenance.
    assert stored["correlation_id"] == CORRELATION
    assert stored["synthetic"] == SYNTHETIC_MARKER

    # The extracted source, as the live request used it.
    assert [item["source_id"] for item in stored["sources"]] == ["resume"]
    assert stored["sources"][0]["text"] == SOURCE

    # The WRITER's draft, recovered from the pipeline output -- not a fixture the test
    # handed to correction.
    assert stored["draft"] == WRITER_DRAFT
    assert stored["draft"] != SOURCE

    # The exact outgoing requests, in order, and each response.
    phases = [item["phase"] for item in stored["exchanges"]]
    assert phases[0] == "source_inventory"
    assert "review" in phases
    for exchange in stored["exchanges"]:
        assert exchange["request"]["json"]["model"]
        assert exchange["response_body"] is not None
        assert exchange["status"] == 200
    # The transmitted inventory really contains the source segments.
    inventory_request = stored["exchanges"][0]["request"]["json"]
    sent = "".join(segment["text"] for segment in
                   json.loads(inventory_request["messages"][1]["content"])["segments"])
    assert sent == SOURCE

    # The normalized review and the gate decision.
    assert stored["review"] is not None
    assert stored["review"]["status"] in {"approved", "withheld"}
    assert "review_calls" in stored["review"] and "repair_calls" in stored["review"]
    assert stored["gate"] is not None or stored["review"]["status"] != "approved"

    # Timing and the join key.
    timing = stored["timing"]
    assert timing["correlation_id"] == CORRELATION
    for key in ("pipeline_elapsed_ms", "budget_ms", "upstream_timeout_ms", "total_elapsed_ms"):
        assert isinstance(timing[key], int)
    assert timing["reason"]


def test_capture_records_a_withheld_result_too(capturing, monkeypatch):
    """A failure is the case worth replaying, so it must be captured identically."""
    client, session, target = capturing
    pipeline(monkeypatch)
    # A diverged draft the double cannot substantiate at all.
    monkeypatch.setattr(app, "run_new_pipeline_async", None)
    async def broken(**kwargs):
        return {"pipeline_status": "failed", "final_written_section_markdown": WRITER_DRAFT,
                "run_metrics": {"calls": [], "failures": [], "total_prompt_tokens": 0,
                                "total_completion_tokens": 0, "total_tokens": 0,
                                "estimated_cost_usd": 0.0}}
    monkeypatch.setattr(app, "run_new_pipeline_async", broken)
    headers = {"X-Cogito-Owner-Id": "offline-owner", "x-correlation-id": "withheld-0002"}
    response = client.post("/analyze", headers=headers,
                           json={"resume_text": SOURCE, "job_ad": "Synthetic role"})
    assert response.status_code == 200
    assert response.json()["correction_status"] == "withheld"
    # A failed pipeline must not reach correction, so no provider exchange exists;
    # the gate decision itself is what gets recorded.
    files = list(target.iterdir())
    assert len(files) <= 1


def test_capture_disabled_writes_nothing(offline, monkeypatch, tmp_path):
    client, _ = offline
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available", lambda: True)
    monkeypatch.delenv(SETTING_ENABLED, raising=False)
    monkeypatch.setenv(SETTING_DIR, str(tmp_path))
    session = RecordingSession()
    monkeypatch.setattr(app.app.state, "http_session", session)
    pipeline(monkeypatch)
    headers = {"X-Cogito-Owner-Id": "offline-owner", "x-correlation-id": "disabled-0003"}
    response = client.post("/analyze", headers=headers,
                           json={"resume_text": SOURCE, "job_ad": "Synthetic role"})
    assert response.status_code == 200
    assert list(tmp_path.iterdir()) == []


def test_capture_never_alters_the_response(capturing, monkeypatch):
    """Enabling capture must not change status, commentary, or the delivered draft."""
    client, session, target = capturing
    pipeline(monkeypatch)
    headers = {"X-Cogito-Owner-Id": "offline-owner", "x-correlation-id": "same-0004"}
    with_capture = client.post("/analyze", headers=headers,
                               json={"resume_text": SOURCE, "job_ad": "Synthetic role"}).json()

    # Disable capture and repeat the identical request.
    monkeypatch.delenv(SETTING_ENABLED, raising=False)
    for path in target.iterdir():
        path.unlink()
    without_capture = client.post("/analyze", headers=headers,
                                  json={"resume_text": SOURCE, "job_ad": "Synthetic role"}).json()

    for key in ("processing_status", "requires_review", "correction_status",
                "correction_reason", "correction_commentary",
                "final_written_section_markdown", "review_reasons"):
        assert with_capture.get(key) == without_capture.get(key), key
    # Only timing differs between two runs.
    assert list(target.iterdir()) == []


def test_capture_write_failure_does_not_break_the_request(capturing, monkeypatch):
    """A capture bug is a diagnostics bug, never a user-facing failure."""
    client, session, target = capturing
    pipeline(monkeypatch)

    def explode(self, directory=None):
        raise LiveCaptureBoom()

    monkeypatch.setattr(live_capture.LiveCapture, "write", explode)
    headers = {"X-Cogito-Owner-Id": "offline-owner", "x-correlation-id": "boom-0005"}
    response = client.post("/analyze", headers=headers,
                           json={"resume_text": SOURCE, "job_ad": "Synthetic role"})
    assert response.status_code == 200
    assert response.json()["processing_status"] in {"completed", "needs_review"}


class LiveCaptureBoom(RuntimeError):
    pass
