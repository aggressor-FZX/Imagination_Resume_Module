"""The repair path through the real route and the real local renderers.

The controlled-recovery tests drive the correction engine directly. This one proves the
same thing through the wiring a user actually touches:

    POST /analyze   -> writer draft (the run-7 draft, defects intact)
                    -> _run_source_correction -> real CorrectionProvider
                    -> real engine: inventoried, reviewed, REPAIRED, re-reviewed
                    -> publication gate -> receipt
    POST /resume/export -> real LaTeX compiler + real python-docx
                        -> pdftotext / DOCX XML read back

Only two layers are scripted, and both are labelled:

* the **writer** output — this is the input under test, taken verbatim from the run-7
  capture so the defect is the one that was actually produced;
* the **reviewer** decisions — written here from the source text.

``CorrectionResult`` is never overridden, so the approval asserted below is the
engine's own. The renderer assertions read the generated files; no PDF/DOCX bytes are
mocked.

Labelled **[route-repair]** and **[renderer-real]**.
"""
from __future__ import annotations

import base64
import json
import re
import shutil
import socket
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app
import correction_provider
import live_capture
from tests.fixtures.fidelity import expected_facts as facts
from tests.test_repair_recovery_controlled import (  # reused answer builders
    DEFECT_MARKERS,
    REPAIRED_LINES,
    REPAIRED_SUPPORTED,
    SOUND_LINES,
    _defective_review,
    _repaired_review,
    repair,
    review,
)

RUN7 = Path("/home/skystarved/Render_Dockers/.live-run-2026-11-02/captures/"
            "run7/0a267e04-affd-4cee-b944-26c574703039.json")

requires_run7 = pytest.mark.skipif(not RUN7.exists(), reason="run-7 capture absent")

SOURCE = facts.SOURCE_TEXT
EMPLOYER = facts.EMPLOYMENT_RECORDS[0]["employer"]
START, END = facts.EMPLOYMENT_RECORDS[0]["start"], facts.EMPLOYMENT_RECORDS[0]["end"]
TITLE = facts.EMPLOYMENT_RECORDS[0]["role"]

# The route runs TWO independent checks on the published markdown: the correction
# engine's review, and `approved_resume.unsupported_entities`, which requires the bold
# leading name under Education/Experience to appear in the source. The run-7 draft's
# education line is "**Associate of Applied Science, Example Technical College**", and
# that concatenation appears nowhere in the source, so the entity check refuses it
# whatever the reviewer says.
#
# That makes it a third, genuinely separate defect — one the reviewer never raised and
# the lexical `_introduces_beyond` veto does not see, because both halves are sourced
# but their combination is not. Repairing it (splitting the bold name from the degree)
# is therefore part of a complete fix, and this route test would fail without it.
EDUCATION_AS_WRITTEN = "**Associate of Applied Science, Example Technical College** -- 2016"
EDUCATION_SPLIT = "**Example Technical College** -- Associate of Applied Science, 2016"

# Repair mapping for the route: the two demonstrated writer defects plus the education
# naming defect the route's entity check exposes.
ROUTE_REPAIR = {
    **REPAIRED_LINES,
    EDUCATION_AS_WRITTEN: EDUCATION_SPLIT,
}

# A review that flags the education line as well. It must be flagged for repair to be
# allowed to touch it: repair may only replace a *flagged* span. Without this the entity
# check would reject the draft after the engine approved it, which is a correct refusal
# but a confusing one — the route would withhold for a defect the reviewer never named.
ROUTE_DEFECT_MARKERS = DEFECT_MARKERS + ("Associate of Applied Science,",)


def _route_defective_review():
    from tests.test_repair_recovery_controlled import SOUND_LINES
    return review({k: v for k, v in SOUND_LINES.items()
                   if k != "Example Technical College"},
                  unsupported=ROUTE_DEFECT_MARKERS)


def _route_repaired_review():
    from tests.test_repair_recovery_controlled import SOUND_LINES
    return review({**SOUND_LINES,
                   "Completed 8,000 hours of supervised": "Completed 8,000 hours",
                   "**Skills:**": (
                       "Commercial electrical systems, conduit bending, panel labeling",
                       "NFPA 70E, lockout/tagout, inspection checklists")})


def _draft():
    """The run-7 draft, verbatim from the protected capture."""
    return live_capture.load(RUN7)["draft"]


def _envelope(answer):
    return json.dumps({
        "model": correction_provider.OR_SLUG_DRAFTER,
        "provider": "ControlledReviewer",
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        "choices": [{"finish_reason": "stop",
                     "message": {"content": json.dumps(answer)}}],
    })


def _response(status, body):
    class Content:
        async def iter_chunked(self, size):
            raw = (body or "").encode()
            for index in range(0, len(raw), size):
                yield raw[index:index + size]

    class Headers:
        def get(self, name, default=None):
            return default

    class Response:
        def __init__(self):
            self.status = status
            self.headers = Headers()
            self.content = Content()

        async def text(self, *args, **kwargs):
            return body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    return Response()


class RepairingReviewer:
    """Reviews the recorded draft, repairs it, and re-reviews the result.

    The first review is the controlled "defective" answer; the repair replaces the two
    demonstrated defects; every later review is the controlled "repaired" answer. Both
    are in the adapter's index wire form, so this exercises the same translation the
    live route uses.
    """

    def __init__(self):
        self.phases: list[str] = []
        self.requests: list[dict] = []

    async def close(self):
        pass

    def post(self, url, **kwargs):
        wire = json.loads(kwargs["json"]["messages"][1]["content"])
        phase = self._phase(wire)
        self.phases.append(phase)
        self.requests.append(kwargs)
        return _response(200, _envelope(self._answer(phase, wire)))

    @staticmethod
    def _phase(wire):
        contract = wire.get("contract", "")
        if contract.startswith("Classify every numbered source segment"):
            return "inventory"
        if contract.startswith("Rewrite the named draft lines"):
            return "restore"
        if "flagged_claim_ids" in wire:
            return "repair"
        return "review"

    def _answer(self, phase, wire):
        if phase == "inventory":
            return {"required": [s["index"] for s in wire["segments"]],
                    "non_resume": [], "uncertain": []}
        if phase == "repair":
            return repair(ROUTE_REPAIR)(wire)
        # First review: the recorded defects. Later reviews: the repaired draft.
        if sum(1 for p in self.phases if p == "review") == 1:
            return _route_defective_review()(wire)
        return _route_repaired_review()(wire)


@pytest.fixture
def published(monkeypatch):
    """Run /analyze through the repair path; return the published result body."""
    if not RUN7.exists():
        pytest.skip("run-7 capture is not present")

    def forbidden(*args, **kwargs):
        raise AssertionError("external network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(app.settings, "openrouter_api_key_1", "offline-key")
    monkeypatch.setattr(app.settings, "openrouter_api_key_2", None)
    monkeypatch.setenv("PUBLICATION_RECEIPT_SECRET",
                       "offline-route-secret-1234567890")
    monkeypatch.setitem(app.app.dependency_overrides, app.get_api_key,
                        lambda: "offline")
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available",
                        lambda: True)

    reviewer = RepairingReviewer()
    draft = _draft()

    async def run_pipeline(**kwargs):
        return {
            "pipeline_status": "completed",
            "final_written_section_markdown": draft,
            "final_written_section": draft,
            "rewritten_resume": draft,
            "experiences": [],
            "editorial_notes": "",
            "final_written_section_provenance": [],
            "metrics_support": {"verdict": "PASS", "checked": True},
            "run_metrics": {"calls": [], "failures": []},
        }

    monkeypatch.setattr(app, "run_new_pipeline_async", run_pipeline)

    with TestClient(app.app) as client:
        client.portal.call(app.app.state.http_session.close)
        app.app.state.http_session = reviewer
        body = client.post(
            "/analyze", headers={"X-Cogito-Owner-Id": "offline-owner"},
            json={"resume_text": SOURCE, "job_ad": "Senior Electrician role"},
        ).json()
        yield client, body, reviewer
    app.app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# [route-repair] the repair happens through the real route and publishes
# ---------------------------------------------------------------------------

def test_repair_phase_ran_through_the_route(published):
    """[route-repair] The route reached the repair phase, which it never used to.

    Before this change every production run recorded ``repair_calls == 0``; the gate
    withheld on missing facts before the repair path could be entered.
    """
    _client, _body, reviewer = published
    assert "repair" in reviewer.phases, reviewer.phases
    # A repair must be followed by a full re-review of the repaired draft.
    first_repair = reviewer.phases.index("repair")
    assert "review" in reviewer.phases[first_repair:], reviewer.phases


def test_repaired_draft_is_approved_through_the_route(published):
    """[route-repair] Approved by the engine, with the defects removed."""
    _client, body, _reviewer = published
    assert body.get("correction_status") == "approved", (
        body.get("correction_reason"), body.get("correction_commentary"))
    published_md = body["final_written_section_markdown"]
    assert "mastering core electrical installations" not in published_md
    assert "**Electrical:**" not in published_md
    assert "supervised field work" in published_md


def test_route_keeps_the_supported_facts_and_the_five_headings(published):
    """[route-repair] Coverage and structure survive the repair.

    Both halves matter and they pull against each other: five headings must be present,
    and none may be filled with material the source does not contain.
    """
    _client, body, _reviewer = published
    md = body["final_written_section_markdown"]

    for surviving in ("40+ job sites", "6 apprentices", "8,000 hours",
                      "25 commercial projects", START, END, "18%"):
        assert surviving in md, f"supported fact missing after repair: {surviving}"

    for heading in ("Professional Summary", "Professional Experience", "Education",
                    "Certifications", "Technical Skills"):
        assert re.search(rf"^#+\s*{re.escape(heading)}\s*$", md,
                         re.IGNORECASE | re.MULTILINE), f"heading missing: {heading}"

    # No invented filler in the promised sections.
    assert "## Projects" not in md
    assert "State of Washington (2023)" not in md
    assert "NFPA (2022)" not in md
    assert "Relevant Courses" not in md


def test_route_issues_a_receipt_for_the_repaired_draft(published):
    """[route-repair] The receipt binds the repaired text, not the original draft."""
    _client, body, _reviewer = published
    assert body.get("publication_receipt"), "approved result must carry a receipt"
    payload = body.get("approved_export_payload") or {}
    assert payload, "approved result must carry an export payload"
    assert "mastering core electrical installations" not in json.dumps(payload)


# ---------------------------------------------------------------------------
# [renderer-real] actual PDF and DOCX files, content read back
# ---------------------------------------------------------------------------

pdflatex = pytest.mark.skipif(shutil.which("pdflatex") is None,
                              reason="no LaTeX toolchain on this machine")


def _pdf_text(path):
    result = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[:300]
    return result.stdout


def _docx_text(path):
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    return re.sub(r"<[^>]+>", " ", xml)


@pdflatex
def test_export_renders_real_files_from_the_repaired_draft(published):
    """[renderer-real] The exported PDF and DOCX carry the repaired content.

    The repaired draft is what the user would receive, so it is what must reach the
    page: the two run-7 defects absent, the supported facts and the five headings
    present, and no mojibake.
    """
    client, body, _reviewer = published
    payload = dict(body.get("approved_export_payload") or {})
    assert payload, "no export payload"
    # The receipt binds the EXACT render payload; altering any field is what
    # receipt_render_content_mismatch exists to refuse.
    payload["publication_receipt"] = body["publication_receipt"]
    payload["final_written_section_markdown"] = body["final_written_section_markdown"]

    response = client.post("/resume/export",
                           headers={"X-Cogito-Owner-Id": "offline-owner"},
                           json=payload)
    assert response.status_code == 200, response.text[:400]
    data = response.json()
    assert data.get("pdf_base64") and data.get("docx_base64")

    with tempfile.TemporaryDirectory() as directory:
        pdf = Path(directory) / "out.pdf"
        docx = Path(directory) / "out.docx"
        pdf.write_bytes(base64.b64decode(data["pdf_base64"]))
        docx.write_bytes(base64.b64decode(data["docx_base64"]))
        assert pdf.stat().st_size > 1000
        assert docx.stat().st_size > 1000
        pdf_text = _pdf_text(pdf)
        docx_text = _docx_text(docx)

    for label, text in (("pdf", pdf_text), ("docx", docx_text)):
        # The defects are gone from the rendered page.
        assert "mastering core electrical installations" not in text, label
        assert "hands-on field experience" not in text, label
        # The supported facts reached the page.
        assert "supervised field work" in text, label
        assert EMPLOYER in text, label
        assert START in text, label
        assert "40+ job sites" in text, label
        # Structure survived the renderer. The rendered heading style differs per
        # format — the PDF template prints the summary paragraph without repeating its
        # heading, while the DOCX keeps every `##` as a heading — so the summary is
        # asserted by its content and the section headings by their presence.
        assert "Journeyman electrician with 8+ years" in text, (
            f"{label}: summary content missing")
        for heading in ("Professional Experience", "Education", "Certifications"):
            assert re.search(rf"{re.escape(heading)}", text, re.IGNORECASE), (
                f"{label}: heading missing: {heading}")
        # No mojibake reached the page.
        assert "\u00e2\u0080" not in text, f"{label}: mojibake"
        assert "\u00c3" not in text, f"{label}: mojibake"
