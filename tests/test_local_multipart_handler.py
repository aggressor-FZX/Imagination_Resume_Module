"""The Imaginator's OWN multipart handler, measured with scripted models.

**This is NOT the website's multi-file path.** Read `docs/ROUTE_MAP.md` before relying
on anything here.

The website's two-file traffic goes frontend -> orchestrator
``POST /api/analysis/multi-file`` -> Imaginator ``POST /analyze`` (see
``backend/tests/integration/multiFileRoute.test.js``, which follows the request). The
handler exercised by this file is ``app.py``'s own
``POST /api/analysis/multi-file``: an Imaginator-local multipart endpoint that no
orchestrator route targets. Earlier versions of this file called its assertions
"multi-file acceptance", which was wrong — naming the endpoint after the traffic that
does *not* reach it is exactly the confusion the route map exists to prevent.

What this handler is worth testing for: it shares the publication gate, it is covered by
``tests/test_composite_publication.py``, and it is a plausible future entry point. So it
is measured honestly, and labelled as the local handler it is.

The handler is a DIFFERENT pipeline from ``/analyze``:

    POST /api/analysis/multi-file          (Imaginator-local, multipart)
      -> run_analysis_async -> run_generation_async -> run_criticism_async
      -> run_synthesis_async -> publication gate -> receipt
    POST /resume/export

It runs **no source correction and no preservation**, and it has **no live-capture
wiring**. On the website's route both do run, because that route ends at ``/analyze``.

Every model call is scripted. ``CorrectionResult`` does not exist on this path, and no
bytes are mocked.
"""
from __future__ import annotations

import hashlib
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
import live_capture


ROOT = Path("/home/skystarved/Render_Dockers/.live-run-2026-11-02")

# Two synthetic documents with deliberately DISTINCT employment and education records,
# which is what makes employer/degree/date mixing detectable at all.
DOC_ELECTRICIAN = """## Professional Summary
SYNTHETIC-ONLY Journeyman electrician with 8+ years of commercial electrical experience.

## Professional Experience
**SYNTHETIC-ONLY Cascade Power Solutions, LLC** -- Example City, WA
*Journeyman Electrician | 03/2020 - Present*
- Installed commercial electrical systems across 40+ job sites.
- Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices.

## Education
**SYNTHETIC-ONLY Example Technical College** -- Example City, WA
*Associate of Applied Science, Electrical Systems Technology | 2016*

## Skills
- Commercial electrical systems, panel labeling
"""

DOC_FACILITIES = """## Professional Summary
SYNTHETIC-ONLY Facilities maintenance technician focused on preventive maintenance.

## Professional Experience
**SYNTHETIC-ONLY Bluegate Property Services** -- Sampletown, OR
*Facilities Maintenance Technician | 01/2014 - 05/2016*
- Maintained electrical distribution and HVAC controls for 12 commercial buildings.
- Reduced unplanned downtime by 22% through a preventive maintenance schedule.

## Education
**SYNTHETIC-ONLY Example Community College** -- Sampletown, OR
*Certificate, Building Systems Maintenance | 2013*

## Skills
- Preventive maintenance scheduling, HVAC controls
"""

# The two records must not be confusable. Asserted, so a future edit to either fixture
# cannot silently make the mixing checks vacuous.
DISTINCT = {
    "employers": ("Cascade Power Solutions, LLC", "Bluegate Property Services"),
    "roles": ("Journeyman Electrician", "Facilities Maintenance Technician"),
    "institutions": ("Example Technical College", "Example Community College"),
    "degrees": ("Associate of Applied Science", "Certificate"),
    "years": ("2016", "2013"),
}


def _distinct_fixtures_really_are_distinct():
    for a, b in DISTINCT.values():
        assert a != b
    blob = DOC_ELECTRICIAN
    for value in (DISTINCT["employers"][1], DISTINCT["institutions"][1]):
        assert value not in blob, f"{value!r} leaks into the electrician document"


def _synthesis_markdown() -> str:
    """What the synthesis stage returns: both records, correctly kept apart.

    Written by hand, not by a model, so this measures the ROUTE — gate, receipt, export
    — and not a model's quality. It deliberately keeps both records separate so the
    mixing checks have something correct to compare against.
    """
    return (
        "## Professional Summary\n"
        "Journeyman electrician and facilities maintenance technician with commercial "
        "electrical and preventive maintenance experience.\n\n"
        "## Professional Experience\n"
        "**Cascade Power Solutions, LLC** -- Example City, WA\n"
        "*Journeyman Electrician | 03/2020 - Present*\n"
        "- Installed commercial electrical systems across 40+ job sites.\n"
        "- Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices.\n\n"
        "**Bluegate Property Services** -- Sampletown, OR\n"
        "*Facilities Maintenance Technician | 01/2014 - 05/2016*\n"
        "- Maintained electrical distribution and HVAC controls for 12 commercial "
        "buildings.\n"
        "- Reduced unplanned downtime by 22% through a preventive maintenance "
        "schedule.\n\n"
        "## Education\n"
        "**Example Technical College** -- Example City, WA\n"
        "*Associate of Applied Science, Electrical Systems Technology | 2016*\n\n"
        "**Example Community College** -- Sampletown, OR\n"
        "*Certificate, Building Systems Maintenance | 2013*\n\n"
        "## Technical Skills\n"
        "- Commercial electrical systems, panel labeling\n"
        "- Preventive maintenance scheduling, HVAC controls\n"
    )


@pytest.fixture
def local_multipart(monkeypatch):
    """Drive the Imaginator's own ``POST /api/analysis/multi-file`` with two files.

    The orchestrator never calls this endpoint; the website's route is ``/analyze``. See
    the module docstring and `docs/ROUTE_MAP.md`.
    """
    _distinct_fixtures_really_are_distinct()

    def forbidden(*args, **kwargs):
        raise AssertionError("external network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(app.settings, "IMAGINATOR_AUTH_TOKEN", "local-multipart-test-key")
    monkeypatch.setenv("IMAGINATOR_AUTH_TOKEN", "local-multipart-test-key")
    monkeypatch.setenv("PUBLICATION_RECEIPT_SECRET",
                       "local-multipart-signing-key-0123456789")

    captured = {}
    markdown = _synthesis_markdown()

    async def analysis(**kwargs):
        captured["submitted_text"] = kwargs.get("resume_text")
        return {
            "pipeline_status": "completed",
            "experiences": [],
            "metrics_support": {"verdict": "PASS (tier A only)", "checked": True},
            "source_projects": [],
        }

    async def generation(**kwargs):
        return {}

    async def criticism(**kwargs):
        return {}

    async def synthesis(**kwargs):
        return {"final_written_section_markdown": markdown,
                "final_written_section": markdown,
                "final_written_section_provenance": []}

    monkeypatch.setattr(app, "run_analysis_async", analysis)
    monkeypatch.setattr(app, "run_generation_async", generation)
    monkeypatch.setattr(app, "run_criticism_async", criticism)
    monkeypatch.setattr(app, "run_synthesis_async", synthesis)
    monkeypatch.setattr(app, "_multifile_llm_client", lambda _keys: None)

    files = [
        ("files", ("electrician.txt", DOC_ELECTRICIAN.encode(), "text/plain")),
        ("files", ("facilities.txt", DOC_FACILITIES.encode(), "text/plain")),
    ]
    with TestClient(app.app) as client:
        response = client.post(
            "/api/analysis/multi-file", files=files,
            data={"job_ad": "Senior Electrician"},
            headers={"X-API-Key": "local-multipart-test-key",
                     "X-Cogito-Owner-Id": "local-multipart-owner"})
        yield client, response, captured, markdown
    app.app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# What THIS handler does today
# ---------------------------------------------------------------------------

def test_both_documents_reach_the_pipeline(local_multipart):
    """Both uploads are concatenated and submitted, so neither is silently dropped."""
    _client, _response, captured, _md = local_multipart
    submitted = captured["submitted_text"]

    for value in ("Cascade Power Solutions", "Bluegate Property Services",
                  "Example Technical College", "Example Community College"):
        assert value in submitted, f"document content dropped before analysis: {value}"


def test_local_multipart_handler_has_no_correction_or_preservation_or_capture():
    """This handler has no correction, preservation or capture. Structural, not transient.

    Asserted from the source so this cannot be mistaken for a flaky observation: the
    ``/analyze`` path calls ``live_capture.start`` and runs ``_run_source_correction``;
    this multipart handler does neither. That asymmetry is a property of THIS endpoint.

    It is **not** evidence about the website's two-file traffic, which reaches
    ``/analyze`` (see ``docs/ROUTE_MAP.md`` and
    ``Login_Landing_page/backend/tests/integration/multiFileRoute.test.js``).
    """
    source = Path(app.__file__).read_text()

    handler_start = source.index("async def analyze_multi_file(")
    handler_end = source.index("except HTTPException as e:", handler_start)
    handler = source[handler_start:handler_end]

    assert "_run_source_correction" not in handler, (
        "this handler now runs source correction; update this analysis")
    assert "live_capture.start" not in handler, (
        "this handler now records a live capture; update this analysis")
    assert "preserve_source_facts" not in handler, (
        "this handler now runs preservation; update this analysis")

    # And the /analyze path does all three, so the asymmetry is real.
    single = source[source.index("async def _analyze_resume_impl("):]
    assert "_run_source_correction" in single
    assert "live_capture.start" in single


def test_the_final_document_is_the_synthesis_output_verbatim(local_multipart):
    """No preservation stage means whatever synthesis returns is what gets judged.

    This is why preservation-based guarantees cannot be claimed for this endpoint: the
    gate and the receipt bind the synthesis text as-is.
    """
    _client, response, _captured, markdown = local_multipart
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["final_written_section_markdown"] == markdown


def test_gate_and_receipt_do_run_on_this_route(local_multipart):
    """The parts that DO exist: Tier-A verification against the combined uploads,
    the publication gate, and a signed receipt.

    Stated because it bounds the work: the missing pieces are capture and
    preservation, not verification or signing.
    """
    _client, response, _captured, _md = local_multipart
    body = response.json()

    assert body["metrics_support"]["verdict"] == "PASS (tier A only)"
    assert body.get("publication_receipt"), "an approved result must carry a receipt"
    assert body.get("approved_export_payload")
    assert body["requires_review"] is False


# ---------------------------------------------------------------------------
# The four requirements, measured against this handler as it stands
#
# These compare the scripted synthesis output with the two uploads. They measure THIS
# handler's gate and receipt, and the scripted content, not a model's quality and not
# the website's route.
# ---------------------------------------------------------------------------

def test_both_documents_supported_facts_are_retained(local_multipart):
    _client, response, _captured, _md = local_multipart
    md = response.json()["final_written_section_markdown"]

    for value in ("40+ job sites", "6 apprentices", "12 commercial buildings", "22%",
                  "lockout/tagout", "HVAC controls", "preventive maintenance"):
        assert value in md, f"supported fact from one of the two documents lost: {value}"


def test_no_employer_role_or_date_mixing(local_multipart):
    """Each employer keeps its own role AND its own dates.

    Positional, not "the strings all appear somewhere": mixing is precisely the case
    where every string is present but attached to the wrong record.
    """
    _client, response, _captured, _md = local_multipart
    md = response.json()["final_written_section_markdown"]

    cascade = md.index("**Cascade Power Solutions, LLC**")
    bluegate = md.index("**Bluegate Property Services**")
    education = md.index("## Education")

    assert cascade < bluegate < education, "employer order changed"

    cascade_block = md[cascade:bluegate]
    assert "Journeyman Electrician" in cascade_block
    assert "03/2020" in cascade_block
    assert "Facilities Maintenance Technician" not in cascade_block
    assert "01/2014" not in cascade_block

    bluegate_block = md[bluegate:education]
    assert "Facilities Maintenance Technician" in bluegate_block
    assert "01/2014" in bluegate_block and "05/2016" in bluegate_block
    assert "Journeyman Electrician" not in bluegate_block
    assert "03/2020" not in bluegate_block


def test_no_degree_or_institution_mixing(local_multipart):
    """Each institution keeps its own degree and year."""
    _client, response, _captured, _md = local_multipart
    md = response.json()["final_written_section_markdown"]

    technical = md.index("**Example Technical College**")
    community = md.index("**Example Community College**")
    skills = md.index("## Technical Skills")

    technical_block = md[technical:community]
    assert "Associate of Applied Science" in technical_block
    assert "Electrical Systems Technology" in technical_block
    assert "2016" in technical_block
    assert "Certificate" not in technical_block
    assert "Building Systems Maintenance" not in technical_block

    community_block = md[community:skills]
    assert "Certificate" in community_block
    assert "Building Systems Maintenance" in community_block
    assert "2013" in community_block
    assert "Associate of Applied Science" not in community_block


def test_no_duplicate_or_invented_qualifications(local_multipart):
    """Each qualification appears once, and nothing appears that the sources lack."""
    _client, response, _captured, _md = local_multipart
    md = response.json()["final_written_section_markdown"]

    for once_only in ("Associate of Applied Science", "Certificate,",
                      "Cascade Power Solutions", "Bluegate Property Services"):
        assert md.count(once_only) == 1, f"{once_only!r} appears {md.count(once_only)}×"

    for invented in ("AutoCAD", "Revit", "Python", "SCADA", "PLC", "Agile", "Gantt",
                     "GPA", "Relevant Courses", "OSHA"):
        assert invented not in md, f"invented qualification: {invented}"


# ---------------------------------------------------------------------------
# Export: receipt accepted, and the real files checked against BOTH documents
# ---------------------------------------------------------------------------

pdflatex = pytest.mark.skipif(shutil.which("pdflatex") is None,
                              reason="no LaTeX toolchain on this machine")


@pdflatex
def test_receipt_accepted_by_export_and_files_checked_against_both_documents(local_multipart):
    """The receipt travels, export accepts it, and both records reach the page.

    Real LaTeX and real python-docx, read back from the produced files.
    """
    client, response, _captured, _md = local_multipart
    body = response.json()

    payload = dict(body.get("approved_export_payload") or {})
    assert payload, "no export payload"
    payload["publication_receipt"] = body["publication_receipt"]
    payload["final_written_section_markdown"] = body["final_written_section_markdown"]

    export = client.post("/resume/export",
                         headers={"X-Cogito-Owner-Id": "local-multipart-owner",
                                  "X-API-Key": "local-multipart-test-key"},
                         json=payload)
    assert export.status_code == 200, export.text[:400]
    data = export.json()
    assert data.get("pdf_base64") and data.get("docx_base64")

    import base64
    with tempfile.TemporaryDirectory() as directory:
        pdf_file = Path(directory) / "out.pdf"
        docx_file = Path(directory) / "out.docx"
        pdf_file.write_bytes(base64.b64decode(data["pdf_base64"]))
        docx_file.write_bytes(base64.b64decode(data["docx_base64"]))

        pdf_text = subprocess.run(["pdftotext", "-layout", str(pdf_file), "-"],
                                  capture_output=True, text=True).stdout
        with zipfile.ZipFile(docx_file) as archive:
            xml = archive.read("word/document.xml").decode()
        docx_text = re.sub(r"<[^>]+>", " ", xml)

    for label, text in (("pdf", pdf_text), ("docx", docx_text)):
        flat = " ".join(text.split())
        # Both documents' records reached the page, each with its own identifiers.
        for value in ("Cascade Power Solutions", "Bluegate Property Services",
                      "Example Technical College", "Example Community College",
                      "Journeyman Electrician", "Facilities Maintenance Technician",
                      "03/2020", "01/2014", "Electrical Systems Technology",
                      "Building Systems Maintenance"):
            assert value in flat, f"{label}: missing {value!r}"

        # No mixing survived the renderer.
        cascade_at = flat.index("Cascade Power Solutions")
        bluegate_at = flat.index("Bluegate Property Services")
        assert cascade_at < bluegate_at
        between = flat[cascade_at:bluegate_at]
        assert "Facilities Maintenance Technician" not in between

        # No mojibake.
        assert "\u00e2\u0080" not in text, f"{label}: mojibake"
        assert "\u00c3" not in text, f"{label}: mojibake"


# ---------------------------------------------------------------------------
# Evidence that this measurement used the frozen revision
# ---------------------------------------------------------------------------

def test_no_preservation_runs_so_the_frozen_revision_cannot_help_this_handler():
    """Records the architectural gap in one assertion.

    ``preserve_source_facts`` is called from ``StarEditor.polish``. This handler never
    constructs a ``StarEditor``: it goes analysis -> generation -> criticism -> synthesis.
    So the skills/qualifier/education rules in the frozen revision do not apply here.

    They DO apply to the website's two-file route, which ends at ``/analyze`` and
    therefore runs ``StarEditor.polish`` — which is why the multi-document preservation
    fix matters for that route and not this one.
    """
    source = Path(app.__file__).read_text()
    handler = source[source.index("async def analyze_multi_file("):
                     source.index("except HTTPException as e:",
                                  source.index("async def analyze_multi_file("))]
    for stage in ("run_analysis_async", "run_generation_async",
                  "run_criticism_async", "run_synthesis_async"):
        assert stage in handler, stage
    assert "StarEditor" not in handler
