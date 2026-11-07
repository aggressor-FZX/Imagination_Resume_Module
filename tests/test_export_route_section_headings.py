"""The real export ROUTE returns files carrying every promised heading.

``test_export_section_headings`` proves the renderer emits the headings. This
file proves the same thing through ``POST /resume/export`` — the endpoint the
website actually calls — so a defect in the route's payload handling cannot hide
behind a renderer-only test.

Measured 2026-11-07: a live website run produced a PDF and DOCX that carried four
of the six headings in ``source_preservation.REQUIRED_HEADINGS``. The summary
text was present but unlabelled, and the empty Projects section was dropped
entirely. Both files were read back with real extractors, not asserted on bytes.

The route is exercised in-process with the LaTeX compiler stubbed, because the
claim under test is which headings reach the file, not that pdflatex runs. The
DOCX half is a real ``python-docx`` document.
"""

import base64
import copy
import importlib
import io
import socket

import pytest
from fastapi.testclient import TestClient

REQUIRED = (
    "Professional Summary",
    "Professional Experience",
    "Projects",
    "Education",
    "Certifications",
    "Technical Skills",
)

# The approved markdown the browser previews. It carries all six headings,
# including an empty Projects section — the exact shape that lost two of them.
# It is long enough to clear the gate's 80-word floor, which is a separate
# control and not what this file is testing.
MARKDOWN = """## Professional Summary
Operations analyst with 9 years of experience in depot reporting and fuel cost
control across public transit operations. Operations analyst supporting stock
planning and parts availability for a regional equipment service network.

## Professional Experience
**Harborline Transit Authority** -- Tacoma, WA
*Fuel Reporting Analyst | 04/2019 - Present*
- Tracked monthly fuel consumption across 14 depots and compared each month
  against the same month in the prior year.
- Built the depot scorecard that supervisors review in the Monday operations
  meeting.
- Reconciled 1,180 fuel invoices quarterly against depot dispensing records.

**Cascade Freight Partners** -- Kent, WA
*Cost Reporting Coordinator | 07/2015 - 03/2019*
- Audited 3 years of maintenance cost records to find which equipment failed
  repeatedly.
- Prepared monthly cost summaries for regional management review.
- Rebuilt the shipment delay log that the dispatch team relies on.

**Ridgeline Equipment Services** -- Fife, WA
*Inventory Analyst | 08/2023 - Present*
- Maintained parts availability records supporting 7 service branches.
- Flagged slow-moving stock and proposed reduction plans adopted across 2
  branches.
- Compared quarterly parts usage against the prior four quarters to set reorder
  points.

## Projects

## Education
**Puget Sound Technical College** -- Tacoma, WA
*Associate of Applied Science, Business Analytics | 2015*

**Inland Records Institute** -- Fife, WA
*Certificate, Inventory Control | 2022*

## Certifications
- Certified Production and Inventory Management (CPIM)

## Technical Skills
- Spreadsheet modelling, SQL
- Invoice reconciliation
- Parts inventory, stock level planning
- Records accuracy
"""

OWNER = "synthetic-owner"


@pytest.fixture
def export_setup(monkeypatch, tmp_path):
    import dotenv
    from pydantic_settings.sources import DotEnvSettingsSource

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(DotEnvSettingsSource, "_read_env_files", lambda self: {})
    monkeypatch.setenv("PUBLICATION_RECEIPT_SECRET", "synthetic-signing-key-0123456789")

    def no_network(*args, **kwargs):
        raise AssertionError("Export tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)

    app = importlib.import_module("app")
    gate = importlib.import_module("publication_gate")
    renderer = importlib.import_module("resume_builder.build_resume_from_export")
    from approved_resume import export_payload

    monkeypatch.setitem(app.app.dependency_overrides, app.get_api_key,
                        lambda: "synthetic-auth")
    monkeypatch.setattr(renderer, "DEFAULT_OUTPUT_DIR", tmp_path)

    def fake_compile(tex_path, output_dir):
        pdf = tex_path.with_suffix(".pdf")
        pdf.write_bytes(b"%PDF-1.4\nsynthetic test double\n")
        return pdf

    monkeypatch.setattr(renderer, "compile_latex", fake_compile)
    client = TestClient(app.app)
    yield client, gate, export_payload(MARKDOWN)
    client.close()


def _signed_body(gate, approved):
    receipt = gate.issue_receipt(
        markdown=MARKDOWN, verdict=gate.Verdict.PASS_TIER_A.value,
        owner=OWNER, approved_payload=approved,
    )
    assert receipt
    body = {
        "data": copy.deepcopy(approved),
        "experience": copy.deepcopy(approved["experience"]),
        "education": copy.deepcopy(approved["education"]),
    }
    body.update(final_written_section_markdown=MARKDOWN,
                publication_receipt=receipt)
    return body


def _docx_text(encoded):
    from docx import Document

    document = Document(io.BytesIO(base64.b64decode(encoded)))
    return "\n".join(p.text for p in document.paragraphs)


def test_export_route_returns_a_docx_with_every_required_heading(export_setup):
    client, gate, approved = export_setup
    response = client.post("/resume/export", json=_signed_body(gate, approved),
                           headers={"X-Cogito-Owner-Id": OWNER})
    assert response.status_code == 200, response.text
    text = _docx_text(response.json()["docx_base64"])
    for heading in REQUIRED:
        assert heading in text, f"exported DOCX dropped the {heading!r} heading"


def test_export_route_labels_the_summary(export_setup):
    """The summary must be a labelled section, not a bare paragraph."""
    client, gate, approved = export_setup
    response = client.post("/resume/export", json=_signed_body(gate, approved),
                           headers={"X-Cogito-Owner-Id": OWNER})
    assert response.status_code == 200, response.text
    from docx import Document

    document = Document(io.BytesIO(base64.b64decode(response.json()["docx_base64"])))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    index = paragraphs.index("Professional Summary")
    assert paragraphs[index + 1].startswith("Operations analyst with 9")


def test_export_route_keeps_the_empty_projects_heading(export_setup):
    """An empty section keeps its heading and gains no invented content."""
    client, gate, approved = export_setup
    response = client.post("/resume/export", json=_signed_body(gate, approved),
                           headers={"X-Cogito-Owner-Id": OWNER})
    assert response.status_code == 200, response.text
    from docx import Document

    document = Document(io.BytesIO(base64.b64decode(response.json()["docx_base64"])))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    index = paragraphs.index("Projects")
    assert paragraphs[index + 1] == "Education"


def test_export_route_heading_set_matches_the_approved_markdown(export_setup):
    """The file's sections must equal the markdown's sections.

    This is the property that failed on the live run: the approved markdown had
    six sections and the files had four, and nothing compared them.
    """
    client, gate, approved = export_setup
    response = client.post("/resume/export", json=_signed_body(gate, approved),
                           headers={"X-Cogito-Owner-Id": OWNER})
    assert response.status_code == 200, response.text
    text = _docx_text(response.json()["docx_base64"])
    in_markdown = {h for h in REQUIRED if f"## {h}" in MARKDOWN}
    in_file = {h for h in REQUIRED if h in text}
    assert in_file == in_markdown == set(REQUIRED)
