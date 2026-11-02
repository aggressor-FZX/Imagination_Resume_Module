"""Offline route-to-renderer checks for receipt-bound headline content."""

import base64
import copy
import importlib
import io
import socket

import pytest
from fastapi.testclient import TestClient


MARKDOWN = """## Professional Experience
**Synthetic Employer Alpha**
- Maintained electrical equipment throughout the service region with documented inspections and daily safety briefings, coordinating scheduled repairs with technicians and recording completion details for commercial installations
- Reduced diagnostic time by 30% with structured testing and regular maintenance across commercial customer installations, documenting findings for supervisor review and maintaining clear equipment service records for follow-up inspections

## Education
**Synthetic Technical College**
- Diploma in Electrical Technology with practical instruction in wiring, distribution panels, troubleshooting, safety procedures and standards
"""
OWNER = "synthetic-owner"
UNAPPROVED = "Invented Neurosurgeon Qualification"


@pytest.fixture
def export_setup(monkeypatch, tmp_path):
    # Import the real app without reading dotenv files or making network calls.
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

    monkeypatch.setitem(app.app.dependency_overrides, app.get_api_key, lambda: "synthetic-auth")
    monkeypatch.setattr(renderer, "DEFAULT_OUTPUT_DIR", tmp_path)
    compiled = []

    def fake_compile(tex_path, output_dir):
        compiled.append(tex_path.read_text(encoding="utf-8"))
        pdf = tex_path.with_suffix(".pdf")
        pdf.write_bytes(b"%PDF-1.4\nsynthetic test double\n")
        return pdf

    monkeypatch.setattr(renderer, "compile_latex", fake_compile)
    client = TestClient(app.app)
    yield client, gate, export_payload(MARKDOWN), compiled
    client.close()


def signed_request(gate, approved, *, wrapped=True, **outer):
    receipt = gate.issue_receipt(
        markdown=MARKDOWN, verdict=gate.Verdict.PASS_TIER_A.value,
        owner=OWNER, approved_payload=approved,
    )
    assert receipt
    if wrapped:
        # The current export gate reads evidence from outer fields, while the
        # renderer and receipt parser deliberately consume only `data`.
        body = {"data": copy.deepcopy(approved), "experience": copy.deepcopy(approved["experience"]),
                "education": copy.deepcopy(approved["education"])}
    else:
        body = copy.deepcopy(approved)
    body.update(outer, final_written_section_markdown=MARKDOWN, publication_receipt=receipt)
    return body


def export(client, body):
    return client.post("/resume/export", json=body, headers={"X-Cogito-Owner-Id": OWNER})


def docx_text(response):
    from docx import Document

    doc = Document(io.BytesIO(base64.b64decode(response.json()["docx_base64"])))
    return "\n".join(p.text for p in doc.paragraphs)


@pytest.mark.parametrize("field", ["job_title", "headline"])
@pytest.mark.parametrize("approved_title", ["", "Approved Electrical Technician"])
def test_outer_title_cannot_override_signed_nested_content(export_setup, field, approved_title):
    client, gate, approved, compiled = export_setup
    approved["headline"] = approved_title
    body = signed_request(gate, approved, **{field: UNAPPROVED})
    assert gate.verify_receipt(body["publication_receipt"], markdown=MARKDOWN,
                               owner=OWNER, render_payload=body) == (True, "ok")

    response = export(client, body)
    assert response.status_code == 200, response.text
    text = docx_text(response)
    assert len(compiled) == 1
    assert UNAPPROVED not in text and UNAPPROVED not in compiled[0]
    if approved_title:
        assert approved_title in compiled[0]
    assert "Synthetic Employer Alpha" in text and "Synthetic Employer Alpha" in compiled[0]
    assert response.json()["filename"] != UNAPPROVED.replace(" ", "_")


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("title_key", ["headline", "job_title", "target_job_title"])
def test_valid_title_aliases_render_and_keep_filename(export_setup, wrapped, title_key):
    client, gate, approved, compiled = export_setup
    approved[title_key] = "Approved Electrical Technician"
    approved["name"] = "Synthetic Candidate"
    approved["email"] = "candidate@example.test"
    body = signed_request(gate, approved, wrapped=wrapped)
    response = export(client, body)
    assert response.status_code == 200, response.text
    text = docx_text(response)
    # The current DOCX renderer omits the headline; the PDF template includes it.
    assert "Approved Electrical Technician" in compiled[0]
    assert "Synthetic Candidate" in text and "candidate@example.test" in text
    assert response.json()["filename"] == "Approved_Electrical_Technician"


def test_nested_title_tampering_is_rejected_before_render(export_setup):
    client, gate, approved, compiled = export_setup
    approved["headline"] = "Approved Electrical Technician"
    body = signed_request(gate, approved)
    body["data"]["job_title"] = UNAPPROVED
    response = export(client, body)
    assert response.status_code == 409
    assert response.json()["reasons"] == ["receipt_render_content_mismatch"]
    assert compiled == []


@pytest.mark.parametrize("change,reason", [
    ("missing_receipt", "receipt_missing"),
    ("wrong_owner", "receipt_owner_mismatch"),
    ("markdown", "receipt_content_mismatch"),
    ("flat_title", "receipt_render_content_mismatch"),
])
def test_invalid_approval_never_reaches_renderer(export_setup, change, reason):
    client, gate, approved, compiled = export_setup
    body = signed_request(gate, approved, wrapped=False)
    headers = {"X-Cogito-Owner-Id": OWNER}
    if change == "missing_receipt":
        del body["publication_receipt"]
    elif change == "wrong_owner":
        headers["X-Cogito-Owner-Id"] = "different-owner"
    elif change == "markdown":
        body["final_written_section_markdown"] += "\nInvented change"
    else:
        body["headline"] = UNAPPROVED
    response = client.post("/resume/export", json=body, headers=headers)
    assert response.status_code == 409
    assert response.json()["reasons"] == [reason]
    assert compiled == []


def test_signed_alias_precedence_not_outer_or_flat_override(export_setup):
    client, gate, approved, compiled = export_setup
    approved.update(job_title="", target_job_title="Approved Target", headline="Other Approved Alias")
    body = signed_request(gate, approved, wrapped=False)
    response = export(client, body)
    assert response.status_code == 200, response.text
    text = docx_text(response)
    assert "Approved Target" in compiled[0]
    assert "Other Approved Alias" not in text and "Other Approved Alias" not in compiled[0]
    assert response.json()["filename"] == "Approved_Target"