"""Executed export regressions: approval must bind the renderer, not raw aliases."""
import copy
import os
import pytest
from unittest.mock import Mock

os.environ.setdefault("IMAGINATOR_AUTH_TOKEN", "render-test-key")
os.environ.setdefault("OPENROUTER_API_KEY_1", "render-dummy")
os.environ.setdefault("PUBLICATION_RECEIPT_SECRET", "render-test-signing-key-0123456789")

from fastapi.testclient import TestClient
import app
import publication_gate as gate
from resume_builder.build_resume_from_export import parse_resume_payload


@pytest.fixture(autouse=True)
def synthetic_auth(monkeypatch):
    monkeypatch.setattr(app.settings, "IMAGINATOR_AUTH_TOKEN", "render-test-key")
    monkeypatch.setenv("IMAGINATOR_AUTH_TOKEN", "render-test-key")
    monkeypatch.setenv("PUBLICATION_RECEIPT_SECRET", "render-test-signing-key-0123456789")

MD = """## Professional Experience
**Synthetic Employer Alpha**
- Maintained electrical equipment throughout the service region with documented inspections and daily safety briefings, coordinating scheduled repairs with technicians and recording completion details for commercial installations
- Reduced diagnostic time by 30% with structured testing and regular maintenance across commercial customer installations, documenting findings for supervisor review and maintaining clear equipment service records for follow-up inspections

## Education
**Synthetic Technical College**
- Diploma in Electrical Technology with practical instruction in wiring, distribution panels, troubleshooting, safety procedures and standards
"""


def approved():
    return {"final_written_section_markdown": MD, "summary": "",
            "experience": [{"company": "Synthetic Employer Alpha", "bullets": [
                "Maintained electrical equipment throughout the service region with documented inspections and daily safety briefings, coordinating scheduled repairs with technicians and recording completion details for commercial installations",
                "Reduced diagnostic time by 30% with structured testing and regular maintenance across commercial customer installations, documenting findings for supervisor review and maintaining clear equipment service records for follow-up inspections",
            ]}], "skills": [{"category": "Tools", "items": ["Multimeter"]}],
            "education": [{"degree": "Diploma", "institution": "Synthetic Technical College"}],
            "certifications": [{"name": "Safety Certificate", "issuer": "Synthetic Board"}]}


def test_empty_renderer_has_no_substantive_sample_defaults():
    parsed = parse_resume_payload({"experience": [], "skills": []})
    assert not parsed.summary
    assert not parsed.headline
    assert not parsed.phone and not parsed.email and not parsed.name
    assert not parse_resume_payload({}).experience


@pytest.mark.parametrize("field,value", [
    ("professional_summary", "Chief scientist with 40 years of invented experience"),
    ("job_title", "Licensed Neurosurgeon"),
    ("skills", [{"category": "Tools", "items": ["Invented qualification"]}]),
    ("education", [{"degree": "Diploma", "institution": "Invented University"}]),
    ("certifications", [{"name": "Safety Certificate", "issuer": "Invented Board"}]),
])
def test_actual_export_rejects_alias_render_injection(monkeypatch, field, value):
    original = approved()
    # Use the real issuer. Once fixed it accepts an approved_payload explicitly;
    # current code signs only its flat claim set, reproducing the bypass.
    kwargs = {"markdown": MD, "verdict": gate.Verdict.PASS_TIER_A.value,
              "owner": "render-owner", "approved_claims": gate.approved_render_claims(original)}
    import inspect
    if "approved_payload" in inspect.signature(gate.issue_receipt).parameters:
        kwargs["approved_payload"] = original
    receipt = gate.issue_receipt(**kwargs)
    rendered = Mock(return_value={"filename": "test", "pdf_base64": "JVBERg==", "docx_base64": "UEs="})
    monkeypatch.setattr("resume_builder.build_resume_from_export.build_resume_from_payload_base64", rendered)
    client = TestClient(app.app)
    control = client.post("/resume/export", json={**original, "publication_receipt": receipt}, headers={
        "X-API-Key": os.environ["IMAGINATOR_AUTH_TOKEN"], "X-Cogito-Owner-Id": "render-owner"})
    assert control.status_code == 200, control.text
    rendered.reset_mock()
    tampered = copy.deepcopy(original)
    tampered[field] = value
    tampered["publication_receipt"] = receipt
    response = client.post("/resume/export", json=tampered, headers={
        "X-API-Key": os.environ["IMAGINATOR_AUTH_TOKEN"], "X-Cogito-Owner-Id": "render-owner"})
    assert response.status_code == 409, (field, response.status_code)
    rendered.assert_not_called()


@pytest.mark.parametrize("change", [
    "bullet-owner", "project-kind", "dates", "role-location", "short-headline",
    "delete-achievement", "nested-summary", "publication", "skill-label",
])
def test_signed_render_structure_rejects_relationship_and_alternate_field_changes(change):
    original = approved()
    original["experience"].append({"company": "Synthetic Employer Beta", "start_date": "2018",
                                   "end_date": "2020", "bullets": ["Maintained backup instruments"]})
    original["projects"] = [{"title": "Synthetic Project", "bullets": ["Built test instruments"]}]
    receipt = gate.issue_receipt(markdown=MD, verdict=gate.Verdict.PASS_TIER_A.value,
                                 owner="render-owner", approved_payload=original)
    assert gate.verify_receipt(receipt, markdown=MD, owner="render-owner", render_payload=original)[0]
    changed = copy.deepcopy(original)
    if change == "bullet-owner":
        changed["experience"][1]["bullets"].append(changed["experience"][0]["bullets"].pop())
    elif change == "project-kind":
        changed["experience"].extend(changed.pop("projects"))
    elif change == "dates":
        changed["experience"][0]["start_date"] = "2018"
    elif change == "role-location":
        changed["experience"][0]["location"] = "Invented City"
    elif change == "short-headline":
        changed["headline"] = "MD"
    elif change == "delete-achievement":
        changed["experience"][0]["bullets"].pop()
    elif change == "nested-summary":
        changed = {"data": changed}
        changed["data"]["profile_summary"] = "Invented research career"
    elif change == "publication":
        changed["publication_line"] = "Invented Journal"
    elif change == "skill-label":
        changed["skills"][0]["category"] = "Medical qualification"
    ok, reason = gate.verify_receipt(receipt, markdown=MD, owner="render-owner", render_payload=changed)
    assert not ok
    assert reason == "receipt_render_content_mismatch"