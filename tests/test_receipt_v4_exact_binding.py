"""Offline receipt tests: bind renderer input, not the generated file bytes."""

import base64
import copy
from dataclasses import asdict, dataclass
import hashlib
import hmac
import importlib
import json
import socket
import time
from unittest.mock import Mock

import pytest

import publication_gate as gate
from resume_builder import build_resume_from_export as renderer


SECRET = "synthetic-only-receipt-key-123456789"
OWNER = "synthetic-owner"
MD = "## Experience\n- Synthetic role\n\n## Education\n- Synthetic school\n"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("PUBLICATION_RECEIPT_SECRET", SECRET)

    def no_network(*args, **kwargs):
        raise AssertionError("Network access forbidden in receipt tests")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)


@pytest.fixture
def approved():
    return {
        "name": "Synthetic Candidate", "location": "Example City",
        "phone": "555 0100", "email": "person@example.test",
        "linkedin": "example.test/profile", "github": "example.test/code",
        "headline": "Synthetic Specialist", "summary": "A synthetic summary.",
        "skills": [{"category": "Test Tools", "items": ["Python", "SQL"]}],
        "experience": [{"role": "Analyst", "company": "Example Labs",
                        "location": "Example City", "duration": "2020 -- 2022",
                        "bullets": ["Built synthetic reports.", "Reviewed test data."]}],
        "projects": [{"title": "Example Project", "duration": "2021 -- 2022",
                      "bullets": ["Built a sample tool."]}],
        "education": ["Synthetic School"], "certifications": ["Test Certificate"],
        "publication": "Example Study",
    }


def issue(approved, markdown=MD):
    receipt = gate.issue_receipt(markdown=markdown, verdict=gate.Verdict.PASS_TIER_A.value,
                                 owner=OWNER, approved_payload=approved)
    assert receipt
    return receipt


def verify(receipt, payload, markdown=MD, owner=OWNER):
    return gate.verify_receipt(receipt, markdown=markdown, owner=owner, render_payload=payload)


def signed_envelope(body):
    serialized = json.dumps(body, sort_keys=True, separators=(",", ":"))
    encoded = base64.urlsafe_b64encode(serialized.encode()).decode()
    signature = hmac.new(SECRET.encode(), serialized.encode(), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def test_v4_matches_complete_renderer_object_and_exact_utf8(approved):
    markdown = MD + "Café\t  \n"
    receipt = issue(approved, markdown)
    token = receipt.split(".")[0]
    envelope = json.loads(base64.urlsafe_b64decode(token))
    actual = renderer.parse_resume_payload({"data": approved})
    assert envelope["v"] == 4
    assert envelope["fp"] == hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    canonical = gate.canonical_render_content(approved)
    assert json.loads(canonical) == json.loads(json.dumps(asdict(actual)))
    assert envelope["render_fp"] == hashlib.sha256(canonical.encode()).hexdigest()
    assert verify(receipt, approved, markdown) == (True, "ok")


@pytest.mark.parametrize("change", [
    lambda s: " " + s, lambda s: s + "\n", lambda s: s.replace("\n\n", "\n"),
    lambda s: s.replace("\n", "\r\n"), lambda s: s.replace("Synthetic role", "Synthetic  role"),
    lambda s: s.replace("Café", "Cafe\u0301"),
])
def test_markdown_whitespace_and_unicode_are_exact(approved, change):
    markdown = MD + "Café\n"
    receipt = issue(approved, markdown)
    assert verify(receipt, approved, change(markdown)) == (False, "receipt_content_mismatch")


@pytest.mark.parametrize("field", [
    "name", "location", "phone", "email", "linkedin", "github", "headline", "summary",
    "skills", "experience", "projects", "education", "certifications", "publication",
])
def test_every_renderer_field_is_bound_including_whitespace(approved, field):
    receipt = issue(approved)
    changed = copy.deepcopy(approved)
    if field == "skills":
        changed[field][0]["items"][0] = "Py  thon"
    elif field in ("experience", "projects"):
        changed[field][0]["bullets"][0] = "Built  synthetic reports."
    elif field in ("education", "certifications"):
        changed[field][0] = changed[field][0].replace(" ", "  ", 1)
    else:
        changed[field] += "  changed"
    assert verify(receipt, changed) == (False, "receipt_render_content_mismatch")


@pytest.mark.parametrize("field", ["name", "location", "phone", "email", "linkedin", "github", "headline", "summary"])
def test_internal_whitespace_change_in_text_fields_is_rejected(approved, field):
    # Even a line break replacing a space must not be collapsed by the receipt.
    receipt = issue(approved)
    changed = copy.deepcopy(approved)
    changed[field] = approved[field].replace(" ", "\n", 1) if " " in approved[field] else approved[field] + "\nX"
    assert verify(receipt, changed) == (False, "receipt_render_content_mismatch")


def test_parser_normalized_whitespace_and_alias_precedence(approved):
    receipt = issue(approved)
    unchanged = copy.deepcopy(approved)
    unchanged["name"] = "  " + unchanged["name"] + "  "  # parser strips it
    unchanged["full_name"] = "Unused alias"
    assert verify(receipt, unchanged) == (True, "ok")
    unchanged["job_title"] = "Higher priority title"
    assert verify(receipt, unchanged) == (False, "receipt_render_content_mismatch")


def test_nested_data_is_the_same_object_renderer_parses(approved):
    receipt = issue(approved)
    wrapped = {"data": copy.deepcopy(approved), "job_title": "Ignored outer title",
               "name": "Ignored outer name", "experience": [{"role": "Ignored"}]}
    assert verify(receipt, wrapped) == (True, "ok")
    wrapped["data"]["email_address"] = "ignored@example.test"  # lower priority alias
    assert verify(receipt, wrapped) == (True, "ok")
    wrapped["data"]["job_title"] = "Unexpected title"
    assert verify(receipt, wrapped) == (False, "receipt_render_content_mismatch")
    assert verify(receipt, {"data": None}) == (False, "receipt_render_content_mismatch")


@pytest.mark.parametrize("change", [
    "experience-title", "experience-date", "experience-order", "project-date",
    "skill-label", "skill-order", "education-order", "certification-order",
])
def test_nested_render_fields_and_order_are_bound(approved, change):
    approved["education"].append("Example Institute")
    approved["certifications"].append("Example Certificate")
    receipt = issue(approved)
    changed = copy.deepcopy(approved)
    if change == "experience-title":
        changed["experience"][0]["company"] = "Other Example Labs"
    elif change == "experience-date":
        changed["experience"][0]["duration"] = "2020 -- 2023"
    elif change == "experience-order":
        changed["experience"][0]["bullets"].reverse()
    elif change == "project-date":
        changed["projects"][0]["duration"] = "2020 -- 2022"
    elif change == "skill-label":
        changed["skills"][0]["category"] = "Other Tools"
    elif change == "skill-order":
        changed["skills"][0]["items"].reverse()
    elif change == "education-order":
        changed["education"].reverse()
    else:
        changed["certifications"].reverse()
    assert verify(receipt, changed) == (False, "receipt_render_content_mismatch")


def test_future_renderer_dataclass_fields_are_not_silently_omitted(monkeypatch, approved):
    @dataclass
    class ExtendedResume(renderer.ResumePayload):
        extra_render_field: str = "additional synthetic content"

    real_parser = renderer.parse_resume_payload

    def extended(payload):
        return ExtendedResume(**asdict(real_parser(payload)))

    monkeypatch.setattr(renderer, "parse_resume_payload", extended)
    canonical = json.loads(gate.canonical_render_content(approved))
    assert canonical["extra_render_field"] == "additional synthetic content"


def test_v3_signed_receipt_requires_reanalysis(approved):
    legacy = signed_envelope({"v": 3, "fp": gate.content_fingerprint(MD),
                              "verdict": gate.Verdict.PASS_TIER_A.value, "owner": OWNER,
                              "iat": int(time.time()), "ttl": 1800, "sb": 1,
                              "render_fp": "legacy-not-exact"})
    assert verify(legacy, approved) == (False, "receipt_version_3_reanalysis_required")
    assert verify(legacy[:-1] + ("0" if legacy[-1] != "0" else "1"), approved) == (False, "receipt_signature_invalid")


@pytest.mark.parametrize("version", [True, "4", 4.1, 2])
def test_unknown_or_non_integer_versions_are_rejected(approved, version):
    receipt = signed_envelope({"v": version})
    assert verify(receipt, approved) == (False, "receipt_version_unknown")


def test_v5_is_recognised_as_a_version():
    """v5 exists now, so it must not be rejected as an unknown version.

    This assertion used to include ``5`` in the unknown-version list because v5
    did not exist. The meaningful guarantee today is that a v5 envelope is
    recognised and then fails on its own merits (here: no verdict), rather than
    being reported as an unrecognised wire version.
    """
    receipt = signed_envelope({"v": 5})
    # The version gate itself must pass; the envelope then fails on its own
    # merits (no verdict, no policy fingerprint), not as an unknown version.
    ok, reason = gate.verify_receipt(receipt, markdown=MD, owner=OWNER)
    assert reason != "receipt_version_unknown"
    assert ok is False


def test_v4_does_not_allow_missing_renderer_or_invalid_markdown(approved):
    receipt = issue(approved)
    assert gate.verify_receipt(receipt, markdown=MD, owner=OWNER) == (False, "receipt_render_payload_missing")
    assert verify(receipt, approved, None) == (False, "receipt_content_mismatch")
    assert verify(receipt, approved, "\ud800") == (False, "receipt_content_mismatch")
    assert gate.issue_receipt(markdown=None, verdict=gate.Verdict.PASS_TIER_A.value,
                              owner=OWNER, approved_payload=approved) is None
    assert gate.issue_receipt(markdown="\ud800", verdict=gate.Verdict.PASS_TIER_A.value,
                              owner=OWNER, approved_payload=approved) is None


def test_real_export_passes_verified_payload_to_renderer(monkeypatch, approved):
    import dotenv
    from fastapi.testclient import TestClient
    from pydantic_settings.sources import DotEnvSettingsSource

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(DotEnvSettingsSource, "_read_env_files", lambda self: {})
    app = importlib.import_module("app")
    monkeypatch.setitem(app.app.dependency_overrides, app.get_api_key, lambda: "synthetic")
    # Gate structure validation is tested separately; this test isolates receipt
    # verification and the exact payload passed to the actual renderer entry point.
    monkeypatch.setattr(app.publication_gate, "decide", lambda **kwargs: gate.PublishDecision(True, "completed"))
    render = Mock(return_value={"filename": "synthetic", "pdf_base64": "", "docx_base64": ""})
    monkeypatch.setattr(renderer, "build_resume_from_payload_base64", render)
    receipt = issue(approved)
    body = {"data": copy.deepcopy(approved), "experience": copy.deepcopy(approved["experience"]),
            "education": approved["education"], "job_title": "Ignored outer title",
            "final_written_section_markdown": MD, "publication_receipt": receipt}
    with TestClient(app.app) as client:
        response = client.post("/resume/export", json=body, headers={"X-Cogito-Owner-Id": OWNER})
        assert response.status_code == 200, response.text
        render.assert_called_once_with(body)
        render.reset_mock()
        body["data"]["phone"] = "555  0100"
        response = client.post("/resume/export", json=body, headers={"X-Cogito-Owner-Id": OWNER})
        assert response.status_code == 409
        assert response.json()["reasons"] == ["receipt_render_content_mismatch"]
        render.assert_not_called()


def test_analysis_issuer_binds_empty_profile_without_late_contact_injection(monkeypatch):
    import dotenv
    from pydantic_settings.sources import DotEnvSettingsSource

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(DotEnvSettingsSource, "_read_env_files", lambda self: {})
    app = importlib.import_module("app")
    markdown = "## Experience\n**Example Labs**\n- Built synthetic reports.\n\n## Education\n**Synthetic School**\n"
    output = app._issue_publication_receipt({
        "final_written_section_markdown": markdown,
        "processing_status": "completed",
        "metrics_support": {"verdict": gate.Verdict.PASS_TIER_A.value, "checked": True},
    }, owner=OWNER)
    receipt = output["publication_receipt"]
    approved = output["approved_export_payload"]
    assert receipt and approved
    assert verify(receipt, approved, markdown) == (True, "ok")
    changed = {**approved, "name": "Synthetic Candidate"}
    assert verify(receipt, changed, markdown) == (False, "receipt_render_content_mismatch")