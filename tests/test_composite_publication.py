"""Actual two-file multipart requests, ONE composite decision, immutable evidence."""
import os
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
import app
from tests.test_render_approval_contract import MD


@pytest.fixture(autouse=True)
def synthetic_auth(monkeypatch):
    monkeypatch.setattr(app.settings, "IMAGINATOR_AUTH_TOKEN", "composite-test-key")
    monkeypatch.setenv("IMAGINATOR_AUTH_TOKEN", "composite-test-key")
    monkeypatch.setenv("PUBLICATION_RECEIPT_SECRET", "composite-signing-key-0123456789")


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("unsupported", [False, True])
def test_composite_checks_final_synthesis_not_earlier_pass(monkeypatch, reverse, unsupported):
    # Deliberately stale PASS plus poisoned generated source echoes. Neither is
    # authoritative for the final synthesis decision.
    analysis = AsyncMock(return_value={"metrics_support": {
        "verdict": "PASS (tier A only)", "checked": True,
        "unsupported_metrics": []}, "source_projects": ["Invented Project"]})
    monkeypatch.setattr(app, "run_analysis_async", analysis)
    monkeypatch.setattr(app, "run_generation_async", AsyncMock(return_value={}))
    monkeypatch.setattr(app, "run_criticism_async", AsyncMock(return_value={}))
    final = MD.replace("30%", "87%") if unsupported else MD
    monkeypatch.setattr(app, "run_synthesis_async", AsyncMock(return_value={
        "final_written_section_markdown": final, "final_written_section": final}))
    monkeypatch.setattr(app, "_multifile_llm_client", lambda _: None)
    files = [
        ("files", ("career.txt", b"Synthetic Employer Alpha\nReduced diagnostic time by 30%", "text/plain")),
        ("files", ("education.txt", b"Education\nSynthetic Technical College\nDiploma in Electrical Technology", "text/plain")),
    ]
    if reverse:
        files.reverse()
    response = TestClient(app.app).post("/api/analysis/multi-file", files=files,
        data={"job_ad": "Electrical maintenance"}, headers={
            "X-API-Key": os.environ["IMAGINATOR_AUTH_TOKEN"],
            "X-Cogito-Owner-Id": "composite-owner"})
    assert response.status_code == 200, response.text
    body = response.json()
    submitted = analysis.await_args.kwargs["resume_text"]
    assert "Synthetic Employer Alpha" in submitted and "Synthetic Technical College" in submitted
    if unsupported:
        assert body["requires_review"] is True
        assert body["publication_receipt"] is None
        assert body["final_written_section_markdown"] is None
    else:
        assert body["requires_review"] is False, body.get("review_reasons")
        assert body["publication_receipt"]
        assert body["approved_export_payload"]["experience"]


def test_generated_source_echo_cannot_approve_an_invented_employer(monkeypatch):
    invented = MD.replace("Synthetic Employer Alpha", "Invented Employer Omega")
    monkeypatch.setattr(app, "run_analysis_async", AsyncMock(return_value={
        "source_experiences": [{"company": "Invented Employer Omega"}]}))
    monkeypatch.setattr(app, "run_generation_async", AsyncMock(return_value={}))
    monkeypatch.setattr(app, "run_criticism_async", AsyncMock(return_value={}))
    monkeypatch.setattr(app, "run_synthesis_async", AsyncMock(return_value={
        "final_written_section_markdown": invented, "final_written_section": invented}))
    response = TestClient(app.app).post("/api/analysis/multi-file", files=[
        ("files", ("career.txt", b"Synthetic Employer Alpha\nReduced diagnostic time by 30%", "text/plain")),
        ("files", ("education.txt", b"Synthetic Technical College", "text/plain")),
    ], data={"job_ad": "Electrical maintenance"}, headers={
        "X-API-Key": os.environ["IMAGINATOR_AUTH_TOKEN"], "X-Cogito-Owner-Id": "composite-owner"})
    assert response.status_code == 200
    assert response.json()["requires_review"] is True
    assert response.json()["publication_receipt"] is None