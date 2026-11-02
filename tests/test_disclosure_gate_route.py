"""Route-level checks for the disclosure gate: flag on vs off, redaction honored.

Uses the real /analyze route, real correction engine and real CorrectionProvider
with an in-memory scripted transport. Synthetic data only; no network.
"""
import json

import pytest

import app
import correction_provider
from tests.test_analyze_source_correction_route import offline, pipeline, post, result_from_sse  # noqa: F401
from tests.test_analyze_attribution_conflict import (  # noqa: F401
    ScriptedSession, BAD_DRAFT, GOOD_DRAFT, REPAIR_FRAGMENT,
)


PII_SOURCE = (
    "## Professional Experience\n"
    "**Synthetic Workshop**\n"
    "- Alpha Works: Improved diagnostic accuracy by 30% over 6 months using "
    "structured test procedures, careful review of repair records, and written "
    "guidance shared with technicians during regular team meetings; contact "
    "alex.example@example.test or 555-010-0199 for scheduling questions.\n"
    "- Betaa Works: Maintained workshop inventory, coordinated equipment "
    "inspections, documented repair procedures, and kept complete maintenance "
    "records for service technicians and colleagues across the workshop.\n"
    "\n## Education\n"
    "- North College: Bachelor of Mathematics, 2014, with coursework in applied "
    "statistics, quantitative reasoning, technical communication, and practical "
    "problem solving across several completed projects.\n"
    "- South College: Master of Physics, 2016, with coursework in experimental "
    "methods, technical research, laboratory safety, and scientific reporting "
    "for engineering teams.\n"
)
PII_DRAFT = PII_SOURCE.replace("\n\n## Education", "\n\n\n## Education")


@pytest.mark.parametrize("sse", [False, True], ids=["json", "sse"])
def test_flag_off_withholds_before_any_provider_request(offline, monkeypatch, sse):
    client, _ = offline
    monkeypatch.delenv(correction_provider.DISCLOSURE_SETTING, raising=False)
    session = ScriptedSession()
    monkeypatch.setattr(app.app.state, "http_session", session)
    pipeline(monkeypatch, draft=PII_DRAFT)
    response = post(client, source=PII_SOURCE, sse=sse)
    assert response.status_code == 200
    result = result_from_sse(response)[1] if sse else response.json()
    assert result["correction_reason"] == "privacy_review_required"
    assert result["requires_review"] is True
    assert result["publication_receipt"] is None
    assert session.requests == []


@pytest.mark.parametrize("sse", [False, True], ids=["json", "sse"])
def test_flag_on_sends_only_redacted_sources_with_preserved_offsets(offline, monkeypatch, sse):
    client, _ = offline
    monkeypatch.setenv(correction_provider.DISCLOSURE_SETTING, "1")
    session = ScriptedSession()
    monkeypatch.setattr(app.app.state, "http_session", session)
    pipeline(monkeypatch, draft=PII_DRAFT)
    response = post(client, source=PII_SOURCE, sse=sse)
    assert response.status_code == 200
    result = result_from_sse(response)[1] if sse else response.json()
    assert session.requests, "correction should reach the provider with the flag on"

    first = json.loads(session.requests[0]["json"]["messages"][1]["content"])
    # The wire carries the application's segments; concatenated they are the
    # source, so offsets are preserved by construction. No `sources` array is sent.
    assert "sources" not in first
    sent_text = "".join(segment["text"] for segment in first["segments"])
    # Identifiers are removed; length (and therefore every citation offset) is not.
    for secret in ("alex.example@example.test", "555-010-0199"):
        assert secret not in sent_text
    assert len(sent_text) == len(PII_SOURCE)
    assert "Alpha Works" in sent_text and "30% over 6 months" in sent_text
    # Offsets are the app's own: contiguous, gapless, and inside the source.
    position = 0
    for segment in first["segments"]:
        assert segment["start"] == position and segment["end"] > segment["start"]
        position = segment["end"]
    assert position == len(PII_SOURCE)

    # The disclosure policy governs what leaves the platform. A secret appearing
    # in the returned resume is the echo of the caller's own input (they uploaded
    # it in resume_text), so the meaningful assertions are: nothing sensitive is
    # sent to the provider, and no diagnostic surface repeats it.
    diagnostics = {
        "correction_commentary": result.get("correction_commentary"),
        "review_message": result.get("review_message"),
        "review_reason": result.get("review_reason"),
        "review_reasons": result.get("review_reasons"),
        "run_metrics": result.get("run_metrics"),
    }
    for secret in ("alex.example@example.test", "555-010-0199"):
        assert secret not in json.dumps(diagnostics)
    # The approved document is the caller's own resume, so redaction must not have
    # been applied to it: their contact details survive intact.
    if result["correction_status"] == "approved":
        md = result["final_written_section_markdown"]
        assert "alex.example@example.test" in md and "555-010-0199" in md
    assert result["processing_status"] in {"completed", "needs_review"}


def test_flag_on_still_withholds_when_redaction_cannot_be_proven(offline, monkeypatch):
    """If the redactor cannot guarantee correspondence, nothing is sent."""
    client, _ = offline
    monkeypatch.setenv(correction_provider.DISCLOSURE_SETTING, "1")
    session = ScriptedSession()
    monkeypatch.setattr(app.app.state, "http_session", session)
    pipeline(monkeypatch, draft=PII_DRAFT)

    import app as app_module

    def unprovable(_documents):
        return (), None

    monkeypatch.setattr("disclosure.redact_sources", unprovable)
    result = post(client, source=PII_SOURCE).json()
    assert result["correction_reason"] == "privacy_redaction_failed"
    assert result["requires_review"] is True
    assert result["publication_receipt"] is None
    assert session.requests == []
