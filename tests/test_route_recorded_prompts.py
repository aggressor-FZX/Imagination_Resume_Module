"""Drive the real website `/analyze` route through the live stages, recording prompts.

``test_website_writer_path.py`` asserts that the module source contains the right
wording and that the route wiring points at the stages package. That is necessary but
weaker than it looks: a prompt can be built differently at runtime from how it reads in
source, and a stage can be reached with inputs the static check never sees.

This module drives the actual route:

    POST /analyze  ->  app._analyze_resume_impl
                   ->  run_new_pipeline_async      (not stubbed)
                   ->  PipelineOrchestrator
                   ->  stages/researcher.py, stages/drafter.py, stages/star_editor.py

and records the `(system_prompt, user_prompt)` pairs that reach the model transport at
each stage. Only ``LLMClientAdapter.call_llm_async`` is replaced: that is the seam a real
network call sits behind, so everything above it — the route, the orchestrator, the
stages and their prompt construction — is production code.

The correction engine is stubbed to *approve the draft unchanged* here on purpose, so
these tests observe the writer's output rather than the correction outcome. The
correction and publication path is exercised separately in
``test_route_semantic_cases.py``.

Labelled **[route-recorded]**: real route and real stages, recorded transport, no live
model.
"""
import asyncio
import json
import socket

import pytest
from fastapi.testclient import TestClient

import app
import llm_client_adapter
from source_correction import CorrectionResult


SOURCE_TEXT = (
    "SIMON EXAMPLE\n"
    "Cascade Power Solutions, LLC -- Example City, WA\n"
    "Journeyman Electrician | 03/2020 - Present\n"
    "- Installed and maintained commercial electrical systems across 40+ job sites.\n"
    "- Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices.\n"
    "- Cut rework by 18% by standardizing panel labeling and inspection checklists.\n"
    "\n"
    "Example Technical College -- Example City, WA\n"
    "Associate of Applied Science, Electrical Systems Technology | 2016\n"
    "\n"
    "Washington Journey Level Electrician License\n"
    "NFPA 70E Arc Flash Safety\n"
)

JOB_AD = ("Senior Electrician. Requires PLC programming, SCADA and AutoCAD. "
          "Must hold OSHA 30 and have 12 years of experience.")


class RecordingTransport:
    """Stands in for the OpenRouter adapter and records every prompt.

    Answers use each stage's real contract so the pipeline completes normally and the
    route reaches publication; the point of this suite is the prompts, not fallbacks.
    """

    FINAL_MARKDOWN = (
        "## Professional Experience\n\n"
        "**Cascade Power Solutions, LLC** -- Example City, WA\n"
        "*Journeyman Electrician | 03/2020 - Present*\n"
        "- Installed and maintained commercial electrical systems across 40+ job sites.\n"
        "- Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices.\n"
        "- Cut rework by 18% by standardizing panel labeling and inspection checklists.\n"
        "\n## Education\n\n"
        "**Example Technical College** -- Example City, WA\n"
        "*Associate of Applied Science, Electrical Systems Technology | 2016*\n"
        "\n## Certifications\n\n"
        "- Washington Journey Level Electrician License\n"
        "- NFPA 70E Arc Flash Safety\n"
        "\n## Skills\n\n"
        "- Commercial electrical systems, conduit bending, panel labeling\n"
    )

    def __init__(self):
        self.calls = []

    async def call_llm_async(self, system_prompt, user_prompt, model=None,
                             temperature=0.7, response_format=None, timeout=None,
                             fallback_models=None, **kwargs):
        self.calls.append({"system": system_prompt, "user": user_prompt,
                           "model": model})
        joined = (system_prompt + "\n" + user_prompt).lower()
        # Dispatch on the stage's own contract text, not on keywords that also
        # occur inside the drafter's job-ad and boundary wording.
        if "career transition strategist" in joined or "implied_skills" in joined:
            return json.dumps({
                "implied_skills": ["SCADA", "PLC Programming"],
                "implied_metrics": ["Reduced downtime by 40%"],
                "industry_metrics": ["Reduced downtime by 40%"],
                "domain_vocab": ["maintenance"],
                "research_notes": "Electricians often hold OSHA 30.",
            })
        if "star editor" in joined or "final_markdown" in joined:
            return json.dumps({
                "final_markdown": self.FINAL_MARKDOWN,
                "final_plain_text": self.FINAL_MARKDOWN,
                "editorial_notes": "synthetic",
            })
        if "ats scorer" in joined:
            return json.dumps({"score": 0.9})
        if "job data extractor" in joined:
            return json.dumps({"job_title": "Senior Electrician"})
        # Drafter (and anything else): return its rewritten_experiences contract.
        return json.dumps({
            "rewritten_experiences": [{
                "company": "Cascade Power Solutions, LLC",
                "role": "Journeyman Electrician",
                "bullets": [
                    "Installed and maintained commercial electrical systems "
                    "across 40+ job sites.",
                    "Trained 6 apprentices on lockout/tagout and NFPA 70E safe "
                    "work practices.",
                    "Cut rework by 18% by standardizing panel labeling and "
                    "inspection checklists.",
                ],
                "metrics_used": ["18%"],
                "style_transfer_rationale": "Restated the source bullets.",
            }],
            "seniority_applied": "senior",
            "quantification_score": 0.9,
        })

    def get_usage_stats(self):
        return {"calls": [], "total_prompt_tokens": 0,
                "total_completion_tokens": 0, "total_tokens": 0,
                "estimated_cost_usd": 0.0}

    def stage_text(self, needle):
        """Joined text of every recorded call that mentions ``needle``."""
        return " ".join(" ".join((c["system"] + " " + c["user"]).split()).lower()
                        for c in self.calls
                        if needle in (c["system"] + c["user"]).lower())


@pytest.fixture
def recorded(monkeypatch):
    """Run the real route with a recording transport and an approving engine."""
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

    transport = RecordingTransport()

    async def call(self, *args, **kwargs):
        return await transport.call_llm_async(*args, **kwargs)

    monkeypatch.setattr(llm_client_adapter.LLMClientAdapter, "call_llm_async", call)
    monkeypatch.setattr(llm_client_adapter.LLMClientAdapter, "get_usage_stats",
                        lambda self: transport.get_usage_stats())

    # Approve the writer's output unchanged: this suite observes the WRITER, and the
    # correction path has its own suite. The result is still the engine's own.
    async def approve(markdown, sources, provider, **kwargs):
        return CorrectionResult("approved", markdown, None,
                                "Model-assisted review is not proof.", None, 1, 0)

    monkeypatch.setattr(app, "correct", approve)

    with TestClient(app.app) as client:
        yield client, transport

    app.app.dependency_overrides.clear()


def _analyze(client, source=SOURCE_TEXT):
    return client.post("/analyze", headers={"X-Cogito-Owner-Id": "offline-owner"},
                       json={"resume_text": source, "job_ad": JOB_AD})


# ---------------------------------------------------------------------------
# [route-recorded] the prompts actually sent
# ---------------------------------------------------------------------------

def test_route_reaches_the_stage_transport(recorded):
    """The real route reaches the live stages and the transport records them."""
    client, transport = recorded
    response = _analyze(client)
    assert response.status_code == 200
    assert transport.calls, "no stage ever reached the model transport"


def test_recorded_drafter_prompt_carries_the_tightened_instruction(recorded):
    """The prompt SENT at drafting time contains the fixed wording, not just source.

    Asserting on the recorded prompt is the point: it is what a model would read.
    """
    client, transport = recorded
    _analyze(client)
    drafter = transport.stage_text("technical resume writer")
    assert drafter, "drafter stage prompt not recorded"
    assert "normalize terminology only" in drafter
    assert "does not license" in drafter
    assert "evidence boundary" in drafter
    # The two permissive instructions are gone from what is actually sent.
    assert 'when user said "database work"' not in drafter
    assert "heavily implied by the research data" not in drafter


def test_recorded_drafter_prompt_never_offers_professional_as_a_title(recorded):
    """The sent prompt must not name the placeholder as an allowed title."""
    client, transport = recorded
    _analyze(client)
    drafter = transport.stage_text("technical resume writer")
    assert drafter
    assert '"professional"' not in drafter
    assert "professional role" not in drafter
    # The real title is offered.
    assert "journeyman electrician" in drafter


def test_recorded_drafter_prompt_keeps_research_out_of_evidence(recorded):
    """Research output must not be offered as candidate evidence in the sent prompt."""
    client, transport = recorded
    _analyze(client)
    drafter = transport.stage_text("technical resume writer")
    assert drafter
    # The research block is gone. Note that SCADA / PLC programming legitimately
    # appear in the prompt as part of the JOB AD, which is labelled context-only and
    # which the model must not treat as evidence. What must not appear is research
    # output presented as facts about this candidate.
    assert "research insights" not in drafter
    assert "reduced downtime by 40%" not in drafter
    assert "implied skills" not in drafter
    assert "domain vocabulary" not in drafter
    # The evidence boundary is stated where the JD is supplied.
    assert "evidence boundary" in drafter
    assert "is not evidence about" in drafter


def test_route_returns_a_published_result(recorded):
    """The route completes and publishes, so the stages really ran end to end."""
    client, transport = recorded
    result = _analyze(client).json()
    assert result.get("correction_status") == "approved", result
    assert result.get("publication_receipt"), "a receipt must be issued"
    assert result.get("approved_export_payload")
    assert result.get("processing_status") == "completed"
