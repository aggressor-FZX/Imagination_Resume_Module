"""Offline regressions through real stages, orchestration, integration and /analyze."""

import asyncio
import copy
import json
import socket
from unittest.mock import AsyncMock

import pytest

import app
import imaginator_new_integration as integration
from orchestrator import PipelineOrchestrator
from stages.researcher import RESEARCHER_SYSTEM_PROMPT
from stages.star_editor import STAR_EDITOR_SYSTEM_PROMPT, StarEditor
from tests.test_analyze_source_correction_route import offline, result_from_sse  # noqa: F401
from tests.test_correction_integration_offline import DRAFT, SOURCE


EXPERIENCES = [
    {"company": f"Synthetic Employer {i}", "role": f"Technician {i}",
     "duration": "2020 - 2022", "location": "Example City",
     "bullets": [f"Synthetic draft {i} bullet {j}" for j in range(8)]}
    for i in range(4)
]
RESEARCH = {"implied_metrics": ["impact"], "domain_vocab": ["maintenance"],
            "implied_skills": []}


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network access forbidden")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setenv("STAGE_COOLDOWN_SECONDS", "0")


class ScriptedClient:
    def __init__(self, *, research="good", editor=None, **kwargs):
        self.research = research
        self.editor = editor
        self.calls = []

    async def call_llm_async(self, **kwargs):
        system = kwargs["system_prompt"]
        if system == RESEARCHER_SYSTEM_PROMPT:
            self.calls.append("researcher")
            if self.research == "error":
                raise RuntimeError("synthetic provider failure")
            if self.research == "malformed":
                return '{"private":'
            if self.research == "empty":
                return "{}"
            return json.dumps(RESEARCH)
        if system == STAR_EDITOR_SYSTEM_PROMPT:
            self.calls.append("editor")
            return self.editor if self.editor is not None else json.dumps({
                "final_markdown": DRAFT, "final_plain_text": DRAFT})
        if "ATS scorer" in system:
            self.calls.append("ats")
            return '{"score": 0.8}'
        self.calls.append("drafter")
        return json.dumps({"rewritten_experiences": [{
            "company": "Synthetic Workshop", "role": "Technician",
            "bullets": ["PRIVATE_GENERATED_DRAFT_SENTINEL"]}], "seniority_applied": "mid"})

    def get_usage_stats(self):
        return {"calls": [], "total_prompt_tokens": 0, "total_completion_tokens": 0,
                "total_tokens": 0, "estimated_cost_usd": 0.0}


@pytest.mark.parametrize("response", ["", "  \n", "{}", "{ }", '{"final_markdown":',
                                      '{"final_markdown":""}', "[]", None,
                                      "## Experience\n- Unverified model text"])
def test_polish_preserves_every_draft_experience_on_unusable_response(response):
    class Client:
        async def call_llm_async(self, **kwargs):
            return response

    draft = {"rewritten_experiences": copy.deepcopy(EXPERIENCES), "seniority_applied": "senior"}
    before = copy.deepcopy(draft)
    result = asyncio.run(StarEditor(Client()).polish(
        draft, RESEARCH, original_resume_text="source",
        education=[{"degree": "Diploma", "institution": "Synthetic College"}]))
    assert result.get("fallback") or result.get("fallback_used")
    assert result["seniority_level"] == "senior"
    for exp in EXPERIENCES:
        assert exp["company"] in result["final_markdown"]
        assert exp["role"] in result["final_markdown"]
        for bullet in exp["bullets"]:
            assert bullet in result["final_markdown"]
    assert draft == before


def test_prompt_failure_also_preserves_existing_draft(monkeypatch):
    editor = StarEditor(None)
    def fail(*args, **kwargs):
        raise ValueError("synthetic prompt failure")
    monkeypatch.setattr(editor, "_create_user_prompt", fail)
    result = asyncio.run(editor.polish({"rewritten_experiences": EXPERIENCES}, RESEARCH))
    assert result["fallback"]
    assert EXPERIENCES[-1]["bullets"][-1] in result["final_markdown"]


def test_empty_editor_does_not_sanitize_existing_draft_bullets():
    class Client:
        async def call_llm_async(self, **kwargs):
            return "{}"
    draft = {"rewritten_experiences": [{"company": "Synthetic Employer 12/2019 - 05/2023",
        "role": "Technician", "bullets": ["Draft mentioning Previous Company"]}]}
    result = asyncio.run(StarEditor(Client()).polish(
        draft, {"company_name": "Previous Company", "domain_vocab": []}))
    assert result["fallback"]
    assert "Synthetic Employer 12/2019 - 05/2023" in result["final_markdown"]
    assert "Draft mentioning Previous Company" in result["final_markdown"]


@pytest.mark.parametrize("mode,expected", [
    ("exception", "fallback"), ("error", "failed"),
    ("fallback", "fallback"), ("fallback_used", "fallback"),
    ("malformed", "fallback"), ("empty", "fallback"), ("healthy", "completed"),
])
def test_real_orchestration_researcher_status(mode, expected):
    client = ScriptedClient(research=mode if mode in {"malformed", "empty"} else "good")
    orchestrator = PipelineOrchestrator(client)
    if mode == "exception":
        orchestrator.researcher.analyze = AsyncMock(side_effect=RuntimeError("private"))
    elif mode == "error":
        orchestrator.researcher.analyze = AsyncMock(return_value={**RESEARCH, "error": "private"})
    elif mode in {"fallback", "fallback_used"}:
        orchestrator.researcher.analyze = AsyncMock(return_value={**RESEARCH, mode: True})
    result = asyncio.run(orchestrator.run_pipeline(
        SOURCE, "synthetic job", [{"company": "Synthetic Workshop", "role": "Technician"}]))
    assert result["stages"]["researcher"]["status"] == expected
    assert result["stages"]["drafter"]["status"] == "completed"
    assert result["stages"]["star_editor"]["status"] == "completed"
    assert result["metrics"]["pipeline_status"] == (
        "completed" if mode == "healthy" else "completed_with_fallback")
    assert "private" not in repr(result["errors"])


@pytest.mark.parametrize("research,expected", [
    ({**RESEARCH, "fallback": True}, "precomputed_fallback"),
    ({**RESEARCH, "fallback_used": True}, "precomputed_fallback"),
    ({**RESEARCH, "error": "private"}, "precomputed_fallback"),
    ({"fallback": False}, "precomputed_fallback"),
    (RESEARCH, "precomputed"),
])
def test_precomputed_research_status_and_skip(research, expected):
    client = ScriptedClient()
    orchestrator = PipelineOrchestrator(client)
    before = copy.deepcopy(research)
    result = asyncio.run(orchestrator.run_pipeline(
        SOURCE, "synthetic job", [{"company": "Synthetic Workshop", "role": "Technician"}],
        research_data=research))
    assert result["stages"]["researcher"]["status"] == expected
    assert result["metrics"]["pipeline_status"] == (
        "completed" if expected == "precomputed" else "completed_with_fallback")
    assert "researcher" not in client.calls
    assert research == before
    assert "private" not in repr(result["errors"])


@pytest.mark.parametrize("sse", [False, True])
@pytest.mark.parametrize("mode", ["research_error", "research_malformed", "research_empty", "precomputed_fallback",
                                  "precomputed_fallback_used", "precomputed_error", "editor_empty",
                                  "editor_malformed", "healthy"])
def test_real_route_and_integration_withhold_fallback_before_correction(offline, monkeypatch, mode, sse):
    client, session = offline
    instances = []

    def make_client(**kwargs):
        scripted = ScriptedClient(
            research={"research_error": "error", "research_malformed": "malformed",
                      "research_empty": "empty"}.get(mode, "good"),
            editor={"editor_empty": "{}", "editor_malformed": '{"final_markdown":'}.get(mode),
            **kwargs)
        instances.append(scripted)
        return scripted

    calls = []
    async def correction(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("synthetic correction stopped")

    monkeypatch.setattr(integration, "LLMClientAdapter", make_client)
    monkeypatch.setattr(integration, "CAREER_ALCHEMY_AVAILABLE", False)
    monkeypatch.setattr(app, "_run_source_correction", correction)
    precomputed = {
        "precomputed_fallback": {**RESEARCH, "fallback": True},
        "precomputed_fallback_used": {**RESEARCH, "fallback_used": True},
        "precomputed_error": {**RESEARCH, "error": "private"},
    }.get(mode)
    body = {"resume_text": SOURCE, "job_ad": "synthetic maintenance role",
            "job_title": "Technician", "research_data": precomputed,
            "experience": [{"company": "Synthetic Workshop", "role": "Technician",
                            "bullets": ["Maintained equipment"]}]}
    headers = {"X-Cogito-Owner-Id": "offline-owner"}
    if sse:
        headers["Accept"] = "text/event-stream"
    response = client.post("/analyze", json=body, headers=headers)
    assert response.status_code == 200
    if sse:
        blocks, result = result_from_sse(response)
        assert not any(block.startswith("event: token") for block in blocks)
        assert any(block.startswith("event: withheld") for block in blocks)
    else:
        result = response.json()
    assert instances and "drafter" in instances[0].calls and "editor" in instances[0].calls
    assert ("researcher" not in instances[0].calls) is (precomputed is not None)
    assert not session.requests
    assert result["requires_review"] and result["final_written_section_markdown"] is None
    assert result["publication_receipt"] is None
    assert "PRIVATE_GENERATED_DRAFT_SENTINEL" not in response.text
    if mode == "healthy":
        assert result["pipeline_status"] == "completed" and calls == [True]
    else:
        assert result["pipeline_status"] == "completed_with_fallback"
        assert "fallback_output" in result["review_reasons"]
        assert not calls