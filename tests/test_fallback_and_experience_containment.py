"""Synthetic, offline regressions for fallback status and blocked experience output."""

import asyncio
import copy
import importlib
import json
import socket
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from orchestrator import PipelineOrchestrator
from publication_gate import apply_to_payload, decide
from stages.star_editor import StarEditor


MARKDOWN = (
    "## Professional Experience\n**Synthetic Workshop**\n- "
    + "Maintained equipment and documented repair procedures for the workshop team. " * 9
    + "\n## Education\nSynthetic College\n"
)
GENERATED = "SYNTHETIC_UNVERIFIED_ACHIEVEMENT_87_PERCENT"
SOURCE = {"title_line": "Technician | Synthetic Workshop", "snippet": "Maintained equipment"}
SUPPORT = {"verdict": "PASS (tier A only)", "checked": True}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access forbidden in synthetic tests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setenv("STAGE_COOLDOWN_SECONDS", "0")


@pytest.mark.parametrize("editor_flags,expected", [
    ({"fallback_used": True}, "completed_with_fallback"),
    ({"fallback": True}, "completed_with_fallback"),
    ({"fallback_used": False, "fallback": False}, "completed"),
])
def test_editor_fallback_flags_propagate_to_pipeline_and_gate(editor_flags, expected):
    orchestrator = PipelineOrchestrator(None)
    orchestrator.drafter.draft = AsyncMock(return_value={
        "rewritten_experiences": [], "total_bullets": 0, "quantification_score": 0,
    })
    orchestrator.star_editor.polish = AsyncMock(return_value={
        "final_markdown": MARKDOWN, "final_plain_text": MARKDOWN,
        "metrics_support_checked": True, **editor_flags,
    })
    result = asyncio.run(orchestrator.run_pipeline(
        resume_text=MARKDOWN, job_ad="Synthetic maintenance role",
        experiences=[{"company": "Synthetic Workshop", "role": "Technician"}],
        research_data={"domain_vocab": ["maintenance"], "implied_metrics": []},
    ))
    assert result["metrics"]["pipeline_status"] == expected
    assert result["stages"]["star_editor"]["status"] == (
        "fallback" if expected == "completed_with_fallback" else "completed"
    )
    decision = decide(resume_markdown=result["final_output"]["final_written_section_markdown"],
                      metrics_support=result["final_output"]["metrics_support"],
                      fallback_used=result["metrics"]["pipeline_status"] != "completed",
                      approved_entities=[])
    assert decision.blocked() is (expected != "completed")


def test_real_editor_direct_markdown_parser_sets_fallback_used():
    parsed = StarEditor(None)._parse_response(MARKDOWN)
    assert parsed["fallback_used"] is True
    assert not parsed.get("fallback")


def test_actual_editor_parser_fallback_blocks_pipeline():
    orchestrator = PipelineOrchestrator(None)
    parsed = orchestrator.star_editor._parse_response(MARKDOWN)
    parsed["metrics_support_checked"] = True
    orchestrator.star_editor.polish = AsyncMock(return_value=parsed)
    orchestrator.drafter.draft = AsyncMock(return_value={
        "rewritten_experiences": [], "total_bullets": 0, "quantification_score": 0,
    })
    result = asyncio.run(orchestrator.run_pipeline(
        resume_text=MARKDOWN, job_ad="Synthetic maintenance role",
        experiences=[{"company": "Synthetic Workshop", "role": "Technician"}],
        research_data={"domain_vocab": ["maintenance"], "implied_metrics": []},
    ))
    assert result["metrics"]["pipeline_status"] == "completed_with_fallback"
    decision = decide(resume_markdown=result["final_output"]["final_written_section_markdown"],
                      metrics_support=result["final_output"]["metrics_support"],
                      fallback_used=result["metrics"]["pipeline_status"] != "completed",
                      approved_entities=[])
    assert "fallback_output" in decision.reason_codes


def test_blocked_generated_experiences_withhold_aliases_and_draft_without_mutation():
    payload = {
        "pipeline_version": "3.0", "final_written_section": MARKDOWN,
        "experience": [{"bullets": [GENERATED]}],
        "experiences": [{"snippet": GENERATED}],
        "stages": {"drafter": {"rewritten_experiences": [{"bullets": [GENERATED]}],
                                "total_bullets": 1}},
        "source_experiences": [SOURCE], "education": [{"institution": "Synthetic College"}],
        "aggregate_skills": ["repair"],
    }
    before = copy.deepcopy(payload)
    decision = decide(resume_markdown=MARKDOWN, metrics_support=None, approved_entities=[])
    out = apply_to_payload(payload, decision)
    assert GENERATED not in json.dumps(out)
    assert not out["experience"] and not out["experiences"]
    assert not out["stages"]["drafter"]["rewritten_experiences"]
    assert out["stages"]["drafter"]["total_bullets"] == 1
    assert out["source_experiences"] == [SOURCE]
    assert out["education"] == payload["education"]
    assert out["aggregate_skills"] == ["repair"]
    assert payload == before


@pytest.mark.parametrize("key", ["experience", "experiences"])
def test_blocked_source_only_experience_is_preserved(key):
    decision = decide(resume_markdown=MARKDOWN, metrics_support=None, approved_entities=[])
    assert apply_to_payload({key: [SOURCE]}, decision)[key] == [SOURCE]


def test_blocked_pipeline_with_single_stale_experience_alias_is_withheld():
    payload = {"pipeline_status": "completed_with_fallback",
               "final_written_section": MARKDOWN,
               "experiences": [{"snippet": GENERATED}], "source_experiences": [SOURCE]}
    decision = decide(resume_markdown=MARKDOWN, metrics_support=None, approved_entities=[])
    out = apply_to_payload(payload, decision)
    assert not out["experiences"]
    assert out["source_experiences"] == [SOURCE]


def test_publishable_generated_experiences_are_unchanged():
    payload = {"pipeline_version": "3.0", "experience": [{"snippet": GENERATED}],
               "experiences": [{"snippet": GENERATED}]}
    decision = decide(resume_markdown=MARKDOWN, metrics_support=SUPPORT, approved_entities=[])
    assert decision.publishable
    out = apply_to_payload(payload, decision)
    assert out["experience"] == payload["experience"]
    assert out["experiences"] == payload["experiences"]


@pytest.fixture
def route_client(monkeypatch):
    import dotenv
    from pydantic_settings.sources import DotEnvSettingsSource

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: False)
    monkeypatch.setattr(DotEnvSettingsSource, "_read_env_files", lambda self: {})
    app = importlib.import_module("app")
    monkeypatch.setitem(app.app.dependency_overrides, app.get_api_key, lambda: "synthetic")
    monkeypatch.setattr(app.settings, "openrouter_api_key_1", None)
    monkeypatch.setattr(app.settings, "openrouter_api_key_2", None)

    async def pipeline(**kwargs):
        return {
            "pipeline_version": "3.0", "pipeline_status": "completed_with_fallback",
            "final_written_section_markdown": MARKDOWN, "final_written_section": MARKDOWN,
            "rewritten_resume": MARKDOWN, "experience": [{"bullets": [GENERATED]}],
            "experiences": [{"snippet": GENERATED}], "aggregate_skills": ["repair"],
            "metrics_support": SUPPORT,
        }

    monkeypatch.setattr(app, "run_new_pipeline_async", pipeline)
    client = TestClient(app.app)
    yield client
    client.close()


@pytest.mark.parametrize("sse", [False, True])
def test_real_analyze_response_withholds_generated_experience(route_client, sse):
    response = route_client.post(
        "/analyze", json={"resume_text": MARKDOWN, "job_ad": "Synthetic maintenance"},
        headers={"Accept": "text/event-stream"} if sse else {},
    )
    assert response.status_code == 200, response.text
    assert GENERATED not in response.text
    if sse:
        events = [block for block in response.text.split("\n\n") if block.strip()]
        assert any(block.startswith("event: withheld\n") for block in events)
        assert not any(block.startswith("event: token\n") for block in events)
        body = json.loads(next(block.split("data: ", 1)[1] for block in events
                               if block.startswith("event: result\n")))
    else:
        body = response.json()
    assert body["processing_status"] == "needs_review"
    assert "fallback_output" in body["review_reasons"]
    assert not body["experiences"]
    assert body["aggregate_skills"] == ["repair"]


@pytest.mark.parametrize("sse", [False, True])
def test_route_withholds_real_integration_experience_shape(route_client, monkeypatch, sse):
    app = importlib.import_module("app")
    seen = []
    original = app.publication_gate.apply_to_payload

    def record_gate_input(payload, decision):
        seen.append(copy.deepcopy(payload))
        return original(payload, decision)

    async def pipeline(**kwargs):
        exp = [{"company": "Synthetic Workshop", "role": "Technician", "duration": "",
                "start_date": "", "end_date": "", "location": "",
                "bullets": [GENERATED]}]
        return {"pipeline_version": "3.0", "pipeline_status": "completed_with_fallback",
                "final_written_section_markdown": MARKDOWN, "final_written_section": MARKDOWN,
                "experience": exp, "experiences": exp, "metrics_support": SUPPORT}

    monkeypatch.setattr(app, "run_new_pipeline_async", pipeline)
    monkeypatch.setattr(app.publication_gate, "apply_to_payload", record_gate_input)
    response = route_client.post(
        "/analyze", json={"resume_text": MARKDOWN, "job_ad": "Synthetic maintenance"},
        headers={"Accept": "text/event-stream"} if sse else {},
    )
    assert response.status_code == 200, response.text
    assert seen[0]["experiences"][0]["bullets"] == [GENERATED]
    assert GENERATED not in response.text
    if sse:
        assert "event: withheld" in response.text and "event: token" not in response.text
        data = next(block.split("data: ", 1)[1] for block in response.text.split("\n\n")
                    if block.startswith("event: result\n"))
        body = json.loads(data)
    else:
        body = response.json()
    assert not body["experiences"]