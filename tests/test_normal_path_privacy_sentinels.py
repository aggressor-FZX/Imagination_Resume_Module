"""Offline synthetic probes for the scoped normal-pipeline logging boundary."""
import asyncio
import json
import logging
import socket

import pytest

import imaginator_new_integration as integration
from orchestrator import PipelineOrchestrator
from stages.drafter import Drafter
from stages.researcher import Researcher
from stages.star_editor import StarEditor


MARKER = "SYNTHETIC_PRIVATE_SENTINEL_7F3"


@pytest.fixture(autouse=True)
def offline(monkeypatch, caplog):
    def blocked(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setenv("STAGE_COOLDOWN_SECONDS", "0")
    caplog.set_level(logging.DEBUG)


def assert_no_disclosure(caplog, *errors):
    assert MARKER not in caplog.text
    assert MARKER not in repr(errors)
    assert not any(record.exc_info or record.stack_info for record in caplog.records)


@pytest.mark.parametrize("stage", ["researcher", "drafter", "editor"])
def test_malformed_model_responses_are_contained(stage, caplog):
    malformed = '{"private": "' + MARKER  # JSON error can include untrusted text
    if stage == "researcher":
        result = Researcher(None)._parse_response(malformed)
        assert result["fallback_used"]
        assert result["parse_error"] == "invalid_json"
    elif stage == "drafter":
        result = Drafter(None)._parse_response(malformed, [])
        assert result["fallback_used"]
        assert result["parse_error"] == "invalid_json"
    else:
        result = StarEditor(None)._parse_response(malformed)
        assert result["fallback_used"]
        assert result["error"] == "invalid_json"
    assert_no_disclosure(caplog, result)


def test_editor_json_extraction_and_markdown_fallback(caplog):
    editor = StarEditor(None)
    extracted = editor._parse_response(MARKER + ' {"final_markdown":"## Experience"}')
    assert extracted["final_markdown"] == "## Experience"
    markdown = editor._parse_response("## " + MARKER)
    assert markdown["fallback_used"] and markdown["final_markdown"] == "## " + MARKER
    assert markdown["parse_error"] == "invalid_json"
    empty = editor._parse_response("")
    assert empty["fallback"]
    assert_no_disclosure(caplog, markdown["parse_error"])


def test_editor_polish_exception_falls_back_without_traceback(caplog):
    class FailingClient:
        async def call_llm_async(self, **kwargs):
            raise RuntimeError(MARKER)

    result = asyncio.run(StarEditor(FailingClient()).polish(
        {"rewritten_experiences": [{"company": "Example", "role": "Engineer"}]}, {}))
    assert result["fallback"] and "Example" in result["final_markdown"]
    assert_no_disclosure(caplog, result["editorial_notes"])


@pytest.mark.parametrize("failed_stage", ["researcher_exception", "researcher_error", "drafter", "editor", "outer"])
def test_orchestrator_contains_unknown_stage_and_pipeline_errors(monkeypatch, caplog, failed_stage):
    orchestrator = PipelineOrchestrator(None)

    async def research(*args, **kwargs):
        if failed_stage == "researcher_exception":
            raise RuntimeError(MARKER)
        if failed_stage == "researcher_error":
            return {"error": MARKER, "implied_metrics": [], "domain_vocab": []}
        return {"implied_metrics": [], "domain_vocab": [], "implied_skills": []}

    async def draft(*args, **kwargs):
        if failed_stage == "drafter":
            raise RuntimeError(MARKER)
        return {"rewritten_experiences": [], "fallback": False}

    async def edit(*args, **kwargs):
        if failed_stage == "editor":
            raise RuntimeError(MARKER)
        return {"final_markdown": "## Experience", "model_used": MARKER}

    monkeypatch.setattr(orchestrator.researcher, "analyze", research)
    monkeypatch.setattr(orchestrator.drafter, "draft", draft)
    monkeypatch.setattr(orchestrator.star_editor, "polish", edit)
    if failed_stage == "outer":
        def fail(*args, **kwargs):
            raise RuntimeError(MARKER)
        monkeypatch.setattr("orchestrator.get_temperatures", fail)

    result = asyncio.run(orchestrator.run_pipeline(
        MARKER, "synthetic job", [{"company": "Example", "role": "Engineer"}]))
    assert result["errors"]
    assert result["metrics"]["pipeline_status"] == (
        "failed" if failed_stage == "outer" else
        "completed_with_fallback")
    assert_no_disclosure(caplog, result["errors"], result["metrics"].get("error"))


def test_integration_top_level_failure_hides_unknown_error(monkeypatch, caplog):
    def fail(*args, **kwargs):
        raise RuntimeError(MARKER)
    monkeypatch.setattr(integration, "sanitize_resume_pii", fail)
    result = asyncio.run(integration.run_new_pipeline_async(MARKER, "job"))
    assert result["pipeline_status"] == "failed"
    assert result["pipeline_metrics"]["errors"] == ["Pipeline failed"]
    assert result["run_metrics"]["failures"] == ["Pipeline failed"]
    assert_no_disclosure(caplog, result)


@pytest.mark.parametrize("provided_title", [False, True])
def test_normal_integration_logs_no_input_or_model_content(monkeypatch, caplog, provided_title):
    class Client:
        def __init__(self, **kwargs):
            pass

        async def call_llm_async(self, **kwargs):
            if "job_title" in kwargs["system_prompt"]:
                return json.dumps({"job_title": MARKER})
            raise RuntimeError(MARKER)

        def get_usage_stats(self):
            return {"total_tokens": 0}

    class Orchestrator:
        def __init__(self, client):
            pass

        async def run_pipeline(self, **kwargs):
            return {"errors": [MARKER], "stages": {}, "metrics": {"pipeline_status": "completed"},
                    "final_output": {"final_written_section": "experience", "final_written_section_markdown": "## Experience"}}

    monkeypatch.setattr(integration, "LLMClientAdapter", Client)
    monkeypatch.setattr(integration, "PipelineOrchestrator", Orchestrator)
    monkeypatch.setattr(integration, "CAREER_ALCHEMY_AVAILABLE", False)
    monkeypatch.setattr(integration, "parse_experiences", lambda text: [
        {"company": MARKER, "role": MARKER, "title_line": MARKER, "bullets": ["capstone"]}])
    result = asyncio.run(integration.run_new_pipeline_async(
        MARKER, "synthetic job", openrouter_api_keys=["fake"],
        job_title=MARKER if provided_title else None, onet_code=MARKER))
    assert result["pipeline_status"] == "completed"
    assert result["critique_score"] == 0.75
    assert result["projects"][0]["company"] == MARKER
    assert_no_disclosure(caplog)


def test_real_stages_through_integration_keep_candidate_and_model_data_out_of_logs(monkeypatch, caplog):
    class Client:
        def __init__(self, **kwargs):
            pass

        async def call_llm_async(self, **kwargs):
            system = kwargs["system_prompt"]
            if "ATS scorer" in system:
                return '{"score": 0.8}'
            if "STAR" in system and "EDITOR" not in system.upper():
                return json.dumps({"rewritten_experiences": [{"company": MARKER,
                    "role": MARKER, "bullets": ["Maintained equipment"]}]})
            return json.dumps({"final_markdown": "## Professional Experience\n- Maintained equipment",
                               "final_plain_text": "Professional Experience\nMaintained equipment"})

        def get_usage_stats(self):
            return {"total_tokens": 0}

    monkeypatch.setattr(integration, "LLMClientAdapter", Client)
    monkeypatch.setattr(integration, "CAREER_ALCHEMY_AVAILABLE", False)
    result = asyncio.run(integration.run_new_pipeline_async(
        MARKER, "synthetic maintenance role", openrouter_api_keys=["fake"], job_title=MARKER,
        experience=[{"company": MARKER, "role": MARKER, "bullets": ["Maintained equipment"]}],
        research_data={"implied_metrics": [MARKER], "domain_vocab": [MARKER],
                       "implied_skills": []}))
    assert result["pipeline_status"] == "completed"
    assert_no_disclosure(caplog)


def test_onet_resolution_errors_do_not_expose_title_or_exception(monkeypatch, caplog):
    async def fail(*args, **kwargs):
        raise RuntimeError(MARKER)

    class Client:
        call_llm_async = fail

    monkeypatch.setattr(integration, "_search_onet_keyword", fail)
    monkeypatch.setattr(integration, "_build_onet_search_variants", lambda title: [title])
    assert asyncio.run(integration._llm_resolve_onet(MARKER, Client())) == {}
    assert asyncio.run(integration._scored_keyword_resolve(MARKER)) == {}
    assert_no_disclosure(caplog)


def test_editor_company_guard_keeps_identity_out_of_log(caplog):
    result = StarEditor(None)._apply_hallucination_guard(
        {"final_markdown": "## Experience\nEngineer | **" + MARKER + "**"},
        [{"company": "Example"}], target_company=MARKER)
    assert "Example" in result["final_markdown"]
    assert_no_disclosure(caplog)