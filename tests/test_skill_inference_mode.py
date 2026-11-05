"""Phase A: the two-mode skill-inference settings contract.

These tests pin the *permission policy* boundary, not prose style:

* omission means ``imaginative`` (the policy default);
* ``conservative`` and ``imaginative`` are the only accepted values;
* an unknown value fails validation instead of silently choosing a mode — the
  previous three-mode ``creativity_mode`` vocabulary must not leak in;
* ``conservative`` performs **zero** Researcher calls, because the entire point
  of the mode is that the bridging inference is not paid for;
* the effective mode and policy version travel with the result.

No network, no provider calls. The orchestrator's real methods are replaced with
spies, but the mode branch under test is the production code path.
"""
import asyncio
import os

import pytest

from skill_inference import (
    DEFAULT_SKILL_INFERENCE_MODE,
    POLICY_VERSION,
    SkillInferenceMode,
    UnknownSkillInferenceMode,
    is_imaginative,
    mode_contract,
    normalize_mode,
    policy_version_for,
)


# ── Module-level contract ───────────────────────────────────────────────────


def test_default_mode_is_imaginative():
    assert DEFAULT_SKILL_INFERENCE_MODE is SkillInferenceMode.IMAGINATIVE
    assert normalize_mode(None) is SkillInferenceMode.IMAGINATIVE


@pytest.mark.parametrize("absent", [None, "", "   ", "\t\n"])
def test_absent_value_takes_the_policy_default(absent):
    """JSON and multipart both send "" for an unset control; treat it as absent."""
    assert normalize_mode(absent) is SkillInferenceMode.IMAGINATIVE


@pytest.mark.parametrize(
    "value,expected",
    [
        ("imaginative", SkillInferenceMode.IMAGINATIVE),
        ("conservative", SkillInferenceMode.CONSERVATIVE),
        ("IMAGINATIVE", SkillInferenceMode.IMAGINATIVE),
        ("  Conservative  ", SkillInferenceMode.CONSERVATIVE),
        (SkillInferenceMode.CONSERVATIVE, SkillInferenceMode.CONSERVATIVE),
    ],
)
def test_the_two_supported_values_are_accepted(value, expected):
    assert normalize_mode(value) is expected


@pytest.mark.parametrize("value", ["balanced", "bold", "creative", "strict", "none"])
def test_unknown_values_are_rejected(value):
    """Never silently choose a mode: that turns a permission into an accident."""
    with pytest.raises(UnknownSkillInferenceMode):
        normalize_mode(value)


def test_non_string_values_are_rejected():
    for value in (1, True, [], {}, object()):
        with pytest.raises(UnknownSkillInferenceMode):
            normalize_mode(value)


def test_policy_version_distinguishes_the_two_modes():
    assert policy_version_for(SkillInferenceMode.IMAGINATIVE) == POLICY_VERSION
    assert (
        policy_version_for(SkillInferenceMode.CONSERVATIVE)
        != policy_version_for(SkillInferenceMode.IMAGINATIVE)
    )


def test_contract_is_server_normalised_only():
    contract = mode_contract("conservative")
    assert contract == {
        "skill_inference_mode": "conservative",
        "skill_inference_policy_version": policy_version_for(
            SkillInferenceMode.CONSERVATIVE
        ),
    }


def test_is_imaginative_helper():
    assert is_imaginative(None) is True
    assert is_imaginative("imaginative") is True
    assert is_imaginative("conservative") is False


# ── Request model boundary ─────────────────────────────────────────────────


def _request(**kwargs):
    from models import AnalysisRequest

    return AnalysisRequest(
        resume_text="Python developer with five years of experience.",
        job_ad="Data Analyst role requiring SQL and Python.",
        **kwargs,
    )


def test_request_defaults_to_imaginative_when_omitted():
    assert _request().skill_inference_mode is None
    from skill_inference import from_payload

    assert from_payload(_request().model_dump()) is SkillInferenceMode.IMAGINATIVE


def test_request_accepts_both_values_and_aliases():
    assert _request(skill_inference_mode="conservative").skill_inference_mode is (
        SkillInferenceMode.CONSERVATIVE
    )
    assert _request(skillInferenceMode="IMAGINATIVE").skill_inference_mode is (
        SkillInferenceMode.IMAGINATIVE
    )


def test_request_rejects_unknown_mode():
    with pytest.raises(Exception):
        _request(skill_inference_mode="balanced")


def test_request_does_not_infer_mode_from_creativity_mode():
    """Style and permission are separate concerns.

    ``creativity_mode=bold`` must not be read as permission to bridge skills.
    """
    req = _request(creativity_mode="bold")
    from skill_inference import from_payload

    assert req.skill_inference_mode is None
    assert from_payload(req.model_dump()) is SkillInferenceMode.IMAGINATIVE
    req2 = _request(creativity_mode="conservative")
    assert req2.skill_inference_mode is None


# ── Response model boundary ────────────────────────────────────────────────


def test_response_carries_effective_policy_fields():
    from models import AnalysisResponse

    resp = AnalysisResponse(
        processing_time_seconds=0.1,
        skill_inference_mode="conservative",
        skill_inference_policy_version=policy_version_for(
            SkillInferenceMode.CONSERVATIVE
        ),
    )
    assert resp.skill_inference_mode == "conservative"
    assert resp.skill_inference_policy_version == "conservative-v1"


# ── Orchestrator behaviour ─────────────────────────────────────────────────


def _orchestrator(research=None):
    from unittest.mock import AsyncMock, Mock

    from orchestrator import PipelineOrchestrator

    client = Mock()
    orchestrator = PipelineOrchestrator(client)
    orchestrator.researcher.analyze = AsyncMock(
        return_value=research
        or {
            "implied_metrics": ["m"],
            "domain_vocab": ["v"],
            "implied_skills": ["s"],
        }
    )
    orchestrator.drafter.draft = AsyncMock(
        return_value={"rewritten_experiences": [], "fallback": True}
    )
    orchestrator.star_editor.polish = AsyncMock(
        return_value={"final_markdown": "## Experience\n- x", "fallback": True}
    )
    return orchestrator


def _run(orchestrator, **overrides):
    kwargs = dict(
        resume_text="Python developer with five years of experience. " * 2,
        job_ad="Data Analyst role requiring SQL.",
        experiences=[{"company": "A", "title": "Dev", "bullets": ["did work"]}],
    )
    kwargs.update(overrides)
    return asyncio.run(orchestrator.run_pipeline(**kwargs))


def test_conservative_makes_zero_researcher_calls():
    orchestrator = _orchestrator()
    orchestrator.researcher.analyze.side_effect = AssertionError(
        "Conservative mode must not invoke the Researcher at all"
    )
    result = _run(orchestrator, skill_inference_mode="conservative")

    assert orchestrator.researcher.analyze.call_count == 0
    assert result["stages"]["researcher"]["status"] == "skipped_by_policy"
    assert result["stages"]["researcher"]["model_calls"] == 0
    assert result["skill_inference_mode"] == "conservative"


def test_conservative_ignores_precomputed_imaginative_research():
    """A cached bundle must not acquire permission from this request's mode."""
    orchestrator = _orchestrator()
    orchestrator.researcher.analyze.side_effect = AssertionError("must not run")
    result = _run(
        orchestrator,
        skill_inference_mode="conservative",
        research_data={"implied_skills": ["Time-Series Analysis"], "domain_vocab": ["x"]},
    )

    assert orchestrator.researcher.analyze.call_count == 0
    assert result["stages"]["researcher"]["status"] == "skipped_by_policy"
    assert result["final_output"]["skill_inference_grounding"] == {"status": "not_requested"}
    # The proposal is not carried into the effective bundle for this request.
    assert "precomputed_research_ignored_by_policy" in result.get("warnings", [])


def test_conservative_bundle_has_no_default_qualifications():
    orchestrator = _orchestrator()
    result = _run(orchestrator, skill_inference_mode="conservative")
    # No invented fallback skills, unlike the old "Problem Solving" defaults.
    processed = result["processed_skills"]
    assert processed["inferred_skills"] == []


def test_imaginative_calls_the_researcher_once():
    orchestrator = _orchestrator()
    result = _run(orchestrator, skill_inference_mode="imaginative")

    assert orchestrator.researcher.analyze.call_count == 1
    assert result["stages"]["researcher"]["status"] == "completed"
    assert result["skill_inference_mode"] == "imaginative"


def test_omitted_mode_defaults_to_imaginative_at_the_orchestrator():
    orchestrator = _orchestrator()
    result = _run(orchestrator)

    assert result["skill_inference_mode"] == "imaginative"
    assert orchestrator.researcher.analyze.call_count == 1


def test_orchestrator_rejects_unknown_mode():
    orchestrator = _orchestrator()
    with pytest.raises(UnknownSkillInferenceMode):
        _run(orchestrator, skill_inference_mode="balanced")


def test_effective_policy_travels_on_the_final_output():
    orchestrator = _orchestrator()
    result = _run(orchestrator, skill_inference_mode="conservative")

    assert result["final_output"]["skill_inference_mode"] == "conservative"
    assert (
        result["final_output"]["skill_inference_policy_version"]
        == "conservative-v1"
    )
    assert result["skill_inference_policy_version"] == "conservative-v1"


def test_mode_does_not_change_prose_style_temperature():
    """The permission policy must not silently replace the style control.

    Conservative is a *policy* value here; it is not the legacy creativity
    preset, and the two are resolved independently.
    """
    orchestrator = _orchestrator()
    _run(orchestrator, skill_inference_mode="conservative", creativity_mode="bold")
    # The Drafter still received the bold prose temperature.
    _, kwargs = orchestrator.drafter.draft.call_args
    assert kwargs["temperature_override"] == pytest.approx(0.65)
