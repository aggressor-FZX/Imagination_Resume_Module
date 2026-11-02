"""The writer must not be offered invented material as candidate evidence.

Delivery run 4 produced a draft containing a Projects section, a GPA, technologies
(Python/SQL/VBA/AutoCAD/Revit/PLC), invented credential issuers and dates, and about six
fabricated metrics. The publication gate caught all of it, but catching is the last line
of defence; the source of the invention is the prompt.

These tests pin the *boundary* rather than individual phrases. They assert the prompts no
longer present web-research output as the candidate's own facts, and that the job ad is
labelled as requirement rather than evidence. They are deliberately structural: they do
not enumerate a blacklist of invented strings, because a blacklist would pass a prompt
that still invites invention with different words.
"""
import asyncio
import json
import socket

import pytest

import imaginator_flow


EXPERIENCES = [
    {
        "title_line": "Journeyman Electrician | 03/2020 - Present",
        "company": "Cascade Power Solutions, LLC",
        "snippet": "Installed commercial electrical systems across 40+ job sites.",
        "bullets": ["Trained 6 apprentices on lockout/tagout and NFPA 70E."],
    }
]

ANALYSIS = {
    "experiences": EXPERIENCES,
    "aggregate_skills": ["conduit bending", "panel labeling"],
    "seniority": {"level": "senior"},
}

JOB_AD = ("Senior Electrician. Requires PLC programming, SCADA, Python and AutoCAD. "
          "Must hold OSHA 30 and have 12 years of experience.")

RESEARCH = {
    "implied_skills": ["SCADA", "PLC Programming", "Python"],
    "industry_metrics": ["Reduced downtime by 40%", "Cut costs by $2M"],
    "star_suggestions": [{"experience_index": 0,
                          "result_metrics": ["Reduced downtime by 40%"]}],
    "research_notes": "Electricians typically hold OSHA 30.",
}


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network access forbidden")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


def _flat(text: str) -> str:
    """Collapse whitespace so assertions test wording, not source line wrapping."""
    return " ".join(text.split()).lower()


def _capture(monkeypatch):
    """Capture the (system, user) prompts of every LLM call; return them."""
    calls = []

    async def fake(system_prompt, user_prompt, **kwargs):
        calls.append((system_prompt, user_prompt))
        return "# Professional Experience\n\nSynthetic draft output."

    monkeypatch.setattr(imaginator_flow, "call_llm_async", fake)
    return calls


def _creative_prompt(monkeypatch):
    calls = _capture(monkeypatch)
    asyncio.run(imaginator_flow.run_creative_draft_async(
        ANALYSIS, JOB_AD, RESEARCH, resume_text="Synthetic resume text",
        openrouter_api_keys=["synthetic-key"]))
    assert calls, "creative drafter made no LLM call"
    return calls[-1]


def _final_editor_prompts(monkeypatch):
    calls = _capture(monkeypatch)

    async def fake(system_prompt, user_prompt, **kwargs):
        calls.append((system_prompt, user_prompt))
        # The final editor parses JSON; give it a well-formed envelope.
        return json.dumps({
            "final_written_section_markdown": "## Professional Experience\n- bullet",
            "final_written_section": "PROFESSIONAL EXPERIENCE\n- bullet",
            "editorial_notes": "synthetic",
        })

    monkeypatch.setattr(imaginator_flow, "call_llm_async", fake)
    asyncio.run(imaginator_flow.run_final_editor_async(
        "creative draft text", "star formatted text", RESEARCH, ANALYSIS, JOB_AD,
        openrouter_api_keys=["synthetic-key"]))
    assert calls
    return calls[-1]


def test_creative_prompt_does_not_offer_research_as_candidate_facts(monkeypatch):
    """Research output must not be presented as evidence of the candidate's history."""
    system_prompt, user_prompt = _creative_prompt(monkeypatch)

    combined = _flat(system_prompt + "\n" + user_prompt)
    # The invented-metric source must be gone from the drafting prompt entirely.
    assert "industry metrics" not in combined
    assert "implied skills" not in combined
    assert "web research insights" not in combined
    # Run 4's own values must not appear as candidate material.
    assert "reduced downtime by 40%" not in combined
    assert "cut costs by $2m" not in combined
    # And the boundary must be stated, not merely implied.
    assert "evidence boundary" in combined
    assert "never evidence of what the candidate" in combined


def test_creative_prompt_forbids_importing_job_requirements(monkeypatch):
    """The job ad may inform wording; it may not supply candidate facts."""
    system_prompt, user_prompt = _creative_prompt(monkeypatch)
    combined = _flat(system_prompt + "\n" + user_prompt)

    assert "never evidence of what the candidate has done" in combined
    assert "never restate a job requirement as a candidate achievement" in _flat(system_prompt)
    # The instruction to surface unlisted "implied" skills is what produced invented
    # technologies. It must be replaced by the opposite instruction.
    assert "surface these implied skills" not in combined
    assert "do not expand what the candidate is claimed to have done" in combined


def test_creative_prompt_forbids_inventing_metrics(monkeypatch):
    """No-metric claims must stay qualitative rather than estimated."""
    _, user_prompt = _creative_prompt(monkeypatch)
    lowered = _flat(user_prompt)
    assert "never invent a percentage" in lowered
    assert "qualitative achievement with no number" in lowered
    assert "never add a project the candidate did not list" in lowered


def test_final_editor_labels_drafts_as_editing_material_not_evidence(monkeypatch):
    """The final editor must not promote a claim merely because a draft asserts it."""
    system_prompt, user_prompt = _final_editor_prompt_pair(monkeypatch)
    lowered = _flat(user_prompt)

    assert "editing material, not evidence" in lowered
    assert "do not promote a claim merely because it appears there" in lowered
    # Research is no longer offered at all.
    assert "industry metrics" not in lowered
    assert "implied skills" not in lowered


def test_final_editor_forbids_filling_sections_with_invented_content(monkeypatch):
    """A required heading may appear empty; it may not be filled with inventions."""
    _, user_prompt = _final_editor_prompt_pair(monkeypatch)
    lowered = _flat(user_prompt)

    assert "do not add a projects section unless the candidate's material already contains projects" in lowered
    assert "filling it with invented projects is forbidden" in lowered
    assert "do not add a gpa" in lowered
    assert "do not add an issuer or an issue date" in lowered
    assert "do not add a technology or tool the candidate never mentions" in lowered
    assert "an incomplete resume is correct; an invented one is not" in lowered


def test_star_editor_labels_metric_shapes_as_illustrative(monkeypatch):
    """STAR research metrics are shapes, never this candidate's achievements."""
    calls = _capture(monkeypatch)
    asyncio.run(imaginator_flow.run_star_editor_async(
        "draft text", RESEARCH, EXPERIENCES, openrouter_api_keys=["synthetic-key"]))
    _, user_prompt = calls[-1]
    lowered = _flat(user_prompt)

    assert "illustrative star metric shapes" in lowered
    assert "not facts about this candidate" in lowered
    assert "use a number only if the draft you are formatting already contains that exact number" in lowered


def _final_editor_prompt_pair(monkeypatch):
    return _final_editor_prompts(monkeypatch)
