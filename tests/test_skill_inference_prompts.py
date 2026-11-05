"""The mode must be enforced by the *prompts*, not by the sampling temperature.

A previous review point was explicit: keeping the legacy ``creativity_mode``
style dial separate from the permission policy is only acceptable if the
effective prompts still prohibit skill bridging and invented duties in
Conservative mode. Temperature is a style knob; it is not an authority.

These tests capture the real generated prompts from the real builders:

* ``stages.drafter._build_drafter_prompts`` — the offline builder that mirrors
  what ``Drafter.draft`` sends;
* ``Drafter.draft`` itself, via a fake client, so the sparse-input instruction is
  included;
* ``StarEditor._create_user_prompt`` — the final writer's prompt.

They are deliberately structural. They do not search for a blacklist of invented
words; they assert that the *instruction* to invent is gone and that the mode
boundary is stated, in both the normal and sparse-input paths.
"""
import asyncio
import json

import pytest

from skill_inference import SkillInferenceMode


EXPERIENCES = [
    {
        "company": "Cascade Power Solutions",
        "title": "Journeyman Electrician",
        "duration": "03/2020 -- Present",
        "bullets": [
            "Compared weekly sales figures to identify trends across 12 branches.",
            "Trained 6 apprentices on lockout/tagout.",
        ],
    }
]

# Under 50 words once serialised in a deliberately small payload: exercises the
# sparse-input branch that previously told the model to invent industry tasks.
SPARSE_EXPERIENCES = [
    {"company": "Acme", "title": "Analyst", "bullets": ["Did analysis."]}
]

JOB_AD = "Senior Data Analyst. Requires Python, SQL and trend analysis."


def _flat(text: str) -> str:
    """Collapse whitespace so assertions test wording, not source line wrapping."""
    return " ".join(text.split()).lower()


# ── The retired invention instruction ───────────────────────────────────────


INVENTED_TASK_PHRASES = (
    "inferring standard industry tasks",
    "how the work was likely done",
    "expand them into full bullets",
)


@pytest.mark.parametrize("mode", ["imaginative", "conservative"])
def test_drafter_prompt_never_asks_for_invented_tasks(mode):
    """No mode licenses guessed duties. This was the sparse-input instruction."""
    from stages.drafter import _build_drafter_prompts

    system_prompt, user_prompt = _build_drafter_prompts(
        EXPERIENCES, JOB_AD, {"domain_vocab": []}, skill_inference_mode=mode
    )
    combined = _flat(system_prompt + "\n" + user_prompt)

    for phrase in INVENTED_TASK_PHRASES:
        assert phrase not in combined, f"{mode} prompt still invites invention: {phrase!r}"


def test_sparse_input_no_longer_authorizes_invention_in_either_mode():
    """The sparse branch is the highest-risk path: it fires on thin input only."""
    from unittest.mock import AsyncMock, Mock

    from stages.drafter import Drafter

    captured = {}

    async def fake_call(**kwargs):
        captured["system_prompt"] = kwargs["system_prompt"]
        captured["user_prompt"] = kwargs["user_prompt"]
        return json.dumps({"rewritten_experiences": []})

    for mode in ("imaginative", "conservative"):
        client = Mock()
        client.call_llm_async = AsyncMock(side_effect=fake_call)
        drafter = Drafter(client)
        asyncio.run(
            drafter.draft(
                SPARSE_EXPERIENCES,
                JOB_AD,
                {"domain_vocab": []},
                skill_inference_mode=mode,
            )
        )
        combined = _flat(captured["system_prompt"] + "\n" + captured["user_prompt"])

        for phrase in INVENTED_TASK_PHRASES:
            assert phrase not in combined, f"sparse {mode} prompt invites invention"
        # The replacement instruction is present and bounds the output to source.
        assert "keep every bullet limited to what the notes state" in combined
        assert "shorter bullet is the correct output" in combined


# ── Conservative: no bridging permission at all ─────────────────────────────


def test_conservative_drafter_prompt_forbids_skill_inference():
    from stages.drafter import _build_drafter_prompts

    system_prompt, _ = _build_drafter_prompts(
        EXPERIENCES, JOB_AD, {"domain_vocab": []}, skill_inference_mode="conservative"
    )
    text = _flat(system_prompt)

    assert "skill-inference policy: conservative" in text
    assert "do not infer transferable skills" in text
    # And it must not also hand out the permission in the same prompt.
    assert "you may use a general, transferable-skill label" not in text


def test_conservative_editor_prompt_forbids_skill_inference():
    from stages.star_editor import StarEditor

    editor = StarEditor(None)
    prompt = editor._create_user_prompt(
        [{"company": "Cascade", "role": "Electrician", "bullets": ["Compared figures."]}],
        "mid",
        [],
        original_resume_text="Skills:\n- conduit bending",
        skill_inference_mode="conservative",
    )
    text = _flat(prompt)

    assert "skill-inference policy: conservative" in text
    assert "do not infer transferable skills" in text
    # No role vocabulary is handed to the conservative editor at all.
    assert "domain vocabulary to incorporate" not in text


def test_conservative_editor_receives_no_domain_vocab_even_if_supplied():
    """The vocabulary is research output; Conservative has nothing to relabel."""
    from unittest.mock import AsyncMock, Mock

    from stages.star_editor import StarEditor

    captured = {}

    async def fake_call(**kwargs):
        captured["user_prompt"] = kwargs["user_prompt"]
        return json.dumps(
            {
                "final_markdown": "## Professional Experience\n**A** -- B\n- did work",
                "final_plain_text": "did work",
                "editorial_notes": "synthetic",
            }
        )

    client = Mock()
    client.call_llm_async = AsyncMock(side_effect=fake_call)
    editor = StarEditor(client)
    asyncio.run(
        editor.polish(
            {"rewritten_experiences": EXPERIENCES, "seniority_applied": "mid"},
            {"domain_vocab": ["Kubernetes", "Snowflake"], "implied_skills": ["x"]},
            original_resume_text="Skills:\n- conduit bending",
            skill_inference_mode="conservative",
        )
    )
    text = _flat(captured["user_prompt"])

    assert "kubernetes" not in text
    assert "snowflake" not in text


# ── Imaginative: permitted scope is stated with its limits ──────────────────


def test_imaginative_drafter_prompt_states_permission_and_limits():
    from stages.drafter import _build_drafter_prompts

    system_prompt, _ = _build_drafter_prompts(
        EXPERIENCES, JOB_AD, {"domain_vocab": []}, skill_inference_mode="imaginative"
    )
    text = _flat(system_prompt)

    assert "skill-inference policy: imaginative" in text
    # Allowed: a general transferable-skill label.
    assert "general, transferable-skill label" in text
    # Forbidden in the same breath: specifics that would manufacture a skill.
    for prohibited in (
        "specific implementation",
        "algorithm",
        "named software",
        "proficiency level",
    ):
        assert prohibited in text, f"missing limit: {prohibited}"
    # Duties/activities/outcomes are explicitly outside the permission.
    assert "you may not invent duties" in text
    assert "never adds work" in text


def test_imaginative_editor_prompt_forbids_proficiency_upgrades():
    from stages.star_editor import StarEditor

    prompt = StarEditor(None)._create_user_prompt(
        [{"company": "Cascade", "role": "Electrician", "bullets": ["Compared figures."]}],
        "mid",
        ["trend analysis"],
        original_resume_text="Skills:\n- conduit bending",
        skill_inference_mode="imaginative",
    )
    text = _flat(prompt)

    assert "skill-inference policy: imaginative" in text
    assert "transferable-skill label" in text
    # Neutral-wording requirement from the policy v1.1 clarification.
    for phrase in ("expert", "advanced", "proven", "years of experience", "leadership"):
        assert phrase in text, f"neutral-wording rule missing: {phrase}"


def test_imaginative_is_the_default_when_mode_is_omitted():
    from stages.drafter import _build_drafter_prompts

    system_prompt, _ = _build_drafter_prompts(EXPERIENCES, JOB_AD, {"domain_vocab": []})
    assert "skill-inference policy: imaginative" in _flat(system_prompt)


# ── The two modes are genuinely different prompts ───────────────────────────


def test_the_two_modes_produce_different_system_prompts():
    from stages.drafter import _build_drafter_prompts

    imaginative, _ = _build_drafter_prompts(
        EXPERIENCES, JOB_AD, {"domain_vocab": []}, skill_inference_mode="imaginative"
    )
    conservative, _ = _build_drafter_prompts(
        EXPERIENCES, JOB_AD, {"domain_vocab": []}, skill_inference_mode="conservative"
    )
    assert imaginative != conservative
    assert "conservative" not in _flat(imaginative).split("skill-inference policy")[1][:40]
