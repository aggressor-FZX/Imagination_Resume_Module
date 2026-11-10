"""Positive and negative offline controls for the binding gate.

Why a POSITIVE control
----------------------
The retained matrix released nothing, so every "the gate is fail-closed" statement
was supported only by the absence of releases. Absence is not a test: a gate that
never releases at all would look identical. `test_a_valid_approve_releases` is
therefore the load-bearing test in this file — it is the only evidence that the
approve path works rather than merely being unreached.

The three controls are deliberately symmetric:

    approve        -> release, on the first answer, with no repair attempted
    revise         -> no release on the first answer; release only after an
                      independently approved correction
    cannot_assess  -> withhold, and withhold for the stated reason

A gate that honoured only `revise` would pass a revise-only test suite. Running all
three is what makes the behaviour a decision rule rather than a default.

Nothing here calls a provider; every answer is a literal string.
"""

from __future__ import annotations

import asyncio

import pytest

import decision_contract as dc
from experiment.binding_candidate import run_binding
from experiment.cases_v2 import SOURCES, by_id

APPROVE = "Decision: approve\n\nIssues:\n- none\n"

# The contract requires a STATED reason for cannot_assess, and a reason is prose
# outside an issue block. Putting only an issue here is refused as
# `cannot_assess_without_reason`, which is the contract working as designed — so
# the fixture has to carry the sentence the contract asks for.
CANNOT_ASSESS = ("Decision: cannot_assess\n\n"
                 "The summary's years figure cannot be reconciled with the "
                 "employment dates, and no source states which is authoritative.\n"
                 "\nIssues:\n"
                 "- passage: Professional Summary\n"
                 "  problem: unresolved\n"
                 "  sources: [candidate-main.txt]\n"
                 "  why: the years figure and the dates disagree.\n")


def revise_for(passage: str, correction: str) -> str:
    return ("Decision: revise\n\nIssues:\n"
            f"- passage: {passage}\n"
            "  problem: unsupported\n"
            "  sources: [candidate-main.txt]\n"
            "  why: the method is not stated in the source.\n"
            f"  correction: {correction}\n")


def run(draft: str, answers: list[str]):
    seen: list[str] = []

    async def ask(current: str, _sources: dict[str, str]) -> str:
        seen.append(current)
        return answers[len(seen) - 1]

    return asyncio.run(run_binding(draft, SOURCES, ask=ask)), seen


# ── The positive control ───────────────────────────────────────────────────────

@pytest.mark.parametrize("case_id", [
    "faithful_paraphrase", "supported_method", "transferable_skill"])
def test_a_valid_approve_releases(case_id):
    """A valid approve must RELEASE, immediately, without a repair.

    Parametrised over the three cases whose preregistered expectation is approve,
    so the control cannot be satisfied by one lucky draft.
    """
    draft = by_id(case_id).draft_before_trim
    result, calls = run(draft, [APPROVE])
    assert result.release is True
    assert result.release_reason == "initial_llm_approve"
    assert result.repair_status == "not_attempted"
    assert result.proposed_document == draft
    assert calls == [draft], "an approval must not trigger a second call"


def test_a_valid_approve_releases_even_when_rules_observe_a_problem():
    """Heuristics are advisory: they explain, they do not block.

    The gate's contract is that the semantic decision is the model's. If an
    observation could veto, the release rule would be a rule system again.
    """
    draft = by_id("faithful_paraphrase").draft_before_trim
    result, _ = run(draft, [APPROVE])
    result_with_rules = asyncio.run(run_binding(
        draft, SOURCES,
        ask=_const(APPROVE),
        observe_rules=lambda _d, _s: ["rule would trim the summary"]))
    assert result.release and result_with_rules.release
    # The observation is retained and reported, so a reader can see what the rules
    # thought without the rules having blocked anything.
    assert result_with_rules.rule_observations == ["rule would trim the summary"]


def _const(value: str):
    """An `ask` that always returns the same answer."""
    async def ask(*_args, **_kwargs) -> str:
        return value
    return ask


# ── The revise control ────────────────────────────────────────────────────────

def test_revise_never_releases_without_a_new_approval():
    draft = ('## Professional Experience\n**Cascade Freight Partners** -- Kent, WA\n'
             '- Cut rework by 18% through targeted quality audits across the '
             'terminal fleet.\n')
    first = revise_for(
        "Cut rework by 18% through targeted quality audits across the terminal "
        "fleet.",
        "Cut rework by 18% by standardizing panel labeling across the terminal "
        "fleet.")
    result, calls = run(draft, [first, "Decision: revise\n"])
    assert result.initial_decision == "revise"
    assert result.proposed_document != draft
    assert result.release is False
    assert result.release_reason == "re_review_not_approved"
    assert len(calls) == 2


def test_revise_releases_only_after_approval_of_the_corrected_text():
    draft = ('## Professional Experience\n**Cascade Freight Partners** -- Kent, WA\n'
             '- Cut rework by 18% through targeted quality audits across the '
             'terminal fleet.\n')
    first = revise_for(
        "Cut rework by 18% through targeted quality audits across the terminal "
        "fleet.",
        "Cut rework by 18% by standardizing panel labeling across the terminal "
        "fleet.")
    result, calls = run(draft, [first, APPROVE])
    assert result.release is True
    assert result.release_reason == "corrected_document_llm_approved"
    # The second call saw the PROPOSAL, not the original.
    assert calls[1] == result.proposed_document != calls[0]


# ── The cannot_assess control ─────────────────────────────────────────────────

def test_cannot_assess_withholds_and_does_not_repair():
    draft = by_id("faithful_paraphrase").draft_before_trim
    result, calls = run(draft, [CANNOT_ASSESS])
    assert result.initial_decision == "cannot_assess"
    assert result.release is False
    assert result.release_reason == "llm_cannot_assess"
    assert result.repair_status == "not_attempted"
    assert len(calls) == 1, "cannot_assess must not spend a repair or re-review call"


# ── The three controls are genuinely distinct outcomes ────────────────────────

def test_the_three_decisions_produce_three_distinct_gate_behaviours():
    draft = by_id("faithful_paraphrase").draft_before_trim
    approve, _ = run(draft, [APPROVE])
    cannot, _ = run(draft, [CANNOT_ASSESS])
    revise_draft = ('## Professional Experience\n'
                    '**Cascade Freight Partners** -- Kent, WA\n'
                    '- Cut rework by 18% through targeted quality audits.\n')
    revise, _ = run(revise_draft, [revise_for(
        "Cut rework by 18% through targeted quality audits.",
        "Cut rework by 18% by standardizing panel labeling across the terminal "
        "fleet.")])

    assert approve.release is True
    assert cannot.release is False
    assert revise.release is False
    # Three different reasons: the gate distinguishes them rather than lumping
    # every non-release into one bucket.
    assert len({approve.release_reason, cannot.release_reason,
                revise.release_reason}) == 3


# ── Negative controls: the gate must not release on these ─────────────────────

@pytest.mark.parametrize("answer", [
    "Looks good to me, ship it.",                       # no decision word
    "Decision: probably fine\n",                        # word outside the set
    "Decision: approve\n\nIssues:\n- passage: Cut rework by 18%.\n"
    "  problem: unsupported\n  sources: [missing.txt]\n  why: x\n",  # bad source
])
def test_unusable_first_answers_never_release(answer):
    draft = by_id("faithful_paraphrase").draft_before_trim
    result, _ = run(draft, [answer])
    assert result.release is False
    assert result.release_reason == "initial_review_unverifiable"


def test_a_review_error_is_reported_as_an_error_not_a_refusal():
    draft = by_id("faithful_paraphrase").draft_before_trim

    async def boom(*_a, **_k):
        raise TimeoutError("provider timed out")

    result = asyncio.run(run_binding(draft, SOURCES, ask=boom))
    assert result.release is False
    assert result.release_reason == "review_error:TimeoutError"
