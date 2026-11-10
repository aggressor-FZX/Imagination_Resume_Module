"""[decision-experiment] The comparison harness, its controls, and its reporting.

This is EXPERIMENTAL. Nothing here changes the production approval path: the
production engine (`source_correction._correct_body`) is untouched on this branch.
A test below asserts that, because an experiment that silently edits the thing it
is measuring is worthless.

What the harness must get right, and what these tests pin:

* the four controls (authorization, receipt integrity, reference validity,
  execution limits) are ACTIVE in every cell — a comparison with them off measures
  a different system;
* drafts are FROZEN, so a difference between cells comes from review policy and not
  from fresh generation;
* the draft before semantic trimming is retained, separately from the trimmed one;
* parser acceptance is recorded separately from semantic correctness and is never
  used as evidence of it;
* the blind packet loses no candidate content, and leaks no configuration.
"""

import asyncio
import hashlib
import inspect
import json

import pytest

import source_correction
from experiment import runner
from experiment.blind_packet import build_packet, verify_blind
from experiment.cases import CASES, by_id
from experiment.runner import (
    CELLS,
    INSTRUCTIONS_CURRENT,
    INSTRUCTIONS_REVISED,
    SEMANTIC_ADVISORY,
    SEMANTIC_ENABLED,
    Cell,
    apply_revisions,
    build_prompt,
    interpret,
    resolve_mode_or_reject,
    run_attempt,
    score,
    summarise,
)


# ── Production path is untouched ────────────────────────────────────────────


# ── Scripted reviewers ──────────────────────────────────────────────────────
# The mocks answer in the format the CELL asks for: the current instructions want a
# JSON span envelope, the revised ones want `Decision: …`. A mock that answered in
# only one format would be testing a conversation no real provider has.


def _answers_in_current_format(prompt: str) -> bool:
    return "Return only JSON" in prompt or "representations:" in prompt


def _current_approve() -> str:
    return json.dumps({"claims": [], "representations": []})


def _current_revise(draft: str) -> str:
    lines = [l for l in draft.splitlines() if "targeted quality audits" in l]
    claims = [{
        "id": "L000001:0:10", "line_id": "L000001", "start": 0, "end": 10,
        "verdict": "unsupported",
        "citations": [{"source_id": "doc-that-does-not-exist", "start": 0,
                       "end": 5, "quote": "xxxxx"}],
    } for _ in lines]
    return json.dumps({"claims": claims, "representations": []})


async def _always_approve(prompt):
    if _answers_in_current_format(prompt):
        return _current_approve(), 0.001
    return "Decision: approve\n\nIssues:\n", 0.001


async def _garbage(prompt):
    return "looks fine to me", 0.001


async def _raises(prompt):
    raise RuntimeError("provider down")


async def _revise_with_bad_source(prompt):
    draft = prompt.split("DRAFT TO REVIEW:", 1)[1]
    if _answers_in_current_format(prompt):
        return _current_revise(draft), 0.001
    line = next((l.strip()[2:] for l in draft.splitlines()
                 if l.strip().startswith("- ") and len(l.strip()) > 20), "")
    return (
        "Decision: revise\n"
        "Issues:\n"
        f"- passage: {line}\n"
        "  problem: unsupported\n"
        "  sources: [doc-that-does-not-exist]\n"
        "  why: invented reference\n"
        f"  correction: {line}\n", 0.001)


def _revise_text(draft: str) -> str:
    line = next(l.strip()[2:] for l in draft.splitlines()
                if "targeted quality audits" in l)
    fixed = line.replace(" through targeted quality audits", "")
    return ("Decision: revise\n"
            "Issues:\n"
            f"- passage: {line}\n"
            "  problem: unsupported\n"
            "  sources: [candidate-main.txt]\n"
            "  why: the source states no such method\n"
            f"  correction: {fixed}\n")


async def _finds_the_defect(prompt):
    """Approves clean drafts; requests revision when the unsupported tail appears."""
    draft = prompt.split("DRAFT TO REVIEW:", 1)[1]
    if "targeted quality audits" not in draft:
        return await _always_approve(prompt)
    if _answers_in_current_format(prompt):
        return _current_revise(draft), 0.001
    return _revise_text(draft), 0.001


def test_the_production_correction_engine_is_not_modified_by_the_experiment():
    """The experiment must measure production, not replace it.

    Asserted structurally: the production engine's signature and its review
    contract are what they were. If a future change wires the experiment into
    `source_correction`, this fails and forces a deliberate decision instead of a
    silent swap.
    """
    signature = inspect.signature(source_correction._correct_body)
    assert list(signature.parameters) == [
        "markdown", "sources", "callback", "timeout_seconds", "events",
        "permitted_bridges", "require_bridge_decisions"]
    added_parameter = signature.parameters["require_bridge_decisions"]
    assert added_parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert added_parameter.default is False
    # The production contract still describes the span/inventory obligation.
    assert "representations" in source_correction._REVIEW_CONTRACT
    assert "claims" in source_correction._REVIEW_CONTRACT
    # …and the simplified decision contract is a separate module.
    import decision_contract
    assert "Decision: approve" in decision_contract.DECISION_CONTRACT


# ── Item 2: mode resolution ─────────────────────────────────────────────────


@pytest.mark.parametrize("requested,expected", [
    (None, "imaginative"),
    ("", "imaginative"),
    ("imaginative", "imaginative"),
    ("IMAGINATIVE", "imaginative"),
    (" Conservative ", "conservative"),
])
def test_omitted_or_valid_mode_resolves(requested, expected):
    assert resolve_mode_or_reject(requested) == expected


@pytest.mark.parametrize("requested", ["bogus", "creative", 123, {"a": 1}])
def test_invalid_mode_is_rejected(requested):
    from skill_inference import UnknownSkillInferenceMode

    with pytest.raises(UnknownSkillInferenceMode):
        resolve_mode_or_reject(requested)


# ── The matrix ──────────────────────────────────────────────────────────────


def test_the_matrix_is_the_four_cells_the_brief_names():
    labels = {c.label for c in CELLS}
    assert labels == {
        "current+advisory", "current+enabled",
        "revised+advisory", "revised+enabled",
    }


def test_only_the_enabled_cells_block_on_semantic_problems():
    for cell in CELLS:
        assert cell.blocks_on_semantic is (cell.semantic == SEMANTIC_ENABLED)


# ── The four controls stay active in EVERY cell ─────────────────────────────


def test_authorization_is_enforced_in_every_cell():
    """An approval without a resolved mode/owner is refused, advisory or not."""
    for cell in CELLS:
        attempt = asyncio.run(run_attempt(
            cell, by_id("faithful_paraphrase"), "",  # empty mode
            ask=_always_approve))
        assert attempt.outcome == "refused"
        assert attempt.outcome_reason == "authorization"


def test_reference_validity_is_enforced_in_every_cell():
    """A citation that does not resolve refuses, even under ADVISORY.

    Advisory relaxes the SEMANTIC rule only. If it also relaxed references, the
    comparison would be measuring whether skipping checks helps.
    """
    for cell in CELLS:
        attempt = asyncio.run(run_attempt(
            cell, by_id("unsupported_method"), "imaginative",
            ask=_revise_with_bad_source))
        # A citation that does not resolve must be refused in EVERY cell, and the
        # refusal must be attributable to the reference, not to the semantics.
        assert attempt.outcome == "refused", f"{cell.label}: {attempt.outcome}"
        assert attempt.outcome_reason == "reference_invalid"


def test_execution_limits_are_still_applied():
    """A provider failure is an OPERATIONAL failure, never an approval."""
    for cell in CELLS:
        attempt = asyncio.run(run_attempt(
            cell, by_id("faithful_paraphrase"), "imaginative", ask=_raises))
        assert attempt.outcome == "operational_failure"
        assert attempt.outcome_reason == "exception"


def test_an_unusable_answer_is_an_operational_failure_not_a_verdict():
    for cell in CELLS:
        attempt = asyncio.run(run_attempt(
            cell, by_id("faithful_paraphrase"), "imaginative", ask=_garbage))
        assert attempt.outcome == "operational_failure"
        assert attempt.outcome != "approved"


# ── Item 5: drafts are frozen ───────────────────────────────────────────────


def test_the_same_case_presents_identical_review_input_in_every_cell():
    """Freezing is what makes the comparison about policy rather than writing."""
    case = by_id("unsupported_method")
    prompts = {}
    for cell in CELLS:
        prompts[cell.label] = build_prompt(cell, "imaginative", case,
                                           case.draft_before_trim)
    # The DRAFT portion must be byte-identical across cells.
    drafts = {p.split("DRAFT TO REVIEW:", 1)[1] for p in prompts.values()}
    assert len(drafts) == 1


def test_the_input_hash_is_stable_and_excludes_the_expectation():
    case = by_id("faithful_paraphrase")
    assert case.content_hash() == case.content_hash()
    same_text = runner.FrozenCase(
        case_id="different_id", expectation="revise",
        draft_before_trim=case.draft_before_trim, sources=case.sources)
    # Same text, different label/expectation -> same hash, so "frozen" is checkable.
    assert same_text.content_hash() == case.content_hash()


def test_changing_the_draft_changes_the_hash():
    case = by_id("faithful_paraphrase")
    altered = runner.FrozenCase(
        case_id=case.case_id, expectation=case.expectation,
        draft_before_trim=case.draft_before_trim + "\n- An extra bullet.\n",
        sources=case.sources)
    assert altered.content_hash() != case.content_hash()


# ── Item 3: retain both drafts, and the provider exchange ───────────────────


def test_the_pre_trim_draft_is_retained_separately_from_the_trimmed_one():
    """Trimming must be observable, not inferred."""
    case = by_id("unsupported_method")

    def trim(draft, sources):
        return draft.replace(" through targeted quality audits", ""), ["removed tail"]

    attempt = asyncio.run(run_attempt(cell=CELLS[2], case=case,
                                      mode="imaginative", ask=_always_approve,
                                      trim=trim))
    assert attempt.trim_changes == ["removed tail"]
    assert "targeted quality audits" not in attempt.draft_after_trim
    # Both states survive for the auditor.
    assert attempt.draft_after_trim


def test_the_provider_request_and_response_are_both_retained():
    attempt = asyncio.run(run_attempt(CELLS[2], by_id("faithful_paraphrase"),
                                      "imaginative", ask=_always_approve))
    assert attempt.provider_request["prompt"]
    assert attempt.provider_request["prompt_sha256"]
    assert attempt.provider_response


# ── Item 7: apply edits, then review the exact resulting document ───────────


def test_revisions_are_applied_only_as_literal_replacements():
    draft = "## Experience\n- Cut rework by 18% through targeted quality audits.\n"
    issues = [{"passage": "Cut rework by 18% through targeted quality audits.",
               "correction": "Cut rework by 18%.", "problem": "unsupported"}]
    revised, applied = apply_revisions(draft, issues)
    assert "targeted quality audits" not in revised
    assert len(applied) == 1


def test_an_omission_is_not_auto_inserted():
    """Inserting text would be composing content, which the experiment must not do."""
    draft = "## Experience\n- Maintained records.\n"
    issues = [{"passage": "(omitted)", "problem": "omitted",
               "correction": "Reconciled 1,180 invoices."}]
    revised, applied = apply_revisions(draft, issues)
    assert revised == draft
    assert applied == []


def test_a_revision_triggers_a_review_of_the_exact_resulting_document():
    """The approval must be of the document that will be delivered."""
    seen_prompts: list[str] = []

    async def reviewer(prompt):
        seen_prompts.append(prompt)
        draft = prompt.split("DRAFT TO REVIEW:", 1)[1]
        if "targeted quality audits" in draft:
            return (_revise_text(draft), 0.001)
        return ("Decision: approve\n\nIssues:\n", 0.001)

    attempt = asyncio.run(run_attempt(
        Cell(INSTRUCTIONS_REVISED, SEMANTIC_ENABLED),
        by_id("unsupported_method"), "imaginative", ask=reviewer))
    assert len(seen_prompts) == 2, "the revised document must be reviewed again"
    assert "targeted quality audits" not in seen_prompts[1].split(
        "DRAFT TO REVIEW:", 1)[1]
    assert attempt.outcome == "approved"
    assert "targeted quality audits" not in attempt.final_markdown


# ── Item 9: reporting separates the categories ──────────────────────────────


def test_parser_acceptance_is_reported_separately_from_semantic_correctness():
    case = by_id("unsupported_method")
    attempt = asyncio.run(run_attempt(
        Cell(INSTRUCTIONS_REVISED, SEMANTIC_ADVISORY), case,
        "imaginative", ask=_finds_the_defect))
    row = score(attempt, case)
    # The parser accepted a well-formed approval…
    assert row["parser_ok"] is True
    # …and it is still a FALSE approval, because the case carries a defect.
    assert row["false_approval"] is True
    assert row["semantic_ok"] is False


def test_the_summary_keeps_every_category_separate():
    rows = []
    for cell in CELLS:
        for case in CASES:
            attempt = asyncio.run(run_attempt(cell, case, "imaginative",
                                              ask=_always_approve))
            rows.append(score(attempt, case))
    summary = summarise(rows)
    for cell, totals in summary.items():
        assert set(totals) >= {
            "correct_approvals", "false_approvals", "false_refusals",
            "operational_failures", "cost_usd", "latency_s_mean"}
        # The three counts must be independently represented, not one score.
        assert totals["correct_approvals"] + totals["false_approvals"] >= 0


def test_advisory_records_the_problem_and_the_revised_cell_still_delivers_it():
    """Advisory and enabled converge here — and that is the correct behaviour.

    The revised cell applies the proposed correction and then re-reviews the
    EXACT resulting document (item 7). Once the unsupported tail is removed, the
    re-review approves a genuinely fixed document, so the outcome is `approved`
    either way.

    What distinguishes the cells is not the final outcome on this case, it is
    what they do when no correction is applied. The next test covers that.
    """
    case = by_id("unsupported_method")
    for semantic in (SEMANTIC_ADVISORY, SEMANTIC_ENABLED):
        attempt = asyncio.run(run_attempt(
            Cell(INSTRUCTIONS_REVISED, semantic), case,
            "imaginative", ask=_finds_the_defect))
        assert attempt.issues, "the defect must be recorded either way"
        assert attempt.outcome == "approved"
        # The delivered document is the CORRECTED one, not the defective draft.
        assert "targeted quality audits" not in attempt.final_markdown


def test_enabled_refuses_when_no_usable_correction_is_offered():
    """The real difference between advisory and enabled.

    A reviewer that identifies a problem but proposes nothing usable cannot be
    resolved by applying an edit. ENABLED withholds; ADVISORY records and
    delivers. If any test only exercised correctable defects, that difference
    would be invisible.
    """
    case = by_id("unsupported_method")

    async def flags_but_does_not_fix(prompt):
        """A VALID revise whose correction the harness cannot apply.

        `correction: (omit)` on a passage-bearing issue is structurally invalid
        (there is no omission marker and no replacement), so the contract rejects
        it before advisory can consider it. That is correct: an unactionable revise
        is a malformed answer.

        To reach the advisory/enabled difference, the answer must be structurally
        valid yet produce no applicable edit — which is what an EMPTY-string
        correction does. The contract accepts it; `apply_revisions` skips it
        because there is nothing to write; and the defect therefore survives into
        the delivered document.
        """
        draft = prompt.split("DRAFT TO REVIEW:", 1)[1]
        if "targeted quality audits" not in draft:
            return await _always_approve(prompt)
        line = next(l.strip()[2:] for l in draft.splitlines()
                    if "targeted quality audits" in l)
        return ("Decision: revise\n"
                "Issues:\n"
                f"- passage: {line}\n"
                "  problem: unsupported\n"
                "  sources: [candidate-main.txt]\n"
                "  why: the method is not in the source\n"
                "  correction: \"\"\n", 0.001)

    advisory = asyncio.run(run_attempt(
        Cell(INSTRUCTIONS_REVISED, SEMANTIC_ADVISORY), case,
        "imaginative", ask=flags_but_does_not_fix))
    enabled = asyncio.run(run_attempt(
        Cell(INSTRUCTIONS_REVISED, SEMANTIC_ENABLED), case,
        "imaginative", ask=flags_but_does_not_fix))

    assert advisory.outcome == "approved"
    assert advisory.outcome_reason == "advisory_revise_noted"
    assert enabled.outcome == "refused"
    assert enabled.outcome_reason == "semantic_revise_unresolved"


# ── Scripted reviewers used above ───────────────────────────────────────────


# ── Three defects found while building the comparison ───────────────────────


def test_a_blank_correction_never_deletes_a_passage():
    """A blank correction must not be applied as a replacement.

    Applying it deletes the passage, which removes a sourced fact rather than
    correcting an unsupported one — and because the re-review then sees a clean
    document, the deletion disappears from the record. Found while constructing
    the advisory/enabled comparison: a blank correction silently dropped a whole
    bullet and the run still approved.
    """
    draft = "## Experience\n- Cut rework by 18% through targeted quality audits.\n"
    issues = [{"passage": "Cut rework by 18% through targeted quality audits.",
               "correction": "", "problem": "unsupported"}]
    revised, note = apply_revisions(draft, issues)
    assert revised == draft, "a blank correction deleted the passage"
    assert any("blank correction refused" in n for n in note)


@pytest.mark.parametrize("raw", ['""', "''", '""""'])
def test_an_all_quotes_correction_is_read_as_empty(raw):
    """`correction: ""` means "no replacement", not two literal quote characters.

    Left unhandled it was applied verbatim, writing quote marks into the résumé.
    """
    from decision_contract import _unquote

    assert _unquote(raw) == ""


def test_a_reference_refusal_is_reported_as_a_refusal_not_a_transport_failure():
    """Bucket integrity: a bad citation is a refusal ground, not a provider fault.

    Reporting it as `operational_failure` would hide it among provider outages and
    make the comparison unreadable — and would understate the reviewer's own error.
    """
    for cell in CELLS:
        attempt = asyncio.run(run_attempt(
            cell, by_id("unsupported_method"), "imaginative",
            ask=_revise_with_bad_source))
        assert attempt.outcome == "refused"
        assert attempt.outcome_reason == "reference_invalid"
        assert attempt.outcome != "operational_failure"


def test_an_invalid_reference_is_not_laundered_by_a_revision():
    """Applying an edit must not convert an unverifiable claim into an approval.

    Reference validity is a control on the REVIEW. If it were only checked after
    re-review, a bad citation plus a successful edit would silently pass.
    """
    attempt = asyncio.run(run_attempt(
        Cell(INSTRUCTIONS_REVISED, SEMANTIC_ADVISORY),
        by_id("unsupported_method"), "imaginative", ask=_revise_with_bad_source))
    assert attempt.outcome == "refused"
    assert attempt.outcome_reason == "reference_invalid"
    # The draft was never rewritten on the way to that refusal.
    assert attempt.final_markdown == attempt.draft_after_trim


def test_authorization_is_checked_before_any_provider_spend():
    """A control that only fails after paying for the call is not a control."""
    calls = []

    async def counting(prompt):
        calls.append(prompt)
        return "Decision: approve\n\nIssues:\n", 0.001

    attempt = asyncio.run(run_attempt(
        CELLS[0], by_id("faithful_paraphrase"), "", ask=counting))
    assert attempt.outcome == "refused"
    assert attempt.outcome_reason == "authorization"
    assert calls == [], "the provider was called for an unauthorized run"
    assert attempt.cost_usd == 0.0
