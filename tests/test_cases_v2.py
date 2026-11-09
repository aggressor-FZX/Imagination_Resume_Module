"""Tests for the v2 fixture, auditor, taxonomy and omission handling.

Every test here corresponds to a defect that was MEASURED in the v1 run or found
while building v2. None is decorative: each one fails if the specific defect
returns.

The controls matter as much as the assertions. A guard that cannot reject a known
bad input is not a guard, so `test_auditor_rejects_invalid_fixture` runs the
auditor against the v1 fixture and requires it to FAIL.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import decision_contract as dc  # noqa: E402
from experiment import replay_v2  # noqa: E402
from experiment.case_audit import audit_cases, employers_in, figures_in  # noqa: E402
from experiment.cases_v2 import (  # noqa: E402
    BASELINE,
    CASES_V2,
    MUTATIONS,
    SOURCES,
    by_id,
    cases_fingerprint,
)
from experiment.runner_v2 import (  # noqa: E402
    OUTCOME_APPROVED,
    OUTCOME_OPERATIONAL,
    OUTCOME_REFERENCE_FAILURE,
    OUTCOME_REFUSED,
    AttemptV2,
    RunBudget,
    apply_revisions_v2,
    describe_omission,
    score_v2,
)

FROZEN_RUN = ROOT / "results" / "comparison-20261109-0719"


# ══ 1. The fixture is VALID before any expectation means anything ══════════
def test_v2_fixture_passes_mechanical_audit():
    result = audit_cases(CASES_V2, baseline=BASELINE, mutations=MUTATIONS)
    assert result.ok, [f.__dict__ for f in result.failures]


def test_auditor_rejects_invalid_fixture():
    """CONTROL: the auditor must fail on the v1 fixture, or it proves nothing.

    v1 supplied both source documents but gave every draft only the main one. Three
    cases expected `approve`. The auditor must find that independently.
    """
    from experiment.cases import CASES as V1

    permissive = {c.case_id: {"kind": "undeclared"} for c in V1}
    result = audit_cases(V1, baseline=BASELINE, mutations=permissive)
    assert not result.ok
    missing = [f for f in result.failures
               if f.check == "covers_sources"
               and "Ridgeline Equipment Services" in f.detail]
    assert len(missing) == 3, "all three v1 approve-cases must be flagged"


def test_every_approve_case_contains_all_supplied_employers():
    """The precise v1 defect, asserted directly."""
    supplied = set()
    for text in SOURCES.values():
        supplied |= set(employers_in(text))
    assert "Ridgeline Equipment Services" in supplied
    for case in CASES_V2:
        if case.expectation != "approve":
            continue
        present = set(employers_in(case.draft_before_trim))
        assert supplied <= present, f"{case.case_id} omits {supplied - present}"


def test_approve_cases_are_the_complete_two_document_baseline():
    """The brief forbids fixing the fixture by DELETING the supplement."""
    assert len(SOURCES) == 2
    assert "candidate-supplement.txt" in SOURCES
    assert "Ridgeline Equipment Services" in SOURCES["candidate-supplement.txt"]

    baseline_employers = set(employers_in(BASELINE))
    assert {"Harborline Transit Authority", "Ridgeline Equipment Services",
            "Cascade Freight Partners"} <= baseline_employers


def test_approve_case_figures_all_trace_to_a_source():
    sourced = set()
    for text in SOURCES.values():
        sourced |= set(figures_in(text))
    for case in CASES_V2:
        if case.expectation != "approve":
            continue
        invented = set(figures_in(case.draft_before_trim)) - sourced
        assert not invented, f"{case.case_id} invents {invented}"


def test_cases_are_isolated_single_mutations():
    """Each case is the baseline plus one declared change, so a verdict is
    attributable to that change and nothing else."""
    for case in CASES_V2:
        mutation = MUTATIONS[case.case_id]
        if mutation["kind"] == "reword_only":
            assert case.draft_before_trim == BASELINE
            continue
        changed = (len([l for l in BASELINE.splitlines()
                        if l not in case.draft_before_trim.splitlines()])
                   + len([l for l in case.draft_before_trim.splitlines()
                          if l not in BASELINE.splitlines()]))
        assert 0 < changed <= 6, f"{case.case_id} changes {changed} lines"


def test_every_case_declares_a_mutation():
    assert set(MUTATIONS) == {c.case_id for c in CASES_V2}


def test_fixture_fingerprint_is_stable():
    assert cases_fingerprint() == cases_fingerprint()
    assert len(cases_fingerprint()) == 64


def test_omitted_record_case_really_drops_the_record():
    case = by_id("omitted_record")
    assert case.expectation == "revise"
    assert "Ridgeline Equipment Services" not in case.draft_before_trim
    assert "Ridgeline Equipment Services" in SOURCES["candidate-supplement.txt"]
    # Every OTHER record survives, so the only defect is the omission.
    others = set(employers_in(BASELINE)) - {"Ridgeline Equipment Services"}
    assert others <= set(employers_in(case.draft_before_trim))


def test_missing_fact_case_drops_exactly_one_figure():
    case = by_id("missing_fact")
    baseline_figures = figures_in(BASELINE)
    draft_figures = figures_in(case.draft_before_trim)
    assert baseline_figures.count("1180") == 1
    assert "1180" not in draft_figures
    # Nothing else numeric was lost.
    assert sorted(set(baseline_figures) - set(draft_figures)) == ["1180"]


# ══ 2. Reference taxonomy: convention is NOT refusal ═══════════════════════
def test_citation_real_quote_wrong_offset_is_a_convention_defect():
    """Measured on the retained run: all 10 quotes were real, all offsets wrong.

    Filing that as a semantic refusal credited the reviewer with a judgement it
    never made, and understated the legacy contract at the same time.
    """
    text = "alpha beta gamma delta"
    citation = {"source_id": "s", "start": 0, "end": 5, "quote": "beta"}
    assert replay_v2.resolve_citation(citation, {"s": text}) == "offsets_off"


def test_citation_exact_offset_matches():
    text = "alpha beta gamma"
    citation = {"source_id": "s", "start": 6, "end": 10, "quote": "beta"}
    assert replay_v2.resolve_citation(citation, {"s": text}) == "matches"


def test_citation_length_convention_is_accepted():
    """Some models write `end` as a LENGTH. The quote is still verifiable."""
    text = "alpha beta gamma"
    citation = {"source_id": "s", "start": 6, "end": 4, "quote": "beta"}
    assert replay_v2.resolve_citation(
        citation, {"s": text}) == "matches_length_convention"


def test_citation_quote_in_no_source_is_fabricated():
    citation = {"source_id": "s", "start": 0, "end": 4, "quote": "zeta"}
    assert replay_v2.resolve_citation(
        citation, {"s": "alpha beta gamma"}) == "fabricated"


def test_citation_unknown_source_is_named_as_such():
    citation = {"source_id": "nope", "start": 0, "end": 4, "quote": "beta"}
    assert replay_v2.resolve_citation(
        citation, {"s": "beta"}) == "unknown_source"


def test_whitespace_rewrapped_quote_is_not_fabrication():
    """A quote that reproduces a wrapped line is evidence, not invention."""
    text = "alpha\nbeta gamma"
    citation = {"source_id": "s", "start": 0, "end": 6, "quote": "alpha beta"}
    assert replay_v2.resolve_citation(citation, {"s": text}) == "offsets_off"


def test_quote_from_the_draft_is_its_own_category():
    """Measured on the v2 probe: 5 of 20 citations were draft text.

    Quoting the document under review and attributing it to a source is neither
    fabrication (the text exists) nor an offset error (the pointer is wrong, not
    mis-numbered). It needs its own name or the two get conflated again.
    """
    sources = {"s": "alpha beta gamma"}
    draft = "delta epsilon zeta"
    citation = {"source_id": "s", "start": 0, "end": 5, "quote": "epsilon"}
    assert replay_v2.resolve_citation(
        citation, sources, draft) == "cites_draft"


def test_quote_in_neither_document_is_still_fabricated():
    """The draft test must not become a loophole for invention."""
    sources = {"s": "alpha beta gamma"}
    draft = "delta epsilon zeta"
    citation = {"source_id": "s", "start": 0, "end": 5, "quote": "omega"}
    assert replay_v2.resolve_citation(
        citation, sources, draft) == "fabricated"


def test_cites_draft_outranks_an_approve_decision():
    """An approval whose evidence is the draft itself is not a semantic result."""
    envelope = {"claims": [{
        "id": "1:1:0:5", "verdict": "supported",
        "citations": [{"source_id": "candidate-main.txt", "start": 0, "end": 4,
                       "quote": "Produ"}]}]}
    result = replay_v2.classify("current+advisory", json.dumps(envelope),
                                BASELINE, SOURCES, None)
    assert result["category"] == replay_v2.CAT_REF_CITES_DRAFT


def test_fabrication_outranks_the_decision_word():
    """A decision reached by citing non-existent evidence is not semantic."""
    envelope = {"claims": [{
        "id": "1:1:0:5", "verdict": "supported",
        "citations": [{"source_id": "candidate-main.txt", "start": 0, "end": 4,
                       "quote": "zzzz"}]}]}
    result = replay_v2.classify("current+advisory", json.dumps(envelope),
                                BASELINE, SOURCES, None)
    assert result["category"] == replay_v2.CAT_REF_FABRICATED


# ══ 3. The frozen archive is preserved and read-only ═══════════════════════
@pytest.mark.skipif(not FROZEN_RUN.exists(), reason="archive not present")
def test_frozen_archive_verifies():
    import subprocess
    proc = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=FROZEN_RUN,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.skipif(not FROZEN_RUN.exists(), reason="archive not present")
def test_replay_v2_does_not_modify_the_archive():
    before = {p.name: p.stat().st_mtime_ns
              for p in FROZEN_RUN.iterdir() if p.is_file()}
    replay_v2.replay(FROZEN_RUN)
    after = {p.name: p.stat().st_mtime_ns
             for p in FROZEN_RUN.iterdir() if p.is_file()}
    assert before == after


@pytest.mark.skipif(not FROZEN_RUN.exists(), reason="archive not present")
def test_replay_v2_finds_no_verified_legacy_judgement():
    """The corrected reading: the legacy arm never produced a usable judgment.

    11 truncated + 1 offset-convention mismatch. Any `semantic:` category in the
    legacy arm would mean either a truncation was scored as a decision or an offset
    problem was called a refusal.
    """
    rows = replay_v2.replay(FROZEN_RUN)
    legacy = [r for r in rows if r["cell"].startswith("current")]
    assert len(legacy) == 12
    semantic = [r for r in legacy
                if r["category"] in replay_v2.SEMANTIC_CATEGORIES]
    assert semantic == [], [(r["case_id"], r["category"]) for r in semantic]
    truncated = [r for r in legacy
                 if r["category"] == replay_v2.CAT_TRUNCATION]
    assert len(truncated) == 11
    convention = [r for r in legacy
                  if r["category"] == replay_v2.CAT_REF_CONVENTION]
    assert len(convention) == 1


@pytest.mark.skipif(not FROZEN_RUN.exists(), reason="archive not present")
def test_replay_v2_separates_the_axes():
    summary = replay_v2.summarise(replay_v2.replay(FROZEN_RUN))
    assert summary["attempts"] == 24
    axes = summary["by_axis"]
    # No attempt may be counted on two axes, and all must be accounted for.
    assert sum(axes.values()) == 24


# ══ 4. Omission handling: identify, never auto-insert ══════════════════════
def test_omission_report_names_source_and_destination():
    """Item 6: the cited source AND where it belongs — without editing."""
    draft = by_id("omitted_record").draft_before_trim
    issue = {
        "passage": "- Maintained parts availability records for 7 service branches.",
        "problem": dc.PROBLEM_OMITTED,
        "sources": ["candidate-supplement.txt"],
        "correction": ("Ridgeline Equipment Services -- Fife, WA Inventory Analyst "
                       "08/2023 - Present Maintained parts availability records for "
                       "7 service branches."),
    }
    report = describe_omission(issue, draft, SOURCES)
    assert report.source_id == "candidate-supplement.txt"
    assert report.cites_real_source is True
    # The missing record's neighbours in the sources place it in the draft. The
    # supplement is ordered after the main document, so its single record genuinely
    # follows Cascade Freight — the inference reports document order, not a guess.
    assert report.destination_basis == "source_order"
    assert "Cascade Freight Partners" in report.destination
    assert report.applied is False


def test_omission_destination_prefers_the_reviewers_own_anchor():
    """When the reviewer points at real draft text, use it rather than infer."""
    issue = {
        "passage": "- Monitored monthly fuel use across 14 depots, comparing each "
                   "month with the same month a year earlier.",
        "problem": dc.PROBLEM_OMITTED,
        "sources": ["candidate-main.txt"],
        "correction": "Reconciled 1,180 fuel invoices",
    }
    report = describe_omission(issue, BASELINE, SOURCES)
    assert report.destination_basis == "anchor_in_draft"
    assert report.anchor_occurrences == 1
    assert report.applied is False


def test_omission_with_no_identifiable_destination_says_so():
    """Never invent a location. `undeterminable` is a valid, honest answer."""
    issue = {
        "passage": dc.OMITTED_MARKER,
        "problem": dc.PROBLEM_OMITTED,
        "sources": ["candidate-main.txt"],
        "correction": "Some sentence about a facility nobody named",
    }
    report = describe_omission(issue, BASELINE, SOURCES)
    if not report.anchor_resolves:
        assert report.destination_basis == "undeterminable"
        assert report.destination == ""
    assert report.applied is False


def test_omission_is_never_applied():
    """The harness must not compose résumé text and then grade its own work."""
    draft = by_id("omitted_record").draft_before_trim
    issues = [{
        "passage": dc.OMITTED_MARKER,
        "problem": dc.PROBLEM_OMITTED,
        "sources": ["candidate-supplement.txt"],
        "correction": "**Ridgeline Equipment Services** -- Fife, WA",
    }]
    revised, log = apply_revisions_v2(draft, issues, SOURCES)
    assert revised == draft, "an omission must not change the draft"
    assert log and log[0]["kind"] == "omission"
    assert log[0]["applied"] is False


def test_omission_with_ambiguous_anchor_is_flagged_not_guessed():
    """If the anchor occurs twice, the harness cannot know where it goes."""
    draft = "- same line\n- same line\n"
    issue = {
        "passage": "- same line",
        "problem": dc.PROBLEM_OMITTED,
        "sources": ["candidate-main.txt"],
        "correction": "Reconciled 1,180 fuel invoices",
    }
    report = describe_omission(issue, draft, SOURCES)
    assert report.ambiguous is True
    assert report.anchor_occurrences == 2
    assert report.applied is False
    assert "ambiguous" in report.note


def test_omission_citing_text_in_no_source_is_not_grounded():
    issue = {
        "passage": dc.OMITTED_MARKER,
        "problem": dc.PROBLEM_OMITTED,
        "sources": ["candidate-main.txt"],
        "correction": "Invented an award that no document mentions",
    }
    report = describe_omission(issue, BASELINE, SOURCES)
    assert report.cites_real_source is False
    assert "does not contain" in report.note


def test_duplicate_passage_correction_is_refused():
    """A passage occurring twice cannot be corrected unambiguously."""
    draft = "- bullet\n- bullet\n"
    issues = [{"passage": "- bullet", "problem": dc.PROBLEM_UNSUPPORTED,
               "sources": [], "correction": "- fixed"}]
    revised, log = apply_revisions_v2(draft, issues, {})
    assert revised == draft
    assert log[0]["kind"] == "refused"
    assert "2 times" in log[0]["reason"]


def test_blank_correction_still_refused():
    """v1 behaviour kept: applying it deletes a sourced fact and then approves."""
    issues = [{"passage": "- Cut rework by 18% by standardizing panel labeling.",
               "problem": dc.PROBLEM_UNSUPPORTED, "sources": [], "correction": "  "}]
    revised, log = apply_revisions_v2(BASELINE, issues, SOURCES)
    assert revised == BASELINE
    assert log[0]["kind"] == "refused"


def test_unambiguous_correction_is_applied_once():
    issues = [{
        "passage": "- Cut rework by 18% by standardizing panel labeling across the terminal fleet.",
        "problem": dc.PROBLEM_UNSUPPORTED, "sources": [],
        "correction": "- Cut rework by 18% by standardizing panel labeling."}]
    revised, log = apply_revisions_v2(BASELINE, issues, SOURCES)
    assert revised != BASELINE
    assert log[0]["kind"] == "replaced"
    assert "quality audits" not in revised


# ══ 5. Budget: a ceiling enforced BEFORE the call ══════════════════════════
def test_budget_refuses_when_calls_exhausted():
    budget = RunBudget(max_calls=2, max_cost_usd=100.0)
    budget.record(0.0)
    budget.record(0.0)
    allowed, why = budget.can_call()
    assert allowed is False
    assert "call budget" in why


def test_budget_refuses_when_cost_exhausted():
    budget = RunBudget(max_calls=100, max_cost_usd=0.50)
    budget.record(0.60)
    allowed, why = budget.can_call()
    assert allowed is False
    assert "cost budget" in why


def test_budget_refusal_is_operational_not_semantic():
    attempt = AttemptV2(case_id="x", cell="revised+enabled", mode="imaginative")
    attempt.outcome = OUTCOME_OPERATIONAL
    attempt.outcome_reason = "budget_refused:call budget exhausted"
    attempt.complete = False
    row = score_v2(attempt, by_id("faithful_paraphrase"))
    assert row["false_refusal"] is False
    assert row["scored"] is False


# ══ 6. Scoring: incomplete attempts are never judged ═══════════════════════
def test_truncated_attempt_is_not_a_false_refusal():
    """Item 5: a truncated legacy answer is not a judgment-quality result."""
    attempt = AttemptV2(case_id="faithful_paraphrase", cell="current+advisory",
                        mode="imaginative")
    attempt.outcome = OUTCOME_OPERATIONAL
    attempt.outcome_reason = "truncation"
    attempt.complete = False
    row = score_v2(attempt, by_id("faithful_paraphrase"))
    assert row["false_refusal"] is False
    assert row["semantic_ok"] is None
    assert row["scored"] is False


def test_complete_refusal_of_an_approve_case_is_still_a_false_refusal():
    """The guard must not be so permissive that real refusals escape."""
    attempt = AttemptV2(case_id="faithful_paraphrase", cell="revised+enabled",
                        mode="imaginative")
    attempt.outcome = OUTCOME_REFUSED
    attempt.outcome_reason = "cannot_assess"
    row = score_v2(attempt, by_id("faithful_paraphrase"))
    assert row["false_refusal"] is True
    assert row["scored"] is True


def test_revise_on_a_revise_case_is_correct_behaviour_not_failure():
    attempt = AttemptV2(case_id="unsupported_method", cell="revised+advisory",
                        mode="imaginative")
    attempt.outcome = OUTCOME_APPROVED  # advisory records and delivers
    attempt.outcome_reason = "advisory_revise_noted"
    row = score_v2(attempt, by_id("unsupported_method"))
    assert row["false_approval"] is True


# ══ 6b. A reference failure is neither a refusal nor a judgement ═══════════
def test_reference_failure_is_not_a_false_refusal():
    """Item 3. v1 filed exactly this as a semantic refusal.

    Measured on the retained run: the one legacy answer that completed was
    recorded `refused/reference_invalid` while all 10 of its quotes were real
    source text. The reviewer's evidence failed to verify; its judgement about the
    document was never established either way.
    """
    attempt = AttemptV2(case_id="faithful_paraphrase", cell="current+advisory",
                        mode="imaginative")
    attempt.outcome = OUTCOME_REFERENCE_FAILURE
    attempt.outcome_reason = "reference_cites_draft"
    row = score_v2(attempt, by_id("faithful_paraphrase"))
    assert row["false_refusal"] is False
    assert row["scored"] is False
    assert row["reference_failure"] is True
    assert row["semantic_ok"] is None


def test_reference_failure_is_counted_in_its_own_column():
    attempt = AttemptV2(case_id="faithful_paraphrase", cell="current+advisory",
                        mode="imaginative")
    attempt.outcome = OUTCOME_REFERENCE_FAILURE
    attempt.outcome_reason = "reference_fabricated"
    rows = [score_v2(attempt, by_id("faithful_paraphrase"))]
    from experiment.runner_v2 import summarise_v2
    summary = summarise_v2(rows)["current+advisory"]
    assert summary["reference_failures"] == 1
    assert summary["false_refusals"] == 0
    assert summary["correct_approvals"] == 0
    assert summary["scored"] == 0


def test_reference_failure_outcome_is_distinct_from_refusal():
    from experiment import runner_v2
    assert runner_v2.OUTCOME_REFERENCE_FAILURE != runner_v2.OUTCOME_REFUSED
    assert runner_v2.OUTCOME_REFERENCE_FAILURE != runner_v2.OUTCOME_OPERATIONAL


# ══ 7. Provider contract ═══════════════════════════════════════════════════
def test_default_output_budget_is_sufficient():
    """1200 tokens could not hold a span envelope. That truncated 11 of 12."""
    from experiment import provider_v2
    assert provider_v2.DEFAULT_MAX_TOKENS >= 4096


def test_provider_reports_truncation_as_an_error_class():
    from experiment import provider_v2
    assert issubclass(provider_v2.TruncatedResponse, RuntimeError)
    result = provider_v2.CallResult(text="x", finish_reason="length",
                                    truncated=True)
    assert result.to_dict()["truncated"] is True
    assert result.to_dict()["finish_reason"] == "length"


def test_call_result_retains_usage_and_cost():
    from experiment import provider_v2
    result = provider_v2.CallResult(
        text="t", finish_reason="stop", prompt_tokens=900, completion_tokens=120,
        total_tokens=1020, cost_usd=0.0012, latency_s=3.4)
    payload = result.to_dict()
    for key in ("finish_reason", "prompt_tokens", "completion_tokens",
                "total_tokens", "cost_usd", "latency_s", "max_tokens"):
        assert key in payload


# ══ 8. Contract behaviour preserved (v2 must not weaken a guard) ═══════════
def test_invented_passage_still_fails():
    answer = ("Decision: revise\n\nIssues:\n"
              "- passage: text that appears nowhere in the draft\n"
              "  problem: unsupported\n"
              "  sources: [candidate-main.txt]\n"
              "  why: invented\n"
              "  correction: something\n")
    parsed = dc.decide_from_text(answer, BASELINE, SOURCES)
    assert any(p.startswith("passage_not_in_draft")
               for p in parsed.structural_problems)


def test_unknown_problem_word_still_fails():
    answer = ("Decision: revise\n\nIssues:\n"
              "- passage: Operations analyst with nine years\n"
              "  problem: vaguely_wrong\n"
              "  sources: [candidate-main.txt]\n"
              "  why: x\n"
              "  correction: y\n")
    parsed = dc.decide_from_text(answer, BASELINE, SOURCES)
    assert any(p.startswith("problem_word_unknown")
               for p in parsed.structural_problems)


def test_faithful_approval_of_the_complete_baseline_parses():
    """The known-good control: a correct approval must not be rejected.

    Without this, every other test could pass on a contract that refuses
    everything.
    """
    answer = "Decision: approve\n\nIssues:\n- none\n"
    parsed = dc.decide_from_text(answer, BASELINE, SOURCES)
    assert parsed.decision == dc.DECISION_APPROVE
    assert parsed.structural_problems == []


# ══ 9. Markup-tolerant passage location ═══════════════════════════════════
def test_passage_without_markup_locates_the_real_draft_text():
    """Measured on the v2 probe: 3 of 4 attempts rejected for this.

    The model restates a bolded record without its emphasis. The passage is
    correct and its evidence is real; only the formatting differs.

    The locator returns the RAW draft span, not a cleaned-up version, because the
    returned string has to be replaceable in the document. A tidied span would
    look right and then fail to apply.
    """
    passage = ("Ridgeline Equipment Services -- Fife, WA "
               "Inventory Analyst | 08/2023 - Present")
    located = dc.locate_passage(passage, BASELINE)
    assert located is not None
    assert "Ridgeline Equipment Services" in located
    assert "Inventory Analyst" in located
    # The contract that matters: it is literally present, so an edit can land.
    assert located in BASELINE


def test_locator_rejects_an_invented_passage():
    """Stripping emphasis must not turn into accepting anything."""
    assert dc.locate_passage("A sentence no document contains", BASELINE) is None


def test_locator_prefers_an_exact_match():
    bullet = ("- Reconciled 1,180 fuel invoices against depot dispensing "
              "records every quarter.")
    assert dc.locate_passage(bullet, BASELINE) == bullet


def test_markup_variant_passage_is_actionable_and_edits_real_text():
    """A correction must target draft text, not the reviewer's re-typed version."""
    issue = {
        "passage": ("Ridgeline Equipment Services -- Fife, WA "
                    "Inventory Analyst | 08/2023 - Present"),
        "problem": dc.PROBLEM_UNSUPPORTED,
        "sources": [],
        "correction": "Ridgeline Equipment Services -- Fife, WA",
    }
    revised, log = apply_revisions_v2(BASELINE, [issue], SOURCES)
    assert revised != BASELINE
    assert log[0]["kind"] == "replaced"
    # The edit landed on the Ridgeline header, and the sourced bullets under it
    # survive untouched.
    assert "Maintained parts availability records" in revised
    assert "Compared quarterly parts usage" in revised


def test_markup_variant_does_not_make_an_invented_issue_actionable():
    answer = ("Decision: revise\n\nIssues:\n"
              "- passage: A sentence no document contains at all\n"
              "  problem: unsupported\n"
              "  sources: [candidate-main.txt]\n"
              "  why: invented\n"
              "  correction: something\n")
    parsed = dc.decide_from_text(answer, BASELINE, SOURCES)
    assert parsed.structural_problems
    assert any("passage_not_in_draft" in p
               for p in parsed.structural_problems)
