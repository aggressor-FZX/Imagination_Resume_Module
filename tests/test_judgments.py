"""Tests for the source-linked judgment classifier.

The classifier decides whether an objection NAMED the defect that actually exists.
That is the claim the packet makes, so it needs its own tests — and the tests that
matter most are the ones proving a failure cannot be scored as a success.

The dangerous case is specific. When a defect is that text is MISSING from the
draft, a reviewer that quotes that text as though it were present has written the
needle. A naive "does the objection contain the needle?" check would score that
exactly backwards. `test_quoting_absent_text_is_not_a_detection` pins the ordering
that prevents it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import decision_contract as dc
from experiment.judgments import (
    APPROVE_CASES, DETECTION, assert_needles, build, classify,
)
from experiment.cases_v2 import SOURCES, by_id

ROOT = Path(__file__).resolve().parents[1]
RETAINED = ROOT / "results/comparison-v2b-20261109/attempts.json"


def _decision(passage: str, why: str, problem: str = "unsupported",
              correction: str | None = None) -> dc.Decision:
    """A decision whose passage really is in the draft, so it resolves."""
    draft = by_id("faithful_paraphrase").draft_before_trim
    issue = dc.Issue(passage, problem, why, sources=["candidate-main.txt"],
                     correction=correction)
    return dc.decide_from_text(
        f"Decision: revise\n\nIssues:\n- passage: {passage}\n"
        f"  problem: {problem}\n  sources: [candidate-main.txt]\n"
        f"  why: {why}\n" + (f"  correction: {correction}\n" if correction else ""),
        draft, SOURCES)


# ── The inversion hazard ──────────────────────────────────────────────────────

def test_quoting_absent_text_is_not_a_detection():
    """For a deletion defect, citing the deleted text as PRESENT is the opposite.

    Built as a raw decision so the passage deliberately does NOT resolve — which is
    what a reviewer asserting an absent passage has actually produced.
    """
    draft = by_id("omitted_record").draft_before_trim
    raw = ("Decision: revise\n\nIssues:\n"
           "- passage: **Ridgeline Equipment Services** -- Fife, WA\n"
           "  problem: misattributed\n"
           "  sources: [candidate-supplement.txt]\n"
           "  why: this record conflicts with the other dates.\n"
           "  correction: (omit)\n")
    decision = dc.decide_from_text(raw, draft, SOURCES)
    assert not decision.issues[0].passage_resolves, "fixture assumption"
    verdict = classify("omitted_record", decision)
    assert verdict["verdict"] == "asserted_the_opposite"


def test_a_grounded_omission_claim_is_a_detection():
    """`problem: omitted` + `(omitted)` + the missing text named in `why`."""
    draft = by_id("omitted_record").draft_before_trim
    raw = ("Decision: revise\n\nIssues:\n"
           "- passage: (omitted)\n"
           "  problem: omitted\n"
           "  sources: [candidate-supplement.txt]\n"
           "  why: the draft omits the Ridgeline Equipment Services record that "
           "candidate-supplement.txt supplies.\n"
           "  correction: (omit)\n")
    decision = dc.decide_from_text(raw, draft, SOURCES)
    verdict = classify("omitted_record", decision)
    assert verdict["verdict"] == "detected"


def test_a_miscitation_alongside_a_real_catch_is_reported_as_both():
    """Neither crediting the miscitation nor discarding the catch is honest."""
    draft = by_id("omitted_record").draft_before_trim
    raw = ("Decision: revise\n\nIssues:\n"
           "- passage: **Ridgeline Equipment Services** -- Fife, WA\n"
           "  problem: misattributed\n"
           "  sources: [candidate-supplement.txt]\n"
           "  why: concurrent employment is impossible.\n"
           "  correction: (omit)\n"
           "- passage: (omitted)\n"
           "  problem: omitted\n"
           "  sources: [candidate-supplement.txt]\n"
           "  why: the Ridgeline Equipment Services record is missing from the "
           "draft.\n"
           "  correction: (omit)\n")
    decision = dc.decide_from_text(raw, draft, SOURCES)
    assert classify("omitted_record", decision)["verdict"] == \
        "detected_with_a_miscitation"


def test_a_missing_defect_that_is_never_mentioned_is_not_detected():
    decision = _decision(
        "Operations analyst with nine years of experience in depot reporting",
        "the summary should name the transit authority explicitly.")
    verdict = classify("missing_fact", decision)
    assert verdict["verdict"] == "not_detected"
    assert "the defect exists" in verdict["basis"]


# ── Approve cases: any issue is invented ─────────────────────────────────────

def test_an_approve_case_with_issues_is_an_invented_objection():
    decision = dc.decide_from_text(
        "Decision: revise\n\nIssues:\n"
        "- passage: Operations analyst with nine years of experience in depot "
        "reporting\n"
        "  problem: unsupported\n  sources: [candidate-main.txt]\n"
        "  why: the year count should be written differently.\n"
        "  correction: Operations analyst with 9 years of experience.\n",
        by_id("faithful_paraphrase").draft_before_trim, SOURCES)
    verdict = classify("faithful_paraphrase", decision)
    assert verdict["verdict"] == "invented_objection"


def test_an_actual_approval_of_an_approve_case_is_correct():
    decision = dc.decide_from_text("Decision: approve\n\nIssues:\n- none\n",
                                   by_id("faithful_paraphrase").draft_before_trim,
                                   SOURCES)
    assert classify("faithful_paraphrase", decision)["verdict"] == "correct_approval"


def test_the_approve_cases_are_exactly_the_ones_with_no_declared_defect():
    from experiment.cases_v2 import MUTATIONS
    for case_id in APPROVE_CASES:
        assert MUTATIONS[case_id]["kind"] in {"reword_only", "reword_sourced_method",
                                              "add_capability_clause"}
    for case_id in DETECTION:
        assert MUTATIONS[case_id]["kind"] not in {"reword_only",
                                                  "reword_sourced_method"}


# ── Needle integrity ────────────────────────────────────────────────────────

def test_needles_sit_where_the_classifier_assumes():
    """A fixture change that moves a needle invalidates every verdict."""
    assert_needles()


def test_the_retained_run_reproduces_the_reported_verdicts():
    payload = build()
    counts = payload["summary"]["verdict_counts"]
    assert payload["summary"]["prose_cells"] == 14
    # Every prose cell gets exactly one verdict; nothing is silently dropped.
    assert sum(counts.values()) == 14
    # The two arms agree on the four defect cases except where they genuinely
    # differ, and `missing_fact` is the documented difference.
    by_cell = payload["summary"]["by_cell"]
    assert by_cell["revised+advisory"]["missing_fact"] == "not_detected"
    assert by_cell["revised+enabled"]["missing_fact"] == "detected"
    # The approve cases are refused on invented grounds in BOTH arms.
    for case_id in APPROVE_CASES:
        assert by_cell["revised+advisory"][case_id] == "invented_objection"
        assert by_cell["revised+enabled"][case_id] == "invented_objection"


def test_a_detection_does_not_imply_a_safe_correction():
    """The axis boundary: naming a defect is not the same as fixing it."""
    payload = build()
    assert "does not mean the proposed correction was safe to apply" in \
        payload["summary"]["caveat"]
