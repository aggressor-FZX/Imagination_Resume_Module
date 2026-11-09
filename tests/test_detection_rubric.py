"""The five-criterion detection rubric, and the parser defects it exposed.

Two things are tested here:

1. The rubric grades the five criteria the brief names, and does NOT treat
   "the needle appears somewhere" as accuracy. The decisive test is a case where
   the needle is present but the proposition is wrong — a reviewer that says a
   source-backed fact is unsourced has named the needle and failed the rubric.

2. Two real parser defects found while building it. Both turned a CORRECT answer
   into a rejected one, which is the error class that most flatters a reviewer's
   apparent accuracy:
     * `**problem:** value` — bold field names parsed as zero issues;
     * a passage with trailing emphasis (`... each quarter.*`) failing an exact
       source match, so a legitimate omission quote became a "miscitation".
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import decision_contract as dc
from experiment.detection_rubric import (
    ABSENT_FROM_DRAFT, ABSENT_FROM_SOURCE, TRUTH, _contains_phrase,
    assert_truth_matches_fixtures, grade_case, grade_issue,
)
from experiment.calibration_cases import by_id as cal_by_id
from experiment.cases_v2 import by_id as fam_by_id, SOURCES as FAM_SOURCES

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/calibration-v1/calibration.json"


# ── The central methodological point ─────────────────────────────────────────

def test_naming_the_needle_is_not_sufficient():
    """A wrong proposition that contains the needle must FAIL.

    This is the brief's point stated as an executable test: the needle locates a
    passage, it does not establish that the reviewer got the fact situation right.
    """
    case = fam_by_id("missing_fact")
    # The 1,180 bullet was DELETED from the draft and IS in the source.
    # This issue claims the opposite on both counts.
    raw = ("Decision: revise\n\nIssues:\n"
           "- passage: Reconciled 1,180 fuel invoices against depot dispensing "
           "records each quarter.\n"
           "  problem: unsupported\n"
           "  sources: [candidate-main.txt]\n"
           "  why: this is not present in the source and it appears in the draft "
           "already.\n"
           "  correction: remove it.\n")
    decision = dc.decide_from_text(raw, case.draft_before_trim, FAM_SOURCES)
    graded = grade_case("missing_fact", decision)

    assert graded["best"]["fact"] is True, "the needle IS present"
    assert graded["all_complete"] is False, "and it must still fail"


def test_the_five_criteria_are_graded_independently():
    case = fam_by_id("wrong_record_attribution")
    raw = ("Decision: revise\n\nIssues:\n"
           "- passage: Prepared the monthly cost summary that the regional manager "
           "reviewed and reconciled 1,180 fuel invoices each quarter.\n"
           "  problem: misattributed\n"
           "  sources: [candidate-main.txt]\n"
           "  why: the 1,180 figure belongs to Harborline Transit Authority, not "
           "Cascade Freight Partners.\n"
           "  correction: Prepared the monthly cost summary reviewed by the "
           "regional manager.\n")
    decision = dc.decide_from_text(raw, case.draft_before_trim, FAM_SOURCES)
    best = grade_case("wrong_record_attribution", decision)["best"]
    assert best["fact"] and best["in_source"] and best["in_draft"]
    assert best["owner"] is True
    assert best["correction"] is True


def test_ownership_is_not_demanded_for_a_local_clause_removal():
    """Naming the employer is not required when the defect is a local insertion.

    Requiring it would fail a correct answer for brevity; the rubric therefore
    declares ownership not-applicable rather than failed.
    """
    assert TRUTH["unsupported_method"]["owner_required"] is False
    case = fam_by_id("unsupported_method")
    raw = ("Decision: revise\n\nIssues:\n"
           "- passage: Cut rework by 18% through standardizing panel labeling and "
           "targeted quality audits across the terminal fleet.\n"
           "  problem: unsupported\n"
           "  sources: [candidate-main.txt]\n"
           "  why: 'targeted quality audits' is not mentioned in the source.\n"
           "  correction: Cut rework by 18% by standardizing panel labeling.\n")
    decision = dc.decide_from_text(raw, case.draft_before_trim, FAM_SOURCES)
    graded = grade_case("unsupported_method", decision)
    assert graded["best"]["owner"] is None
    assert graded["all_complete"] is True


def test_absence_phrases_are_scoped_to_the_source_or_the_draft():
    """"not in the source" answers criterion 2, not criterion 3."""
    assert _contains_phrase("this is not mentioned in the source", "not mentioned")
    for phrase in ABSENT_FROM_SOURCE:
        assert phrase not in ABSENT_FROM_DRAFT or phrase in (
            "not in the draft", "omitted from the draft")
    # And a source-scoped phrase must not count as a draft-absence claim.
    assert not any(_contains_phrase("not mentioned in the source", p)
                   for p in ABSENT_FROM_DRAFT)


def test_phrase_matching_does_not_fire_inside_longer_words():
    """"invent" must not match "inventory" — it made a correct claim look wrong."""
    assert _contains_phrase("the inventory record", "invent") is False
    assert _contains_phrase("they invent a method", "invent") is True
    assert _contains_phrase("1,180 invoices", "1,180") is True
    assert _contains_phrase("11,801 invoices", "1,180") is False


def test_a_restoration_must_not_be_a_deletion():
    """`(omit)` deletes; restoring supplies. They are opposites.

    The parser maps a bare `(omit)` correction to `None` — "no replacement text" —
    so the rubric reports the correction as UNDETERMINED, not failed. Either way it
    must never count as a correct restoration, and this test pins that: `(omit)`
    is not credited, and a real replacement is.
    """
    case = fam_by_id("missing_fact")
    omit = ("Decision: revise\n\nIssues:\n"
            "- passage: Reconciled 1,180 fuel invoices against depot dispensing "
            "records each quarter.\n"
            "  problem: omitted\n"
            "  sources: [candidate-main.txt]\n"
            "  why: the 1,180 bullet is missing from the draft.\n"
            "  correction: (omit)\n")
    best = grade_case(
        "missing_fact",
        dc.decide_from_text(omit, case.draft_before_trim, FAM_SOURCES))["best"]
    assert best["fact"] is True and best["in_draft"] is True
    assert best["correction"] is not True, "a deletion must never be credited"

    restore = omit.replace(
        "  correction: (omit)\n",
        "  correction: Reconciled 1,180 fuel invoices against depot dispensing "
        "records each quarter.\n")
    best = grade_case(
        "missing_fact",
        dc.decide_from_text(restore, case.draft_before_trim, FAM_SOURCES))["best"]
    assert best["correction"] is True


# ── Fixture integrity ────────────────────────────────────────────────────────

def test_the_truth_table_matches_the_fixtures():
    """A fixture edit that moves a fact invalidates every grade."""
    assert_truth_matches_fixtures()


# ── Parser defect 1: bold field names ────────────────────────────────────────

def test_bold_field_names_are_parsed():
    """`**problem:** value` used to parse as ZERO issues."""
    draft = ("## Professional Experience\n**Harborline Transit Authority** -- Tacoma, WA\n"
             "- Reconciled 1,180 fuel invoices against depot dispensing records "
             "each quarter.\n")
    raw = ("Decision: **revise**\n\nIssues:\n"
           "- **passage:** *Reconciled 1,180 fuel invoices against depot dispensing "
           "records each quarter.*\n"
           "  **problem:** **unsupported**\n"
           "  **sources:** [candidate-main.txt]\n"
           "  **why:** the figure is in the source but the method is not.\n"
           "  **correction:** Reconciled 1,180 fuel invoices each quarter.\n")
    decision = dc.decide_from_text(raw, draft, {"candidate-main.txt": draft})
    assert decision.decision == "revise"
    assert len(decision.issues) == 1, decision.structural_problems
    assert decision.issues[0].problem == "unsupported"
    assert "Reconciled 1,180" in decision.issues[0].passage


def test_bold_field_names_do_not_break_plain_fields():
    """The widening must be a widening only."""
    draft = "- Reconciled 1,180 fuel invoices each quarter.\n"
    raw = ("Decision: revise\n\nIssues:\n"
           "- passage: Reconciled 1,180 fuel invoices each quarter.\n"
           "  problem: unsupported\n  sources: [candidate-main.txt]\n"
           "  why: x\n  correction: Reconciled 1,180 invoices.\n")
    decision = dc.decide_from_text(raw, draft, {"candidate-main.txt": draft})
    assert len(decision.issues) == 1
    assert decision.issues[0].problem == "unsupported"


# ── Parser defect 2: trailing emphasis on a passage value ────────────────────

def test_trailing_emphasis_is_stripped_from_a_passage():
    draft = "- Reconciled 1,180 fuel invoices each quarter.\n"
    raw = ("Decision: revise\n\nIssues:\n"
           "- passage: Reconciled 1,180 fuel invoices each quarter.*\n"
           "  problem: unsupported\n  sources: [candidate-main.txt]\n"
           "  why: x\n  correction: Reconciled invoices each quarter.\n")
    decision = dc.decide_from_text(raw, draft, {"candidate-main.txt": draft})
    assert decision.issues[0].passage == "Reconciled 1,180 fuel invoices each quarter."
    assert decision.issues[0].passage_resolves is True


def test_a_legitimate_omission_quote_survives_trailing_emphasis():
    """The specific regression: a real source quote must stay accepted.

    Before the fix the stray `*` defeated the exact source match, the `omitted`
    issue lost its quote acceptance, and a correct detection was recorded as a
    miscitation.
    """
    case = cal_by_id("cal_lab_missing")
    # This exact text is in the source and was deleted from the draft.
    quoted = ("Reported results to the state drinking-water program within the "
              "24-hour regulatory window.")
    raw = ("Decision: revise\n\nIssues:\n"
           f"- passage: {quoted}*\n"
           "  problem: omitted\n  sources: [lab-main.txt]\n"
           "  why: the draft omits the 24-hour reporting obligation.\n"
           "  correction: add the 24-hour reporting bullet back.\n")
    decision = dc.decide_from_text(raw, case.draft_before_trim, case.sources)
    issue = decision.issues[0]
    assert issue.reports_omission is True
    assert issue.passage_resolves is True
    assert not any(p.startswith("passage_not_in_draft")
                   for p in decision.structural_problems)


def test_stripping_emphasis_does_not_invent_a_match():
    """Removing decoration must not make absent text resolve."""
    draft = "- Some other bullet entirely.\n"
    raw = ("Decision: revise\n\nIssues:\n"
           "- passage: A sentence never written in this document.*\n"
           "  problem: unsupported\n  sources: [candidate-main.txt]\n"
           "  why: x\n  correction: y\n")
    decision = dc.decide_from_text(raw, draft, {"candidate-main.txt": draft})
    assert decision.issues[0].passage_resolves is False


def test_a_fully_bolded_answer_parses_clean_with_no_dangling_markers():
    """The asymmetry defect: one side of an emphasis pair consumed, the other left.

    Accepting `**problem:**` in the field regex consumed the leading marker but
    left the trailing one, so a passage came out as `text.*` and a correction as
    `text.**`. `_strip_markup_wrapping` only unwraps a BALANCED pair, so it could
    not help. That is why correction values are now cleaned with a balanced-first
    pass and the field regex captures the value verbatim.
    """
    draft = "- Reconciled 1,180 fuel invoices each quarter.\n"
    bold = ("Decision: **revise**\n\nIssues:\n"
            "- **passage:** *Reconciled 1,180 fuel invoices each quarter.*\n"
            "  **problem:** **unsupported**\n"
            "  **sources:** [candidate-main.txt]\n"
            "  **why:** x\n"
            "  **correction:** **Reconciled 1,180 invoices.**\n")
    plain = ("Decision: revise\n\nIssues:\n"
             "- passage: Reconciled 1,180 fuel invoices each quarter.\n"
             "  problem: unsupported\n  sources: [candidate-main.txt]\n"
             "  why: x\n  correction: Reconciled 1,180 invoices.\n")
    b = dc.decide_from_text(bold, draft, {"candidate-main.txt": draft}).issues[0]
    p = dc.decide_from_text(plain, draft, {"candidate-main.txt": draft}).issues[0]
    # Emphasis must make NO difference to the parsed values.
    assert b.passage == p.passage == "Reconciled 1,180 fuel invoices each quarter."
    assert b.problem == p.problem == "unsupported"
    assert b.correction == p.correction == "Reconciled 1,180 invoices."
    assert "*" not in b.passage and "*" not in b.correction


def test_interior_emphasis_in_a_field_value_is_preserved():
    """Only decoration on the edges is removed; content is not rewritten."""
    draft = "- Reconciled 1,180 fuel invoices each quarter.\n"
    raw = ("Decision: revise\n\nIssues:\n"
           "- passage: Reconciled 1,180 fuel invoices each quarter.\n"
           "  problem: unsupported\n  sources: [candidate-main.txt]\n"
           "  why: the *only* issue is the method\n"
           "  correction: Reconciled 1,180 invoices.\n")
    issue = dc.decide_from_text(raw, draft,
                                {"candidate-main.txt": draft}).issues[0]
    assert issue.why == "the *only* issue is the method"


# ── The run, graded ──────────────────────────────────────────────────────────

@pytest.mark.skipif(not RUN.is_file(), reason="calibration artifact absent")
def test_the_run_grades_ten_of_twelve_rubric_agreements():
    """Locked number: 10 of 12 defective cases agree with all five criteria.

    This is an AUTOMATED-RUBRIC AGREEMENT count, not verified accuracy. The test
    name and the rendered report both say so, because a bare "10/12" reads as an
    accuracy figure and it is not one — the rubric can pass a finding whose
    correction is unsafe (see MANUAL_INSPECTION.md).
    """
    from experiment.detection_rubric import grade_rows
    run = json.loads(RUN.read_text())
    summary = grade_rows(run["rows"])
    assert summary["all"]["cases"] == 12
    assert summary["all"]["complete_detection"] == 10
    assert summary["all"]["fact_identified"] == 12
    # The two failures are named, so the prose can be checked against them.
    incomplete = {r["case_id"] for r in summary["cases"] if not r["all_complete"]}
    assert incomplete == {"cal_lab_missing", "cal_ops_missing"}


def test_the_report_labels_its_numbers_as_rubric_agreement():
    """A bare '10/12' must never be published without the caveat.

    This guards the exact correction requested: the count is agreement with a
    hand-written table, not independently verified detection accuracy.
    """
    from experiment.detection_rubric import grade_rows, render
    payload = {"all": {"cases": 0, "complete_detection": 0, "fact_identified": 0,
                       "no_fact_identified": 0, "criteria": {}, "undetermined": {}},
               "cases": []}
    text = render(payload)
    flat = " ".join(text.lower().split())
    assert "automated-rubric agreement" in flat
    assert "not independently verified detection accuracy" in flat
    # The old, misleading wording must not come back.
    assert "complete detection (all five)" not in flat
