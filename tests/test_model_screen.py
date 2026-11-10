"""The screen's four outcomes must not be inferable from a decision word.

The brief is explicit: do not infer correctness from keywords or decision words
alone. These tests pin that behaviour using answers ALREADY RECORDED, so the
classifier is validated before a single paid call is made.

Each test names a real failure observed in the earlier runs and asserts the screen
classifies it correctly — as an invented objection, a miss, or an invalid answer —
rather than as a correct judgment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import decision_contract as dc
from experiment.calibration_cases import by_id as cal_by_id
from experiment.cases_v2 import by_id as fam_by_id, SOURCES as FAM_SOURCES
from experiment.detection_rubric import grade_case
from experiment.model_screen import (
    CAST, MODELS, MODEL_SUBSTITUTION, budget_for, classify,
)

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "results/calibration-v1/calibration.json"
GPT = ROOT / "results/cross-model-v1/cross-model-openai_gpt-4o-mini.json"


def _case(case_id: str):
    try:
        return cal_by_id(case_id)
    except KeyError:
        return fam_by_id(case_id)


def _decision(case_id: str, raw: str) -> dc.Decision:
    c = _case(case_id)
    return dc.decide_from_text(raw, c.draft_before_trim, c.sources)


def _recorded_decision(source: Path, case_id: str) -> dc.Decision:
    data = json.loads(source.read_text())
    row = next(r for r in data["rows"] if r["case_id"] == case_id)
    raw = row.get("first_response") or row.get("response") or ""
    return _decision(case_id, raw)


# ── The case list and the model substitution ────────────────────────────────

def test_the_five_cases_and_their_expectations():
    assert [c for c, _g, _f, _e in CAST] == [
        "faithful_paraphrase", "cal_ops_faithful", "unsupported_method",
        "wrong_record_attribution", "omitted_record"]
    assert [e for _c, _g, _f, e in CAST].count("approve") == 2
    assert [e for _c, _g, _f, e in CAST].count("revise") == 3


def test_the_requested_mimo_slug_is_substituted_openly():
    """The exact slug requested does not exist; the change is declared, not silent."""
    assert MODEL_SUBSTITUTION["requested"] == "xiaomi/mimo-v2-flash"
    assert MODEL_SUBSTITUTION["used"] in MODELS
    assert "not present" in MODEL_SUBSTITUTION["reason"]
    # The requested slug must NOT appear as an executable model.
    assert "xiaomi/mimo-v2-flash" not in MODELS


def test_the_budget_is_bounded_and_matches_four_models():
    plan = budget_for(MODELS)
    n = len(MODELS)
    assert plan["logical_calls"] == 5 * n
    assert plan["max_requests"] == plan["logical_calls"] * plan["attempts_per_call"]
    assert plan["max_spend_usd"] == 0.10
    assert "HARD" in plan["ceiling_statement"]
    assert "not a proof of maximum spend" in plan["ceiling_statement"]


# ── Outcome 1: validity ─────────────────────────────────────────────────────

def test_approve_with_issues_is_not_a_valid_judgment():
    """gpt-4o-mini's `omitted_record` answer: decision word `approve`, issues attached.

    It must be reported invalid. It must NOT be reported as a correct approval, and
    it must NOT be reported as a demonstrated release.
    """
    d = _recorded_decision(GPT, "omitted_record")
    assert d.decision == "approve"
    assert d.structural_problems, "fixture assumption: this answer is invalid"
    out = classify("omitted_record", "revise", d)
    assert out["valid_judgment"] is False
    assert out["actual_detection"] is False
    assert out["missed_defect"] is True


def test_a_refusal_word_alone_earns_no_detection_credit():
    """`revise` with nothing material is a miss, not a detection."""
    raw = ("Decision: revise\n\nIssues:\n"
           "- passage: Operations analyst with nine years of experience in depot "
           "reporting\n"
           "  problem: unsupported\n  sources: [candidate-main.txt]\n"
           "  why: the summary should read differently.\n"
           "  correction: Operations analyst with 9 years of experience.\n")
    out = classify("omitted_record", "revise", _decision("omitted_record", raw))
    assert out["decision"] == "revise"
    assert out["actual_detection"] is False, "a refusal word is not a detection"
    assert out["missed_defect"] is True


def test_a_needle_without_the_full_finding_earns_no_credit():
    """Mentioning the fact is not detection; the five criteria decide."""
    c = fam_by_id("omitted_record")
    raw = ("Decision: revise\n\nIssues:\n"
           "- passage: **Ridgeline Equipment Services** -- Fife, WA\n"
           "  problem: misattributed\n  sources: [candidate-supplement.txt]\n"
           "  why: the Ridgeline dates conflict with Harborline.\n"
           "  correction: (omit)\n")
    d = dc.decide_from_text(raw, c.draft_before_trim, c.sources)
    out = classify("omitted_record", "revise", d)
    assert "Ridgeline" in (d.issues[0].passage or "")
    assert out["actual_detection"] is False, \
        "naming the needle while asserting it is present is not detection"


# ── Outcome 2: detection, validated on real recorded answers ────────────────

@pytest.mark.skipif(not CAL.is_file(), reason="calibration artifact absent")
def test_recorded_correct_detection_is_credited():
    d = _recorded_decision(CAL, "wrong_record_attribution")
    out = classify("wrong_record_attribution", "revise", d)
    assert out["actual_detection"] is True
    assert out["missed_defect"] is False


@pytest.mark.skipif(not CAL.is_file(), reason="calibration artifact absent")
def test_recorded_correct_approval_is_credited():
    """qwen approved cal_ops_faithful. That is a valid judgment and not an invention."""
    d = _recorded_decision(CAL, "cal_ops_faithful")
    out = classify("cal_ops_faithful", "approve", d)
    assert out["valid_judgment"] is True
    assert out["invented_objection"] is False


# ── Outcome 3: invented objections ──────────────────────────────────────────

@pytest.mark.skipif(not CAL.is_file(), reason="calibration artifact absent")
def test_a_refused_clean_draft_is_an_invented_objection():
    d = _recorded_decision(CAL, "faithful_paraphrase")
    out = classify("faithful_paraphrase", "approve", d)
    assert out["invented_objection"] is True
    assert out["actual_detection"] is None, "detection is not applicable here"


@pytest.mark.skipif(not GPT.is_file(), reason="cross-model artifact absent")
def test_gpt_refusing_a_clean_draft_on_word_fidelity_is_an_invention():
    d = _recorded_decision(GPT, "cal_ops_faithful")
    out = classify("cal_ops_faithful", "approve", d)
    assert out["invented_objection"] is True


def test_a_clean_approval_is_not_an_invention():
    d = dc.decide_from_text("Decision: approve\n\nIssues:\n- none\n",
                            _case("cal_ops_faithful").draft_before_trim,
                            _case("cal_ops_faithful").sources)
    out = classify("cal_ops_faithful", "approve", d)
    assert out["invented_objection"] is False
    assert out["valid_judgment"] is True


# ── Outcome 4: misses, including the ownership distinction ──────────────────

def test_owner_not_explicit_is_still_a_miss_on_an_owner_case():
    """A correct fact with no owning record fails where ownership is at issue."""
    c = fam_by_id("omitted_record")
    raw = ("Decision: revise\n\nIssues:\n"
           "- passage: (omitted)\n"
           "  problem: omitted\n  sources: [candidate-supplement.txt]\n"
           "  why: a role with two bullets is missing from the draft.\n"
           "  correction: add the missing role back.\n")
    d = dc.decide_from_text(raw, c.draft_before_trim, c.sources)
    out = classify("omitted_record", "revise", d)
    assert out["missed_defect"] is True, "ownership is at issue for this case"
    assert out["detection_detail"]["owner_required"] is True


def test_owner_is_not_required_for_a_local_clause_removal():
    d = _recorded_decision(CAL, "unsupported_method") if CAL.is_file() else None
    if d is None:
        pytest.skip("calibration artifact absent")
    out = classify("unsupported_method", "revise", d)
    assert out["detection_detail"]["owner_required"] is False
    assert out["actual_detection"] is True


# ── The four outcomes stay independent ──────────────────────────────────────

def test_the_outcomes_are_exclusive_per_case_kind():
    """Clean cases report invention; defective cases report detection/miss."""
    if not CAL.is_file():
        pytest.skip("calibration artifact absent")
    clean = classify("faithful_paraphrase", "approve",
                      _recorded_decision(CAL, "faithful_paraphrase"))
    assert clean["actual_detection"] is None and clean["missed_defect"] is False
    defective = classify("unsupported_method", "revise",
                          _recorded_decision(CAL, "unsupported_method"))
    assert defective["invented_objection"] is None
