"""The cross-model comparison: matched design, and honest named checks.

Two things must hold for the comparison to mean anything:

1. **Only the model varies.** The drafts, sources, mode and instruction text are
   the frozen artifacts already used by the calibration run. A test asserts the
   prompt text and case list are unchanged, so a later edit cannot quietly turn
   this into a different experiment.

2. **The named checks detect the two behaviours the brief calls out**, without
   over-firing. The concurrency check is only meaningful on a clean draft: on a
   defective draft an overlap objection can be a legitimate finding, so applying
   the check there and then explaining it away would be exactly the kind of
   after-the-fact scoring this work exists to remove.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import decision_contract as dc
from experiment import cross_model
from experiment.calibration_cases import by_id as cal_by_id
from experiment.cases_v2 import by_id as fam_by_id
from experiment.cross_model import CAST, compare, named_checks

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/calibration-v1/calibration.json"


# ── The design is matched ────────────────────────────────────────────────────

def test_the_case_list_covers_every_behaviour_the_brief_names():
    probes = {friendly for _cid, _g, friendly in CAST}
    assert probes == {"clean_paraphrase", "source_backed_concurrency",
                      "unsupported_method", "omission_or_wrong_record"}
    # Two clean drafts, so "approved a clean draft" is not a single-observation claim.
    assert sum(1 for _c, _g, f in CAST if f == "clean_paraphrase") == 2
    # The concurrency probe must be a draft whose SOURCES state an overlap.
    concurrency_case = next(cid for cid, _g, f in CAST
                            if f == "source_backed_concurrency")
    case = cal_by_id(concurrency_case)
    both = " ".join(case.sources.values())
    assert "03/2021 - Present" in both and "09/2023 - Present" in both


def test_the_instructions_are_the_frozen_ones():
    """Only the model may vary. The prompt text must be the calibration's."""
    instructions = cross_model.frozen_instructions()
    assert hashlib.sha256(instructions.encode()).hexdigest() == \
        cross_model.hashlib.sha256(instructions.encode()).hexdigest()
    assert instructions == dc.contract_for(cross_model.MODE)
    assert cross_model.MODE == "imaginative"
    # The four general evidence principles are present, so this is the refined
    # instruction set and not the original one.
    flat = " ".join(instructions.lower().split())
    assert "overlapping dates are a fact" in flat
    assert "synonymous wording is permitted" in flat


def test_the_plan_is_bounded():
    assert cross_model.MAX_REQUESTS == len(CAST) * cross_model.ATTEMPTS_PER_CALL
    assert cross_model.MAX_REQUESTS == 14
    assert cross_model.MAX_SPEND_USD <= 1.0


# ── The named checks ─────────────────────────────────────────────────────────

def _decision(issues: list[tuple[str, str]]) -> dc.Decision:
    return dc.Decision(
        decision="revise",
        issues=[dc.Issue(passage="", problem=p, why=w, correction=None)
                for p, w in issues])


def test_concurrency_is_detected_on_a_clean_draft():
    """The observed phrasing, verbatim from the run.

    The model wrote "concurrent employment (overlapping dates) … roles as
    sequential rather than concurrent" — no word "impossible", yet the same
    invented exclusivity rule.
    """
    decision = _decision([("misattributed",
                           "the source documents show concurrent employment "
                           "(overlapping dates), misrepresenting the roles as "
                           "sequential rather than concurrent")])
    checks = named_checks("cal_lab_faithful", "source_backed_concurrency", decision)
    assert checks["concurrency_claimed_impossible"] is True


def test_the_other_observed_phrasing_is_also_detected():
    decision = _decision([("unsupported",
                           "the concurrent 'Present' roles without clarification "
                           "violate the timeline")])
    assert named_checks("faithful_paraphrase", "clean_paraphrase",
                        decision)["concurrency_claimed_impossible"] is True


def test_a_clean_approval_raises_no_flags():
    decision = dc.Decision(decision="approve", issues=[])
    checks = named_checks("cal_ops_faithful", "clean_paraphrase", decision)
    assert checks["concurrency_claimed_impossible"] is False
    assert checks["synonymy_objected"] is False
    assert checks["approved_clean_draft"] is True
    assert checks["clean_refused"] is False


def test_the_checks_are_not_applied_to_defective_drafts():
    """An overlap objection can be legitimate where a defect exists."""
    decision = _decision([("misattributed",
                           "the figure belongs to Harborline, causing an overlap "
                           "in the Cascade record")])
    checks = named_checks("wrong_record_attribution",
                          "omission_or_wrong_record", decision)
    assert checks["concurrency_claimed_impossible"] is False
    assert checks["approved_clean_draft"] is None


def test_synonymy_objection_is_detected_on_a_clean_draft():
    decision = _decision([("unsupported",
                           "the draft adds a causal mechanism ('through') the "
                           "source does not state")])
    assert named_checks("faithful_paraphrase", "clean_paraphrase",
                        decision)["synonymy_objected"] is True


# ── The comparison shape ─────────────────────────────────────────────────────

def _row(case_id: str, label: str, decision: str, objections=()) -> dict:
    return {"case_id": case_id, "model_label": label, "decision": decision,
            "valid": True, "error": "", "cost_usd": 0.0,
            "objections": [{"problem": p, "passage": "", "why": w,
                            "correction": None, "passage_resolves": True}
                           for p, w in objections]}


def test_compare_puts_two_models_side_by_side_per_case():
    rows = [
        _row("cal_ops_faithful", "model-a", "approve"),
        _row("cal_ops_faithful", "model-b", "revise",
             [("unsupported", "not mentioned in the source")]),
    ]
    out = compare(rows)
    entry = out["per_case"]["cal_ops_faithful"]
    assert set(entry["models"]) == {"model-a", "model-b"}
    assert entry["models"]["model-a"]["checks"]["approved_clean_draft"] is True
    assert entry["models"]["model-b"]["checks"]["clean_refused"] is True


@pytest.mark.skipif(not RUN.is_file(), reason="calibration artifact absent")
def test_the_qwen_baseline_for_the_matched_cast():
    """Locked baseline: what the incumbent model does on exactly these 7 drafts.

    Recorded so the second model's numbers are compared against a fixed value
    rather than against a re-run that might drift.
    """
    run = json.loads(RUN.read_text())
    rows = []
    for case_id, _group, _friendly in CAST:
        src = next(r for r in run["rows"] if r["case_id"] == case_id)
        decision = dc.decide_from_text(src["first_response"], src["draft"],
                                        src["sources"])
        rows.append({
            "case_id": case_id, "model_label": "qwen/qwen3-coder",
            "decision": decision.decision,
            "valid": bool(decision.decision) and not decision.structural_problems,
            "error": "", "cost_usd": 0.0,
            "objections": [{"problem": i.problem, "passage": i.passage,
                            "why": i.why, "correction": i.correction,
                            "passage_resolves": i.passage_resolves}
                           for i in decision.issues]})
    summary = compare(rows)["qwen/qwen3-coder"]
    assert summary["usable_answers"] == 5
    assert summary["concurrency_claims"] == 2
    assert summary["synonymy_objections"] == 2
    assert summary["clean_approved"] == 1
