"""Lock the claims in `results/calibration-v1/RESULTS.md`.

The results are a one-shot run: they cannot be regenerated without spending money
again, so the numbers must be pinned to the retained artifact. If the artifact is
edited or a classifier changes, these tests fail and the prose becomes visibly stale
instead of quietly wrong.

Skipped when the run artifact is absent, so a fresh clone without the paid run still
passes the suite.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/calibration-v1/calibration.json"

pytestmark = pytest.mark.skipif(not RUN.is_file(),
                                reason="calibration run artifact not present")

FAMILIAR_DEFECT_NEEDLES = {
    "unsupported_method": ["targeted quality audits"],
    "missing_fact": ["1,180"],
    "wrong_record_attribution": ["1,180"],
    "omitted_record": ["Ridgeline", "Inventory Analyst"],
}
UNFAMILIAR_DEFECT_NEEDLES = {
    "cal_lab_unsupported": ["EPA Method 1604"],
    "cal_lab_missing": ["24-hour"],
    "cal_lab_wrong_record": ["640"],
    "cal_lab_omitted": ["Cascade Microbiology"],
    "cal_ops_unsupported": ["LTL"],
    "cal_ops_missing": ["94%"],
    "cal_ops_wrong_record": ["260"],
    "cal_ops_omitted": ["Gorge Logistics"],
}
FAITHFUL = ("faithful_paraphrase", "supported_method", "transferable_skill",
            "cal_lab_faithful", "cal_ops_faithful")


@pytest.fixture(scope="module")
def run() -> dict:
    return json.loads(RUN.read_text())


# ── Budget compliance ────────────────────────────────────────────────────────

def test_the_run_stayed_inside_every_ceiling(run):
    frozen, meter = run["frozen"], run["meter"]
    assert meter["requests_sent"] == 18
    assert meter["requests_sent"] <= meter["max_requests"] == 81
    assert meter["logical_calls"] == 18 <= frozen["max_logical_calls"] == 27
    assert meter["cost_usd"] < frozen["max_spend_usd"]
    assert meter["cost_usd"] == pytest.approx(0.008488, abs=1e-9)


def test_every_request_produced_a_billable_response(run):
    """The accounting claim: 18 requests, 18 billable, 0 unmetered."""
    meter = run["meter"]
    assert meter["billable_requests"] == 18
    assert meter["unmetered_requests"] == 0
    assert meter["cost_is_known"] is True
    assert meter["unmetered_caveat"] is None
    # No retries were needed, so requests == logical calls.
    assert meter["requests_sent"] == meter["logical_calls"]


def test_the_frozen_identity_is_unchanged_by_the_run(run):
    frozen = run["frozen"]
    assert frozen["instructions_sha256"] == frozen["contract_fingerprint"]
    assert frozen["instructions_sha256"].startswith("4261e9cb5b031230")
    assert frozen["calibration_fixture_fingerprint"].startswith("b4b73e68b4d69b39")
    assert frozen["mode"] == "imaginative"
    assert len(run["rows"]) == 17


# ── Detection: 12 of 12 ──────────────────────────────────────────────────────

def test_every_defective_case_named_the_real_defect(run):
    """The strongest claim: 4/4 familiar + 8/8 unfamiliar."""
    checked = familiar = unfamiliar = 0
    for row in run["rows"]:
        spec = (FAMILIAR_DEFECT_NEEDLES if row["group"] == "familiar"
                else UNFAMILIAR_DEFECT_NEEDLES)
        if row["case_id"] not in spec:
            continue
        checked += 1
        raw = (row["first_response"] or "").lower()
        named = any(n.lower() in raw for n in spec[row["case_id"]])
        assert named, f"{row['case_id']} did not name its defect"
        if row["group"] == "familiar":
            familiar += 1
        else:
            unfamiliar += 1
    assert (familiar, unfamiliar, checked) == (4, 8, 12)


# ── Approval: 1 of 5 faithful ────────────────────────────────────────────────

def test_only_one_faithful_draft_was_approved(run):
    approved = [r["case_id"] for r in run["rows"]
                if r["case_id"] in FAITHFUL and r["first_decision"] == "approve"]
    assert approved == ["cal_ops_faithful"], approved
    # The other four were refused.
    refused = [r["case_id"] for r in run["rows"]
               if r["case_id"] in FAITHFUL and r["first_decision"] != "approve"]
    assert len(refused) == 4


def test_no_corrected_document_was_approved(run):
    """Approval of an exact proposal is still 0 — the axis has no positive case."""
    assert run["summary"]["familiar"]["corrected_document_approved"] == 0
    assert run["summary"]["unfamiliar"]["corrected_document_approved"] == 0
    # Exactly one proposal was ever drawn, and it was refused.
    drawn = [r for r in run["rows"] if r["proposal_drawn"]]
    assert len(drawn) == 1
    assert drawn[0]["re_review_decision"] == "revise"
    assert drawn[0]["release"] is False


def test_the_one_re_review_refused_on_the_invented_timeline(run):
    """The most informative cell: the real defect was fixed and it still refused."""
    row = next(r for r in run["rows"] if r["proposal_drawn"])
    assert row["re_review_valid"] is True, "a clean refusal, not an envelope failure"
    text = row["re_review_response"].lower()
    assert "contradiction" in text and "present" in text


# ── Invalid responses and scaffold limits ────────────────────────────────────

def test_seven_responses_were_structurally_invalid(run):
    invalid = [r["case_id"] for r in run["rows"] if r["first_valid"] is False]
    assert len(invalid) == 7
    assert set(invalid) == {
        "faithful_paraphrase", "missing_fact", "omitted_record",
        "cal_lab_unsupported", "cal_lab_missing", "cal_ops_missing",
        "cal_ops_omitted"}


def test_no_operational_failures(run):
    assert not [r for r in run["rows"] if r["first_error"]]
    assert run["summary"]["familiar"]["agreement_classes"]["operational_failure"] == 0
    assert run["summary"]["unfamiliar"]["agreement_classes"]["operational_failure"] == 0


def test_the_scaffold_drew_exactly_one_proposal(run):
    statuses = collections.Counter(r["repair_status"] for r in run["rows"])
    assert statuses["proposed_exact_owned_source_bullet"] == 1
    assert statuses["not_attempted"] == 8
    assert statuses["ambiguous_or_multiple_issues"] == 4
    assert sum(statuses.values()) == 17


# ── Pair separation: 2 of 8 ──────────────────────────────────────────────────

def test_pair_separation_is_two_of_eight(run):
    """Corrected from an earlier 4/8: an invalid variant is not a separation."""
    pairs = run["summary"]["pair_separation"]
    real = {k: v for k, v in pairs.items()
            if v["baseline_case"] != v["variant_case"]}
    assert len(real) == 8
    separated = sorted(k for k, v in real.items() if v["separated"])
    assert separated == ["cal_ops_faithful > cal_ops_unsupported",
                         "cal_ops_faithful > cal_ops_wrong_record"]


def test_the_two_unseparated_ops_pairs_are_invalid_not_missed(run):
    """They named the defect; the envelope failed. Not a detection failure."""
    for case_id in ("cal_ops_missing", "cal_ops_omitted"):
        row = next(r for r in run["rows"] if r["case_id"] == case_id)
        assert row["first_decision"] == "revise"
        assert row["first_valid"] is False
        assert row["first_structural_problems"]
