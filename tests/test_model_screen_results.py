"""Lock the screen's reported results to the retained artifact.

The run cannot be regenerated without spending money again, so its numbers are
pinned here. If the artifact is edited or a classifier changes, these fail and the
prose becomes visibly stale instead of quietly wrong.

Also pinned: the THREE measurement limitations that make the MiMo row an
understatement. Those are the difference between "MiMo missed 2 of 3 defects" and
"the parser discarded 2 of MiMo's answers before reading them", and that distinction
must not be lost in a later summary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import decision_contract as dc
from experiment.cases_v2 import by_id

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/model-screen-v1/screen.json"

pytestmark = pytest.mark.skipif(not RUN.is_file(), reason="screen artifact absent")


@pytest.fixture(scope="module")
def run() -> dict:
    return json.loads(RUN.read_text())


# ── Budget ───────────────────────────────────────────────────────────────────

def test_the_run_stayed_inside_every_ceiling(run):
    m = run["meter"]
    assert m["requests_sent"] == 20
    assert m["requests_sent"] <= m["max_requests"] == 40
    assert m["logical_calls"] == 20
    assert m["unmetered_requests"] == 0
    assert m["cost_is_known"] is True
    assert m["cost_usd"] == pytest.approx(0.005143, abs=1e-9)
    assert m["cost_usd"] < run["plan"]["max_spend_usd"]


def test_no_operational_failures_and_no_retries(run):
    assert not [r for r in run["rows"] if r["error"]]
    assert all(r["requests"] == 1 for r in run["rows"]), "no retries should be needed"
    assert run["meter"]["requests_sent"] == run["meter"]["logical_calls"]


# ── Headline: gemma ──────────────────────────────────────────────────────────

def test_gemma_is_the_only_model_meeting_the_criterion(run):
    s = run["summary"]["google/gemma-4-31b-it"]
    assert s["valid_judgment"] == 5
    assert s["actual_detection"] == 3
    assert s["invented_objections"] == 0
    assert s["missed_defects"] == 0
    assert s["clean_approved"] == 2
    # No other model reaches this on all four axes.
    for model, other in run["summary"].items():
        if model == "google/gemma-4-31b-it":
            continue
        assert not (other["valid_judgment"] == 5
                    and other["actual_detection"] == 3
                    and other["invented_objections"] == 0)


def test_gemma_approved_both_clean_drafts_with_no_issues(run):
    for case_id in ("faithful_paraphrase", "cal_ops_faithful"):
        row = next(r for r in run["rows"]
                   if r["model"] == "google/gemma-4-31b-it"
                   and r["case_id"] == case_id)
        assert row["decision"] == "approve"
        assert row["outcomes"]["issues"] == 0
        assert row["outcomes"]["invented_objection"] is False


def test_gemma_is_the_cheapest(run):
    costs = {m: s["cost_usd"] for m, s in run["summary"].items()}
    assert min(costs, key=costs.get) == "google/gemma-4-31b-it"


# ── The pin held ─────────────────────────────────────────────────────────────

def test_the_cohere_pin_held_on_every_call(run):
    """Case-insensitive: the pin is `cohere/fp8`, the report says `Cohere`."""
    mimo = [r for r in run["rows"] if r["model"] == "xiaomi/mimo-v2.6-flash"]
    assert len(mimo) == 5
    for row in mimo:
        assert row["provider_pinned"] == ["cohere/fp8"]
        assert row["provider_reported"] == "Cohere"
        assert row["provider_reported"].lower() in \
            [p.split("/")[0].lower() for p in row["provider_pinned"]]
    # The fallback must never have been contacted.
    assert all(r["provider_reported"] != "Makora" for r in run["rows"])


# ── The three measurement limitations ───────────────────────────────────────

def test_a_missing_decision_line_discards_the_whole_answer(run):
    """Limitation 1: MiMo's correct issue blocks score zero.

    This is a HARNESS defect, not a model failure, and it must stay visible.
    """
    blind = [r for r in run["rows"] if r["decision"] is None]
    assert len(blind) == 2
    assert all(r["model"] == "xiaomi/mimo-v2.6-flash" for r in blind)
    for row in blind:
        assert "decision_missing" in row["outcomes"]["structural"]
        # ...yet the raw text clearly contains an issue block.
        assert "Issues:" in row["response"]
        assert "problem:" in row["response"]


def test_issues_are_not_parsed_when_the_decision_line_is_absent():
    """The mechanism, stated directly."""
    raw = ("Issues: - passage: Cut rework by 18% through X. problem: unsupported "
           "sources: [a] why: y correction: z")
    d = dc.decide_from_text(raw, "draft text\n", {"a": "source"})
    assert d.decision == ""
    assert d.structural_problems == ["decision_missing"]
    assert d.issues == [], "the issue block is discarded, not parsed"


def test_a_correct_answer_is_recoverable_from_the_recorded_bytes():
    """The diagnostic: prefixing a decision line recovers the finding."""
    run_data = json.loads(RUN.read_text())
    row = next(r for r in run_data["rows"]
               if r["model"] == "xiaomi/mimo-v2.6-flash"
               and r["case_id"] == "wrong_record_attribution")
    case = by_id("wrong_record_attribution")
    m = re.search(r"^\s*Issues\s*:", row["response"], re.I | re.M)
    assert m is not None
    recovered = dc.decide_from_text("Decision: revise\n" + row["response"][m.start():],
                                    case.draft_before_trim, case.sources)
    assert len(recovered.issues) == 1
    assert recovered.issues[0].problem == "misattributed"


def test_the_absence_phrase_list_misses_never_states():
    """Limitation 2: correct reasoning phrased outside the list scores undetermined."""
    from experiment.detection_rubric import ABSENT_FROM_SOURCE, _any_phrase
    phrase = "an additional method the source never states was used"
    assert _any_phrase(phrase.lower(), ABSENT_FROM_SOURCE) == []


def test_approve_with_issues_is_invalid_but_keeps_its_issues(run):
    """Limitation 3: the contract is right to reject it; the issues still exist."""
    invalid_approvals = [r for r in run["rows"]
                         if r["decision"] == "approve"
                         and r["outcomes"]["valid_judgment"] is False]
    assert invalid_approvals
    for row in invalid_approvals:
        assert "approve_carries_issues" in row["outcomes"]["structural"]
        assert row["outcomes"]["issues"] >= 1


# ── Missed defects are recorded, not inferred ───────────────────────────────

def test_no_approval_of_a_defective_document_was_counted_as_a_release(run):
    """`approve` on a defective case is a MISS, never a success."""
    for row in run["rows"]:
        if row["expected"] != "revise":
            continue
        if row["decision"] == "approve":
            assert row["outcomes"]["missed_defect"] is True
            assert row["outcomes"]["actual_detection"] is False


def test_gemini_claimed_a_fact_was_missing_that_is_in_the_draft(run):
    """Verified against the fixture: `14 depots` IS present in the draft."""
    case = by_id("omitted_record")
    assert "14 depots" in case.draft_before_trim
    row = next(r for r in run["rows"]
               if r["model"] == "google/gemini-2.5-flash-lite"
               and r["case_id"] == "omitted_record")
    assert "14 depots" in row["response"]
    assert row["outcomes"]["missed_defect"] is True


def test_the_recorded_pin_verdict_is_case_insensitive_and_consistent(run):
    """The stored `provider_pin_held` must agree with the recorded provider names.

    It did not at first: the check compared `cohere/fp8` to `Cohere`
    case-sensitively and marked all five MiMo calls as violations. This asserts the
    rows and the summary agree with each other AND with the plain reading of the
    data, so the field cannot silently go stale again.
    """
    for row in run["rows"]:
        pin = row.get("provider_pinned") or []
        reported = row.get("provider_reported") or ""
        expected = (not pin) or (reported.lower()
                                 in [p.split("/")[0].lower() for p in pin])
        assert row["provider_pin_held"] is expected, (row["model"], row["case_id"])

    for model, summary in run["summary"].items():
        n = sum(1 for r in run["rows"] if r["model"] == model
                and r["provider_pin_held"] is False)
        assert summary["pin_violations"] == n, model


def test_the_summary_states_that_pinning_was_recomputed(run):
    assert "recomputed" in run.get("summary_note", "").lower()
