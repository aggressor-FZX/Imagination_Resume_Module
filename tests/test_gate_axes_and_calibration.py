"""Tests for the four reporting axes, the calibration fixture, and the re-review guard.

These guard the claims the packet makes, so the claims cannot drift from the code
that produced them:

  * the denominator excludes exactly the cells that answered a different contract,
    and reports them rather than dropping them;
  * the four axes are computed independently and never collapsed into one number;
  * a re-review of a different document is reported as no observation — the item 7
    rule, and the one that is easiest to get silently wrong;
  * the calibration fixture is isolated, paired, and frozen.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import decision_contract as dc
from experiment import calibration_cases as cal
from experiment import gate_axes
from experiment.calibration_cases import (
    CALIBRATION_CASES, CALIBRATION_MUTATIONS, DEFECTIVE_IDS, FAITHFUL_IDS,
    check as check_calibration,
)
from experiment.cases_v2 import CASES_V2

ROOT = Path(__file__).resolve().parents[1]
RETAINED = ROOT / "results/comparison-v2b-20261109/attempts.json"


# ── The denominator rule (item 2) ─────────────────────────────────────────────

def test_the_denominator_excludes_legacy_json_and_truncated_cells():
    summary = gate_axes.build()
    assert summary["cells_total"] == 28
    assert summary["denominator_cells"] == 14
    assert summary["excluded"] == {"legacy_json_envelope": 13, "no_response": 1}
    # Excluded cells are still REPORTED, with a reason, not silently dropped.
    excluded = [r for r in summary["rows"] if not r["in_candidate_denominator"]]
    assert len(excluded) == 14
    assert all(r["excluded_reason"] for r in excluded)
    # ...and they contribute nothing to any semantic axis.
    assert all(r["axis_initial_judgment"] is None for r in excluded)
    assert all(r["axis_release"] is None for r in excluded)


def test_classification_is_by_contract_not_by_outcome():
    assert gate_axes.classify_answer("") == "no_response"
    assert gate_axes.classify_answer('  {"claims": []}') == "legacy_json_envelope"
    assert gate_axes.classify_answer("Decision: approve\n") == "prose"
    # A prose answer that happens to be wrong is still contract-compatible.
    assert gate_axes.classify_answer("nonsense") == "prose"


def test_the_packet_readme_and_the_axes_module_agree_on_the_denominator(tmp_path):
    """Two generators, one rule. They diverged once (15 vs 14) and must not again.

    The drift happened because the README inferred "prose" from the gate's reason
    string, and a missing response shares the reason `initial_review_unverifiable`
    with unreadable prose. This test compares the two independently computed
    numbers.
    """
    from experiment.review_packet import build as build_packet
    build_packet(tmp_path)
    readme = (tmp_path / "README.md").read_text()
    axes = gate_axes.build()
    expected = f"prose cells: **{axes['denominator_cells']}**"
    assert expected in readme, readme[readme.index("Denominator"):][:400]


# ── The four axes stay separate (item 7) ──────────────────────────────────────

def test_the_four_axes_are_reported_separately_and_never_summed():
    summary = gate_axes.build()
    for key in ("axis_1_initial_judgment", "axis_2_edit_applicability",
                "axis_3_corrected_document_approval", "axis_4_release"):
        assert key in summary, key
    # No single composite field exists that could be quoted as a score.
    assert "score" not in json.dumps({k: v for k, v in summary.items()
                                      if k != "rows"})

    a1 = summary["axis_1_initial_judgment"]
    a2 = summary["axis_2_edit_applicability"]
    a3 = summary["axis_3_corrected_document_approval"]
    a4 = summary["axis_4_release"]
    # Each axis carries its own caveat that forbids the conflation.
    assert "not evidence the objection was correct" in a1["caveat"]
    assert "limitation of the narrow repair scaffold" in a2["caveat"]
    assert "never as an approval or a refusal" in a3["caveat"]
    assert a4["released"] + a4["withheld"] + a4["undetermined"] == \
        summary["denominator_cells"]


def test_a_re_review_of_a_different_document_is_no_observation():
    """Item 7. Five retained re-reviews answered about a DIFFERENT document.

    They must not be read as approving or refusing the gate's proposal. This test
    pins the mechanism: identity is computed from the recorded prompt, and a
    mismatch forces `None` on the approval axis.
    """
    rows = gate_axes.build()["rows"]
    different = [r for r in rows
                 if r["axis_re_review_document_identity"] == "different_document"]
    assert len(different) == 5
    for row in different:
        assert row["axis_corrected_document_approval"] is None
        assert "DIFFERENT document" in row["axis_corrected_document_note"]

    same = [r for r in rows
            if r["axis_re_review_document_identity"] == "same_document"]
    assert len(same) == 2
    for row in same:
        # Item 2: the three valid-document outcomes are distinct. Neither a valid
        # refusal nor an unreadable answer may be reported as approval.
        assert row["axis_corrected_document_approval"] in (
            "approved", "not_approved", "invalid_re_review")


def test_document_from_prompt_recovers_the_reviewed_text_exactly():
    draft = "## Summary\nText.\n"
    built = "instructions\n\nSOURCE DOCUMENTS:\n[a]\nx\n\nDRAFT TO REVIEW:\n" + draft
    assert gate_axes.document_from_prompt(built) == draft
    assert gate_axes.document_from_prompt("no marker here") is None


# ── The calibration fixture (item 6) ──────────────────────────────────────────

def test_calibration_fixture_is_isolated_and_paired():
    verdict = check_calibration()
    assert verdict["problems"] == [], verdict["problems"]
    assert verdict["cases"] == 10
    assert verdict["faithful"] == 2 and verdict["defective"] == 8
    # Every defective draft differs from its baseline in exactly ONE region.
    for case_id, regions in verdict["regions_from_baseline"].items():
        assert regions <= 1, (case_id, regions)
    for case_id in DEFECTIVE_IDS:
        assert verdict["regions_from_baseline"][case_id] == 1, case_id


def test_every_unfamiliar_pair_has_a_faithful_control():
    for case_id, mutation in CALIBRATION_MUTATIONS.items():
        assert mutation["baseline"] in FAITHFUL_IDS, case_id
        assert mutation["contrast"] in {c.case_id for c in CALIBRATION_CASES}


def test_calibration_cases_use_unfamiliar_sources():
    """The controls must not reuse the familiar fixture's CONTENT.

    Markdown section headings are structure, not evidence: every resume has a
    `## Professional Experience`, and requiring the controls to invent new headings
    would make them less realistic without making them more independent. What must
    be unfamiliar is the substantive lines — employers, figures, dates, bullets.
    """
    def content_lines(text: str) -> set[str]:
        return {line.strip() for line in text.splitlines()
                if line.strip() and not line.strip().startswith("#")}

    familiar = set()
    from experiment.cases_v2 import SOURCES as FAMILIAR_SOURCES
    for text in FAMILIAR_SOURCES.values():
        familiar |= content_lines(text)

    for case in CALIBRATION_CASES:
        for name, text in case.sources.items():
            assert name not in FAMILIAR_SOURCES, (case.case_id, name)
            overlap = familiar & content_lines(text)
            assert not overlap, (case.case_id, sorted(overlap)[:3])


def test_calibration_fixture_fingerprint_is_stable():
    first = cal.cases_fingerprint()
    second = cal.cases_fingerprint()
    assert first == second
    # A change to any draft must change the identity.
    import hashlib
    blob = [{"case_id": c.case_id, "expectation": c.expectation,
             "draft": c.draft_before_trim,
             "sources": dict(sorted(c.sources.items())),
             "mutation": CALIBRATION_MUTATIONS[c.case_id]}
            for c in CALIBRATION_CASES]
    assert hashlib.sha256(json.dumps(blob, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest() == first


# ── Instruction freeze (item 6) ──────────────────────────────────────────────

def test_instructions_carry_the_four_general_evidence_principles():
    flat = " ".join(dc.contract_for("imaginative").lower().split())
    assert "overlapping dates are a fact, not a contradiction" in flat
    assert "synonymous wording is permitted" in flat
    assert "a materially missing source fact must be identified" in flat
    assert "a fact must stay under the record that stated it" in flat


def test_the_evidence_principles_are_mode_independent():
    """They are general rules, so they belong to the shared contract."""
    shared = " ".join(dc.DECISION_CONTRACT.lower().split())
    assert "overlapping dates are a fact" in shared
    assert "synonymous wording is permitted" in shared


def test_forging_a_different_timeline_is_still_rejected_by_the_contract():
    """The new rule must not be read as 'never object to dates'.

    The rule permits a CONCURRENT timeline the source states. It does not permit a
    document to assert dates the source contradicts, and the contract must still
    name that as a defect class.
    """
    flat = " ".join(dc.contract_for("imaginative").lower().split())
    assert "object only when the draft asserts a timeline the source denies" in flat
    assert "dates" in flat and "relationships" in flat


def test_contract_fingerprint_tracks_the_text():
    before = dc.contract_fingerprint("imaginative")
    assert before == dc.contract_fingerprint("imaginative")
    assert before != dc.contract_fingerprint("conservative")
    # Conservative and unknown modes share the strict text, so they share a hash.
    assert dc.contract_fingerprint(None) == dc.contract_fingerprint("conservative")


# ── The re-review guard in the calibration runner (item 7) ────────────────────

def test_the_calibration_prompt_round_trips_the_exact_document():
    from experiment.run_calibration import build_prompt, document_from_prompt
    draft = "## Professional Summary\nExact text.\n"
    prompt = build_prompt("INSTRUCTIONS", {"a.txt": "src"}, draft)
    assert document_from_prompt(prompt) == draft


def test_the_calibration_plan_is_bounded_and_frozen(tmp_path):
    from experiment.run_calibration import (
        ATTEMPTS_PER_CALL, DEFAULT_MAX_REQUESTS, MAX_LOGICAL_CALLS, freeze,
    )
    frozen = freeze()
    assert frozen["max_logical_calls"] == len(CASES_V2) + 2 * len(CALIBRATION_CASES)
    assert frozen["max_logical_calls"] == 27
    # The provable ceiling is REQUEST-based, because a logical call can retry.
    assert frozen["max_requests"] == MAX_LOGICAL_CALLS * ATTEMPTS_PER_CALL
    assert frozen["max_requests"] == DEFAULT_MAX_REQUESTS == 81
    assert frozen["instructions_sha256"] == frozen["contract_fingerprint"]
    assert frozen["calibration_fixture_fingerprint"] == cal.cases_fingerprint()
    # The frozen text is the text the contract produces NOW; any edit breaks this.
    assert frozen["instructions"] == dc.contract_for("imaginative")
    # The mode must match the retained matrix, or `transferable_skill`'s label
    # becomes incoherent: it is approvable only under Imaginative.
    assert frozen["mode"] == "imaginative"


def test_provider_cost_reads_the_cost_key_not_total_cost():
    """The $0.00 cost bug: `usage.cost` is the field OpenRouter actually sends.

    The parsing moved into `_usage_cost` when per-request metering was added, so
    this asserts on that helper directly — behaviour, not source text.
    """
    from experiment.provider_v2 import _usage_cost
    assert _usage_cost({"cost": 0.004}) == (0.004, True)
    assert _usage_cost({"total_cost": 0.007}) == (0.007, True)
    # `cost` wins when both are present.
    assert _usage_cost({"cost": 0.004, "total_cost": 0.007}) == (0.004, True)
    # Missing usage is UNKNOWN, not free: reported=False, cost 0.0.
    assert _usage_cost({}) == (0.0, False)
