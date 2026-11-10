"""The four repair gaps, driven by Gemma's REAL retained corrections.

Each test below uses the exact passage and correction text Gemma produced in the
retained screen run (`results/model-screen-v1/screen.json`), not a hand-written
example. The current `safe_repair` fails all four:

1. **Paraphrase rejected.** `"Cut rework by 18% through standardizing…"` is refused
   because the replacement is not a byte-exact source bullet. The source says "by";
   the model wrote "through". That is ordinary paraphrase and must be allowed.
2. **Wrong-record correction refused** as `replacement_not_exact_owned_source_bullet`
   for the same reason — it drops the merged clause rather than restating the source.
3. **Omission has no destination.** `safe_repair` returns
   `omission_requires_explicit_destination` and never restores anything, so a
   source-backed record can never be put back.
4. Ambiguity, invented dates and cross-record leakage must still be refused — those
   are the constraints that make 1–3 safe.

The tests are written to fail against the CURRENT implementation; the module under
test (`experiment/repair.py`) does not exist yet, so collection fails first. That
failure is recorded in the commit message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import decision_contract as dc
from experiment.binding_candidate import safe_repair as legacy_safe_repair
from experiment.cases_v2 import SOURCES, by_id

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "results/model-screen-v1/screen.json"


# ── Gemma's exact retained corrections ──────────────────────────────────────

def gemma_issue(case_id: str) -> dc.Issue:
    """The real issue block Gemma returned for this case in the retained run."""
    data = json.loads(SCREEN.read_text())
    row = next(r for r in data["rows"]
               if r["model"] == "google/gemma-4-31b-it" and r["case_id"] == case_id)
    case = by_id(case_id)
    decision = dc.decide_from_text(row["response"], case.draft_before_trim,
                                   case.sources)
    assert len(decision.issues) == 1, decision.structural_problems
    return decision.issues[0]


@pytest.mark.skipif(not SCREEN.is_file(), reason="screen artifact absent")
def test_the_legacy_repair_refuses_gemma_paraphrase():
    """Documents the gap. This is the CURRENT behaviour, asserted so the fix is
    provable: the same input must be refused before and accepted after."""
    issue = gemma_issue("unsupported_method")
    assert issue.correction == (
        "Cut rework by 18% through standardizing panel labeling across the "
        "terminal fleet.")
    case = by_id("unsupported_method")
    doc, status = legacy_safe_repair(case.draft_before_trim, SOURCES, [issue])
    assert doc == case.draft_before_trim, "legacy refuses to change anything"
    assert status == "replacement_not_exact_owned_source_bullet"


@pytest.mark.skipif(not SCREEN.is_file(), reason="screen artifact absent")
def test_the_legacy_repair_already_handles_gemma_wrong_record_correction():
    """This gap is NOT present for wrong-record: the correction restates a bullet
    that IS an exact source bullet, so the legacy check accepts it.

    Recorded explicitly so the fix is only credited for the gaps that are real.
    """
    issue = gemma_issue("wrong_record_attribution")
    case = by_id("wrong_record_attribution")
    doc, status = legacy_safe_repair(case.draft_before_trim, SOURCES, [issue])
    assert status == "proposed_exact_owned_source_bullet"
    assert doc != case.draft_before_trim


@pytest.mark.skipif(not SCREEN.is_file(), reason="screen artifact absent")
def test_the_legacy_repair_cannot_restore_an_omitted_record():
    issue = gemma_issue("omitted_record")
    case = by_id("omitted_record")
    doc, status = legacy_safe_repair(case.draft_before_trim, SOURCES, [issue])
    assert doc == case.draft_before_trim
    assert status == "omission_requires_explicit_destination"


# ── The new repair must satisfy all four behaviours ─────────────────────────

def _repair():
    from experiment.repair import apply_repair
    return apply_repair


@pytest.mark.skipif(not SCREEN.is_file(), reason="screen artifact absent")
def test_1_paraphrased_correction_preserves_the_supported_achievement():
    """Remove the invented clause, keep the sourced figure, method and employer."""
    case = by_id("unsupported_method")
    issue = gemma_issue("unsupported_method")
    doc, status = _repair()(case.draft_before_trim, SOURCES, [issue])

    assert status == "applied", status
    assert "targeted quality audits" not in doc
    # Everything the source supports must survive.
    assert "18%" in doc
    assert "standardizing panel labeling" in doc
    assert "Cascade Freight Partners" in doc
    assert "terminal fleet" in doc


@pytest.mark.skipif(not SCREEN.is_file(), reason="screen artifact absent")
def test_2_wrong_record_correction_keeps_the_fact_under_its_owner():
    """The 1,180 figure must remain, under Harborline, and leave Cascade."""
    case = by_id("wrong_record_attribution")
    issue = gemma_issue("wrong_record_attribution")
    doc, status = _repair()(case.draft_before_trim, SOURCES, [issue])

    assert status == "applied", status
    # The merged bullet under Cascade no longer claims the invoice figure.
    cascade_block = doc.split("**Cascade Freight Partners**")[1]
    cascade_bullets = "\n".join(
        line for line in cascade_block.splitlines() if line.startswith("- "))
    assert "1,180" not in cascade_bullets, cascade_block
    # The fact must still be present under its correct owner. The Harborline
    # bullet is untouched by the edit, so it must appear exactly once.
    harborline_block = doc.split("**Harborline Transit Authority**")[1].split("**")[0]
    assert "Reconciled 1,180 fuel invoices" in harborline_block
    # No employer was merged away or duplicated.
    assert doc.count("**Cascade Freight Partners**") == 1
    assert doc.count("**Harborline Transit Authority**") == 1
    assert doc.count("**Ridgeline Equipment Services**") == 1
    # The surviving Cascade bullet is the sourced one, not the merged version.
    cascade_block = doc.split("**Cascade Freight Partners**")[1]
    assert "Prepared the monthly cost summary reviewed by the regional manager." \
        in cascade_block
    assert "and reconciled 1,180" not in cascade_block


@pytest.mark.skipif(not SCREEN.is_file(), reason="screen artifact absent")
def test_3_missing_record_is_restored_with_its_own_dates_and_bullets():
    """Employer, title, dates and bullets come from the SOURCE, not the model."""
    case = by_id("omitted_record")
    issue = gemma_issue("omitted_record")
    doc, status = _repair()(case.draft_before_trim, SOURCES, [issue])

    assert status == "applied", status
    assert "Ridgeline Equipment Services" in doc
    assert "Inventory Analyst" in doc
    assert "08/2023" in doc and "Present" in doc
    # The source's own bullets, verbatim.
    assert "Maintain parts availability records for 7 service branches." in doc
    assert ("Compared quarterly parts usage against the prior four quarters to set "
            "reorder points.") in doc
    # Nothing invented.
    for invented in ("08/2022", "03/2023"):
        assert invented not in doc


@pytest.mark.skipif(not SCREEN.is_file(), reason="screen artifact absent")
def test_4_ambiguous_target_is_refused_not_first_match():
    """Two identical bullets: refuse, do not edit the first one found."""
    case = by_id("unsupported_method")
    issue = gemma_issue("unsupported_method")
    draft = case.draft_before_trim + "\n" + case.draft_before_trim
    doc, status = _repair()(draft, SOURCES, [issue])
    assert doc == draft
    assert "ambiguous" in status


@pytest.mark.skipif(not SCREEN.is_file(), reason="screen artifact absent")
def test_5_invented_dates_are_refused_for_restoration():
    """A restoration whose dates are not in the source must not be applied."""
    case = by_id("omitted_record")
    issue = gemma_issue("omitted_record")
    issue.correction = issue.correction.replace("08/2023 - Present",
                                                "08/2022 - 03/2023")
    doc, status = _repair()(case.draft_before_trim, SOURCES, [issue])
    assert doc == case.draft_before_trim
    assert status.startswith("restoration_") or status == "invented_content"


@pytest.mark.skipif(not SCREEN.is_file(), reason="screen artifact absent")
def test_6_qualifications_are_not_moved_between_records():
    """A correction that relocates a credential must be refused."""
    case = by_id("omitted_record")
    issue = gemma_issue("omitted_record")
    issue.correction = ("Ridgeline Equipment Services** -- Fife, WA *Inventory "
                        "Analyst | 08/2023 - Present* - Certified Inventory Control "
                        "Specialist (CICS)")
    doc, status = _repair()(case.draft_before_trim, SOURCES, [issue])
    assert doc == case.draft_before_trim
    assert status != "applied"


# ── The controls must never be edited ──────────────────────────────────────

def test_7_an_approval_produces_no_edit():
    """`apply_repair` is only reached on a revise; an empty issue list is a no-op."""
    case = by_id("faithful_paraphrase")
    from experiment.repair import apply_repair
    doc, status = apply_repair(case.draft_before_trim, SOURCES, [])
    assert doc == case.draft_before_trim
    assert status in ("no_change", "not_applicable")
