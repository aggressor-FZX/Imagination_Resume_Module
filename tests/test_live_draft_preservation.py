"""Regressions for the defects observed in Conservative diagnostic 3abb7811.

The live writer output (retained privately in the protected capture) replaced
all three employer names with ``**Current Employer** --``, reduced every role to
``*Analyst |*``, and emitted empty ``## Certifications`` and ``## Technical
Skills`` sections. The source states three employers, three role/date lines, one
certification and four skill bullets. Preservation returned that draft
UNCHANGED, so the reviewer reported those source facts missing and the run
withheld as ``source_facts_missing``.

These tests drive the exact draft shape through ``preserve_source_facts`` and
assert the candidate's own facts are restored. The frozen fixture is unchanged.
"""
import json
from pathlib import Path

import source_preservation as sp

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bridge_opportunity"
CAPTURE = Path.home() / ".local/state/imaginator-capture" / (
    "3abb7811-3ff7-4b7e-beb9-c356844e22c0.json")


def _frozen_source() -> str:
    main = (FIXTURE / "candidate-main.txt").read_text()
    supp = (FIXTURE / "candidate-supplement.txt").read_text()
    return main + "\n\n---\n\n" + supp


# The exact writer output from the capture, verbatim (no private data: the
# fixture is synthetic). Kept inline so the test runs from a clean checkout.
DRAFT = """## Professional Summary

Operations analyst with 9 years of experience in depot reporting and fuel cost control across public transit operations. Operations analyst supporting stock planning and parts availability for a regional equipment service network.

## Professional Experience
**Current Employer** --
*Analyst |*
- Tracked monthly fuel consumption across 14 depots, comparing each month to the same month in the prior year.
- Built the depot scorecard reviewed by supervisors during weekly operations meetings.
- Reconciled 1,180 fuel invoices against depot dispensing records each quarter.

**Current Employer** --
*Analyst |*
- Audited 3 years of maintenance cost records to identify equipment with repeated failures.
- Prepared monthly cost summaries reviewed by the regional manager.
- Rebuilt the shipment delay log used by the dispatch team.

**Current Employer** --
*Analyst |*
- Maintained parts availability records for 7 service branches.
- Flagged slow-moving stock and proposed a reduction plan adopted by 2 branches.
- Compared quarterly parts usage against previous four quarters to set reorder points.

## Projects

## Education

**Puget Sound Technical College** -- Tacoma, WA
*Associate of Applied Science, Business Analytics | 2015*

**Inland Records Institute** -- Fife, WA
*Certificate, Inventory Control | 2022*

## Certifications

## Technical Skills
"""


def test_employer_names_and_role_lines_are_restored():
    """Placeholder employer heads and dateless roles are replaced from source."""
    out, changes = sp.preserve_source_facts(DRAFT, _frozen_source())
    assert "restored the source's employer and role records" in changes
    for employer in ("Harborline Transit Authority", "Cascade Freight Partners",
                     "Ridgeline Equipment Services"):
        assert employer in out, f"employer lost: {employer}"
    for role in ("Fuel Reporting Analyst | 04/2019 - Present",
                 "Cost Reporting Coordinator | 07/2015 - 03/2019",
                 "Inventory Analyst | 08/2023 - Present"):
        assert role in out, f"role/date lost: {role}"
    assert "Current Employer" not in out
    assert "*Analyst |*" not in out


def test_empty_skills_and_certifications_are_filled_from_source():
    """An empty section is not a faithful resume when the source has content."""
    out, changes = sp.preserve_source_facts(DRAFT, _frozen_source())
    assert any("filled an empty skills section" in c for c in changes)
    assert any("filled an empty certifications section" in c for c in changes)
    for skill in ("Spreadsheet modelling, SQL", "Invoice reconciliation",
                  "Parts inventory, stock level planning", "Records accuracy"):
        assert skill in out, f"skill lost: {skill}"
    assert "Certified Production and Inventory Management (CPIM)" in out


def test_the_candidates_real_bullets_survive_the_repair():
    """Restoring records must not delete the writer's faithful bullets."""
    out, _ = sp.preserve_source_facts(DRAFT, _frozen_source())
    for fact in ("1,180 fuel invoices", "14 depots", "7 service branches",
                 "2 branches", "3 years", "9 years", "previous four quarters"):
        assert fact in out, f"supported fact lost: {fact}"


def test_a_real_employer_name_is_not_overwritten():
    """Paired control: the rule acts only on placeholders, not real names."""
    source = ("## Professional Experience\n"
              "**SYNTHETIC-ONLY Alpha Corp** -- City, ST\n"
              "*Technician | 01/2020 - Present*\n"
              "- Led 4 projects.\n")
    draft = ("## Professional Experience\n"
             "**Beta Corp** -- Other City, ST\n"
             "*Engineer | 02/2021 - Present*\n"
             "- Led 4 projects.\n")
    out, changes = sp.preserve_source_facts(draft, source)
    assert "Beta Corp" in out, "a real employer name was overwritten"
    assert "restored the source's employer and role records" not in changes


def test_ambiguous_record_count_is_left_for_review():
    """Two draft records vs three source records: refuse to guess a mapping."""
    source = _frozen_source()
    draft = DRAFT.replace("**Current Employer** --\n*Analyst |*\n", "", 1)
    out, changes = sp.preserve_source_facts(draft, source)
    assert "restored the source's employer and role records" not in changes
    assert "Current Employer" in out


def test_the_retained_capture_matches_the_inline_draft():
    """Optional: prove the inline fixture is the real observed output."""
    if not CAPTURE.exists():
        return
    data = json.loads(CAPTURE.read_text())
    assert data["draft"].strip() == DRAFT.strip(), (
        "the inline draft no longer matches the retained capture")
