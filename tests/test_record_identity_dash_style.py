"""Record identity must not depend on dash style.

Measured on the approved Conservative website run ``104b7f7a`` (2026-11-07,
revision ``95db56e``). The writer emitted the role line with an en dash
(``04/2019 – Present``) while the extracted source uses a hyphen
(``04/2019 - Present``). ``preserve_specific_experience_details`` compares the
draft record's detail line to the source record's detail line to establish
ownership before repairing a bullet. Raw string comparison made that check fail,
so the repair never ran and the delivered bullet kept a dropped comparison
basis:

  source  "Tracked monthly fuel consumption across 14 depots and compared each
           month against the same month in the prior year."
  output  "Tracked monthly fuel consumption across 14 depots; benchmarked each
           month against the prior year."

Dash style is formatting, not identity. These tests pin the normalization and
the paired refusals that keep it from merging genuinely different records.
"""
from pathlib import Path

import pytest

import source_preservation as sp

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bridge_opportunity"

# Verbatim extracted source shape from the capture: no Markdown markers.
EXTRACTED_SOURCE = """Professional Summary
SYNTHETIC-ONLY Operations analyst with 9 years of experience in depot reporting and fuel cost control across public transit operations.
Professional Experience
SYNTHETIC-ONLY Harborline Transit Authority -- Tacoma, WA
Fuel Reporting Analyst | 04/2019 - Present
SYNTHETIC-ONLY Tracked monthly fuel consumption across 14 depots and compared each month against the same month in the prior year.
SYNTHETIC-ONLY Reconciled 1,180 fuel invoices against depot dispensing records each quarter.
"""

# Verbatim draft from the capture, with the en dash the writer produced.
CAPTURED_DRAFT = """## Professional Experience
**Harborline Transit Authority** -- Tacoma, WA
*Fuel Reporting Analyst | 04/2019 \u2013 Present*
- Tracked monthly fuel consumption across 14 depots; benchmarked each month against the prior year.
- Reconciled 1,180 fuel invoices quarterly against depot dispensing records.
"""


def test_en_dash_role_line_still_identifies_the_record():
    out, changes = sp.preserve_source_facts(CAPTURED_DRAFT, EXTRACTED_SOURCE)
    assert "restored a same-month prior-year comparison" in changes
    assert "same month in the prior year" in out
    assert "benchmarked each month against the prior year" not in out


def test_supported_figures_survive_the_repair():
    out, _ = sp.preserve_source_facts(CAPTURED_DRAFT, EXTRACTED_SOURCE)
    for fact in ("14 depots", "1,180 fuel invoices"):
        assert fact in out


def test_employer_ownership_is_unchanged():
    out, _ = sp.preserve_source_facts(CAPTURED_DRAFT, EXTRACTED_SOURCE)
    assert "**Harborline Transit Authority**" in out


@pytest.mark.parametrize("dash", ["-", "\u2013", "\u2014", "\u2011"])
def test_every_dash_variant_normalizes_to_the_same_identity(dash):
    draft = CAPTURED_DRAFT.replace("\u2013", dash)
    out, changes = sp.preserve_source_facts(draft, EXTRACTED_SOURCE)
    assert "restored a same-month prior-year comparison" in changes


def test_a_different_date_range_is_still_a_different_record():
    """Normalization must not merge records that genuinely differ."""
    draft = CAPTURED_DRAFT.replace("04/2019 \u2013 Present", "04/2018 \u2013 Present")
    out, changes = sp.preserve_source_facts(draft, EXTRACTED_SOURCE)
    assert "restored a same-month prior-year comparison" not in changes
    assert "benchmarked each month against the prior year" in out


def test_a_different_role_is_still_a_different_record():
    draft = CAPTURED_DRAFT.replace("Fuel Reporting Analyst", "Senior Analyst")
    out, changes = sp.preserve_source_facts(draft, EXTRACTED_SOURCE)
    assert "restored a same-month prior-year comparison" not in changes
