"""Offline Loader-to-preservation checks; no model, network, or fixture writes.

The frozen DOCX files are read directly. The two-page synthetic PDF is created
only in pytest's temporary directory. These tests are not claims about the
separate semantic reviewer or the complete pipeline.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import fitz
import pytest

import source_preservation as sp


ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT.parents[2] / "Loader" / "modules" / "extractors.py"
PACK = ROOT / "tests" / "fixtures" / "bridge_opportunity"
DOCX_HASHES = {
    "candidate-main.docx": "7883ee26ab0dc57f8493be8856317d84e3a5c9bed3a48f123715347c9ef7a2b9",
    "candidate-supplement.docx": "d2263b6311f501da6e925ab401cb279ab4ed2d72ec53b666f11a40548bacd8b9",
}

spec = importlib.util.spec_from_file_location("offline_loader_extractors", LOADER)
assert spec and spec.loader, f"Loader extractor not found: {LOADER}"
extractors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractors)


@pytest.fixture(scope="module")
def docx_sources():
    texts = []
    for name, expected in DOCX_HASHES.items():
        path = PACK / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        result = extractors.extract_from_word(str(path))
        assert result["method"] == "python-docx", result
        assert result["confidence"] == "high" and not result["warnings"], result
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        texts.append(result["text"])
    return texts


@pytest.fixture(scope="module")
def docx_source(docx_sources):
    return "\n\n---\n\n".join(docx_sources)


@pytest.fixture
def pdf_source(tmp_path):
    # Plain text, two pages and realistic section/record boundaries. This is a
    # digitally generated PDF, not an OCR, table, or complex-layout test.
    pages = [
        ["Professional Summary", "Maintenance electrician with twelve years of experience.",
         "Professional Experience", "Alpha Electric -- Tacoma, WA",
         "Electrician | 01/2015 - 12/2019",
         "Completed 8,000 hours of supervised field work on electrical installations.",
         "Performed maintenance on distribution equipment."],
        ["Professional Experience", "Beta Power -- Kent, WA",
         "Technician | 01/2020 - Present",
         "Inspected electrical systems and maintained safety records.",
         "Compared quarterly inspections against the prior four quarters to set maintenance priorities.",
         "Skills", "Electrical distribution, safety inspection", "Lockout/tagout"],
    ]
    path = tmp_path / "synthetic-candidate.pdf"
    with fitz.open() as doc:
        for lines in pages:
            page = doc.new_page()
            for index, line in enumerate(lines):
                page.insert_text((48, 55 + index * 24), line, fontsize=11)
        doc.save(str(path))
    result = extractors.extract_from_pdf(str(path))
    assert result["method"] == "pymupdf", result
    assert result["confidence"] == "very_high" and result["page_count"] == 2, result
    assert not result["warnings"], result
    return result["text"]


def _draft(employer, location, role, bullet, skills=""):
    return ("## Professional Experience\n"
            f"**{employer}** -- {location}\n*{role}*\n- {bullet}\n"
            f"\n## Technical Skills\n{skills}\n")


def test_docx_extraction_returns_both_candidates_and_plain_text(docx_sources):
    main, supplement = docx_sources
    assert "Harborline Transit Authority -- Tacoma, WA" in main
    assert "Fuel Reporting Analyst | 04/2019 - Present" in main
    assert "1,180 fuel invoices" in main and "Spreadsheet modelling, SQL" in main
    assert "Ridgeline Equipment Services -- Fife, WA" in supplement
    assert "prior four quarters" in supplement and "Parts inventory" in supplement
    assert not any(line.lstrip().startswith(("#", "-", "*"))
                   for text in docx_sources for line in text.splitlines())


def test_pdf_extraction_has_both_pages_and_resume_facts(pdf_source):
    for fact in ("Alpha Electric -- Tacoma, WA", "8,000 hours of supervised field work",
                 "Beta Power -- Kent, WA", "Electrical distribution, safety inspection"):
        assert fact in pdf_source
    assert pdf_source.index("Alpha Electric") < pdf_source.index("Beta Power")


def test_frozen_docx_empty_skills_restore_both_documents(docx_source):
    draft = _draft("Harborline Transit Authority", "Tacoma, WA",
                   "Fuel Reporting Analyst | 04/2019 - Present",
                   "Reconciled 1,180 fuel invoices against depot dispensing records each quarter.")
    output, changes = sp.preserve_source_facts(draft, docx_source)
    for fact in ("Spreadsheet modelling, SQL", "Invoice reconciliation",
                 "Parts inventory, stock level planning", "Records accuracy"):
        assert fact in output
    assert any("filled an empty skills section" in change for change in changes)
    assert "SYNTHETIC-ONLY" not in output


def test_docx_ownership_can_be_restored_from_exact_role_and_dates(docx_source):
    draft = _draft("Current Employer", "Tacoma, WA",
                   "Fuel Reporting Analyst | 04/2019 - Present",
                   "Reconciled 1,180 fuel invoices against depot dispensing records each quarter.")
    output, changes = sp.preserve_source_facts(draft, docx_source)
    assert "**Harborline Transit Authority** -- Tacoma, WA" in output
    assert "restored the source's employer and role records" in changes


def test_docx_exact_bullet_alone_identifies_placeholder(docx_source):
    bullet = "Reconciled 1,180 fuel invoices against depot dispensing records each quarter."
    draft = _draft("Current Employer", "Tacoma, WA", "Analyst |", bullet)
    output, changes = sp.preserve_source_facts(draft, docx_source)
    assert bullet in docx_source and bullet in output
    assert "**Harborline Transit Authority**" in output
    assert "restored the source's employer and role records" in changes


def test_docx_exact_bullet_should_identify_placeholder(docx_source):
    draft = _draft("Current Employer", "Tacoma, WA", "Analyst |",
                   "Reconciled 1,180 fuel invoices against depot dispensing records each quarter.")
    output, _ = sp.preserve_source_facts(draft, docx_source)
    assert "**Harborline Transit Authority** -- Tacoma, WA" in output


def test_pdf_quantified_qualifier_restored_under_correct_employer(pdf_source):
    draft = _draft("Alpha Electric", "Tacoma, WA", "Electrician | 01/2015 - 12/2019",
                   "Completed 8,000 hours of hands-on field work on electrical installations.")
    output, changes = sp.preserve_source_facts(draft, pdf_source)
    assert "8,000 hours of supervised field work" in output
    assert "hands-on" not in output
    assert "restored the source wording for a quantified bullet" in changes


def test_pdf_does_not_insert_other_employers_quantified_fact(pdf_source):
    bullet = "Completed 8,000 hours of hands-on field work on electrical installations."
    draft = _draft("Beta Power", "Kent, WA", "Technician | 01/2020 - Present", bullet)
    output, changes = sp.preserve_source_facts(draft, pdf_source)
    assert bullet in output
    assert "supervised field work" not in output
    assert "restored the source wording for a quantified bullet" not in changes
    # Refusal to substitute does not mean the transplanted claim was validated.


def test_pdf_empty_skills_restore_extracted_facts(pdf_source):
    draft = _draft("Beta Power", "Kent, WA", "Technician | 01/2020 - Present",
                   "Inspected electrical systems and maintained safety records.")
    output, _ = sp.preserve_source_facts(draft, pdf_source)
    assert "Electrical distribution, safety inspection" in output
    assert "Lockout/tagout" in output


def test_pdf_missing_four_quarters_should_be_restored(pdf_source):
    draft = _draft("Beta Power", "Kent, WA", "Technician | 01/2020 - Present",
                   "Compared quarterly inspections against historical records to set maintenance priorities.")
    output, _ = sp.preserve_source_facts(draft, pdf_source)
    assert "prior four quarters" in output


def test_pdf_unsupported_appended_clause_should_be_removed(pdf_source):
    draft = _draft("Beta Power", "Kent, WA", "Technician | 01/2020 - Present",
                   "Inspected electrical systems and maintained safety records, ensuring flawless regulatory compliance.")
    output, _ = sp.preserve_source_facts(draft, pdf_source)
    assert "flawless regulatory compliance" not in output


@pytest.mark.parametrize("suffix,removed", [
    ("applying trend analysis and ensuring flawless regulatory compliance.", True),
    ("applying trend analysis.", False),
])
def test_pdf_bridge_survives_with_or_without_separable_outcome(pdf_source, suffix, removed):
    draft = _draft("Beta Power", "Kent, WA", "Technician | 01/2020 - Present",
                   "Inspected electrical systems and maintained safety records, " + suffix)
    output, changes = sp.preserve_source_facts(draft, pdf_source, ["trend analysis"])
    assert "applying trend analysis" in output
    assert "flawless regulatory compliance" not in output
    assert ("removed an unsupported appended clause" in changes) is removed


def test_docx_missing_qualifier_and_unsupported_clause_are_repaired(docx_source):
    missing = "Compared quarterly parts usage against historical records to set reorder points."
    unsupported = ("Reconciled 1,180 fuel invoices against depot dispensing records each quarter, "
                   "ensuring flawless regulatory compliance.")
    draft = (_draft("Harborline Transit Authority", "Tacoma, WA",
                    "Fuel Reporting Analyst | 04/2019 - Present", unsupported)
             + _draft("Ridgeline Equipment Services", "Fife, WA",
                      "Inventory Analyst | 08/2023 - Present", missing))
    output, changes = sp.preserve_source_facts(draft, docx_source)
    assert "prior four quarters" in docx_source
    assert "flawless regulatory compliance" not in docx_source
    assert missing not in output and unsupported not in output
    assert "prior four quarters" in output
    assert "flawless regulatory compliance" not in output
    assert "restored a dropped material qualifier" in changes
    assert "removed an unsupported appended clause" in changes


def test_docx_missing_four_quarters_should_be_restored(docx_source):
    draft = _draft("Ridgeline Equipment Services", "Fife, WA", "Inventory Analyst | 08/2023 - Present",
                   "Compared quarterly parts usage against historical records to set reorder points.")
    output, _ = sp.preserve_source_facts(draft, docx_source)
    assert "prior four quarters" in output


def test_docx_unsupported_appended_clause_should_be_removed(docx_source):
    draft = _draft("Harborline Transit Authority", "Tacoma, WA", "Fuel Reporting Analyst | 04/2019 - Present",
                   "Reconciled 1,180 fuel invoices against depot dispensing records each quarter, ensuring flawless regulatory compliance.")
    output, _ = sp.preserve_source_facts(draft, docx_source)
    assert "flawless regulatory compliance" not in output


def test_marked_source_controls_prove_repairs_are_sensitive_to_extracted_shape(docx_source):
    # Change only the in-memory extracted text, never the frozen documents.
    lines = docx_source.splitlines()
    for index, line in enumerate(lines):
        if "Reconciled 1,180 fuel invoices" in line or "Compared quarterly parts usage" in line:
            lines[index] = "- " + line
    marked = "\n".join(lines)
    missing = _draft("Ridgeline Equipment Services", "Fife, WA", "Inventory Analyst | 08/2023 - Present",
                     "Compared quarterly parts usage against historical records to set reorder points.")
    unsupported = _draft("Harborline Transit Authority", "Tacoma, WA", "Fuel Reporting Analyst | 04/2019 - Present",
                         "Reconciled 1,180 fuel invoices against depot dispensing records each quarter, ensuring flawless regulatory compliance.")
    missing_out, missing_changes = sp.preserve_source_facts(missing, marked)
    unsupported_out, unsupported_changes = sp.preserve_source_facts(unsupported, marked)
    assert "prior four quarters" in missing_out
    assert "restored a dropped material qualifier" in missing_changes
    assert "flawless regulatory compliance" not in unsupported_out
    assert "removed an unsupported appended clause" in unsupported_changes