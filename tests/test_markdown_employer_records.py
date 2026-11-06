"""Regression for the 'Current Employer' collapse on the frozen fixture.

Measured on Conservative diagnostics 3abb7811 and c84a6a8d: the direct
``/analyze`` route supplies ``resume_text`` + ``source_documents`` but no
structured ``experience``. ``parse_experiences`` returned ZERO records for the
frozen Markdown fixture, so the pipeline fell back to a single
``"Current Employer"`` record. The writer then merged all three employers'
bullets under that one placeholder and the reviewer correctly reported the real
employers, roles and dates missing -- the run withheld as
``source_facts_missing``.

``_markdown_employer_records`` is a structural parse of the candidate's own
Markdown, used only when the existing parser finds nothing usable. It copies
values; it never generates them.
"""
from pathlib import Path

from imaginator_new_integration import _markdown_employer_records

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bridge_opportunity"


def _frozen_source() -> str:
    main = (FIXTURE / "candidate-main.txt").read_text()
    supp = (FIXTURE / "candidate-supplement.txt").read_text()
    return main + "\n\n---\n\n" + supp


def test_the_frozen_fixture_yields_three_distinct_employer_records():
    records = _markdown_employer_records(_frozen_source())
    assert len(records) == 3
    assert [r["company"] for r in records] == [
        "Harborline Transit Authority",
        "Cascade Freight Partners",
        "Ridgeline Equipment Services",
    ]


def test_each_record_keeps_its_own_role_dates_and_bullets():
    records = _markdown_employer_records(_frozen_source())
    assert [r["role"] for r in records] == [
        "Fuel Reporting Analyst", "Cost Reporting Coordinator", "Inventory Analyst"]
    assert [r["duration"] for r in records] == [
        "04/2019 - Present", "07/2015 - 03/2019", "08/2023 - Present"]
    assert all(len(r["bullets"]) == 3 for r in records)
    # No cross-employer bullet mixing.
    assert "1,180 fuel invoices" in records[0]["body"]
    assert "1,180 fuel invoices" not in records[1]["body"]
    assert "parts availability" in records[2]["body"]


def test_education_institutions_are_not_treated_as_employers():
    records = _markdown_employer_records(_frozen_source())
    names = " ".join(r["company"] for r in records)
    assert "Puget Sound Technical College" not in names
    assert "Inland Records Institute" not in names


def test_no_markdown_records_returns_empty_not_a_placeholder():
    """Paired control: the caller keeps its existing fallback when nothing parses."""
    assert _markdown_employer_records("Just a paragraph of prose.\n") == []
    assert _markdown_employer_records("") == []


def test_provenance_markers_are_stripped_from_company_names():
    records = _markdown_employer_records(_frozen_source())
    assert all("SYNTHETIC-ONLY" not in r["company"] for r in records)
