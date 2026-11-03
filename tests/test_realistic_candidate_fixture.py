"""Structural checks for the realistic candidate pack. Offline, no provider calls.

These are **self-checks on the fixture**, not acceptance tests. They exist because the
pack's stated value is that its traps are real: an ownership ledger nobody verified is a
claim, not a fixture. Each test asserts one property of the README's trap list against
the two source documents.

The pack is NOT frozen. Nothing here freezes it, hashes it for approval, or submits it.
When the owner approves it, the hashes get recorded and these tests stay valid because
they assert structure, not digests.

Labelled **[realistic-fixture]**.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import live_capture

HERE = Path(__file__).resolve().parent
PACK = HERE / "fixtures" / "realistic_candidate"
MAIN = PACK / "candidate-main.txt"
SUPPLEMENT = PACK / "candidate-supplement.txt"


def _main() -> str:
    return MAIN.read_text(encoding="utf-8")


def _supplement() -> str:
    return SUPPLEMENT.read_text(encoding="utf-8")


def _combined() -> str:
    # The orchestrator's join. Asserted in the request tests; used here for whole-pack
    # checks so the fixture is exercised the way the route will see it.
    return _main() + "\n\n---\n\n" + _supplement()


# ---------------------------------------------------------------------------
# Capture-safety compatibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["main", "supplement"])
def test_every_source_document_carries_the_synthetic_marker(text):
    """[realistic-fixture] The capture guard must accept this pack unchanged.

    ``live_capture.assert_synthetic()`` raises unless EVERY source document contains
    ``SYNTHETIC-ONLY``. Realistic cities are not a reason to relax that, so the pack
    keeps the marker on every factual line.
    """
    body = _main() if text == "main" else _supplement()
    assert live_capture.SYNTHETIC_MARKER in body
    # The guard's own function, not a re-implementation of it.
    live_capture.assert_synthetic([_main(), _supplement()])


def test_the_marker_is_on_factual_lines_and_not_only_in_a_header():
    """[realistic-fixture] A single header marker would be defeated by any section split.

    The convention, identical to the run-10 sources: every block-level factual line —
    employer, bullet, certification, skills — carries the marker, while the italic
    ``*Role | dates*`` / ``*Degree | year*`` sub-line does not. That sub-line is never
    the sole carrier of its fact: the labelled line above it is marked, so dropping a
    section cannot drop the provenance proof with it.

    Asserting the real convention rather than an idealised one keeps this fixture
    substitutable for the run-10 pack without touching the capture guard.
    """
    emphasis = 0
    for body in (_main(), _supplement()):
        for line in body.splitlines():
            stripped = line.strip(" -*")
            if not stripped or stripped.startswith("#"):
                continue
            if line.strip().startswith("*") and line.strip().endswith("*"):
                emphasis += 1
                continue
            assert live_capture.SYNTHETIC_MARKER in line, line
    # Non-vacuous: the sub-lines really are present and really are unmarked.
    assert emphasis == 6


def test_an_unmarked_document_is_still_refused():
    """[realistic-fixture] Pair for the test above: the guard is not disabled by this pack.

    The pack is compatible with the safeguard; it does not switch it off.
    """
    with pytest.raises(live_capture.LiveCaptureRefused):
        live_capture.assert_synthetic([
            _main(), _supplement().replace(live_capture.SYNTHETIC_MARKER, "")])


# ---------------------------------------------------------------------------
# Trap 1: two states
# ---------------------------------------------------------------------------

def test_the_two_states_are_distinct_and_correctly_split():
    """[realistic-fixture] Coeur d'Alene is ID; Spokane and Spokane Valley are WA.

    A location-based guess would move the current role to WA. The pack must make that
    guess wrong.
    """
    main, supp = _main(), _supplement()
    assert "-- Spokane, WA" in main
    assert "-- Spokane Valley, WA" in main
    assert "Spokane, WA" not in supp.split("## Professional Experience")[1]
    assert "-- Coeur d'Alene, ID" in supp


# ---------------------------------------------------------------------------
# Trap 2: two electrical education records
# ---------------------------------------------------------------------------

def test_the_two_education_records_differ_in_institution_level_and_year():
    """[realistic-fixture] AAS 2015 vs Certificate 2021, two institutions."""
    main, supp = _main(), _supplement()
    assert "Spokane Falls Institute of Technology" in main
    assert "Associate of Applied Science, Electrical Systems Technology | 2015" in main
    assert "Inland Northwest Safety Academy" in supp
    assert "Certificate, NFPA 70E Arc Flash Standards | 2021" in supp
    # Neither institution nor year leaks across documents.
    assert "Spokane Falls Institute of Technology" not in supp
    assert "Inland Northwest Safety Academy" not in main
    assert "| 2021" not in main


# ---------------------------------------------------------------------------
# Trap 3: three metrics, three owners, no movement
# ---------------------------------------------------------------------------

def _owner_of(needle: str) -> set[str]:
    """Employers whose record block contains ``needle``, per document structure."""
    text = _combined()
    blocks = re.split(r"\n\*\*SYNTHETIC-ONLY ", text)
    owners: set[str] = set()
    for block in blocks[1:]:
        employer = block.split("**")[0].strip()
        if needle in block:
            owners.add(employer)
    return owners


@pytest.mark.parametrize("needle,owner", [
    ("14 industrial contracts", "Northline Fabrication, Inc."),
    ("31 commercial buildouts", "Kettle Falls Electrical Contractors"),
    ("8,000 apprenticeship hours", "Kettle Falls Electrical Contractors"),
    ("9 substations", "Pend Oreille Power Systems"),
    ("4.5 hours to 2.75 hours", "Pend Oreille Power Systems"),
    ("4 apprentices", "Pend Oreille Power Systems"),
    ("6 commercial properties", "Riverbend Facilities Group"),
    ("after-hours callouts by 27%", "Riverbend Facilities Group"),
])
def test_every_metric_has_exactly_one_owner(needle, owner):
    """[realistic-fixture] Each figure appears under one employer and nowhere else.

    This is what makes a transplant detectable: if a figure had two owners, moving it
    would be indistinguishable from a legitimate restatement.
    """
    assert _owner_of(needle) == {owner}


def test_no_metric_appears_in_a_summary_or_skills_section():
    """[realistic-fixture] Figures belong in records, not in roll-ups.

    A metric restated in the summary has no employer attached, so a draft that lifts it
    there cannot be checked for ownership.
    """
    for body in (_main(), _supplement()):
        head = body.split("## Professional Experience")[0]
        assert not re.search(r"\d+%|\d+ (?:substations|contracts|buildouts|properties)",
                             head)


# ---------------------------------------------------------------------------
# Trap 4: the current employer lives only in the supplement
# ---------------------------------------------------------------------------

def test_the_current_employer_and_the_certificate_are_supplement_only():
    """[realistic-fixture] The single-document failure must be reproducible.

    A reader that takes only the first document loses the present role, the current
    employer, the certificate and the licence. That is the run-10 defect class.
    """
    main, supp = _main(), _supplement()
    for fact in ("Riverbend Facilities Group", "03/2023 - Present",
                 "Inland Northwest Safety Academy",
                 "Washington 01 Journey Level Electrician License"):
        assert fact in supp
        assert fact not in main


# ---------------------------------------------------------------------------
# Trap 5 and 6: near-miss figures
# ---------------------------------------------------------------------------

def test_the_two_apprentice_counts_are_different_numbers():
    """[realistic-fixture] 4 apprentices must not drift to 6.

    Run 10's sources said "6 apprentices". A writer carrying that habit into this pack
    would produce a figure error, and the pack must be able to expose it.
    """
    assert "4 apprentices" in _main()
    assert "6 apprentices" not in _combined()


def test_the_two_hour_values_are_both_present_so_the_reduction_is_unambiguous():
    """[realistic-fixture] 4.5 hours to 2.75 hours: both endpoints sourced."""
    main = _main()
    assert "4.5 hours" in main and "2.75 hours" in main


# ---------------------------------------------------------------------------
# Formatting realism
# ---------------------------------------------------------------------------

def test_the_documents_look_like_resumes_and_not_placeholder_data():
    """[realistic-fixture] Real cities, real-looking employers, no placeholder words.

    "Example City" and "Sampletown" are what the run-10 pack used; they cannot
    demonstrate location preservation.
    """
    text = _combined()
    for placeholder in ("Example City", "Sampletown", "Example College", "TODO"):
        assert placeholder not in text
    assert "Spokane" in text and "Coeur d'Alene" in text
    for section in ("## Professional Summary", "## Professional Experience",
                    "## Education", "## Skills"):
        assert section in text


def test_the_contact_details_are_test_only_and_unroutable():
    """[realistic-fixture] Documented, and reserved so they cannot deliver mail."""
    readme = (PACK / "README.md").read_text(encoding="utf-8")
    assert "example.invalid" in readme
    assert "555-0147" in readme


def test_the_chronology_is_consistent_and_has_no_overlap_or_gap():
    """[realistic-fixture] Employment periods tile the timeline in order."""
    expected = [
        ("03/2012", "09/2015"), ("10/2015", "06/2019"),
        ("07/2019", "02/2023"), ("03/2023", "Present"),
    ]
    rows = re.findall(r"\*[^*|]+\|\s*(\d{2}/\d{4})\s*-\s*(\d{2}/\d{4}|Present)", _combined())
    assert rows == expected
