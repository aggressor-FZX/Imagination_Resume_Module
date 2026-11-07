"""An education record must bold the institution, not the degree.

Measured on Conservative website run ``94c209ae`` (2026-11-07, revision
``4a90e44``). The writer emitted:

  **Associate of Applied Science, Business Analytics** -- Puget Sound Technical
  College -- 01/2015

The institution and the field are both present, so the presence checks passed
and the record was left alone. But the degree is bolded as if it were the
institution, and the reviewer reported the education record unconfirmed, so the
run withheld as ``source_completeness_uncertain``.

The source states the institution as the head of its own line. A record whose
institution is not its bold head is mis-shaped, not preserved.
"""
from pathlib import Path

import pytest

import source_preservation as sp

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bridge_opportunity"


@pytest.fixture(scope="module")
def source():
    return "\n\n---\n\n".join(
        (FIXTURE / name).read_text()
        for name in ("candidate-main.txt", "candidate-supplement.txt"))


MIS_SHAPED = """## Education
**Associate of Applied Science, Business Analytics** -- Puget Sound Technical College -- 01/2015
**Certificate, Inventory Control** -- Inland Records Institute -- 06/2022
"""

CORRECT = """## Education
**Puget Sound Technical College** -- Tacoma, WA
*Associate of Applied Science, Business Analytics | 2015*

**Inland Records Institute** -- Fife, WA
*Certificate, Inventory Control | 2022*
"""


def test_a_degree_bolded_as_the_institution_is_restored(source):
    out, changes = sp.preserve_source_facts(MIS_SHAPED, source)
    assert "restored the source's education record (degree, field, institution, date)" in changes
    assert "**Puget Sound Technical College**" in out
    assert "**Associate of Applied Science, Business Analytics**" not in out


def test_the_restored_record_keeps_every_source_value(source):
    out, _ = sp.preserve_source_facts(MIS_SHAPED, source)
    for value in ("Puget Sound Technical College", "Tacoma, WA",
                  "Associate of Applied Science, Business Analytics", "2015",
                  "Inland Records Institute", "Fife, WA",
                  "Certificate, Inventory Control", "2022"):
        assert value in out, f"source value lost: {value}"


def test_a_correct_record_is_left_alone(source):
    out, changes = sp.preserve_source_facts(CORRECT, source)
    assert not any("education" in change for change in changes), changes
    assert out.count("**Puget Sound Technical College**") == 1


def test_a_missing_second_record_is_still_restored(source):
    draft = CORRECT.split("**Inland Records Institute**")[0]
    out, changes = sp.preserve_source_facts(draft, source)
    assert "restored the source's education record (degree, field, institution, date)" in changes
    assert "**Inland Records Institute**" in out
