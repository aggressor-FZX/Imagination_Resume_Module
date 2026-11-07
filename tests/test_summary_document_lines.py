"""Each document's summary is written as its own line, not one merged line.

Measured on the Conservative website runs ``64145e6c`` and ``a09a004a``
(2026-11-07, revision ``ae354e7``). Preservation joined both documents'
summaries into a single paragraph:

  "Operations analyst with 9 years ... public transit operations. Operations
   analyst supporting stock planning ... service network."

That joined string is not a contiguous span of either source document, so the
reviewer could not cite it. It flagged the line ``uncertain`` with no evidence,
the repair correctly returned the verbatim text unchanged, and the run withheld
as ``no_op_repair`` and then ``source_completeness_uncertain``. Two consecutive
runs failed on the same line, so this is systematic, not reviewer variance.

Writing each document's paragraph on its own line keeps every line verbatim
source text. These tests pin that shape and the paired controls.
"""
from pathlib import Path

import pytest

import source_preservation as sp

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bridge_opportunity"

MAIN_SUMMARY = ("Operations analyst with 9 years of experience in depot reporting "
                "and fuel cost control across public transit operations.")
SUPPLEMENT_SUMMARY = ("Operations analyst supporting stock planning and parts "
                      "availability for a regional equipment service network.")


@pytest.fixture(scope="module")
def source():
    return "\n\n---\n\n".join(
        (FIXTURE / name).read_text()
        for name in ("candidate-main.txt", "candidate-supplement.txt"))


def test_summary_parts_are_separate(source):
    parts = sp.source_summary_parts(source)
    assert len(parts) == 2
    assert parts[0] == MAIN_SUMMARY
    assert parts[1] == SUPPLEMENT_SUMMARY


def test_joined_form_is_still_available_for_presence_checks(source):
    assert sp.source_summary(source) == f"{MAIN_SUMMARY} {SUPPLEMENT_SUMMARY}"


def test_restored_summary_uses_one_line_per_document():
    draft = "## Professional Experience\n**Acme** -- City, ST\n*Analyst | 2020 - 2021*\n"
    out, changes = sp.preserve_source_facts(draft, _source())
    assert "restored the source summary" in changes
    section = sp._find_section(sp._split_sections(out), "professional summary")
    lines = [line.strip() for line in section["lines"] if line.strip()]
    assert lines == [MAIN_SUMMARY, SUPPLEMENT_SUMMARY]


def test_each_summary_line_is_contiguous_source_text(source):
    draft = "## Professional Experience\n**Acme** -- City, ST\n*Analyst | 2020 - 2021*\n"
    out, _ = sp.preserve_source_facts(draft, source)
    flat = " ".join(source.replace("SYNTHETIC-ONLY", "").split())
    section = sp._find_section(sp._split_sections(out), "professional summary")
    for line in section["lines"]:
        text = " ".join(line.split())
        if text:
            assert text in flat, f"summary line is not source text: {text!r}"


def test_a_merged_draft_summary_is_replaced_by_separate_lines(source):
    merged = f"## Professional Summary\n{MAIN_SUMMARY} {SUPPLEMENT_SUMMARY}\n"
    out, _ = sp.preserve_source_facts(merged, source)
    section = sp._find_section(sp._split_sections(out), "professional summary")
    lines = [line.strip() for line in section["lines"] if line.strip()]
    assert lines == [MAIN_SUMMARY, SUPPLEMENT_SUMMARY]


def test_a_single_document_summary_is_unchanged():
    source = f"## Professional Summary\n{MAIN_SUMMARY}\n"
    draft = f"## Professional Summary\n{MAIN_SUMMARY}\n"
    out, changes = sp.preserve_source_facts(draft, source)
    assert not any("summary" in change for change in changes)
    assert MAIN_SUMMARY in out


def test_an_appended_proficiency_list_is_still_trimmed(source):
    draft = ("## Professional Summary\n"
             f"{MAIN_SUMMARY} {SUPPLEMENT_SUMMARY}, proficient in Kubernetes.\n")
    out, changes = sp.preserve_source_facts(draft, source)
    assert "trimmed the appended proficiency list from the summary" in changes
    assert "Kubernetes" not in out


def _source():
    return "\n\n---\n\n".join(
        (FIXTURE / name).read_text()
        for name in ("candidate-main.txt", "candidate-supplement.txt"))
