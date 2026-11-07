"""A lexically supported skills block can still omit the candidate's skills.

Measured on Conservative website run ``56bc8c88`` (2026-11-07, revision
``3dec9ad``). The writer emitted:

  ## Technical Skills
  **Analytics:** analytics

Every word of that term occurs in the source, so the existing support check
passed and the block was left alone. But all four of the candidate's own skills
were gone. The reviewer flagged the block, two repair passes could not fix it,
and the run withheld as ``unresolved_claims``.

A source skill the block does not carry is a missing fact, so the block is
replaced by the source's own list — the same replacement the unsupported case
performs. These tests pin that and the paired controls.
"""
from pathlib import Path

import pytest

import source_preservation as sp

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bridge_opportunity"

SOURCE_SKILLS = ("Spreadsheet modelling, SQL", "Invoice reconciliation",
                 "Parts inventory, stock level planning", "Records accuracy")


@pytest.fixture(scope="module")
def source():
    return "\n\n---\n\n".join(
        (FIXTURE / name).read_text()
        for name in ("candidate-main.txt", "candidate-supplement.txt"))


def _draft(skills_block):
    return ("## Professional Experience\n"
            "**Harborline Transit Authority** -- Tacoma, WA\n"
            "*Fuel Reporting Analyst | 04/2019 - Present*\n"
            "- Reconciled 1,180 fuel invoices against depot dispensing records each quarter.\n"
            "\n## Technical Skills\n" + skills_block + "\n")


def test_a_lexically_supported_block_that_omits_source_skills_is_replaced(source):
    out, changes = sp.preserve_source_facts(_draft("**Analytics:** analytics"), source)
    assert "replaced an unsupported skills block with the source's own skills" in changes
    body = out.split("## Technical Skills", 1)[1]
    for skill in SOURCE_SKILLS:
        assert skill in body, f"source skill missing: {skill}"
    assert "**Analytics:**" not in body


def test_a_complete_source_skills_block_is_left_alone(source):
    block = "\n".join(f"- {skill}" for skill in SOURCE_SKILLS)
    out, changes = sp.preserve_source_facts(_draft(block), source)
    assert not any("skills block" in change for change in changes), changes
    body = out.split("## Technical Skills", 1)[1]
    for skill in SOURCE_SKILLS:
        assert skill in body


def test_a_partially_complete_block_is_completed(source):
    out, changes = sp.preserve_source_facts(
        _draft("- Spreadsheet modelling, SQL"), source)
    assert "replaced an unsupported skills block with the source's own skills" in changes
    body = out.split("## Technical Skills", 1)[1]
    for skill in SOURCE_SKILLS:
        assert skill in body


def test_an_invented_term_is_still_replaced(source):
    out, changes = sp.preserve_source_facts(
        _draft("- Spreadsheet modelling, SQL\n- Kubernetes orchestration"), source)
    assert "replaced an unsupported skills block with the source's own skills" in changes
    assert "Kubernetes" not in out


def test_a_permitted_bridge_survives_a_complete_block(source):
    block = "\n".join(f"- {skill}" for skill in SOURCE_SKILLS) + "\n- trend analysis"
    out, changes = sp.preserve_source_facts(
        _draft(block), source, permitted_bridges=["trend analysis"])
    assert not any("skills block" in change for change in changes), changes
    assert "trend analysis" in out
