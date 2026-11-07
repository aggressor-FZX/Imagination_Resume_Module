"""Appended purpose clauses are corrected, not merely approved.

Measured on the approved Conservative website run ``845e8e80`` (2026-11-07,
revision ``d2b9578``). The reviewer marked all 31 lines ``supported`` and the
export succeeded, yet four bullets carried a trailing purpose phrase no source
states:

  source  "Built the depot scorecard that supervisors review in the Monday
           operations meeting."
  output  "Built depot scorecard used weekly by supervisors in operations
           meetings to assess performance."

  source  "Prepared the monthly cost summary reviewed by the regional manager."
  output  "Prepared monthly cost summaries reviewed by regional manager for
           budget oversight."

  source  "Rebuilt the shipment delay log that the dispatch team relies on."
  output  "Rebuilt shipment delay log used daily by dispatch team to track
           delivery disruptions."

  source  "Flagged slow-moving stock and proposed the reduction plan adopted
           for 2 branches."
  output  "Flagged slow-moving stock and proposed reduction plan adopted by 2
           branches to optimize inventory."

The previous rule only inspected the last comma, so none of these were seen.
These tests pin the corrected behaviour on the captured draft and the paired
refusals that keep the rule from becoming a general rewriter.
"""
from pathlib import Path

import pytest

import source_preservation as sp

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bridge_opportunity"

# Verbatim draft from the capture, trimmed to the affected records.
CAPTURED_DRAFT = """## Professional Experience
**Harborline Transit Authority** -- Tacoma, WA
*Fuel Reporting Analyst | 04/2019 - Present*
- Tracked monthly fuel consumption across 14 depots and compared each month against the same month in the prior year.
- Built depot scorecard used weekly by supervisors in operations meetings to assess performance.
- Reconciled 1,180 fuel invoices against depot dispensing records each quarter.

**Cascade Freight Partners** -- Kent, WA
*Cost Reporting Coordinator | 07/2015 - 03/2019*
- Audited 3 years of maintenance cost records to identify equipment with repeated failures.
- Prepared monthly cost summaries reviewed by regional manager for budget oversight.
- Rebuilt shipment delay log used daily by dispatch team to track delivery disruptions.

**Ridgeline Equipment Services** -- Fife, WA
*Inventory Analyst | 08/2023 - Present*
- Maintained parts availability records supporting 7 service branches across regional operations.
- Flagged slow-moving stock and proposed reduction plan adopted by 2 branches to optimize inventory.
- Compared quarterly parts usage against the prior four quarters to set reorder points.
"""

UNSUPPORTED = (
    "to assess performance",
    "for budget oversight",
    "to track delivery disruptions",
    "to optimize inventory",
)


@pytest.fixture(scope="module")
def source():
    return "\n\n---\n\n".join(
        (FIXTURE / name).read_text()
        for name in ("candidate-main.txt", "candidate-supplement.txt"))


def test_captured_purpose_clauses_are_removed(source):
    out, changes = sp.preserve_source_facts(CAPTURED_DRAFT, source)
    for clause in UNSUPPORTED:
        assert clause not in out, f"unsupported purpose clause survived: {clause}"
    assert changes.count("removed an unsupported appended clause") == 4


def test_captured_supported_facts_survive(source):
    out, _ = sp.preserve_source_facts(CAPTURED_DRAFT, source)
    for fact in ("14 depots", "1,180 fuel invoices", "3 years", "7 service branches",
                 "2 branches", "prior four quarters", "same month in the prior year"):
        assert fact in out, f"supported fact lost: {fact}"


def test_captured_employer_ownership_is_unchanged(source):
    out, _ = sp.preserve_source_facts(CAPTURED_DRAFT, source)
    for employer in ("Harborline Transit Authority", "Cascade Freight Partners",
                     "Ridgeline Equipment Services"):
        assert employer in out
    # Each employer's own bullet stays under its own heading.
    harborline = out.split("**Cascade Freight Partners**")[0]
    assert "1,180 fuel invoices" in harborline
    assert "7 service branches" not in harborline
    ridgeline = out.split("**Ridgeline Equipment Services**")[1]
    assert "7 service branches" in ridgeline
    assert "1,180 fuel invoices" not in ridgeline


def test_captured_correction_does_not_repeat_clause_removal(source):
    """A second pass must not keep "correcting" the same bullets.

    The captured draft omits the Skills and Certifications sections, so a second
    pass legitimately fills those empty sections from the source. What must not
    recur is another clause removal.
    """
    once, _ = sp.preserve_source_facts(CAPTURED_DRAFT, source)
    twice, changes = sp.preserve_source_facts(once, source)
    assert "removed an unsupported appended clause" not in changes
    for clause in UNSUPPORTED:
        assert clause not in twice


def test_every_changed_bullet_is_the_candidates_own_wording(source):
    """A changed BULLET is source text or a prefix of the draft line.

    A prefix is a deletion of an unsupported trailing clause: it writes no new
    sentence, so it is not a second writer. Anything else would be composed prose.
    """
    out, _ = sp.preserve_source_facts(CAPTURED_DRAFT, source)
    flat_source = " ".join(source.replace("SYNTHETIC-ONLY", "").split())
    draft_lines = {" ".join(line.strip().lstrip("- ").split())
                   for line in CAPTURED_DRAFT.splitlines()}
    for line in out.splitlines():
        if not line.strip().startswith("-"):
            continue
        candidate = " ".join(line.strip().lstrip("- ").split())
        if not candidate or candidate in draft_lines:
            continue
        assert candidate in flat_source or any(
            draft_line.startswith(candidate) for draft_line in draft_lines), (
            f"preservation invented a bullet: {candidate!r}")


@pytest.mark.parametrize("draft,source_line", [
    # A purpose phrase that restates the source's own purpose is not removed.
    ("- Compared quarterly parts usage against the prior four quarters to set reorder points.",
     "- SYNTHETIC-ONLY Compared quarterly parts usage against the prior four quarters to set reorder points."),
    # A coordinated activity is not a purpose clause.
    ("- Maintained parts availability records for 7 service branches.",
     "- SYNTHETIC-ONLY Maintain parts availability records for 7 service branches."),
])
def test_supported_purpose_and_coordination_are_kept(draft, source_line):
    out, changes = sp.preserve_experience_bullets(draft, source_line)
    assert "removed an unsupported appended clause" not in changes
    assert draft.strip() in out


def test_an_invented_figure_in_the_tail_is_removed():
    """A tail figure the matched bullet never states is invented, not sourced.

    Measured on the approved Imaginative website run ``9b291b32`` (2026-11-07):
    "enabling supervisors to assess performance across 14 sites" was appended to
    a bullet that states no figure, and the invented "14 sites" reached an
    approved export because the transplant guard refused any tail figure.
    """
    source = "- SYNTHETIC-ONLY Reduced after-hours callouts by 27%.\n"
    draft = "- Reduced after-hours callouts by 27% to improve uptime by 40%.\n"
    out, changes = sp.preserve_experience_bullets(draft, source)
    assert "removed an unsupported appended clause" in changes
    assert "40%" not in out
    # The candidate's own figure survives.
    assert "27%" in out


def test_a_sourced_figure_only_in_the_tail_is_left_for_review():
    """Removing a tail figure the bullet DOES state would discard a fact."""
    source = ("- SYNTHETIC-ONLY Reduced after-hours callouts by 27% across "
              "6 commercial properties.\n")
    draft = ("- Reduced after-hours callouts by 27% to support 6 commercial "
             "properties.\n")
    out, changes = sp.preserve_experience_bullets(draft, source)
    assert "removed an unsupported appended clause" not in changes
    assert "6 commercial properties" in out


def test_a_tail_restating_source_content_is_not_dropped():
    """A tail that carries source content the prefix lacks is not removable."""
    source = ("- SYNTHETIC-ONLY Inspected electrical systems and maintained "
              "safety records.\n")
    draft = "- Inspected electrical systems to maintain safety records.\n"
    out, changes = sp.preserve_experience_bullets(draft, source)
    assert "removed an unsupported appended clause" not in changes
    assert "safety records" in out


def test_a_tail_word_found_only_elsewhere_in_the_source_is_still_removed():
    """The tail is judged against the matched bullet, not the whole document.

    Measured on Conservative run ``eae427ea``: the writer appended "for accuracy"
    to the reconciled-invoices bullet. "accuracy" occurs elsewhere in the source
    (the skills line "Records accuracy"), so a document-wide vocabulary check
    found nothing ungrounded, left the addition in place, and the reviewer then
    reported the line unconfirmed — the run withheld.
    """
    source = ("## Professional Experience\n"
              "**Acme** -- City, ST\n"
              "*Analyst | 2020 - 2021*\n"
              "- SYNTHETIC-ONLY Reconciled 1,180 fuel invoices against depot "
              "dispensing records each quarter.\n"
              "## Skills\n"
              "- SYNTHETIC-ONLY Records accuracy\n")
    draft = ("## Professional Experience\n"
             "**Acme** -- City, ST\n"
             "*Analyst | 2020 - 2021*\n"
             "- Reconciled 1,180 fuel invoices quarterly against depot dispensing "
             "records for accuracy.\n")
    out, changes = sp.preserve_experience_bullets(draft, source)
    assert "removed an unsupported appended clause" in changes
    assert "for accuracy" not in out
    assert "1,180 fuel invoices" in out


# ---------------------------------------------------------------------------
# The source-word escape
# ---------------------------------------------------------------------------
# Measured on the Conservative acceptance batch of 2026-11-07 (runs
# ``dac8b3f5``, ``b2c9686f``, ``0365614d``). Three approved bullets carried an
# appended outcome whose tail ALSO echoed a word the source states. The old rule
# refused the whole tail because it contained a source word, so the unsupported
# outcome reached an approved export. Trimming instead would have dropped the
# sourced fact. Neither is faithful, so the candidate's own source bullet is
# substituted: every fact kept, the addition gone, nothing composed.

ESCAPE_SOURCE = (
    "## Professional Experience\n"
    "**Cascade Freight Partners** -- Kent, WA\n"
    "*Cost Reporting Coordinator | 07/2015 - 03/2019*\n"
    "- SYNTHETIC-ONLY Prepared the monthly cost summary reviewed by the regional "
    "manager.\n"
    "- SYNTHETIC-ONLY Rebuilt the shipment delay log that the dispatch team "
    "relies on.\n"
    "\n"
    "**Ridgeline Equipment Services** -- Fife, WA\n"
    "*Inventory Analyst | 08/2023 - Present*\n"
    "- SYNTHETIC-ONLY Maintain parts availability records for 7 service "
    "branches.\n"
)


def test_a_bare_participle_tail_mixing_a_source_fact_is_substituted():
    """No comma, and the tail carries a sourced fact plus an unsupported outcome.

    "…cost summaries enabling regional manager oversight of key operational
    expenses" — "regional manager" is in the source, "oversight of key
    operational expenses" is not. Trimming would drop the sourced fact; keeping
    would publish the outcome.
    """
    draft = ("## Professional Experience\n"
             "**Cascade Freight Partners** -- Kent, WA\n"
             "*Cost Reporting Coordinator | 07/2015 - 03/2019*\n"
             "- Prepared monthly cost summaries enabling regional manager "
             "oversight of key operational expenses.\n")
    out, changes = sp.preserve_experience_bullets(draft, ESCAPE_SOURCE)
    assert "restored the source bullet for a mixed clause" in changes
    assert "oversight of key operational expenses" not in out
    # The sourced fact survives, because the source bullet carries it.
    assert "regional manager" in out
    assert "Prepared the monthly cost summary reviewed by the regional manager." in out


def test_a_comma_tail_mixing_a_source_fact_is_substituted():
    """The comma form of the same escape, from run ``b2c9686f``."""
    draft = ("## Professional Experience\n"
             "**Cascade Freight Partners** -- Kent, WA\n"
             "*Cost Reporting Coordinator | 07/2015 - 03/2019*\n"
             "- Prepared monthly cost summaries for regional manager review, "
             "enabling data-driven decision-making by supervisors.\n")
    out, changes = sp.preserve_experience_bullets(draft, ESCAPE_SOURCE)
    assert "enabling data-driven decision-making by supervisors" not in out
    assert "regional manager" in out


def test_a_bare_participle_tail_with_no_source_fact_is_trimmed():
    """A purely foreign tail is still trimmed, preserving the writer's prefix."""
    draft = ("## Professional Experience\n"
             "**Cascade Freight Partners** -- Kent, WA\n"
             "*Cost Reporting Coordinator | 07/2015 - 03/2019*\n"
             "- Prepared monthly cost summaries enabling data-driven "
             "decision-making by supervisors.\n")
    out, changes = sp.preserve_experience_bullets(draft, ESCAPE_SOURCE)
    assert "removed an unsupported appended clause" in changes
    assert "data-driven decision-making" not in out
    assert "Prepared monthly cost summaries" in out


def test_a_bare_participle_noun_phrase_is_not_split():
    """An ordinary noun phrase after a participle is not an appended outcome.

    "…records supporting 7 service branches" states a sourced figure and adds no
    ungrounded content, so the line is left exactly as written.
    """
    draft = ("## Professional Experience\n"
             "**Ridgeline Equipment Services** -- Fife, WA\n"
             "*Inventory Analyst | 08/2023 - Present*\n"
             "- Maintained parts availability records supporting 7 service "
             "branches through accurate inventory tracking.\n")
    out, changes = sp.preserve_experience_bullets(draft, ESCAPE_SOURCE)
    assert changes == []
    assert "supporting 7 service branches" in out


def test_a_faithful_restatement_of_a_source_figure_is_not_substituted():
    """A tail restating the source's own outcome is a paraphrase, not an addition.

    "Optimized panel labeling …, cutting rework by 18%" restates the source's
    "Cut rework by 18% by standardizing panel labeling". The figure guard keeps
    this class out of the substitution.
    """
    source = ("## Professional Experience\n"
              "**Acme** -- City, ST\n"
              "*Electrician | 2020 - 2021*\n"
              "- SYNTHETIC-ONLY Cut rework by 18% by standardizing panel "
              "labeling and inspection checklists.\n")
    draft = ("## Professional Experience\n"
             "**Acme** -- City, ST\n"
             "*Electrician | 2020 - 2021*\n"
             "- Optimized panel labeling and inspection workflows, cutting "
             "rework by 18% across multiple commercial projects.\n")
    out, changes = sp.preserve_experience_bullets(draft, source)
    assert "restored the source bullet for a mixed clause" not in changes
    assert "cutting rework by 18%" in out
