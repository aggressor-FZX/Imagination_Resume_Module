"""Unit tests for deterministic source preservation.

These are the focused checks behind ``source_preservation``. The route-level proof lives
in ``test_route_source_preservation.py``; this file isolates each rule so a failure names
the rule rather than "the route".

Every case is driven by the delivered run-7 draft and the answer-key source, so the
before/after pairs are the real defect, not an author's invention.
"""
from __future__ import annotations

import pytest

import live_capture
import source_preservation as sp
from tests.fixtures.fidelity import expected_facts as facts

RUN7 = "/home/skystarved/Render_Dockers/.live-run-2026-11-02/captures/" \
       "run7/0a267e04-affd-4cee-b944-26c574703039.json"

requires_capture = pytest.mark.skipif(
    not __import__("pathlib").Path(RUN7).exists(), reason="run-7 capture absent")

SOURCE = facts.SOURCE_TEXT


def _draft() -> str:
    return live_capture.load(RUN7)["draft"]


# ---------------------------------------------------------------------------
# 1. Education record
# ---------------------------------------------------------------------------

@requires_capture
def test_education_record_is_restored_with_all_four_parts():
    """Degree, field, institution and date survive as one record.

    Before: ``**Associate of Applied Science, Example Technical College** -- 2016``
            — the bold span concatenates degree and institution and the field is gone.
    After:  the source's own two-line record.
    """
    out, changes = sp.preserve_source_facts(_draft(), SOURCE)

    assert "restored the source's education record" in " ".join(changes)

    section = sp._find_section(sp._split_sections(out), "education")
    assert section is not None
    body = "\n".join(section["lines"])

    assert "**Example Technical College**" in body
    assert "*Associate of Applied Science, Electrical Systems Technology | 2016*" in body
    # The concatenation is gone: the bold span is the institution alone.
    for bold in [line.split("**")[1] for line in section["lines"] if "**" in line]:
        assert bold.replace("Example Technical College", "").strip(" ,-–") == ""


@requires_capture
def test_education_record_values_all_come_from_the_source():
    """No value is composed: each appears in the source text as written."""
    records = sp.source_education_records(SOURCE)
    assert len(records) == 1
    record = records[0]

    assert record["institution"] in SOURCE
    assert record["degree"] in SOURCE
    assert record["field"] in SOURCE
    assert record["dates"] in SOURCE
    assert record["detail_line"] in SOURCE


def test_education_is_left_alone_when_the_source_has_no_recognisable_record():
    """A source without a recognisable Education record must not be given one.

    The education text is preserved verbatim. Headings ARE inserted for the other
    promised sections, which is correct and is why this asserts on the education line
    rather than on equality with the input.
    """
    markdown = "## Education\n**Some School** -- 2019\n"
    out, changes = sp.preserve_source_facts(markdown, "## Experience\n- Did work.\n")
    assert "**Some School** -- 2019" in out
    assert not any("education record" in c for c in changes), (
        "education must not be rewritten when the source states no record")


def test_education_is_not_rewritten_when_already_correct():
    """A correct record is left exactly as it is.

    Asserted on the education section rather than on the whole document: other promised
    headings are legitimately inserted here, and conflating the two would make this test
    about heading insertion instead of about the education rule.
    """
    source = ("## Education\n"
              "**Example Technical College** -- Example City, WA\n"
              "*Associate of Applied Science, Electrical Systems Technology | 2016*\n")
    markdown = source
    out, changes = sp.preserve_source_facts(markdown, source)

    assert not any("education record" in c for c in changes)
    section = sp._find_section(sp._split_sections(out), "education")
    assert "\n".join(section["lines"]).strip() == "\n".join(
        sp._find_section(sp._split_sections(markdown), "education")["lines"]).strip()


# ---------------------------------------------------------------------------
# 2. Quantified qualifier
# ---------------------------------------------------------------------------

@requires_capture
def test_supervised_qualifier_is_restored_and_mastery_removed():
    """``8,000 hours of supervised field work`` keeps ``supervised``.

    Before: ``- Accumulated 8,000 hours of hands-on field experience toward
             journeyman licensure, mastering core electrical installations.``
    After:  the source line, verbatim.
    """
    out, changes = sp.preserve_source_facts(_draft(), SOURCE)

    assert "restored the source wording for a quantified bullet" in changes

    line = next(l for l in out.splitlines() if "8,000" in l)
    assert "supervised field work" in line
    assert "hands-on" not in line
    assert "mastering core electrical installations" not in line
    assert "journeyman licensure" not in line
    # And the replacement is the source's own sentence.
    assert line.strip().lstrip("- ").strip() in SOURCE


@requires_capture
def test_a_faithful_paraphrase_is_not_rewritten():
    """The rule must not revert legitimate paraphrase.

    Segments 6 and 8 of this source are classified by the answer key as
    "accepted, not missing": the writer's "Led installation and maintenance ..." and
    "Optimized panel labeling ... cutting rework by 18%" keep every quantified fact.
    An earlier version of this module rewrote both, which is a different defect from the
    one being fixed.
    """
    out, _changes = sp.preserve_source_facts(_draft(), SOURCE)

    assert "Led installation and maintenance of commercial electrical systems" in out
    assert "cutting rework by 18%" in out
    assert "Trained 6 apprentices in lockout/tagout procedures" in out


@requires_capture
def test_dates_and_role_headers_are_never_treated_as_quantities():
    """``03/2020`` and ``2016`` are records, not measured facts.

    The first version of this module accepted any digit run, so it treated every date as
    a quantified fact and rewrote the role headers and the summary — silent corruption of
    the fields the other rules exist to protect.
    """
    for fact in sp.source_quantified_facts(SOURCE):
        assert "/" not in fact["value"], fact["value"]
        assert fact["unit"], "a quantity must carry a unit"

    out, _changes = sp.preserve_source_facts(_draft(), SOURCE)
    for header in ("*Journeyman Electrician | 03/2020 – Present*",
                   "*Electrical Apprentice | 06/2016 – 02/2020*"):
        assert header in out, f"role header was rewritten: {header}"


def test_qualifier_window_does_not_reach_across_clauses():
    """The qualifier set is the words ADJACENT to the figure, not the whole sentence.

    A 42-character window reached from "8,000 hours" to "safe work practices" in the next
    clause, so a legitimate paraphrase read as a lost qualifier. The window is now 28
    characters and word-aligned.
    """
    facts_found = sp.source_quantified_facts(SOURCE)
    eight = next(f for f in facts_found if f["value"] == "8,000")
    assert "supervised" in eight["qualifiers"]
    assert "safe" not in eight["qualifiers"], (
        "the qualifier window reached into the next clause")
    # No word fragments from a mid-word slice.
    assert "ined" not in eight["qualifiers"]


# ---------------------------------------------------------------------------
# 3. Summary
# ---------------------------------------------------------------------------

@requires_capture
def test_summary_keeps_source_facts_and_drops_the_proficiency_list():
    """Before/after for the appended proficiency list."""
    out, changes = sp.preserve_source_facts(_draft(), SOURCE)

    assert "trimmed the appended proficiency list from the summary" in changes

    section = sp._find_section(sp._split_sections(out), "professional summary")
    text = " ".join(section["lines"])
    for fact in ("Journeyman electrician", "8+ years", "commercial and industrial",
                 "construction", "facilities maintenance", "power distribution"):
        assert fact in text, f"source summary fact lost: {fact}"
    assert "proficient in" not in text
    assert "apprentice supervision" not in text


def test_summary_is_removed_when_the_source_has_none():
    """Summary content appears only when source-backed."""
    markdown = ("## Professional Summary\nInvented summary.\n\n"
                "## Experience\n- Did work.\n")
    out, changes = sp.preserve_source_facts(markdown, "## Experience\n- Did work.\n")
    assert "removed an unsourced professional summary" in changes
    assert "Invented summary" not in out


def test_no_summary_heading_is_inserted_when_the_source_has_none():
    """An empty summary heading would claim a summary the candidate never wrote."""
    out, _ = sp.preserve_source_facts(
        "## Experience\n- Did work.\n",
        "## Experience\n- Did work.\n## Education\n**School**\n")
    assert not sp._find_section(sp._split_sections(out), "professional summary")


# ---------------------------------------------------------------------------
# 4. Required headings
# ---------------------------------------------------------------------------

@requires_capture
def test_all_required_headings_are_present_and_projects_is_empty():
    """Six headings, and Projects carries nothing."""
    out, changes = sp.preserve_source_facts(_draft(), SOURCE)
    assert "added an empty Projects heading" in changes

    sections = sp._split_sections(out)
    names = [s["heading"] for s in sections if s["heading"]]
    for required in sp.REQUIRED_HEADINGS:
        assert required in names, f"heading missing: {required}"

    projects = sp._find_section(sections, "projects")
    assert "".join(projects["lines"]).strip() == "", (
        "Projects must be empty for a candidate with none")


def test_headings_are_inserted_in_the_required_order():
    markdown = "## Professional Experience\n- Did work.\n"
    out, _ = sp.preserve_source_facts(markdown, "## Professional Summary\nElec.\n")
    names = [s["heading"] for s in sp._split_sections(out) if s["heading"]]
    order = [n for n in sp.REQUIRED_HEADINGS if n in names]
    assert names == order, f"headings out of order: {names}"


def test_skill_heading_alias_is_renamed_not_duplicated():
    """The source's ``## Skills`` becomes ``## Technical Skills``, once."""
    out, _ = sp.preserve_source_facts(_draft(), SOURCE)
    names = [s["heading"] for s in sp._split_sections(out) if s["heading"]]
    assert names.count("Technical Skills") == 1
    assert "Skills" not in names


# ---------------------------------------------------------------------------
# Contract: substitution only, never composition
# ---------------------------------------------------------------------------

@requires_capture
def test_no_output_line_is_invented():
    """Every changed line is either source text or a heading.

    This is the module's whole safety claim: it substitutes the candidate's own words,
    removes an unsupported span, or emits a structural heading. It never writes prose.
    Putting a new sentence in a resume is the writer's job, and a post-process that
    composed one would be a second, unaudited writer.

    The comparison is against the WHOLE source, whitespace-collapsed, not against its
    individual lines. The summary paragraph is wrapped across two source lines, so a
    rebuilt summary line is correct text that matches no single source line — and an
    earlier version of this test called that an invention. Comparing against the source
    as a stream is the right granularity for "is this the candidate's own wording".
    """
    draft, source = _draft(), SOURCE
    out, _ = sp.preserve_source_facts(draft, source)

    def plain(text: str) -> str:
        """Drop markdown markup and the provenance marker, then collapse whitespace.

        Markup is formatting, not wording: the draft writes the institution as
        ``**Example Technical College**`` and the source writes
        ``**SYNTHETIC-ONLY Example Technical College**``. Comparing raw strings would
        call a faithful restoration an invention purely because the asterisks and the
        marker differ.
        """
        text = text.replace("SYNTHETIC-ONLY", "")
        text = text.replace("**", "").replace("*", "")
        text = text.lstrip("- ").strip()
        return " ".join(text.split())

    flat_source = plain(source)
    draft_lines = {plain(line) for line in draft.splitlines() if line.strip()}

    for line in out.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        candidate = plain(line)
        if not candidate or candidate in draft_lines:
            continue
        # A changed line must be the source's own wording, contiguously.
        assert candidate in flat_source, (
            f"preservation invented a line: {line.strip()!r}")


@requires_capture
def test_preservation_is_idempotent():
    """A second pass changes nothing, so the route cannot oscillate."""
    once, _ = sp.preserve_source_facts(_draft(), SOURCE)
    twice, changes = sp.preserve_source_facts(once, SOURCE)
    assert changes == []
    assert twice.strip() == once.strip()


def test_empty_or_missing_inputs_are_passed_through():
    assert sp.preserve_source_facts("", SOURCE) == ("", [])
    assert sp.preserve_source_facts("## Education\n", "") == ("## Education\n", [])
    assert sp.preserve_source_facts("## Education\n", "   ") == ("## Education\n", [])
