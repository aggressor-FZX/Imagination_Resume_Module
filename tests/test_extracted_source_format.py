"""Regressions for the extracted-`.docx` source format. Portable, offline.

Why this file exists
--------------------

Run 12 (`83e2eb3f-…`) submitted two real `.docx` files. The text extracted from them carries
**no Markdown markers at all** — a `.docx` stores a section title as a styled paragraph and
an employer as plain text, so the source arrived as::

    Professional Summary
    SYNTHETIC-ONLY Journeyman electrician with 12 years of electrical experience spanning
    ...
    SYNTHETIC-ONLY Northline Fabrication, Inc. -- Spokane, WA
    Welder / Maintenance Helper | 03/2012 - 09/2015

Every preservation rule was written against hand-written Markdown fixtures that carry
``##``, ``**`` and ``*``. On the extracted shape:

* ``_split_sections`` matched no heading, so the document was one section with
  ``heading=None`` and ``source_summary()`` returned ``""``;
* ``_education_records_in`` required ``**Institution**`` and ``*Detail | Year*``, so both
  degrees vanished;
* ``_employer_of`` required a bold employer line, so it returned ``""`` for every line and
  **silently disabled the cross-employer attribution guard**.

The engine then withheld with ``source_facts_missing`` for facts the source plainly stated.

What this file asserts, in the order it is asked for
---------------------------------------------------

1. Section recognition — a closed allowlist, whole-line, marked or bare; no arbitrary
   title-case line is a heading.
2. Summary restoration — section absent, and section present but empty.
3. Education and ownership after formatting is stripped.
4. The demonstrated source-only cleanup (``motor controls``, provenance markers).
5. Both source formats agree, which is the invariant the bug violated.

Every assertion runs from the committed fixture
``tests/fixtures/tier_b/extracted_format_excerpt.json``, which is a verbatim excerpt of the
preserved run-12 capture. The capture itself is not read here, so the suite is portable.

**Scripted downstream answers, labelled.** No assertion in this file replays the original
reviewer's line IDs or verdicts. Those were produced against this draft under the old rule
order; applying them to patched output would be a false claim of faithful replay. Where a
downstream verdict is needed it is constructed here and marked ``[scripted]``.

Labelled **[extracted-format]**.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import source_preservation as sp

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "tier_b" / "extracted_format_excerpt.json"


def _fixture() -> dict:
    """The committed excerpt. Fails loudly rather than skipping: a missing REQUIRED
    fixture means a broken checkout, not an inapplicable test."""
    if not FIXTURE.exists():
        raise AssertionError(
            f"required fixture missing: {FIXTURE}. Regenerate with "
            f"`python tests/fixtures/tier_b/extract_extracted_format.py --write` from a "
            f"checkout that still has the preserved capture.")
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _sources() -> dict[str, str]:
    return {sid: blob["text"] for sid, blob in _fixture()["sources"].items()}


def _combined() -> str:
    # The orchestrator's join. Asserted in the route tests; used here so the fixture is
    # read the way the pipeline reads it.
    return "\n\n---\n\n".join(_sources()[sid] for sid in sorted(_sources()))


def _draft() -> str:
    return _fixture()["draft"]["text"]


def _preserved() -> tuple[str, list[str]]:
    return sp.preserve_source_facts(_draft(), _combined())


# ---------------------------------------------------------------------------
# 0. The fixture is the shape it claims to be
# ---------------------------------------------------------------------------

def test_the_fixture_really_is_extracted_text_and_not_markdown():
    """[extracted-format] Guard the precondition, so the tests cannot silently stop
    testing the thing they exist for."""
    fixture = _fixture()
    assert fixture["fixture_version"] == 1
    for sid, blob in fixture["sources"].items():
        assert hashlib.sha256(blob["text"].encode()).hexdigest() == blob["sha256"]
        assert "**" not in blob["text"], f"{sid} carries bold markers"
        assert not blob["text"].lstrip().startswith("#"), f"{sid} carries a # heading"
    assert hashlib.sha256(fixture["draft"]["text"].encode()).hexdigest() == \
        fixture["draft"]["sha256"]
    # And no reviewer material was copied in, which is the anti-replay guarantee.
    assert set(fixture) == {"fixture_version", "provenance", "sources", "draft"}


# ---------------------------------------------------------------------------
# 1. Section recognition
# ---------------------------------------------------------------------------

def test_every_section_is_found_in_the_extracted_source():
    """[extracted-format] Both documents' sections, including the repeats.

    A multi-file source repeats each heading once per document. Every repetition must be
    visible, or a per-document rule reads one document and deletes the other's content.
    """
    sections = sp._split_sections(_combined())
    headings = [s["heading"] for s in sections if s["heading"]]
    assert headings.count("Professional Summary") == 2
    assert headings.count("Professional Experience") == 2
    assert headings.count("Education") == 2
    assert headings.count("Skills") == 2
    assert headings.count("Certifications") == 1
    # Nothing is left unheaded at the start: the document begins with a heading.
    assert sections[0]["heading"] == "Professional Summary"


@pytest.mark.parametrize("line", [
    "Professional Summary", "professional summary", "Summary:", "Skills",
    "Technical Skills", "Education", "Certifications", "Professional Experience",
    "Work Experience", "Projects",
])
def test_known_section_names_are_recognised_bare(line):
    """[extracted-format] The allowlist, case-insensitively and with an optional colon."""
    assert sp._heading_of(line) is not None


@pytest.mark.parametrize("line", [
    "Northline Fabrication, Inc.",           # an employer
    "Welder / Maintenance Helper",           # a role
    "Electrical Systems Technology",         # a degree field
    "Spokane Falls Institute of Technology",  # an institution
    "Maintenance Electrician",
    "Completed 8,000 apprenticeship hours under a licensed journeyman.",
    "Education is important to me",          # prose that contains a section word
    "Skills I want to develop",
])
def test_arbitrary_short_title_case_lines_are_not_headings(line):
    """[extracted-format] No heuristic classification of "short title-case line".

    A rule like that would split the document at an employer or a role and corrupt every
    section boundary. The allowlist is closed on purpose.
    """
    assert sp._heading_of(line) is None


def test_a_marked_heading_still_works_and_keeps_its_level():
    """[extracted-format] Supporting the bare shape must not break the Markdown shape."""
    found = sp._heading_of("### Education")
    assert found is not None
    assert found[0] == "Education"
    assert found[1] == 3


# ---------------------------------------------------------------------------
# 2. Summary restoration — both cases
# ---------------------------------------------------------------------------

def test_both_documents_summaries_are_read():
    """[extracted-format] Including the supplied 12-year claim, verbatim."""
    summary = sp.source_summary(_combined())
    assert "12 years of electrical experience" in summary
    assert "commercial construction, industrial maintenance" in summary
    assert "Maintenance electrician focused on electrical distribution" in summary
    assert "building automation systems" in summary


def test_summary_is_restored_when_the_section_is_absent():
    """[extracted-format] CASE A — no summary section in the draft at all.

    This is the run-12 shape. Declining to insert here is what produced the withholding:
    the reviewer reported the summary missing and the engine withheld a resume whose source
    stated it.
    """
    draft_without = _draft()
    assert "Professional Summary" not in draft_without
    out, changes = sp.preserve_source_facts(draft_without, _combined())
    assert "Professional Summary" in out
    assert "12 years" in out
    assert any("summary" in c for c in changes)


def test_summary_is_filled_when_the_section_exists_but_is_empty():
    """[extracted-format] CASE B — heading present, body empty.

    The heading-first rule order creates the section; the summary rule must then fill it.
    This is the exact pair that was broken: the rule that fills ran before the rule that
    creates, so the body stayed permanently empty.
    """
    empty = "## Professional Summary\n\n" + _draft()
    out, _changes = sp.preserve_source_facts(empty, _combined())
    assert "12 years" in out
    # Filled once, not duplicated.
    assert out.count("12 years") == 1


def test_the_summary_is_not_duplicated_and_not_moved_into_a_date_line():
    """[extracted-format] Restoring must not invent, duplicate, or misplace.

    A summary inserted into a role/date header would corrupt attribution, which is worse
    than the omission it repairs.
    """
    out, _ = _preserved()
    assert out.count("Professional Summary") == 1
    assert out.count("12 years") == 1
    for line in out.splitlines():
        if "12 years" in line:
            assert "|" not in line, "summary text was placed on a record line"
            assert "SYNTHETIC-ONLY" not in line


def test_the_summary_comes_from_the_source_and_is_not_composed():
    """[extracted-format] Every content word of the restored summary is the source's."""
    out, _ = _preserved()
    summary_line = next(l for l in out.splitlines() if "12 years" in l)
    source_words = set(sp._strip_markers(sp.source_summary(_combined())).lower().split())
    out_words = set(summary_line.lower().split())
    assert out_words <= source_words, (
        f"composed words not in the source: {sorted(out_words - source_words)}")


# ---------------------------------------------------------------------------
# 3. Education and ownership with formatting stripped
# ---------------------------------------------------------------------------

def test_both_education_records_survive_without_any_emphasis_markers():
    """[extracted-format] The run-12 source has no `**`/`*`, and both records must parse."""
    records = sp.source_education_records(_combined())
    assert len(records) == 2
    institutions = {r["institution"] for r in records}
    assert institutions == {"Spokane Falls Institute of Technology",
                            "Inland Northwest Safety Academy"}
    fields = {r["field"] for r in records}
    assert fields == {"Electrical Systems Technology", "NFPA 70E Arc Flash Standards"}
    assert {r["dates"] for r in records} == {"2015", "2021"}


def test_the_degree_field_is_restored_and_not_invented():
    """[extracted-format] The writer dropped "Electrical Systems Technology".

    The rule must put the source's own field back rather than leaving a bare award.
    """
    out, _ = _preserved()
    assert "Electrical Systems Technology" in out
    for line in out.splitlines():
        if "Associate of Applied Science" in line:
            # The award line must not also carry an invented field or a different year.
            assert "2015" in line or "2015" in out.split(line)[-1][:200]


def test_employer_ownership_is_resolved_on_the_extracted_format():
    """[extracted-format] The guard that silently disabled itself.

    ``_employer_of`` read only bold lines. On extracted text it returned ``""`` for every
    line, and an empty employer DISABLES the cross-record check in
    ``preserve_quantified_qualifiers`` — so attribution protection was off exactly on the
    input shape that reaches production, while still working on the Markdown fixtures.
    """
    combined = _combined()
    position = combined.index("31 commercial buildouts")
    assert sp._employer_of(combined, position) == "kettle falls electrical contractors"

    position2 = combined.index("6 commercial properties")
    assert sp._employer_of(combined, position2) == "riverbend facilities group"


def test_employer_ownership_matches_between_marked_and_extracted_sources():
    """[extracted-format] The invariant: emphasis must not change attribution.

    Same facts, one source marked and one bare. If the resolved employer differs, some
    rule is reading formatting as content.
    """
    bare = _combined()
    marked = "\n".join(
        f"**{m.group('head')}** -- {m.group('location')}"
        if sp._LOCATION_SUFFIX.match(sp._strip_markers(l)) else l
        for l in bare.splitlines()
        for m in [sp._LOCATION_SUFFIX.match(sp._strip_markers(l))]
        if True)
    # The marked variant differs only in the employer/institution lines.
    for needle in ("31 commercial buildouts", "6 commercial properties"):
        assert (sp._employer_of(bare, bare.index(needle))
                == sp._employer_of(marked, marked.index(needle)))


def test_a_record_from_another_employer_is_not_restored_under_the_wrong_heading():
    """[extracted-format] Ownership must still block a cross-record restoration.

    ``27%`` belongs to Riverbend. Presented under Northline, the rule must not accept the
    Riverbend source line as that record's truth.
    """
    moved = _draft().replace(
        "- Fabricated and repaired structural steel assemblies for 14 industrial contracts",
        "- Fabricated assemblies for 14 industrial contracts, cutting callouts by 27%")
    out, _ = sp.preserve_source_facts(moved, _combined())
    position = out.index("Fabricated assemblies for 14 industrial contracts")
    assert sp._employer_of(out, position) == "northline fabrication, inc."


# ---------------------------------------------------------------------------
# 4. The demonstrated source-only cleanup
# ---------------------------------------------------------------------------

def test_the_unsupported_motor_controls_term_is_removed():
    """[extracted-format] The term is in the JOB AD and in neither source.

    Run 12's draft listed it in Technical Skills. Once the source skills are readable the
    block is replaced with the candidate's own list, so the term cannot survive by having
    appeared in the posting.
    """
    assert "motor controls" in _draft().lower(), "precondition: the draft contains it"
    for name, text in _sources().items():
        assert "motor control" not in text.lower(), f"precondition: absent from {name}"
    out, changes = _preserved()
    assert "motor controls" not in out.lower()
    assert any("skills block" in c for c in changes)


def test_synthetic_provenance_markers_are_removed_from_every_section():
    """[extracted-format] Including Certifications, which no earlier rule rewrites.

    Measured on run 12: the marker survived in the licence line because the per-rule strip
    never touched that section. The render point now guarantees the invariant.
    """
    assert "SYNTHETIC-ONLY" in _draft(), "precondition: the draft carries one"
    out, _ = _preserved()
    assert "SYNTHETIC-ONLY" not in out
    # And the licence itself survives, marker-free.
    assert "Washington 01 Journey Level Electrician License" in out


def test_skills_are_the_candidates_own_list_from_both_documents():
    """[extracted-format] Both documents' skills, no invented group headings."""
    out, _ = _preserved()
    for term in ("Conduit bending", "switchgear troubleshooting", "thermographic inspection",
                 "Electrical distribution", "BAS controls", "lockout/tagout"):
        assert term.lower() in out.lower(), term


# ---------------------------------------------------------------------------
# 5. Format parity, and the scope limit
# ---------------------------------------------------------------------------

def test_the_two_source_formats_produce_the_same_preservation_result():
    """[extracted-format] THE invariant the bug violated.

    The same facts with and without Markdown markers must yield the same preservation
    result. Equality is asserted on the normalised text, because the renderer legitimately
    re-emits ``##`` headings.
    """
    bare = _combined()
    marked = "\n".join(
        "#" + l if sp._bare_heading(l.strip()) else l
        for l in bare.splitlines())
    bare_out, _ = sp.preserve_source_facts(_draft(), bare)
    marked_out, _ = sp.preserve_source_facts(_draft(), marked)
    assert bare_out == marked_out


def test_appended_outcome_clauses_are_NOT_repaired_here():
    """[extracted-format] Explicit scope limit — this is not the semantic fix.

    Six bullets still append an outcome the source does not state. Deterministic
    preservation is not a substitute for semantic review, and this test records that the
    repair did not silently broaden into one.
    """
    out, _ = _preserved()
    for clause in ("ensuring consistent uptime", "maintaining grid reliability",
                   "improving safety compliance adherence", "ensuring system uptime"):
        assert clause in out, (
            f"{clause!r} should still be present: repairing outcome embellishment is a "
            f"reviewer/writer change, not a preservation rule")


# ---------------------------------------------------------------------------
# 6. [scripted] downstream behaviour, labelled
# ---------------------------------------------------------------------------

def test_scripted_the_preserved_output_retains_the_review_required_facts():
    """[extracted-format][scripted] A REQUIRED-fact check over the patched output.

    ``[scripted]``: this is a fact checklist written here, not a replay of the original
    reviewer's verdicts. The original answer judged a different document under a different
    rule order, so reusing its line IDs against patched text would not be a faithful
    replay.
    """
    out, _ = _preserved()
    required = [
        "Professional Summary",
        "12 years",
        "Northline Fabrication",
        "Kettle Falls Electrical Contractors",
        "Pend Oreille Power Systems",
        "Riverbend Facilities Group",
        "Spokane Falls Institute of Technology",
        "Inland Northwest Safety Academy",
        "Electrical Systems Technology",
        "14 industrial contracts",
        "31 commercial buildouts",
        "8,000 apprenticeship hours",
        "9 substations",
        "4.5 hours to 2.75 hours",
        "Trained 4 apprentices",
        "6 commercial properties",
        "27%",
        "Washington 01 Journey Level Electrician License",
    ]
    missing = [item for item in required if item not in out]
    assert not missing, f"required facts lost by preservation: {missing}"


def test_scripted_no_metric_moved_to_another_employer():
    """[extracted-format][scripted] Ownership of every figure, checked positionally.

    Asserting "the string appears" would pass even if a figure moved between records, so
    each figure is checked against the employer heading that precedes it.
    """
    out, _ = _preserved()
    expected = {
        "14 industrial contracts": "northline fabrication, inc.",
        "31 commercial buildouts": "kettle falls electrical contractors",
        "8,000 apprenticeship hours": "kettle falls electrical contractors",
        "9 substations": "pend oreille power systems",
        "4.5 hours to 2.75 hours": "pend oreille power systems",
        "Trained 4 apprentices": "pend oreille power systems",
        "6 commercial properties": "riverbend facilities group",
        "27%": "riverbend facilities group",
    }
    for needle, employer in expected.items():
        assert needle in out, f"{needle} missing from output"
        assert sp._employer_of(out, out.index(needle)) == employer, (
            f"{needle!r} no longer owned by {employer!r}")
