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
from source_preservation import _stem
from tests.fixtures.fidelity import expected_facts as facts

RUN7 = "/home/skystarved/Render_Dockers/.live-run-2026-11-02/captures/" \
       "run7/0a267e04-affd-4cee-b944-26c574703039.json"

requires_capture = pytest.mark.skipif(
    not __import__("pathlib").Path(RUN7).exists(), reason="run-7 capture absent")

SOURCE = facts.SOURCE_TEXT



RUN8 = ("/home/skystarved/Render_Dockers/.live-run-2026-11-02/captures/"
        "run8/7f1cc601-990c-4fdb-a2aa-0665c8cb7a0f.json")


def _run8_draft() -> str:
    """The delivery run-8 writer output — the draft that invented the skills block."""
    return live_capture.load(RUN8)["draft"]


def _run8_source() -> str:
    return live_capture.load(RUN8)["sources"][0]["text"]


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


@requires_capture
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


# ---------------------------------------------------------------------------
# Negative controls
#
# Preservation replaces or removes spans. Each control states a way that could go wrong
# and asserts it does not. Without these the module is only known to fix the one fixture
# it was written against, which is how a "fix" becomes a hard-coded answer.
# ---------------------------------------------------------------------------

def test_repeated_number_under_two_employers_is_not_cross_attributed():
    """The same figure appearing twice must not have its fact moved between records.

    A naive "find the line with this number" would match either employer's bullet and
    could rewrite the wrong one. The rule keys on the figure AND the qualifiers beside
    it, so a second employer's bullet that carries the same digit but different
    surrounding words is left alone.

    Here Cascade's bullet has a correct qualifier and Harborline's is defective: only
    Harborline's may be rewritten.
    """
    source = (
        "## Professional Experience\n"
        "**Alpha Corp** -- City, ST\n"
        "*Engineer | 2019 - 2021*\n"
        "- Audited 40+ job sites for safety compliance.\n"
        "**Beta Corp** -- Town, OR\n"
        "*Technician | 2016 - 2019*\n"
        "- Inspected 40+ sites across two states.\n"
    )
    markdown = (
        "## Professional Experience\n"
        "**Alpha Corp** -- City, ST\n"
        "*Engineer | 2019 - 2021*\n"
        "- Audited 40+ job sites for safety compliance.\n"
        "**Beta Corp** -- Town, OR\n"
        "*Technician | 2016 - 2019*\n"
        "- Covered 40+ job sites.\n"
    )
    out, _ = sp.preserve_source_facts(markdown, source)

    alpha = out.index("**Alpha Corp**")
    beta = out.index("**Beta Corp**")
    alpha_bullet = out[alpha:beta]
    # Alpha's correct bullet is untouched.
    assert "Audited 40+ job sites for safety compliance." in alpha_bullet
    # Beta's defective bullet must NOT adopt Alpha's accomplishment. Either it is
    # restored from Beta's own source line or left alone; both are acceptable, and
    # transplanting Alpha's fact under Beta's heading is not.
    assert "Audited 40+ job sites for safety compliance." not in out[beta:]


def test_two_education_records_are_both_preserved():
    """Two degrees must not collapse into one, and each keeps its own institution."""
    source = (
        "## Education\n"
        "**North College** -- City, ST\n"
        "*Associate of Science, Physics | 2014*\n"
        "**South University** -- Town, OR\n"
        "*Bachelor of Science, Engineering | 2018*\n"
    )
    markdown = (
        "## Education\n"
        "**Associate of Science, North College** -- 2014\n"
        "**Bachelor of Science, South University** -- 2018\n"
    )
    out, _ = sp.preserve_source_facts(markdown, source)

    records = sp.source_education_records(source)
    assert len(records) == 2
    assert "**North College** -- City, ST" in out
    assert "*Associate of Science, Physics | 2014*" in out
    assert "**South University** -- Town, OR" in out
    assert "*Bachelor of Science, Engineering | 2018*" in out
    # Neither institution absorbed the other's degree.
    assert "**Associate of Science, North College**" not in out


def test_dates_are_never_treated_as_quantities():
    """A year or a MM/YYYY range is a record, not a measured fact.

    The first version of this module accepted any digit run, so every date became a
    "quantified fact" and the education and role lines were rewritten from neighbouring
    source lines — silent corruption of exactly the fields the other rules protect.
    """
    source = (
        "## Professional Experience\n"
        "**Alpha Corp** -- City, ST\n"
        "*Engineer | 03/2019 - 08/2021*\n"
        "- Led 4 projects.\n"
    )
    markdown = (
        "## Professional Experience\n"
        "**Alpha Corp** -- City, ST\n"
        "*Engineer | 03/2019 - 08/2021*\n"
        "- Ran 4 projects.\n"
    )
    out, _ = sp.preserve_source_facts(markdown, source)

    # The role/date header is untouched.
    assert "*Engineer | 03/2019 - 08/2021*" in out
    # No date was harvested as a quantity.
    for fact in sp.source_quantified_facts(source):
        assert "/" not in fact["value"], fact["value"]
        assert fact["unit"], fact["value"]


def test_ambiguous_match_is_left_unchanged():
    """When two source lines could equally be the origin, nothing is rewritten.

    Choosing one would be a guess, and a guessed restoration is a fabricated one. The
    conservative behaviour is to leave the draft alone; the reviewer can still withhold.

    The ambiguity must be INSIDE one employer record, or employer scoping would resolve
    it first and this test would never reach the code it is named for. That is exactly
    what happened to the first version of this test: its two candidates sat under
    different employers, so scoping silently answered the question, and mutating the
    ambiguity check away left the test green. Mutation-checking exposed it.

    Here both candidate lines belong to Alpha, carry the same figure and unit, and both
    are missing qualifiers the draft no longer has — genuinely indistinguishable.
    """
    source = (
        "## Professional Experience\n"
        "**Alpha Corp** -- City, ST\n"
        "*Engineer | 2019 - 2021*\n"
        "- Delivered 12 projects under supervision in 2019.\n"
        "- Delivered 12 projects under supervision in 2020.\n"
    )
    markdown = (
        "## Professional Experience\n"
        "**Alpha Corp** -- City, ST\n"
        "*Engineer | 2019 - 2021*\n"
        "- Handled 12 projects.\n"
    )
    out, changes = sp.preserve_source_facts(markdown, source)

    assert not any("quantified" in c for c in changes), (
        f"an ambiguous match was resolved by guessing: {changes}")
    assert "- Handled 12 projects." in out, out

    # And prove the fixture really reaches the ambiguity path: the same source with only
    # ONE candidate must be repaired. Without this the test could pass because the rule
    # never engaged at all.
    unambiguous = source.replace(
        "- Delivered 12 projects under supervision in 2020.\n", "")
    repaired, repaired_changes = sp.preserve_source_facts(markdown, unambiguous)
    assert any("quantified" in c for c in repaired_changes), repaired_changes
    assert "Delivered 12 projects under supervision in 2019." in repaired


@requires_capture
def test_preservation_applied_twice_makes_no_further_changes():
    """Idempotence, on a document where every rule actually fires.

    A rule that re-applied on each pass would let the route oscillate, and a second run
    would keep "restoring" text that is already correct. Asserted on the defect-bearing
    document, not on an already-clean one, so every rule has something to do first.
    """
    draft = _draft()
    once, first_changes = sp.preserve_source_facts(draft, SOURCE)
    assert first_changes, "the fixture must actually trigger the rules"

    twice, second_changes = sp.preserve_source_facts(once, SOURCE)
    assert second_changes == [], second_changes
    assert twice.strip() == once.strip()


def test_preservation_does_not_invent_an_education_record():
    """A source with no Education section must not cause one to be written."""
    source = "## Professional Experience\n**Alpha Corp** -- City, ST\n- Led 4 projects.\n"
    markdown = "## Education\n**Somewhere Else** -- 2019\n"
    out, _ = sp.preserve_source_facts(markdown, source)
    assert "**Somewhere Else** -- 2019" in out
    assert "Electrical Systems Technology" not in out


def test_preservation_leaves_a_correct_document_alone():
    """A document that already satisfies every rule is returned byte-identical."""
    # The summary must match the source's own paragraph, or the summary rule will
    # legitimately restore it — which is correct behaviour, not a failure of idempotence.
    already = ("## Professional Summary\n\n"
               "Journeyman electrician with 8+ years of commercial and industrial "
               "electrical experience across construction, facilities maintenance, "
               "and power distribution.\n\n"
               "## Professional Experience\n"
               "**Example Technical College Holder** -- City, ST\n"
               "*Journeyman Electrician | 03/2020 - Present*\n"
               "- Completed 8,000 hours of supervised field work toward licensure.\n\n"
               "## Projects\n\n"
               "## Education\n"
               "**Example Technical College** -- Example City, WA\n"
               "*Associate of Applied Science, Electrical Systems Technology | 2016*\n\n"
               "## Certifications\n\n"
               "## Technical Skills\n")
    out, changes = sp.preserve_source_facts(already, SOURCE)
    assert out.strip() == already.strip(), changes


# ---------------------------------------------------------------------------
# 5. Skills block
# ---------------------------------------------------------------------------

# Phrases from the delivery run-8 draft that are NOT in the source, and the ordinary
# English reason each fails. Recorded because the judgement is semantic, and writing it
# down is what stops the rule from being a word-list.
RUN8_UNSUPPORTED_SKILL_TERMS = {
    "apprentice crew": (
        "every word occurs in the source ('Trained 6 apprentices', a 'crew' is never "
        "mentioned). Training apprentices does not imply supervising a crew."),
    "licensed journeyman electrician": (
        "'licence' and 'journeyman electrician' both appear, but not as a claimed rank."),
    "inspection coordination": (
        "'inspection' and 'coordination' appear separately; coordinating inspections is "
        "an activity the source never states."),
    "electrical systems installation": (
        "'installation' appears as a verb form of 'Installed'; as a skill it is an "
        "activity, not a listed skill."),
    "commercial construction": (
        "the phrase is in the job ad, not the candidate's record."),
}


@requires_capture
def test_run8_invented_skills_are_removed_and_source_skills_kept():
    """The skills block from delivery run 8 is replaced by the source's own list.

    Before: a categorised block inventing ``Crew: apprentice crew, licensed journeyman
    electrician`` and ``Inspection: inspection coordination``.
    After:  the source's two skill bullets, verbatim.
    """
    out, changes = sp.preserve_source_facts(_run8_draft(), _run8_source())

    assert "replaced an unsupported skills block" in " ".join(changes)

    section = sp._find_section(sp._split_sections(out), "technical skills")
    body = "\n".join(section["lines"])

    for invented in RUN8_UNSUPPORTED_SKILL_TERMS:
        assert invented not in body, f"invented skill survived: {invented}"
    # Invented category headings are claims too.
    for label in ("**Crew:**", "**Inspection:**"):
        assert label not in body, label

    for sourced in ("Commercial electrical systems, conduit bending, panel labeling",
                    "NFPA 70E, lockout/tagout, inspection checklists"):
        assert sourced in body, f"source skill lost: {sourced}"


@requires_capture
def test_supervision_is_not_inferred_from_training_apprentices():
    """``supervision`` must not appear unless the source says it.

    The brief's rule, stated as an assertion: training apprentices is not supervision.
    The source has no "supervis*" token outside "supervised field work" (which qualifies
    hours, not people), so no skills term may assert supervision of anyone.
    """
    source = _run8_source()
    out, _ = sp.preserve_source_facts(_run8_draft(), source)

    section = sp._find_section(sp._split_sections(out), "technical skills")
    body = " ".join(section["lines"]).lower()
    assert "supervision" not in body
    assert "supervising" not in body
    assert "apprentice crew" not in body


@requires_capture
def test_installation_and_maintenance_are_supported_by_the_source_verbs():
    """``installation``/``maintenance`` are NOT inventions: the source says them.

    The brief's other half. The source's first Cascade bullet is "Installed and
    maintained commercial electrical systems", so a skills term built from those verbs is
    supported and must survive. A rule that stripped them would be over-fitting to the
    other invented terms on the same line.
    """
    source = _run8_source()
    tokens = [_stem(w) for w in
              __import__("re").findall(r"[a-z0-9][a-z0-9/\.\-]*", source.lower())]

    assert sp._phrase_in_source("installation", tokens)
    assert sp._phrase_in_source("maintenance", tokens)
    # And the unsupported ranks on the same invented line are still rejected.
    assert not sp._phrase_in_source("apprentice crew", tokens)
    assert not sp._phrase_in_source("licensed journeyman electrician", tokens)


@requires_capture
def test_a_fully_sourced_skills_block_is_left_alone():
    """No change when every term is in the source: the rule is not a rewrite."""
    source = _run8_source()
    markdown = ("## Technical Skills\n"
                "- Commercial electrical systems, conduit bending, panel labeling\n"
                "- NFPA 70E, lockout/tagout, inspection checklists\n")
    out, changes = sp.preserve_source_facts(markdown, source)
    assert not any("skills" in c for c in changes), changes
    assert "Commercial electrical systems, conduit bending, panel labeling" in out


# ---------------------------------------------------------------------------
# 6. Multiple source documents
#
# The website's two-file route sends several documents as ONE combined source: the
# orchestrator joins them with "\n\n---\n\n" and only ``/analyze`` is reached
# (``docs/ROUTE_MAP.md``). Preservation therefore has to read a source that repeats
# every heading once per document.
#
# Every per-document reader used to take the FIRST matching section only. Measured
# before the fix, on the fixtures below:
#
#   * ``source_education_records`` returned one record, so rebuilding the Education
#     section deleted the other document's degree;
#   * ``source_summary`` returned the first document's paragraph, so the second
#     document's sentence counted as an unsourced ADDITION and was truncated away by
#     "trim the appended proficiency list".
#
# Both were replacements, so neither raised — the result looked like a normal change.
# ---------------------------------------------------------------------------

DOC_ELECTRICIAN = """## Professional Summary
Journeyman electrician with 8+ years of commercial electrical experience.

## Professional Experience
**Cascade Power Solutions, LLC** -- Example City, WA
*Journeyman Electrician | 03/2020 - Present*
- Installed commercial electrical systems across 40+ job sites.

## Education
**Example Technical College** -- Example City, WA
*Associate of Applied Science, Electrical Systems Technology | 2016*

## Skills
- Commercial electrical systems, panel labeling
"""

DOC_FACILITIES = """## Professional Summary
Facilities maintenance technician focused on preventive maintenance.

## Professional Experience
**Bluegate Property Services** -- Sampletown, OR
*Facilities Maintenance Technician | 01/2014 - 05/2016*
- Maintained electrical distribution and HVAC controls for 12 commercial buildings.

## Education
**Example Community College** -- Sampletown, OR
*Certificate, Building Systems Maintenance | 2013*

## Skills
- Preventive maintenance scheduling, HVAC controls
"""

COMBINED = DOC_ELECTRICIAN + "\n\n---\n\n" + DOC_FACILITIES

DRAFT_BOTH_RECORDS = """## Professional Summary
Journeyman electrician with 8+ years of commercial electrical experience. Facilities maintenance technician focused on preventive maintenance.

## Professional Experience
**Cascade Power Solutions, LLC** -- Example City, WA
*Journeyman Electrician | 03/2020 - Present*
- Installed commercial electrical systems across 40+ job sites.

**Bluegate Property Services** -- Sampletown, OR
*Facilities Maintenance Technician | 01/2014 - 05/2016*
- Maintained electrical distribution and HVAC controls for 12 commercial buildings.

## Education
**Example Technical College** -- Example City, WA
*Associate of Applied Science, Electrical Systems Technology | 2016*

**Example Community College** -- Sampletown, OR
*Certificate, Building Systems Maintenance | 2013*

## Technical Skills
- Commercial electrical systems, panel labeling
- Preventive maintenance scheduling, HVAC controls
"""


def test_every_document_summary_is_read():
    """Both summaries are source text, so neither may be truncated as an addition."""
    summary = sp.source_summary(COMBINED)
    assert "Journeyman electrician with 8+ years" in summary
    assert "Facilities maintenance technician focused on preventive maintenance" in summary


def test_both_documents_summaries_survive_preservation():
    """A merged summary naming both candidates is NOT an appended proficiency list."""
    out, changes = sp.preserve_source_facts(DRAFT_BOTH_RECORDS, COMBINED)
    assert not any("summary" in change for change in changes), changes

    section = sp._find_section(sp._split_sections(out), "professional summary")
    text = " ".join(section["lines"])
    assert "Journeyman electrician with 8+ years" in text
    assert "Facilities maintenance technician focused on preventive maintenance" in text


def test_every_document_education_record_is_read():
    """Records from both documents, in document order, each complete."""
    records = sp.source_education_records(COMBINED)
    assert [record["institution"] for record in records] == [
        "Example Technical College", "Example Community College"]
    assert [record["dates"] for record in records] == ["2016", "2013"]
    # Each record keeps its own degree, field and location.
    assert records[0]["degree"] == "Associate of Applied Science"
    assert records[0]["field"] == "Electrical Systems Technology"
    assert records[0]["location"] == "Example City, WA"
    assert records[1]["degree"] == "Certificate"
    assert records[1]["field"] == "Building Systems Maintenance"
    assert records[1]["location"] == "Sampletown, OR"


def test_a_draft_that_drops_the_second_degree_gets_it_back():
    """The multi-document failure the fix exists for.

    Before: a draft with only the first document's education produced a rebuilt
            Education section from that one record, deleting the second degree — the
            "collapse education records" risk, observed rather than hypothesised.
    After:  both source records are written back, each with its own institution.
    """
    draft = DRAFT_BOTH_RECORDS.replace(
        "\n**Example Community College** -- Sampletown, OR\n"
        "*Certificate, Building Systems Maintenance | 2013*\n", "")

    out, changes = sp.preserve_source_facts(draft, COMBINED)
    assert "restored the source's education record" in " ".join(changes)
    assert "**Example Community College** -- Sampletown, OR" in out
    assert "*Certificate, Building Systems Maintenance | 2013*" in out
    # And the record that was already right is still right.
    assert "**Example Technical College** -- Example City, WA" in out
    assert "*Associate of Applied Science, Electrical Systems Technology | 2016*" in out


def test_two_correct_documents_produce_no_education_change():
    """A draft already holding both records is left alone.

    This is the control for the test above: restoration must be triggered by the gap,
    not by the mere presence of two source records.
    """
    out, changes = sp.preserve_source_facts(DRAFT_BOTH_RECORDS, COMBINED)
    assert not any("education" in change for change in changes), changes
    assert out.count("**Example Technical College**") == 1
    assert out.count("**Example Community College**") == 1
    assert out.count("Associate of Applied Science") == 1
    assert out.count("Certificate, Building Systems Maintenance") == 1


def test_every_document_skills_block_is_read_without_the_document_separator():
    """Both skill lists are read, and the ``---`` join marker is not a skill."""
    lines = sp.source_skills_lines(COMBINED)
    assert lines == [
        "- Commercial electrical systems, panel labeling",
        "- Preventive maintenance scheduling, HVAC controls",
    ]
    assert "---" not in lines


def test_an_unsupported_skills_block_is_replaced_with_BOTH_documents_skills():
    """Replacement draws on every document, not just the first."""
    draft = ("## Technical Skills\n"
             "- Crew: apprentice crew, licensed journeyman electrician\n")
    out, changes = sp.preserve_source_facts(draft, COMBINED)
    assert "replaced an unsupported skills block" in " ".join(changes)

    section = sp._find_section(sp._split_sections(out), "technical skills")
    body = "\n".join(section["lines"])
    assert "Commercial electrical systems, panel labeling" in body
    assert "Preventive maintenance scheduling, HVAC controls" in body
    assert "apprentice crew" not in body


def test_a_single_document_source_is_unaffected():
    """The multi-document change must not alter single-document behaviour.

    With one document there is one matching section, so every reader behaves exactly as
    before. Asserted because the fix touched four functions on the live route.
    """
    assert sp.source_summary(DOC_ELECTRICIAN) == (
        "Journeyman electrician with 8+ years of commercial electrical experience.")
    assert [record["institution"] for record in
            sp.source_education_records(DOC_ELECTRICIAN)] == ["Example Technical College"]
    assert sp.source_skills_lines(DOC_ELECTRICIAN) == [
        "- Commercial electrical systems, panel labeling"]


def test_a_source_with_repeated_headings_but_no_second_document_still_parses():
    """Two Education headings with one record each are both read."""
    source = ("## Education\n**Alpha College** -- City, ST\n"
              "*Associate of Arts, History | 2012*\n"
              "## Education\n**Beta College** -- Town, OR\n"
              "*Bachelor of Arts, English | 2015*\n")
    records = sp.source_education_records(source)
    assert [record["institution"] for record in records] == ["Alpha College", "Beta College"]
