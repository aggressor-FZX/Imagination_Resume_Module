"""Deterministic source preservation for the final resume Markdown.

The writer is a language model and will keep paraphrasing in ways that lose a
qualifier, drop a field, or append a proficiency list. Prompt instructions reduce how
often that happens but cannot guarantee it, and each prompt change has cost a paid
submission to validate. Anything that must hold *every* time belongs here, as a
deterministic post-process over the drafted Markdown, keyed off the candidate's own
source text.

Scope, stated precisely so this is not mistaken for a verifier:

* This module only ever REPLACES a span with text drawn from the source, or REMOVES a
  span the source does not support, or RESTORES a source value the draft dropped. It
  never composes new prose.
* It runs on the live route: ``StarEditor.polish`` calls ``preserve_source_facts``
  before returning, so the orchestrator, the publication gate and export all see the
  normalized document.
* It is not a substitute for the reviewer. A semantic claim the reviewer alone can
  judge is out of scope, and this module says so rather than pretending otherwise.

Four defects are handled, each demonstrated by delivery run 7:

1. **Education record.** The draft wrote
   ``**Associate of Applied Science, Example Technical College** -- 2016``, which
   concatenates degree and institution into one bold *name* and drops the field of
   study. The source keeps them as a record: bold institution line, then
   ``*Degree, Field | Date*``.
2. **Quantified qualifier.** The source says "8,000 hours of **supervised** field
   work"; the draft said "8,000 hours of **hands-on** field experience ... mastering
   core electrical installations". The number survived, its governing word did not, and
   an unsupported outcome was appended. Every token is sourced, which is why no lexical
   addition check catches it.
3. **Summary proficiency list.** The draft restated the source summary and then
   appended ", proficient in NFPA 70E, Arc Flash safety, electrical systems,
   apprentice supervision, and inspections" — an addition past where the source summary
   ends.
4. **Required headings.** The product promises the core headings; an empty Projects
   section is correct and must not be filled, while the summary may only appear when
   the source supplies one.
"""
from __future__ import annotations

import re

# Provenance markers a synthetic fixture carries on every factual line. They are not
# resume facts and must not appear in a rendered resume.
_PROVENANCE_MARKERS = ("SYNTHETIC-ONLY",)

_SECTION_HEADING = re.compile(r"^#{1,6}\s*(?P<name>.+?)\s*$")

# Bare section headings, as real DOCX extraction produces them.
#
# A `.docx` stores a section title as a STYLED PARAGRAPH, not as a "#" prefix, so the
# text extracted from one contains no Markdown heading markers at all. Measured on the
# preserved run-12 capture, whose source begins:
#
#     Professional Summary
#     SYNTHETIC-ONLY Journeyman electrician with 12 years of electrical experience spanning
#
# Parsed with ``_SECTION_HEADING`` alone that document is ONE section with
# ``heading=None``, so every per-section rule (summary, education, skills) matched nothing
# and ``preserve_source_facts`` returned the draft unchanged. The engine then withheld on
# ``source_facts_missing`` for facts the source plainly stated.
#
# The allowlist is deliberately a CLOSED set of known resume section names, matched against
# the WHOLE stripped line (with an optional trailing colon, as in "Skills:"). It is not a
# heuristic for "short title-case line": a rule like that would turn an employer name, a
# role, or a one-line achievement into a heading and split sections in the wrong place.
# Extending it means naming a new resume section here, which is a reviewable act.
_BARE_SECTION_NAMES = (
    "professional summary", "summary", "profile", "objective",
    "professional experience", "experience", "work experience",
    "employment history", "employment",
    "projects", "education", "academic background",
    "certifications", "certificates", "licenses", "certifications and licenses",
    "technical skills", "skills", "core competencies", "awards", "publications",
)
_BARE_SECTION = re.compile(
    r"^(?P<name>" + "|".join(re.escape(name) for name in
                              sorted(_BARE_SECTION_NAMES, key=len, reverse=True))
    + r")\s*:?$",
    re.IGNORECASE,
)

_BOLD_LINE = re.compile(r"^\*\*(?P<bold>.+?)\*\*\s*(?P<rest>.*)$")
_ITALIC_LINE = re.compile(r"^\*(?P<body>[^*]+)\*\s*$")

# Extracted text carries records without emphasis, e.g.
#     SYNTHETIC-ONLY Northline Fabrication, Inc. -- Spokane, WA
#     Welder / Maintenance Helper | 03/2012 - 09/2015
# The institution/role line is recognisable by its trailing location, the detail line by
# a trailing pipe-delimited date. Both are anchored whole-line so prose is never matched.
_LOCATION_SUFFIX = re.compile(
    r"^(?P<head>.+?)\s+[-\u2013\u2014]{1,2}\s+"
    r"(?P<location>[A-Z][A-Za-z .'-]+,\s*(?:[A-Z]{2}|[A-Z][a-z]+))\s*$")
_DETAIL_SUFFIX = re.compile(r"^(?P<body>.+?)\s*\|\s*(?P<dates>.+?)\s*$")


def _bare_heading(raw: str) -> str | None:
    """The canonical section name for a bare extracted heading line, else None."""
    match = _BARE_SECTION.match((raw or "").strip())
    return match.group("name") if match else None


def _heading_of(raw: str) -> tuple[str, int] | None:
    """Return ``(name, level)`` when ``raw`` is a heading, marked or bare."""
    stripped = (raw or "").strip()
    marked = _SECTION_HEADING.match(stripped)
    if marked:
        return marked.group("name"), len(raw) - len(raw.lstrip("#"))
    bare = _bare_heading(stripped)
    if bare:
        # A bare heading has no marker depth; treat it as a top-level section so the
        # renderer emits a stable "## " level for both input shapes.
        return bare, 2
    return None


def _strip_markers(text: str) -> str:
    out = text
    for marker in _PROVENANCE_MARKERS:
        out = out.replace(marker, "")
    return re.sub(r"\s{2,}", " ", out).strip()


def _split_sections(markdown: str) -> list[dict]:
    """Split Markdown OR extracted plain text into ordered sections.

    Text before the first heading is kept as a section with ``heading=None`` so nothing
    is silently dropped.

    Both heading shapes are recognised because both reach this module: the website sends
    a multi-file source whose README fixtures were hand-written with ``##``, while a real
    upload is extracted from ``.docx`` and carries no markers at all. Supporting only the
    marked shape made every downstream rule blind on real uploads (run 12).

    Document boundaries are preserved: the multi-file route joins uploads with
    ``\\n\\n---\\n\\n``, which is not a heading, so a repeated section heading inside the
    second document starts a second section rather than merging into the first. Every
    per-section rule already reads ALL matches via ``_find_sections``.
    """
    sections: list[dict] = []
    current = {"heading": None, "level": 0, "lines": []}
    for raw in (markdown or "").splitlines():
        found = _heading_of(raw)
        if found is not None:
            if current["heading"] is not None or current["lines"]:
                sections.append(current)
            name, level = found
            current = {"heading": name, "level": level, "lines": []}
            continue
        current["lines"].append(raw)
    sections.append(current)
    return sections


def _render_sections(sections: list[dict]) -> str:
    """Render sections, removing provenance markers from EVERY line.

    The marker is fixture metadata, not a resume fact. Stripping it at the single render
    point guarantees it cannot survive into any section — measured failure: run 12's
    draft carried ``- SYNTHETIC-ONLY Washington 01 Journey Level Electrician License``
    into Certifications, a section no earlier rule rewrites, so a per-rule strip left it
    in place.

    This is a no-op for a real candidate, whose text never contains the marker.
    """
    out: list[str] = []
    for section in sections:
        if section["heading"] is not None:
            out.append("#" * max(section["level"], 1) + " " + section["heading"])
        out.extend(_strip_markers(line) if line.strip() else line
                   for line in section["lines"])
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def _find_section(sections: list[dict], *names: str) -> dict | None:
    lowered = {n.lower() for n in names}
    for section in sections:
        if section["heading"] and section["heading"].strip().lower() in lowered:
            return section
    return None


def _find_sections(sections: list[dict], *names: str) -> list[dict]:
    """EVERY section matching ``names``, in document order.

    ``_find_section`` is right for a DRAFT, which has one section per heading. It is
    wrong for a SOURCE that is several documents concatenated, which is exactly what the
    website's multi-file route sends: the documents are joined with
    ``\\n\\n---\\n\\n``, so each one repeats the same headings.

    Reading only the first match made every per-document rule see one document. Measured
    on two synthetic resumes, that produced two silent data losses:

    * the summary rule compared a merged summary against the FIRST document's summary,
      called the second document's sentence an "appended proficiency list", and truncated
      it away;
    * the education rule rebuilt the Education section from the first document's records
      alone, deleting the second document's degree.

    Neither raised: both are replacements, so the result looked like a normal
    preservation change.
    """
    lowered = {n.lower() for n in names}
    return [section for section in sections
            if section["heading"] and section["heading"].strip().lower() in lowered]


# ---------------------------------------------------------------------------
# 1. Source records
# ---------------------------------------------------------------------------

def source_education_records(source_text: str) -> list[dict]:
    """Extract education records from a source document's Education section(s).

    The source shape this recognises is the one the fixture (and ordinary resumes) use:

        **Institution** -- Location
        *Degree [in Field] | Year*

    A record is only produced when BOTH an institution line and a detail line are
    present, because a partial record cannot be restored faithfully. Returning nothing is
    the correct answer when the source does not state the record in this form; the caller
    then leaves the draft alone rather than inventing a shape.

    EVERY Education section is read, not just the first. A multi-file source is several
    documents concatenated, so it repeats the heading once per document; reading only the
    first silently dropped the other candidate's degree, because the education rule
    rebuilds the section from whatever it parsed.
    """
    records: list[dict] = []
    for record_section in _find_sections(_split_sections(source_text), "education",
                                         "academic background"):
        records.extend(_education_records_in(record_section))
    return records


def _education_records_in(record_section: dict) -> list[dict]:
    """The education records in one Education section."""
    records: list[dict] = []
    lines = [line for line in record_section["lines"] if line.strip()]
    index = 0
    while index < len(lines):
        head, location = _record_head(lines[index])
        if head is None:
            index += 1
            continue
        institution = head
        detail = None
        if index + 1 < len(lines):
            detail = _record_detail(lines[index + 1])
        if detail is None:
            index += 1
            continue
        records.append({
            "institution": institution,
            "location": location,
            "detail_line": detail,
            "degree": detail.split("|")[0].split(",")[0].strip(),
            "field": (detail.split("|")[0].split(",", 1)[1].strip()
                      if "," in detail.split("|")[0] else ""),
            "dates": (detail.split("|", 1)[1].strip() if "|" in detail else ""),
        })
        index += 2
    return records


def source_summary(source_text: str) -> str:
    """The candidate's own summary paragraph, or ``""`` when the source has none.

    With several documents the summaries are concatenated in document order rather than
    only the first being returned. Returning the first alone made the summary rule treat
    every later document's summary sentence as text the source does not support.
    """
    parts = [words for words in
             (_summary_in(section) for section in
              _find_sections(_split_sections(source_text), "professional summary",
                             "summary", "profile", "objective"))
             if words]
    return " ".join(parts)


def _summary_in(record_section: dict) -> str:
    """One section's summary text, with provenance markers removed."""
    words = " ".join(line.strip() for line in record_section["lines"] if line.strip())
    return _strip_markers(words)


def source_quantified_facts(source_text: str) -> list[dict]:
    """Sentences in the source that attach a qualifier to a measured quantity.

    Only a figure followed by a **unit** counts: hours, percent, projects, job sites and
    similar. A bare number does not, and that restriction is load-bearing. The first
    version of this function accepted any digit sequence, so it treated ``03/2020``,
    ``2016`` and every date as a quantified fact and rewrote the education and summary
    lines as "source wording" — a destructive, silent corruption of exactly the fields
    the other rules exist to protect.

    A "fact" here is the source LINE containing the figure, kept with the words
    immediately around it. The qualifier is what a rewrite most often loses: "8,000 hours
    of supervised field work" and "8,000 hours of hands-on field experience" share every
    digit and share no meaning.
    """
    facts: list[dict] = []
    for raw in (source_text or "").splitlines():
        line = _strip_markers(raw).strip().lstrip("-").strip()
        if not line:
            continue
        if _is_date_record_line(line):
            continue
        for match in _MEASURED_QUANTITY.finditer(line):
            facts.append({
                "value": match.group("value"),
                "unit": match.group("unit").lower(),
                "source_line": line,
                "qualifiers": _qualifier_words(line, match.start(), match.end()),
            })
    return facts


# A figure only counts as a measured quantity when a unit follows it. Dates are excluded
# structurally: `03/2020` has no unit, and `_is_date_record_line` rejects the header lines
# that carry them.
_MEASURED_QUANTITY = re.compile(
    r"(?P<value>\b\d[\d,]*(?:\.\d+)?\+?)"
    r"\s*(?P<unit>hours?|hrs?|percent|%|projects?|job sites?|sites?|buildings?|"
    r"clients?|employees?|staff|apprentices?|vehicles?|miles?|days?|weeks?|months?)\b",
    re.IGNORECASE,
)


def _is_date_record_line(line: str) -> bool:
    """True for a line that is a date range / role header rather than prose.

    ``*Journeyman Electrician | 03/2020 - Present*`` and
    ``*Associate of Applied Science | 2016*`` are records. Rewriting them from a
    neighbouring source line is never a qualifier repair, and the first version of this
    module did exactly that.
    """
    stripped = line.strip()
    if stripped.startswith("*") and stripped.endswith("*"):
        return True
    if re.search(r"\b\d{2}/\d{4}\s*[-–]\s*(\d{2}/\d{4}|Present)\b", stripped):
        return True
    # A short line that is mostly a year is a record, not a claim.
    return bool(re.fullmatch(r"[^.]{0,60}\b(19|20)\d{2}\b[^.]{0,20}", stripped))


def _qualifier_words(sentence: str, start: int, end: int, window: int = 28) -> set[str]:
    """Content words immediately adjacent to a figure, on both sides.

    The window is deliberately tight and word-aligned, and both properties were measured
    rather than guessed:

    * a 42-character window on the recorded source reached from "8,000 hours" all the way
      to "safe work practices" in the next clause, so a legitimate paraphrase
      ("safety practices") looked like a lost qualifier;
    * slicing at a raw character offset produced word FRAGMENTS — "maintained" cut to
      "ined" — which then never matched and read as a lost qualifier too.

    Between them these two faults rewrote two correct bullets back to their plainer source
    form. Adjacency is the whole point of a qualifier: the word governing a number is the
    one beside it, not one several clauses away.
    """
    left_raw = sentence[max(0, start - window):start]
    right_raw = sentence[end:end + window]
    # Drop a partial word at the outer edge of each slice.
    if start - window > 0:
        left_raw = re.sub(r"^[A-Za-z\-]+", "", left_raw)
    if end + window < len(sentence):
        right_raw = re.sub(r"[A-Za-z\-]+$", "", right_raw)
    words = re.findall(r"[A-Za-z][A-Za-z\-]+", left_raw + " " + right_raw)
    stop = {"of", "and", "the", "to", "in", "on", "for", "with", "a", "an", "across",
            "by", "from", "at", "toward", "towards", "hours", "hour", "plus"}
    return {w.lower() for w in words if w.lower() not in stop and len(w) > 3}


# ---------------------------------------------------------------------------
# 2. Education normalization
# ---------------------------------------------------------------------------

def preserve_education_record(markdown: str, source_text: str) -> tuple[str, list[str]]:
    """Rewrite the Education section into source-backed records.

    The specific defect: a bold line whose text is the degree AND the institution. That
    concatenation is not a name the source contains, so the route's independent entity
    check (`approved_resume.unsupported_entities`) rightly refuses it — while the field
    of study, which IS in the source, was dropped.

    The repair restores the source's own two-line record. It never composes a new
    degree, field or date; every value comes from the parsed record. If the source does
    not state a record in a recognisable form, the draft is left untouched.

    A MISSING section is restored, not skipped. Until this was fixed the rule returned
    early when the draft had no Education heading at all, so
    `ensure_required_headings` added an empty `## Education` afterwards and the
    candidate's degrees were silently dropped. That is the exact asymmetry the summary
    rule had: the section that is created first is filled, and the one that is created
    last stays empty. Both branches now emit the same source-backed record through
    ``_education_lines``.
    """
    records = source_education_records(source_text)
    if not records:
        return markdown, []

    sections = _split_sections(markdown)
    section = _find_section(sections, "education", "academic background")
    if section is None:
        # Absent section. Every value inserted comes from the parsed source record, and
        # the alternative is delivering a resume with no education at all while the
        # reviewer correctly reports the degrees missing.
        #
        # Placement follows the promised section order so this rule does not fight
        # ensure_required_headings: insert before the first section that must come
        # after Education, else append.
        position = len(sections)
        for index, candidate in enumerate(sections):
            canonical = (_HEADING_ALIASES.get(candidate["heading"].strip().lower())
                         if candidate["heading"] else None)
            if canonical in _SECTIONS_AFTER_EDUCATION:
                position = index
                break
        sections.insert(position, {"heading": "Education", "level": 2,
                                   "lines": _education_lines(records)})
        return _render_sections(sections), ["restored the source's education record"]

    changes: list[str] = []
    body = section["lines"]
    body_text = " ".join(body).lower()
    has_institution = any(record["institution"].lower() in body_text for record in records)
    has_field = any(record["field"] and record["field"].lower() in body_text
                    for record in records)
    # A source record the draft never mentions is the multi-document failure mode: the
    # writer keeps the first candidate's degree and drops the second. Checking only
    # "is at least one institution present" cannot see that, because with two source
    # records one present institution satisfies it.
    missing_record = any(record["institution"].lower() not in body_text
                         for record in records)
    if (has_institution and has_field and not missing_record
            and not any(_bolds_a_concatenation(body, record) for record in records)):
        return markdown, []

    section["lines"] = _education_lines(records)
    changes.append("restored the source's education record (degree, field, institution, date)")
    return _render_sections(sections), changes


def _education_lines(records: list[dict]) -> list[str]:
    """The source's education records rendered as section lines.

    Shared by the replace branch and the insert branch so a restored section is
    identical however it came to exist.
    """
    rebuilt: list[str] = [""]
    for record in records:
        line = f"**{record['institution']}**"
        if record["location"]:
            line += f" -- {record['location']}"
        rebuilt.append(line)
        rebuilt.append(f"*{record['detail_line']}*")
        rebuilt.append("")
    return rebuilt


def _bolds_a_concatenation(body: list[str], record: dict) -> bool:
    """True when a bold line carries more than the institution.

    Catches the observed shape (degree + institution in one bold span) without assuming
    the degree and institution appear in any particular order.
    """
    for raw in body:
        match = _BOLD_LINE.match(raw.strip())
        if match is None:
            continue
        bold = _strip_markers(match.group("bold"))
        if record["institution"].lower() not in bold.lower():
            continue
        excess = bold.lower().replace(record["institution"].lower(), "").strip(" ,-\u2013")
        if len(excess) > 3:
            return True
    return False


def _record_head(raw: str) -> tuple[str | None, str]:
    """An institution/employer line split into ``(head, location)``.

    Two shapes reach this module and both must work:

    * ``**Institution** -- Location``  (hand-written Markdown fixtures)
    * ``Institution -- Location``      (text extracted from a real ``.docx``, which has
      no emphasis markers at all)

    The extracted shape is not hypothetical: run 12's source carried
    ``SYNTHETIC-ONLY Spokane Falls Institute of Technology -- Spokane, WA``. Parsing only
    the bold shape meant every education record silently disappeared from a real upload.

    Returns ``(None, "")`` when the line is not a record head, so prose is never eaten.
    """
    stripped = raw.strip()
    bold = _BOLD_LINE.match(stripped)
    if bold is not None:
        head = _strip_markers(bold.group("bold")).strip()
        rest = bold.group("rest").strip().lstrip("-").strip()
        return head, _strip_markers(rest)
    located = _LOCATION_SUFFIX.match(_strip_markers(stripped))
    if located is None:
        return None, ""
    head = located.group("head").strip().rstrip(":")
    # A bullet or a sentence is not a record head, even when it ends in a location.
    if not head or head.startswith("-") or head.endswith("."):
        return None, ""
    return head, located.group("location").strip()


def _record_detail(raw: str) -> str | None:
    """The detail line's text (degree/role, field, dates), or None.

    Accepts both ``*Detail | Dates*`` and the unemphasised extracted ``Detail | Dates``.
    A detail line must carry a pipe-delimited date part; that anchor is what keeps a
    plain prose sentence from being read as a record.
    """
    stripped = _strip_markers(raw.strip())
    italic = _ITALIC_LINE.match(raw.strip())
    if italic is not None:
        return _strip_markers(italic.group("body")).strip()
    detail = _DETAIL_SUFFIX.match(stripped)
    if detail is None:
        return None
    if stripped.startswith("-") or stripped.endswith("."):
        return None
    return stripped.strip()


# ---------------------------------------------------------------------------
# 3. Quantified qualifier + unsupported clause
# ---------------------------------------------------------------------------

def _employer_of(text: str, up_to: int) -> str:
    """The most recent employer line above ``up_to``, normalised.

    Used to keep a restoration inside the record it belongs to. A source bullet states an
    accomplishment *for a particular employer*, and substituting it under a different
    employer's heading moves a fact between records — the misattribution class the answer
    key forbids, and something a figure-plus-unit match cannot see on its own.

    Both record shapes are recognised, for the reason given on ``_record_head``: a source
    extracted from a real ``.docx`` has no bold markers, so reading only bold lines
    returned ``""`` for every line of such a source. An empty employer DISABLES the scoping
    check in the caller, so attribution protection would have switched itself off silently
    on exactly the input shape that reaches production, while continuing to work on the
    hand-written fixtures.

    Returns ``""`` when no employer line is found, which disables the scoping rather than
    guessing at a boundary. At the top of a record that is the correct answer: a section
    heading, not an employer, precedes the first role.
    """
    prefix = text[:up_to]
    for raw in reversed(prefix.splitlines()):
        stripped = raw.strip()
        if not stripped:
            continue
        bold = _BOLD_LINE.match(stripped)
        if bold is not None:
            return _strip_markers(bold.group("bold")).strip().casefold()
        # A section heading ends the search: walking past it would return the PREVIOUS
        # record's employer and scope a restoration to the wrong person's job.
        if _bare_heading(stripped):
            return ""
        located = _LOCATION_SUFFIX.match(_strip_markers(stripped))
        if located is not None:
            head = located.group("head").strip()
            if head and not head.startswith("-") and not head.endswith("."):
                return head.casefold()
    return ""


def preserve_quantified_qualifiers(markdown: str, source_text: str) -> tuple[str, list[str]]:
    """Restore a quantified fact whose governing qualifier the draft lost.

    The rule is deliberately narrow, and every constraint below was added after a measured
    failure, not in anticipation:

    * the draft line must carry the same measured figure AND unit as a source line;
    * the source must attach a qualifier to that figure which the draft line no longer
      carries (after stemming, so ``installation`` still counts as ``installed``);
    * the source line must belong to the SAME employer record as the draft line — adding
      this was necessary because "40+ job sites" matched one employer's source bullet and
      would have been written under another employer's heading;
    * the match must be unambiguous: two equally plausible source lines means no
      restoration, because choosing one would be a guess.

    Replacement with the source line is the conservative move: the text is the
    candidate's own, verbatim. Reconstructing a "fixed" sentence would be composing
    prose, which this module must not do.
    """
    facts = source_quantified_facts(source_text)
    if not facts:
        return markdown, []

    by_key: dict[tuple[str, str], list[dict]] = {}
    for fact in facts:
        fact = dict(fact)
        fact["employer"] = _employer_of(source_text,
                                        source_text.find(fact["source_line"]))
        by_key.setdefault((fact["value"].rstrip("+"), fact["unit"]), []).append(fact)

    changes: list[str] = []
    out_lines: list[str] = []
    offset = 0
    for raw in markdown.splitlines(keepends=True):
        stripped = raw.strip()
        line_employer = _employer_of(markdown, offset)
        offset += len(raw)
        if not stripped or stripped.startswith("#") or _BOLD_LINE.match(stripped) \
                or _ITALIC_LINE.match(stripped) or _is_date_record_line(stripped):
            out_lines.append(raw)
            continue
        replacement = _qualifier_repair_for(stripped, by_key, line_employer)
        if replacement is None:
            out_lines.append(raw)
            continue
        indent = raw[:len(raw) - len(raw.lstrip())]
        prefix = "- " if stripped.startswith("-") else ""
        newline = "\n" if raw.endswith("\n") else ""
        out_lines.append(f"{indent}{prefix}{replacement}{newline}")
        changes.append("restored the source wording for a quantified bullet")
    return "".join(out_lines), changes


def _qualifier_repair_for(line: str, by_key: dict[tuple[str, str], list[dict]],
                          employer: str = "") -> str | None:
    """The source line to substitute for ``line``, or None when it may stand.

    Returns None when the evidence is ambiguous or belongs to another employer. Both
    refusals were measured: the first version silently chose whichever candidate came
    first, which could rewrite one employer's bullet with another's accomplishment.
    """
    words_here = _stemmed_words(line)
    for match in _MEASURED_QUANTITY.finditer(line):
        key = (match.group("value").rstrip("+"), match.group("unit").lower())
        qualifying: list[str] = []
        for fact in by_key.get(key, []):
            if employer and fact["employer"] and fact["employer"] != employer:
                # A different record's accomplishment. Never transplanted.
                continue
            lost = {_stem(word) for word in fact["qualifiers"]} - words_here
            if lost:
                qualifying.append(fact["source_line"])
        distinct = sorted(set(qualifying))
        if len(distinct) > 1:
            # Equally plausible origins: refuse to choose.
            return None
        if distinct:
            return distinct[0]
    return None


def _stem(word: str) -> str:
    """Crude English stem, enough to match ``installed``/``installation``/``installing``.

    A real stemmer is not needed: the comparison is between one source bullet and its
    paraphrase, and the suffixes that vary are the common inflectional and agentive ones.
    Over-stemming risk is low because both sides are stemmed identically.
    """
    value = word.lower()
    for suffix in ("ations", "ation", "ments", "ment", "ings", "ing", "ions", "ion",
                   "ers", "er", "ed", "es", "s"):
        if len(value) - len(suffix) >= 4 and value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _stemmed_words(text: str) -> set[str]:
    return {_stem(word) for word in re.findall(r"[A-Za-z][A-Za-z\-]+", text)
            if len(word) > 3}


# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------

def preserve_source_summary(markdown: str, source_text: str) -> tuple[str, list[str]]:
    """Truncate an appended proficiency list; drop the section when unsourced.

    The reported defect is an addition past where the source summary ends: the draft
    restated the summary correctly and then appended ", proficient in NFPA 70E, ...".
    Appended proficiency is a claim, so the summary is replaced by the source's own
    paragraph.

    When the source has no summary at all, an existing summary heading is removed rather
    than filled — the fourth requirement is "include summary content only when
    source-backed".
    """
    summary = source_summary(source_text)
    sections = _split_sections(markdown)
    section = _find_section(sections, "professional summary", "summary")

    if not summary:
        if section is None:
            return markdown, []
        index = sections.index(section)
        sections.pop(index)
        return _render_sections(sections), ["removed an unsourced professional summary"]

    if section is None:
        # No summary section in the draft but the source has one. Inserting one here is
        # NOT composing prose: every word comes from the candidate's own summary, and it
        # is the exact fact set the reviewer otherwise reports as `source_facts_missing`
        # — which withholds the whole resume instead of delivering it.
        #
        # Measured: run 12 withheld because both documents' summaries were absent from a
        # draft whose source stated them. Declining here produced the withholding, not
        # safety.
        heading = "Professional Summary"
        sections.insert(0, {"heading": heading, "level": 2,
                            "lines": ["", summary, ""]})
        return _render_sections(sections), ["restored the source summary"]

    current = " ".join(line.strip() for line in section["lines"] if line.strip())
    normalized = _strip_markers(current)
    if normalized.casefold() == summary.casefold():
        return markdown, []

    # Only replace when the difference is an ADDITION or a dropped source word. A pure
    # restatement that shares every content word is left alone, so ordinary paraphrase
    # is not overwritten.
    source_words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]+", summary)
                    if len(w) > 3}
    draft_words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]+", normalized)
                   if len(w) > 3}
    added = draft_words - source_words
    lost = source_words - draft_words
    if not added and not lost:
        return markdown, []

    section["lines"] = ["", summary, ""]
    reason = ("trimmed the appended proficiency list from the summary" if added
              else "restored the source summary wording")
    return _render_sections(sections), [reason]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 5. Skills block
# ---------------------------------------------------------------------------

def source_skills_lines(source_text: str) -> list[str]:
    """The candidate's own skill bullets, verbatim, from every Skills section.

    Every section, not only the first: a multi-file source repeats the heading once per
    document, and reading one of them replaced the skills block with a single document's
    list while dropping the other's.

    Document separators are dropped. The multi-file route joins the uploads with
    ``\\n\\n---\\n\\n``, and that separator sits in the trailing lines of the preceding
    section, so without this filter it was emitted as the literal skill ``---``.
    """
    lines: list[str] = []
    for section in _find_sections(_split_sections(source_text), "skills", "technical skills"):
        for raw in section["lines"]:
            value = _strip_markers(raw).strip()
            if not value or _is_document_separator(value):
                continue
            lines.append(value)
    return lines


_DOCUMENT_SEPARATOR = re.compile(r"^[-*_]{3,}$")


def _is_document_separator(value: str) -> bool:
    """True for a Markdown horizontal rule / multi-document join marker."""
    return bool(_DOCUMENT_SEPARATOR.match(value.strip()))


def source_certification_lines(source_text: str) -> list[str]:
    """The candidate's own certification bullets, verbatim, from every section."""
    lines: list[str] = []
    for section in _find_sections(_split_sections(source_text), "certifications"):
        for raw in section["lines"]:
            value = _strip_markers(raw).strip()
            if not value or _is_document_separator(value):
                continue
            lines.append(value)
    return lines


def _source_experience_records(source_text: str) -> list[dict]:
    """Employer records from every Professional Experience section.

    Each record is ``{"head", "location", "detail", "bullets"}``. Only records
    with a recognisable employer head AND a role/detail line are returned, so a
    partial record is never used to invent a shape.
    """
    records: list[dict] = []
    for section in _find_sections(_split_sections(source_text),
                                  "professional experience", "experience",
                                  "work experience"):
        current: dict | None = None
        for raw in section["lines"]:
            head, location = _record_head(raw)
            if head is not None:
                current = {"head": head, "location": location,
                           "detail": "", "bullets": []}
                records.append(current)
                continue
            if current is None:
                continue
            detail = _record_detail(raw)
            if detail is not None:
                current["detail"] = detail
                continue
            value = _strip_markers(raw).strip()
            if value.startswith("-") and not _is_document_separator(value):
                current["bullets"].append(value.lstrip("-").strip())
    return [r for r in records if r["head"] and r["detail"]]


def preserve_employer_records(markdown: str, source_text: str) -> tuple[str, list[str]]:
    """Restore employer names and role lines the writer replaced with placeholders.

    Measured on the Conservative diagnostic ``3abb7811``: the writer emitted
    three identical ``**Current Employer** --`` heads and ``*Analyst |*`` role
    lines, dropping all three real employer names, role titles and dates. The
    reviewer then reported those source facts missing and the run withheld.

    The rule is deliberately narrow and refuses to guess:

    * it acts only when a draft record head is a generic placeholder (or the
      role line has no dates), never on a real employer name;
    * it matches draft records to source records IN ORDER, and only when the
      counts agree -- an ambiguous mapping is left for the reviewer;
    * it substitutes only the candidate's own source head and detail lines.
      Bullets are left to the existing rules and the reviewer.
    """
    source_records = _source_experience_records(source_text)
    if not source_records:
        return markdown, []

    sections = _split_sections(markdown)
    draft_records: list[tuple[dict, int, int]] = []
    for section in _find_sections(sections, "professional experience",
                                  "experience", "work experience"):
        for index, raw in enumerate(section["lines"]):
            head, _ = _record_head(raw)
            if head is not None:
                draft_records.append((section, index))
    if not draft_records or len(draft_records) != len(source_records):
        return markdown, []

    generic = re.compile(r"^(current|previous|present|recent)\s+"
                         r"(employer|company|role|position)$", re.IGNORECASE)
    needs_repair = False
    for (section, index), _source in zip(draft_records, source_records):
        head, _ = _record_head(section["lines"][index])
        if head and generic.match(head.strip()):
            needs_repair = True
            break
        # A role line with no dates is the same placeholder shape.
        for look in range(index + 1, min(index + 3, len(section["lines"]))):
            detail = _record_detail(section["lines"][look])
            if detail is not None and "|" in detail and not re.search(r"\d", detail):
                needs_repair = True
                break
        if needs_repair:
            break
    if not needs_repair:
        return markdown, []

    for (section, index), source in zip(draft_records, source_records):
        head_line = f"**{source['head']}**"
        if source["location"]:
            head_line += f" -- {source['location']}"
        section["lines"][index] = head_line
        for look in range(index + 1, min(index + 3, len(section["lines"]))):
            if _record_detail(section["lines"][look]) is not None:
                section["lines"][look] = f"*{source['detail']}*"
                break
    return _render_sections(sections), ["restored the source's employer and role records"]


def preserve_empty_source_sections(markdown: str, source_text: str) -> tuple[str, list[str]]:
    """Fill a Skills or Certifications section the writer left empty.

    Measured on the Conservative diagnostic ``3abb7811``: the writer emitted
    ``## Technical Skills`` and ``## Certifications`` with no content at all,
    while the source states four skill bullets and one certification. The
    existing skills rule returns early on an empty body, so the candidate's own
    facts were silently dropped and the reviewer then reported them missing.

    Only an EMPTY section is filled, and only with the candidate's own source
    lines. A section with content is left to the existing rules and the
    reviewer; a source with no such section is left alone.
    """
    sections = _split_sections(markdown)
    changes: list[str] = []
    for names, source_lines, label in (
        (("technical skills", "skills"), source_skills_lines(source_text), "skills"),
        (("certifications",), source_certification_lines(source_text), "certifications"),
    ):
        if not source_lines:
            continue
        for section in _find_sections(sections, *names):
            if any(line.strip() for line in section["lines"]):
                continue
            section["lines"] = [""] + [
                line if line.lstrip().startswith("-") else f"- {line}"
                for line in source_lines] + [""]
            changes.append(f"filled an empty {label} section from the source")
    if not changes:
        return markdown, []
    return _render_sections(sections), changes


def _phrase_in_source(term: str, source_tokens: list[str]) -> bool:
    """True when ``term`` appears in the source as contiguous (stemmed) tokens.

    Contiguity is the whole point, and word-level membership was measurably too weak:
    every word of "apprentice crew" and "licensed journeyman electrician" is present
    somewhere in this source, yet neither phrase is stated, and both are invented ranks.
    Requiring the phrase to appear in order rejects them while keeping every genuinely
    sourced term.
    """
    tokens = [_stem(w) for w in re.findall(r"[a-z0-9][a-z0-9/\.\-]*", term.lower())]
    tokens = [t for t in tokens if len(t) > 1]
    if not tokens:
        return True
    size = len(tokens)
    for index in range(len(source_tokens) - size + 1):
        if source_tokens[index:index + size] == tokens:
            return True
    return False


def preserve_skills_block(
    markdown: str,
    source_text: str,
    permitted_bridges: Optional[list[str]] = None,
) -> tuple[str, list[str]]:
    """Replace a skills block the source does not support with the source's own list.

    Delivery run 8 produced a categorised skills block that invented ranks and
    activities: ``Crew: apprentice crew, licensed journeyman electrician`` and
    ``Inspection: inspection coordination``. The source lists six skills and no
    categories. Note the ordinary English judgement the brief asks for and this rule
    encodes structurally rather than semantically: every word of "apprentice crew" and
    "licensed journeyman electrician" DOES occur in the source ("Trained 6 apprentices",
    "journeyman electrician"), so a word-level check would keep them. What is missing is
    the phrase. Supervision is likewise not inferable from training apprentices, so no
    term may assert it.

    The replacement is the source's own bullets, verbatim and ungrouped. Grouping is the
    writer's stylistic choice; it is dropped here only because the invented group
    HEADINGS ("Crew:", "Inspection:") are themselves claims about how the candidate
    organises their skills.

    Nothing happens when every term in the block is sourced, so a faithful skills section
    is untouched.

    ``permitted_bridges`` carries the validated, mode-permitted inferred skill
    labels (Phase B). In Imaginative mode an inferred label is not a defect: it is
    authorised for semantic review. So a term that matches a permitted bridge is
    treated like a sourced term here — it must survive to the reviewer instead of
    being deleted by a lexical check that cannot see intent. The exception is
    per-term and narrow: an unmatchable term still triggers the source-only
    replacement, and Conservative passes no bridges, so its behaviour is
    unchanged.
    """
    permitted = {
        (label or "").strip().casefold()
        for label in (permitted_bridges or [])
        if isinstance(label, str) and label.strip()
    }
    source_lines = source_skills_lines(source_text)
    if not source_lines:
        # No source Skills section to fall back to. Returning early here used to
        # mean an inferred label was simply left in place, which is the correct
        # outcome in Imaginative and irrelevant in Conservative (no bridges).
        return markdown, []

    sections = _split_sections(markdown)
    section = _find_section(sections, "technical skills", "skills")
    if section is None:
        return markdown, []

    body = [line for line in section["lines"] if line.strip()]
    if not body:
        return markdown, []

    source_tokens = [_stem(w) for w in
                     re.findall(r"[a-z0-9][a-z0-9/\.\-]*", source_text.lower())]

    def term_supported(raw: str) -> bool:
        term = re.sub(r"^\s*[-*]\s*", "", raw.strip())
        # A category heading ("**Crew:**") is itself a claim; only its label matters.
        term = term.replace("**", "").rstrip(":").strip()
        if term.casefold() in permitted:
            # An authorised inferred label is not an invented rank: it was tied
            # to a verbatim candidate quote by Phase B and will be judged by the
            # existing semantic review, which this lexical pass cannot replace.
            return True
        return _phrase_in_source(term, source_tokens)

    unsupported = [line for line in body
                   if not _terms_all_supported(
                       line, source_tokens, permitted)]
    if not unsupported:
        return markdown, []

    # The source lines already carry their list marker; adding another produced
    # "- - Commercial electrical systems...".
    section["lines"] = [""] + [
        line if line.lstrip().startswith("-") else f"- {line}"
        for line in source_lines] + [""]
    return _render_sections(sections), ["replaced an unsupported skills block with the source's own skills"]


def _terms_all_supported(
    line: str,
    source_tokens: list[str],
    permitted: Optional[set[str]] = None,
) -> bool:
    """True when EVERY comma-separated term is sourced or is a permitted bridge."""
    allowed = permitted or set()
    text = line.replace("**", "").strip()
    if ":" in text:
        text = text.split(":", 1)[1]
    terms = [t.strip() for t in text.split(",") if t.strip()]
    return bool(terms) and all(
        term.casefold() in allowed or _phrase_in_source(term, source_tokens)
        for term in terms
    )


# 5. Required headings
# ---------------------------------------------------------------------------

# The headings the product promises. "Technical Skills" covers the source's "Skills".
REQUIRED_HEADINGS = (
    "Professional Summary",
    "Professional Experience",
    "Projects",
    "Education",
    "Certifications",
    "Technical Skills",
)

# Sections the product promises AFTER Education. Used to place a restored Education
# section in the promised order rather than wherever the last rule happened to leave the
# document.
_SECTIONS_AFTER_EDUCATION = frozenset({"Certifications", "Technical Skills"})

_HEADING_ALIASES = {
    "professional summary": "Professional Summary",
    "summary": "Professional Summary",
    "professional experience": "Professional Experience",
    "experience": "Professional Experience",
    "work experience": "Professional Experience",
    "projects": "Projects",
    "education": "Education",
    "certifications": "Certifications",
    "technical skills": "Technical Skills",
    "skills": "Technical Skills",
}


def ensure_required_headings(markdown: str, source_text: str) -> tuple[str, list[str]]:
    """Emit every promised heading, in order, adding nothing under an empty one.

    An empty section is the correct representation of "this section exists and the
    candidate has nothing for it", so a heading is inserted with no body. Nothing is
    invented to fill it.

    The summary is the one conditional heading: it is only emitted when the source
    supplies summary text, because an empty summary heading claims a summary the
    candidate never wrote.

    Existing sections are REORDERED into the promised order as well as added. Adding
    only the missing ones left an existing section wherever the writer happened to put
    it — a resume whose summary appeared last because that is where the model emitted it.
    Reordering is a structural move of content that already exists, so it claims nothing
    new.
    """
    sections = _split_sections(markdown)
    present: dict[str, dict] = {}
    others: list[dict] = []
    for section in sections:
        canonical = (_HEADING_ALIASES.get(section["heading"].strip().lower())
                     if section["heading"] else None)
        if canonical and canonical not in present:
            present[canonical] = section
            if section["heading"] != canonical:
                section["heading"] = canonical
        else:
            others.append(section)

    has_summary = bool(source_summary(source_text))
    wanted = [h for h in REQUIRED_HEADINGS
              if h != "Professional Summary" or has_summary]

    changes: list[str] = []
    for name in wanted:
        if name in present:
            continue
        present[name] = {"heading": name, "level": 2, "lines": [""]}
        changes.append(f"added an empty {name} heading")

    # Rebuild in the promised order, then keep any unrecognised section in its original
    # relative position at the end so nothing is dropped.
    ordered: list[dict] = []
    for name in wanted:
        section = present.pop(name, None)
        if section is not None:
            ordered.append(section)
    ordered.extend(present.values())   # wanted-but-filtered (summary without a source)
    ordered.extend(others)
    rebuilt = _render_sections(ordered)

    if not changes and rebuilt.strip() == markdown.strip():
        return markdown, []
    return rebuilt, changes


# ---------------------------------------------------------------------------
# 6. Experience bullets: unsupported additions and dropped qualifiers
# ---------------------------------------------------------------------------

# Spelled-out numbers are material qualifiers too. "the prior four quarters" is a
# specific comparison period; a rewrite that says only "historical comparison" has
# dropped it. This is a general number vocabulary, not a list of phrases from any
# one run, and it lets "six apprentices" and "6 apprentices" compare equal.
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "hundred": "100", "thousand": "1000",
}

_WORD_TOKEN = re.compile(r"[A-Za-z][A-Za-z'’+.-]*|\d[\d,./+%-]*")

# Function words carry no substantiation. Without this filter a source-backed
# clause ("and kept complete records for all routine equipment maintenance") was
# reported as an addition purely because "and", "for" and "all" are not content
# words in the source vocabulary. This is a stop list, not a phrase blacklist.
_CLAUSE_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "their", "we", "our", "each",
    "every", "per", "across", "within", "into", "over", "more", "than", "all",
    "any", "some", "such", "also", "while", "when", "where", "which", "who",
    "not", "no", "under", "through", "during", "before", "after", "between",
    "among", "via", "plus", "about", "both", "other", "including", "include",
})


def _numeric_qualifiers(text: str) -> set[str]:
    """Numeric cores of the figures and number-words a text states."""
    cores: set[str] = set()
    for token in _WORD_TOKEN.findall(text or ""):
        if any(ch.isdigit() for ch in token):
            core = re.sub(r"[^0-9.]", "", token).rstrip(".")
            if core:
                cores.add(core)
        elif token.lower() in _NUMBER_WORDS:
            cores.add(_NUMBER_WORDS[token.lower()])
    return cores


def _source_bullets(source_text: str) -> list[str]:
    """The candidate's own Markdown bullet lines, with markers removed.

    Only lines that ARE Markdown bullets qualify. A DOCX-style extraction stores
    its facts as plain styled paragraphs with no ``-`` prefix; treating every
    line as a bullet made this rule fire on that format, where an existing test
    pins that appended outcome clauses are a reviewer/writer concern, not a
    deterministic preservation one. Requiring the bullet marker keeps the rule
    scoped to the shape it was measured on.
    """
    out: list[str] = []
    for raw in (source_text or "").splitlines():
        stripped = raw.strip()
        if not stripped.startswith("-"):
            continue
        text = _strip_markers(stripped).lstrip("-").strip()
        if text:
            out.append(text)
    return out


def _source_bullets_with_employer(source_text: str) -> list[tuple[str, str]]:
    """``(bullet, employer)`` for each Markdown source bullet.

    The employer is the most recent record heading above the bullet. An empty
    employer means ownership could not be established; a caller must not use
    that bullet for an automatic substitution.
    """
    out: list[tuple[str, str]] = []
    offset = 0
    for raw in (source_text or "").splitlines(keepends=True):
        stripped = raw.strip()
        if stripped.startswith("-"):
            text = _strip_markers(stripped).lstrip("-").strip()
            if text:
                out.append((text, _employer_of(source_text, offset)))
        offset += len(raw)
    return out


def _best_source_bullet(line: str, bullets: list[str],
                        employer: str = "") -> str | None:
    """The unambiguous source line this draft bullet restates, or None.

    ``employer`` scopes the match to the same record. A source bullet from a
    DIFFERENT employer must never be substituted into this one's draft bullet:
    that would move a fact between records. When ownership cannot be
    established on either side, or the best match is tied, return None so the
    reviewer decides instead of the deterministic layer guessing.
    """
    words = _stemmed_words(line)
    best: str | None = None
    best_score = 0
    tie = False
    for bullet in bullets:
        score = len(words & _stemmed_words(bullet))
        if score > best_score:
            best, best_score, tie = bullet, score, False
        elif score == best_score and score > 0:
            tie = True
    if tie or best_score < 3:
        return None
    return best


def _grounded_in_source(word: str, source_words: set[str],
                        source_stems: set[str]) -> bool:
    """True when the word is the source's own or an inflectional variant."""
    if word in source_words or _stem(word) in source_stems:
        return True
    if len(word) >= 6:
        head = word[:6]
        return any(len(v) >= 6 and v.startswith(head) for v in source_words)
    return False


def _bridge_pattern(label: str) -> str:
    """Whole-label, word-bounded, case-insensitive pattern for one bridge."""
    return r"(?<![A-Za-z0-9])" + re.escape(label.strip()) + r"(?![A-Za-z0-9])"


def _contains_permitted_bridge(text: str,
                               permitted_bridges: Optional[list[str]]) -> bool:
    """True when the text carries a validated, mode-permitted inferred label.

    Whole-label, case-insensitive match with word boundaries: a bridge must not
    be confused with a substring of an unrelated word. An empty or absent list
    (Conservative, or a source-only caller) grants no exemption.
    """
    for label in permitted_bridges or []:
        if not isinstance(label, str) or not label.strip():
            continue
        if re.search(_bridge_pattern(label), text or "", re.IGNORECASE):
            return True
    return False


def _without_permitted_bridges(text: str,
                               permitted_bridges: Optional[list[str]]) -> str:
    """The text with every validated bridge phrase removed.

    Used to judge the REST of a clause: an authorised label may stand, but it
    must not authorise an unsupported outcome sitting beside it. Removing only
    the exact whole-label phrase keeps the exemption phrase-scoped.
    """
    residual = text or ""
    for label in permitted_bridges or []:
        if not isinstance(label, str) or not label.strip():
            continue
        residual = re.sub(_bridge_pattern(label), " ", residual,
                          flags=re.IGNORECASE)
    return residual


def _ungrounded_words(text: str, source_words: set[str],
                      source_stems: set[str]) -> list[str]:
    """Content words with no basis in the full source context.

    Verb inflections (``-ing``/``-ed``), figures and number-words are left to the
    qualifier check: a faithful paraphrase may change a verb, but it may not
    introduce a new noun, adjective or outcome the record never states.
    """
    found: list[str] = []
    for token in _WORD_TOKEN.findall(text or ""):
        word = token.strip(".-'’").lower()
        if (len(word) < 3 or word in _CLAUSE_STOPWORDS
                or any(ch.isdigit() for ch in word)
                or word in _NUMBER_WORDS or word.endswith(("ing", "ed"))):
            continue
        if not _grounded_in_source(word, source_words, source_stems):
            found.append(word)
    return found


def preserve_experience_bullets(markdown: str, source_text: str,
                                permitted_bridges: Optional[list[str]] = None
                                ) -> tuple[str, list[str]]:
    """Correct a bullet that adds an unsupported claim or drops a material qualifier.

    Measured on the approved Imaginative run ``de375f60``: the writer appended
    "ensuring accurate fuel expense reporting" and "highlighting key cost drivers"
    to two source bullets, and replaced "against the prior four quarters" with the
    vaguer "with historical comparison". The reviewer marked every line supported,
    so the engine's addition check (which sees only capitalised names and figures)
    and its completeness check (which asks only whether the item is represented)
    both let the inaccurate text through to export.

    This rule is a CORRECTION, not a gate, and stays inside this module's stated
    scope: it either removes a span the source does not support, or restores a
    source value the draft dropped. It never composes prose.

    * A bullet that dropped a numeric qualifier (a figure or a spelled-out number
      such as "four quarters") is replaced by its source line verbatim.
    * A bullet whose PREFIX is fully grounded in the source (a faithful
      paraphrase) but whose trailing clause introduces content words with no
      basis in the FULL source context is replaced by its source line verbatim.
      This is the observed false-approval shape: "Reconciled 1,180 fuel
      invoices quarterly against depot records" is faithful; ", ensuring
      accurate fuel expense reporting" is not. A faithful paraphrase that
      restates a source figure ("cutting rework by 18%") introduces no
      ungrounded content word, so it is left alone by the same rule -- no
      special figure exemption is needed. That matters: exempting any clause
      carrying a source number would let "Trained 4 apprentices ... increased
      productivity by 4%" through, because "4" occurs in the source even
      though the 4% outcome does not.
    * A faithful paraphrase -- different verbs and inflections, no new content
      word, every figure kept -- is left untouched. Verbatim copying is not
      required.

    Three refusals keep this from weakening the gate, each measured on an
    existing control:

    * A draft that carries a DIFFERENT source figure is a transplant, not a
      dropped qualifier. Restoring it to the matched source line would erase the
      very defect the reviewer must withhold, so the line is left untouched.
    * A bullet whose PREFIX is itself ungrounded is the writer's own paraphrase
      with an appended clause. Existing tests pin that class as a reviewer/writer
      concern, not a deterministic preservation one, so it is left untouched.
    * Only a Markdown-bullet source line can be matched. A DOCX-style extraction
      has no bullet markers, so this rule leaves that format to the reviewer.
    * A clause that is (or contains) a validated, mode-permitted skill bridge is
      left for the semantic reviewer. A lexical check cannot see intent, and
      erasing an authorised inferred label would delete legitimate content.

    A correction always substitutes the candidate's own source line. It never
    truncates or composes a new sentence, which would be prose generation and
    would produce a line that exists in neither the draft nor the source.
    """
    source_words = {w.lower() for w in
                    re.findall(r"[A-Za-z][A-Za-z\-]+", source_text or "")
                    if len(w) > 3}
    source_stems = _stemmed_words(source_text or "")
    scoped_bullets = _source_bullets_with_employer(source_text)
    source_figures = _numeric_qualifiers(source_text)

    out_lines: list[str] = []
    changes: list[str] = []
    draft_offset = 0
    for raw in markdown.splitlines(keepends=True):
        stripped = raw.strip()
        if not stripped.startswith("-"):
            draft_offset += len(raw)
            out_lines.append(raw)
            continue
        # Ownership scoping. A source bullet from a DIFFERENT employer must
        # never be substituted into this draft bullet -- that would move a fact
        # between records. When the source has only one employer (or none is
        # marked at all), ownership is unambiguous and the rule may act. When
        # there are several employers and this draft bullet's owner cannot be
        # established, leave it for the reviewer rather than guessing.
        draft_employer = _employer_of(markdown, draft_offset)
        draft_offset += len(raw)
        source_employers = {employer for _, employer in scoped_bullets
                            if employer}
        if len(source_employers) > 1 and not draft_employer:
            out_lines.append(raw)
            continue
        if draft_employer and source_employers:
            candidates = [text for text, employer in scoped_bullets
                          if employer == draft_employer]
        else:
            candidates = [text for text, _ in scoped_bullets]
        if not candidates:
            out_lines.append(raw)
            continue
        bullet = _best_source_bullet(stripped, candidates)
        if bullet is None:
            out_lines.append(raw)
            continue
        # The same bullet text under two different employers is ambiguous
        # ownership: we cannot tell which record this draft bullet belongs to,
        # even after scoping. Leave it for the reviewer rather than guessing.
        owners_of_bullet = {employer for text, employer in scoped_bullets
                            if text == bullet and employer}
        if len(owners_of_bullet) > 1:
            out_lines.append(raw)
            continue
        indent = raw[:len(raw) - len(raw.lstrip())]
        newline = "\n" if raw.endswith("\n") else ""
        draft_figures = _numeric_qualifiers(stripped)
        bullet_figures = _numeric_qualifiers(bullet)
        # A different source figure is a transplant. Leave it for the reviewer.
        transplanted = bool(draft_figures - bullet_figures)
        if not transplanted and bullet_figures - draft_figures:
            out_lines.append(f"{indent}- {bullet}{newline}")
            changes.append("restored a dropped material qualifier")
            continue
        comma = stripped.rfind(",")
        if comma > 0 and not transplanted:
            prefix = stripped[:comma]
            clause = stripped[comma + 1:]
            # A permitted inferred label is authorised for semantic review, but
            # ONLY for its exact phrase. It must not shield an unsupported
            # outcome sitting beside it in the same clause ("applying trend
            # analysis and ensuring accurate reporting"). Remove the label from
            # the clause before judging the rest; if the remainder still
            # introduces ungrounded content, the whole bullet is corrected.
            residual = _without_permitted_bridges(clause, permitted_bridges)
            # No figure exemption: a faithful paraphrase that restates a source
            # number introduces no ungrounded content word, while an invented
            # outcome does ("increased productivity by 4%"). Exempting any
            # clause carrying a source number would admit the latter.
            if (not _ungrounded_words(prefix, source_words, source_stems)
                    and _ungrounded_words(residual, source_words, source_stems)):
                out_lines.append(f"{indent}- {bullet}{newline}")
                changes.append("removed an unsupported appended clause")
                continue
        out_lines.append(raw)
    return "".join(out_lines), changes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def preserve_source_facts(
    markdown: str,
    source_text: str,
    permitted_bridges: Optional[list[str]] = None,
) -> tuple[str, list[str]]:
    """Apply every source-preservation rule, in dependency order.

    Education and summary are normalized before headings are ensured, so a heading is
    never inserted for a section that normalization is about to fill or remove. The
    quantified-qualifier pass runs on prose, which the education/summary passes may have
    rewritten, so it runs last.

    I reordered this pipeline during the run-12 fix and then measured that the reorder was
    UNNECESSARY: the actual defect was that ``preserve_source_summary`` declined to create
    a missing summary section, and once that branch inserts the section itself the original
    order produces identical output on the captured evidence. Confirmed by reverting the
    order and re-running the full preservation suites (97 passed). The reorder was dropped
    rather than kept as unearned risk.

    The final marker sweep is not part of the ordered rules because it is an invariant, not
    a repair: a rule that copies a source line verbatim can reintroduce a provenance
    marker, so the guarantee is enforced once at the render point regardless of which rule
    last touched the text.

    Returns ``(markdown, changes)``. ``changes`` is a list of short, content-free
    descriptions fit for a log line; it never contains resume text.
    """
    if not isinstance(markdown, str) or not markdown.strip():
        return markdown, []
    if not isinstance(source_text, str) or not source_text.strip():
        return markdown, []

    changes: list[str] = []
    for step in (preserve_education_record,
                 preserve_source_summary,
                 preserve_quantified_qualifiers,
                 preserve_employer_records,
                 preserve_experience_bullets,
                 preserve_skills_block,
                 preserve_empty_source_sections,
                 ensure_required_headings):
        if step in (preserve_skills_block, preserve_experience_bullets):
            # Both rules understand the bridge exception: an authorised inferred
            # label must survive to semantic review. Every other rule keeps its
            # original source-only contract.
            markdown, step_changes = step(
                markdown, source_text, permitted_bridges=permitted_bridges
            )
        else:
            markdown, step_changes = step(markdown, source_text)
        changes.extend(step_changes)
    # A marker can be reintroduced by a rule that copies a source line verbatim, so the
    # final pass guarantees the invariant rather than trusting each rule to uphold it.
    if any(marker in markdown for marker in _PROVENANCE_MARKERS):
        markdown = _render_sections(_split_sections(markdown))
        changes.append("removed synthetic provenance markers")
    return markdown, changes
