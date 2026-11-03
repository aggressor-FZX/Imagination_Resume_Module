"""Classification of every residual run-7 reviewer objection.

The reviewer is a model. It can be wrong, and treating its verdicts as ground truth
would make the answer key unfalsifiable: every objection would become a defect to
"fix", including objections that are paraphrase-recognition failures. So each run-7
objection is classified against the *source text* here, before any of them is acted on.

**A checker rejection is not itself proof of fabrication.** A deterministic checker
rejecting a line establishes only that the line failed *that checker*. Three different
things make a line fail, and they demand different responses:

  * the line asserts something the source does not — a writer defect;
  * the line omits something the source states — also a writer defect, but the fix is
    different and nothing false was claimed;
  * the checker itself is scope-limited and cannot see that the line is fine.

Collapsing those into "fabrication" overstates the defect and, worse, licenses code
changes aimed at content that was never wrong.

Categories
----------
``formatting``          the line is structure — a heading, a label, a separator. It
                        asserts no candidate fact, so it cannot be "missing" and it
                        cannot be fabricated. Present-but-empty is its correct state.
``omission``            the source states a fact and the draft does not carry it. The
                        draft asserts nothing false; it *drops* something. Run-7 lost
                        "Electrical Systems Technology" this way and run-4 lost both
                        role titles. The fix is coverage, not removal.
``unsupported_addition`` the draft adds content the source does not establish. This is
                        the only class where the draft claims more than the record
                        supports, and the one that must be repaired or withheld.
``lost_qualifier``      the fact is present but the word governing it changed, so its
                        meaning changed. The most dangerous class, because every token
                        is sourced and a lexical check sees nothing wrong.
``reviewer_error``      the reviewer's verdict contradicts the text it was given. NOT a
                        defect. Recording these is what keeps the key falsifiable.

A line may have a primary class and a secondary defect inside it. Where that happens
both are recorded, because the primary class drives the fix and the secondary one must
still be repaired.
"""
from __future__ import annotations

import pytest

import live_capture

RUN7 = "/home/skystarved/Render_Dockers/.live-run-2026-11-02/captures/" \
       "run7/0a267e04-affd-4cee-b944-26c574703039.json"

requires_run7 = pytest.mark.skipif(
    not __import__("pathlib").Path(RUN7).exists(), reason="run-7 capture absent")

# The agreed vocabulary. Fixed, so a classification cannot be fudged into a new word.
CLASSES = frozenset({"formatting", "omission", "unsupported_addition",
                     "lost_qualifier", "reviewer_error"})

# Which classes justify which kind of fix. ``formatting`` changes structure only and
# ``reviewer_error`` changes nothing; ``omission`` justifies carrying a fact rather than
# suppressing a claim, which is a different remedy from the assertion classes.
ASSERTION_CLASSES = frozenset({"unsupported_addition", "lost_qualifier"})
COVERAGE_CLASSES = frozenset({"omission"})


# ---------------------------------------------------------------------------
# The classification, written from the source and the draft, not the implementation
# ---------------------------------------------------------------------------

OBJECTIONS = (
    {
        "line": "L000014",
        "reviewer_verdict": "unsupported",
        "draft": "- Accumulated 8,000 hours of hands-on field experience toward "
                 "journeyman licensure, mastering core electrical installations.",
        "source": "- Completed 8,000 hours of supervised field work toward licensure.",
        "class": "lost_qualifier",
        "secondary": "unsupported_addition",
        "why": (
            "Primary: a lost qualifier. 'supervised' became 'hands-on'. The source "
            "qualifier is load-bearing because supervised hours are the licensure "
            "pathway, so replacing it changes what the number means while every token "
            "stays sourced. 'licensure' also became 'journeyman licensure', a claim "
            "about which licence the hours counted toward. "
            "Secondary: the trailing clause 'mastering core electrical installations' is "
            "an unsupported addition, an invented outcome with no basis in the source. "
            "Recorded separately because it needs a different repair: the qualifier must "
            "be restored and the clause removed."),
        "asserts_in_source": False,
    },
    {
        "line": "L000024",
        "reviewer_verdict": "unsupported",
        "draft": "**Electrical:** installation, maintenance, commercial electrical "
                 "systems, NFPA 70E, Arc Flash, lockout/tagout, apprentice "
                 "supervision, inspections, panel labeling, conduit bending, wire "
                 "pulling, panel terminations",
        "source": "- Commercial electrical systems, conduit bending, panel labeling\n"
                  "- NFPA 70E, lockout/tagout, inspection checklists",
        "class": "unsupported_addition",
        "secondary": None,
        "why": (
            "'installation' and 'maintenance' are activities, not the listed skills, and "
            "they entered the skills block. 'apprentice supervision' is a rank the "
            "candidate never claims — the source says 'Trained 6 apprentices'. The "
            "category structure ('**Electrical:**') is invented too: the source has no "
            "skill categories. Every remaining word is sourced, which is exactly why the "
            "line passes a lexical check while still overstating the record."),
        "asserts_in_source": False,
    },
    {
        "line": "L000002",
        "reviewer_verdict": "uncertain",
        "draft": "Journeyman electrician with 8+ years of commercial and industrial "
                 "electrical experience across construction, facilities maintenance, "
                 "and power distribution, proficient in NFPA 70E, Arc Flash safety, "
                 "electrical systems, apprentice supervision, and inspections.",
        "source": "SYNTHETIC-ONLY Journeyman electrician with 8+ years of commercial "
                  "and industrial electrical\nexperience across construction, "
                  "facilities maintenance, and power distribution.",
        "class": "unsupported_addition",
        "secondary": "reviewer_error",
        "why": (
            "Primary: the line runs past where the source summary ends, appending "
            "'proficient in NFPA 70E, Arc Flash safety, electrical systems, apprentice "
            "supervision, and inspections'. Appended proficiency is an addition, and "
            "'apprentice supervision' overstates 'trained 6 apprentices' again. "
            "Secondary (partial reviewer error): the reviewer flagged the WHOLE line, but "
            "its first clause faithfully restates the source summary. The objection is "
            "right about the line and wrong about the clause, so both classes are "
            "recorded rather than letting 'the line is bad' stand as 'all of it is "
            "invented'."),
        "asserts_in_source": False,
    },
    {
        "line": "L000017",
        "reviewer_verdict": "missing (segment 15)",
        "draft": "**Associate of Applied Science, Example Technical College** -- 2016",
        "source": "*Associate of Applied Science, Electrical Systems Technology | 2016*",
        "class": "omission",
        "secondary": None,
        "why": (
            "RECLASSIFIED from 'fabrication', which was wrong. The draft omits the field "
            "of study 'Electrical Systems Technology'; it asserts nothing false. The "
            "institution, the degree and the year are all correct. A reader cannot tell "
            "the draft is incomplete, but nothing on the page contradicts the source — "
            "and calling an omission a fabrication overstates the defect and invites a "
            "'stop asserting this' fix for a line whose real fault is that it dropped a "
            "fact. The remedy is coverage: carry the field. "
            "A separate problem exists on this line and is recorded in the defect list "
            "instead: its bold text concatenates degree and institution, so the bold name "
            "is not a name the source contains. That is caught by "
            "`approved_resume.unsupported_entities`. It is a formatting fault with a "
            "rendering consequence, not fabrication either."),
        "asserts_in_source": False,
    },
)

# Run-7 objections that the classification says are NOT defects. Kept separate because
# the point of the exercise is that the reviewer is not always right, and a list with
# nothing in it would suggest the classification never disagreed.
NON_DEFECTS = (
    {
        "line": "L000002 (first clause)",
        "reviewer_verdict": "uncertain",
        "class": "reviewer_error",
        "why": (
            "'Journeyman electrician with 8+ years of commercial and industrial "
            "electrical experience across construction, facilities maintenance, and "
            "power distribution' restates the source summary with only the SYNTHETIC-ONLY "
            "marker and line wrapping removed. Objecting to it demands word-for-word "
            "copying, which is not a fidelity requirement. The reviewer was wrong about "
            "this clause and right about the appended proficiency in the same line, which "
            "is why L000002 appears in both lists."),
    },
    {
        "line": "L000001, L000004, L000016, L000019, L000023",
        "reviewer_verdict": "missing (segments 0, 3, 13, 16, 19)",
        "class": "formatting",
        "why": (
            "Section headings assert no candidate fact. The answer key already says "
            "segment 0 'must not be reported as a missing fact'; the same reasoning "
            "applies to the other four headings. A heading with no content is "
            "present-and-empty, which is its correct state — not omitted, and certainly "
            "not fabricated. Treating a heading as a required *fact* is what forced the "
            "omission/fabrication confusion in the first place."),
    },
)


# ---------------------------------------------------------------------------
# The classification is checkable against the text
# ---------------------------------------------------------------------------

@requires_run7
def test_every_objection_points_at_a_real_line_in_the_capture():
    """A classification is only evidence if it refers to something that exists.

    Line ids come from the engine's own numbering (``source_correction._lines``), which
    skips blank lines. Using that rather than counting lines here keeps the
    classification tied to the ids the reviewer actually emitted.
    """
    from source_correction import _lines

    data = live_capture.load(RUN7)
    by_id = {line["line_id"]: line["text"] for line in _lines(data["draft"])}

    for objection in OBJECTIONS:
        line_id = objection["line"]
        assert line_id in by_id, f"{line_id} is not a line of the captured draft"
        recorded = by_id[line_id]
        assert objection["draft"].strip()[:40] in recorded, (
            f"{line_id} classified text does not match the capture: {recorded!r}")


@requires_run7
def test_every_quoted_source_passes_is_actually_in_the_source():
    """The 'source' field is quoted evidence, so it must be verifiable.

    Compared against the source with its original line wrapping, because the capture
    wraps the summary paragraph mid-sentence. A quote that only matches after reflowing
    would not be a quote of this source.
    """
    data = live_capture.load(RUN7)
    source_lines = {line.strip() for line in data["sources"][0]["text"].splitlines()
                    if line.strip()}

    for objection in OBJECTIONS:
        for fragment in objection["source"].splitlines():
            if not fragment.strip():
                continue
            assert fragment.strip() in source_lines, (
                f"quoted source line not found verbatim: {fragment!r}")


@requires_run7
def test_lost_qualifier_objection_is_demonstrably_a_qualifier_loss():
    """``lost_qualifier`` is asserted by showing the qualifier really is gone.

    That is what makes it a classification rather than an opinion: the number survives
    and the word governing it does not.
    """
    data = live_capture.load(RUN7)
    source = data["sources"][0]["text"]
    draft = data["draft"]

    assert "8,000 hours" in source and "8,000 hours" in draft
    assert "supervised field work" in source
    assert "supervised field work" not in draft
    assert "hands-on field experience" in draft


@requires_run7
def test_omission_objection_asserts_nothing_false():
    """``omission`` must be demonstrated as absence, not as a false claim.

    This is the correction to the answer key. The proof that a line is an omission is
    that the source fact is missing from the draft AND the draft's own text contradicts
    nothing. If the draft asserted the opposite, it would be an unsupported addition.
    """
    data = live_capture.load(RUN7)
    source = data["sources"][0]["text"]
    draft = data["draft"]

    field = "Electrical Systems Technology"
    assert field in source, "the fact must be in the source for omission to apply"
    assert field not in draft, "the fact must be absent from the draft"

    education_line = next(line for line in draft.splitlines()
                          if line.strip().startswith("**Associate of Applied Science"))
    for token in ("Associate of Applied Science", "Example Technical College", "2016"):
        assert token in education_line and token in source, token


@requires_run7
def test_unsupported_addition_objection_names_words_the_source_lacks():
    """``unsupported_addition`` must name content the source does not establish."""
    data = live_capture.load(RUN7)
    source = data["sources"][0]["text"]
    draft = data["draft"]

    for added in ("installation, maintenance", "apprentice supervision"):
        assert added in draft, f"{added!r} should be present in the draft"
        assert added not in source, f"{added!r} is in the source, so it is not an addition"


@requires_run7
def test_secondary_unsupported_addition_is_separated_from_the_lost_qualifier():
    """A line can hold two defects that need two different repairs.

    L000014 has both a replaced qualifier and an invented trailing clause. Collapsing
    them into one class would produce a fix that either restores the qualifier and keeps
    the invention, or deletes the clause and leaves the wrong qualifier.
    """
    data = live_capture.load(RUN7)
    source = data["sources"][0]["text"]
    draft = data["draft"]

    assert "8,000 hours" in source
    assert "supervised" in source and "supervised" not in draft
    assert "hands-on" in draft and "hands-on" not in source
    assert "mastering core electrical installations" in draft
    assert "mastering core electrical installations" not in source


@requires_run7
def test_heading_objection_is_demonstrably_non_factual():
    """A heading asserts nothing, so 'missing' is the wrong verdict for it."""
    data = live_capture.load(RUN7)
    source = data["sources"][0]["text"]

    for heading in ("## Professional Summary", "## Professional Experience",
                    "## Education", "## Certifications", "## Skills"):
        assert heading in source
        assert not any(ch.isdigit() for ch in heading)


# ---------------------------------------------------------------------------
# The vocabulary and the consequence of each class
# ---------------------------------------------------------------------------

def test_every_objection_has_a_class_from_the_agreed_vocabulary():
    """The vocabulary is fixed, so a classification cannot be fudged into a new word."""
    for objection in OBJECTIONS:
        assert objection["class"] in CLASSES, objection["class"]
        secondary = objection.get("secondary")
        assert secondary is None or secondary in CLASSES, secondary
    for item in NON_DEFECTS:
        assert item["class"] in CLASSES, item["class"]


def test_the_classification_does_not_assume_the_reviewer_is_always_right():
    """A classification that never disagrees with the reviewer is not a classification.

    The conflict must be resolved against the text, not against the verdict.
    """
    assert NON_DEFECTS, "the classification contains no disagreement with the reviewer"
    classes = {item["class"] for item in NON_DEFECTS}
    assert classes & {"reviewer_error", "formatting"}, classes


def test_omission_is_not_recorded_as_fabrication():
    """The specific correction: a dropped fact is not a false claim.

    Where the product's own checkers reject a line, the temptation is to log it as
    fabrication because "rejected" and "invented" feel like the same thing. They are not,
    and the distinction decides whether the fix is coverage or removal.
    """
    education = next(o for o in OBJECTIONS if o["line"] == "L000017")
    assert education["class"] == "omission", (
        "an omitted degree field asserts nothing false and must not be called "
        "fabrication")
    assert education["class"] not in ASSERTION_CLASSES
    assert education["class"] in COVERAGE_CLASSES


def test_each_class_maps_to_the_fix_it_justifies():
    """State the mapping, so a later change cannot apply the wrong remedy.

    ``unsupported_addition`` -> remove or narrow the claim.
    ``lost_qualifier``       -> restore the governing word.
    ``omission``             -> carry the missing fact.
    ``formatting``           -> no content change; emit the heading.
    ``reviewer_error``       -> no code change at all.
    """
    remedies = {
        "unsupported_addition": "remove",
        "lost_qualifier": "restore",
        "omission": "carry",
        "formatting": "emit",
        "reviewer_error": "none",
    }
    assert set(remedies) == set(CLASSES)

    for objection in OBJECTIONS:
        # The PRIMARY class is what makes the row a defect, so it must justify a fix.
        assert remedies[objection["class"]] != "none", (
            f"{objection['line']} is a defect, so its primary class cannot map to no fix")
        # A SECONDARY class records a second thing found on the same line. It may be
        # ``reviewer_error`` — L000002's first clause was objected to wrongly — and that
        # correctly maps to no fix. The primary class already carries the remedy.
        secondary = objection.get("secondary")
        assert secondary is None or secondary in remedies

    for item in NON_DEFECTS:
        if item["class"] == "reviewer_error":
            assert remedies[item["class"]] == "none"
        else:
            # A heading is emitted; that is a structural requirement, not a content fix,
            # and it invents nothing.
            assert remedies[item["class"]] == "emit"


def test_no_defect_class_claims_more_than_the_evidence_allows():
    """Every row must carry reasoning, and none may assert support it lacks.

    The ``asserts_in_source`` field exists so a classification cannot quietly claim the
    draft is supported when it is not. The per-class tests above verify the textual side
    individually; this one keeps the bookkeeping honest.
    """
    for objection in OBJECTIONS:
        assert objection["asserts_in_source"] is False
        assert objection["why"].strip(), "a classification without reasoning is not one"
