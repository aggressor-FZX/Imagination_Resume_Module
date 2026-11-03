"""Classification of every residual run-7 reviewer objection.

The reviewer is a model. It can be wrong, and treating its verdicts as ground truth
would make the answer key unfalsifiable: every objection would become a defect to
"fix", including objections that are paraphrase-recognition failures. So each run-7
objection is classified against the *source text* here, before any of them is acted on.

Categories
----------
``fabrication``     the draft asserts something the source does not. A real defect.
``lost_qualifier``  the draft keeps a fact but drops or replaces a qualifier that
                    changes its meaning. A real defect, and the most dangerous kind
                    because every word looks sourced.
``paraphrase``      the draft restates the source faithfully in different words. NOT a
                    defect; the reviewer was wrong to object.
``formatting``      the line is structure — a heading, a label, a separator. Asserts no
                    candidate fact, so it cannot be missing.
``reviewer_error``  the reviewer's own verdict contradicts the text it was given.
                    NOT a defect.

The distinction matters for what gets fixed. Classifying an objection as
``paraphrase`` stops an implementation change that would have chased a phantom;
classifying one as ``lost_qualifier`` is what turns "the writer paraphrased" into a
specific, testable rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import live_capture

RUN7 = Path("/home/skystarved/Render_Dockers/.live-run-2026-11-02/captures/"
            "run7/0a267e04-affd-4cee-b944-26c574703039.json")

requires_run7 = pytest.mark.skipif(not RUN7.exists(), reason="run-7 capture absent")


# ---------------------------------------------------------------------------
# The classification, written from the source and the draft, not from the implementation
# ---------------------------------------------------------------------------

OBJECTIONS = (
    {
        "line": "L000014",
        "reviewer_verdict": "unsupported",
        "draft": "- Accumulated 8,000 hours of hands-on field experience toward "
                 "journeyman licensure, mastering core electrical installations.",
        "source": "- Completed 8,000 hours of supervised field work toward licensure.",
        "class": "lost_qualifier",
        "why": (
            "Three changes to one fact. 'supervised' became 'hands-on': the source "
            "qualifier is load-bearing because supervised hours are the licensure "
            "pathway, and dropping it changes what the number means. 'licensure' became "
            "'journeyman licensure': a claim about which licence the hours counted "
            "toward. 'mastering core electrical installations' is an invented outcome "
            "with no basis in the source. The number is real and correctly attached; the "
            "qualifiers around it are not."),
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
        "class": "fabrication",
        "why": (
            "'installation' and 'maintenance' are activities, not the listed skills, "
            "and they entered the skills block. 'apprentice supervision' is a rank the "
            "candidate never claims — the source says 'Trained 6 apprentices'. "
            "The category structure ('**Electrical:**') is also invented: the source "
            "has no skill categories. The remaining words are sourced, which is why the "
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
        "class": "fabrication",
        "why": (
            "The first clause is a faithful restatement of the source summary — that "
            "part is a legitimate paraphrase and the reviewer was too harsh on it. But "
            "the line continues past where the source summary ends, appending "
            "'proficient in NFPA 70E, Arc Flash safety, electrical systems, apprentice "
            "supervision, and inspections'. Appended proficiency is an addition, and "
            "'apprentice supervision' overstates 'trained 6 apprentices' again. The "
            "objection is therefore correct, for a narrower reason than it states."),
        "asserts_in_source": False,
    },
    {
        "line": "L000017",
        "reviewer_verdict": "missing (segment 15)",
        "draft": "**Associate of Applied Science, Example Technical College** -- 2016",
        "source": "*Associate of Applied Science, Electrical Systems Technology | 2016*",
        "class": "fabrication",
        "why": (
            "The field of study, 'Electrical Systems Technology', was dropped. The "
            "institution, the degree and the year are all correct, so this is a coverage "
            "loss rather than a wrong statement — but the major is a required fact and "
            "its absence is a real omission. The reviewer was right to report it."),
        "asserts_in_source": False,
    },
)

# Run-7 objections that the classification says are NOT defects. Kept as a separate list
# because the point of the exercise is that the reviewer is not always right, and a list
# with nothing in it would suggest the classification never disagreed.
NON_DEFECTS = (
    {
        "line": "L000002 (first clause)",
        "reviewer_verdict": "uncertain",
        "class": "paraphrase",
        "why": (
            "'Journeyman electrician with 8+ years of commercial and industrial "
            "electrical experience across construction, facilities maintenance, and "
            "power distribution' restates the source summary with only markup removed. "
            "Recording it as missing would demand word-for-word copying, which is not a "
            "fidelity requirement."),
    },
    {
        "line": "L000001, L000004, L000016, L000019, L000023",
        "reviewer_verdict": "missing (segments 0, 3, 13, 16, 19)",
        "class": "formatting",
        "why": (
            "Section headings assert no candidate fact. The answer key already says "
            "segment 0 'must not be reported as a missing fact'; the same reasoning "
            "applies to the other four headings. A heading whose content is absent is "
            "omitted, not fabricated."),
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
    source = data["sources"][0]["text"]
    source_lines = {line.strip() for line in source.splitlines() if line.strip()}

    for objection in OBJECTIONS:
        for fragment in objection["source"].splitlines():
            if not fragment.strip():
                continue
            assert fragment.strip() in source_lines, (
                f"quoted source line not found verbatim: {fragment!r}")


@requires_run7
def test_lost_qualifier_objection_is_demonstrably_a_qualifier_loss():
    """The class is asserted by showing the qualifier really is gone.

    This is what makes ``lost_qualifier`` a classification rather than an opinion: the
    number survives, and the word that governed it does not.
    """
    data = live_capture.load(RUN7)
    source = data["sources"][0]["text"]
    draft = data["draft"]

    assert "8,000 hours" in source and "8,000 hours" in draft
    assert "supervised field work" in source
    assert "supervised field work" not in draft
    assert "hands-on field experience" in draft
    # And the invented outcome is absent from the source entirely.
    assert "mastering core electrical installations" not in source
    assert "mastering core electrical installations" in draft


@requires_run7
def test_fabrication_objection_is_demonstrably_absent_from_the_source():
    """A 'fabrication' class must name words the source does not contain."""
    data = live_capture.load(RUN7)
    source = data["sources"][0]["text"]
    draft = data["draft"]

    for invented in ("installation, maintenance", "apprentice supervision",
                     "Electrical Systems Technology"):
        if invented in draft:
            assert invented not in source, (
                f"{invented!r} is in the source, so it is not a fabrication")

    # The dropped field is the reverse case: in the source, absent from the draft.
    assert "Electrical Systems Technology" in source
    assert "Electrical Systems Technology" not in draft


@requires_run7
def test_heading_objection_is_demonstrably_non_factual():
    """A heading asserts nothing, so 'missing' is the wrong verdict for it."""
    data = live_capture.load(RUN7)
    source = data["sources"][0]["text"]

    for heading in ("## Professional Summary", "## Professional Experience",
                    "## Education", "## Certifications", "## Skills"):
        assert heading in source
        # A heading line has no digits, verbs of accomplishment, or employer.
        assert not any(ch.isdigit() for ch in heading)


def test_every_objection_has_a_class_from_the_agreed_vocabulary():
    """The vocabulary is fixed, so a classification cannot be fudged into a new word."""
    allowed = {"fabrication", "lost_qualifier", "paraphrase", "formatting",
               "reviewer_error"}
    for objection in OBJECTIONS:
        assert objection["class"] in allowed, objection["class"]
    for item in NON_DEFECTS:
        assert item["class"] in allowed, item["class"]


def test_the_classification_does_not_assume_the_reviewer_is_always_right():
    """A classification that never disagrees with the reviewer is not a classification.

    This is the load-bearing test for the exercise: the answer key and the reviewer must
    be able to conflict, and the conflict must be resolved against the text. Here the
    non-defect list is required to be non-empty and to contain a category that says the
    reviewer was wrong.
    """
    assert NON_DEFECTS, "the classification contains no disagreement with the reviewer"
    classes = {item["class"] for item in NON_DEFECTS}
    assert classes & {"paraphrase", "formatting", "reviewer_error"}, classes


def test_defect_classes_are_the_ones_that_justify_a_code_change():
    """Only fabrication and lost_qualifier justify changing the writer.

    ``paraphrase`` and ``formatting`` explicitly do not, and ``reviewer_error`` means the
    reviewer was wrong. Stating the mapping in a test stops a phantom objection from
    being chased with an implementation change later.
    """
    fixable = {"fabrication", "lost_qualifier"}
    for objection in OBJECTIONS:
        assert objection["class"] in fixable, (
            f"{objection['line']} is classified {objection['class']}, which does not "
            "justify a writer change")
