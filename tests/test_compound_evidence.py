"""The addition check must not reject a compound the source states verbatim.

Layer: ``[engine]`` — the real ``_introduces_beyond`` and ``_content_words``, no
provider and no model. These tests prove a deterministic string comparison, never
that a model would word a resume this way.

Why this exists
---------------
Run 13 was withheld with ``source_completeness_uncertain`` after a reviewer marked
every draft line supported and every one of the 39 inventory items represented. The
engine downgraded exactly one representation:

    span       "- Lockout/tagout, NFPA 70E"
    item_quote "SYNTHETIC-ONLY Lockout/tagout, NFPA 70E"
    introduced ["Lockout/tagout"]

The source states ``Lockout/tagout`` on that very line, so this was a false
rejection, not a defect in the draft.

The cause is a vocabulary mismatch, not a marker artefact. ``_content_words`` scans
``[A-Za-z][A-Za-z'’+.-]*``, which has no ``/`` or ``#``, so the source contributes
the separate words ``lockout`` and ``tagout``. The entity scan in
``_introduces_beyond`` uses ``[A-Za-z][A-Za-z0-9'’+.#/-]*``, keeps the separator,
and therefore reads one token ``Lockout/tagout`` that the vocabulary can never
contain. Stripping provenance markers does NOT fix it — the compound is still one
token — which is why the earlier "strip the marker" proposal was wrong.

The fix compares the complete compound against the evidence text with its
separators intact and non-alphanumeric boundaries. It does not admit a compound
merely because its parts appear somewhere in the source: that is the weaker rule
these negative tests exist to forbid.
"""
import pytest

from source_correction import _compound_in_evidence, _content_words, _introduces_beyond

# The run-13 source line, verbatim from the preserved capture.
SOURCE = (
    "## Skills\n"
    "- SYNTHETIC-ONLY Conduit bending, switchgear troubleshooting, thermographic inspection\n"
    "- SYNTHETIC-ONLY Lockout/tagout, NFPA 70E\n"
)

# The run-13 inventory item and the span the engine downgraded.
ITEM = "SYNTHETIC-ONLY Lockout/tagout, NFPA 70E"
SPAN = "- Lockout/tagout, NFPA 70E"


# --------------------------------------------------------------------------- #
# The defect: the vocabulary really does store the parts separately.
# This is the premise of the fix, asserted so it cannot silently change.
# --------------------------------------------------------------------------- #

def test_content_words_stores_slash_compound_as_separate_parts():
    words = _content_words(ITEM)
    assert "lockout" in words
    assert "tagout" in words
    assert "lockout/tagout" not in words


def test_entity_scan_reads_the_compound_as_one_token():
    # The observable consequence of the mismatch above, before any fix.
    assert _introduces_beyond(SPAN, [SOURCE], ITEM) == []


# --------------------------------------------------------------------------- #
# Positive: the source states the compound, so it is not an addition.
# --------------------------------------------------------------------------- #

def test_verbatim_compound_is_not_reported_as_introduced():
    assert _introduces_beyond(SPAN, [SOURCE], ITEM) == []


def test_compound_comparison_is_case_normalized():
    assert _introduces_beyond("- LOCKOUT/TAGOUT, NFPA 70E", [SOURCE], ITEM) == []
    assert _compound_in_evidence("Lockout/Tagout", "lockout/tagout procedure")
    assert _compound_in_evidence("LOCKOUT/tagout", "Lockout/TAGOUT procedure")


def test_spacing_around_the_separator_is_tolerated():
    # "Lockout / tagout" and "Lockout/tagout" are the same term written apart.
    assert _introduces_beyond("- Lockout / tagout, NFPA 70E", [SOURCE], ITEM) == []
    assert _compound_in_evidence("Lockout/tagout", "uses Lockout / tagout daily")


def test_compound_found_in_the_item_quote_alone_is_enough():
    # The item is itself evidence; a draft restating its own item is faithful.
    assert _introduces_beyond(SPAN, [], ITEM) == []


# --------------------------------------------------------------------------- #
# Negative: the fix must NOT accept a compound on weaker grounds.
# --------------------------------------------------------------------------- #

def test_parts_present_separately_does_not_admit_the_compound():
    """The rule the fix must not become.

    Both words occur in the source, but never as the compound. A draft that fuses
    them is introducing a term the record does not state, and must still be
    reported.
    """
    split_source = "- lockout procedures and tagout procedures\n"
    split_item = "lockout procedures and tagout procedures"

    # Premise: both parts really are in the vocabulary.
    assert {"lockout", "tagout"} <= _content_words(split_source)

    assert _introduces_beyond("- Lockout/tagout", [split_source], split_item) == [
        "Lockout/tagout"
    ]
    assert not _compound_in_evidence("Lockout/tagout", split_source)


def test_only_one_part_present_does_not_admit_the_compound():
    one_part = "- lockout procedures\n"
    assert _introduces_beyond("- Lockout/tagout", [one_part], "lockout procedures") == [
        "Lockout/tagout"
    ]
    assert not _compound_in_evidence("Lockout/tagout", "lockout procedures")


def test_invented_compound_is_still_reported():
    """Neither part is in the record, so this is a genuine addition."""
    assert _introduces_beyond("- Honeywell/Tridium controls", [SOURCE], ITEM) == [
        "Honeywell/Tridium"
    ]
    assert not _compound_in_evidence("Honeywell/Tridium", SOURCE)


def test_boundary_prevents_a_substring_from_counting():
    # "Lockout/tagout" must not match inside "Lockout/tagoutprocedures".
    glued = "- Lockout/tagoutprocedures and more\n"
    assert not _compound_in_evidence("Lockout/tagout", glued)
    assert _introduces_beyond("- Lockout/tagout", [glued],
                              "Lockout/tagoutprocedures and more") == ["Lockout/tagout"]


def test_boundary_prevents_a_prefix_word_from_counting():
    prefixed = "- preLockout/tagout note\n"
    assert not _compound_in_evidence("Lockout/tagout", prefixed)


def test_run_4_appended_qualification_is_still_caught():
    """The control that keeps the addition check meaningful.

    Run 4 appended "Agile, Scrum, Gantt" to a line about inspection checklists.
    The compound fix must not blunt this: those are separate tokens, not a
    compound, and they are absent from the source.
    """
    assert _introduces_beyond("Agile, Scrum, Gantt", [SOURCE], ITEM) == [
        "Agile", "Gantt", "Scrum"
    ]


def test_multi_item_span_still_accepted_through_cited_evidence():
    """A compound is accepted only when a cited document contains it."""
    other = "- SYNTHETIC-ONLY Lockout/tagout written up for every job\n"
    assert _introduces_beyond("- Lockout/tagout", [other], "unrelated item") == []


# --------------------------------------------------------------------------- #
# Guard: non-compound tokens are unaffected by the new path.
# --------------------------------------------------------------------------- #

def test_helper_ignores_tokens_without_a_separator():
    # Hyphen and dot are already inside the content-word class, so they never
    # reach this helper.
    assert not _compound_in_evidence("NFPA", SOURCE)
    assert not _compound_in_evidence("Lockout", SOURCE)
    assert not _compound_in_evidence("", SOURCE)


@pytest.mark.parametrize("separator", ["/", "#"])
def test_both_separators_are_supported(separator):
    word = f"Alpha{separator}Beta"
    assert _compound_in_evidence(word, f"uses {word} on site")
    assert not _compound_in_evidence(word, "uses Alpha and Beta on site")
