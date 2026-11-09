"""The retained Imaginative false approval, as a desired-behaviour regression.

Run ``43917767-152b-441c-8c72-a4abe71033ea`` delivered two sentences the
candidate never wrote:

  * "- Fabricated structural steel assemblies for 14 industrial contracts using
    blueprint specifications."   — the source says only "Fabricated and repaired
    structural steel assemblies for 14 industrial contracts."
  * "- Completed 8,000 apprenticeship hours toward journeyman certification in
    compliance with state standards."   — the source says "Completed 8,000
    apprenticeship hours under a licensed journeyman."

Neither is covered by a surviving bridge, so neither is authorised Imaginative
inference. Both reached the delivered document.

This file pins the requirement in the terms it was given:

  * unsupported additions must be corrected or withheld;
  * faithful paraphrases must still pass.

## Both defects are fixed, at different layers

1. ``_validate_review``: a candidate-level skills page cited alongside a role no
   longer establishes tool use inside that role. See
   ``test_candidate_skill_vs_role_evidence.py``.

2. ``source_preservation``: the retained additions ARE now corrected. The check
   that missed them was never the right place to catch them — it compares
   content words and only fires on capitalised names and unbacked figures, and no
   lexical measure separates ``using blueprint specifications`` from the faithful
   ``through targeted diagnostics`` (both introduce novel words; measured).
   What actually distinguishes them is SHAPE: each is a trailing qualifying
   phrase that the prefix does not contain. ``_TRAILING_PURPOSE_BOUNDARY``
   recognised ``to`` / ``for`` / ``so that`` / ``in order to`` but not
   ``using`` / ``toward`` / ``in compliance with``, so the existing, already
   tested trimmer never inspected these clauses.

   Widening the boundary is not a phrase blacklist: it only decides WHERE to
   look. The trimmer's measured rules still decide whether anything is removed,
   and a mixed tail substitutes the candidate's own source bullet rather than
   composing prose. Result on the retained draft: ``blueprint specifications`` is
   trimmed, and the apprenticeship bullet is restored to the source's own
   wording.

``test_faithful_paraphrases_are_not_flagged`` pins the four accepted paraphrases
against the SAME widened boundary, so a future change cannot buy detection by
breaking legitimate rewriting.
"""
import pytest

import source_correction as sc


SOURCE = (
    "Fabricated and repaired structural steel assemblies for 14 industrial "
    "contracts.\n"
    "Completed 8,000 apprenticeship hours under a licensed journeyman.\n"
    "Cut average fault-response time from 4.5 hours to 2.75 hours.\n"
    "Reduced after-hours callouts by 27% by adding thermographic inspections.\n"
)

FAB_ITEM = ("SYNTHETIC-ONLY Fabricated and repaired structural steel assemblies "
            "for 14 industrial contracts.\n")
APP_ITEM = ("SYNTHETIC-ONLY Completed 8,000 apprenticeship hours under a licensed "
            "journeyman.\n")
CUT_ITEM = ("SYNTHETIC-ONLY Cut average fault-response time from 4.5 hours to "
            "2.75 hours.\n")
RED_ITEM = ("SYNTHETIC-ONLY Reduced after-hours callouts by 27% by adding "
            "thermographic inspections.\n")


# ── The two delivered additions ───────────────────────────────────────────


RETENTION_DRAFT = """## Professional Experience
**Northline Fabrication, Inc.** -- Spokane, WA
*Welder / Maintenance Helper | 03/2012 - 09/2015*
- Fabricated structural steel assemblies for 14 industrial contracts using blueprint specifications.

**Kettle Falls Electrical Contractors** -- Spokane, WA
*Electrical Apprentice | 10/2015 - 06/2019*
- Completed 8,000 apprenticeship hours toward journeyman certification in compliance with state standards.
"""

RETENTION_SOURCE = (
    "Professional Experience\n"
    "Northline Fabrication, Inc. -- Spokane, WA\n"
    "Welder / Maintenance Helper | 03/2012 - 09/2015\n"
    "Fabricated and repaired structural steel assemblies for 14 industrial "
    "contracts.\n"
    "Kettle Falls Electrical Contractors -- Spokane, WA\n"
    "Electrical Apprentice | 10/2015 - 06/2019\n"
    "Completed 8,000 apprenticeship hours under a licensed journeyman.\n"
)


def _preserved(markdown=RETENTION_DRAFT, source=RETENTION_SOURCE):
    from source_preservation import preserve_source_facts
    return preserve_source_facts(markdown, source)


@pytest.mark.parametrize("addition", [
    "blueprint specifications",
    "journeyman certification",
    "state standards",
])
def test_the_retained_additions_are_removed_by_preservation(addition):
    """THE RETAINED DEFECT, verbatim, through the real correction path.

    These are NOT asserted against ``_introduces_beyond``. That helper compares
    content words and only fires on capitalised names and unbacked figures, and
    no lexical measure separates these clauses from the faithful paraphrases
    below — ``through targeted diagnostics`` introduces just as many novel words.
    The cause was READER-side: the trailing-purpose boundary did not recognise
    ``using`` / ``toward`` / ``in compliance with``, so the existing trimmer
    never looked at these clauses.

    Asserting at the helper would have proved nothing while looking green.
    """
    out, _changes = _preserved()
    assert addition not in out, (
        f"{addition!r} survived preservation and would have been published")


def test_the_retained_additions_module_message_stays_faithful():
    """The corrections must not damage the facts they were attached to."""
    out, _changes = _preserved()
    assert ("Fabricated structural steel assemblies for 14 industrial contracts"
            in out)
    assert "8,000 apprenticeship hours" in out
    assert "journeyman" in out


def test_preservation_reports_the_corrections_it_made():
    """A silent correction and a missed correction look the same from outside."""
    out, changes = _preserved()
    bullet_changes = [c for c in changes
                      if "appended clause" in c or "mixed clause" in c]
    assert bullet_changes, f"no bullet correction was reported: {changes}"
    assert len(out) > 0


# ── Additions the check DOES catch (these must never regress) ─────────────


def test_an_invented_tool_in_an_added_clause_is_caught():
    span = ("- Fabricated structural steel assemblies for 14 industrial contracts "
            "using Agile and Scrum.")
    assert "Agile" in sc._introduces_beyond(span, [SOURCE], FAB_ITEM)


def test_an_invented_figure_in_an_added_clause_is_caught():
    span = ("- Fabricated structural steel assemblies for 14 industrial contracts, "
            "cutting rework by 42%.")
    assert "42" in sc._introduces_beyond(span, [SOURCE], FAB_ITEM)


def test_an_invented_credential_is_caught():
    span = "- Completed 8,000 apprenticeship hours, earning a PMP certification."
    assert sc._introduces_beyond(span, [SOURCE], APP_ITEM) != []


# ── Faithful paraphrases must still pass ─────────────────────────────────


@pytest.mark.parametrize("span,item", [
    # Conservative-mode paraphrases accepted on the final build.
    ("- Fabricated structural steel assemblies for 14 industrial contracts "
     "using welding.", FAB_ITEM),
    ("- Completed 8,000 apprenticeship hours under supervision of a licensed "
     "journeyman electrician.", APP_ITEM),
    ("- Cut average fault-response time from 4.5 hours to 2.75 hours through "
     "targeted diagnostics.", CUT_ITEM),
    ("- Reduced after-hours callouts by 27% by implementing thermographic "
     "inspections.", RED_ITEM),
])
def test_faithful_paraphrases_are_not_flagged(span, item):
    """These delivered on the accepted Conservative run and must keep passing.

    Any future change that makes the additions detectable by Flagging every
    novel lowercase word will fail here first, which is the point: the
    requirement is to catch the first two WITHOUT breaking these four.
    """
    assert sc._introduces_beyond(span, [SOURCE], item) == []


def test_the_control_figures_are_not_flagged():
    """The candidate's own numbers restated must never count as additions."""
    span = "- Cut average fault-response time from 4.5 hours to 2.75 hours."
    assert sc._introduces_beyond(span, [SOURCE], CUT_ITEM) == []
