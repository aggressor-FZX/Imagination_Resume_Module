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

## What is fixed here and what is not

FIXED (``_validate_review``): a candidate-level skills page cited alongside a
role no longer establishes tool use inside that role. See
``test_candidate_skill_vs_role_evidence.py``.

NOT FIXED: the deterministic check catches a capitalised NAME or an unbacked
FIGURE in an added clause. It does not catch added LOWERCASE qualifiers, and
``blueprint specifications`` / ``state standards`` are exactly that shape. The
tests below mark those cases ``xfail`` with a strict reason rather than being
deleted, softened, or quietly skipped: the requirement stands and is unmet.
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


@pytest.mark.xfail(
    strict=True,
    reason="UNFIXED: an added lowercase qualifier ('using blueprint "
           "specifications') is not caught - the check only flags capitalised "
           "names and unbacked figures in added clauses",
)
def test_blueprint_specifications_addition_is_caught():
    """THE RETAINED DEFECT, verbatim."""
    span = ("- Fabricated structural steel assemblies for 14 industrial contracts "
            "using blueprint specifications.")
    assert sc._introduces_beyond(span, [SOURCE], FAB_ITEM) != []


@pytest.mark.xfail(
    strict=True,
    reason="UNFIXED: 'toward journeyman certification in compliance with state "
           "standards' adds no capitalised name and no figure",
)
def test_state_standards_addition_is_caught():
    """THE RETAINED DEFECT, verbatim."""
    span = ("- Completed 8,000 apprenticeship hours toward journeyman "
            "certification in compliance with state standards.")
    assert sc._introduces_beyond(span, [SOURCE], APP_ITEM) != []


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
