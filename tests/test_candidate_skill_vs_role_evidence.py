"""A candidate-level skill is not evidence of tool use in a particular role.

The candidate's document may list a tool once, at the top, as a skill they have.
That is a claim about the CANDIDATE. It is not evidence that the tool was used
inside any one employer's bullet, and it must not authorise the writer to attach
it to a specific role.

This distinction is easy to lose. One earlier attempt widened the evidence set
for the introduced-content check to "every document the run touched", which
would have made the candidate's own skills page justify any per-role attribution
the writer invented. It was measured, it changed nothing for the case it was
aimed at, and it was reverted — these tests are what stop it coming back.

The rule under test:

  * a tool named in the document an item CAME FROM, on that item's own line, may
    support restating that item;
  * a tool named only on a different document's skills list does not.
"""
import source_correction as sc


# document-0 is the candidate's skills summary. document-1 is the role record.
SKILLS_DOC = "Skills\nSQL, Excel, Tableau\n"
ROLE_DOC = "Maintained parts records for 7 branches.\n"
ROLE_ITEM = "SYNTHETIC-ONLY Maintained parts records for 7 branches.\n"


def test_a_candidate_level_tool_does_not_justify_per_role_tool_use():
    """Listing Tableau as a skill must not license "using Tableau" in a role."""
    span = "- Maintained parts records for 7 branches using Tableau."
    introduced = sc._introduces_beyond(span, [ROLE_DOC], ROLE_ITEM)
    assert introduced == ["Tableau"], (
        "a tool from the candidate's skills list was accepted as evidence for "
        "tool use inside a specific role")


def test_citing_the_skills_page_does_authorise_the_attribution_TODAY():
    """Documents the CURRENT behaviour, including its known weakness.

    This is deliberately written as a description, not a wish. When the skills
    page is the cited evidence, the check treats "Tableau" as known and the
    attribution passes — even though the skills page says nothing about where the
    tool was used.

    It is recorded rather than changed because no evidence shows this path being
    reached in production: a representation is only accepted when the claim's
    citations COVER the item's own source offsets, so the item's own document is
    always among the cited documents, and the normal case (below) does flag the
    attribution. Changing it without a demonstrated failure would risk the
    legitimate multi-document synthesis the check is documented to permit.

    If a live run is ever captured where a per-role tool attribution passes only
    because the candidate's skills page was cited, this test is the place to
    assert the stricter rule.
    """
    span = "- Maintained parts records for 7 branches using Tableau."
    introduced = sc._introduces_beyond(span, [SKILLS_DOC], ROLE_ITEM)
    assert introduced == [], (
        "behaviour changed: citing the skills page no longer authorises the "
        "attribution. That is stricter than the recorded behaviour — update this "
        "documentation and add a live-capture-backed test.")


def test_a_tool_named_on_the_items_own_line_is_evidence():
    """The same tool IS acceptable when the role record itself states it."""
    role = "Maintained parts records for 7 branches using Tableau.\n"
    item = "SYNTHETIC-ONLY Maintained parts records for 7 branches using Tableau.\n"
    assert sc._introduces_beyond(
        "- Maintained parts records for 7 branches using Tableau.",
        [role], item) == []


def test_an_invented_tool_is_still_refused():
    """The control: a name in neither place must never pass."""
    span = "- Maintained parts records for 7 branches using Snowflake."
    introduced = sc._introduces_beyond(span, [ROLE_DOC], ROLE_ITEM)
    assert introduced == ["Snowflake"]


def test_an_invented_figure_is_still_refused():
    """Numbers are a separate, independent guard and must keep firing."""
    span = "- Maintained parts records for 7 branches, cutting errors by 42%."
    introduced = sc._introduces_beyond(span, [ROLE_DOC], ROLE_ITEM)
    assert "42" in introduced
