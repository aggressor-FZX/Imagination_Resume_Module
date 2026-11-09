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


def test_a_skills_page_on_its_own_is_never_the_items_own_document():
    """Why the earlier, weaker test was wrong to accept this outcome.

    ``_introduces_beyond`` is a pure function of the evidence it is HANDED. Fed
    only a skills page it reports nothing, because from its point of view the
    tool is documented. The defect was never in this helper: it was in which
    documents ``_validate_review`` chose to hand it.

    That is why this file asserts the boundary at the validator
    (``test_compound_citation_does_not_establish_role_tool_use``) rather than
    here. This test pins the helper's documented behaviour so a future reader
    does not "fix" the wrong layer.
    """
    span = "- Maintained parts records for 7 branches using Tableau."
    assert sc._introduces_beyond(span, [SKILLS_DOC], ROLE_ITEM) == [], (
        "the helper changed shape; the real boundary test is in _validate_review")


# ── compound citation: the demonstrated defect ────────────────────────────
#
# Measured 2026-11-08. For the same bullet, citing the ROLE alone flagged
# "Tableau", but citing the ROLE **and** the candidate's skills page returned
# nothing. Attaching a skills list to the citation set silently established tool
# use inside the role, even though the role record never mentions the tool.
#
# The fix drops a skills-only document from the evidence set when it is not the
# item's own document. A document that merely CONTAINS a skills section is left
# alone, so genuine cross-document synthesis between role records still works.

ROLE_DOC = "Maintained parts records for 7 branches.\n"
ROLE_QUOTE = "Maintained parts records for 7 branches.\n"
ROLE_START = 0
ROLE_END = len(ROLE_QUOTE)
ROLE_INVENTORY = [{
    "id": f"document-0:{ROLE_START}:{ROLE_END}", "source_id": "document-0",
    "start": ROLE_START, "end": ROLE_END, "quote": ROLE_QUOTE,
    "classification": "required",
}]
TOOL_LINE = "- Maintained parts records for 7 branches using Tableau."
TOOL_LINES = [{"line_id": "L1", "text": TOOL_LINE}]


def _review(cite_skills_page):
    citations = [{"source_id": "document-0", "start": ROLE_START,
                  "end": ROLE_END, "quote": ROLE_QUOTE}]
    if cite_skills_page:
        citations.append({"source_id": "document-1", "start": 0,
                          "end": len(SKILLS_DOC), "quote": SKILLS_DOC})
    return {
        "claims": [{"id": f"L1:0:{len(TOOL_LINE)}", "line_id": "L1", "start": 0,
                    "end": len(TOOL_LINE), "verdict": "supported",
                    "citations": citations}],
        "representations": [{
            "source_claim_id": ROLE_INVENTORY[0]["id"], "status": "represented",
            "spans": [{"line_id": "L1", "start": 0, "end": len(TOOL_LINE)}]}],
    }


def _validate(extra_doc):
    sources = {"document-0": ROLE_DOC, "document-1": SKILLS_DOC}
    response = _review(extra_doc)
    claims, incomplete, _ = sc._validate_review(
        response, TOOL_LINES, sources, ROLE_INVENTORY)
    return incomplete, response


def test_citing_the_role_alone_flags_the_tool_attribution():
    """The baseline: without a skills page attached, it is flagged."""
    incomplete, response = _validate(False)
    assert "uncertain" in incomplete
    assert response["representations"][0]["status"] == "uncertain"


def test_compound_citation_does_not_establish_role_tool_use():
    """THE DEFECT: attaching the skills page must not clear the attribution.

    Citing the role record and the candidate's skills page together previously
    returned no findings, so the representation stayed ``represented`` and the
    run could deliver a per-role tool claim the role never made.
    """
    incomplete, response = _validate(True)
    assert "uncertain" in incomplete, (
        "citing the candidate's skills page alongside the role authorised "
        "per-role tool use")
    assert response["representations"][0]["status"] == "uncertain"


def test_a_document_that_merely_contains_skills_is_still_evidence():
    """Only an ENTIRELY skills-only document is set aside.

    A real role record that happens to end with a skills section must keep
    contributing evidence, or legitimate synthesis between role records breaks.
    """
    mixed = ("Maintained parts records for 7 branches.\n"
             "Trained 6 apprentices.\nSkills\nSQL, Excel\n")
    assert sc._is_skills_only_document(mixed) is False


def test_a_pure_skills_page_is_recognised():
    assert sc._is_skills_only_document(SKILLS_DOC) is True


def test_a_record_with_a_year_is_not_a_skills_page():
    """A dated line marks an employment record, not a skill list."""
    dated = "Maintained parts records.\n2019 - Present\n"
    assert sc._is_skills_only_document(dated) is False


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
