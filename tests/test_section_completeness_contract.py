"""Every source-backed section must still carry content when the résumé is approved.

The honest statement before this contract existed was: "the system can produce
complete, correctly ordered résumés and has meaningful safety checks" — not
"every submission reliably produces a complete résumé with every required
section".

The gap was specific and measurable. A writer draft that omits Professional
Experience receives an empty heading from ``ensure_required_headings``; the
employer records vanish, and nothing deterministic stopped publication. Education,
Skills and Certifications were already protected by their restorers. Experience
and Projects were not, and inventing a restorer for them would compose prose out
of employer records, which the preservation rules refuse to do. The honest answer
for a lost section is to REFUSE, not to synthesise.

These tests cover the four cases the contract has to get right:

  1. a lost source-backed section blocks publication;
  2. a genuinely empty section is NOT a loss — the appearance decision;
  3. sparse and multi-document sources behave correctly;
  4. the same document yields the same decision in Markdown, PDF and DOCX.
"""
import pytest

import section_contract as contract
import source_preservation as sp


TWO_DOC = (
    "Professional Summary\n"
    "SYNTHETIC-ONLY Journeyman electrician with 12 years of electrical "
    "experience spanning commercial construction and facility operations.\n"
    "Professional Experience\n"
    "SYNTHETIC-ONLY Northline Fabrication, Inc. -- Spokane, WA\n"
    "SYNTHETIC-ONLY Welder / Maintenance Helper | 03/2012 - 09/2015\n"
    "SYNTHETIC-ONLY Fabricated and repaired structural steel assemblies for 14 "
    "industrial contracts.\n"
    "SYNTHETIC-ONLY Kettle Falls Electrical Contractors -- Spokane, WA\n"
    "SYNTHETIC-ONLY Electrical Apprentice | 10/2015 - 06/2019\n"
    "SYNTHETIC-ONLY Completed 8,000 apprenticeship hours under a licensed "
    "journeyman.\n"
    "Education\n"
    "SYNTHETIC-ONLY Spokane Falls Institute of Technology -- Spokane, WA\n"
    "Certifications\n"
    "SYNTHETIC-ONLY Washington 01 Journey Level Electrician License\n"
    "Skills\n"
    "SYNTHETIC-ONLY Conduit bending, switchgear troubleshooting\n"
)

COMPLETE = """## Professional Summary
Journeyman electrician with 12 years of electrical experience.

## Professional Experience
**Northline Fabrication, Inc.** -- Spokane, WA
*Welder / Maintenance Helper | 03/2012 - 09/2015*
- Fabricated and repaired structural steel assemblies for 14 industrial contracts.

**Kettle Falls Electrical Contractors** -- Spokane, WA
*Electrical Apprentice | 10/2015 - 06/2019*
- Completed 8,000 apprenticeship hours under a licensed journeyman.

## Education
**Spokane Falls Institute of Technology** -- Spokane, WA

## Certifications
- Washington 01 Journey Level Electrician License

## Technical Skills
- Conduit bending, switchgear troubleshooting
"""


def _sections(markdown):
    return [loss["section"] for loss in contract.shortfall(markdown, TWO_DOC)]


# ── 1. A lost source-backed section blocks publication ────────────────────


def test_a_complete_document_satisfies_the_contract():
    assert _sections(COMPLETE) == []


@pytest.mark.parametrize("heading,removed", [
    ("## Professional Experience",
     """## Professional Experience
**Northline Fabrication, Inc.** -- Spokane, WA
*Welder / Maintenance Helper | 03/2012 - 09/2015*
- Fabricated and repaired structural steel assemblies for 14 industrial contracts.

**Kettle Falls Electrical Contractors** -- Spokane, WA
*Electrical Apprentice | 10/2015 - 06/2019*
- Completed 8,000 apprenticeship hours under a licensed journeyman.
"""),
    ("## Education", "## Education\n**Spokane Falls Institute of Technology** -- Spokane, WA\n"),
    ("## Certifications", "## Certifications\n- Washington 01 Journey Level Electrician License\n"),
    ("## Technical Skills", "## Technical Skills\n- Conduit bending, switchgear troubleshooting\n"),
])
def test_dropping_a_source_backed_section_is_reported(heading, removed):
    """The section's content disappearing is a loss, whichever section it is."""
    without = COMPLETE.replace(removed, "")
    assert heading.split(" ", 1)[1] in _sections(without)


def test_an_empty_heading_where_the_source_has_content_is_a_loss():
    """A heading with no body is NOT completeness.

    This is the exact measured failure: ``ensure_required_headings`` inserts the
    heading, the records are gone, and the document claims a section it does not
    have. That is worse than omitting it.
    """
    dropped_body = COMPLETE.replace(
        """**Northline Fabrication, Inc.** -- Spokane, WA
*Welder / Maintenance Helper | 03/2012 - 09/2015*
- Fabricated and repaired structural steel assemblies for 14 industrial contracts.

**Kettle Falls Electrical Contractors** -- Spokane, WA
*Electrical Apprentice | 10/2015 - 06/2019*
- Completed 8,000 apprenticeship hours under a licensed journeyman.
""", "")
    # The heading survives; the content does not. That is still a loss.
    assert "## Professional Experience" in dropped_body
    assert "Professional Experience" in _sections(dropped_body)


# ── 2. Genuinely empty sections are not losses ───────────────────────────


def test_a_section_the_source_lacks_is_not_a_loss():
    """A candidate with no projects keeps an empty Projects heading, correctly.

    The source provides no Projects content, so an empty section is the honest
    appearance for "nothing to show". The contract must not report it.
    """
    source_without_projects = TWO_DOC
    assert "Projects" not in contract.source_backed_sections(source_without_projects)
    with_empty_projects = COMPLETE.replace(
        "## Education", "## Projects\n\n## Education")
    assert "Projects" not in _sections(with_empty_projects)


def test_a_sparse_resume_with_only_a_summary_is_not_a_loss():
    """Nothing is missing when the source has almost nothing either."""
    sparse = "Professional Summary\nSYNTHETIC-ONLY Electrician with 10 years.\n"
    assert contract.source_backed_sections(sparse) == {"Professional Summary": 1}
    delivered = "## Professional Summary\nElectrician with 10 years.\n"
    assert contract.shortfall(delivered, sparse) == []


def test_a_summary_heading_is_conditional_on_the_source():
    """An empty summary heading claims a summary the candidate never wrote."""
    source_no_summary = """Professional Experience
SYNTHETIC-ONLY Northline Fabrication, Inc. -- Spokane, WA
SYNTHETIC-ONLY Welder | 03/2012 - 09/2015
SYNTHETIC-ONLY Fabricated steel assemblies for 14 contracts.
"""
    assert "Professional Summary" not in contract.source_backed_sections(
        source_no_summary)
    delivered = ("## Professional Experience\n"
                 "**Northline Fabrication, Inc.** -- Spokane, WA\n"
                 "- Fabricated steel assemblies for 14 contracts.\n")
    assert contract.shortfall(delivered, source_no_summary) == []


# ── 3. Multi-document sources ────────────────────────────────────────────


def test_multi_document_inputs_are_judged_together():
    """Two documents compose one candidate; a loss in either is a loss."""
    doc_a = ("Professional Experience\n"
             "SYNTHETIC-ONLY Northline Fabrication, Inc. -- Spokane, WA\n"
             "SYNTHETIC-ONLY Welder | 03/2012 - 09/2015\n"
             "SYNTHETIC-ONLY Fabricated steel assemblies for 14 contracts.\n")
    doc_b = ("Certifications\n"
             "SYNTHETIC-ONLY Washington 01 Journey Level Electrician License\n")
    combined = doc_a + doc_b
    assert contract.source_backed_sections(combined).keys() == {
        "Professional Experience", "Certifications"}

    only_experience = ("## Professional Experience\n"
                       "**Northline Fabrication, Inc.** -- Spokane, WA\n"
                       "- Fabricated steel assemblies for 14 contracts.\n")
    assert "Certifications" in [l["section"]
                                for l in contract.shortfall(only_experience,
                                                            combined)]


def test_an_alias_heading_counts_as_the_canonical_section():
    """``## Skills`` satisfies ``Technical Skills``; do not report a false loss."""
    delivered = COMPLETE.replace("## Technical Skills", "## Skills")
    assert _sections(delivered) == []


# ── 4. The decision is format-independent ────────────────────────────────


def test_the_contract_decision_does_not_depend_on_the_export_format():
    """Markdown, PDF and DOCX render the same sections, so the contract agrees.

    The contract reads the approved Markdown, which is the single source both
    exporters render. This test pins that the ORDER and the section SET the
    exporters rely on are the same set the contract judges, so a document cannot
    pass here and render incompletely.
    """
    expected_order = list(contract.CONTRACTED)
    present = [name for name in expected_order
               if name in contract.canonical_sections(COMPLETE)]
    assert present == [h for h in expected_order if h in present]
    # Every section the contract requires is one ``REQUIRED_HEADINGS`` names,
    # which is the list the exporters emit headings for.
    assert set(contract.CONTRACTED) == set(sp.REQUIRED_HEADINGS)


# ── The contract must not become a prose comparer ────────────────────────


def test_a_faithful_paraphrase_satisfies_the_contract():
    """Content presence is the question; wording is the reviewer's.

    A reworded bullet is still content. If this contract started judging prose
    it would duplicate the preservation rules and reject legitimate rewriting.
    """
    paraphrased = COMPLETE.replace(
        "- Fabricated and repaired structural steel assemblies for 14 industrial "
        "contracts.",
        "- Assembled structural steel for industrial work across 14 contracts.")
    assert _sections(paraphrased) == []


def test_extra_sections_are_not_losses():
    """A document may carry sections the contract does not list."""
    extra = COMPLETE.replace("## Certifications",
                             "## Publications\n- Detector performance analysis.\n\n"
                             "## Certifications")
    assert _sections(extra) == []


# ── The appearance decision is documented, not accidental ────────────────


def test_the_empty_section_appearance_is_recorded_in_one_place():
    """Deciding how an empty section looks must not be rediscovered per renderer.

    The constant is documentation of measured behaviour. This test exists so the
    decision cannot drift silently: if a renderer changes, the constant and the
    note beside it must be updated deliberately.
    """
    appearance = contract.EMPTY_SECTION_APPEARANCE
    assert appearance["markdown"] == "heading_without_body"
    assert appearance["pdf"] == "heading_without_body"
    assert appearance["docx"] == "heading_without_body"
    # All five core sections are unconditional in every format.
    assert "All five CORE_SECTIONS are unconditional" in appearance["note"]
    assert contract.CORE_SECTIONS == (
        "Professional Experience", "Projects", "Education", "Certifications",
        "Technical Skills")


def test_every_core_section_is_present_in_the_rendered_docx():
    """RESOLVED POLICY: every core section is present, empty, nothing invented.

    Education and Certifications are conditional in the builder, so a sparse
    candidate's DOCX simply has no such heading. This pins the appearance the
    constant describes, so a reader can tell the omission is intended.
    """
    import base64
    import io

    from docx import Document

    from resume_builder.build_resume_from_export import (
        build_resume_from_payload_base64,
    )

    sparse = {
        "name": "Sparse Candidate", "headline": "Electrician",
        "summary": "Electrician with 10 years in commercial construction.",
        "skills": [{"category": "Trades", "items": ["Conduit bending"]}],
        "experience": [{"title": "Welder", "company": "Northline",
                        "duration": "03/2012 - 09/2015",
                        "bullets": ["Fabricated steel assemblies for 14 contracts."]}],
        "education": [], "certifications": [], "projects": [],
    }
    result = build_resume_from_payload_base64(dict(sparse))
    document = Document(io.BytesIO(base64.b64decode(result["docx_base64"])))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    # RESOLVED POLICY 2026-11-08: every core section is present, empty, and
    # nothing is invented under it. This test previously asserted the opposite
    # (that empty sections vanish), which was the behaviour the policy replaced.
    for heading in contract.CORE_SECTIONS:
        assert heading in paragraphs, heading
    for invented in ("None provided", "Statistical and Predictive Modeling"):
        assert invented not in paragraphs


# ── Scope: what the existing checks already cover ────────────────────────
#
# Deliberately NOT tested here: "the writer dropped every employer record".
# That case is already withheld by the existing ``source_facts_missing`` path,
# and ``tests/test_source_correction.py`` pins it. Asserting it here for
# ``source_section_missing`` would have been a test that passes on the wrong
# mechanism — and an attempt to write exactly that failed for that reason.
#
# This contract covers the gap those checks cannot see: a source that DOES carry
# section headings, where the writer keeps the heading and loses the content, so
# no per-item claim is reported missing because the section itself was never
# parsed against the source.


def test_the_contract_only_engages_for_sources_that_carry_headings():
    """A heading-less source is judged by the existing per-item checks.

    This states the boundary between the two mechanisms so a future reader does
    not think the contract is silently covering more than it is.
    """
    headingless = "Alex Example\nEngineer at Acme\n"
    assert contract.source_backed_sections(headingless) == {}
    assert contract.shortfall("## Professional Experience\n", headingless) == []

    headed = "Professional Experience\nAlex Example\nEngineer at Acme\n"
    assert "Professional Experience" in contract.source_backed_sections(headed)
    assert [l["section"] for l in
            contract.shortfall("## Professional Experience\n", headed)] == [
        "Professional Experience"]


def test_the_contract_actually_runs_on_the_approval_path():
    """Wiring, not logic: prove the contract is evaluated when a run approves.

    Every other test in this file would pass with the contract never being
    called — they exercise it directly, and a function that is correct and
    unreachable is worthless. This one counts the calls on a real approval and
    requires exactly one, so the wiring cannot be silently dropped.
    """
    import sys
    sys.path.insert(0, "tests")

    calls = {"n": 0}
    original = contract.shortfall

    def counting(markdown, source_text):
        calls["n"] += 1
        return original(markdown, source_text)

    contract.shortfall = counting
    sys.modules["section_contract"].shortfall = counting
    try:
        import test_source_correction as fixture
        result = fixture.run(fixture.COMPLETE_DRAFT, fixture.complete_callback(),
                             sources=fixture.COMPLETE_SOURCES, raw=True)
    finally:
        contract.shortfall = original
        sys.modules["section_contract"].shortfall = original

    assert result.status == "approved", (result.status, result.reason)
    assert calls["n"] == 1, (
        "the completeness contract was not evaluated on the approval path")


# ── The resolved policy, applied and tested in all three formats ────────

def _render_all(payload):
    """Render Markdown, PDF text and DOCX text for one payload."""
    import base64
    import io
    import shutil
    import subprocess
    import tempfile
    import pathlib

    from docx import Document
    from resume_builder.build_resume_from_export import (
        build_resume_from_payload_base64,
    )

    result = build_resume_from_payload_base64(dict(payload))
    document = Document(io.BytesIO(base64.b64decode(result["docx_base64"])))
    docx = "\n".join(p.text for p in document.paragraphs)

    pdf = ""
    if shutil.which("pdftotext"):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "r.pdf"
            path.write_bytes(base64.b64decode(result["pdf_base64"]))
            pdf = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                                 capture_output=True, text=True).stdout
    return docx, pdf


SPARSE_PAYLOAD = {
    "name": "Sparse Candidate", "headline": "Electrician",
    "summary": "Electrician with 10 years in commercial construction.",
    "skills": [], "experience": [], "education": [],
    "certifications": [], "projects": [],
}


def test_every_core_section_is_present_in_pdf_and_docx_when_empty():
    """The policy, applied: all five core sections appear even with no content.

    Measured before this change: Markdown carried all five, while PDF and DOCX
    carried only ``Projects``. Three artefacts of one run disagreed about which
    sections exist.
    """
    docx, pdf = _render_all(SPARSE_PAYLOAD)
    for heading in contract.CORE_SECTIONS:
        assert heading in docx, f"{heading} missing from DOCX"
        if pdf:
            assert heading in pdf, f"{heading} missing from PDF"


def test_nothing_is_invented_under_an_empty_section():
    """Present and empty — the heading names a section, it claims nothing.

    This is the other half of the policy. Filling an empty section with
    placeholder text would be the safe-looking mistake; it is asserted against
    directly.
    """
    docx, pdf = _render_all(SPARSE_PAYLOAD)
    invented = ("None provided", "N/A", "No experience", "Statistical and "
                "Predictive Modeling", "Boeing Defense", "Entry 1")
    for text in (docx, pdf):
        for phrase in invented:
            assert phrase not in text, f"{phrase!r} was invented"


def test_the_markdown_and_the_rendered_files_agree_on_the_section_set():
    """One run must not describe its own sections differently per artefact."""
    docx, pdf = _render_all(SPARSE_PAYLOAD)
    markdown_sections = set(sp.REQUIRED_HEADINGS)
    for heading in contract.CORE_SECTIONS:
        assert heading in markdown_sections, heading
        assert heading in docx, heading
        if pdf:
            assert heading in pdf, heading


def test_the_summary_heading_stays_conditional_everywhere():
    """The one deliberate exception, in all three formats.

    An empty summary heading would claim a summary the candidate never wrote, so
    it is conditional by product design rather than by renderer.
    """
    payload = dict(SPARSE_PAYLOAD)
    payload["summary"] = ""
    docx, pdf = _render_all(payload)
    assert "Professional Summary" not in docx
    if pdf:
        assert "Professional Summary" not in pdf


# ── Engine-level: the contract ALONE causes the refusal ─────────────────
#
# Item 1 of the brief: a case where upstream review genuinely succeeds, a
# source-backed section is empty, and the new contract is the only thing that
# refuses. This took work to construct, and the reason matters.
#
# For the obvious shape — "the writer deleted every employer record" — the
# existing per-item check already withholds with ``source_facts_missing``, so the
# contract never even runs. Asserting that shape here would produce a test that
# passes on the WRONG mechanism.
#
# The shape where the contract is genuinely alone is DISPLACEMENT: every source
# item is still present verbatim, so the reviewer can legitimately represent all
# of them, but the content sits under a heading that is not the core section. The
# per-item checks are satisfied; only a section-level check notices that
# Professional Experience is now empty while the source populates it.

ENGINE_SOURCE = (
    "Professional Experience\n"
    "Alex Example led diagnostics at Acme\n"
)
DISPLACED_DRAFT = (
    "# Candidate\r\n"
    "## Professional Experience\r\n"
    "## Additional Notes\r\n"
    "Alex Example led diagnostics at Acme\r\n"
)
INTACT_DRAFT = (
    "# Candidate\r\n"
    "## Professional Experience\r\n"
    "Alex Example led diagnostics at Acme\r\n"
)


def _engine_fixture():
    import sys
    sys.path.insert(0, "tests")
    import test_source_correction as fixture
    from source_correction import SourceDocument
    return fixture, SourceDocument("doc", ENGINE_SOURCE)


def test_the_contract_alone_causes_source_section_missing():
    """Upstream review succeeds; only the contract refuses, and nothing publishes.

    The reviewer confirms every line and represents every inventory item, so the
    per-item and coverage checks pass. Professional Experience is nevertheless
    empty because its content was displaced. The contract is what stops it.
    """
    fixture, source = _engine_fixture()
    result = fixture.run(DISPLACED_DRAFT, fixture.complete_callback(),
                         sources=(source,), raw=True)

    assert result.status == "withheld"
    assert result.reason == "source_section_missing", result.reason
    assert result.markdown is None, "a withheld run must publish no Markdown"
    # And the loss is real, not a contract artefact.
    assert [loss["section"] for loss in
            contract.shortfall(DISPLACED_DRAFT, ENGINE_SOURCE)] == [
        "Professional Experience"]


def test_the_same_document_with_its_content_in_place_is_approved():
    """The paired control: identical content, correct section, run approves.

    Without this, the test above would pass for a contract that refuses
    everything.
    """
    fixture, source = _engine_fixture()
    result = fixture.run(INTACT_DRAFT, fixture.complete_callback(),
                         sources=(source,), raw=True)

    assert result.status == "approved", (result.status, result.reason)
    assert result.markdown is not None
    assert contract.shortfall(INTACT_DRAFT, ENGINE_SOURCE) == []


# ── Scope limit: whole-section presence is NOT record-level completeness ─
#
# This is the boundary the contract must not be described as crossing. It answers
# "is this source-backed section still non-empty", NOT "did every qualification
# survive". The two are different questions and this test states the difference
# with a case that passes the contract while losing a record.

def test_a_section_with_one_surviving_record_satisfies_the_contract():
    """The contract passes when ONE record survives and another is lost.

    Both employers are in the source. The output keeps one and drops the other.
    The contract sees a non-empty Professional Experience and is satisfied. That
    is its documented limit, not a bug — a section-level check cannot count
    records it never enumerated against the draft.

    Recording it here keeps the claim honest: passing this contract is NOT proof
    that every qualification survived. Record-level loss is caught by the
    per-item coverage check (``source_facts_missing``) and by the reviewer, which
    is where that question belongs.
    """
    source = (
        "Professional Experience\n"
        "Northline Fabrication, Inc. -- Spokane, WA\n"
        "Welder | 03/2012 - 09/2015\n"
        "Fabricated steel assemblies for 14 contracts.\n"
        "Kettle Falls Electrical Contractors -- Spokane, WA\n"
        "Apprentice | 10/2015 - 06/2019\n"
        "Completed 8,000 apprenticeship hours.\n"
    )
    # One employer kept, the other gone entirely.
    output = (
        "## Professional Experience\n"
        "**Northline Fabrication, Inc.** -- Spokane, WA\n"
        "- Fabricated steel assemblies for 14 contracts.\n"
    )
    assert contract.shortfall(output, source) == [], (
        "the contract is a whole-section check; it is not expected to catch a "
        "record-level loss, and this assertion documents that limit")
    # The lost record IS detectable by inspection, just not by this check.
    assert "Kettle Falls" not in output
    assert "Kettle Falls" in source


def test_the_contract_documents_its_own_scope():
    """The module states what it proves AND what it does not.

    A check whose docstring only lists what it catches invites the reader to
    assume that is all there is to catch. The overstatement this replaces called
    the module "a completeness contract", which a whole-section check cannot
    honour: it counts sections and never counts facts.
    """
    doc = contract.__doc__ or ""
    lowered = doc.lower()
    # The scope statement must be about sections, not about completeness.
    assert "whole-section" in lowered, (
        "the docstring must name the check as whole-section presence")
    assert "not a completeness proof" in lowered or \
        "not prove the résumé is complete" in lowered, (
        "the docstring must explicitly deny being a completeness proof")
    # And it must name the record-level limit concretely.
    assert "record-level" in lowered
    assert "one surviving record" in lowered


def test_the_docstring_does_not_claim_completeness_it_cannot_deliver():
    """Guard against the wording drifting back into overstatement.

    Checked as CLAIMS, not as raw substrings. The docstring legitimately uses
    phrases like "that every qualification survived" inside the sentence "it does
    NOT establish that every qualification survived". A naive substring ban fails
    on the very denial it is trying to require, so this looks for the claim
    without a negation in front of it.
    """
    import re

    doc = (contract.__doc__ or "").lower()
    for phrase in ("every qualification", "every submission reliably"):
        for match in re.finditer(re.escape(phrase), doc):
            before = doc[max(0, match.start() - 120):match.start()]
            negated = re.search(r"(\bnot\b|never|cannot|doesn't|n't)", before)
            assert negated, (
                f"the docstring asserts {phrase!r} without denying it, which "
                "this check cannot establish")
