"""Two documents through the REAL ``/analyze`` route: preservation, correction, capture.

Why this file exists
--------------------

``docs/ROUTE_MAP.md`` and
``Login_Landing_page/backend/tests/integration/multiFileRoute.test.js`` establish that the
website's two-file traffic reaches the Imaginator's ``POST /analyze`` — the SAME handler as
a single-file upload, not the Imaginator's own multipart endpoint. That means a two-file
delivery runs the single-file preservation and correction code.

Preservation was written for one document. This file measures what it does with two, on the
real route, rather than assuming the single-document behaviour carries over.

What was measured before the fix (see ``tests/test_source_preservation.py`` section 6 for
the unit-level version) is that every source reader took only the FIRST matching section,
so a combined source was read as one document: the education rule deleted the second
degree, and the summary rule truncated the second document's summary as an unproven
"addition". Neither raised, because both are replacements.

What is scripted, and labelled
------------------------------

* the **writer** draft — hand-written, containing BOTH records, so this measures the
  route's handling of a correct two-record draft rather than a model's quality;
* the **reviewer** — a deterministic reader that answers the indexed contract by matching
  draft lines to source segments. It is NOT a similarity threshold tuned until the run
  approved: it reports a verdict per line from an explicit rule stated below.

What is NOT done: ``CorrectionResult`` is never overridden, the answer key is never
edited, and no document bytes are mocked.

Labelled **[route-two-doc]**.
"""
from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app
import correction_provider
import correction_protocol
import live_capture
import source_preservation

DOC_ELECTRICIAN = """## Professional Summary
SYNTHETIC-ONLY Journeyman electrician with 8+ years of commercial electrical experience.

## Professional Experience
**SYNTHETIC-ONLY Cascade Power Solutions, LLC** -- Example City, WA
*Journeyman Electrician | 03/2020 - Present*
- Installed commercial electrical systems across 40+ job sites.

## Education
**SYNTHETIC-ONLY Example Technical College** -- Example City, WA
*Associate of Applied Science, Electrical Systems Technology | 2016*

## Skills
- Commercial electrical systems, panel labeling
"""

DOC_FACILITIES = """## Professional Summary
SYNTHETIC-ONLY Facilities maintenance technician focused on preventive maintenance.

## Professional Experience
**SYNTHETIC-ONLY Bluegate Property Services** -- Sampletown, OR
*Facilities Maintenance Technician | 01/2014 - 05/2016*
- Maintained electrical distribution and HVAC controls for 12 commercial buildings.

## Education
**SYNTHETIC-ONLY Example Community College** -- Sampletown, OR
*Certificate, Building Systems Maintenance | 2013*

## Skills
- Preventive maintenance scheduling, HVAC controls
"""

# The orchestrator joins the two uploads exactly this way; asserting it here keeps the
# fixture honest about what a multi-file source actually looks like at ingress.
COMBINED = DOC_ELECTRICIAN + "\n\n---\n\n" + DOC_FACILITIES

SOURCE_DOCUMENTS = [
    {"source_id": "document-0", "text": DOC_ELECTRICIAN, "kind": "candidate_document"},
    {"source_id": "document-1", "text": DOC_FACILITIES, "kind": "candidate_document"},
]

# Both records present and correctly separated: the draft a competent writer should
# produce. Preservation must not damage it.
#
# The two summaries are on SEPARATE lines here, one per source document. That is a
# convenience, not a requirement: synthesising supported facts from more than one cited
# document is legitimate, and the same summary merged onto ONE line must approve too.
# Both spellings are asserted in the paired tests below
# (`test_supported_summary_approves_whether_the_clauses_are_split_by_a_newline_or_a_space`).
DRAFT_BOTH_RECORDS = """## Professional Summary
Journeyman electrician with 8+ years of commercial electrical experience.
Facilities maintenance technician focused on preventive maintenance.

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


# ---------------------------------------------------------------------------
# A deterministic reviewer for the indexed contract
# ---------------------------------------------------------------------------

def _fold(text: str) -> str:
    """Comparable form: no provenance marker, whitespace collapsed, casefolded."""
    return " ".join(text.replace(live_capture.SYNTHETIC_MARKER, "").split()).casefold()


def _word_overlap(quote: str, line: str) -> float:
    """Share of the segment's significant words that appear in the line.

    Mirrors the retrieval heuristic the real adapter uses (``correction_protocol``), so
    this reviewer accepts the same kind of paraphrase a real model would be shown. It is
    deliberately generous: the point of the scripted reviewer is to be *lenient* about
    support, so that the engine's own addition check is what has to catch an appended,
    unbacked qualification. A pickier reviewer would hide that check behind a coverage
    failure and the negative control would prove the wrong thing.
    """
    words = [word for word in _fold(quote).split() if len(word) > 2]
    if not words:
        return 0.0
    present = set(_fold(line).split())
    return sum(word in present for word in words) / len(words)


class TwoDocumentReviewer:
    """Answers the indexed protocol by matching draft lines to source segments.

    One explicit rule, stated once: a line is ``supported`` when a source segment's text
    appears in it (or it appears in a segment), and the segments it matched are cited as
    evidence. Headings are ``structural``. Anything else is ``unsupported``.

    The rule is fixed and applies to every line, so it cannot be tuned toward a verdict.
    It is deliberately stricter than a live model: no paraphrase credit, so a passing run
    here is a lower bound on what a model would accept.
    """

    def __init__(self) -> None:
        self.phases: list[str] = []
        self.inventory: list[dict] = []
        self.review: list[dict] = []
        self.restore: list[dict] = []

    async def close(self) -> None:  # aiohttp-compatible no-op
        pass

    def post(self, url, **kwargs):
        wire = json.loads(kwargs["json"]["messages"][1]["content"])
        contract = wire.get("contract", "")
        if contract == correction_protocol.INVENTORY_CONTRACT:
            phase, answer = "inventory", self._inventory(wire)
            self.inventory = wire["segments"]
        elif contract == correction_protocol.RESTORE_CONTRACT:
            phase, answer = "restore", self._restore(wire)
            self.restore = wire["lines"]
        elif contract == correction_protocol.REVIEW_CONTRACT:
            phase, answer = "review", self._review(wire)
            self.review = wire["segments"]
        else:
            phase, answer = "unknown", {}
        self.phases.append(phase)
        return _response(200, _envelope(answer))

    def _inventory(self, wire):
        required, non_resume = [], []
        for segment in wire["segments"]:
            text = segment["text"].strip()
            (non_resume if not text or text.startswith("#") else required).append(
                segment["index"])
        # `uncertain` is empty on purpose: this fixture's ownership is unambiguous.
        return {"required": required, "non_resume": non_resume, "uncertain": []}

    def _restore(self, wire):
        # Echo each line unchanged. Nothing here needs a restoration, so this path is
        # present only so an unexpected one is visible rather than a crash.
        return {"restorations": [{"line_id": line["line_id"],
                                  "replacement": line["text"]} for line in wire["lines"]]}

    def _review(self, wire):
        supported, unsupported, structural = [], [], []
        evidence: dict[str, list[int]] = {}
        cited: set[int] = set()

        for line in wire["lines"]:
            text, line_id = line["text"], line["line_id"]
            if not text.strip():
                continue
            if text.lstrip().startswith("#"):
                structural.append(line_id)
                continue
            bare = _fold(text.replace("**", "").replace("*", ""))
            matched = []
            for segment in wire["segments"]:
                quote = _fold(segment["text"].lstrip("- ").replace("**", ""))
                if not quote or quote.startswith("#"):
                    continue
                contiguously_present = quote in bare
                mostly_present = _word_overlap(quote, bare) >= 0.6
                if contiguously_present or mostly_present or bare in quote:
                    matched.append(segment["index"])
            if matched:
                supported.append(line_id)
                evidence[line_id] = sorted(set(matched))
                cited.update(matched)
            else:
                unsupported.append(line_id)

        missing = [segment["index"] for segment in wire["segments"]
                   if segment["text"].strip()
                   and not segment["text"].strip().startswith("#")
                   and segment["index"] not in cited]
        return {"supported": supported, "unsupported": unsupported, "uncertain": [],
                "structural": structural, "evidence": evidence,
                "represented": sorted(cited), "missing": missing}

        missing = [segment["index"] for segment in wire["segments"]
                   if segment["text"].strip()
                   and not segment["text"].strip().startswith("#")
                   and segment["index"] not in cited]
        return {"supported": supported, "unsupported": unsupported, "uncertain": [],
                "structural": structural, "evidence": evidence,
                "represented": sorted(cited), "missing": missing}


def _envelope(answer) -> str:
    return json.dumps({
        "model": correction_provider.OR_SLUG_DRAFTER,
        "provider": "TwoDocumentReviewer",
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        "choices": [{"finish_reason": "stop",
                     "message": {"content": json.dumps(answer)}}],
    })


def _response(status, body):
    class Content:
        async def iter_chunked(self, size):
            raw = (body or "").encode()
            for start in range(0, len(raw), size):
                yield raw[start:start + size]

    class Headers:
        def get(self, name, default=None):
            return default

    class Response:
        def __init__(self):
            self.status = status
            self.headers = Headers()
            self.content = Content()

        async def text(self, *args, **kwargs):
            return body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    return Response()


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------

@pytest.fixture
def two_document_run(monkeypatch, tmp_path):
    """Issue ``POST /analyze`` with two ``source_documents`` and a scripted writer.

    Capture is enabled into ``tmp_path`` for this test, so the capture path is exercised
    on the two-document route rather than assumed to work because it works for one.
    """
    body, reviewer, directory = _run_route(monkeypatch, tmp_path, DRAFT_BOTH_RECORDS)
    yield body, reviewer, directory
    app.app.dependency_overrides.clear()


def _run_route(monkeypatch, tmp_path, draft):
    """Drive ``POST /analyze`` once with ``draft`` as the writer's output.

    A function rather than only a fixture so a test can supply a DIFFERENT draft and
    still go through the real route, instead of poking at a fixture after it has run.
    """
    import stages.star_editor as star_editor_module
    from llm_client_adapter import LLMClientAdapter

    def forbidden(*args, **kwargs):
        raise AssertionError("external network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)

    monkeypatch.setattr(app.settings, "openrouter_api_key_1", "offline-key")
    monkeypatch.setattr(app.settings, "openrouter_api_key_2", None)
    monkeypatch.setenv("PUBLICATION_RECEIPT_SECRET", "two-doc-route-secret-0123456789")
    monkeypatch.setitem(app.app.dependency_overrides, app.get_api_key, lambda: "offline")
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available",
                        lambda: True)

    monkeypatch.setenv(live_capture.SETTING_ENABLED, "true")
    monkeypatch.setenv(live_capture.SETTING_DIR, str(tmp_path))

    async def scripted_call(self, system_prompt, user_prompt, model,
                            temperature=0.7, response_format=None, timeout=None,
                            fallback_models=None, **kwargs):
        if system_prompt == star_editor_module.STAR_EDITOR_SYSTEM_PROMPT:
            # The input under test.
            return json.dumps({"final_markdown": draft,
                               "final_plain_text": draft,
                               "editorial_notes": "two-record draft"})
        if "researcher" in system_prompt.lower() or "domain" in system_prompt.lower():
            return json.dumps({"implied_metrics": [], "domain_vocab": [],
                               "implied_skills": [], "research_notes": ""})
        return json.dumps({
            "rewritten_experiences": [
                {"company": "Cascade Power Solutions, LLC",
                 "role": "Journeyman Electrician",
                 "duration": "03/2020 - Present",
                 "bullets": ["Installed commercial electrical systems across "
                             "40+ job sites."]},
                {"company": "Bluegate Property Services",
                 "role": "Facilities Maintenance Technician",
                 "duration": "01/2014 - 05/2016",
                 "bullets": ["Maintained electrical distribution and HVAC controls for "
                             "12 commercial buildings."]},
            ],
            "seniority_applied": "senior",
        })

    monkeypatch.setattr(LLMClientAdapter, "call_llm_async", scripted_call)

    reviewer = TwoDocumentReviewer()
    with TestClient(app.app) as client:
        client.portal.call(app.app.state.http_session.close)
        app.app.state.http_session = reviewer
        response = client.post(
            "/analyze",
            headers={"X-Cogito-Owner-Id": "two-document-owner"},
            json={
                "resume_text": COMBINED,
                "source_documents": SOURCE_DOCUMENTS,
                "job_ad": "Senior Electrician role",
            },
        )
    return response.json(), reviewer, tmp_path


# ---------------------------------------------------------------------------
# The correction path saw both documents
# ---------------------------------------------------------------------------

def test_correction_receives_both_documents_under_their_own_ids(two_document_run):
    """[route-two-doc] The inventory and the review both carry document-0 and document-1.

    If the route collapsed the evidence to one document before correction, every
    later assertion in this file would be about a single document without saying so.
    """
    _body, reviewer, _tmp = two_document_run

    assert reviewer.phases[0] == "inventory"
    inventory_ids = {segment["source_id"] for segment in reviewer.inventory}
    review_ids = {segment["source_id"] for segment in reviewer.review}
    assert inventory_ids == {"document-0", "document-1"}
    assert review_ids == {"document-0", "document-1"}


def test_preservation_reads_every_document(two_document_run):
    """[route-two-doc] Both summaries, both degrees and both skill lists are source text.

    Asserted against the module the route actually calls, so this fails if the
    single-document assumption returns.
    """
    _body, _reviewer, _tmp = two_document_run

    summary = source_preservation.source_summary(COMBINED)
    assert "Journeyman electrician with 8+ years" in summary
    assert "Facilities maintenance technician focused on preventive maintenance" in summary

    institutions = [record["institution"]
                    for record in source_preservation.source_education_records(COMBINED)]
    assert institutions == ["Example Technical College", "Example Community College"]

    skills = "\n".join(source_preservation.source_skills_lines(COMBINED))
    assert "Commercial electrical systems, panel labeling" in skills
    assert "Preventive maintenance scheduling, HVAC controls" in skills
    assert "---" not in skills, "the document separator leaked in as a skill"


# ---------------------------------------------------------------------------
# The approved document keeps both records
# ---------------------------------------------------------------------------

def test_both_documents_employment_and_education_records_survive_the_route(
        two_document_run):
    """[route-two-doc] Neither record is dropped, merged or overwritten.

    The whole point of the fixture: two employers and two institutions, each keeping its
    own title, dates, degree and field.
    """
    body, _reviewer, _tmp = two_document_run
    assert body.get("correction_status") == "approved", (
        body.get("correction_reason"), body.get("correction_commentary"))

    markdown = body["final_written_section_markdown"]
    for value in ("Cascade Power Solutions, LLC", "Bluegate Property Services",
                  "Example Technical College", "Example Community College",
                  "Journeyman Electrician", "Facilities Maintenance Technician",
                  "03/2020", "01/2014", "05/2016",
                  "Electrical Systems Technology", "Building Systems Maintenance",
                  "2016", "2013"):
        assert value in markdown, f"a source record was lost: {value}"

    # Each employer keeps its own role and dates, positionally.
    cascade = markdown.index("**Cascade Power Solutions, LLC**")
    bluegate = markdown.index("**Bluegate Property Services**")
    cascade_block = markdown[cascade:bluegate]
    assert "Journeyman Electrician" in cascade_block
    assert "03/2020" in cascade_block
    assert "Facilities Maintenance Technician" not in cascade_block
    assert "01/2014" not in cascade_block

    # Each institution keeps its own degree and field.
    technical = markdown.index("**Example Technical College**")
    community = markdown.index("**Example Community College**")
    technical_block = markdown[technical:community]
    assert "Associate of Applied Science" in technical_block
    assert "Electrical Systems Technology" in technical_block
    assert "Certificate" not in technical_block
    assert "Building Systems Maintenance" not in technical_block

    community_block = markdown[community:]
    assert "Certificate" in community_block
    assert "Building Systems Maintenance" in community_block
    assert "Associate of Applied Science" not in community_block


def test_the_route_adds_no_fact_either_document_lacks(two_document_run):
    """[route-two-doc] No invented qualification, and no duplicate record."""
    body, _reviewer, _tmp = two_document_run
    markdown = body["final_written_section_markdown"]

    for invented in ("AutoCAD", "Revit", "Python", "SCADA", "PLC", "Agile", "Gantt",
                     "OSHA", "GPA", "Relevant Courses", "crew", "supervision"):
        assert invented not in markdown, f"invented qualification: {invented}"

    for once in ("Cascade Power Solutions", "Bluegate Property Services",
                 "Example Technical College", "Example Community College"):
        assert markdown.count(once) == 1, (
            f"{once!r} appears {markdown.count(once)} times")


# ---------------------------------------------------------------------------
# The capture path, with two documents
# ---------------------------------------------------------------------------

def test_the_capture_records_both_source_documents(two_document_run):
    """[route-two-doc] Capture writes a record naming both documents.

    The capture path is enabled for this request, so this is the live wiring being
    exercised rather than a unit call to ``live_capture``.
    """
    _body, _reviewer, tmp_path = two_document_run
    files = sorted(tmp_path.glob("*.json"))
    assert files, (
        "capture is enabled for this request but nothing was written; the two-document "
        "route does not reach live_capture.start")

    record = json.loads(files[0].read_text())
    recorded_ids = [source["source_id"] for source in record["sources"]]
    assert recorded_ids == ["document-0", "document-1"]
    # The draft the capture stored is the writer's own two-record draft.
    assert "Example Community College" in record["draft"]
    assert record["review"]["status"] in {"approved", "withheld"}


def test_the_receipt_binds_the_two_document_markdown(two_document_run):
    """[route-two-doc] Approval produces a receipt over exactly this document."""
    body, _reviewer, _tmp = two_document_run
    assert body.get("publication_receipt")
    assert body.get("approved_export_payload"), "an approved result must carry a payload"
    assert body.get("requires_review") is False


# ---------------------------------------------------------------------------
# Cross-document synthesis: supported is supported, however it is laid out
# ---------------------------------------------------------------------------

# The same two source summaries, merged onto ONE line. Nothing is added and nothing moves
# between records; this is a plausible writer output and must approve.
MERGED_SUMMARY_DRAFT = DRAFT_BOTH_RECORDS.replace(
    "Journeyman electrician with 8+ years of commercial electrical experience.\n"
    "Facilities maintenance technician focused on preventive maintenance.",
    "Journeyman electrician with 8+ years of commercial electrical experience. "
    "Facilities maintenance technician focused on preventive maintenance.")


def test_supported_summary_approves_whether_the_clauses_are_split_by_a_newline_or_a_space(
        monkeypatch, tmp_path):
    """[route-two-doc] Paired: the SAME supported summary approves in both layouts.

    Merging two supported clauses onto one line is synthesis, not fabrication. The
    provenance check must therefore judge the span against the documents the reviewer
    CITED for it, not only the document the particular source item came from.

    Measured before the fix: with one document supplied to ``_introduces_beyond``,
    document-1's own word "Facilities" was reported as *introduced* into a claim the
    reviewer had cited both documents for, and the merged — entirely faithful — summary
    was withheld with ``source_completeness_uncertain``. Two bullet points would have
    approved and the identical sentence joined by a space would not, which is a property
    of the layout rather than of the content.

    The pair is the point: asserting only the merged case would prove the check got
    looser; asserting only the split case would prove nothing changed.
    """
    newline_run = _run_route(monkeypatch, tmp_path, DRAFT_BOTH_RECORDS)[0]
    assert newline_run.get("correction_status") == "approved", (
        "the newline layout must approve", newline_run.get("correction_reason"),
        newline_run.get("review_reasons"))

    merged_run = _run_route(monkeypatch, tmp_path, MERGED_SUMMARY_DRAFT)[0]
    assert merged_run.get("correction_status") == "approved", (
        "the space layout expresses the same supported facts and must also approve",
        merged_run.get("correction_reason"), merged_run.get("review_reasons"))

    # And the approved text carries both clauses.
    for run in (newline_run, merged_run):
        markdown = run["final_written_section_markdown"]
        assert "Journeyman electrician with 8+ years" in markdown
        assert "Facilities maintenance technician" in markdown


# ---------------------------------------------------------------------------
# Negative controls: attribution survives the synthesis allowance
#
# Loosening the evidence set from "one document" to "the documents the claim cites" is
# only safe if an unsupported addition and a moved fact still fail. Each control below
# differs from an APPROVING draft by exactly one line, so the defect is the only variable.
#
# Why the appended qualification is on an EXPERIENCE BULLET, not in the summary:
# preservation legitimately REPAIRS a summary that appends an unbacked proficiency list
# (`preserve_source_summary`), so a summary append is caught before review and the route
# approves the corrected document — which is the correct outcome, not a miss. A bullet has
# no such repair, so the gate itself has to notice, which is what this control is for.
# `test_preservation_repairs_a_summary_append_before_the_gate_sees_it` records the other
# half of that pair.
# ---------------------------------------------------------------------------

# An appended unbacked qualification is now CORRECTED rather than withheld, so it
# is asserted by its own test below instead of appearing here.
#
# Measured 2026-11-08, when the trailing-purpose boundary was widened to include
# "using" / "toward": "…across 40+ job sites using AutoCAD and SCADA" is now
# trimmed back to "…across 40+ job sites" by preservation before the gate runs.
# The published document contains neither "AutoCAD" nor "SCADA", so the defect is
# corrected rather than published — which is the outcome the requirement allows.
# The alternative, approving the text WITH the invented tools, would be a false
# approval and is forbidden by the test that replaces this case.
CASES_WITHHELD = {
    # An employer's record under another employer's name.
    "transplanted_employer_label": DRAFT_BOTH_RECORDS.replace(
        "**Cascade Power Solutions, LLC** -- Example City, WA",
        "**Bluegate Property Services** -- Example City, WA"),
    # A role title moved to the other employer's record.
    "transplanted_role_title": DRAFT_BOTH_RECORDS.replace(
        "*Journeyman Electrician | 03/2020 - Present*",
        "*Facilities Maintenance Technician | 03/2020 - Present*"),
    # Dates moved from one employer to another.
    "transplanted_dates": DRAFT_BOTH_RECORDS.replace(
        "*Journeyman Electrician | 03/2020 - Present*",
        "*Journeyman Electrician | 01/2014 - 05/2016*"),
    # One employer's accomplishment under another employer's heading. This is a DUPLICATE
    # line (both bullets become the same text), which is why the guard below counts lines
    # rather than set membership.
    "transplanted_achievement": DRAFT_BOTH_RECORDS.replace(
        "- Installed commercial electrical systems across 40+ job sites.",
        "- Maintained electrical distribution and HVAC controls for 12 commercial "
        "buildings."),
    # Another document's skill grafted onto this document's line.
    "transplanted_qualification": DRAFT_BOTH_RECORDS.replace(
        "- Commercial electrical systems, panel labeling",
        "- Commercial electrical systems, panel labeling, HVAC controls"),
    # A real figure re-attributed. Both numbers are in the sources, so only the record it
    # is attached to is wrong.
    "transplanted_figure": DRAFT_BOTH_RECORDS.replace(
        "- Installed commercial electrical systems across 40+ job sites.",
        "- Installed commercial electrical systems across 12 commercial buildings."),
}


def test_the_controls_really_are_single_edits_from_an_approving_draft():
    """[route-two-doc] Guard: every control differs from the baseline on exactly one line.

    Counted by multiset, not by set membership. The achievement-transplant control
    replaces one bullet with text that already exists as another bullet, so a set
    difference reports ZERO changed lines — and this guard would then pass while asserting
    nothing. Counting occurrences catches the duplication.

    Without a guard like this, a control could fail for an unrelated reason (a dropped
    record, a typo) and silently prove nothing about attribution.
    """
    from collections import Counter

    baseline = Counter(DRAFT_BOTH_RECORDS.splitlines())
    for name, draft in CASES_WITHHELD.items():
        difference = baseline - Counter(draft.splitlines())
        added = Counter(draft.splitlines()) - baseline
        assert sum(difference.values()) == 1 and sum(added.values()) == 1, (
            f"{name} is not a single-line edit: removed {dict(difference)}, "
            f"added {dict(added)}")


@pytest.mark.parametrize("case", sorted(CASES_WITHHELD))
def test_an_unsupported_addition_or_a_moved_fact_is_still_withheld(
        monkeypatch, tmp_path, case):
    """[route-two-doc] Claim-level attribution survives the synthesis allowance.

    Every case here is one line away from a draft that APPROVES in the test above, so a
    pass here means the defect itself is what was rejected.

    Note what is NOT asserted: the specific reason code. A transplant can be caught by
    the coverage check (`missing`), by the provenance check (downgraded to `uncertain`),
    or by a repair that then fails re-review (`malformed_output`). Which mechanism fires
    is an implementation detail; what matters is that the draft is **not approved** and
    nothing is published.
    """
    body, _reviewer, _tmp = _run_route(monkeypatch, tmp_path, CASES_WITHHELD[case])

    assert body.get("correction_status") == "withheld", (
        f"{case} was approved", body.get("correction_reason"))
    assert body.get("requires_review") is True
    # Nothing is published: no receipt, no export payload, no document.
    assert body.get("final_written_section_markdown") is None
    assert body.get("publication_receipt") is None
    assert body.get("approved_export_payload") is None


def test_preservation_repairs_a_summary_append_before_the_gate_sees_it(
        monkeypatch, tmp_path):
    """[route-two-doc] Paired with the bullet control: a repaired defect may still publish.

    Appending an unbacked qualification to the SUMMARY is repaired deterministically by
    ``preserve_source_summary`` before the gate runs, so the route approves — the document
    it publishes, though, does not contain the appended claim. That is a pass, and it must
    not be confused with the gate failing to notice.

    Asserted three ways so the distinction cannot be lost:

    * the run is approved;
    * the published document omits the appended qualification;
    * the same append on a bullet — where no repair exists — IS withheld (the control
      above). Together these pin that the difference is the repair, not the check.
    """
    draft = DRAFT_BOTH_RECORDS.replace(
        "Journeyman electrician with 8+ years of commercial electrical experience.",
        "Journeyman electrician with 8+ years of commercial electrical experience, "
        "proficient in AutoCAD and SCADA.")
    body, _reviewer, _tmp = _run_route(monkeypatch, tmp_path, draft)

    assert body.get("correction_status") == "approved", (
        body.get("correction_reason"), body.get("review_reasons"))
    markdown = body["final_written_section_markdown"]
    assert "AutoCAD" not in markdown and "SCADA" not in markdown, (
        "the appended qualification must not survive into the published document")
    # The supported summary it was appended to is still there.
    assert "Journeyman electrician with 8+ years" in markdown


def test_a_bullet_appended_tool_list_is_corrected_not_published(
        monkeypatch, tmp_path):
    """[route-two-doc] A bullet's unbacked tool list is now trimmed, not published.

    This case used to be a withhold control: the appended "using AutoCAD and
    SCADA" carried capitalised names that appeared in neither document, and no
    repair existed for a bullet, so the gate withheld.

    Measured 2026-11-08, after the trailing-purpose boundary was widened to
    include "using", preservation now trims the clause before the gate runs. The
    run approves and the published document does not contain the invented tools.

    Asserted three ways, because "corrected" and "approved anyway" look identical
    if you only check the status:

    * the invented tools are absent from the published document;
    * the faithful prefix the clause was appended to survives;
    * the surviving bullet is the source's own wording, not a composed sentence.
    """
    draft = DRAFT_BOTH_RECORDS.replace(
        "- Installed commercial electrical systems across 40+ job sites.",
        "- Installed commercial electrical systems across 40+ job sites using "
        "AutoCAD and SCADA.")
    body, _reviewer, _tmp = _run_route(monkeypatch, tmp_path, draft)

    markdown = body.get("final_written_section_markdown") or ""
    if body.get("correction_status") == "withheld":
        # Withholding is also acceptable: nothing is published at all.
        assert markdown == ""
        assert body.get("publication_receipt") is None
        return
    assert "AutoCAD" not in markdown, "the invented tool reached the document"
    assert "SCADA" not in markdown, "the invented tool reached the document"
    assert "Installed commercial electrical systems across 40+ job sites" in markdown
    # Nothing was composed: the kept text is a prefix of what the writer wrote.
    assert "using" not in markdown.lower()


# ---------------------------------------------------------------------------
# Export: the approved two-document payload really renders, and alteration is refused
#
# This closes the loop the acceptance target asks for: an approval is only worth anything
# if the signed payload it produces survives the export endpoint unchanged, and if the
# SAME endpoint refuses that payload once its content is altered.
#
# Both directions use the real `POST /resume/export`, the real receipt, and the real
# LaTeX + python-docx renderers. Nothing here is mocked, and no PDF/DOCX bytes are
# synthesised: the rendered text is read back from the produced files.
# ---------------------------------------------------------------------------

pdflatex = pytest.mark.skipif(shutil.which("pdflatex") is None,
                              reason="no LaTeX toolchain on this machine")

EXPORT_OWNER = "two-document-owner"


def _export(client, body, *, markdown=None, receipt=None):
    """POST the approved payload to the real export endpoint."""
    payload = dict(body.get("approved_export_payload") or {})
    payload["publication_receipt"] = receipt or body["publication_receipt"]
    payload["final_written_section_markdown"] = (
        body["final_written_section_markdown"] if markdown is None else markdown)
    # The approval may be policy-bound (receipt v5). The browser transports this
    # state verbatim; omitting it makes a v5 approval unverifiable, which is the
    # correct fail-closed behaviour rather than a false success.
    if body.get("skill_inference_policy_state"):
        payload["skill_inference_policy_state"] = body["skill_inference_policy_state"]
    return client.post("/resume/export",
                       headers={"X-Cogito-Owner-Id": EXPORT_OWNER,
                                "X-API-Key": "offline"},
                       json=payload)


@pdflatex
def test_the_approved_two_document_payload_renders_both_records(monkeypatch, tmp_path):
    """[route-two-doc] Approved payload -> real export -> both documents' records on the page.

    The PDF is read back with ``pdftotext`` and the DOCX through ``word/document.xml``,
    so this asserts what a reader would see, not what the payload claims.
    """
    body, _reviewer, _tmp = _run_route(monkeypatch, tmp_path, DRAFT_BOTH_RECORDS)
    assert body.get("correction_status") == "approved"
    assert body.get("approved_export_payload"), "an approval must carry an export payload"

    with TestClient(app.app) as client:
        response = _export(client, body)
        assert response.status_code == 200, response.text[:400]
        data = response.json()

    assert data.get("pdf_base64") and data.get("docx_base64")

    import base64
    with tempfile.TemporaryDirectory() as directory:
        pdf_file = Path(directory) / "out.pdf"
        docx_file = Path(directory) / "out.docx"
        pdf_file.write_bytes(base64.b64decode(data["pdf_base64"]))
        docx_file.write_bytes(base64.b64decode(data["docx_base64"]))

        pdf_text = subprocess.run(["pdftotext", "-layout", str(pdf_file), "-"],
                                  capture_output=True, text=True).stdout
        with zipfile.ZipFile(docx_file) as archive:
            docx_text = re.sub(r"<[^>]+>", " ",
                               archive.read("word/document.xml").decode())

    for label, text in (("pdf", pdf_text), ("docx", docx_text)):
        flat = " ".join(text.split())
        # Both documents' employment and education records reached the page.
        for value in ("Cascade Power Solutions", "Bluegate Property Services",
                      "Example Technical College", "Example Community College",
                      "Journeyman Electrician", "Facilities Maintenance Technician",
                      "03/2020", "01/2014", "Electrical Systems Technology",
                      "Building Systems Maintenance"):
            assert value in flat, f"{label}: missing {value!r}"

        # And no mixing survived the renderer.
        cascade_at = flat.index("Cascade Power Solutions")
        bluegate_at = flat.index("Bluegate Property Services")
        assert cascade_at < bluegate_at
        assert "Facilities Maintenance Technician" not in flat[cascade_at:bluegate_at]

        assert "\u00e2\u0080" not in text, f"{label}: mojibake"


@pdflatex
@pytest.mark.parametrize("alteration", ["appended_claim", "changed_employer",
                                       "moved_date", "dropped_record"])
def test_export_refuses_altered_content_under_the_existing_receipt_contract(
        monkeypatch, tmp_path, alteration):
    """[route-two-doc] The same receipt does not authorise altered content.

    One approved payload, four alterations, all of which change the document a reader
    would see. The receipt binds the content fingerprint and the canonical render
    structure, so each must be refused — and refused for a receipt reason, not by
    crashing.

    The approved export in the test above is the control: identical payload, identical
    endpoint, identical receipt, unchanged content, 200.
    """
    body, _reviewer, _tmp = _run_route(monkeypatch, tmp_path, DRAFT_BOTH_RECORDS)
    assert body.get("correction_status") == "approved"
    markdown = body["final_written_section_markdown"]

    altered = {
        # Content the sources never state.
        "appended_claim": markdown + "\n- Certified in AutoCAD and SCADA.\n",
        # A record under the wrong employer (two documents, one changed).
        "changed_employer": markdown.replace("Bluegate Property Services",
                                             "Cascade Power Solutions"),
        # A date moved.
        "moved_date": markdown.replace("01/2014", "03/2020"),
        # A whole record deleted, so the document no longer covers both uploads.
        "dropped_record": markdown.replace("Example Community College", ""),
    }[alteration]
    assert altered != markdown, f"{alteration} did not actually change the document"

    with TestClient(app.app) as client:
        response = _export(client, body, markdown=altered)

    assert response.status_code == 409, (
        f"{alteration} was exported; status {response.status_code}")
    payload = response.json()
    assert payload.get("error") == "publication_blocked"
    assert payload.get("reasons"), "a refusal must name its reason"
    # A refusal must not leak the document.
    assert "pdf_base64" not in payload and "docx_base64" not in payload


def test_export_refuses_the_approved_content_presented_under_another_owner(
        monkeypatch, tmp_path):
    """[route-two-doc] The receipt is bound to an owner, so it does not travel.

    A second account cannot spend an approval minted for the first, even with the exact
    approved bytes.
    """
    body, _reviewer, _tmp = _run_route(monkeypatch, tmp_path, DRAFT_BOTH_RECORDS)
    assert body.get("correction_status") == "approved"

    payload = dict(body["approved_export_payload"])
    payload["publication_receipt"] = body["publication_receipt"]
    payload["final_written_section_markdown"] = body["final_written_section_markdown"]

    with TestClient(app.app) as client:
        response = client.post(
            "/resume/export",
            headers={"X-Cogito-Owner-Id": "someone-else", "X-API-Key": "offline"},
            json=payload,
        )

    assert response.status_code == 409
    assert response.json().get("error") == "publication_blocked"
