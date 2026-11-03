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
import socket

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
# The two summaries are on SEPARATE lines, one per source document, because the route's
# provenance check (`source_correction._introduces_beyond`) compares a draft span against
# the ONE source document each inventory item came from. A single summary line merging
# both documents is therefore not substantiable from either document alone, and the route
# withholds it with `source_completeness_uncertain`. That is the correct behaviour for a
# line that mixes two records, and
# `test_a_merged_two_document_summary_is_withheld_not_silently_truncated` asserts it.
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
                if quote in bare or bare in quote:
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
# The limitation this measurement found, stated rather than hidden
# ---------------------------------------------------------------------------

MERGED_SUMMARY_DRAFT = DRAFT_BOTH_RECORDS.replace(
    "Journeyman electrician with 8+ years of commercial electrical experience.\n"
    "Facilities maintenance technician focused on preventive maintenance.",
    "Journeyman electrician with 8+ years of commercial electrical experience. "
    "Facilities maintenance technician focused on preventive maintenance.")


def test_a_merged_two_document_summary_is_withheld_not_silently_truncated(
        monkeypatch, tmp_path):
    """[route-two-doc] One summary line mixing both documents is withheld, not cut.

    The provenance check compares a draft span against the ONE source document each
    inventory item came from. A line that merges document-0's summary with document-1's
    is therefore not substantiable from either alone, and the route withholds with
    ``source_completeness_uncertain``.

    This is the correct outcome and the important one to pin: the alternative observed
    before the preservation fix was that the summary rule called document-1's sentence an
    "appended proficiency list" and TRUNCATED it — a silent deletion of a real source
    fact. Withholding is visible; truncation is not.

    Recorded as a limitation, not a passing feature: a writer that merges two summaries
    into one line will have that draft withheld rather than approved. Fixing that needs a
    line that cites two documents, which the current per-document provenance check cannot
    express.
    """
    body, _reviewer, _tmp = _run_route(monkeypatch, tmp_path, MERGED_SUMMARY_DRAFT)

    assert body.get("correction_status") == "withheld"
    assert "source_completeness_uncertain" in (body.get("review_reasons") or []), (
        body.get("review_reasons"))
    # And crucially: neither summary was deleted, because nothing was published at all.
    assert body.get("final_written_section_markdown") is None
