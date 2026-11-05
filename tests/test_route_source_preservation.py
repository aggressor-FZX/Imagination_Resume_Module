"""Source preservation proven through the real route and the real renderers.

    POST /analyze
      -> writer draft (run-7 output, defects intact, verbatim from the capture)
      -> real StarEditor.polish -> deterministic source preservation
      -> real CorrectionProvider -> real engine -> publication gate -> receipt
    POST /resume/export -> real LaTeX compiler + real python-docx
                        -> pdftotext / DOCX XML read back

What is scripted, and labelled
------------------------------

* the **writer** output — the input under test, verbatim from the capture;
* the **reviewer** decisions — authored by hand from the source text, per line, and
  given as an explicit table. They are NOT produced by a coverage threshold. An
  earlier version scored similarity and lowered the threshold until the run approved,
  which is indistinguishable from scripting the result: the reviewer stops being an
  independent check the moment its verdict is tuned to the outcome. The table below
  states, for each line of this one fixture, which source segments establish it, and
  the test asserts the decisions were actually used.

What is NOT done: ``CorrectionResult`` is never overridden, the answer key is never
edited, and no PDF/DOCX bytes are mocked.

Harness note — why the instrumentation lives in a fixture
---------------------------------------------------------

The request must be issued AFTER the spies are installed. This file previously built
the response inside the ``preserved`` fixture and let tests install spies afterwards,
so a spy could never observe anything: the route had already completed. That produced a
long detour of "the validator was never called / maybe another module / perhaps cached"
about code that was simply never observed. Two things prevent a repeat:

* ``boundaries`` is a fixture that installs every recorder and then yields a *factory*;
  the factory issues the request. Nothing can run before the recorders exist, because
  the request is a call the test makes.
* ``captured["boundaries"]`` holds the real values at each named seam, so a test reads
  what happened instead of re-deriving it. No offsets or source ids are reconstructed by
  hand.

Labelled **[route-preserve]**, **[harness]** and **[renderer-real]**.
"""
from __future__ import annotations

import base64
import copy
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
import live_capture
import source_correction
import source_preservation
from tests.fixtures.fidelity import expected_facts as facts

RUN7 = Path("/home/skystarved/Render_Dockers/.live-run-2026-11-02/captures/"
            "run7/0a267e04-affd-4cee-b944-26c574703039.json")

requires_run7 = pytest.mark.skipif(not RUN7.exists(), reason="run-7 capture absent")

SOURCE = facts.SOURCE_TEXT


def _recorded_draft() -> str:
    """The run-7 writer output, verbatim — all four defects intact."""
    return live_capture.load(RUN7)["draft"]


# ---------------------------------------------------------------------------
# The controlled reviewer: an explicit, hand-authored decision table
# ---------------------------------------------------------------------------

# For each line of the PRESERVED run-7 document, which source segments establish it.
#
# Keyed by a substring unique to the line. Written by reading the source and the
# preserved document side by side, not by scoring overlap. A line absent from this table
# is reported `unsupported` — the reviewer declines to substantiate it.
LINE_DECISIONS: dict[str, tuple[int, ...]] = {
    # Summary restates the source summary, which is wrapped across segments 1 and 2.
    "Journeyman electrician with 8+ years": (1, 2),
    # Cascade record: employer line, role/date line, then its three bullets.
    "**Cascade Power Solutions, LLC**": (4,),
    "*Journeyman Electrician | 03/2020": (5,),
    "Led installation and maintenance": (6,),
    "Trained 6 apprentices": (7,),
    "Optimized panel labeling": (8,),
    # Harborline record.
    "**Harborline Mechanical Contractors**": (9,),
    "*Electrical Apprentice | 06/2016": (10,),
    "Supported conduit bending": (11,),
    "Completed 8,000 hours of supervised field work": (12,),
    # Education: institution on its own line, then degree + field + date.
    "**Example Technical College**": (14,),
    "*Associate of Applied Science, Electrical Systems Technology | 2016*": (15,),
    "- Washington Journey Level Electrician License": (17,),
    "- NFPA 70E Arc Flash Safety": (18,),
    # The skills block. Preservation now replaces an unsupported skills block with the
    # source's own two bullets, so each line rests on exactly one segment — and the
    # invented categorised block from the writer (`**Electrical:** installation,
    # maintenance, ..., apprentice supervision, ...`) is gone before the reviewer sees it.
    "- Commercial electrical systems, conduit bending, panel labeling": (20,),
    "- NFPA 70E, lockout/tagout, inspection checklists": (21,),
}


def decisions_for(text: str) -> tuple[int, ...] | None:
    """The authored segments for a draft line, or None when it is not substantiated."""
    for marker, segments in LINE_DECISIONS.items():
        if marker in text:
            return segments
    return None


class ControlledReviewer:
    """Answers the correction phases from the authored table above.

    Test-only. It is deliberately not "aggressive"; it states what the source says.
    """

    def __init__(self):
        self.phases: list[str] = []
        self.requests: list[dict] = []
        self.matched_decisions: list[str] = []

    async def close(self):
        pass

    def post(self, url, **kwargs):
        wire = json.loads(kwargs["json"]["messages"][1]["content"])
        phase = self._phase(wire)
        self.phases.append(phase)
        self.requests.append(kwargs)
        return _response(200, _envelope(self._answer(phase, wire)))

    @staticmethod
    def _phase(wire):
        contract = wire.get("contract", "")
        if contract.startswith("Classify every numbered source segment"):
            return "inventory"
        if contract.startswith("Rewrite the named draft lines"):
            return "restore"
        if "flagged_claim_ids" in wire:
            return "repair"
        return "review"

    def _answer(self, phase, wire):
        if phase == "inventory":
            # Headings are `non_resume`: they assert no candidate fact, so demanding a
            # representation for one demands the impossible. `required` for a heading
            # produced a representation with no supporting span, which the engine
            # correctly downgraded to `uncertain`; `uncertain` for a heading blocks the
            # run outright. `non_resume` is the accurate classification for a line that
            # makes no claim about the candidate.
            headings, body = [], []
            for segment in wire["segments"]:
                stripped = segment["text"].strip()
                (headings if stripped.startswith("#") or not stripped
                 else body).append(segment["index"])
            return {"required": body, "non_resume": headings, "uncertain": []}
        if phase == "repair":
            # Echo each flagged line unchanged. A no-op repair is reported as such rather
            # than silently accepted; nothing in these fixtures needs a repair, so this
            # path is only here so an unexpected repair is visible rather than a crash.
            by_line = {line["line_id"]: line["text"] for line in wire["lines"]}
            return {"edits": [{"claim_id": cid,
                               "replacement": by_line[cid.split(":", 1)[0]]}
                              for cid in wire["flagged_claim_ids"]]}
        return self._review(wire)

    def _review(self, wire):
        segments = wire["segments"]
        verdicts = {"supported": [], "unsupported": [], "uncertain": [], "structural": []}
        evidence: dict[str, list[int]] = {}

        for line in wire["lines"]:
            text, lid = line["text"], line["line_id"]
            if not text.strip():
                continue
            if text.lstrip().startswith("#"):
                verdicts["structural"].append(lid)
                continue
            authored = decisions_for(text)
            if authored is None:
                # Not in the table: the reviewer does not substantiate it. Deliberately
                # listed as `unsupported` rather than `uncertain` so an unanticipated
                # line is loudly visible instead of quietly withholding.
                verdicts["unsupported"].append(lid)
                continue
            verdicts["supported"].append(lid)
            evidence[lid] = sorted(set(authored))
            for marker in LINE_DECISIONS:
                if marker in text:
                    self.matched_decisions.append(marker)
                    break

        # Coverage: every segment named by an authored decision is represented. Segments
        # carrying a heading or only a provenance marker assert no candidate fact and are
        # classified `non_resume` in the inventory, so they are not part of this list.
        cited = {index for indices in evidence.values() for index in indices}
        represented = sorted(cited)
        missing = [s["index"] for s in segments
                   if s["index"] not in cited
                   and not s["text"].strip().startswith("#")
                   and not decisions_for(s["text"]) and _is_required(s["text"])]
        return {**verdicts, "evidence": evidence,
                "represented": represented, "missing": missing}


def _is_required(text: str) -> bool:
    """True for a source segment that states a candidate fact."""
    stripped = text.strip()
    return bool(stripped) and not stripped.startswith("#")


def _envelope(answer):
    return json.dumps({
        "model": correction_provider.OR_SLUG_DRAFTER,
        "provider": "ControlledReviewer",
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        "choices": [{"finish_reason": "stop",
                     "message": {"content": json.dumps(answer)}}],
    })


def _response(status, body):
    class Content:
        async def iter_chunked(self, size):
            raw = (body or "").encode()
            for index in range(0, len(raw), size):
                yield raw[index:index + size]

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
# [harness] every recorder installed BEFORE the request is issued
# ---------------------------------------------------------------------------

@pytest.fixture
def boundaries(monkeypatch):
    """Install all recorders, then hand back a factory that issues the request.

    Yielding a *factory* is the point. The previous version of this file ran the route
    inside the fixture and returned the finished response, so any spy a test installed
    afterwards observed nothing — and a long detour followed about code that was merely
    never watched. Here nothing can execute before the recorders exist, because the
    request is a call the test makes.

    Recorded seams, in the order the route crosses them:

    1. ``editor_before_preservation`` — the writer's own markdown, straight out of
       ``StarEditor.polish`` input handling;
    2. ``editor_after_preservation`` — the same document after preservation, plus the
       change list;
    3. ``correction_input`` — the exact markdown handed to ``correct()``;
    4. ``provider_response`` — each raw phase payload the engine asked for;
    5. ``adapter_review`` — the strict review shape the adapter normalised it into;
    6. ``validator`` — the engine validator's claims / incomplete set / missing items.
    """
    if not RUN7.exists():
        pytest.skip("run-7 capture is not present")

    captured: dict[str, object] = {"boundaries": []}

    def note(name, value):
        captured[name] = value
        captured["boundaries"].append(name)

    # ── offline guards ──────────────────────────────────────────────────────
    def forbidden(*args, **kwargs):
        raise AssertionError("external network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(app.settings, "openrouter_api_key_1", "offline-key")
    monkeypatch.setattr(app.settings, "openrouter_api_key_2", None)
    monkeypatch.setenv("PUBLICATION_RECEIPT_SECRET",
                       "offline-route-secret-1234567890")
    monkeypatch.setitem(app.app.dependency_overrides, app.get_api_key,
                        lambda: "offline")
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available",
                        lambda: True)

    # ── 1/2. StarEditor: before and after preservation ──────────────────────
    from stages.star_editor import StarEditor
    import source_preservation

    real_polish = StarEditor.polish

    async def recording_polish(self, draft_data=None, research_data=None, **kwargs):
        out = await real_polish(self, draft_data, research_data, **kwargs)
        note("editor_after_preservation", out.get("final_markdown"))
        note("preservation_changes", out.get("source_preservation"))
        return out

    monkeypatch.setattr(StarEditor, "polish", recording_polish)

    # The writer's raw markdown is captured where preservation reads it, i.e. as the
    # input to preserve_source_facts and the output of the model call. Recording the
    # model's own return is the honest "before" value.
    real_preserve = source_preservation.preserve_source_facts

    def recording_preserve(markdown, source_text):
        note("editor_before_preservation", markdown)
        out = real_preserve(markdown, source_text)
        note("preservation_result", out[0])
        return out

    monkeypatch.setattr(source_preservation, "preserve_source_facts",
                        recording_preserve)
    monkeypatch.setattr("stages.star_editor.source_preservation",
                        source_preservation)

    # ── 3. correction input ─────────────────────────────────────────────────
    real_correct = source_correction.correct

    async def recording_correct(markdown, sources, callback, **kw):
        note("correction_input", markdown)
        note("correction_sources", [(s.source_id, s.text) for s in sources])
        out = await real_correct(markdown, sources, callback, **kw)
        note("correction_result", (out.status, out.reason, out.review_calls,
                                   out.repair_calls))
        return out

    monkeypatch.setattr(source_correction, "correct", recording_correct)
    monkeypatch.setattr(app, "correct", recording_correct)

    # ── 4. provider response (the engine's own payload, pre-translation) ────
    real_invoke = source_correction._invoke

    async def recording_invoke(callback, phase, payload, timeout, events=None):
        note(f"provider_request[{phase}]", copy.deepcopy(payload))
        answer = await real_invoke(callback, phase, payload, timeout, events)
        note(f"provider_response[{phase}]", copy.deepcopy(answer))
        return answer

    monkeypatch.setattr(source_correction, "_invoke", recording_invoke)

    # ── 5. adapter-normalised review ────────────────────────────────────────
    real_strict_review = correction_protocol_strict_review()

    def recording_strict_review(response, request):
        out = real_strict_review(response, request)
        note("adapter_review", out)
        return out

    monkeypatch.setattr("correction_protocol.strict_review_from_indexed",
                        recording_strict_review)

    # ── 6. validator result ─────────────────────────────────────────────────
    real_validate = source_correction._validate_review

    def recording_validate(response, lines, source_map, inventory):
        out = real_validate(response, lines, source_map, inventory)
        claims, incomplete, missing_items = out
        note("validator", {
            "verdicts": [c["verdict"] for c in claims],
            "incomplete": sorted(incomplete),
            "missing_items": [i["quote"] for i in missing_items],
        })
        return out

    monkeypatch.setattr(source_correction, "_validate_review", recording_validate)

    # ── the factory ────────────────────────────────────────────────────────
    reviewer = ControlledReviewer()
    draft = _recorded_draft()

    from llm_client_adapter import LLMClientAdapter
    import stages.star_editor as star_editor_module
    import stages.drafter as drafter_module

    async def scripted_call(self, system_prompt, user_prompt, model,
                            temperature=0.7, response_format=None, timeout=None,
                            fallback_models=None, **kwargs):
        if system_prompt == star_editor_module.STAR_EDITOR_SYSTEM_PROMPT:
            # The recorded writer output, unchanged. This is the input under test.
            return json.dumps({"final_markdown": draft,
                               "final_plain_text": draft,
                               "editorial_notes": "recorded writer output"})
        if "researcher" in system_prompt.lower() or "domain" in system_prompt.lower():
            return json.dumps({"implied_metrics": [], "domain_vocab": [],
                               "implied_skills": [], "research_notes": ""})
        return json.dumps({
            "rewritten_experiences": [
                {"company": "Cascade Power Solutions, LLC",
                 "role": "Journeyman Electrician",
                 "duration": "03/2020 - Present",
                 "bullets": [
                     "Installed and maintained commercial electrical systems across "
                     "40+ job sites.",
                 ]},
                {"company": "Harborline Mechanical Contractors",
                 "role": "Electrical Apprentice",
                 "duration": "06/2016 - 02/2020",
                 "bullets": [
                     "Completed 8,000 hours of supervised field work toward "
                     "licensure.",
                 ]},
            ],
            "seniority_applied": "senior",
        })

    monkeypatch.setattr(LLMClientAdapter, "call_llm_async", scripted_call)

    def run():
        """Issue the request. Called by a test, after every recorder is installed."""
        with TestClient(app.app) as client:
            client.portal.call(app.app.state.http_session.close)
            app.app.state.http_session = reviewer
            body = client.post(
                "/analyze", headers={"X-Cogito-Owner-Id": "offline-owner"},
                json={"resume_text": SOURCE, "job_ad": "Senior Electrician role"},
            ).json()
            return client, body, reviewer, captured

    yield run
    app.app.dependency_overrides.clear()


def correction_protocol_strict_review():
    """The real ``strict_review_from_indexed``, captured before it is replaced.

    Looked up through the module so the recorder wraps the function the adapter actually
    calls. Reading it from a local import would capture a different binding and the
    spy would silently do nothing — the same class of mistake this harness exists to
    avoid.
    """
    import correction_protocol
    return correction_protocol.strict_review_from_indexed


@pytest.fixture
def preserved(boundaries):
    """Convenience: run the route once with recorders active."""
    return boundaries()


# ---------------------------------------------------------------------------

def test_education_record_is_one_source_backed_line(preserved):
    """[route-preserve] Degree, field, institution and date survive as one record.

    Before (recorded draft):
        **Associate of Applied Science, Example Technical College** -- 2016
      — one bold span concatenating degree and institution, and the field of study
      missing. The bold text is not a name the source contains, so the route's
      independent entity check refuses it.

    After:
        **Example Technical College** -- Example City, WA
        *Associate of Applied Science, Electrical Systems Technology | 2016*
      — the source's own two-line record, with every value from the source.
    """
    _client, body, _reviewer, _captured = preserved
    assert body.get("correction_status") == "approved", (
        body.get("correction_reason"), body.get("correction_commentary"))
    md = body["final_written_section_markdown"]

    education = re.search(r"^##\s+Education\s*\n(.*?)(?=^##\s|\Z)", md,
                          re.MULTILINE | re.DOTALL)
    assert education, "Education section missing"
    section = education.group(1)

    # One record, both parts sourced.
    assert "**Example Technical College**" in section
    assert "*Associate of Applied Science, Electrical Systems Technology | 2016*" in section

    # The concatenation is gone: the bold span is the institution alone.
    for bold in re.findall(r"\*\*(.+?)\*\*", section):
        excess = bold.replace("Example Technical College", "").strip(" ,-–")
        assert excess == "", f"bold span still concatenates: {bold!r}"

    # And the field of study, which the draft had dropped, is present.
    assert "Electrical Systems Technology" in section

    # The route's own entity check must agree.
    from approved_resume import unsupported_entities
    assert unsupported_entities(md, SOURCE) == []


def test_quantified_qualifier_and_no_unsupported_mastery(preserved):
    """[route-preserve] "8,000 hours of supervised field work" keeps "supervised".

    Before (recorded draft):
        - Accumulated 8,000 hours of hands-on field experience toward journeyman
          licensure, mastering core electrical installations.
      — the number survived, its governing word did not, and an outcome was appended.

    After:
        - Completed 8,000 hours of supervised field work toward licensure.
      — the candidate's own sentence, verbatim.
    """
    _client, body, _reviewer, _captured = preserved
    md = body["final_written_section_markdown"]

    assert "8,000 hours" in md
    assert "supervised field work" in md, "the source qualifier was lost"
    assert "hands-on" not in md, "the replaced qualifier survived"
    assert "mastering core electrical installations" not in md, (
        "the unsupported outcome survived")
    assert "journeyman licensure" not in md, (
        "the claim about which licence the hours counted toward survived")


def test_summary_keeps_source_facts_without_proficiency_list(preserved):
    """[route-preserve] Summary facts survive; the appended proficiency list does not.

    Before:
        Journeyman electrician with 8+ years ... power distribution, proficient in
        NFPA 70E, Arc Flash safety, electrical systems, apprentice supervision, and
        inspections.
    After:
        Journeyman electrician with 8+ years ... power distribution.
    """
    _client, body, _reviewer, _captured = preserved
    md = body["final_written_section_markdown"]

    summary = re.search(r"^##\s+Professional Summary\s*\n(.*?)(?=^##\s|\Z)", md,
                        re.MULTILINE | re.DOTALL)
    assert summary, "Professional Summary section missing"
    text = summary.group(1)

    for fact in ("Journeyman electrician", "8+ years", "commercial and industrial",
                 "construction", "facilities maintenance", "power distribution"):
        assert fact in text, f"source summary fact lost: {fact}"

    assert "proficient in" not in text, "the appended proficiency list survived"
    assert "apprentice supervision" not in text, (
        "an invented rank survived in the summary")


def test_core_headings_present_with_projects_empty(preserved):
    """[route-preserve] Five core headings present; Projects empty, summary sourced.

    An empty Projects section is the correct representation of a candidate with no
    projects. It must be present and must contain nothing.
    """
    _client, body, _reviewer, _captured = preserved
    md = body["final_written_section_markdown"]

    for heading in ("Professional Summary", "Professional Experience", "Projects",
                    "Education", "Certifications", "Technical Skills"):
        assert re.search(rf"^##\s+{re.escape(heading)}\s*$", md, re.MULTILINE), (
            f"heading missing: {heading}")

    projects = re.search(r"^##\s+Projects\s*\n(.*?)(?=^##\s|\Z)", md,
                         re.MULTILINE | re.DOTALL)
    assert projects is not None
    assert projects.group(1).strip() == "", (
        f"Projects must be empty, got: {projects.group(1)!r}")

    # The summary heading is only justified because the source supplies a summary.
    assert source_preservation.source_summary(SOURCE).strip()


def test_preservation_made_no_unsourced_edits(preserved):
    """[route-preserve] Every preserved line exists verbatim in the source.

    The module's contract is substitution, never composition. Checked against the source
    rather than against the module, so this cannot pass because the module agrees with
    itself.
    """
    _client, body, _reviewer, _captured = preserved
    md = body["final_written_section_markdown"]
    source_lines = {line.strip().lstrip("-").strip()
                    for line in SOURCE.splitlines() if line.strip()}

    for preserved_text in (
        "Completed 8,000 hours of supervised field work toward licensure.",
        "*Associate of Applied Science, Electrical Systems Technology | 2016*",
    ):
        assert preserved_text in md, preserved_text

    assert "Completed 8,000 hours of supervised field work toward licensure." in source_lines
    assert "*Associate of Applied Science, Electrical Systems Technology | 2016*" in source_lines


# ---------------------------------------------------------------------------
# [renderer-real] the real files carry the preserved record
# ---------------------------------------------------------------------------

pdflatex = pytest.mark.skipif(shutil.which("pdflatex") is None,
                              reason="no LaTeX toolchain on this machine")


def _pdf_text(path):
    result = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[:300]
    return result.stdout


def _docx_text(path):
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    return re.sub(r"<[^>]+>", " ", xml)


@pdflatex
def test_exported_files_carry_the_preserved_record(preserved):
    """[renderer-real] Real PDF and DOCX, content read back.

    The receipt binds the approved Markdown, so this also proves the files rendered from
    the content that was approved — not a re-parse that happened to look similar.
    """
    client, body, _reviewer, _captured = preserved
    payload = dict(body.get("approved_export_payload") or {})
    assert payload, "no export payload"
    payload["publication_receipt"] = body["publication_receipt"]
    payload["final_written_section_markdown"] = body["final_written_section_markdown"]
    # The approval may be policy-bound (receipt v5). The browser transports this
    # state verbatim; omitting it makes a v5 approval unverifiable, which is the
    # correct fail-closed behaviour rather than a false success.
    if body.get("skill_inference_policy_state"):
        payload["skill_inference_policy_state"] = body["skill_inference_policy_state"]

    response = client.post("/resume/export",
                           headers={"X-Cogito-Owner-Id": "offline-owner"},
                           json=payload)
    assert response.status_code == 200, response.text[:400]
    data = response.json()
    assert data.get("pdf_base64") and data.get("docx_base64")

    with tempfile.TemporaryDirectory() as directory:
        pdf = Path(directory) / "out.pdf"
        docx = Path(directory) / "out.docx"
        pdf.write_bytes(base64.b64decode(data["pdf_base64"]))
        docx.write_bytes(base64.b64decode(data["docx_base64"]))
        assert pdf.stat().st_size > 1000
        assert docx.stat().st_size > 1000
        pdf_text = _pdf_text(pdf)
        docx_text = _docx_text(docx)

    for label, text in (("pdf", pdf_text), ("docx", docx_text)):
        flat = " ".join(text.split())
        # The education record reached the page with all four parts.
        for part in ("Example Technical College", "Associate of Applied Science",
                     "Electrical Systems Technology", "2016"):
            assert part in flat, f"{label}: education part missing: {part}"
        # The qualifier survived to the page and the corruptions did not.
        assert "supervised field work" in flat, f"{label}: qualifier missing"
        assert "hands-on" not in flat, f"{label}: replaced qualifier on the page"
        assert "mastering core electrical installations" not in flat, label
        # The summary facts reached the page without the proficiency list.
        assert "facilities maintenance" in flat, label
        assert "proficient in" not in flat, f"{label}: proficiency list on the page"
        # No mojibake.
        assert "\u00e2\u0080" not in text, f"{label}: mojibake"
        assert "\u00c3" not in text, f"{label}: mojibake"


# ---------------------------------------------------------------------------
# [harness] the recorders are live, and the seams are readable
# ---------------------------------------------------------------------------

def test_recorders_are_installed_before_the_request_runs(preserved):
    """[harness] Every named seam recorded a value.

    This is the regression test for the mistake that cost this session hours: spies
    installed after the request observes nothing, and the absence looks exactly like
    "the code never ran". If any recorder is missing from the trace, this fails loudly
    and names it, instead of leaving a later assertion to fail for a misleading reason.
    """
    _client, _body, _reviewer, captured = preserved
    seen = captured["boundaries"]

    expected = ("editor_before_preservation", "editor_after_preservation",
                "correction_input", "correction_result")
    for name in expected:
        assert name in seen, f"recorder never fired: {name} (saw {seen})"

    # The per-phase seams only exist once the engine asks for them.
    assert any(k.startswith("provider_response[") for k in seen), seen
    assert "adapter_review" in seen, seen
    assert "validator" in seen, seen


def test_boundary_order_is_the_order_the_route_crosses_them(preserved):
    """[harness] The seams fired in causal order, not out of sequence."""
    _client, _body, _reviewer, captured = preserved
    seen = captured["boundaries"]

    def first(predicate):
        for index, name in enumerate(seen):
            if predicate(name):
                return index
        return None

    before = first(lambda n: n == "editor_before_preservation")
    after = first(lambda n: n == "editor_after_preservation")
    corr = first(lambda n: n == "correction_input")
    adapter = first(lambda n: n == "adapter_review")
    validator = first(lambda n: n == "validator")

    assert None not in (before, after, corr, adapter, validator), seen
    assert before < after < corr < adapter < validator, seen


def test_preservation_changed_the_document_handed_to_correction(preserved):
    """[boundary 2 vs 3] Preservation is what the engine reviewed.

    The first boundary values make the causal chain checkable without re-deriving
    anything: the editor's pre-preservation markdown still carries the defects, the
    post-preservation markdown does not, and the engine's input is the latter.
    """
    _client, _body, _reviewer, captured = preserved

    before = captured["editor_before_preservation"]
    after = captured["editor_after_preservation"]
    correction_input = captured["correction_input"]

    assert "mastering core electrical installations" in before, (
        "the writer output under test should still carry the defect")
    assert "mastering core electrical installations" not in after
    assert correction_input == after, (
        "the engine must review the preserved document, not an earlier copy")


def test_validator_found_nothing_incomplete_for_the_preserved_draft(preserved):
    """[boundary 6] The first seam where uncertainty appeared was the adapter's
    translation, not preservation or the engine.

    Recorded so the specific failure this session chased cannot recur silently: three
    source items arrived at the validator as `uncertain` because the reviewer cited only
    one of the two segments each draft line restates, and the adapter emits `uncertain`
    for an uncited required item. With decisions authored per line — including both
    skills segments on the combined line — the validator reports nothing incomplete.
    """
    _client, _body, _reviewer, captured = preserved
    validator = captured["validator"]

    assert validator["incomplete"] == [], validator
    assert validator["missing_items"] == [], validator
    assert "unsupported" not in validator["verdicts"], validator["verdicts"]


def test_reviewer_used_the_authored_decisions_it_matched(preserved):
    """[harness] The verdicts come from the authored table, not from a threshold.

    The reviewer records which decisions it matched. Asserting on them shows the table
    was actually consulted, so the approval is traceable to hand-authored judgements
    about this fixture rather than to a similarity score tuned until the run passed.
    """
    _client, _body, reviewer, _captured = preserved

    matched = set(reviewer.matched_decisions)
    assert matched, "the reviewer matched no authored decision"
    # Spot-check the cases that matter, including the combined skills line.
    assert "Journeyman electrician with 8+ years" in matched
    assert "Completed 8,000 hours of supervised field work" in matched
    assert "*Associate of Applied Science, Electrical Systems Technology | 2016*" in matched
    # The invented categorised skills block is replaced by preservation BEFORE the
    # reviewer sees the draft, so the reviewer's table covers the source's own two skill
    # lines. The old assertion looked for `**Electrical:**`, which no longer reaches it.
    assert "- Commercial electrical systems, conduit bending, panel labeling" in matched
    assert "- NFPA 70E, lockout/tagout, inspection checklists" in matched
    assert "**Electrical:**" not in matched, (
        "the invented skills block should have been removed before review")


def test_engine_approved_the_preserved_draft(preserved):
    """[boundary 4] The engine's own verdict, read from the recorded boundary."""
    _client, body, _reviewer, captured = preserved
    status, reason, reviews, repairs = captured["correction_result"]

    assert status == "approved", (status, reason)
    assert reason is None
    assert body.get("correction_status") == "approved"
    assert body.get("publication_receipt"), "an approval must carry a receipt"


def test_eight_thousand_hours_is_attached_to_the_harborline_record(preserved):
    """The quantified fact lands with the employer it belongs to.

    "8,000 supervised hours" is Harborline's apprenticeship bullet, not Cascade's. A
    preserved fact attached to the wrong employer is the misattribution class the
    answer key forbids, and an end-to-end assertion that merely finds the string
    somewhere in the document would not catch it.
    """
    _client, body, _reviewer, _captured = preserved
    md = body["final_written_section_markdown"]

    cascade = md.index("**Cascade Power Solutions, LLC**")
    harborline = md.index("**Harborline Mechanical Contractors**")
    education = md.index("## Education")
    hours = md.index("Completed 8,000 hours of supervised field work toward licensure.")

    assert cascade < harborline < education
    assert harborline < hours < education, (
        "the 8,000-hour fact must sit under Harborline, not Cascade")
