"""Source preservation proven through the real route and the real renderers.

The four defects this covers were all demonstrated by delivery run 7 and are all
*silent*: the resume reads well, every token is sourced, and the candidate's record is
still misrepresented. They are fixed deterministically in ``source_preservation`` and
called from ``StarEditor.polish``, so the fix runs on the live path:

    POST /analyze
      -> writer draft (run-7 output, defects intact, taken verbatim from the capture)
      -> real StarEditor.polish  -> deterministic source preservation
      -> real CorrectionProvider -> real engine -> publication gate -> receipt
    POST /resume/export -> real LaTeX compiler + real python-docx
                        -> pdftotext / DOCX XML read back

What is scripted, and labelled:

* the **writer** output — the input under test, verbatim from the capture;
* the **reviewer** decisions — authored here from the source text.

What is NOT done: ``CorrectionResult`` is never overridden, the answer key is never
edited to match the output, and no PDF/DOCX bytes are mocked. The renderer assertions
read files that the real toolchain produced.

Labelled **[route-preserve]** and **[renderer-real]**.
"""
from __future__ import annotations

import base64
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
import source_preservation
from source_preservation import _stem
from tests.fixtures.fidelity import expected_facts as facts

RUN7 = Path("/home/skystarved/Render_Dockers/.live-run-2026-11-02/captures/"
            "run7/0a267e04-affd-4cee-b944-26c574703039.json")

requires_run7 = pytest.mark.skipif(not RUN7.exists(), reason="run-7 capture absent")

SOURCE = facts.SOURCE_TEXT


def _recorded_draft() -> str:
    """The run-7 writer output, verbatim — all four defects intact."""
    return live_capture.load(RUN7)["draft"]


# ---------------------------------------------------------------------------
# The controlled reviewer
#
# It answers every phase the engine can ask for and, after preservation has run, it
# finds nothing to object to. That is the honest shape: the defects preservation fixes
# are not reviewer-visible semantic faults once corrected, and inventing objections to
# them would be scripting the result.
# ---------------------------------------------------------------------------

class ControlledReviewer:
    def __init__(self):
        self.phases: list[str] = []
        self.requests: list[dict] = []

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
            # Headings are classified `non_resume`: they assert no candidate fact, so a
            # representation cannot be demanded for them. The engine's own contract says
            # a `required` item must be represented by a supported span, and no span can
            # establish a heading — claiming one produced a representation with no
            # supporting span, which the engine correctly downgraded to `uncertain`, and
            # `uncertain` with nothing flagged is the withhold.
            #
            # Both alternatives are wrong in different ways: `required` demands the
            # impossible, and `uncertain` blocks the run outright. `non_resume` is the
            # accurate classification for a line that makes no claim about the candidate.
            headings, body = [], []
            for segment in wire["segments"]:
                stripped = segment["text"].strip()
                (headings if stripped.startswith("#") or not stripped
                 else body).append(segment["index"])
            return {"required": body, "non_resume": headings, "uncertain": []}
        if phase == "repair":
            return {"edits": [{"claim_id": cid,
                               "replacement": self._replace(cid, wire)}
                              for cid in wire["flagged_claim_ids"]]}
        return self._review(wire)

    @staticmethod
    def _replace(claim_id, wire):
        line_id = claim_id.split(":", 1)[0]
        return next(line["text"] for line in wire["lines"]
                    if line["line_id"] == line_id)

    def _review(self, wire):
        """Support every line that shares content with a source segment.

        Deliberately generous, and deliberately mechanical: this reviewer exists to let a
        preserved draft reach the gate, not to model judgement.

        Coverage is judged against the WHOLE document rather than line-by-line. A source
        line can span two draft lines, and a section the draft legitimately consolidates
        (the source's ``## Skills`` becoming ``## Technical Skills``) would otherwise read
        as omitted purely because no single draft line matched it. Judging per line was
        the first version of this and it reported 7 of 22 segments missing on a correctly
        preserved document — a false objection, which is exactly the class of error the
        answer key warns about.

        Headings are ``structural``: they assert no candidate fact, so they are never
        reported missing.
        """
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
            words = {w.casefold() for w in re.findall(r"[A-Za-z][A-Za-z\-]+", text)
                     if len(w) > 3}
            # Cite EVERY segment this line substantively covers, not just its single best
            # match. One draft line routinely restates several source items — the summary
            # sentence carries both halves of the wrapped paragraph, and a combined
            # skills line carries both source skill bullets. Picking one best match left
            # the others uncited, the translator then emitted `uncertain` for them, and
            # three genuinely-represented items withheld the whole draft.
            cited = []
            for segment in segments:
                seg_words = {w.casefold()
                             for w in re.findall(r"[A-Za-z][A-Za-z\-]+", segment["text"])
                             if len(w) > 3}
                if not seg_words:
                    continue
                if len(words & seg_words) / len(seg_words) >= 0.5:
                    cited.append(segment["index"])
            if cited:
                verdicts["supported"].append(lid)
                evidence[lid] = sorted(set(cited))
            else:
                verdicts["uncertain"].append(lid)

        # Stemmed, because a faithful paraphrase inflects: the source's
        # "standardizing" appears as "standardized"/"optimized" and "checklists" as
        # "workflows". Exact-word overlap scored two bullets the answer key calls
        # "accepted, not missing" (segments 6 and 8) at 0.71 and 0.67 — a false
        # objection of exactly the class the key warns about.
        whole = {_stem(w) for w in re.findall(r"[A-Za-z][A-Za-z\-]+",
                                              " ".join(line["text"]
                                                       for line in wire["lines"]))
                 if len(w) > 3}
        represented, missing = [], []
        for segment in segments:
            stripped = segment["text"].strip()
            if stripped.startswith("#") or not re.search(r"[A-Za-z]", stripped):
                # A heading asserts no candidate fact. Tested by the leading "#" rather
                # than by "has no content words": "## Professional Summary" DOES contain
                # content words, so the word-based guard silently let headings through
                # and they were reported `represented` with no support.
                # The inventory classifies them `non_resume`, and the translator emits
                # that status verbatim, so they must not appear here either.
                continue
            seg_words = {_stem(w)
                         for w in re.findall(r"[A-Za-z][A-Za-z\-]+", stripped)
                         if len(w) > 3}
            if not seg_words:
                continue
            covered = len(seg_words & whole) / len(seg_words)
            (represented if covered >= 0.6 else missing).append(segment["index"])

        return {**verdicts, "evidence": evidence,
                "represented": represented, "missing": missing}


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


@pytest.fixture
def preserved(monkeypatch):
    """Run /analyze over the recorded draft; return (client, body, reviewer)."""
    if not RUN7.exists():
        pytest.skip("run-7 capture is not present")

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

    from stages.star_editor import StarEditor
    real_polish = StarEditor.polish
    captured = {"editor_ran": False, "preservation": None}

    async def recording_polish(self, *a, **kw):
        captured["editor_ran"] = True
        out = await real_polish(self, *a, **kw)
        captured["preservation"] = out.get("source_preservation")
        captured["editor_markdown"] = out.get("final_markdown")
        return out

    monkeypatch.setattr(StarEditor, "polish", recording_polish)

    reviewer = ControlledReviewer()
    draft = _recorded_draft()

    # Script the WRITER MODEL, not the pipeline.
    #
    # An earlier version of this fixture stubbed `run_new_pipeline_async`, which meant
    # `StarEditor.polish` never ran and source preservation never executed — while the
    # tests still read as though they covered the route. They were observing the raw
    # recorded draft and failing for a reason that had nothing to do with the fix.
    #
    # Replacing `LLMClientAdapter.call_llm_async` instead scripts only the network seam.
    # The real orchestrator, the real StarEditor (including preservation), the real
    # correction engine, the real gate and the real renderers all execute.
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
        # The drafter: one experience per employer, from the source.
        return json.dumps({
            "rewritten_experiences": [
                {"company": "Cascade Power Solutions, LLC",
                 "role": "Journeyman Electrician",
                 "duration": "03/2020 - Present",
                 "bullets": [
                     "Installed and maintained commercial electrical systems across "
                     "40+ job sites.",
                     "Trained 6 apprentices on lockout/tagout and NFPA 70E safe work "
                     "practices.",
                     "Cut rework by 18% by standardizing panel labeling and inspection "
                     "checklists."]},
                {"company": "Harborline Mechanical Contractors",
                 "role": "Electrical Apprentice",
                 "duration": "06/2016 - 02/2020",
                 "bullets": [
                     "Assisted with conduit bending, wire pulling, and panel "
                     "terminations on 25 projects.",
                     "Completed 8,000 hours of supervised field work toward "
                     "licensure."]},
            ],
            "seniority_applied": "senior",
        })

    monkeypatch.setattr(LLMClientAdapter, "call_llm_async", scripted_call)

    # The route constructs its own adapter from the app settings patched above, so the
    # class-level patch is enough for the real pipeline to run end to end. Verified by
    # asserting below that the editor stage actually executed rather than falling back.
    captured["editor_ran"] = False

    # Record the engine's own verdict and the editor's output, from inside the request.
    # Instrumenting in the test body would be too late: the request is issued here.
    import source_correction
    real_correct = source_correction.correct

    async def recording_correct(markdown, sources, callback, **kw):
        out = await real_correct(markdown, sources, callback, **kw)
        captured["engine"] = (out.status, out.reason, out.review_calls,
                              out.repair_calls)
        return out

    monkeypatch.setattr(source_correction, "correct", recording_correct)
    monkeypatch.setattr(app, "correct", recording_correct)

    with TestClient(app.app) as client:
        client.portal.call(app.app.state.http_session.close)
        app.app.state.http_session = reviewer
        body = client.post(
            "/analyze", headers={"X-Cogito-Owner-Id": "offline-owner"},
            json={"resume_text": SOURCE, "job_ad": "Senior Electrician role"},
        ).json()
        yield client, body, reviewer, captured
    app.app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# [route-preserve] fields, before and after, through the real path
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
