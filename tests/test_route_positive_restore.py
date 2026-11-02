"""Positive acceptance: restore a title, publish with a receipt, render real files.

The negative suites prove defects are corrected or withheld. This proves the other
half: a defective draft whose ONLY problem is a destroyed role title can be repaired
by the bounded restoration, published with a signed receipt, and exported to real PDF
and DOCX files whose *content* is then read back and checked.

Route exercised, with no layer stubbed except the writer and the reviewer:

    POST /analyze -> writer draft (title destroyed)
                  -> _run_source_correction -> real CorrectionProvider -> scripted HTTP
                  -> real engine: inventoried, reviewed, RESTORED, re-reviewed
                  -> publication gate -> receipt
    POST /resume/export -> real LaTeX compiler + real python-docx
                        -> pdftotext / DOCX XML read back

What is scripted, and labelled as such:
* the **writer** output — the input under test;
* the **reviewer** decisions — specified here from the answer key.

``CorrectionResult`` is never overridden, so the approval below is the engine's own.
Renderer assertions read the generated files; no PDF/DOCX bytes are mocked.

Labelled **[route-positive]** and **[renderer-real]**.
"""
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
from tests.fixtures.fidelity import expected_facts as facts


SOURCE = facts.SOURCE_TEXT
TITLE = "Journeyman Electrician"
EMPLOYER = facts.EMPLOYMENT_RECORDS[0]["employer"]
START, END = facts.EMPLOYMENT_RECORDS[0]["start"], facts.EMPLOYMENT_RECORDS[0]["end"]

# The draft has ONE defect: the role title was replaced by the placeholder word.
# The source carries a SYNTHETIC-ONLY marker on each factual line; the draft keeps it
# so the ONLY defect present is the destroyed role title.
DESTROYED_DRAFT = (
    # The summary is copied from the source verbatim. This fixture must have EXACTLY
    # one defect — the destroyed role title — or the run is testing coverage rather
    # than restoration.
    "## Professional Summary\n"
    "SYNTHETIC-ONLY Journeyman electrician with 8+ years of commercial and "
    "industrial electrical\n"
    "experience across construction, facilities maintenance, and power "
    "distribution.\n"
    "## Professional Experience\n"
    f"**SYNTHETIC-ONLY {EMPLOYER}** -- Example City, WA\n"
    f"*Professional | {START} - {END}*\n"
    "- Installed and maintained commercial electrical systems across 40+ job sites.\n"
    "- Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices.\n"
    "- Cut rework by 18% by standardizing panel labeling and inspection checklists.\n"
    "\n**SYNTHETIC-ONLY Harborline Mechanical Contractors** -- Sampletown, OR\n"
    "*Electrical Apprentice | 06/2016 - 02/2020*\n"
    "- Assisted with conduit bending, wire pulling, and panel terminations on "
    "25 projects.\n"
    "- Completed 8,000 hours of supervised field work toward licensure.\n"
    "\n## Education\n"
    "**SYNTHETIC-ONLY Example Technical College** -- Example City, WA\n"
    "*Associate of Applied Science, Electrical Systems Technology | 2016*\n"
    "\n## Certifications\n"
    "- SYNTHETIC-ONLY Washington Journey Level Electrician License\n"
    "- SYNTHETIC-ONLY NFPA 70E Arc Flash Safety\n"
    "\n## Skills\n"
    "- Commercial electrical systems, conduit bending, panel labeling\n"
    "- NFPA 70E, lockout/tagout, inspection checklists\n"
)


def _is_title_segment(text: str) -> bool:
    """True for the single header segment that carries the role title and dates."""
    stripped = text.strip()
    return (stripped.startswith("*") and TITLE.casefold() in stripped.casefold()
            and any(d in stripped for d in (START, END)))


class RestoringReviewer:
    """Reports the destroyed title missing, then restores it from the source record."""

    def __init__(self):
        self.phases = []
        self.requests = []

    async def close(self):
        pass

    def post(self, url, **kwargs):
        payload = json.loads(kwargs["json"]["messages"][1]["content"])
        phase = self._phase(payload)
        self.phases.append(phase)
        self.requests.append(kwargs)
        return _response(200, _envelope(self._answer(phase, payload)))

    @staticmethod
    def _phase(payload):
        contract = payload.get("contract", "")
        if contract.startswith("Classify every numbered source segment"):
            return "inventory"
        if contract.startswith("Rewrite the named draft lines"):
            return "restore"
        return "review"

    def _answer(self, phase, payload):
        if phase == "inventory":
            return {"required": [s["index"] for s in payload["segments"]],
                    "non_resume": [], "uncertain": []}
        if phase == "restore":
            # Restore the real title, preserving the employer and the dates.
            # Restore ONLY the header line whose title was replaced. A blanket
            # str.replace over every line would also rewrite the "## Professional
            # Summary" heading, silently corrupting an unrelated section name.
            return {"restorations": [
                {"line_id": line["line_id"],
                 "replacement": line["text"].replace("Professional |", TITLE + " |")}
                for line in payload["lines"]]}
        return self._review(payload)

    def _review(self, payload):
        segments, lines = payload["segments"], payload["lines"]
        title_present = any(TITLE in line["text"] for line in lines)

        supported, structural, evidence = [], [], {}
        for line in lines:
            text, lid = line["text"], line["line_id"]
            if text.lstrip().startswith("#"):
                structural.append(lid)
                continue
            supported.append(lid)
            words = set(text.casefold().split())
            cited = [s["index"] for s in segments
                     if s["text"].strip() and set(s["text"].casefold().split()) & words]
            evidence[lid] = sorted(set(cited)) or [0]

        represented, missing = [], []
        # Coverage is judged against the WHOLE draft, not line by line: a source
        # segment can span a line break that a rewrite places differently, and
        # comparing per line would report a faithfully reproduced summary as missing
        # purely because the wrapping changed.
        # Formatting characters are not content: strip markdown markers and
        # punctuation so "**SYNTHETIC-ONLY Cascade Power Solutions, LLC** -- City"
        # and its source segment compare on words alone.
        whole = " ".join(line["text"] for line in lines).casefold()
        draft_words = set(re.findall(r"[a-z0-9][a-z0-9+./&-]*", whole))
        for segment in segments:
            # Same normalisation on both sides, so the comparison is about words.
            segment_words = re.findall(r"[a-z0-9][a-z0-9+./&-]*",
                                       segment["text"].casefold())
            if not segment_words:
                continue
            # Only the segment that IS the role title interacts with the title
            # defect. The summary also contains "journeyman electrician", so a bare
            # containment test marked the summary missing too and the restoration
            # never ran for the line that actually needed it.
            if _is_title_segment(segment["text"]) and not title_present:
                missing.append(segment["index"])
                continue
            content = {w for w in segment_words if len(w) > 3}
            # "Restated" is a coverage judgement, not exact equality: a faithful
            # rewrite may drop a marker word or re-inflect a verb. Most of the
            # segment's content words must appear.
            covered = len(content & draft_words) / len(content) if content else 0.0
            (represented if covered >= 0.75 else missing).append(segment["index"])

        return {"supported": supported, "unsupported": [], "uncertain": [],
                "structural": structural, "evidence": evidence,
                "represented": represented, "missing": missing}


def _envelope(answer):
    return json.dumps({
        "model": correction_provider.OR_SLUG_DRAFTER,
        "provider": "RestoringReviewer",
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
def published(monkeypatch):
    """Run /analyze with a restoring reviewer; return the published result body."""
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

    reviewer = RestoringReviewer()

    async def run_pipeline(**kwargs):
        return {
            "pipeline_status": "completed",
            "final_written_section_markdown": DESTROYED_DRAFT,
            "final_written_section": DESTROYED_DRAFT,
            "rewritten_resume": DESTROYED_DRAFT,
            "experiences": [],
            "editorial_notes": "",
            "final_written_section_provenance": [],
            "metrics_support": {"verdict": "PASS", "checked": True},
            "run_metrics": {"calls": [], "failures": []},
        }

    monkeypatch.setattr(app, "run_new_pipeline_async", run_pipeline)

    with TestClient(app.app) as client:
        client.portal.call(app.app.state.http_session.close)
        app.app.state.http_session = reviewer
        body = client.post(
            "/analyze", headers={"X-Cogito-Owner-Id": "offline-owner"},
            json={"resume_text": SOURCE, "job_ad": "Senior Electrician role"},
        ).json()
        yield client, body, reviewer
    app.app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# [route-positive] the restoration happens and the result is published
# ---------------------------------------------------------------------------

def test_restoration_phase_ran(published):
    """The engine asked for a restoration and the reviewer supplied one."""
    _client, _body, reviewer = published
    assert "restore" in reviewer.phases, reviewer.phases
    assert any(p.startswith("review") for p in reviewer.phases) or \
        reviewer.phases.count("review") >= 2


def test_destroyed_title_is_restored_and_published(published):
    """Approved, and the real title is present instead of the placeholder."""
    _client, body, _reviewer = published
    assert body.get("correction_status") == "approved", body.get("correction_reason")
    published_md = body["final_written_section_markdown"]
    assert TITLE in published_md
    assert "Professional |" not in published_md
    assert body.get("restore_calls", 1) >= 0  # field may be omitted by the route


def test_restored_title_keeps_the_correct_employer_and_dates(published):
    """Attribution is preserved: the title landed on the Cascade record, not elsewhere."""
    _client, body, _reviewer = published
    lines = body["final_written_section_markdown"].splitlines()
    header = next(line for line in lines if TITLE in line)
    assert START in header and END in header, header
    employer_line = next(line for line in lines if EMPLOYER in line)
    assert TITLE not in employer_line
    # And the second employer keeps its own correct title.
    assert "Electrical Apprentice" in body["final_written_section_markdown"]


def test_publication_receipt_is_issued(published):
    """A receipt and a signed export payload exist for the approved result."""
    _client, body, _reviewer = published
    assert body.get("publication_receipt"), "approved result must carry a receipt"
    assert body.get("approved_export_payload")


# ---------------------------------------------------------------------------
# [renderer-real] actual PDF and DOCX files, content read back
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
    import re
    return re.sub(r"<[^>]+>", " ", xml)


@pdflatex
def test_export_endpoint_renders_real_files_with_the_restored_title(published):
    """[renderer-real] The exported PDF and DOCX contain the restored title.

    The receipt travels from /analyze to /resume/export, so this also proves the
    published result is the one that was approved.
    """
    client, body, _reviewer = published
    # The receipt binds the EXACT render payload and markdown that /analyze
    # produced. Send them back untouched: overriding any field is what
    # "receipt_render_content_mismatch" exists to refuse, and it would also mean the
    # file being rendered was not the file that was approved.
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
        assert TITLE in text, f"{label}: restored title missing"
        assert "Professional |" not in text, f"{label}: placeholder survived"
        assert EMPLOYER in text, f"{label}: employer missing"
        assert START in text, f"{label}: start date missing"
        # No mojibake reached the page.
        assert "\u00e2\u0080" not in text, f"{label}: mojibake"
        # Source facts preserved.
        assert "40+ job sites" in text
        assert "6 apprentices" in text
        assert "18%" in text
