"""Captured semantic negatives through the real route, adapter, engine and gate.

The four cases below are the exact semantic failures from delivery run 4:

  1. "six apprentices"       -> "six apprentice crews"           (people -> teams)
  2. rework reduction 18%    -> safety-incident reduction 18%    (re-attributed metric)
  3. credential name         -> credential + invented issue year (fabricated date)
  4. "inspection checklists" -> + Agile / Scrum / Gantt charts    (added tools)

Each is driven through the REAL website route:

    POST /analyze -> app._analyze_resume_impl
                  -> _run_source_correction -> CorrectionProvider -> scripted HTTP
                  -> source_correction.correct -> publication gate -> receipt

Only two things are scripted, both explicitly labelled:

* the **writer** output (the pipeline returns a fixed draft). That is the input under
  test, not the thing being proved.
* the **reviewer decisions** (a scripted HTTP session answering the index-list
  protocol). Those answers are written here, independently of the engine, from the
  answer key's expectations — not derived from what the engine would like to hear.

``CorrectionResult`` is never overridden: whatever the engine decides is what the route
publishes or withholds.

Labelled **[route-semantic]**.
"""
import json
import socket

import pytest
from fastapi.testclient import TestClient

import app
import correction_provider
from approved_resume import export_payload
from tests.fixtures.fidelity import expected_facts as facts


SOURCE = facts.SOURCE_TEXT
TITLE = "Journeyman Electrician"
EMPLOYER = facts.EMPLOYMENT_RECORDS[0]["employer"]


# ---------------------------------------------------------------------------
# Scripted reviewer: answers the index-list protocol as an honest reviewer would
# ---------------------------------------------------------------------------

class ScriptedReviewer:
    """A reviewer that flags the draft lines containing any of ``defects``.

    It answers the *index-list* protocol a real model sees, so the adapter's
    translation is exercised as well as the engine's validation. Lines it considers
    unsupported are reported unsupported; everything else is supported and cites the
    source segments whose words it reuses.
    """

    def __init__(self, defects=(), restore_title=False):
        self.defects = tuple(defects)
        self.restore_title = restore_title
        self.requests = []
        self.phases = []

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
            return {"restorations": [
                {"line_id": line["line_id"],
                 "replacement": line["text"].replace("Professional", TITLE)}
                for line in payload["lines"]]}
        return self._review(payload)

    def _review(self, payload):
        # The adapter's review request carries the source items as numbered
        # ``segments`` (one per inventory item) plus the draft ``lines``. There is no
        # ``source_claims`` key on the wire: that is the engine's internal shape.
        segments, lines = payload["segments"], payload["lines"]
        flagged = {line["line_id"] for line in lines
                   if any(mark in line["text"] for mark in self.defects)}

        supported, unsupported, structural, evidence = [], [], [], {}
        for line in lines:
            text, lid = line["text"], line["line_id"]
            if text.lstrip().startswith("#"):
                structural.append(lid)
                continue
            if lid in flagged:
                unsupported.append(lid)
                continue
            supported.append(lid)
            # Cite every segment whose words this line reuses. The engine requires a
            # representation's claim to cite the item across its exact span, so a
            # reviewer naming only a stray word would be answered with _Invalid; that
            # is the engine being strict, not this fixture being generous.
            words = set(text.casefold().split())
            cited = [s["index"] for s in segments
                     if s["text"].strip() and set(s["text"].casefold().split()) & words]
            evidence[lid] = sorted(set(cited)) or [0]

        represented, missing = [], []
        supported_text = {line["line_id"]: line["text"] for line in lines
                          if line["line_id"] in set(supported)}
        for segment in segments:
            quote = segment["text"]
            restated = any(set(quote.casefold().split()) & set(text.casefold().split())
                           for text in supported_text.values())
            (represented if restated else missing).append(segment["index"])

        return {"supported": supported, "unsupported": unsupported, "uncertain": [],
                "structural": structural, "evidence": evidence,
                "represented": represented, "missing": missing}


def _envelope(answer):
    return json.dumps({
        "model": correction_provider.OR_SLUG_DRAFTER,
        "provider": "ScriptedReviewer",
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
# Route harness: real route, scripted writer, REVIEWER decisions injected
# ---------------------------------------------------------------------------

@pytest.fixture
def route(monkeypatch):
    """Provide a factory: route(reviewer, draft) -> (response_json, reviewer)."""
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
    # The disclosure policy is the existing approved one; this fixture does not
    # broaden it. It only stands in for the availability check that needs keys.
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available",
                        lambda: True)

    def build(reviewer, draft):
        monkeypatch.setattr(app.app.state, "http_session", reviewer, raising=False)
        monkeypatch.setattr(app.app.state, "http_session",
                            reviewer, raising=False)

        async def run_pipeline(**kwargs):
            return {
                "pipeline_status": "completed",
                "final_written_section_markdown": draft,
                "final_written_section": draft,
                "rewritten_resume": draft,
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
            response = client.post(
                "/analyze", headers={"X-Cogito-Owner-Id": "offline-owner"},
                json={"resume_text": SOURCE, "job_ad": "Senior Electrician role"})
        app.app.dependency_overrides.clear()
        return response

    return build


# ---------------------------------------------------------------------------
# The defective drafts, one per captured semantic failure
# ---------------------------------------------------------------------------

def _draft(*bullet_lines):
    # The summary line is included because the real source's summary spans two
    # source lines. Omitting it leaves a required source fact genuinely unstated,
    # which the engine correctly treats as incomplete coverage — a fault of the
    # fixture, not of the resume under test.
    return (
        "## Professional Summary\n"
        "Journeyman electrician with 8+ years of commercial and industrial "
        "electrical experience across construction, facilities maintenance, and "
        "power distribution.\n"
        "## Professional Experience\n"
        f"**{EMPLOYER}** -- Example City, WA\n"
        "*Journeyman Electrician | 03/2020 - Present*\n"
        + "\n".join(f"- {line}" for line in bullet_lines) + "\n"
        f"\n**{facts.EMPLOYMENT_RECORDS[1]['employer']}** -- Sampletown, OR\n"
        "*Electrical Apprentice | 06/2016 - 02/2020*\n"
        "- Assisted with conduit bending, wire pulling, and panel terminations "
        "on 25 projects.\n"
        "- Completed 8,000 hours of supervised field work toward licensure.\n"
        "\n## Education\n"
        "**Example Technical College** -- Example City, WA\n"
        "*Associate of Applied Science, Electrical Systems Technology | 2016*\n"
        "\n## Certifications\n"
        "- Washington Journey Level Electrician License\n"
        "- NFPA 70E Arc Flash Safety\n"
        "\n## Skills\n"
        "- Commercial electrical systems, conduit bending, panel labeling\n"
        "- NFPA 70E, lockout/tagout, inspection checklists\n"
    )


CASES = {
    # name: (defective line the writer produced, substring identifying it)
    "apprentice_crews": (
        "Trained and supervised six apprentice crews, delivering arc flash "
        "training and lockout/tagout procedures.",
        "apprentice crews"),
    "reattributed_metric": (
        "Cut safety incidents by 18% through improved inspection checklists.",
        "safety incidents"),
    "invented_credential_date": (
        "Washington Journey Level Electrician License -- Washington State "
        "Department of Labor & Industries (2020)",
        "Labor & Industries"),
    "added_tools": (
        "**Project Management:** Agile, Scrum, Gantt Charts, Inspection Checklists",
        "Agile"),
}


# ---------------------------------------------------------------------------
# [route-semantic] each captured failure is corrected or withheld
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(CASES))
def test_captured_semantic_defect_is_corrected_or_withheld(route, case):
    """A reviewer flagging the defect must stop it reaching the customer.

    The outcome accepted here is exactly the acceptance table's: corrected, or
    withheld. Both are safe; silently publishing the defect is not.
    """
    line, marker = CASES[case]
    draft = _draft(
        "Installed and maintained commercial electrical systems across 40+ "
        "job sites.",
        "Cut rework by 18% by standardizing panel labeling and inspection "
        "checklists.",
        line,
    )
    reviewer = ScriptedReviewer(defects=[marker])
    response = route(reviewer, draft)
    body = response.json()

    published = body.get("final_written_section_markdown") or ""
    assert body.get("correction_status") in {"approved", "withheld"}, body
    if body.get("correction_status") == "approved":
        # CORRECTED: the defect is gone and a receipt exists, so it was published
        # only after correction.
        assert marker.lower() not in published.lower(), (
            f"{case}: defect survived into the published resume")
        assert body.get("publication_receipt"), f"{case}: approved without receipt"
    else:
        # WITHHELD: nothing exportable, no receipt.
        assert body.get("approved_export_payload") is None
        assert not body.get("publication_receipt")
        assert body.get("processing_status") in {"needs_review", "withheld"}


@pytest.mark.parametrize("case", sorted(CASES))
def test_reviewer_actually_ran_for_each_case(route, case):
    """The decision came from the reviewer through the real adapter, not by accident."""
    line, marker = CASES[case]
    draft = _draft(
        "Installed and maintained commercial electrical systems across 40+ job sites.",
        line,
    )
    reviewer = ScriptedReviewer(defects=[marker])
    route(reviewer, draft)
    assert "inventory" in reviewer.phases
    assert "review" in reviewer.phases
    # The provider really posted to the wire (one request per phase at minimum).
    assert len(reviewer.requests) >= 2


def test_defect_free_draft_is_approved_with_a_receipt(route):
    """[route-semantic] The control: a faithful draft publishes with a receipt.

    Without this, "everything is withheld" would pass every negative above.
    """
    draft = _draft(
        "Installed and maintained commercial electrical systems across 40+ job sites.",
        "Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices.",
        "Cut rework by 18% by standardizing panel labeling and inspection checklists.",
    )
    reviewer = ScriptedReviewer(defects=[])
    body = route(reviewer, draft).json()

    assert body.get("correction_status") == "approved", body.get("correction_reason")
    assert body.get("publication_receipt"), "an approved draft must carry a receipt"
    assert body.get("approved_export_payload")
    published = body["final_written_section_markdown"]
    assert "40+ job sites" in published
    assert "6 apprentices" in published
    assert "18%" in published
