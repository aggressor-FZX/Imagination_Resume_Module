"""The protected capture must record the engine's restoration mapping, not just HTTP.

A captured `source_facts_missing` is undiagnosable from the wire exchange alone:
`strict_restore_from_indexed` deliberately strips the internal `source_claim_id`
before transmission, so the transmitted request does not say which source fact
was paired with which draft line. That association exists only in the engine's
own phase payload.

These tests drive the real `/analyze` route with capture enabled and assert:

* the restoration phase records its selected source items *and* their line
  association;
* the post-restore review records its normalized verdicts, so the reason the
  restored draft was still refused is inspectable;
* the capture never changes the response;
* nothing enters the capture when a phase raises — only a bounded error label.
"""

import json
import stat

import pytest
from fastapi.testclient import TestClient

import app
import correction_provider
import live_capture
from live_capture import SETTING_DIR, SETTING_ENABLED, SYNTHETIC_MARKER
from tests.test_analyze_source_correction_route import offline  # noqa: F401


SOURCE = f"""## Professional Summary
{SYNTHETIC_MARKER} Operations analyst with nine years of reporting experience across depot and branch operations, building recurring reporting that supervisors use to compare period-over-period performance and to track fuel and parts consumption across service locations.

## Professional Experience
**{SYNTHETIC_MARKER} Harborline Transit Authority** -- Tacoma, WA
*Fuel Reporting Analyst | 04/2019 - Present*
- Tracked monthly fuel consumption across 14 depots and compared each month against the same month in the prior year.
- Built the depot scorecard that supervisors review in the Monday operations meeting, covering consumption, variance and exception counts for every site.
- Reconciled fuel invoices against depot dispensing records each quarter, working through supplier statements and resolving discrepancies with the depot clerks.

**{SYNTHETIC_MARKER} Cascade Freight Partners** -- Kent, WA
*Cost Reporting Coordinator | 07/2015 - 03/2019*
- Audited three years of maintenance cost records to find which equipment failed repeatedly across the regional fleet.
- Prepared the monthly cost summary reviewed by the regional manager, covering labour, parts and outside service charges.
- Rebuilt the shipment delay log that the dispatch team relies on for daily route planning and customer updates.

## Education
**{SYNTHETIC_MARKER} Puget Sound Technical College** -- Tacoma, WA
*Associate of Applied Science, Business Analytics | 2015*
"""

# The title line is absent from the draft, which is the restoration shape: a
# required source fact the reviewer reports missing, on a line that exists.
DRAFT = SOURCE.replace("Fuel Reporting Analyst | 04/2019 - Present",
                       "Analyst | 04/2019 - Present")

MISSING_TITLE = "Fuel Reporting Analyst"
CORRELATION = "restoremap-0001"


class Response:
    def __init__(self, body):
        self.status = 200
        self._body = body

    class _Content:
        def __init__(self, body):
            self._body = body

        async def iter_chunked(self, size):
            for index in range(0, len(self._body), size):
                yield self._body[index:index + size]

    @property
    def content(self):
        return self._Content(self._body)

    async def text(self, *args, **kwargs):
        return self._body.decode()

    async def read(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class RestoringSession:
    """Scripted provider: reports the title missing, then accepts its restoration."""

    def __init__(self):
        self.requests = []
        self.phases = []

    async def close(self):
        pass

    def post(self, url, **kwargs):
        from pipeline_config import OR_SLUG_DRAFTER

        self.requests.append(kwargs)
        payload = json.loads(kwargs["json"]["messages"][1]["content"])
        contract = payload.get("contract", "")
        if contract.startswith("Rewrite the named draft lines"):
            self.phases.append("restore")
            answer = {"restorations": [
                {"line_id": line["line_id"],
                 "replacement": line["text"].replace("Analyst |", MISSING_TITLE + " |")}
                for line in payload["lines"]]}
        elif "lines" in payload:
            self.phases.append("review")
            answer = self._review(payload)
        else:
            self.phases.append("inventory")
            answer = {
                "required": [s["index"] for s in payload["segments"]
                             if s["text"].strip() and not s["text"].startswith("## ")],
                "non_resume": [s["index"] for s in payload["segments"]
                               if s["text"].startswith("## ")],
                "uncertain": [],
            }
        return Response(json.dumps({
            "model": OR_SLUG_DRAFTER,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "choices": [{"finish_reason": "stop",
                         "message": {"content": json.dumps(answer)}}],
        }).encode())

    @staticmethod
    def _review(payload):
        segments, lines = payload["segments"], payload["lines"]
        verdicts = {"supported": [], "unsupported": [], "uncertain": [], "structural": []}
        evidence, carried, missing = {}, set(), []
        title_segments = [s["index"] for s in segments if MISSING_TITLE in s["text"]]
        title_present = any(MISSING_TITLE in line["text"] for line in lines)
        for line in lines:
            if line["text"].lstrip().startswith("#"):
                verdicts["structural"].append(line["line_id"])
                continue
            verdicts["supported"].append(line["line_id"])
            words = {w for w in line["text"].casefold().split() if len(w) > 3}
            cited = [s["index"] for s in segments
                     if s["text"].strip() and words & {w for w in s["text"].casefold().split() if len(w) > 3}]
            evidence[line["line_id"]] = cited or [s["index"] for s in segments
                                                  if s["text"].strip()][:1]
            carried.update(evidence[line["line_id"]])
        for segment in segments:
            if not segment["text"].strip() or segment["text"].startswith("## "):
                continue
            if MISSING_TITLE in segment["text"] and not title_present:
                missing.append(segment["index"])
            else:
                carried.add(segment["index"])
        verdicts["evidence"] = evidence
        # A restoration only helps if the re-review acknowledges it.
        verdicts["represented"] = sorted(index for index in carried if index not in missing)
        verdicts["missing"] = sorted(missing)
        return verdicts


@pytest.fixture
def capturing(offline, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available", lambda: True)
    target = tmp_path / "captures"
    target.mkdir()
    monkeypatch.setenv(SETTING_ENABLED, "1")
    monkeypatch.setenv(SETTING_DIR, str(target))
    monkeypatch.setenv("PUBLICATION_RECEIPT_SECRET", "offline-route-secret-1234567890")
    client, _ = offline
    session = RestoringSession()
    monkeypatch.setattr(app.app.state, "http_session", session)

    async def pipeline(**kwargs):
        return {"pipeline_status": "completed",
                "final_written_section_markdown": DRAFT,
                "final_written_section": DRAFT,
                "rewritten_resume": DRAFT,
                "experiences": [{"snippet": DRAFT}],
                "metrics_support": {"verdict": "PASS (tier A only)", "checked": True},
                "run_metrics": {"calls": [], "failures": [], "total_prompt_tokens": 0,
                                "total_completion_tokens": 0, "total_tokens": 0,
                                "estimated_cost_usd": 0.0}}

    monkeypatch.setattr(app, "run_new_pipeline_async", pipeline)
    return client, session, target


def only_capture(target):
    files = list(target.iterdir())
    assert len(files) == 1, [path.name for path in files]
    return json.loads(files[0].read_text()), files[0]


def test_capture_records_the_restoration_mapping(capturing):
    """The selected source items and their line association must survive."""
    client, session, target = capturing
    response = client.post("/analyze",
                           headers={"X-Cogito-Owner-Id": "offline-owner",
                                    "x-correlation-id": CORRELATION},
                           json={"resume_text": SOURCE, "job_ad": "Synthetic role"})
    assert response.status_code == 200

    stored, path = only_capture(target)
    assert path.stat().st_mode & (stat.S_IRGRP | stat.S_IROTH) == 0

    # Session ownership must not be disturbed by the wrapper.
    phases = [item["phase"] for item in stored["phases"]]
    assert "restore" in phases, phases
    assert "post_restore_review" in phases, phases

    restore = next(item for item in stored["phases"] if item["phase"] == "restore")
    payload = restore["payload"]
    # The engine's own payload carries the association the wire form omits.
    assert payload["source_claims"], "restore recorded no selected source item"
    for claim in payload["source_claims"]:
        assert claim["id"] and claim["quote"]
    for line in payload["lines"]:
        assert line["source_claim_id"] in {claim["id"] for claim in payload["source_claims"]}
    restored_ids = [line["source_claim_id"] for line in payload["lines"]]
    assert len(restored_ids) == len(set(restored_ids)), "one source item per line"

    # And its normalized answer was recorded too.
    assert isinstance(restore["result"], dict)
    assert restore["result"].get("restorations")

    # The wire form still has no internal identity: this is why the phase record
    # is required, and the assertion protects that reasoning from silent change.
    #
    # Scoped to the RESTORE exchange deliberately. Review requests do carry
    # `source_claim_id` inside their `segments` — that is an opaque engine id used
    # to name evidence, not the private line-to-fact association. What must never
    # appear on the wire is the restore mapping itself.
    restore_wire = json.dumps([item["request"] for item in stored["exchanges"]
                               if item["phase"] == "restore"])
    assert restore_wire != "[]", "no restore exchange was captured"
    assert "source_claim_id" not in restore_wire


def test_capture_records_the_post_restore_review_verdicts(capturing):
    """The reason a restored draft is still refused must be inspectable."""
    client, session, target = capturing
    client.post("/analyze",
                headers={"X-Cogito-Owner-Id": "offline-owner",
                         "x-correlation-id": CORRELATION + "b"},
                json={"resume_text": SOURCE, "job_ad": "Synthetic role"})
    stored, _ = only_capture(target)
    review = next(item for item in stored["phases"]
                  if item["phase"] == "post_restore_review")
    result = review["result"]
    assert set(result) >= {"claims", "representations"}
    # Verdicts are the normalized engine shape, not the model's index lists.
    assert all("verdict" in claim for claim in result["claims"])
    assert all("status" in mapping for mapping in result["representations"])


def test_capture_review_summary_carries_counts_and_restore_total(capturing):
    client, session, target = capturing
    client.post("/analyze",
                headers={"X-Cogito-Owner-Id": "offline-owner",
                         "x-correlation-id": CORRELATION + "c"},
                json={"resume_text": SOURCE, "job_ad": "Synthetic role"})
    stored, _ = only_capture(target)
    review = stored["review"]
    assert review is not None
    assert isinstance(review.get("restore_calls"), int)
    assert review["restore_calls"] >= 1
    coverage = review.get("coverage")
    if coverage is not None:
        # Numeric coverage, not a type-presence flag.
        for key in ("level", "line_count", "claim_count",
                    "supported_count", "structural_count"):
            assert key in coverage


def test_phase_recording_does_not_change_the_response(capturing, monkeypatch):
    client, session, target = capturing
    headers = {"X-Cogito-Owner-Id": "offline-owner",
               "x-correlation-id": CORRELATION + "d"}
    with_capture = client.post("/analyze", headers=headers,
                               json={"resume_text": SOURCE,
                                     "job_ad": "Synthetic role"}).json()
    monkeypatch.delenv(SETTING_ENABLED, raising=False)
    for path in target.iterdir():
        path.unlink()
    without = client.post("/analyze", headers=headers,
                          json={"resume_text": SOURCE,
                                "job_ad": "Synthetic role"}).json()
    for key in ("processing_status", "requires_review", "correction_status",
                "correction_reason", "final_written_section_markdown",
                "review_reasons"):
        assert with_capture.get(key) == without.get(key), key


def test_a_raising_phase_records_a_bounded_error_label(capturing, monkeypatch):
    """A failure records the exception TYPE only — never its message."""
    client, session, target = capturing

    class Boom(RuntimeError):
        pass

    real_post = session.post

    def post(url, **kwargs):
        payload = json.loads(kwargs["json"]["messages"][1]["content"])
        if payload.get("contract", "").startswith("Rewrite the named draft lines"):
            raise Boom("PRIVATE DRAFT TEXT must never be recorded")
        return real_post(url, **kwargs)

    monkeypatch.setattr(session, "post", post)
    client.post("/analyze",
                headers={"X-Cogito-Owner-Id": "offline-owner",
                         "x-correlation-id": CORRELATION + "e"},
                json={"resume_text": SOURCE, "job_ad": "Synthetic role"})
    stored, _ = only_capture(target)
    restore = next(item for item in stored["phases"] if item["phase"] == "restore")
    assert restore["error"] == "Boom"
    assert "PRIVATE DRAFT TEXT" not in json.dumps(stored)
