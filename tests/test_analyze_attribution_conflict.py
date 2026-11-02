"""Synthetic offline /analyze coverage; scripted reviews are not semantic proof.

The provider double below is a deliberately *lying* reviewer: it approves text by
record body and cites every source record with that body, without ever checking
the employer label. That makes the deterministic label correction (and the
ambiguity rejection) observable in the real route rather than attributing the
outcome to an honest model.
"""
import copy
import json

import pytest

import app
import correction_provider
from pipeline_config import OR_SLUG_DRAFTER
from tests.test_analyze_source_correction_route import offline, pipeline, post, result_from_sse  # noqa: F401
from tests.test_correction_integration_offline import (
    FRAGMENT, SECOND_FRAGMENT, SOURCE as BASE_SOURCE, FakeSession, Response, inventory,
)


SOURCE = BASE_SOURCE.replace(
    "- Maintained the workshop inventory", "- Betaa Works: Maintained the workshop inventory"
).replace(
    "- Improved diagnostic accuracy", "- Alpha Works: Improved diagnostic accuracy"
)
CORRECT_LINE = next(line for line in SOURCE.splitlines() if line.startswith("- Alpha Works:"))
BETA_LINE = next(line for line in SOURCE.splitlines() if line.startswith("- Betaa Works:"))
WRONG_LINE = CORRECT_LINE.replace("Alpha Works:", "Betaa Works:", 1)
# Extra blank line makes the positive draft distinct from the source; the route
# deliberately treats an exact source copy as a pipeline fallback.
GOOD_DRAFT = SOURCE.replace("\n\n## Education", "\n\n\n## Education")
BAD_DRAFT = GOOD_DRAFT.replace(CORRECT_LINE, WRONG_LINE)
REPAIR_FRAGMENT = " in 6 months"
AMBIGUOUS_BODY = BETA_LINE.split(": ", 1)[1]
# Two different source labels legitimately own this exact record body, so no
# unique owner exists. Alpha Works is a known source label that owns a different
# record, so assigning the shared body to Alpha Works is neither an unknown
# alias nor a uniquely repairable label.
AMBIGUOUS_SOURCE = SOURCE.replace(
    BETA_LINE,
    f"- Gamma Works: {AMBIGUOUS_BODY}\n- Beta Works: {AMBIGUOUS_BODY}",
)
AMBIGUOUS_DRAFT = GOOD_DRAFT.replace(BETA_LINE, f"- Alpha Works: {AMBIGUOUS_BODY}")


def _record_body(text):
    """Return the record body of an explicit ``label: body`` line, else None."""
    stripped = text[2:] if text.startswith("- ") else text
    if ": " not in stripped:
        return None
    label, body = stripped.split(": ", 1)
    if not label or label.startswith("#") or len(label) > 80:
        return None
    return body


def record_inventory(payload):
    """Scripted inventory: explicit ``label: body`` lines are required.

    Headings and blank lines are declared non_resume by this test double so the
    attribution contract can be observed in isolation; that is a fixture
    decision, not a claim that headings are irrelevant in production. Blank
    lines are folded into an adjacent nonblank span because the real validator
    rejects whitespace-only items and requires an exact contiguous partition.
    """
    assert set(payload) == {"contract", "sources"}
    items = []
    for source in payload["sources"]:
        text, sid = source["text"], source["source_id"]
        start = position = 0
        for part in text.splitlines(keepends=True):
            position += len(part)
            if not part.strip():
                continue
            classification = "required" if _record_body(part.strip()) else "non_resume"
            items.append({"id": f"{sid}:{start}:{position}", "source_id": sid,
                          "start": start, "end": position,
                          "quote": text[start:position],
                          "classification": classification})
            start = position
        if position > start:
            # Trailing whitespace joins the final span rather than standing alone.
            last = items[-1]
            last["end"] = position
            last["id"] = f"{sid}:{last['start']}:{position}"
            last["quote"] = text[last["start"]:position]
    return {"source_claims": items}


def body_review(payload):
    """Approve draft lines by record body, citing every source owner.

    Schema-valid but semantically wrong: it never verifies that the label on the
    draft line owns the body. It is the adversarial reviewer, not an oracle. An
    inserted duration suffix is flagged at its exact offsets so the repair step
    removes only that suffix.
    """
    claims = []
    spans = {item["id"]: [] for item in payload["source_claims"]}
    required = {item["id"]: _record_body(item["quote"].strip().splitlines()[-1])
                for item in payload["source_claims"]
                if item["classification"] == "required"}
    for line in payload["lines"]:
        text, line_id = line["text"], line["line_id"]
        body = _record_body(text)
        if body is None:
            claims.append({"id": f"{line_id}:0:{len(text)}", "line_id": line_id,
                           "start": 0, "end": len(text), "verdict": "structural",
                           "citations": []})
            continue
        suffix = next((part for part in (FRAGMENT, SECOND_FRAGMENT, REPAIR_FRAGMENT)
                       if part in body), None)
        if suffix:
            start = text.index(suffix)
            end = start + len(suffix)
            clean_body = _record_body(text[:start] + text[end:])
        else:
            start = end = None
            clean_body = body
        owners = [item for item in payload["source_claims"]
                  if item["id"] in required and required[item["id"]] == clean_body]
        if not owners:
            raise ValueError("unscripted synthetic draft line")
        citation = [{key: item[key] for key in ("source_id", "start", "end", "quote")}
                    for item in owners]
        if suffix is None:
            claims.append({"id": f"{line_id}:0:{len(text)}", "line_id": line_id,
                           "start": 0, "end": len(text), "verdict": "supported",
                           "citations": citation})
            for item in owners:
                spans[item["id"]].append({"line_id": line_id, "start": 0, "end": len(text)})
        else:
            claims.append({"id": f"{line_id}:0:{start}", "line_id": line_id, "start": 0,
                           "end": start, "verdict": "supported", "citations": citation})
            claims.append({"id": f"{line_id}:{start}:{end}", "line_id": line_id,
                           "start": start, "end": end, "verdict": "unsupported",
                           "citations": []})
            claims.append({"id": f"{line_id}:{end}:{len(text)}", "line_id": line_id,
                           "start": end, "end": len(text), "verdict": "supported",
                           "citations": citation})
            for item in owners:
                spans[item["id"]].append({"line_id": line_id, "start": 0, "end": start})
                spans[item["id"]].append({"line_id": line_id, "start": end,
                                          "end": len(text)})
    mappings = []
    for item in payload["source_claims"]:
        if item["classification"] == "non_resume":
            mappings.append({"source_claim_id": item["id"], "status": "non_resume", "spans": []})
            continue
        if not spans[item["id"]]:
            # Test-double policy: a required record whose exact body is absent is
            # reported missing, never silently accepted.
            mappings.append({"source_claim_id": item["id"], "status": "missing", "spans": []})
            continue
        mappings.append({"source_claim_id": item["id"], "status": "represented",
                         "spans": spans[item["id"]]})
    return {"claims": claims, "representations": mappings}


class ScriptedSession(FakeSession):
    """Real adapter, in-memory transport, body-based lying reviewer."""

    def post(self, url, **kwargs):
        self.requests.append(kwargs)
        payload = json.loads(kwargs["json"]["messages"][1]["content"])
        try:
            if "flagged_claim_ids" in payload:
                body = {"edits": [{"claim_id": cid, "replacement": ""}
                                  for cid in payload["flagged_claim_ids"]]}
            elif "lines" not in payload:
                body = record_inventory(payload)
            else:
                body = body_review(payload)
        except Exception:
            import os
            import traceback
            if os.environ.get("CORRECTION_FIXTURE_DEBUG"):
                traceback.print_exc()
            raise
        return Response(json.dumps({
            "model": OR_SLUG_DRAFTER,
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(body)}}],
        }).encode())


@pytest.fixture
def synthetic_transport(offline, monkeypatch):
    """Opt in only for these synthetic documents and in-memory transport."""
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available", lambda: True)
    session = ScriptedSession()
    monkeypatch.setattr(app.app.state, "http_session", session)
    return offline[0], session


def capture_real_correction(monkeypatch):
    actual = app.correct
    results = []

    async def observe(*args, **kwargs):
        result = await actual(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(app, "correct", observe)
    return results


def phases(result):
    return [call["phase"] for call in result["run_metrics"]["calls"]
            if call["stage"] == "source_correction"]


@pytest.mark.parametrize("sse", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize("repair_first", [False, True], ids=["initial", "post-patch"])
def test_wrong_employer_is_corrected_and_published(synthetic_transport, monkeypatch, sse, repair_first):
    client, session = synthetic_transport
    draft = BAD_DRAFT.replace("30% using", "30% in 6 months using") if repair_first else BAD_DRAFT
    pipeline(monkeypatch, draft=draft)
    observed = capture_real_correction(monkeypatch)
    response = post(client, source=SOURCE, sse=sse)
    assert response.status_code == 200
    if sse:
        assert response.headers["content-type"].startswith("text/event-stream")
        blocks, result = result_from_sse(response)
        assert [block.split("\n", 1)[0] for block in blocks] == ["event: token", "event: result"]
        assert json.loads(blocks[0].split("data: ", 1)[1]) == {
            "field": "rewrite", "delta": GOOD_DRAFT}
    else:
        assert response.headers["content-type"].startswith("application/json")
        result = response.json()

    expected_phases = ["source_inventory", "initial_review"]
    if repair_first:
        expected_phases += ["repair", "post_patch_review"]
    assert phases(result) == expected_phases
    assert len(session.requests) == len(expected_phases)
    assert len(observed) == 1
    assert (observed[0].status, observed[0].reason) == ("approved", None)
    assert observed[0].markdown == GOOD_DRAFT
    assert (observed[0].review_calls, observed[0].repair_calls) == (
        (2, 1) if repair_first else (1, 0))
    assert result["processing_status"] == "completed" and result["requires_review"] is False
    assert result["correction_status"] == "approved"
    assert result["final_written_section_markdown"] == GOOD_DRAFT
    assert result["publication_receipt"] and result["approved_export_payload"]
    assert "restored 1 source-owned record label" in result["correction_commentary"]
    assert "independent review is required" in result["correction_commentary"]
    # The accomplishment and unrelated facts survive the correction. The
    # unsupported duration fragment is removed by repair in the post-patch case,
    # so neither variant contains it; the supported 30% metric and both
    # employers' own lines must remain.
    export = result["approved_export_payload"]
    bullets = [bullet for item in export["experience"] for bullet in item["bullets"]]
    assert any(bullet.startswith("Alpha Works: Improved diagnostic accuracy")
               and "30%" in bullet for bullet in bullets), bullets
    assert any("Betaa Works: Maintained the workshop inventory" in bullet
               for bullet in bullets), bullets
    assert not any("Betaa Works: Improved diagnostic accuracy" in bullet
                   for bullet in bullets), bullets
    assert "diagnostic accuracy" in result["final_written_section_markdown"]
    assert len(export["education"]) == 2
    assert sum("Synthetic College" in item for item in export["education"]) == 1
    for private in ("PRIVATE_", FRAGMENT, REPAIR_FRAGMENT):
        assert private not in response.text


@pytest.mark.parametrize("sse", [False, True], ids=["json", "sse"])
def test_unresolved_multi_owner_attribution_is_publicly_withheld(offline, monkeypatch, sse):
    # Negative control: two source records legitimately share this body and the
    # draft assigns it to a third known label. A body-only reviewer approves it,
    # so only the deterministic ambiguity check can reject publication.
    client, _ = offline
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available", lambda: True)
    session = ScriptedSession()
    monkeypatch.setattr(app.app.state, "http_session", session)
    pipeline(monkeypatch, draft=AMBIGUOUS_DRAFT)
    observed = capture_real_correction(monkeypatch)
    response = post(client, source=AMBIGUOUS_SOURCE, sse=sse)
    assert response.status_code == 200
    if sse:
        blocks, result = result_from_sse(response)
        assert [block.split("\n", 1)[0] for block in blocks] == ["event: withheld", "event: result"]
        withheld = json.loads(blocks[0].split("data: ", 1)[1])
    else:
        result = response.json()
    assert len(observed) == 1
    # Rejection must be a decision, not a callback, schema or timeout failure.
    assert (observed[0].status, observed[0].reason) == ("withheld", "attribution_conflict")
    assert observed[0].repair_calls == 0
    assert phases(result) == ["source_inventory", "initial_review"]
    assert result["processing_status"] == "needs_review" and result["requires_review"] is True
    assert result["correction_status"] == "withheld"
    assert result["correction_reason"] == "attribution_conflict"
    assert result["publication_receipt"] is None and result["approved_export_payload"] is None
    assert result["final_written_section_markdown"] is None
    if sse:
        assert withheld["reasons"] == ["attribution_conflict"]


@pytest.mark.parametrize("sse", [False, True], ids=["json", "sse"])
def test_correct_attribution_positive_control(synthetic_transport, monkeypatch, sse):
    client, session = synthetic_transport
    pipeline(monkeypatch, draft=GOOD_DRAFT)
    observed = capture_real_correction(monkeypatch)
    response = post(client, source=SOURCE, sse=sse)
    assert response.status_code == 200
    if sse:
        blocks, result = result_from_sse(response)
        assert [block.split("\n", 1)[0] for block in blocks] == ["event: token", "event: result"]
        assert json.loads(blocks[0].split("data: ", 1)[1]) == {
            "field": "rewrite", "delta": GOOD_DRAFT}
    else:
        result = response.json()
    assert len(observed) == 1 and observed[0].status == "approved"
    assert observed[0].markdown == GOOD_DRAFT and observed[0].reason is None
    assert phases(result) == ["source_inventory", "initial_review"] and len(session.requests) == 2
    assert result["processing_status"] == "completed" and result["requires_review"] is False
    assert result["final_written_section_markdown"] == GOOD_DRAFT
    assert result["publication_receipt"] and result["approved_export_payload"]
    assert "restored" not in result["correction_commentary"]
    assert "PRIVATE_" not in response.text


@pytest.mark.parametrize("sse", [False, True], ids=["json", "sse"])
def test_normal_disclosure_boundary_remains_closed(offline, monkeypatch, sse):
    client, session = offline
    pipeline(monkeypatch, draft=BAD_DRAFT)
    observed = capture_real_correction(monkeypatch)
    response = post(client, source=SOURCE, sse=sse)
    assert response.status_code == 200
    if sse:
        blocks, result = result_from_sse(response)
        assert [block.split("\n", 1)[0] for block in blocks] == ["event: withheld", "event: result"]
    else:
        result = response.json()
    assert len(observed) == 1 and observed[0].reason == "privacy_review_required"
    assert result["correction_reason"] == "privacy_review_required"
    assert result["requires_review"] and result["publication_receipt"] is None
    assert result["approved_export_payload"] is None and result["final_written_section_markdown"] is None
    assert not session.requests and "PRIVATE_" not in response.text and WRONG_LINE not in response.text