"""Synthetic offline /analyze coverage; scripted reviews are not semantic proof."""
import copy
import json

import pytest

import app
import correction_provider
from pipeline_config import OR_SLUG_DRAFTER
from tests.test_analyze_source_correction_route import offline, pipeline, post, result_from_sse  # noqa: F401
from tests.test_correction_integration_offline import (
    FRAGMENT, SOURCE as BASE_SOURCE, FakeSession, Response, inventory, review,
)


SOURCE = BASE_SOURCE.replace(
    "- Maintained the workshop inventory", "- Betaa Works: Maintained the workshop inventory"
).replace(
    "- Improved diagnostic accuracy", "- Alpha Works: Improved diagnostic accuracy"
)
CORRECT_LINE = next(line for line in SOURCE.splitlines() if line.startswith("- Alpha Works:"))
WRONG_LINE = CORRECT_LINE.replace("Alpha Works:", "Betaa Works:", 1)
# Extra blank line makes the positive draft distinct from the source; the route
# deliberately treats an exact source copy as a pipeline fallback.
GOOD_DRAFT = SOURCE.replace("\n\n## Education", "\n\n\n## Education")
BAD_DRAFT = GOOD_DRAFT.replace(CORRECT_LINE, WRONG_LINE)


class ScriptedSession(FakeSession):
    """Real adapter, fake transport: the reviewer lies about one attribution."""

    def post(self, url, **kwargs):
        self.requests.append(kwargs)
        payload = json.loads(kwargs["json"]["messages"][1]["content"])
        if "flagged_claim_ids" in payload:
            body = {"edits": [{"claim_id": cid, "replacement": ""}
                              for cid in payload["flagged_claim_ids"]]}
        elif "lines" not in payload:
            body = inventory(payload)
        else:
            # Equal-length labels preserve valid span offsets and citations, while
            # making the reviewer wrongly claim the Alpha source is represented.
            substituted = copy.deepcopy(payload)
            for line in substituted["lines"]:
                if line["text"].startswith("- Betaa Works: Improved diagnostic accuracy"):
                    line["text"] = line["text"].replace("Betaa Works:", "Alpha Works:", 1)
            body = review(substituted)
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
def test_attribution_conflict_is_publicly_withheld(synthetic_transport, monkeypatch, sse, repair_first):
    client, session = synthetic_transport
    draft = BAD_DRAFT.replace("30% using", "30% in 6 months using") if repair_first else BAD_DRAFT
    pipeline(monkeypatch, draft=draft)
    observed = capture_real_correction(monkeypatch)
    response = post(client, source=SOURCE, sse=sse)
    assert response.status_code == 200
    if sse:
        assert response.headers["content-type"].startswith("text/event-stream")
        blocks, result = result_from_sse(response)
        assert [block.split("\n", 1)[0] for block in blocks] == ["event: withheld", "event: result"]
        withheld = json.loads(blocks[0].split("data: ", 1)[1])
    else:
        assert response.headers["content-type"].startswith("application/json")
        result = response.json()

    expected_phases = ["source_inventory", "initial_review"]
    if repair_first:
        expected_phases += ["repair", "post_patch_review"]
    assert phases(result) == expected_phases
    assert len(session.requests) == len(expected_phases)
    assert len(observed) == 1
    assert (observed[0].status, observed[0].reason, observed[0].markdown) == (
        "withheld", "attribution_conflict", None)
    assert (observed[0].review_calls, observed[0].repair_calls) == ((2, 1) if repair_first else (1, 0))
    assert result["processing_status"] == "needs_review" and result["requires_review"] is True
    assert result["correction_status"] == "withheld"
    for field in ("final_written_section_markdown", "final_written_section", "rewritten_resume",
                  "publication_receipt", "approved_export_payload"):
        assert result[field] is None
    for private in ("PRIVATE_", WRONG_LINE, CORRECT_LINE, FRAGMENT):
        assert private not in response.text
    assert result["correction_reason"] == result["review_reason"] == "attribution_conflict"
    assert result["review_reasons"] == ["attribution_conflict"]
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
        assert json.loads(blocks[0].split("data: ", 1)[1]) == {"field": "rewrite", "delta": GOOD_DRAFT}
    else:
        result = response.json()
    assert len(observed) == 1 and observed[0].status == "approved"
    assert observed[0].markdown == GOOD_DRAFT and observed[0].reason is None
    assert phases(result) == ["source_inventory", "initial_review"] and len(session.requests) == 2
    assert result["processing_status"] == "completed" and result["requires_review"] is False
    assert result["correction_status"] == "approved"
    assert result["final_written_section_markdown"] == GOOD_DRAFT
    assert result["publication_receipt"] and result["approved_export_payload"]
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