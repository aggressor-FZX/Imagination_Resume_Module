"""Restoration maps selected lines to their own source facts, then re-reviews."""

import asyncio
import copy
import json

import pytest

import correction_protocol as protocol
import correction_provider
import source_correction as engine
from correction_provider import CorrectionProvider
from source_correction import SourceDocument, correct
from tests.test_restoration_integration import ScriptedHTTP


def request():
    return {
        "source_claims": [{"id": "A", "quote": "Alpha role"},
                          {"id": "B", "quote": "Beta role"}],
        "lines": [{"line_id": "L1", "text": "original alpha", "source_claim_id": "A"},
                  {"line_id": "L2", "text": "original beta", "source_claim_id": "B"}],
    }


def test_reversed_answers_keep_engine_assigned_mapping():
    indexed = protocol.indexed_restore_request(request())
    assert [line["segment_index"] for line in indexed["lines"]] == [0, 1]
    assert all("source_claim_id" not in line for line in indexed["lines"])
    actual = protocol.strict_restore_from_indexed({"restorations": [
        {"line_id": "L2", "replacement": "Beta role"},
        {"line_id": "L1", "replacement": "Alpha role"},
    ]}, indexed)
    assert [(x["line_id"], x["source_claim_id"], x["before"]) for x in actual["restorations"]] == [
        ("L2", "B", "original beta"), ("L1", "A", "original alpha")]


@pytest.mark.parametrize("mutate", [
    lambda p: p["lines"][0].pop("source_claim_id"),
    lambda p: p["lines"][0].update(source_claim_id="unknown"),
    lambda p: p["lines"][1].update(source_claim_id="A"),
    lambda p: p["lines"][1].update(line_id="L1"),
    lambda p: p["source_claims"][1].update(id="A"),
    lambda p: p["lines"].pop(),
])
def test_ambiguous_request_is_rejected(mutate):
    payload = request()
    mutate(payload)
    with pytest.raises(protocol.MalformedIndexedResponse):
        protocol.indexed_restore_request(payload)


@pytest.mark.parametrize("answers", [
    [],
    [{"line_id": "L1", "replacement": "Alpha role"}],
    [{"line_id": "L1", "replacement": "Alpha role"},
     {"line_id": "L1", "replacement": "Alpha role"}],
    [{"line_id": "L1", "replacement": "Alpha role"},
     {"line_id": "L3", "replacement": "Beta role"}],
    [{"line_id": "L1", "replacement": "Alpha\nrole"},
     {"line_id": "L2", "replacement": "Beta role"}],
])
def test_bad_answer_cannot_be_translated(answers):
    with pytest.raises(protocol.MalformedIndexedResponse):
        protocol.strict_restore_from_indexed({"restorations": answers},
                                            protocol.indexed_restore_request(request()))


def test_duplicate_quote_text_keeps_distinct_source_identity():
    payload = request()
    payload["source_claims"][1]["quote"] = payload["source_claims"][0]["quote"]
    indexed = protocol.indexed_restore_request(payload)
    result = protocol.strict_restore_from_indexed({"restorations": [
        {"line_id": "L1", "replacement": "Alpha role"},
        {"line_id": "L2", "replacement": "Alpha role"},
    ]}, indexed)
    assert [item["source_claim_id"] for item in result["restorations"]] == ["A", "B"]


class TwoTargetModel:
    def __init__(self, missing):
        self.missing = missing
        self.phases = []
        self.restore_wire = None

    def answer(self, phase, payload):
        self.phases.append(phase)
        if phase == "inventory":
            return {"required": [s["index"] for s in payload["segments"]],
                    "non_resume": [], "uncertain": []}
        if phase == "restore":
            self.restore_wire = copy.deepcopy(payload)
            return {"restorations": [
                {"line_id": line["line_id"],
                 "replacement": payload["segments"][line["segment_index"]]["text"].strip()}
                for line in reversed(payload["lines"])]}
        segments = payload["segments"]
        first = self.phases.count("review") == 1
        missing = self.missing if first else []
        if self.missing == [0, 2]:
            missing = [0] if not first else self.missing
        return {"supported": [line["line_id"] for line in payload["lines"]],
                "unsupported": [], "uncertain": [], "structural": [],
                "evidence": {line["line_id"]: [line["candidates"][0]]
                             for line in payload["lines"]},
                "represented": [i for i in range(len(segments)) if i not in missing],
                "missing": missing}


def run_engine(monkeypatch, source, draft, missing):
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available", lambda: True)
    model = TwoTargetModel(missing)
    session = ScriptedHTTP(model)
    provider = CorrectionProvider("synthetic-key", session)
    result = asyncio.run(correct(draft, (SourceDocument("candidate", source),),
                                 provider, timeout_seconds=60))
    return result, model, session


def test_two_targets_travel_through_real_provider_and_full_review(monkeypatch):
    source = "Alpha manager role\nBeta technician role\n"
    draft = "Alpha role\nBeta role\n"
    result, model, session = run_engine(monkeypatch, source, draft, [0, 1])
    assert result.status == "approved", result.reason
    assert model.phases == ["inventory", "review", "restore", "review"]
    assert result.restore_calls == 2
    assert "Alpha manager role" in result.markdown
    assert "Beta technician role" in result.markdown
    assert [x["segment_index"] for x in model.restore_wire["lines"]] == [0, 1]
    assert "source_claim_id" not in json.dumps(model.restore_wire)
    assert len(session.requests) == 4


def test_skipped_first_missing_item_cannot_be_mapped_to_later_line(monkeypatch):
    original = engine._pick_target_line

    def pick(lines, quote, source_line=""):
        if "Zebra qualification" in quote:
            return None
        return original(lines, quote, source_line)

    monkeypatch.setattr(engine, "_pick_target_line", pick)
    source = "Zebra qualification\nAlpha role\nBeta technician role\n"
    draft = "Alpha role\nBeta role\n"
    result, model, _ = run_engine(monkeypatch, source, draft, [0, 2])
    assert result.status == "withheld"
    assert result.reason == "source_facts_missing"
    assert model.restore_wire["segments"] == [{"index": 0, "text": "Beta technician role\n"}]
    assert model.phases == ["inventory", "review", "restore", "review"]


def test_invalid_mapping_never_reaches_provider(monkeypatch):
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available", lambda: True)
    model = TwoTargetModel([])
    session = ScriptedHTTP(model)
    provider = CorrectionProvider("synthetic-key", session)
    payload = request()
    payload["lines"][0]["source_claim_id"] = "unknown"
    payload["sources"] = [{"source_id": "candidate", "text": "Alpha role\nBeta role\n"}]
    with pytest.raises(protocol.MalformedIndexedResponse):
        asyncio.run(provider("restore", payload))
    assert not session.requests


def test_wrong_source_id_rejects_entire_batch():
    markdown = "Alpha role\nBeta role\n"
    lines = engine._lines(markdown)
    targets = {lines[0]["line_id"]: {"id": "A", "quote": "Alpha role"},
               lines[1]["line_id"]: {"id": "B", "quote": "Beta role"}}
    response = {"restorations": [
        {"line_id": lines[0]["line_id"], "source_claim_id": "A",
         "before": "Alpha role", "replacement": "Alpha manager role"},
        {"line_id": lines[1]["line_id"], "source_claim_id": "A",
         "before": "Beta role", "replacement": "Beta technician role"},
    ]}
    with pytest.raises(engine._Invalid):
        engine._apply_restorations(markdown, response, targets, lines)
    assert markdown == "Alpha role\nBeta role\n"


def test_post_restore_unsupported_fact_cannot_publish(monkeypatch):
    original = TwoTargetModel.answer

    def answer(self, phase, payload):
        result = original(self, phase, payload)
        if phase == "review" and self.phases.count("review") == 2:
            line_id = payload["lines"][0]["line_id"]
            result["supported"].remove(line_id)
            result["unsupported"].append(line_id)
        return result

    monkeypatch.setattr(TwoTargetModel, "answer", answer)
    result, model, _ = run_engine(monkeypatch, "Alpha manager role\nBeta technician role\n",
                                  "Alpha role\nBeta role\n", [0, 1])
    assert model.phases[:4] == ["inventory", "review", "restore", "review"]
    assert result.status == "withheld"
    assert result.markdown is None