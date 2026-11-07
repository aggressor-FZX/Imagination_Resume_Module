"""A permitted bridge label is authorised wording, not an introduced claim.

Measured on the Imaginative website run ``12ecfaa4`` (2026-11-07, revision
``8606610``). The writer inserted validated bridge labels into otherwise faithful
bullets:

  "- Audited 3 years of maintenance records using failure pattern analysis;
     identified equipment with repeated failures."

The reviewer marked every line ``supported``, but the engine's lexical
introduced-content check counted "failure pattern analysis" as content the source
never states. It downgraded the representation, and the run withheld as
``source_completeness_uncertain`` — the mode's defining benefit was unreachable.

The exemption is phrase-scoped: only the exact whole label is removed before the
check, so an unsupported outcome sitting beside a bridge is still introduced.
Conservative passes no bridges, so its behaviour is unchanged.

These tests drive ``_validate_review`` directly, which is where the check lives.
"""
import pytest

import source_correction as sc

SOURCE = ("Professional Experience\n"
          "SYNTHETIC-ONLY Audited 3 years of maintenance records to find which "
          "equipment failed repeatedly.\n")

BRIDGE = {"id": "bridge-01", "skill": "failure pattern analysis",
          "origin": "researcher_inference", "policy_version": "imaginative-v1"}

ITEM_START = SOURCE.index("SYNTHETIC-ONLY Audited")
ITEM_END = SOURCE.index("\n", ITEM_START) + 1
ITEM_QUOTE = SOURCE[ITEM_START:ITEM_END]


def _validate(text, bridges):
    lines = [{"line_id": "L1", "text": text}]
    review = {
        "claims": [{"id": f"L1:0:{len(text)}", "line_id": "L1", "start": 0,
                    "end": len(text), "verdict": "supported",
                    "citations": [{"source_id": "doc", "start": 0,
                                   "end": len(SOURCE), "quote": SOURCE}]}],
        "representations": [{"source_claim_id": f"doc:{ITEM_START}:{ITEM_END}",
                             "status": "represented",
                             "spans": [{"line_id": "L1", "start": 0,
                                        "end": len(text)}]}],
    }
    inventory = [{"id": f"doc:{ITEM_START}:{ITEM_END}", "source_id": "doc",
                  "start": ITEM_START, "end": ITEM_END, "quote": ITEM_QUOTE,
                  "classification": "required"}]
    token = sc._ACTIVE_BRIDGES.set(tuple(bridges))
    try:
        return sc._validate_review(review, lines, {"doc": SOURCE}, inventory)
    finally:
        sc._ACTIVE_BRIDGES.reset(token)


BRIDGED = ("Audited 3 years of maintenance records to find which equipment "
           "failed repeatedly using failure pattern analysis.")


def test_a_permitted_bridge_label_is_not_introduced_content():
    claims, incomplete, missing = _validate(BRIDGED, [BRIDGE])
    assert claims[0]["verdict"] == "supported"
    assert "uncertain" not in incomplete
    assert missing == []


def test_the_same_span_without_the_bridge_is_flagged():
    """Conservative passes no bridges, so the label is introduced content.

    The detector flags capitalised names and unbacked figures, so the label here
    is a capitalised tool name — the shape it actually catches.
    """
    text = ("Audited 3 years of maintenance records to find which equipment "
            "failed repeatedly using Tableau.")
    claims, incomplete, _ = _validate(text, [])
    assert "uncertain" in incomplete


def test_a_bridge_label_does_not_authorise_a_new_figure():
    text = BRIDGED + " Cutting downtime by 42%."
    claims, incomplete, _ = _validate(text, [BRIDGE])
    assert "uncertain" in incomplete


def test_a_bridge_label_does_not_authorise_a_new_tool():
    text = BRIDGED + " Using Tableau."
    claims, incomplete, _ = _validate(text, [BRIDGE])
    assert "uncertain" in incomplete


def test_a_bridge_label_does_not_authorise_a_new_credential():
    text = BRIDGED + " Holding a PMP certification."
    claims, incomplete, _ = _validate(text, [BRIDGE])
    assert "uncertain" in incomplete
