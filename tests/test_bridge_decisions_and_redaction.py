"""Phase C.7/C.8: bridge decisions (accept/narrow/remove) and redacted anchors.

C.7 — the lifecycle must not be decorative:

* narrowing cannot escalate into an algorithm, tool, metric or proficiency claim;
* removal never destroys candidate facts — an embedded label is refused and left to
  the repair path;
* a **partial** removal is refused, because reporting success while the label
  survives is worse than refusing;
* a refused decision leaves the text byte-identical.

C.8 — correction receives *length-preservingly redacted* sources while a bridge
keeps the raw quote it was validated against. When PII sits inside the quoted
activity the two differ character-for-character, and an exact-equality anchor
check silently dropped a legitimate bridge. The anchor must tolerate the redacted
form without tolerating a fabricated one.
"""
import pytest

import correction_protocol as protocol
from disclosure import redact_text
from skill_bridge import build_bridges
from skill_bridge_decisions import (
    ACCEPT,
    NARROW,
    REMOVE,
    BridgeDecisionError,
    apply_decisions,
    find_occurrences,
    surviving_labels,
    validate_decision,
)


SKILLS_MD = (
    "## Technical Skills\n"
    "- trend analysis\n"
    "- reporting\n"
)


# ── C.7 decision validation ────────────────────────────────────────────────


def test_accept_is_valid_without_a_replacement():
    decision = validate_decision(
        bridge_id="b1", decision=ACCEPT, skill="trend analysis"
    )
    assert decision.decision == ACCEPT
    assert decision.replacement == ""


@pytest.mark.parametrize(
    "replacement",
    [
        "expert trend analysis",             # proficiency claim
        "proven trend analysis",             # proficiency claim
        "trend analysis 20%",                # metric with a unit
        "3 years of trend analysis",         # metric with a unit
        "a complete sentence about trends.",  # prose
        "Analysis: trend analysis",          # category markup
    ],
)
def test_narrowing_cannot_escalate_structurally(replacement):
    """A narrowing is a *shape*-checked label, same as a proposal.

    Semantic escalation (an algorithm or tool the candidate never used) is caught
    by the reviewer against the original evidence, not here — this check refuses
    only shapes that are never a label.
    """
    with pytest.raises(BridgeDecisionError):
        validate_decision(
            bridge_id="b1", decision=NARROW, skill="trend analysis",
            replacement=replacement,
        )


@pytest.mark.parametrize(
    "replacement",
    ["data analysis", "reporting", "Python", "SQL", "statistical analysis"],
)
def test_narrowing_accepts_well_shaped_labels(replacement):
    """Structural acceptance only; the reviewer still judges the wording."""
    decision = validate_decision(
        bridge_id="b1", decision=NARROW, skill="trend analysis",
        replacement=replacement,
    )
    assert decision.replacement == replacement


def test_narrowing_to_an_identical_label_is_refused():
    with pytest.raises(BridgeDecisionError):
        validate_decision(
            bridge_id="b1", decision=NARROW, skill="trend analysis",
            replacement="Trend Analysis",
        )


def test_narrowing_requires_a_replacement():
    with pytest.raises(BridgeDecisionError):
        validate_decision(bridge_id="b1", decision=NARROW, skill="trend analysis")


def test_only_narrow_may_supply_a_replacement():
    with pytest.raises(BridgeDecisionError):
        validate_decision(
            bridge_id="b1", decision=ACCEPT, skill="trend analysis",
            replacement="data analysis",
        )


def test_unknown_decision_is_refused():
    with pytest.raises(BridgeDecisionError):
        validate_decision(bridge_id="b1", decision="approve", skill="trend analysis")


def test_unknown_reason_code_is_dropped_not_echoed():
    decision = validate_decision(
        bridge_id="b1", decision=REMOVE, skill="trend analysis",
        reason="because the reviewer said so",
    )
    assert decision.reason == ""


# ── C.7 applying decisions ─────────────────────────────────────────────────


def test_accept_leaves_the_text_untouched():
    text, applied, rejected = apply_decisions(
        SKILLS_MD, [{"bridge_id": "b1", "decision": ACCEPT, "skill": "trend analysis"}]
    )
    assert text == SKILLS_MD
    assert len(applied) == 1
    assert rejected == []


def test_narrow_rewrites_the_standalone_label():
    text, applied, rejected = apply_decisions(
        SKILLS_MD,
        [{"bridge_id": "b1", "decision": NARROW, "skill": "trend analysis",
          "replacement": "data analysis"}],
    )
    assert "- data analysis" in text
    assert "trend analysis" not in text
    assert "- reporting" in text, "an unrelated skill must survive"
    assert len(applied) == 1


def test_remove_deletes_a_standalone_skill_line():
    text, applied, rejected = apply_decisions(
        SKILLS_MD, [{"bridge_id": "b1", "decision": REMOVE, "skill": "trend analysis"}]
    )
    assert "trend analysis" not in text
    assert "- reporting" in text
    assert len(applied) == 1
    assert rejected == []


def test_remove_refuses_an_embedded_label():
    """Removing a phrase from a sentence would destroy the facts around it."""
    prose = "## Experience\n- Used Python for trend analysis and increased revenue 20%\n"
    text, applied, rejected = apply_decisions(
        prose, [{"bridge_id": "b1", "decision": REMOVE, "skill": "trend analysis"}]
    )
    assert text == prose, "a refused decision must not modify the text"
    assert applied == []
    assert rejected == [{"bridge_id": "b1", "reason": "embedded_label_requires_repair"}]


def test_partial_removal_is_refused():
    """A removal that leaves the label in prose is not a removal."""
    mixed = (
        "## Technical Skills\n- trend analysis\n"
        "## Experience\n- Used Python for trend analysis and increased revenue 20%\n"
    )
    text, applied, rejected = apply_decisions(
        mixed, [{"bridge_id": "b1", "decision": REMOVE, "skill": "trend analysis"}]
    )
    assert text == mixed
    assert applied == []
    assert rejected[0]["reason"] == "embedded_label_requires_repair"


def test_accept_refuses_a_label_that_is_not_standalone():
    prose = "## Experience\n- Used Python for trend analysis and increased revenue 20%\n"
    _text, applied, rejected = apply_decisions(
        prose, [{"bridge_id": "b1", "decision": ACCEPT, "skill": "trend analysis"}]
    )
    assert applied == []
    assert rejected[0]["reason"] == "label_not_standalone"


@pytest.mark.parametrize("decision", [ACCEPT, NARROW, REMOVE])
@pytest.mark.parametrize("md", ["\n", "  \r\n", "---\n", "- **\n", "...\n"])
def test_blank_and_punctuation_only_lines_do_not_match_absent_label(decision, md):
    payload = {"bridge_id": "b1", "decision": decision, "skill": "trend analysis"}
    if decision == NARROW:
        payload["replacement"] = "data analysis"

    text, applied, rejected = apply_decisions(md, [payload])

    assert text == md
    assert applied == []
    reason = "label_not_standalone" if decision == ACCEPT else "embedded_label_requires_repair"
    assert rejected == [{"bridge_id": "b1", "reason": reason}]


@pytest.mark.parametrize("decision", [ACCEPT, NARROW, REMOVE])
def test_decisions_preserve_unrelated_blank_and_punctuation_lines(decision):
    md = "## Skills\r\n\r\n---\r\n- **Trend Analysis**\r\n...\r\n- reporting\r\n"
    payload = {"bridge_id": "b1", "decision": decision, "skill": "trend analysis"}
    if decision == NARROW:
        payload["replacement"] = "data analysis"

    text, applied, rejected = apply_decisions(md, [payload])

    expected_line = {ACCEPT: "- **Trend Analysis**\r\n",
                     NARROW: "- data analysis\r\n", REMOVE: ""}[decision]
    assert text == "## Skills\r\n\r\n---\r\n" + expected_line + "...\r\n- reporting\r\n"
    assert len(applied) == 1
    assert rejected == []


def test_invalid_decision_is_rejected_without_touching_the_text():
    text, applied, rejected = apply_decisions(
        SKILLS_MD, [{"bridge_id": "b1", "decision": NARROW, "skill": "trend analysis"}]
    )
    assert text == SKILLS_MD
    assert applied == []
    assert rejected == [{"bridge_id": "b1", "reason": "invalid_decision"}]


def test_non_object_decisions_are_rejected():
    text, applied, rejected = apply_decisions(SKILLS_MD, ["accept", None, 7])
    assert text == SKILLS_MD
    assert applied == []
    assert len(rejected) == 3


def test_word_boundaries_prevent_accidental_matches():
    """`analysis` must not match inside `analyses`."""
    assert find_occurrences("- analyses of data", "analysis") == []
    assert find_occurrences("- analysis of data", "analysis")


def test_markdown_emphasis_still_counts_as_standalone():
    md = "## Technical Skills\n- **trend analysis**\n"
    text, applied, _ = apply_decisions(
        md,
        [{"bridge_id": "b1", "decision": NARROW, "skill": "trend analysis",
          "replacement": "data analysis"}],
    )
    assert "data analysis" in text
    assert len(applied) == 1


def test_repeated_labels_narrow_every_standalone_occurrence():
    md = "## Skills\n- trend analysis\n- trend analysis\n"
    text, applied, _ = apply_decisions(
        md,
        [{"bridge_id": "b1", "decision": NARROW, "skill": "trend analysis",
          "replacement": "data analysis"}],
    )
    assert "trend analysis" not in text
    assert text.count("data analysis") == 2
    assert len(applied) == 1


def test_surviving_labels_reflects_the_decisions():
    decisions = [
        {"bridge_id": "b1", "decision": ACCEPT, "skill": "trend analysis"},
        {"bridge_id": "b2", "decision": NARROW, "skill": "time-series analysis",
         "replacement": "data analysis"},
        {"bridge_id": "b3", "decision": REMOVE, "skill": "reporting"},
    ]
    assert surviving_labels(decisions) == ["trend analysis", "data analysis"]


def test_surviving_labels_of_a_removal_is_empty():
    assert surviving_labels([{"bridge_id": "b1", "decision": REMOVE,
                              "skill": "trend analysis"}]) == []


# ── C.8 redacted anchors ───────────────────────────────────────────────────

RAW_SOURCE = (
    "Jordan Example\n"
    "Reached jordan@example.com about weekly sales figures to identify trends.\n"
)
QUOTE = "weekly sales figures to identify trends"
LABEL = "trend analysis"


def _bridge_payloads(source_text=RAW_SOURCE, quote=QUOTE):
    accepted, _rejected = build_bridges(
        [{"skill": LABEL, "evidence_quote": quote, "rationale": "Changes over time.",
          "source_id": "document-0"}],
        sources={"document-0": source_text},
        mode="imaginative",
    )
    return [bridge.to_dict() for bridge in accepted]


def _review_with(bridges, source_text):
    """Run the real adapter over ``source_text`` and return the strict claims."""
    inventory_request = protocol.indexed_inventory_request(
        {"sources": [{"source_id": "document-0", "text": source_text}]}
    )
    segments = inventory_request["_segments"]
    inventory = protocol.strict_inventory_from_indexed(
        {"required": [segment["index"] for segment in segments],
         "non_resume": [], "uncertain": []},
        inventory_request,
    )["source_claims"]

    payload = {
        "sources": [{"source_id": "document-0", "text": source_text}],
        "source_claims": inventory,
        "lines": [{"line_id": "L1", "text": "- trend analysis"}],
    }
    if bridges:
        payload["permitted_bridges"] = bridges
    request = protocol.indexed_review_request(payload)
    answer = {
        "supported": ["L1"],
        "evidence": {"L1": [len(inventory) - 1]},
        "represented": [], "missing": [], "unsupported": [], "uncertain": [],
        "structural": [],
    }
    return protocol.strict_review_from_indexed(answer, request)["claims"]


def test_redaction_is_length_preserving_so_offsets_survive():
    redacted, _count = redact_text(RAW_SOURCE)
    assert len(redacted) == len(RAW_SOURCE)


def test_bridge_survives_when_correction_receives_redacted_sources():
    """The route validates on raw text and corrects on redacted text."""
    bridges = _bridge_payloads()
    assert bridges, "the bridge should validate against the raw source"
    redacted, _count = redact_text(RAW_SOURCE)

    claims = _review_with(bridges, redacted)
    assert claims[-1]["verdict"] == "supported", (
        "a legitimate bridge must survive length-preserving redaction"
    )


def test_quote_containing_pii_still_anchors_after_redaction():
    """The hardest case: the redaction lands INSIDE the quoted activity."""
    source = "Compared sales for jordan@example.com and identified trends.\n"
    quote = "Compared sales for jordan@example.com and identified trends"
    bridges = _bridge_payloads(source_text=source, quote=quote)
    assert bridges, "the bridge should validate against the raw source"
    redacted, _count = redact_text(source)
    assert quote not in redacted, "the PII itself must be gone"

    claims = _review_with(bridges, redacted)
    assert claims[-1]["verdict"] == "supported", (
        "redaction inside the quote must not invalidate the anchor"
    )


def test_a_fabricated_anchor_still_fails_under_redaction():
    """Tolerating redaction must not tolerate forgery."""
    bridges = _bridge_payloads()
    assert bridges
    for bridge in bridges:
        bridge["quote_start"] = 0
        bridge["quote_end"] = 5
    redacted, _count = redact_text(RAW_SOURCE)
    claims = _review_with(bridges, redacted)
    assert all(claim["verdict"] != "supported" for claim in claims)


def test_a_fabricated_quote_still_fails_under_redaction():
    bridges = _bridge_payloads()
    assert bridges
    for bridge in bridges:
        bridge["evidence_quote"] = "Built a Kubernetes platform"
    redacted, _count = redact_text(RAW_SOURCE)
    claims = _review_with(bridges, redacted)
    assert all(claim["verdict"] != "supported" for claim in claims)


def test_no_redaction_needed_still_uses_the_plain_path():
    plain = "Compared weekly sales figures to identify trends.\n"
    bridges = _bridge_payloads(
        source_text=plain, quote="Compared weekly sales figures to identify trends"
    )
    claims = _review_with(bridges, plain)
    assert claims[-1]["verdict"] == "supported"
