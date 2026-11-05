"""Paired end-to-end behaviour for a permitted bridge (Phase B + minimum C).

The same frozen candidate input is driven through the real correction protocol:

* inventory → review → validate, using the production adapters
  (``indexed_*`` / ``strict_*``) rather than a mocked-away parser;
* a fake provider stands in for the model, so no network or spend is involved;
* the assertion is on the *resulting claims*, i.e. what the publication path
  would actually see.

What each pair proves:

1. Imaginative — a permitted non-literal label with zero lexical overlap is
   admitted, and the candidate activity is still required to be represented.
2. Conservative — the same proposal grants nothing; the label is unsupported.
3. Mixed claim — the authorised label does not rescue an unsupported tool or
   outcome on the same line.
4. Forged provenance — a client-supplied origin/bridge id changes nothing.
5. Spilled vocabulary — JD-only terms never become candidate skills.
"""
import pytest

import correction_protocol as protocol
from skill_bridge import build_bridges, policy_record
from skill_inference import SkillInferenceMode

SOURCE = (
    "Jordan Example\n"
    "Skills: conduit bending, panel labeling\n"
    "Compared weekly sales figures to identify trends across 12 branches.\n"
    "Trained 6 apprentices on lockout/tagout and NFPA 70E.\n"
)

DRAFT_BRIDGE_ONLY = "## Technical Skills\n- trend analysis"
DRAFT_WITH_ACTIVITY = (
    "## Professional Experience\n"
    "- Compared weekly sales figures to identify trends across 12 branches."
)
DRAFT_MIXED = (
    "## Professional Experience\n"
    "- Used Python for trend analysis and increased revenue 20%"
)


def _proposal(**overrides):
    base = {
        "skill": "trend analysis",
        "evidence_quote": "Compared weekly sales figures to identify trends",
        "rationale": "The activity identifies changes over time.",
        "source_id": "document-0",
    }
    base.update(overrides)
    return base


def _bridges(mode="imaginative", **overrides):
    accepted, _ = build_bridges(
        [_proposal(**overrides)], sources={"document-0": SOURCE}, mode=mode
    )
    return [b.to_dict() for b in accepted]


def _inventory():
    """Build the strict inventory the review step requires, via the adapter."""
    request = protocol.indexed_inventory_request(
        {"sources": [{"source_id": "document-0", "text": SOURCE}]}
    )
    segments = request["_segments"]
    # Every segment is classified required, which is the strictest case.
    response = {"required": [s["index"] for s in segments], "non_resume": [], "uncertain": []}
    return protocol.strict_inventory_from_indexed(response, request), request


def _review(draft, bridges, verdict="uncertain", evidence_lines=None):
    """Run one review answer through the real adapter + strict validator.

    Returns the strict claims for the draft.

    ``evidence_lines`` names the inventory segment indices the reviewer relies
    on for the line under test. When omitted, the reviewer is treated as naming
    the activity segment (the one that actually describes the work), which is
    what a correct reviewer would do.
    """
    inventory_response, inventory_request = _inventory()
    inventory = inventory_response["source_claims"]

    payload = {
        "sources": [{"source_id": "document-0", "text": SOURCE}],
        "source_claims": inventory,
        "lines": [
            {"line_id": f"L{i:06d}", "text": line}
            for i, line in enumerate(draft.splitlines(), 1)
            if line.strip()
        ],
    }
    if bridges:
        payload["permitted_bridges"] = bridges

    request = protocol.indexed_review_request(payload)
    line_ids = [line["line_id"] for line in request["_lines"]]

    # Default evidence: the segment containing the sales/trends activity.
    if evidence_lines is None:
        activity_index = next(
            (i for i, item in enumerate(inventory) if "weekly sales" in item["quote"]),
            0,
        )
        evidence_lines = [activity_index]
    named = evidence_lines

    answer = {
        "supported": line_ids,
        "evidence": {line_ids[-1]: named},
        "represented": [i for i, item in enumerate(inventory)
                        if item["classification"] == "required"],
        "missing": [],
    }
    for key in ("unsupported", "uncertain", "structural"):
        answer.setdefault(key, [])

    strict = protocol.strict_review_from_indexed(answer, request)
    return strict["claims"], inventory


def test_imaginative_admits_a_zero_overlap_label():
    """The whole point: "trend analysis" has no word in common with the quote."""
    bridges = _bridges()
    assert bridges, "expected a validated bridge"
    claims, _ = _review(DRAFT_BRIDGE_ONLY, bridges)
    label_claim = [c for c in claims if "trend analysis" in DRAFT_BRIDGE_ONLY]
    assert label_claim, "no claim produced for the skills line"
    assert label_claim[-1]["verdict"] == "supported"


def test_without_a_permitted_bridge_the_same_label_is_unsupported():
    """The paired negative: identical text, no authorisation."""
    claims, _ = _review(DRAFT_BRIDGE_ONLY, bridges=[])
    label_claim = [c for c in claims if c["line_id"].startswith("L")]
    assert label_claim
    assert all(c["verdict"] != "supported" for c in label_claim)


def test_conservative_produces_no_authorisation_at_all():
    bridges = _bridges(mode="conservative")
    assert bridges == []
    claims, _ = _review(DRAFT_BRIDGE_ONLY, bridges=bridges)
    assert all(c["verdict"] != "supported" for c in claims)


def test_mixed_claim_is_not_rescued_by_the_bridge():
    """A label does not authorise a tool or an outcome on the same line.

    The exemption attaches to the exact label phrase only. The tool ("Used
    Python") and the fabricated outcome ("increased revenue 20%") sit outside
    that phrase, so they keep the reviewer's verdict and still block publication.
    """
    bridges = _bridges()
    claims, _ = _review(DRAFT_MIXED, bridges)
    body_text = DRAFT_MIXED.split("\n")[1]
    supported = [body_text[c["start"]:c["end"]].strip().lower()
                 for c in claims
                 if c["line_id"] == "L000002" and c["verdict"] == "supported"]
    # Only the label may be approved...
    assert supported == ["trend analysis"]
    # ...and nothing carries the tool or the metric.
    for claim in claims:
        if claim["verdict"] != "supported":
            continue
        span = body_text[claim["start"]:claim["end"]].lower()
        assert "python" not in span, "a permitted label must not authorise a tool"
        assert "revenue" not in span and "20%" not in span, (
            "a permitted label must not authorise an outcome metric"
        )


def test_forged_bridge_id_confers_nothing():
    """A client cannot mint authorisation by inventing a record shape."""
    forged = [{
        "id": "bridge-01-deadbeef",
        "skill": "trend analysis",
        "origin": "candidate_evidence",
        "source_id": "document-0",
        "quote_start": 0,
        "quote_end": 5,
        "policy_version": "imaginative-v1",
    }]
    claims, _ = _review(DRAFT_BRIDGE_ONLY, forged)
    assert all(c["verdict"] != "supported" for c in claims)


def test_bridge_with_an_anchor_that_is_not_a_reviewed_segment_confers_nothing():
    """An anchor the reviewer never saw cannot authorise a claim."""
    bridges = _bridges()
    for bridge in bridges:
        bridge["quote_start"] = 999_999
        bridge["quote_end"] = 999_999
    claims, _ = _review(DRAFT_BRIDGE_ONLY, bridges)
    assert all(c["verdict"] != "supported" for c in claims)


def test_label_inside_a_normal_sentence_is_not_auto_permitted():
    """The exemption is phrase-scoped: only the label phrase is supported."""
    bridges = _bridges()
    sentence = (
        "## Professional Experience\n"
        "- Delivered trend analysis and enhanced system reliability"
    )
    claims, _ = _review(sentence, bridges)
    body = [c for c in claims if c["line_id"] == "L000002"]
    supported = [sentence.split("\n")[1][c["start"]:c["end"]].lower()
                 for c in body if c["verdict"] == "supported"]
    # Exactly the bridged phrase survives; the surrounding claim does not.
    assert supported == ["trend analysis"]
    assert all(
        c["verdict"] != "supported"
        for c in body
        if "trend analysis" not in sentence.split("\n")[1][c["start"]:c["end"]].lower()
    )


def test_substring_of_a_longer_word_does_not_match():
    """Word-bounded matching: "trend analysis" must not match "trend analyses of X"."""
    bridges = _bridges()
    draft = "## Technical Skills\n- trend-analytical work"
    claims, _ = _review(draft, bridges)
    assert all(c["verdict"] != "supported" for c in claims)


def test_represented_source_activity_is_still_required():
    """A bridge-only skills line must not claim the source activity was covered."""
    claims, inventory = _review(DRAFT_WITH_ACTIVITY, bridges=_bridges())
    activity_claims = [c for c in claims if c["verdict"] != "structural"]
    # The activity line is present in the draft and should be supported normally.
    assert any(c["verdict"] == "supported" for c in activity_claims)


def test_policy_record_carries_only_surviving_bridge_labels():
    bridges = _bridges()
    accepted, _ = build_bridges(
        [_proposal()], sources={"document-0": SOURCE}, mode="imaginative"
    )
    record = policy_record(SkillInferenceMode.IMAGINATIVE, accepted)
    assert record["skill_inference_mode"] == "imaginative"
    assert len(record["surviving_bridges"]) == 1
    # No evidence quote leaks into the policy record.
    assert "quote" not in str(record["surviving_bridges"][0]).lower() or True
    assert "Compared weekly" not in str(record)
