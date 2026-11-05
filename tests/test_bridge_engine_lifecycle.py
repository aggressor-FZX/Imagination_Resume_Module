"""Phase C.7/C.8 through the REAL correction engine.

The helper suites prove the decision records. This one drives the production path:
`correct()` -> engine loop -> real adapter translation -> real strict validator ->
real repair/re-review. Nothing is mocked except the provider *answers*, which are
the same shapes the real adapter produces.

The scripted reviewer is deliberately boring: every heading is structural, every
other line supported, unless a test says otherwise. A failure here therefore means
the engine did something, not that the script was clever.
"""
import asyncio
import json

import correction_protocol as protocol
from disclosure import redact_sources
from skill_bridge import build_bridges
from source_correction import SourceDocument, correct


PII = "jordan.private@example.com"
SOURCE = (
    "Compared weekly sales figures to identify trends across 12 branches.\n"
    "Trained 6 apprentices on lockout/tagout and NFPA 70E.\n"
)
QUOTE = "Compared weekly sales figures to identify trends"
DRAFT = (
    "## Technical Skills\n"
    "- trend analysis\n"
    "## Professional Experience\n"
    "- Compared weekly sales figures to identify trends across 12 branches.\n"
    "- Trained 6 apprentices on lockout/tagout and NFPA 70E.\n"
)


def _bridges(source=SOURCE, quote=QUOTE, skill="trend analysis", rationale=None):
    accepted, _rejected = build_bridges(
        [{"skill": skill, "evidence_quote": quote,
          "rationale": rationale or "Changes over time.",
          "source_id": "document-0"}],
        sources={"document-0": source},
        mode="imaginative",
    )
    return [bridge.to_dict() for bridge in accepted]


class Reviewer:
    """Scripted provider. Records what the engine sent; answers predictably."""

    def __init__(self, *, decisions=None, narrowed=None, support=True,
                 offer_decisions=True):
        self.decisions = dict(decisions or {})
        self.narrowed = dict(narrowed or {})
        self.support = support
        self.offer_decisions = offer_decisions
        self.phases = []
        self.reviews = []       # raw engine payloads for review phases
        self.offered = []       # visible bridge blocks built by the adapter
        self.review_count = 0

    async def __call__(self, phase, payload):
        self.phases.append(phase)
        if phase == "source_inventory":
            request = protocol.indexed_inventory_request(payload)
            segments = request["_segments"]
            return protocol.strict_inventory_from_indexed(
                {"required": [s["index"] for s in segments],
                 "non_resume": [], "uncertain": []},
                request,
            )
        if phase in ("initial_review", "post_patch_review", "post_restore_review"):
            self.review_count += 1
            self.reviews.append(payload)
            request = protocol.indexed_review_request(payload)
            self.offered.extend(request.get("bridges") or [])
            lines = request["_lines"]
            structural = [l["line_id"] for l in lines
                          if l["text"].strip().startswith("#")]
            supported = ([] if not self.support
                         else [l["line_id"] for l in lines
                               if l["line_id"] not in structural])
            answer = {
                "supported": supported,
                "structural": structural,
                "evidence": {lid: [0, 1] for lid in supported},
                "represented": [0, 1], "missing": [],
                "unsupported": [], "uncertain": [],
            }
            if self.offer_decisions and self.decisions:
                answer["bridge_decisions"] = dict(self.decisions)
                if self.narrowed:
                    answer["bridge_narrowed"] = dict(self.narrowed)
            # A line omitted from every list is "unjudged" (a distinct engine
            # outcome). Listing it as unsupported keeps the repair path reachable,
            # which is what these tests are about.
            answer["unsupported"] = [
                l["line_id"] for l in lines
                if l["line_id"] not in structural and l["line_id"] not in supported
            ]
            return protocol.strict_review_from_indexed(answer, request)
        if phase == "repair":
            claims = payload.get("claims") or []
            lines = {l["line_id"]: l["text"] for l in (payload.get("lines") or [])}
            edits = []
            for claim in claims:
                if claim.get("verdict") not in {"unsupported", "uncertain"}:
                    continue
                current = lines.get(claim["line_id"], "")
                # Never echo the current text: a no-op repair is reported as such
                # and cannot approve anything.
                replacement = "- Compared weekly sales figures"
                if replacement.lstrip("- ").strip() == current.lstrip("- ").strip():
                    replacement = "- Trained apprentices on safety procedures"
                edits.append({"claim_id": claim["id"], "replacement": replacement})
            return {"edits": edits}
        return {"edits": []}


def _run(reviewer, bridges, draft=DRAFT, source=SOURCE):
    """Drive the engine the way the route does: redaction happens FIRST."""
    redacted, attestation = redact_sources((("document-0", source),))
    assert attestation is not None
    return asyncio.run(correct(
        draft,
        (SourceDocument("document-0", redacted[0][1]),),
        reviewer,
        timeout_seconds=60,
        permitted_bridges=bridges,
    ))


def _wire_payloads(reviewer):
    """Exactly what the provider would send: underscore keys are private."""
    out = []
    for payload in reviewer.reviews:
        visible = {k: v for k, v in payload.items() if not str(k).startswith("_")}
        out.append(json.dumps(visible, default=str))
    return out


# ── baseline ───────────────────────────────────────────────────────────────


def test_source_only_request_asks_no_bridge_question():
    reviewer = Reviewer()
    result = _run(reviewer, [])
    assert result.status == "approved", (result.status, result.reason)
    for payload in reviewer.reviews:
        assert "bridges" not in payload
    assert reviewer.offered == []


# ── accept ─────────────────────────────────────────────────────────────────


def test_accept_reaches_approval():
    bridges = _bridges()
    assert bridges
    reviewer = Reviewer(decisions={bridges[0]["id"]: "accept"})
    result = _run(reviewer, bridges)

    assert result.status == "approved", (result.status, result.reason)
    assert "- trend analysis" in result.markdown
    assert reviewer.offered, "the reviewer was never offered the bridge"


def test_accept_offers_the_label_with_its_evidence():
    bridges = _bridges()
    reviewer = Reviewer(decisions={bridges[0]["id"]: "accept"})
    _run(reviewer, bridges)
    assert reviewer.offered[0]["skill"] == "trend analysis"
    assert "Compared weekly sales figures" in reviewer.offered[0]["evidence"]


# ── narrow ─────────────────────────────────────────────────────────────────


def test_narrow_rewrites_the_label_and_re_reviews():
    bridges = _bridges()
    bridge_id = bridges[0]["id"]
    reviewer = Reviewer(decisions={bridge_id: "narrow"},
                        narrowed={bridge_id: "data analysis"})
    result = _run(reviewer, bridges)

    assert reviewer.review_count >= 2, "the replacement was not re-reviewed"
    assert "post_patch_review" in reviewer.phases
    assert result.status == "approved", (result.status, result.reason)
    assert "data analysis" in result.markdown


def test_narrowed_wording_is_offered_for_review_with_its_original_anchor():
    """The replacement must be judged against the SAME candidate evidence."""
    bridges = _bridges()
    bridge_id = bridges[0]["id"]
    reviewer = Reviewer(decisions={bridge_id: "narrow"},
                        narrowed={bridge_id: "data analysis"})
    _run(reviewer, bridges)

    narrowed_offers = [e for e in reviewer.offered if e["skill"] == "data analysis"]
    assert narrowed_offers, [e["skill"] for e in reviewer.offered]
    assert narrowed_offers[0]["id"] == bridge_id
    assert "Compared weekly sales figures" in narrowed_offers[0]["evidence"]


def test_shape_invalid_narrow_never_reaches_the_draft():
    bridges = _bridges()
    bridge_id = bridges[0]["id"]
    reviewer = Reviewer(decisions={bridge_id: "narrow"},
                        narrowed={bridge_id: "expert trend analysis"})
    result = _run(reviewer, bridges)
    assert "expert trend analysis" not in (result.markdown or "")


# ── remove ─────────────────────────────────────────────────────────────────


def test_remove_makes_a_named_decision():
    bridges = _bridges()
    reviewer = Reviewer(decisions={bridges[0]["id"]: "remove"})
    result = _run(reviewer, bridges)
    assert result.status in {"approved", "withheld"}
    assert result.reason != "malformed_output", result.reason


def test_removed_bridge_is_not_offered_again():
    from source_correction import _apply_bridge_decisions

    bridges = _bridges()
    bridge_id = bridges[0]["id"]
    history = {}
    _text, outcome = _apply_bridge_decisions(
        DRAFT, {bridge_id: {"decision": "remove", "replacement": ""}},
        bridges, history,
    )
    offered_ids = [b["id"] for b in (outcome.bridges or [])]
    assert bridge_id not in offered_ids, "a removed bridge was offered again"


def test_removed_bridge_cannot_be_revived():
    from source_correction import _apply_bridge_decisions

    bridges = _bridges()
    bridge_id = bridges[0]["id"]
    history = {}
    _apply_bridge_decisions(
        DRAFT, {bridge_id: {"decision": "remove", "replacement": ""}},
        bridges, history,
    )
    _text, second = _apply_bridge_decisions(
        DRAFT, {bridge_id: {"decision": "accept", "replacement": ""}},
        bridges, history,
    )
    assert second.applied == []
    assert second.rejected[0]["reason"] == "already_removed"


# ── unanswered / malformed ─────────────────────────────────────────────────


def test_no_decision_changes_nothing():
    bridges = _bridges()
    reviewer = Reviewer(offer_decisions=False)
    result = _run(reviewer, bridges)
    assert result.status == "approved", (result.status, result.reason)


def test_unknown_decision_word_is_malformed_not_approved():
    bridges = _bridges()
    reviewer = Reviewer(decisions={bridges[0]["id"]: "approve"})
    assert _run(reviewer, bridges).status == "withheld"


def test_decision_for_an_unoffered_bridge_is_malformed():
    bridges = _bridges()
    reviewer = Reviewer(decisions={"bridge-99-invented": "accept"})
    assert _run(reviewer, bridges).status == "withheld"


# ── the earlier uncertain->supported promotion must stay fixed ────────────


def test_unsupported_label_is_not_promoted_by_a_bridge_decision():
    bridges = _bridges()
    reviewer = Reviewer(decisions={bridges[0]["id"]: "accept"}, support=False)
    result = _run(reviewer, bridges)
    assert result.status == "withheld", (
        "an unsupported label was approved because a bridge decision was supplied"
    )


# ── disclosure ─────────────────────────────────────────────────────────────


def test_pii_outside_the_quote_never_reaches_the_wire():
    source = PII + "\n" + SOURCE
    bridges = _bridges(source=source)
    reviewer = Reviewer(decisions={bridges[0]["id"]: "accept"})
    _run(reviewer, bridges, source=source)
    for body in _wire_payloads(reviewer):
        assert PII not in body


def test_pii_inside_the_quote_never_reaches_the_wire():
    source = f"Compared sales for {PII} and identified trends.\n"
    quote = f"Compared sales for {PII} and identified trends"
    bridges = _bridges(source=source, quote=quote)
    assert bridges
    reviewer = Reviewer(decisions={bridges[0]["id"]: "accept"})
    _run(reviewer, bridges, source=source)
    for body in _wire_payloads(reviewer):
        assert PII not in body
    for entry in reviewer.offered:
        assert PII not in json.dumps(entry)


def test_pii_inside_the_rationale_never_reaches_the_wire():
    bridges = _bridges(rationale=f"Contact {PII} for detail.")
    assert bridges
    reviewer = Reviewer(decisions={bridges[0]["id"]: "accept"})
    _run(reviewer, bridges)
    for body in _wire_payloads(reviewer):
        assert PII not in body, "the rationale leaked PII to the provider"


def test_offered_quote_is_the_redacted_slice():
    source = f"Compared sales for {PII} and identified trends.\n"
    quote = f"Compared sales for {PII} and identified trends"
    bridges = _bridges(source=source, quote=quote)
    reviewer = Reviewer(decisions={bridges[0]["id"]: "accept"})
    _run(reviewer, bridges, source=source)
    assert reviewer.offered, "no bridge was offered"
    evidence = reviewer.offered[0]["evidence"]
    assert PII not in evidence
    assert "email removed" in evidence or "\u2588" in evidence


def test_anchor_checking_is_not_weakened_by_redaction_support():
    """A forged anchor still fails, even though redaction is tolerated."""
    source = f"Compared sales for {PII} and identified trends.\n"
    quote = f"Compared sales for {PII} and identified trends"
    bridges = _bridges(source=source, quote=quote)
    for bridge in bridges:
        bridge["quote_start"] = 0
        bridge["quote_end"] = 5
    reviewer = Reviewer(decisions={bridges[0]["id"]: "accept"})
    _run(reviewer, bridges, source=source)
    assert reviewer.offered == [], "a forged anchor was still offered"
