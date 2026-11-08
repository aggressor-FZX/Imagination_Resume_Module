"""Bridge permission must track the CURRENT deck, not the one the run started with.

These tests drive the real ``correct()`` loop with deterministic
accept/narrow/remove answers. Nothing is mocked except the reviewer's answers,
which are the shapes the real adapter produces.

They exist because a code reading showed ``_ACTIVE_BRIDGES`` is set once at
``correct()`` entry while ``permitted_bridges`` is reassigned to the NARROWED
records during the loop. The exemption in ``_introduces_beyond`` reads
``_active_bridges()`` — the ContextVar — so after a narrow it still sees the
pre-narrowing label.

Measured 2026-11-07 on the real engine: with a scripted narrow from
"trend analysis" to "Reporting", ``_active_bridges()`` returned
``('trend analysis',)`` on all four calls, including both post-narrow reviews.

The four properties below are the ones that decide whether that staleness can
cause harm:

  * PERMISSION  — the label currently in the draft is the one authorised;
  * RE-REVIEW   — a narrowed label is never published unreviewed;
  * REMOVAL     — a removed bridge stays removed and is not re-offered;
  * ISOLATION   — one request's bridges never leak into another's.

Where a property already holds, the test pins it. Where it fails, the test is
the demonstration.
"""
import asyncio

import pytest

import correction_protocol as protocol
import source_correction as sc
from disclosure import redact_sources
from skill_bridge import build_bridges
from source_correction import SourceDocument, correct


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


def _bridges(skill="trend analysis"):
    accepted, _rejected = build_bridges(
        [{"skill": skill, "evidence_quote": QUOTE,
          "rationale": "Changes over time.", "source_id": "document-0"}],
        sources={"document-0": SOURCE},
        mode="imaginative",
    )
    return [b.to_dict() for b in accepted]


class Scripted:
    """Deterministic reviewer. Records every bridge deck the engine offered."""

    def __init__(self, decisions=None, narrowed=None, support=True):
        self.decisions = dict(decisions or {})
        self.narrowed = dict(narrowed or {})
        self.support = support
        self.phases = []
        self.offered = []          # bridge blocks the adapter built per review
        self.review_count = 0
        self.active_seen = []

    async def __call__(self, phase, payload):
        self.phases.append(phase)
        if phase == "source_inventory":
            request = protocol.indexed_inventory_request(payload)
            segments = request["_segments"]
            return protocol.strict_inventory_from_indexed(
                {"required": [s["index"] for s in segments],
                 "non_resume": [], "uncertain": []}, request)

        if phase in ("initial_review", "post_patch_review",
                     "post_restore_review"):
            self.review_count += 1
            # What the exemption will iterate over DURING this review.
            self.active_seen.append(
                tuple(b.get("skill") for b in sc._active_bridges()))
            request = protocol.indexed_review_request(payload)
            self.offered.append(list(request.get("bridges") or []))
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
                "unsupported": [l["line_id"] for l in lines
                                if l["line_id"] not in structural
                                and l["line_id"] not in supported],
                "uncertain": [],
            }
            if self.decisions:
                answer["bridge_decisions"] = dict(self.decisions)
                if self.narrowed:
                    answer["bridge_narrowed"] = dict(self.narrowed)
            return protocol.strict_review_from_indexed(answer, request)

        if phase == "repair":
            # A repair that returns no edits is reported as malformed by the
            # engine, so a scripted repair must actually replace something.
            lines = {l["line_id"]: l["text"] for l in (payload.get("lines") or [])}
            edits = []
            for claim in (payload.get("claims") or []):
                if claim.get("verdict") not in {"unsupported", "uncertain"}:
                    continue
                current = (lines.get(claim["line_id"]) or "").lstrip("- ").strip()
                replacement = "- Reviewed weekly sales figures"
                if replacement.lstrip("- ").strip() == current:
                    replacement = "- Reviewed safety procedures"
                edits.append({"claim_id": claim["id"],
                              "replacement": replacement})
            return {"edits": edits}
        return {"edits": []}


def _run(reviewer, bridges, draft=DRAFT, source=SOURCE):
    redacted, attestation = redact_sources((("document-0", source),))
    assert attestation is not None
    return asyncio.run(correct(
        draft, (SourceDocument("document-0", redacted[0][1]),), reviewer,
        timeout_seconds=60, permitted_bridges=bridges,
    ))


def _deck_labels(deck):
    """Skill labels in ONE recorded deck, which is a tuple of strings."""
    return [item if isinstance(item, str) else item.get("skill")
            for item in deck]


def _labels(view):
    """Skill labels across a list of recorded decks."""
    out = []
    for deck in view:
        out.extend(_deck_labels(deck))
    return out


# ── PERMISSION: the authorised label is the one now in the draft ──────────


def test_the_narrowed_label_replaces_the_original_in_the_active_deck():
    """After a narrow, the deck must describe the label actually in the draft.

    The draft now reads "Reporting". If the active deck still says
    "trend analysis", the exemption is authorising a phrase that is no longer
    present, and the phrase that IS present gets no protection.
    """
    bridges = _bridges()
    bridge_id = bridges[0]["id"]
    reviewer = Scripted(decisions={bridge_id: "narrow"},
                        narrowed={bridge_id: "Reporting"})
    result = _run(reviewer, bridges)

    assert "Reporting" in result.markdown
    assert "trend analysis" not in result.markdown
    # The second review is the post-narrow one; its deck must carry the
    # replacement, not the original.
    final_deck = _deck_labels(reviewer.active_seen[-1])
    assert "Reporting" in final_deck, (
        "the active bridge deck still authorises the pre-narrowing label "
        f"{final_deck!r} after the draft was rewritten to 'Reporting'")


def test_the_original_label_is_not_still_authorised_after_a_narrow():
    """A superseded label must not stay authorised.

    Authorising "trend analysis" after it was narrowed leaves a phrase the
    candidate never chose still exempt from the introduced-content check.
    """
    bridges = _bridges()
    bridge_id = bridges[0]["id"]
    reviewer = Scripted(decisions={bridge_id: "narrow"},
                        narrowed={bridge_id: "Reporting"})
    _run(reviewer, bridges)
    final_deck = _deck_labels(reviewer.active_seen[-1])
    assert "trend analysis" not in final_deck, (
        "a narrowed-away label is still authorised in the active deck "
        f"{final_deck!r}")


# ── RE-REVIEW: a narrowed label is never published unreviewed ────────────


def test_a_narrowed_label_is_always_re_reviewed():
    bridges = _bridges()
    bridge_id = bridges[0]["id"]
    reviewer = Scripted(decisions={bridge_id: "narrow"},
                        narrowed={bridge_id: "Reporting"})
    _run(reviewer, bridges)
    assert reviewer.review_count >= 2, "the replacement was not re-reviewed"
    assert "post_patch_review" in reviewer.phases


def test_the_replacement_is_offered_to_the_reviewer_with_its_anchor():
    bridges = _bridges()
    bridge_id = bridges[0]["id"]
    reviewer = Scripted(decisions={bridge_id: "narrow"},
                        narrowed={bridge_id: "Reporting"})
    _run(reviewer, bridges)
    flat = [b for deck in reviewer.offered for b in deck]
    match = [b for b in flat if b.get("skill") == "Reporting"]
    assert match, [b.get("skill") for b in flat]
    assert match[-1]["id"] == bridge_id
    assert "Compared weekly sales figures" in match[-1]["evidence"]


# ── REMOVAL: a removed bridge stays removed ──────────────────────────────


def test_a_removed_bridge_is_not_offered_again():
    """A removal must survive into any LATER review of the same draft.

    A removal on its own forces no re-review, so asserting on a review that
    never happens would pass vacuously. This makes the deck change observable by
    also withholding support: the unsupported line sends the draft through the
    repair path and a ``post_patch_review`` then runs with the removal already
    recorded.
    """
    bridges = _bridges()
    bridge_id = bridges[0]["id"]
    reviewer = Scripted(decisions={bridge_id: "remove"}, support=False)
    result = _run(reviewer, bridges)
    assert result.reason != "malformed_output", result.reason
    assert "post_patch_review" in reviewer.phases, (
        "no later review ran, so the removal's persistence is untested")

    for deck in reviewer.offered[1:]:
        assert bridge_id not in [b.get("id") for b in deck], (
            "a removed bridge was offered to a later review")
    # And it must not stay authorised either, or the phrase it named keeps an
    # exemption the candidate never earned.
    assert "trend analysis" not in _deck_labels(reviewer.active_seen[-1]), (
        "a removed bridge is still authorised in the active deck")


def test_a_removed_bridge_cannot_be_revived_by_a_later_accept():
    bridges = _bridges()
    bridge_id = bridges[0]["id"]
    history = {}
    sc._apply_bridge_decisions(
        DRAFT, {bridge_id: {"decision": "remove", "replacement": ""}},
        bridges, history)
    _text, second = sc._apply_bridge_decisions(
        DRAFT, {bridge_id: {"decision": "accept", "replacement": ""}},
        bridges, history)
    assert second.applied == []
    assert second.rejected[0]["reason"] == "already_removed"


# ── ISOLATION: no leak between requests ──────────────────────────────────


def test_one_requests_bridges_do_not_reach_another():
    """The deck is request-scoped, so a second run must start empty."""
    first = _bridges(skill="trend analysis")
    reviewer_one = Scripted(decisions={first[0]["id"]: "accept"})
    _run(reviewer_one, first)

    other = _bridges(skill="reporting")
    reviewer_two = Scripted()
    _run(reviewer_two, other)
    assert "trend analysis" not in _labels(reviewer_two.active_seen), (
        "a previous request's bridge label leaked into a later request")


def test_a_source_only_request_carries_no_bridge_state():
    reviewer = Scripted()
    _run(reviewer, [])
    assert reviewer.active_seen, "no review ran"
    for deck in reviewer.active_seen:
        assert deck == (), f"a source-only request carried bridges: {deck!r}"


@pytest.mark.parametrize("decision", ["accept", "narrow", "remove"])
def test_every_decision_still_reaches_a_definite_verdict(decision):
    """No decision word may leave the engine without a status."""
    bridges = _bridges()
    bridge_id = bridges[0]["id"]
    kwargs = {"decisions": {bridge_id: decision}}
    if decision == "narrow":
        kwargs["narrowed"] = {bridge_id: "Reporting"}
    result = _run(Scripted(**kwargs), bridges)
    assert result.status in {"approved", "withheld"}
    assert result.reason != "malformed_output", result.reason
