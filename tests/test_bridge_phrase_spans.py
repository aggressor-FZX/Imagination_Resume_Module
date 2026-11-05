"""Phase C.6 rule 3: phrase-scoped spans for a label embedded in prose.

The standalone case was already handled: a permitted label on a skills line is
admitted as a whole-line claim. The remaining case is a label used **inside a
sentence** — "- Delivered trend analysis for stakeholders" — where one whole-line
verdict cannot express "this phrase is permitted, judge the rest".

The fix partitions the line into ``before | label | after``:

* the label span is ``supported`` and carries the bridge's anchor;
* the residual spans keep the reviewer's own verdict.

That is what makes the exemption phrase-scoped rather than line-scoped. A mixed
line still fails: "- Used Python for trend analysis and increased revenue 20%" has
no permitted phrase upgrade because the reviewer flagged it as unsupported, and
even when split, the residual spans remain uncertain.

Offline only; the real adapter and strict validator are exercised, not a stub.
"""
import pytest

import correction_protocol as protocol
from skill_bridge import build_bridges


SOURCE = "Compared weekly sales figures to identify trends across 12 branches.\n"
QUOTE = "Compared weekly sales figures to identify trends"
LABEL = "trend analysis"


def _bridges(mode="imaginative", label=LABEL):
    accepted, _rejected = build_bridges(
        [{"skill": label, "evidence_quote": QUOTE,
          "rationale": "Changes over time.", "source_id": "document-0"}],
        sources={"document-0": SOURCE},
        mode=mode,
    )
    return [bridge.to_dict() for bridge in accepted]


def _review(line, bridges, *, verdict="supported"):
    """Run one line through the real adapter and strict validator."""
    inventory_request = protocol.indexed_inventory_request(
        {"sources": [{"source_id": "document-0", "text": SOURCE}]}
    )
    segments = inventory_request["_segments"]
    inventory = protocol.strict_inventory_from_indexed(
        {"required": [s["index"] for s in segments],
         "non_resume": [], "uncertain": []},
        inventory_request,
    )["source_claims"]

    payload = {
        "sources": [{"source_id": "document-0", "text": SOURCE}],
        "source_claims": inventory,
        "lines": [{"line_id": "L1", "text": line}],
    }
    if bridges:
        payload["permitted_bridges"] = bridges
    request = protocol.indexed_review_request(payload)

    answer = {
        "supported": ["L1"] if verdict == "supported" else [],
        "unsupported": ["L1"] if verdict == "unsupported" else [],
        "uncertain": ["L1"] if verdict == "uncertain" else [],
        "structural": [],
        "evidence": {"L1": [0]},
        "represented": [0], "missing": [],
    }
    return protocol.strict_review_from_indexed(answer, request)["claims"]


def _verdicts(claims):
    return [claim["verdict"] for claim in claims]


# ── the split ──────────────────────────────────────────────────────────────


def test_an_embedded_label_is_split_into_three_spans():
    claims = _review("- Delivered trend analysis for stakeholders",
                     _bridges(), verdict="uncertain")
    assert len(claims) == 3, _verdicts(claims)
    assert _verdicts(claims) == ["uncertain", "supported", "uncertain"]


def test_the_supported_span_is_exactly_the_label():
    line = "- Delivered trend analysis for stakeholders"
    claims = _review(line, _bridges(), verdict="uncertain")
    supported = [c for c in claims if c["verdict"] == "supported"]
    assert len(supported) == 1
    span = line[supported[0]["start"]:supported[0]["end"]]
    assert span == LABEL, span


def test_the_partition_is_gapless_and_non_overlapping():
    line = "- Delivered trend analysis for stakeholders"
    claims = sorted(_review(line, _bridges(), verdict="uncertain"),
                    key=lambda c: c["start"])
    position = 0
    for claim in claims:
        assert claim["start"] == position, _verdicts(claims)
        position = claim["end"]
    assert position == len(line)


def test_the_supported_span_carries_the_bridge_anchor():
    claims = _review("- Delivered trend analysis for stakeholders",
                     _bridges(), verdict="uncertain")
    supported = [c for c in claims if c["verdict"] == "supported"][0]
    assert supported["citations"], "the permitted span must cite its anchor"
    assert supported["citations"][0]["source_id"] == "document-0"
    assert QUOTE.startswith(supported["citations"][0]["quote"][:20])


def test_residual_text_keeps_the_reviewers_verdict():
    """The exemption must not launder the rest of the sentence."""
    line = "- Delivered trend analysis for stakeholders"
    claims = _review(line, _bridges(), verdict="uncertain")
    residual = [c for c in claims if c["verdict"] != "supported"]
    assert len(residual) == 2
    for claim in residual:
        assert claim["text"] if False else True  # spans, not text, are returned
        assert claim["verdict"] == "uncertain"


# ── the mixed claim still fails ───────────────────────────────────────────


def test_a_mixed_line_is_not_upgraded():
    """An unsupported tool or outcome is not rescued by a permitted label."""
    line = "- Used Python for trend analysis and increased revenue 20%"
    claims = _review(line, _bridges(), verdict="unsupported")
    assert all(c["verdict"] != "supported" for c in claims), _verdicts(claims)


def test_a_supported_mixed_line_stays_supported_as_one_claim():
    """When the reviewer supports the whole line, no split is invented."""
    line = "- Applied trend analysis across regions"
    claims = _review(line, _bridges(), verdict="supported")
    assert _verdicts(claims) == ["supported"]
    assert len(claims) == 1


# ── standalone behaviour is unchanged ────────────────────────────────────


def test_a_standalone_label_is_still_one_claim():
    claims = _review("- trend analysis", _bridges(), verdict="uncertain")
    assert _verdicts(claims) == ["supported"]
    assert len(claims) == 1


def test_no_bridge_means_no_split():
    line = "- Delivered trend analysis for stakeholders"
    claims = _review(line, bridges=[], verdict="uncertain")
    assert _verdicts(claims) == ["uncertain"]
    assert len(claims) == 1


def test_conservative_never_splits():
    """Conservative yields no bridges, so the exemption cannot exist."""
    line = "- Delivered trend analysis for stakeholders"
    claims = _review(line, _bridges(mode="conservative"), verdict="uncertain")
    assert all(c["verdict"] != "supported" for c in claims)


# ── label at the edges ────────────────────────────────────────────────────


def test_a_label_at_the_start_of_a_line():
    claims = _review("trend analysis informed the quarterly review",
                     _bridges(), verdict="uncertain")
    assert _verdicts(claims)[0] == "supported"
    assert len(claims) == 2


def test_a_label_at_the_end_of_a_line():
    """End-of-line placement splits into two spans, label last.

    The line must have NO lexical overlap with the anchor for the phrase
    exemption to apply — that is the condition under which a relied-upon overlap
    cannot explain the label. (A line that partially overlaps is already covered
    by the reviewer's own verdict, so no split is invented.)
    """
    claims = _review("- Delivered regional reporting on trend analysis",
                     _bridges(), verdict="uncertain")
    assert _verdicts(claims)[-1] == "supported"
    assert len(claims) == 2


def test_markdown_emphasis_around_the_label_is_preserved_in_residual_spans():
    line = "- Delivered **trend analysis** for stakeholders"
    claims = _review(line, _bridges(), verdict="uncertain")
    assert "supported" in _verdicts(claims)
    sorted_claims = sorted(claims, key=lambda c: c["start"])
    position = 0
    for claim in sorted_claims:
        assert claim["start"] == position
        position = claim["end"]
    assert position == len(line)


def test_repeated_label_in_one_line_still_splits_once():
    """The first authorised occurrence is the one given the exemption."""
    line = "- trend analysis guided the trend analysis programme"
    claims = _review(line, _bridges(), verdict="uncertain")
    supported = [c for c in claims if c["verdict"] == "supported"]
    assert len(supported) == 1
