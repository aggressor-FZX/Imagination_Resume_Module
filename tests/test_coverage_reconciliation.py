"""A reviewer that contradicts itself on unchanged text is reconciled, bounded.

Measured on Imaginative website run ``61bce5ef`` (2026-11-07, revision
``bc9e6df``). Five employer/institution head lines were confirmed in pass 1 and
reported missing in pass 2, while the line text was byte-identical:

  pass 1  L000007  '**Harborline Transit Authority** -- Tacoma, WA'
          claim verdict 'supported', 1 citation
          representation 'represented', span L000007:0:46
  pass 2  L000007  '**Harborline Transit Authority** -- Tacoma, WA'   (identical)
          claim verdict 'structural', 0 citations
          representation 'missing', no spans

The repair in between changed only a different line (it stripped a bullet
marker), so nothing about these five lines justified the reversal. The run
withheld as ``source_facts_missing`` with every fact present.

The reconciliation is deliberately narrow. It fires only when the prior pass
recorded the item ``represented`` with a span, that span's line is byte-identical
now, the item's quote is verbatim in that line, and the quote is verbatim in the
source document it came from. Every other shape keeps the refusal.
"""
import pytest

import source_correction as sc

SOURCE = ("Professional Experience\n"
          "SYNTHETIC-ONLY Harborline Transit Authority -- Tacoma, WA\n"
          "SYNTHETIC-ONLY Fuel Reporting Analyst | 04/2019 - Present\n"
          "SYNTHETIC-ONLY Reconciled 1,180 fuel invoices each quarter.\n")

HEAD_QUOTE = "SYNTHETIC-ONLY Harborline Transit Authority -- Tacoma, WA\n"
HEAD_START = SOURCE.index(HEAD_QUOTE)
HEAD_END = HEAD_START + len(HEAD_QUOTE)

BULLET_QUOTE = "SYNTHETIC-ONLY Reconciled 1,180 fuel invoices each quarter.\n"
BULLET_START = SOURCE.index(BULLET_QUOTE)
BULLET_END = BULLET_START + len(BULLET_QUOTE)

LINES = [
    {"line_id": "L1", "text": "## Professional Experience"},
    {"line_id": "L2", "text": "**Harborline Transit Authority** -- Tacoma, WA"},
    {"line_id": "L3", "text": "*Fuel Reporting Analyst | 04/2019 - Present*"},
    {"line_id": "L4", "text": "- Reconciled 1,180 fuel invoices each quarter."},
]

INVENTORY = [
    {"id": f"doc:{HEAD_START}:{HEAD_END}", "source_id": "doc", "start": HEAD_START,
     "end": HEAD_END, "quote": HEAD_QUOTE, "classification": "required"},
    {"id": f"doc:{BULLET_START}:{BULLET_END}", "source_id": "doc",
     "start": BULLET_START, "end": BULLET_END, "quote": BULLET_QUOTE,
     "classification": "required"},
]

HEAD_ID = INVENTORY[0]["id"]
BULLET_ID = INVENTORY[1]["id"]


def _review(head_status, head_verdict, head_citations, bullet_status="represented"):
    """A review answer with the head line in the requested state."""
    claims = [
        {"id": "L1:0:26", "line_id": "L1", "start": 0, "end": 26,
         "verdict": "structural", "citations": []},
        {"id": "L2:0:46", "line_id": "L2", "start": 0, "end": 46,
         "verdict": head_verdict, "citations": head_citations},
        {"id": "L3:0:44", "line_id": "L3", "start": 0, "end": 44,
         "verdict": "structural", "citations": []},
        {"id": "L4:0:46", "line_id": "L4", "start": 0, "end": 46,
         "verdict": "supported",
         "citations": [{"source_id": "doc", "start": BULLET_START,
                        "end": BULLET_END, "quote": BULLET_QUOTE}]},
    ]
    representations = [
        {"source_claim_id": HEAD_ID, "status": head_status,
         "spans": ([{"line_id": "L2", "start": 0, "end": 46}]
                   if head_status == "represented" else [])},
        {"source_claim_id": BULLET_ID, "status": bullet_status,
         "spans": ([{"line_id": "L4", "start": 0, "end": 46}]
                   if bullet_status == "represented" else [])},
    ]
    return {"claims": claims, "representations": representations}


HEAD_CITATION = [{"source_id": "doc", "start": HEAD_START, "end": HEAD_END,
                  "quote": HEAD_QUOTE}]


def _validate(review, prior, lines=None):
    token = sc._ACTIVE_PRIOR_COVERAGE.set(prior)
    try:
        return sc._validate_review(review, lines or LINES, {"doc": SOURCE},
                                   INVENTORY)
    finally:
        sc._ACTIVE_PRIOR_COVERAGE.reset(token)


def _prior_represented(line_text=None):
    return {
        HEAD_ID: {"status": "represented",
                  "spans": [{"line_id": "L2", "start": 0, "end": 46}],
                  "line_text": line_text or {line["line_id"]: line["text"]
                                             for line in LINES}},
        BULLET_ID: {"status": "represented",
                    "spans": [{"line_id": "L4", "start": 0, "end": 46}],
                    "line_text": {line["line_id"]: line["text"] for line in LINES}},
    }


# --- The captured contradiction -------------------------------------------

def test_the_captured_contradiction_is_reconciled():
    """Pass 1 confirmed the head; pass 2 calls it missing on identical text."""
    review = _review("missing", "structural", [])
    claims, incomplete, missing = _validate(review, _prior_represented())
    assert "missing" not in incomplete
    assert missing == []
    head = next(c for c in claims if c["line_id"] == "L2")
    assert head["verdict"] == "supported"
    assert head["citations"][0]["quote"] == HEAD_QUOTE


def test_the_reconciled_span_is_the_current_draft_line():
    review = _review("missing", "structural", [])
    _validate(review, _prior_represented())
    mapping = next(m for m in review["representations"]
                   if m["source_claim_id"] == HEAD_ID)
    assert mapping["status"] == "represented"
    assert mapping["spans"] == [{"line_id": "L2", "start": 0, "end": 46}]


# --- Refusals: every other shape keeps the withhold ------------------------

def test_a_changed_line_keeps_the_refusal():
    """The repair changed this line, so the prior verdict does not transfer."""
    changed = [dict(line) for line in LINES]
    changed[1]["text"] = "**Harborline Transit Authority** -- Seattle, WA"
    review = _review("missing", "structural", [])
    # The claim must still cover the new line exactly, or the envelope is
    # malformed for an unrelated reason and the test would pass for the wrong one.
    for claim in review["claims"]:
        if claim["line_id"] == "L2":
            claim["end"] = len(changed[1]["text"])
            claim["id"] = f"L2:0:{claim['end']}"
    claims, incomplete, missing = _validate(review, _prior_represented(), changed)
    assert "missing" in incomplete
    assert missing


def test_a_paraphrased_line_keeps_the_refusal():
    """The line no longer carries the item verbatim."""
    changed = [dict(line) for line in LINES]
    changed[1]["text"] = "**Harborline Transit Authority** in Tacoma, WA"
    review = _review("missing", "structural", [])
    claims, incomplete, _ = _validate(review, _prior_represented(), changed)
    assert "missing" in incomplete


def test_no_prior_evidence_keeps_the_refusal():
    """A first-pass ``missing`` is never reconciled."""
    review = _review("missing", "structural", [])
    claims, incomplete, missing = _validate(review, {})
    assert "missing" in incomplete
    assert missing


def test_a_prior_missing_verdict_keeps_the_refusal():
    """Only a prior ``represented`` verdict can be contradicted."""
    prior = _prior_represented()
    prior[HEAD_ID]["status"] = "missing"
    review = _review("missing", "structural", [])
    claims, incomplete, _ = _validate(review, prior)
    assert "missing" in incomplete


def test_an_item_absent_from_the_source_keeps_the_refusal():
    """The quote must be verbatim in the source document it came from.

    The inventory is rebuilt against a document that does not contain the head
    line, so the item is genuinely unsupported and the refusal must stand.
    """
    other = "Unrelated source text.\n"
    inventory = [{"id": f"doc:0:{len(other)}", "source_id": "doc", "start": 0,
                  "end": len(other), "quote": other, "classification": "required"}]
    review = _review("missing", "structural", [])
    # The bullet claim cites the real document, which is not the one under test,
    # so it is replaced by a structural claim that needs no citation.
    for claim in review["claims"]:
        if claim["line_id"] == "L4":
            claim["verdict"] = "structural"
            claim["citations"] = []
    review["representations"] = [{"source_claim_id": inventory[0]["id"],
                                  "status": "missing", "spans": []}]
    token = sc._ACTIVE_PRIOR_COVERAGE.set(
        {inventory[0]["id"]: {"status": "represented",
                              "spans": [{"line_id": "L2", "start": 0, "end": 46}],
                              "line_text": {l["line_id"]: l["text"] for l in LINES}}})
    try:
        claims, incomplete, _ = sc._validate_review(
            review, LINES, {"doc": other}, inventory)
    finally:
        sc._ACTIVE_PRIOR_COVERAGE.reset(token)
    assert "missing" in incomplete


def test_a_genuinely_absent_item_keeps_the_refusal():
    """The draft does not carry the item at all."""
    lines = [line for line in LINES if line["line_id"] != "L2"]
    review = _review("missing", "structural", [])
    review["claims"] = [c for c in review["claims"] if c["line_id"] != "L2"]
    claims, incomplete, _ = _validate(review, _prior_represented(), lines)
    assert "missing" in incomplete


def test_a_prior_span_on_a_different_line_keeps_the_refusal():
    """The prior evidence must point at the line that carries the item now."""
    prior = _prior_represented()
    prior[HEAD_ID]["spans"] = [{"line_id": "L4", "start": 0, "end": 46}]
    review = _review("missing", "structural", [])
    claims, incomplete, _ = _validate(review, prior)
    assert "missing" in incomplete


def test_an_uncertain_verdict_is_not_reconciled():
    """Only ``missing`` is reconciled; ``uncertain`` has its own narrow path."""
    review = _review("uncertain", "uncertain", [])
    claims, incomplete, _ = _validate(review, _prior_represented())
    assert "uncertain" in incomplete


# --- The write-back chain: a downgrade must not be re-approved next pass ----

def test_a_downgraded_claim_is_not_reconciled_on_the_next_review():
    """End-to-end: pass 1 downgrades an over-claim, pass 2 cannot restore it.

    ``_validate_review`` downgrades a ``represented`` claim to ``uncertain``
    when ``_introduces_beyond`` finds a name the cited sources never state. That
    downgrade used to be assigned to a local variable and never written back to
    the mapping, so ``_ACTIVE_PRIOR_COVERAGE`` — which reads the mapping — filed
    the over-claim as ``represented`` WITH spans.

    That is precisely the shape coverage reconciliation restores. So the defect
    was not only a wrong diagnostic count: a later pass could have re-approved
    the very over-claim the check had just refused.

    This drives both passes through the real ``_validate_review`` and the real
    prior-coverage record, so it fails if either half of the chain regresses.
    """
    # Pass 1: the head line is offered as represented, but its span introduces
    # "Agile" and "Scrum", which the source never states.
    over_claim_line = "**Harborline Transit Authority** -- Tacoma, WA (Agile, Scrum)"
    lines = [dict(line) for line in LINES]
    lines[1] = {"line_id": "L2", "text": over_claim_line}
    span_end = len(over_claim_line)

    pass1 = {
        "claims": [
            {"id": "L1:0:26", "line_id": "L1", "start": 0, "end": 26,
             "verdict": "structural", "citations": []},
            {"id": f"L2:0:{span_end}", "line_id": "L2", "start": 0,
             "end": span_end, "verdict": "supported",
             "citations": HEAD_CITATION},
            {"id": "L3:0:44", "line_id": "L3", "start": 0, "end": 44,
             "verdict": "structural", "citations": []},
            {"id": "L4:0:46", "line_id": "L4", "start": 0, "end": 46,
             "verdict": "supported",
             "citations": [{"source_id": "doc", "start": BULLET_START,
                            "end": BULLET_END, "quote": BULLET_QUOTE}]},
        ],
        "representations": [
            {"source_claim_id": HEAD_ID, "status": "represented",
             "spans": [{"line_id": "L2", "start": 0, "end": span_end}]},
            {"source_claim_id": BULLET_ID, "status": "represented",
             "spans": [{"line_id": "L4", "start": 0, "end": 46}]},
        ],
    }

    token = sc._ACTIVE_PRIOR_COVERAGE.set(None)
    try:
        _, incomplete, _ = sc._validate_review(
            pass1, lines, {"doc": SOURCE}, INVENTORY)
    finally:
        sc._ACTIVE_PRIOR_COVERAGE.reset(token)

    # Pass 1 refused the over-claim...
    assert "uncertain" in incomplete
    # ...and the mapping now says so, which is what the next pass reads.
    head_mapping = next(m for m in pass1["representations"]
                        if m["source_claim_id"] == HEAD_ID)
    assert head_mapping["status"] == "uncertain"
    assert head_mapping["spans"] == []

    # Pass 2: the reviewer now calls the head missing. Reconciliation must NOT
    # restore it, because the prior record is `uncertain`, not `represented`.
    prior = {
        HEAD_ID: {"status": head_mapping["status"],
                  "spans": list(head_mapping["spans"]),
                  "line_text": {line["line_id"]: line["text"] for line in lines}},
        BULLET_ID: {"status": "represented",
                    "spans": [{"line_id": "L4", "start": 0, "end": 46}],
                    "line_text": {line["line_id"]: line["text"]
                                  for line in lines}},
    }
    # The head line is longer than the fixture's, so build pass 2 against the
    # actual line length rather than reusing `_review`'s fixed 0:46 span.
    pass2 = _review("missing", "structural", [])
    pass2["claims"][1] = {"id": f"L2:0:{span_end}", "line_id": "L2", "start": 0,
                          "end": span_end, "verdict": "structural",
                          "citations": []}
    claims, incomplete2, _ = _validate(pass2, prior, lines=lines)

    assert "missing" in incomplete2, (
        "a downgraded over-claim was reconciled back to represented on the next "
        "review — the write-back is not reaching prior coverage")
    head_claim = next(c for c in claims if c["line_id"] == "L2")
    assert head_claim["verdict"] == "structural"
    assert head_claim["citations"] == []
