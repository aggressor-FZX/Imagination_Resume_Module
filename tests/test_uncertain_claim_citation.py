"""[uncertain-citation] One bounded repair for `source_completeness_uncertain`.

Traced from the live failure
----------------------------
Five real submissions on 2026-11-09 withheld; three were
`source_completeness_uncertain`, which was the largest single cause. Reproducing
it against the real engine (scripted reviewer, everything else real) showed the
branch precisely:

    phases: [source_inventory, initial_review]   status: withheld
    reason: source_completeness_uncertain        reviews: 1  repairs: 0

The reviewer declined to confirm a REQUIRED source claim and flagged no draft
line. `repair` may only replace a *flagged* span, so no repair was possible and
the run withheld on the reviewer's uncertainty alone.

What was added
--------------
One bounded question: name the draft line that already carries the fact, then
verify it ourselves. The draft is never edited.

Why this is not the fix that was reverted
-----------------------------------------
An earlier attempt INSERTED the source's text (reusing the "missing" restorers)
and approved a transplanted qualification, because inserting the source's complete
skill line makes a fact the draft MOVED into another record look sourced.

This does the opposite: it asks WHERE the fact already is and requires the cited
span to be a verbatim substring of the item's OWN source document. A moved fact
still fails, and that is asserted below alongside the positive case.
"""

import asyncio

import pytest

import source_correction as sc
from source_correction import SourceDocument, correct

DOC_A = """## Professional Experience
- Journeyman Electrician, Cascade Commercial Electric (03/2019 - Present)
- Installed commercial electrical systems across 40+ job sites.
"""

# A SECOND document, from which a fact could be transplanted.
DOC_B = """## Technical Skills
- Panel labeling
"""


def partition(text):
    """Gapless spans covering every character; blank lines merged into the prior span."""
    spans, cursor = [], 0
    for raw in text.splitlines(keepends=True):
        start, end = cursor, cursor + len(raw)
        cursor = end
        if not raw.strip() and spans:
            prev, _, _ = spans[-1]
            spans[-1] = (prev, end, text[prev:end])
        else:
            spans.append((start, end, text[start:end]))
    return spans


def inventory_for(text, source_id):
    return [
        {
            "id": f"{source_id}:{s}:{e}",
            "source_id": source_id,
            "start": s,
            "end": e,
            "quote": q,
            "classification": "non_resume" if q.lstrip().startswith("#") else "required",
        }
        for s, e, q in partition(text)
    ]


def review_answer(payload, citation_ids, *, cite_as=None):
    """A review that marks every non-heading line supported with a real citation.

    ``cite_as`` lets a test point a claim at a DIFFERENT document, which is how the
    transplant case is built without editing the draft.
    """
    out = []
    for line in payload["lines"]:
        text = line["text"]
        head = text.lstrip().startswith("#")
        source_id, source_text = cite_as or ("document-0", DOC_A)
        at = source_text.find(text)
        cites = []
        if not head and at >= 0:
            cites = [{"source_id": source_id, "start": at,
                      "end": at + len(text), "quote": text}]
        out.append({
            "id": f"{line['line_id']}:0:{len(text)}",
            "line_id": line["line_id"],
            "start": 0,
            "end": len(text),
            "verdict": "structural" if head else "supported",
            "citations": cites,
        })
    reps = []
    for item in payload["source_claims"]:
        if item["classification"] == "non_resume":
            reps.append({"source_claim_id": item["id"], "status": "non_resume", "spans": []})
        elif item["id"] in citation_ids:
            reps.append({"source_claim_id": item["id"], "status": "uncertain", "spans": []})
        else:
            reps.append({"source_claim_id": item["id"], "status": "uncertain", "spans": []})
    return {"claims": out, "representations": reps}


def is_citation_question(payload):
    """Shape of the citation question: source_claim_id/quote items, no claims.

    It now has its own phase name (`cite_uncertain`), so this helper is only used
    to assert the payload the engine actually sends -- the phase name is what keeps
    the real provider from mistaking it for a review.
    """
    return (
        not payload.get("claims")
        and any(isinstance(c, dict) and "source_claim_id" in c
                for c in (payload.get("source_claims") or []))
    )


def review_answer_mismatched(payload, uncertain_ids):
    """Supported claims whose citations point at the WRONG document.

    The line text is from DOC_A, so a citation into DOC_B cannot be verbatim --
    exactly the unverifiable provenance the repair must refuse.
    """
    out = []
    for line in payload["lines"]:
        text = line["text"]
        head = text.lstrip().startswith("#")
        cites = []
        if not head:
            cites = [{"source_id": "document-1", "start": 0,
                      "end": len(DOC_B.splitlines()[0]), "quote": DOC_B.splitlines()[0]}]
        out.append({
            "id": f"{line['line_id']}:0:{len(text)}",
            "line_id": line["line_id"],
            "start": 0,
            "end": len(text),
            "verdict": "structural" if head else "supported",
            "citations": cites,
        })
    reps = []
    for item in payload["source_claims"]:
        if item["classification"] == "non_resume":
            reps.append({"source_claim_id": item["id"], "status": "non_resume", "spans": []})
        elif item["id"] in uncertain_ids:
            reps.append({"source_claim_id": item["id"], "status": "uncertain", "spans": []})
        else:
            reps.append({"source_claim_id": item["id"], "status": "uncertain", "spans": []})
    return {"claims": out, "representations": reps}


def run(callback, sources=None):
    docs = sources or (SourceDocument("document-0", DOC_A),)
    return asyncio.run(correct(DOC_A, docs, callback, timeout_seconds=60.0))


def test_uncertain_required_claim_is_repaired_when_the_fact_is_cited():
    """Positive case: the fact IS present, the reviewer just would not confirm it."""
    inv = inventory_for(DOC_A, "document-0")
    uncertain = {item["id"] for item in inv if item["classification"] == "required"}
    phases = []

    async def callback(phase, payload):
        phases.append(phase)
        if phase == "source_inventory":
            return {"source_claims": inv}
        if phase == "cite_uncertain":
            entries = []
            for item in payload["source_claims"]:
                quote = (item.get("quote") or "").strip()
                for line in payload["lines"]:
                    if quote and quote in line["text"]:
                        entries.append({"source_claim_id": item["source_claim_id"],
                                        "claim_id": f"{line['line_id']}:0:{len(line['text'])}"})
                        break
            return {"citations": entries}
        if phase in ("initial_review", "post_patch_review", "post_restore_review"):
            return review_answer(payload, uncertain)
        return {"edits": []}

    result = run(callback)
    assert result.status == "approved", result.reason
    assert result.markdown is not None
    # The draft is returned byte-identical: this repair verifies, it does not write.
    assert result.markdown == DOC_A
    assert "cite_uncertain" in phases


def test_a_transplanted_fact_is_still_withheld():
    """Negative control: the cited provenance is a DIFFERENT document, so it fails.

    This is the shape of the reverted fix's failure --
    tests/test_route_two_documents.py::[transplanted_qualification] -- rebuilt here
    so the new repair cannot regress into it. The draft is unchanged; what differs
    is that the reviewer points every claim at document-1 (DOC_B) while the line
    text lives in document-0. The verifier requires the cited span to be verbatim
    in the ITEM'S OWN document, so the confirmation cannot be granted.
    """
    # The inventory must cover EVERY supplied document, so DOC_B is partitioned
    # too -- an inventory that covers only one source is correctly rejected as
    # malformed, which is a different failure than the one under test.
    inv = inventory_for(DOC_A, "document-0") + inventory_for(DOC_B, "document-1")
    uncertain = {item["id"] for item in inv if item["classification"] == "required"}

    async def callback(phase, payload):
        if phase == "source_inventory":
            return {"source_claims": inv}
        if phase == "cite_uncertain":
            entries = []
            for item in payload["source_claims"]:
                for line in payload["lines"]:
                    entries.append({
                        "source_claim_id": item["source_claim_id"],
                        "claim_id": f"{line['line_id']}:0:{len(line['text'])}",
                    })
                    break
            return {"citations": entries}
        if phase in ("initial_review", "post_patch_review", "post_restore_review"):
            # Every claim cites the OTHER document, so no cited span is verbatim
            # in the item's own source.
            item_src = {}
            return review_answer_mismatched(payload, uncertain)
        return {"edits": []}

    result = run(callback, sources=(SourceDocument("document-0", DOC_A),
                                    SourceDocument("document-1", DOC_B)))
    assert result.status == "withheld"
    assert result.markdown is None
    assert result.reason == "source_completeness_uncertain"


def test_no_citation_offered_keeps_the_original_withhold():
    """A reviewer that cannot place the fact gets the unchanged outcome."""
    inv = inventory_for(DOC_A, "document-0")
    uncertain = {item["id"] for item in inv if item["classification"] == "required"}

    async def callback(phase, payload):
        if phase == "source_inventory":
            return {"source_claims": inv}
        if phase == "cite_uncertain":
            return {"citations": []}
        if phase in ("initial_review", "post_patch_review", "post_restore_review"):
            return review_answer(payload, uncertain)
        return {"edits": []}

    result = run(callback)
    assert result.status == "withheld"
    assert result.reason == "source_completeness_uncertain"


def test_a_malformed_citation_answer_is_not_trusted():
    inv = inventory_for(DOC_A, "document-0")
    uncertain = {item["id"] for item in inv if item["classification"] == "required"}

    async def callback(phase, payload):
        if phase == "source_inventory":
            return {"source_claims": inv}
        if phase == "cite_uncertain":
            return {"citations": "not-a-list"}
        if phase in ("initial_review", "post_patch_review", "post_restore_review"):
            return review_answer(payload, uncertain)
        return {"edits": []}

    result = run(callback)
    assert result.status == "withheld"
    assert result.reason == "source_completeness_uncertain"


def test_a_citation_to_a_claim_the_reviewer_did_not_support_is_rejected():
    """The named claim must be one the reviewer itself marked supported."""
    inv = inventory_for(DOC_A, "document-0")
    uncertain = {item["id"] for item in inv if item["classification"] == "required"}

    async def callback(phase, payload):
        if phase == "source_inventory":
            return {"source_claims": inv}
        if phase == "cite_uncertain":
            return {"citations": [{"source_claim_id": item["source_claim_id"],
                                   "claim_id": "line-that-does-not-exist:0:5"}
                                  for item in payload["source_claims"]]}
        if phase in ("initial_review", "post_patch_review", "post_restore_review"):
            return review_answer(payload, uncertain)
        return {"edits": []}

    result = run(callback)
    assert result.status == "withheld"
    assert result.reason == "source_completeness_uncertain"


def test_the_repair_does_not_loop_when_the_reviewer_repeats_itself():
    """A first version used `continue`, re-asked, and looped 112,377 times.

    The confirmation must be decided once per pass, not by re-entering the review.
    """
    inv = inventory_for(DOC_A, "document-0")
    uncertain = {item["id"] for item in inv if item["classification"] == "required"}
    question_count = {"n": 0}

    async def callback(phase, payload):
        if phase == "source_inventory":
            return {"source_claims": inv}
        if phase == "cite_uncertain":
            question_count["n"] += 1
            return {"citations": []}
        if phase in ("initial_review", "post_patch_review", "post_restore_review"):
            return review_answer(payload, uncertain)
        return {"edits": []}

    result = run(callback)
    assert result.status == "withheld"
    # Asked once, answered once, done.
    assert question_count["n"] <= 1
