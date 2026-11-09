"""[cite-phase] The citation question must not be translated into a review.

Measured live 2026-11-09, and this is the bug that made the first version of the
repair look like it did nothing:

    phases: source_inventory, initial_review, repair, post_patch_review,
            post_patch_review        status: withheld
    reason: source_completeness_uncertain

The citation question reused the `post_patch_review` phase name and carried the
same `sources` + `source_claims` keys, so `CorrectionProvider` matched its review
translation branch first. That branch calls `indexed_review_request`, which
discards the `citations` contract entirely: the model was asked to REVIEW a draft
instead of naming where a fact already lives, answered `uncertain` again, and the
run withheld.

The engine's verifier was never at fault. The request never reached the model in
the right shape. This is the class of failure that a scripted reviewer cannot
show, because a scripted callback reads the payload it is given while the real
adapter rewrites it first.

Two defences, both asserted here:
  1. the citation question has its OWN phase name (`cite_uncertain`);
  2. even if the name were reused, `_is_citation_request` keeps the translation
     from claiming it.
"""

import asyncio

import correction_provider
from correction_provider import CorrectionProvider, _is_citation_request

CITE_PAYLOAD = {
    "contract": "cite",
    "sources": ({"source_id": "document-0", "text": "line one\nline two\n"},),
    "source_claims": [
        {"source_claim_id": "document-0:0:8", "source_id": "document-0",
         "quote": "line one"},
    ],
    "lines": [{"line_id": "L000001", "text": "line one"}],
}

REVIEW_PAYLOAD = {
    "contract": "review",
    "sources": ({"source_id": "document-0", "text": "line one\n"},),
    "source_claims": [
        {"id": "document-0:0:8", "source_id": "document-0", "start": 0,
         "end": 8, "quote": "line one", "classification": "required"},
    ],
    "lines": [{"line_id": "L000001", "text": "line one"}],
}


def test_the_citation_payload_is_recognised_by_shape():
    assert _is_citation_request(CITE_PAYLOAD) is True


def test_a_real_review_payload_is_not_mistaken_for_a_citation_request():
    # A review's source_claims carry `id`/`start`/`end`, never `source_claim_id`.
    assert _is_citation_request(REVIEW_PAYLOAD) is False


def test_an_empty_or_malformed_payload_is_not_a_citation_request():
    for payload in ({}, {"source_claims": []}, {"source_claims": "nope"},
                    {"source_claims": [{"id": "x"}]}):
        assert _is_citation_request(payload) is False


def test_the_cite_phase_is_in_the_adapter_allowlist():
    """An unlisted phase is rejected with the budget message, before the wire.

    Checked by reading the guard's allowlist rather than driving a request: the
    assertion is about which phases the adapter accepts, and a real call would
    also need a session, a key, and a disclosure policy.
    """
    import inspect

    source = inspect.getsource(CorrectionProvider.__call__)
    allowlist = source[source.index("if phase not in {"):]
    allowlist = allowlist[:allowlist.index("}")]
    assert '"cite_uncertain"' in allowlist, (
        "the citation phase was removed from the adapter allowlist; an unlisted "
        "phase raises 'correction call budget exhausted' before reaching the wire")
    # The budget guard must not exclude it either: it is an ordinary call.
    assert "must_judge" in source, (
        "the reserved-slot guard changed; re-check that cite_uncertain is not "
        "mistaken for a repair-path call")


def test_a_citation_request_is_not_sent_through_the_review_translation():
    """The translation branch must skip the citation payload.

    Asserted on the predicate the branch uses, because the alternative is to
    inspect `correction_protocol` output, which is a different module's contract.
    """
    # If the guard were absent, this payload WOULD satisfy the branch's other
    # conditions (phase + sources + source_claims), which is precisely the bug.
    assert "sources" in CITE_PAYLOAD and "source_claims" in CITE_PAYLOAD
    assert _is_citation_request(CITE_PAYLOAD) is True
    # …and the phase name that reaches the adapter is its own.
    import inspect
    import source_correction

    body = inspect.getsource(source_correction._await_citation_for_uncertain)
    assert '"cite_uncertain"' in body, (
        "the citation question no longer uses its own phase name; reusing "
        "post_patch_review lets the review translation discard the contract")
