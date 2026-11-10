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


def test_the_cite_phase_is_in_the_adapter_allowlist(monkeypatch):
    """Citation is an ordinary call, not an unknown phase or reserved recovery."""
    import json
    import pytest

    monkeypatch.setattr(correction_provider, "_approved_disclosure_available", lambda: True)

    class Session:
        def __init__(self):
            self.requests = []

        def post(self, url, **kwargs):
            self.requests.append(kwargs["json"])
            envelope = {"model": correction_provider.OR_SLUG_DRAFTER,
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                  "total_tokens": 2},
                        "choices": [{"finish_reason": "stop", "message": {
                            "content": json.dumps({"citations": []})}}]}

            class Response:
                status = 200

                def __init__(self):
                    self.content = self

                async def iter_chunked(self, size):
                    yield json.dumps(envelope).encode()

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

            return Response()

    session = Session()
    provider = CorrectionProvider("synthetic-key", session)
    assert asyncio.run(provider("cite_uncertain", CITE_PAYLOAD)) == {"citations": []}
    sent = json.loads(session.requests[0]["messages"][1]["content"])
    assert sent == {**CITE_PAYLOAD, "sources": list(CITE_PAYLOAD["sources"])}

    # Give the overlapping shape valid review fields too: without the shape guard
    # this would successfully translate instead of falling back on malformed input.
    overlapping = {**CITE_PAYLOAD, "sources": list(CITE_PAYLOAD["sources"]),
                   "source_claims": [{**CITE_PAYLOAD["source_claims"][0],
                                      "id": "document-0:0:8", "start": 0,
                                      "end": 8, "classification": "required"}]}
    assert asyncio.run(provider("post_patch_review", overlapping)) == {"citations": []}
    assert json.loads(session.requests[1]["messages"][1]["content"]) == overlapping

    with pytest.raises(ValueError, match="correction call budget exhausted"):
        asyncio.run(provider("not_a_phase", CITE_PAYLOAD))
    assert len(session.requests) == 2

    provider.calls = correction_provider.REPAIR_PATH_CALLS
    with pytest.raises(ValueError, match="correction call budget exhausted"):
        asyncio.run(provider("cite_uncertain", CITE_PAYLOAD))
    assert len(session.requests) == 2
    assert asyncio.run(provider("cite_uncertain", {**CITE_PAYLOAD, "must_judge": []})) == {
        "citations": []}
    provider.calls = correction_provider.MAX_CALLS
    with pytest.raises(ValueError, match="correction call budget exhausted"):
        asyncio.run(provider("cite_uncertain", {**CITE_PAYLOAD, "must_judge": []}))
    assert len(session.requests) == 3


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
