"""Bounded recovery for a review the model answered only partially.

``incomplete_review`` is a delivery failure, not a completed correction. Withholding
without trying to finish the answer is only right when the gap is genuinely
unrecoverable, so the engine now attempts **one** completion re-review that names the
exact lines the previous answer omitted.

Every test here checks one of the bounds. The recovery must be able to turn a partial
answer into a full one, and must never become a retry loop, never run after a repair,
never exceed the call budget, and never bypass the completeness or factual checks.
"""
import asyncio
import json

import pytest

import correction_provider
from correction_provider import CorrectionProvider, REPAIR_PATH_CALLS, MAX_CALLS
from pipeline_config import OR_SLUG_DRAFTER
from source_correction import SourceDocument, correct


SOURCE = """## Professional Experience
**Synthetic Workshop**
- Maintained the workshop inventory and documented repair procedures for service technicians.
- Improved diagnostic accuracy by 30% using structured test procedures and careful review.

## Education
**Synthetic College**
- Completed technical training in electrical systems and equipment maintenance.
"""


class Stream:
    def __init__(self, data):
        self.data = data

    async def iter_chunked(self, size):
        for index in range(0, len(self.data), size):
            yield self.data[index:index + size]


class Response:
    status = 200

    def __init__(self, body):
        self.content = Stream(body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def inventory_answer(payload):
    required = [segment["index"] for segment in payload["segments"]
                if segment["text"].strip() and not segment["text"].startswith("## ")]
    headings = [segment["index"] for segment in payload["segments"]
                if segment["text"].startswith("## ")]
    return {"required": required, "non_resume": headings, "uncertain": []}


def full_review_answer(payload):
    """A complete, honest answer: every line judged, every fact accounted for."""
    segments, lines = payload["segments"], payload["lines"]
    verdicts = {"supported": [], "unsupported": [], "uncertain": [], "structural": []}
    evidence, carried = {}, set()
    for line in lines:
        text = line["text"]
        if text.startswith("## "):
            verdicts["structural"].append(line["line_id"])
            continue
        exact = [segment["index"] for segment in segments if segment["text"].strip() == text]
        if exact:
            verdicts["supported"].append(line["line_id"])
            evidence[line["line_id"]] = exact
            carried.update(exact)
        else:
            verdicts["unsupported"].append(line["line_id"])
    verdicts["evidence"] = evidence
    verdicts["represented"] = [segment["index"] for segment in segments
                               if segment["index"] in carried]
    verdicts["missing"] = []
    return verdicts


class PartialSession:
    """Omits the requested lines on the first review, then answers completely.

    ``omit_rounds`` says how many reviews omit the trailing lines. Recovery is
    expected to succeed on the second round when ``omit_rounds`` is 1.
    """

    def __init__(self, omit_rounds=1, omit_count=2):
        self.omit_rounds = omit_rounds
        self.omit_count = omit_count
        self.requests = []
        self.answers = []
        self.reviews = 0

    async def close(self):
        pass

    def post(self, url, **kwargs):
        self.requests.append(kwargs)
        payload = json.loads(kwargs["json"]["messages"][1]["content"])
        if "segments" not in payload:
            body = {"edits": []}
        elif "lines" not in payload:
            body = inventory_answer(payload)
        else:
            self.reviews += 1
            body = full_review_answer(payload)
            if self.reviews <= self.omit_rounds:
                # Leave the trailing lines out of every verdict list, exactly as the
                # probe recording showed.
                omitted = {line["line_id"] for line in payload["lines"][-self.omit_count:]}
                for key in ("supported", "unsupported", "uncertain", "structural"):
                    body[key] = [line_id for line_id in body[key] if line_id not in omitted]
                body["evidence"] = {line_id: value for line_id, value in body["evidence"].items()
                                    if line_id not in omitted}
        self.answers.append(body)
        return Response(json.dumps({
            "model": OR_SLUG_DRAFTER,
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            "choices": [{"finish_reason": "stop",
                         "message": {"content": json.dumps(body)}}],
        }).encode())


@pytest.fixture
def bypass(monkeypatch):
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available", lambda: True)


def run(draft, session, *, timeout_seconds=120.0):
    provider = CorrectionProvider("synthetic-key", session)
    return asyncio.run(correct(draft, (SourceDocument("resume", SOURCE),), provider,
                              timeout_seconds=timeout_seconds)), provider


# --- 1. Recovery works ------------------------------------------------------

def test_partial_answer_is_completed_by_one_recovery(bypass):
    """A one-off omission is recoverable: the completed answer can be approved."""
    session = PartialSession(omit_rounds=1)
    result, provider = run(SOURCE, session)

    assert (result.status, result.reason) == ("approved", None)
    # Two reviews: the partial one and the completion.
    assert result.review_calls == 2
    assert [call["phase"] for call in provider.usage["calls"]] == [
        "source_inventory", "initial_review", "initial_review"]


def test_recovery_request_names_exactly_the_omitted_lines(bypass):
    """The re-ask is a completion, not a blind retry of the same question."""
    session = PartialSession(omit_rounds=1, omit_count=2)
    result, _ = run(SOURCE, session)
    assert result.status == "approved"

    first, second = session.requests[1], session.requests[2]
    first_payload = json.loads(first["json"]["messages"][1]["content"])
    second_payload = json.loads(second["json"]["messages"][1]["content"])

    assert "must_judge" not in first_payload
    # The hint must be exactly the line ids the first answer left out, and must be a
    # list of ids -- never a nested structure that only looks like one.
    review_answers = [answer for answer in session.answers if "supported" in answer]
    first_answer = review_answers[0]
    judged: set[str] = set()
    for key in ("supported", "unsupported", "uncertain", "structural"):
        judged.update(first_answer.get(key, []))
    omitted = [line["line_id"] for line in first_payload["lines"]
               if line["line_id"] not in judged]
    assert omitted == [line["line_id"] for line in first_payload["lines"][-2:]]
    assert second_payload["must_judge"] == omitted
    assert all(isinstance(value, str) for value in second_payload["must_judge"])
    # The hint must not be smuggled into a normal review.
    assert all("must_judge" not in json.loads(req["json"]["messages"][1]["content"])
               for req in session.requests[:1])


# --- 2. Every bound ---------------------------------------------------------

def test_recovery_happens_at_most_once(bypass):
    """If the model omits again, recovery is spent and the result withholds."""
    session = PartialSession(omit_rounds=99)
    result, provider = run(SOURCE, session)

    assert (result.status, result.reason) == ("withheld", "incomplete_review")
    # Exactly two reviews: the partial answer and the single completion attempt.
    assert result.review_calls == 2
    assert session.reviews == 2
    assert result.markdown is None


def test_a_second_identical_omission_is_not_retried_again(bypass):
    """No blind retry: the same unanswered set twice means the model cannot answer."""
    session = PartialSession(omit_rounds=99)
    result, provider = run(SOURCE, session)
    payloads = [json.loads(req["json"]["messages"][1]["content"])
                for req in session.requests if "lines" in json.loads(
                    req["json"]["messages"][1]["content"])]
    # Only one payload carries the hint, and it is the last review performed.
    assert sum("must_judge" in payload for payload in payloads) == 1
    assert result.reason == "incomplete_review"


def test_recovery_is_allowed_after_a_repair_of_the_same_document(bypass):
    """A repair is not a change of document, so it must not disqualify recovery.

    Measured on a live run (``diagcons-28fc7c2a88e3``): the initial review judged
    all 19 lines, the post-patch review then omitted the six experience bullets,
    and the run withheld as ``incomplete_review`` solely because a repair had
    already happened. The completion would have re-asked about the SAME repaired
    markdown, so the flag was a proxy for staleness that did not hold.

    The invariant that does hold is document identity, which the next test pins.
    """
    from source_correction import _recover_unjudged, UnjudgedReview

    markdown = "## Experience\n- one\n- two\n"
    state = {"recovered": False, "repaired": True, "last_unjudged": None,
             "unjudged_digest": None}
    outcome = asyncio.run(_recover_unjudged(
        "post_patch_review", UnjudgedReview(["L000001"]),
        deadline=10 ** 9, state=state, markdown=markdown))
    assert outcome == ["L000001"], "a same-document omission must be recoverable"
    assert state["recovered"] is True


def test_recovery_is_refused_when_the_document_changed(bypass):
    """The real staleness guard: completion must judge the draft that was omitted."""
    from source_correction import _recover_unjudged, UnjudgedReview

    first = "## Experience\n- one\n- two\n"
    changed = "## Experience\n- one\n- two revised\n"
    state = {"recovered": False, "repaired": True, "last_unjudged": None,
             "unjudged_digest": None}
    asyncio.run(_recover_unjudged("initial_review", UnjudgedReview(["L000001"]),
                                  deadline=10 ** 9, state=state, markdown=first))
    state["recovered"] = False  # isolate the digest check from the once-only rule
    outcome = asyncio.run(_recover_unjudged("initial_review", UnjudgedReview(["L000002"]),
                                            deadline=10 ** 9, state=state,
                                            markdown=changed))
    assert outcome == "incomplete_review"


def test_a_post_repair_omission_is_completed_end_to_end(bypass):
    """The live failure shape: repair, then a partial re-review, then approval.

    ``PartialSession`` omits lines on its FIRST review, so this drives the order the
    live run took by omitting on the SECOND instead. The first review must genuinely
    flag a line or the engine approves without repairing, so the draft carries one
    span that differs from the source; the repair restores it and the re-review then
    omits trailing lines, which is exactly what ``diagcons-28fc7c2a88e3`` recorded.
    """
    drift = SOURCE.replace("Maintained the workshop", "Kept the workshop")

    class OmitOnSecondReview(PartialSession):
        def post(self, url, **kwargs):
            self.requests.append(kwargs)
            payload = json.loads(kwargs["json"]["messages"][1]["content"])
            if "flagged_claim_ids" in payload:
                # A real repair: put the drifted span back to its source wording so
                # the edit is neither a no-op nor a new fact.
                lines = {line["line_id"]: line["text"] for line in payload["lines"]}
                edits = []
                for cid in payload["flagged_claim_ids"]:
                    line_id, start, end = cid.rsplit(":", 2)
                    span = lines[line_id][int(start):int(end)]
                    edits.append({"claim_id": cid,
                                  "replacement": span.replace("Kept", "Maintained")})
                body = {"edits": edits}
            elif "segments" not in payload:
                body = {"edits": []}
            elif "lines" not in payload:
                body = inventory_answer(payload)
            else:
                self.reviews += 1
                body = full_review_answer(payload)
                if self.reviews == 2:
                    # The post-patch review omits, exactly as the live run did.
                    omitted = {line["line_id"] for line in payload["lines"][-2:]}
                    for key in ("supported", "unsupported", "uncertain", "structural"):
                        body[key] = [v for v in body[key] if v not in omitted]
                    body["evidence"] = {k: v for k, v in body["evidence"].items()
                                        if k not in omitted}
            self.answers.append(body)
            return Response(json.dumps({
                "model": OR_SLUG_DRAFTER,
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                "choices": [{"finish_reason": "stop",
                             "message": {"content": json.dumps(body)}}],
            }).encode())

    session = OmitOnSecondReview()
    result, provider = run(drift, session)
    phases = [call["phase"] for call in provider.usage["calls"]]
    assert "repair" in phases, phases
    # The completion happened AFTER the repair, and it carried the hint.
    assert phases.count("post_patch_review") >= 2, phases
    hinted = [json.loads(req["json"]["messages"][1]["content"])
              for req in session.requests if "lines" in json.loads(
                  req["json"]["messages"][1]["content"])]
    assert any("must_judge" in payload for payload in hinted), "no completion was asked"
    assert result.status == "approved", result.reason


def test_recovery_is_refused_when_the_deadline_has_passed(bypass):
    from source_correction import _recover_unjudged, UnjudgedReview

    state = {"recovered": False, "repaired": False, "last_unjudged": None,
             "unjudged_digest": None}
    outcome = asyncio.run(_recover_unjudged(
        "initial_review", UnjudgedReview(["L000001"]),
        deadline=0.0, state=state))
    assert outcome == "timeout_initial_review"
    assert state["recovered"] is False


def test_recovery_is_refused_with_no_line_identifiers(bypass):
    """Without a named gap there is nothing to complete, so this is not recovery."""
    from source_correction import _recover_unjudged, UnjudgedReview

    state = {"recovered": False, "repaired": False, "last_unjudged": None,
             "unjudged_digest": None}
    outcome = asyncio.run(_recover_unjudged(
        "initial_review", UnjudgedReview([]),
        deadline=10 ** 9, state=state))
    assert outcome == "incomplete_review"
    assert state["recovered"] is False


def test_recovery_headroom_cannot_be_spent_by_a_plain_call(bypass):
    """The seventh call exists only for a request carrying the recovery hint."""
    session = PartialSession(omit_rounds=1)
    result, provider = run(SOURCE, session)
    assert result.status == "approved"
    # A one-off omission costs exactly one extra call: inventory + 2 reviews.
    assert provider.calls == REPAIR_PATH_CALLS - 3
    assert MAX_CALLS == REPAIR_PATH_CALLS + 1

    # With the budget at the repair-path ceiling, a normal request is refused...
    fresh = CorrectionProvider("synthetic-key", PartialSession())
    fresh.calls = REPAIR_PATH_CALLS
    with pytest.raises(ValueError, match="budget exhausted"):
        asyncio.run(fresh("initial_review", {"contract": "c", "lines": []}))
    # ...and a request that merely mentions the hint key cannot spend it either,
    # because the budget check is positional, not payload-driven.
    probe = PartialSession()
    provider2 = CorrectionProvider("synthetic-key", probe)
    provider2.calls = MAX_CALLS
    with pytest.raises(ValueError, match="budget exhausted"):
        asyncio.run(provider2("initial_review", {
            "contract": "c", "segments": [], "lines": [], "must_judge": ["L000001"],
            "sources": [], "source_claims": []}))


# --- 3. Recovery cannot weaken the gates ------------------------------------

def test_recovery_cannot_turn_an_omitted_fact_into_approval(bypass):
    """The completion answer still faces the unchanged completeness check."""
    class OmitFactThenAdmitMissing(PartialSession):
        @staticmethod
        def _mark_missing(payload, body):
            texts = [line["text"] for line in payload["lines"]]
            absent = [segment["index"] for segment in payload["segments"]
                      if segment["text"].strip() and segment["text"] not in texts]
            body["represented"] = [index for index in body["represented"]
                                   if index not in absent]
            body["missing"] = absent
            return body

        def post(self, url, **kwargs):
            self.requests.append(kwargs)
            payload = json.loads(kwargs["json"]["messages"][1]["content"])
            if "segments" not in payload:
                body = {"edits": []}
            elif "lines" not in payload:
                body = inventory_answer(payload)
            else:
                self.reviews += 1
                body = full_review_answer(payload)
                if self.reviews == 1:
                    omitted = {line["line_id"] for line in payload["lines"][-2:]}
                    for key in ("supported", "unsupported", "uncertain", "structural"):
                        body[key] = [v for v in body[key] if v not in omitted]
                else:
                    # The completion now answers, and truthfully reports an omission.
                    body = self._mark_missing(payload, body)
            return Response(json.dumps({
                "model": OR_SLUG_DRAFTER,
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                "choices": [{"finish_reason": "stop",
                             "message": {"content": json.dumps(body)}}],
            }).encode())

    session = OmitFactThenAdmitMissing(omit_rounds=1)
    draft = SOURCE.replace(
        "- Completed technical training in electrical systems and equipment maintenance.\n", "")
    result, _ = run(draft, session)
    # Recovery produced a complete answer, and the factual check still fired.
    assert (result.status, result.reason) == ("withheld", "source_facts_missing")
    assert result.markdown is None


def test_recovery_does_not_soften_the_evidence_requirement(bypass):
    """A completed answer still needs named evidence for every supported line."""
    class OmitThenSupportWithoutEvidence(PartialSession):
        def post(self, url, **kwargs):
            self.requests.append(kwargs)
            payload = json.loads(kwargs["json"]["messages"][1]["content"])
            if "segments" not in payload:
                body = {"edits": []}
            elif "lines" not in payload:
                body = inventory_answer(payload)
            else:
                self.reviews += 1
                body = full_review_answer(payload)
                if self.reviews == 1:
                    omitted = {line["line_id"] for line in payload["lines"][-2:]}
                    for key in ("supported", "unsupported", "uncertain", "structural"):
                        body[key] = [v for v in body[key] if v not in omitted]
                else:
                    # Second time round: claim support but name no evidence at all.
                    body["evidence"] = {}
            return Response(json.dumps({
                "model": OR_SLUG_DRAFTER,
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                "choices": [{"finish_reason": "stop",
                             "message": {"content": json.dumps(body)}}],
            }).encode())

    session = OmitThenSupportWithoutEvidence(omit_rounds=1)
    result, _ = run(SOURCE, session)
    # Unnamed support becomes uncertain, and the draft is withheld rather than
    # approved: recovery cannot manufacture evidence.
    assert result.status == "withheld"
    assert result.markdown is None


def test_incomplete_review_withholds_when_recovery_cannot_help(bypass):
    """If every review omits lines, the failure stays a withheld delivery."""
    session = PartialSession(omit_rounds=99, omit_count=len(SOURCE.splitlines()))
    result, _ = run(SOURCE, session)
    assert result.status == "withheld"
    assert result.markdown is None
    assert result.reason in {"incomplete_review", "no_supported_claims"}
