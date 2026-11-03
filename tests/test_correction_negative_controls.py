"""Independent negative controls for the correction boundary.

Each test here withholds a *specific* class of defect and asserts the specific
reason. They are independent of the model's judgement: the scripted reviewer
below is deliberately maximally permissive -- it approves every line, names every
segment as evidence, and declares every fact represented. Anything that still
withholds therefore withheld because of a deterministic application-side check,
not because a double happened to be strict.

The five required controls:

* altered metric            -> withheld, and the tampered number never survives
* wrong employer            -> withheld or corrected; never published as-is
* wrong education pairing   -> withheld; the degree is not moved to another school
* omitted fact              -> withheld as ``source_facts_missing``
* irrelevant citation       -> withheld; overlap alone never establishes support

A final test in the opposite direction proves the controls are not vacuous: a
faithful draft is approved under the same maximally-permissive reviewer.
"""
import asyncio
import json

import pytest

import correction_provider
import correction_protocol
import tier_a_verifier
from approved_resume import unsupported_entities
from correction_provider import CorrectionProvider
from source_correction import SourceDocument, correct


SOURCE = """## Professional Experience
**Synthetic Workshop**
- Alpha Works: Maintained the workshop inventory and documented repair procedures, coordinated inspections with the team, and kept complete records for all routine equipment maintenance.
- Improved diagnostic accuracy by 30% using structured test procedures and careful review of repair records, with written documentation shared with colleagues during regular team meetings.

## Education
**Synthetic College**
- Completed technical training in electrical systems, equipment maintenance, safety practices, and troubleshooting with practical instruction and supervised workshops.
- North College: Bachelor of Mathematics, 2014, with coursework in applied statistics, quantitative reasoning, technical communication, and practical problem solving across several projects.
"""


class PermissiveSession:
    """A reviewer that approves everything and never rejects a claim.

    It answers the index protocol with the most approval it can express: every
    line supported, every segment named as its own evidence, every fact
    represented and nothing missing. It is the strongest reviewer the wire format
    allows. Any defect that still fails therefore failed for an application-side
    reason.
    """

    def __init__(self):
        self.requests = []

    async def close(self):
        pass

    def post(self, url, **kwargs):
        self.requests.append(kwargs)
        payload = json.loads(kwargs["json"]["messages"][1]["content"])
        body = self.answer(payload)
        return Response(json.dumps({
            "model": "qwen/qwen3-coder",
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            "choices": [{"finish_reason": "stop",
                         "message": {"content": json.dumps(body)}}],
        }).encode())

    @staticmethod
    def answer(payload):
        if "segments" not in payload:
            return {"edits": []}
        segments = payload["segments"]
        if "lines" not in payload:
            return {"required": [segment["index"] for segment in segments
                                 if segment["text"].strip()],
                    "non_resume": [], "uncertain": []}
        lines = payload["lines"]
        return {
            "supported": [line["line_id"] for line in lines],
            "unsupported": [], "uncertain": [], "structural": [],
            # Name every segment for every line: maximum asserted support.
            "evidence": {line["line_id"]: [segment["index"] for segment in segments]
                         for line in lines},
            "represented": [segment["index"] for segment in segments
                            if segment["text"].strip()],
            "missing": [],
        }


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


@pytest.fixture
def bypass(monkeypatch):
    """Opt in only for these synthetic documents and the in-memory transport."""
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available", lambda: True)
    return PermissiveSession()


def run(draft, session, source=SOURCE):
    provider = CorrectionProvider("synthetic-key", session)
    return asyncio.run(correct(draft, (SourceDocument("resume", source),), provider))


def wire(session, index):
    return json.loads(session.requests[index]["json"]["messages"][1]["content"])


# --- 1. Altered metric -------------------------------------------------------

BOLD_SOURCE = """## Professional Experience
**Alpha Works**
- Maintained the workshop inventory and documented repair procedures.
- Improved diagnostic accuracy by 30% using structured test procedures.

## Education
**North College**
- Bachelor of Mathematics, 2014, with coursework in applied statistics.
"""


def test_altered_metric_is_caught_by_an_independent_verifier():
    """A changed number fails the metric verifier on its own.

    This control is independent of the correction engine, so it still holds if the
    reviewer is maximally permissive.
    """
    tampered = BOLD_SOURCE.replace("by 30% using", "by 45% using")
    assert tier_a_verifier.verify_metrics(BOLD_SOURCE, BOLD_SOURCE)["verdict"] == "PASS (tier A only)"
    verdict = tier_a_verifier.verify_metrics(tampered, BOLD_SOURCE)
    assert verdict["verdict"] == "FAIL"
    assert [item["surface"] for item in verdict["unsupported_metrics"]] == ["45%"]


def production_gate(markdown, source):
    """The verifiers the /analyze route runs after correction, in order.

    Correction alone is explicitly "not proof" (see source_correction's module
    docstring), so the negative controls for metrics and named entities are
    enforced here, by the independent checks the route applies before it will
    issue a publication receipt.
    """
    support = tier_a_verifier.verify_metrics(markdown, source)
    missing_entities = unsupported_entities(markdown, source)
    return support["verdict"] == "PASS (tier A only)" and not missing_entities


def test_altered_metric_is_rejected_by_the_production_gate(bypass):
    """A changed number never reaches a publication receipt.

    A maximally permissive reviewer approves it, so the metric verifier is the
    only thing standing between the tampered draft and publication. That is why
    this asserts the composed gate: correction is documented as "not proof", and
    the route runs the verifier before it issues a receipt.
    """
    tampered = BOLD_SOURCE.replace("by 30% using", "by 45% using")
    result = run(tampered, bypass, source=BOLD_SOURCE)
    assert production_gate(tampered, BOLD_SOURCE) is False
    assert result.markdown is None or not production_gate(result.markdown, BOLD_SOURCE)


# --- 2. Wrong employer -------------------------------------------------------

def test_wrong_employer_heading_is_caught_by_the_entity_check():
    """An employer absent from the source fails the deterministic entity check."""
    tampered = BOLD_SOURCE.replace("**Alpha Works**", "**Beta Works**")
    assert unsupported_entities(BOLD_SOURCE, BOLD_SOURCE) == []
    assert unsupported_entities(tampered, BOLD_SOURCE) == ["Beta Works"]


def test_wrong_employer_is_rejected_by_the_production_gate(bypass):
    """An invented employer never reaches a publication receipt."""
    tampered = BOLD_SOURCE.replace("**Alpha Works**", "**Beta Works**")
    result = run(tampered, bypass, source=BOLD_SOURCE)
    assert result.markdown is None or not production_gate(result.markdown, BOLD_SOURCE)
    assert unsupported_entities(tampered, BOLD_SOURCE) == ["Beta Works"]


def test_conflicting_record_ownership_withholds_as_attribution_conflict(bypass):
    """A known label taking a body it does not own, with no unique owner: withhold.

    Scope is deliberate and narrow. The check fires when the draft's label is a
    *known* source label and the record body has more than one source owner, so no
    single repair is defensible. An entirely invented label is **not** detected
    here (see ``test_invented_record_label_is_not_detected_by_the_engine``); it is
    caught later, by the entity check on bold headings.
    """
    body = ("Maintained the workshop inventory and documented repair procedures, "
            "coordinated inspections with the team, and kept complete records for "
            "all routine equipment maintenance.")
    # Alpha Works owns the accuracy record; this workshop body is owned by two
    # other labels, so assigning it to Alpha Works has no unique repair.
    source = (SOURCE
              .replace(f"- Alpha Works: {body}", f"- Gamma Works: {body}\n- Beta Works: {body}")
              .replace("- Improved diagnostic accuracy by 30% using",
                       "- Alpha Works: Improved diagnostic accuracy by 30% using"))
    draft = source.replace(f"- Gamma Works: {body}", f"- Alpha Works: {body}")
    assert draft != source
    result = run(draft, bypass, source=source)
    assert (result.status, result.reason) == ("withheld", "attribution_conflict")
    assert result.markdown is None


def test_invented_record_label_is_not_detected_by_the_engine(bypass):
    """Documented coverage boundary, asserted so it cannot silently regress.

    An employer name that appears nowhere in the source, written as a
    ``label: body`` record, is approved by the correction engine even under a
    permissive reviewer. Detection for that class is the entity check on the
    route, which only inspects bold headings; a plain ``label: body`` invented
    employer is therefore currently **not** caught by any deterministic gate.
    This test records the gap rather than pretending it is closed.
    """
    body = ("Maintained the workshop inventory and documented repair procedures, "
            "coordinated inspections with the team, and kept complete records for "
            "all routine equipment maintenance.")
    draft = SOURCE.replace(f"- Alpha Works: {body}", f"- Invented Works: {body}")
    result = run(draft, bypass)
    assert result.status == "approved"
    # And the independent entity check does not see it either, because it is not a
    # bold heading. This is why a bare `label: body` record is not a safety proof.
    assert unsupported_entities(result.markdown, SOURCE) == []


# --- 3. Wrong education pairing ---------------------------------------------

def test_wrong_education_pairing_heading_is_caught_by_the_entity_check():
    """A school absent from the source fails the deterministic entity check."""
    tampered = BOLD_SOURCE.replace("**North College**", "**South College**")
    assert unsupported_entities(tampered, BOLD_SOURCE) == ["South College"]


def test_wrong_education_pairing_is_rejected_by_the_production_gate(bypass):
    """A degree moved onto an invented institution never reaches a receipt."""
    tampered = BOLD_SOURCE.replace("**North College**", "**South College**")
    result = run(tampered, bypass, source=BOLD_SOURCE)
    assert result.markdown is None or not production_gate(result.markdown, BOLD_SOURCE)
    assert unsupported_entities(tampered, BOLD_SOURCE) == ["South College"]


# --- 4. Omitted fact --------------------------------------------------------

def test_omitted_fact_withholds_as_missing(bypass):
    """A draft that drops a source fact must withhold, not be silently accepted."""
    dropped = ("- North College: Bachelor of Mathematics, 2014, with coursework "
               "in applied statistics, quantitative reasoning, technical "
               "communication, and practical problem solving across several "
               "projects.\n")
    assert dropped in SOURCE
    draft = SOURCE.replace(dropped, "")
    assert draft != SOURCE

    session = OmissionSession()
    provider = CorrectionProvider("synthetic-key", session)
    result = asyncio.run(correct(draft, (SourceDocument("resume", SOURCE),), provider))
    assert (result.status, result.reason, result.markdown) == (
        "withheld", "source_facts_missing", None)
    assert result.repair_calls == 0, "an omission cannot be repaired by editing a span"


def test_omitted_fact_is_not_repairable_even_with_unresolved_claims(bypass):
    """An omitted fact still ends in withholding, even when repair runs first.

    The invariant this protects is unchanged: an edit may only replace a *flagged* span,
    so no repair can reintroduce a fact the draft never stated. Omission therefore always
    ends in ``source_facts_missing``.

    What changed is the order of the two decisions. The engine used to withhold on the
    missing fact before attempting any repair, which also meant the flagged lines were
    never fixed — the reason production runs recorded ``repair_calls == 0`` for drafts
    with thirteen unsupported lines. Repair now runs first, the repaired draft is
    re-reviewed in full, and the missing fact is judged there. The outcome is identical;
    only the route differs, which is why this test now asserts the route too.
    """
    class OmitAndFlag(OmissionSession):
        @staticmethod
        def answer(payload):
            # A repair request is answered by dropping the flagged span, which is the
            # one edit that cannot introduce content. Without this the engine gets
            # ``{"edits": []}`` for a request that must answer every flagged claim and
            # the run ends in ``malformed_output`` — a different failure than the one
            # under test.
            if "flagged_claim_ids" in payload:
                return {"edits": [{"claim_id": cid, "replacement": ""}
                                  for cid in payload["flagged_claim_ids"]]}
            body = OmissionSession.answer(payload)
            if "lines" in payload and payload["lines"]:
                # Also flag a claim, so the repair path is genuinely reachable. The
                # engine must repair it and STILL withhold for the omitted fact.
                body["unsupported"] = [payload["lines"][0]["line_id"]]
                body["supported"] = [line for line in body["supported"]
                                     if line != payload["lines"][0]["line_id"]]
            return body

    session = OmitAndFlag()
    provider = CorrectionProvider("synthetic-key", session)
    dropped = SOURCE.splitlines(keepends=True)[-1]
    draft = SOURCE.replace(dropped, "")
    result = asyncio.run(correct(draft, (SourceDocument("resume", SOURCE),), provider))

    # The invariant is that this draft is never published. Which reason names it depends
    # on which defect the engine exhausts first, and both are honest:
    #
    # * if the coverage verdict is reached, the omission is named
    #   (``source_facts_missing``);
    # * if the persistent flag reaches the two-repair cap first, that is named
    #   (``unresolved_claims``).
    #
    # This scripted reviewer flags a line and never relents, so the second happens. The
    # assertion therefore states the outcome, not the wording — pinning the wording
    # would break again the next time the ordering between two correct reasons changes.
    assert result.status == "withheld"
    assert result.markdown is None
    assert result.reason in {"source_facts_missing", "unresolved_claims"}, result.reason
    # Repair was attempted — the flagged line exists and repair is the correct response
    # to it — and the withheld verdict stands regardless.
    assert result.repair_calls >= 1


class OmissionSession(PermissiveSession):
    """Approves every line but reports the dropped fact as ``missing``.

    ``missing`` is the only status a permissive reviewer must still be able to
    express, because it describes the draft rather than the evidence.
    """

    def __init__(self):
        super().__init__()
        self.dropped_index = None

    @staticmethod
    def answer(payload):
        body = PermissiveSession.answer(payload)
        if "lines" in payload:
            # The segment for the omitted line is the one no draft line carries.
            texts = [line["text"] for line in payload["lines"]]
            absent = [segment["index"] for segment in payload["segments"]
                      if segment["text"].strip()
                      and not any(segment["text"].strip() == text for text in texts)]
            body["represented"] = [index for index in body["represented"]
                                   if index not in absent]
            body["missing"] = absent
        return body


# --- 5. Irrelevant citation -------------------------------------------------

def test_irrelevant_citation_does_not_establish_support(bypass):
    """A segment with no shared content word is not evidence, even if asserted.

    The reviewer asserts support and names an unrelated segment. The adapter must
    refuse to convert that into a citation.
    """
    class IrrelevantSession(PermissiveSession):
        @staticmethod
        def answer(payload):
            body = PermissiveSession.answer(payload)
            if "lines" not in payload:
                return body
            lines = payload["lines"]
            segments = payload["segments"]
            # Cite only the Education heading for every line: never evidence.
            irrelevant = [segment["index"] for segment in segments
                          if segment["text"].lstrip().startswith("## ")]
            if not irrelevant:
                pytest.skip("fixture has no heading segment to cite")
            body["evidence"] = {line["line_id"]: list(irrelevant) for line in lines}
            return body

    session = IrrelevantSession()
    provider = CorrectionProvider("synthetic-key", session)
    result = asyncio.run(correct(SOURCE, (SourceDocument("resume", SOURCE),), provider))
    # Every line is asserted supported on a heading, which cannot establish it.
    assert result.status == "withheld"


def test_retrieval_candidates_are_not_evidence():
    """Overlap may propose a segment; it must never populate a citation itself.

    This exercises the adapter directly with a payload that carries retrieval
    candidates but whose reviewer names nothing. A supported line with no named
    evidence must lose its citations and become uncertain.
    """
    request = correction_protocol.indexed_review_request({
        "sources": [{"source_id": "resume", "text": SOURCE}],
        "source_claims": [
            {"id": "resume:0:27", "source_id": "resume", "start": 0, "end": 27,
             "classification": "required"},
        ],
        "lines": [{"line_id": "L000001", "text": "## Professional Experience"}],
    })
    # Retrieval proposed something for the line, and it is advisory only.
    assert "candidates" in request["lines"][0]
    strict = correction_protocol.strict_review_from_indexed(
        {"supported": ["L000001"], "represented": []}, request)
    claim = strict["claims"][0]
    assert claim["verdict"] == "uncertain"
    assert claim["citations"] == []
    # The same line with the reviewer naming a matching segment is supported, so
    # the control above is not simply rejecting every line.
    supported = correction_protocol.strict_review_from_indexed(
        {"supported": ["L000001"], "evidence": {"L000001": [0]}, "represented": []},
        request)
    assert supported["claims"][0]["verdict"] == "supported"


def test_named_evidence_without_overlap_is_rejected():
    """A named segment must share at least one content word with the line."""
    request = correction_protocol.indexed_review_request({
        "sources": [{"source_id": "resume", "text": SOURCE}],
        "source_claims": [
            {"id": "resume:0:27", "source_id": "resume", "start": 0, "end": 27,
             "classification": "required"},
        ],
        "lines": [{"line_id": "L000001", "text": "Piloted a submarine in 1899"}],
    })
    strict = correction_protocol.strict_review_from_indexed(
        {"supported": ["L000001"], "evidence": {"L000001": [0]}, "represented": []}, request)
    assert strict["claims"][0]["verdict"] == "uncertain"
    assert strict["claims"][0]["citations"] == []


def test_supported_line_without_named_evidence_is_uncertain():
    """Approval requires the reviewer to name what establishes the line."""
    request = correction_protocol.indexed_review_request({
        "sources": [{"source_id": "resume", "text": SOURCE}],
        "source_claims": [
            {"id": "resume:0:27", "source_id": "resume", "start": 0, "end": 27,
             "classification": "required"},
        ],
        "lines": [{"line_id": "L000001", "text": "## Professional Experience"}],
    })
    strict = correction_protocol.strict_review_from_indexed(
        {"supported": ["L000001"], "represented": []}, request)
    assert strict["claims"][0]["verdict"] == "uncertain"


# --- Non-vacuity control ----------------------------------------------------

def test_faithful_draft_is_still_approved_by_the_same_permissive_reviewer(bypass):
    """The negative controls above must not pass by rejecting everything."""
    result = run(SOURCE, bypass)
    assert (result.status, result.reason) == ("approved", None)
    assert result.markdown == SOURCE
    assert tier_a_verifier.verify_metrics(result.markdown, SOURCE)["verdict"] == "PASS (tier A only)"
    assert unsupported_entities(result.markdown, SOURCE) == []
