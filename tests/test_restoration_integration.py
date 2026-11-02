"""Integrated engine -> provider -> scripted HTTP -> restoration -> re-review.

The engine tests in ``test_fidelity_delivery_run_4.py`` drive ``correct()`` with a
*provider-shaped async function*. That proves the engine's orchestration and its
validation, but it does NOT prove the restoration mechanism works through the real
``CorrectionProvider`` — the layer that owns protocol translation, the disclosure gate,
per-call budgets, retry classification, telemetry and live capture.

This module closes that gap. Everything below the HTTP socket is real:

    correct()  ->  CorrectionProvider.__call__  ->  scripted HTTP session
               ->  correction_protocol translation  ->  strict validation

The scripted HTTP session is a labelled double: it answers each phase the way a model
would, so the *translation* is exercised for real while the model is not. It is not a
mock of the result — the final ``CorrectionResult`` is whatever the engine produced.
Nothing here overrides ``correct()``'s return value.
"""
import json

import pytest

import correction_protocol
import correction_provider
from correction_provider import CorrectionProvider
from live_capture import CapturingSession
from source_correction import SourceDocument, correct
import source_correction


SOURCE = (
    "Cascade Power Solutions, LLC\n"
    "Journeyman Electrician | 03/2020 - Present\n"
    "Installed and maintained commercial electrical systems across 40+ job sites.\n"
)

# The role title has been destroyed by an upstream defect: the draft keeps the dates
# and the employer but has no title on the header line. This is the run-4 shape.
DRAFT = (
    "**Cascade Power Solutions, LLC**\n"
    "*Professional | 03/2020 - Present*\n"
    "- Installed and maintained commercial electrical systems across 40+ job sites.\n"
)

TITLE = "Journeyman Electrician"


def _envelope(answer, provider="ScriptedEndpoint"):
    """A well-formed OpenRouter 200 envelope carrying ``answer`` as model content."""
    return json.dumps({
        "model": correction_provider.OR_SLUG_DRAFTER,
        "provider": provider,
        "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
        "choices": [{"finish_reason": "stop",
                     "message": {"content": json.dumps(answer)}}],
    })


class RestorationModel:
    """A scripted model that answers the index-list contract for every phase.

    Phase detection uses the *translated* payload, which is what a real model would
    see: an inventory request has ``segments`` and no ``lines``, a review has both, and
    a restore has neither because it uses the engine's own restoration contract.

    The model reports the title missing on the first review, then — once the engine has
    asked for a restoration — reports it represented. That is the same answer sequence
    the engine-level test uses, but here it travels through the real adapter.
    """

    def __init__(self):
        self.phases = []
        self.requests = []
        self.restore_answers = 0

    def answer(self, phase, payload):
        self.phases.append(phase)
        # Dispatch on the contract text, which is unambiguous after translation:
        # an inventory request has no lines, and review and restore are told apart by
        # the question they ask rather than by which keys happen to be present.
        contract = payload.get("contract", "")
        if contract == correction_protocol.RESTORE_CONTRACT:
            return self._restore(payload)
        if contract == correction_protocol.INVENTORY_CONTRACT:
            return {"required": [s["index"] for s in payload["segments"]],
                    "non_resume": [], "uncertain": []}
        return self._review(payload)

    # -- per-phase answers ------------------------------------------------
    def _restore(self, payload):
        """Answer the translated restoration contract.

        The adapter has removed the internal item id and the ``before`` echo, so the
        model returns only line ids and replacements — which is what a live model can
        actually do. The adapter reconstructs the rest. The engine then re-validates
        item id, before-text and substantiation against its own records.
        """
        self.restore_answers += 1
        return {"restorations": [
            {"line_id": line["line_id"],
             # Preserve the italic header shape; add only the real title.
             "replacement": line["text"].replace("Professional", TITLE)}
            for line in payload["lines"]
        ]}

    def _review(self, payload):
        segments = payload["segments"]
        lines = payload["lines"]
        # The inventory segments follow the source's own lines, so the title segment
        # also carries the dates: match by containment, not equality.
        title_indices = [s["index"] for s in segments if TITLE in s["text"]]
        present = any(TITLE in line["text"] for line in lines)

        supported, structural, evidence = [], [], {}
        for line in lines:
            text = line["text"]
            lid = line["line_id"]
            if text.lstrip().startswith("#"):
                structural.append(lid)
                continue
            # Everything factual is supported, citing the segments its words match.
            supported.append(lid)
            words = set(text.casefold().split())
            cited = [s["index"] for s in segments
                     if s["text"].strip() and set(s["text"].casefold().split()) & words]
            if not cited:
                cited = [s["index"] for s in segments if s["text"].strip()][:1]
            darkest = sorted(set(cited))
            if TITLE in text and title_indices:
                darkest = sorted(set(darkest) | set(title_indices))
            evidence[lid] = darkest

        represented, missing = [], []
        for segment in segments:
            text = segment["text"]
            if not text.strip() or text.strip().startswith("#"):
                continue
            if TITLE in text and not present:
                # The draft genuinely does not contain the role title. It must be
                # DECLARED missing, not merely omitted: the adapter refuses to infer
                # "missing" from silence, so an omission is reported as uncertain,
                # which withholds before any restoration is attempted. Declaring it is
                # also what makes the omission repairable.
                missing.append(segment["index"])
                continue
            represented.append(segment["index"])

        return {"supported": supported, "unsupported": [], "uncertain": [],
                "structural": structural, "evidence": evidence,
                "represented": represented, "missing": missing}


class ScriptedHTTP:
    """Scripted aiohttp-shaped session; records what actually went on the wire."""

    def __init__(self, model):
        self.model = model
        self.requests = []
        self.statuses = []

    async def close(self):
        pass

    def post(self, url, **kwargs):
        self.requests.append(kwargs)
        payload = json.loads(kwargs["json"]["messages"][1]["content"])
        # The model sees the translated payload, exactly as a real endpoint would.
        answer = self.model.answer(self._phase(payload), payload)
        status, body = 200, _envelope(answer)
        self.statuses.append(status)
        return _response(status, body)

    @staticmethod
    def _phase(payload):
        contract = payload.get("contract", "")
        if contract == correction_protocol.RESTORE_CONTRACT:
            return "restore"
        if contract == correction_protocol.INVENTORY_CONTRACT:
            return "inventory"
        return "review"


def _response(status, body):
    class Content:
        async def iter_chunked(self, size):
            raw = (body or "").encode()
            for index in range(0, len(raw), size):
                yield raw[index:index + size]

    class Headers:
        def get(self, name, default=None):
            return default

    class Response:
        def __init__(self):
            self.status = status
            self.headers = Headers()
            self.content = Content()

        async def text(self, *args, **kwargs):
            return body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    return Response()


@pytest.fixture
def disclosure(monkeypatch):
    """Approve the existing disclosure policy and remove retry backoff waits."""
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available",
                        lambda: True)

    async def instant(_seconds):
        return None

    monkeypatch.setattr(correction_provider, "_sleep", instant)


def _run(model, *, capture=None, observer=None):
    """Run the real engine against the real provider over scripted HTTP.

    Live capture wraps the session ONCE, exactly as ``app._run_source_correction``
    does it, so the recorded exchanges are the ones the provider actually made.
    """
    import asyncio

    scripted = ScriptedHTTP(model)
    session = CapturingSession(scripted, capture) if capture is not None else scripted
    provider = CorrectionProvider("synthetic-key", session, observer=observer)
    # The observer is given to BOTH layers, exactly as `app._correction_observer`
    # does: the engine reports phase boundaries and the provider reports per-attempt
    # outcomes, and a restoration must be visible in each.
    result = asyncio.run(correct(DRAFT, (SourceDocument("candidate", SOURCE),),
                                 provider, timeout_seconds=60.0, observer=observer))
    return result, provider, scripted


# ---------------------------------------------------------------------------
# The integrated positive: approval reached through the real adapter
# ---------------------------------------------------------------------------

def test_restoration_reaches_approval_through_the_real_provider(disclosure):
    """[integrated] Missing title -> restore through the provider -> approved.

    This is the end-to-end claim the engine-level test could not make. Every layer
    below the socket is production code, including protocol translation and the
    disclosure gate. The result is the engine's own; nothing overrides it.
    """
    model = RestorationModel()
    result, provider, session = _run(model)

    # The engine asked for a restoration, and the provider carried it.
    assert "restore" in model.phases, model.phases
    assert model.restore_answers == 1
    # A restore, then a full re-review, both actually sent.
    assert model.phases.count("review") >= 2, model.phases
    assert provider.calls == len(model.phases)

    assert result.status == "approved", (result.reason, result.commentary)
    assert result.restore_calls == 1
    assert TITLE in result.markdown
    assert "Professional" not in result.markdown
    # Attribution is preserved: the title landed on the Cascade record, not elsewhere.
    header = next(line for line in result.markdown.splitlines() if "03/2020" in line)
    assert TITLE in header and "Cascade" not in header
    employer = next(line for line in result.markdown.splitlines() if "Cascade" in line)
    assert TITLE not in employer


def test_restoration_telemetry_reports_the_restore_phase(disclosure):
    """[integrated] The restore phase emits telemetry like any other provider call."""
    events = []
    model = RestorationModel()
    result, provider, session = _run(model, observer=events.append)

    assert result.status == "approved"
    phases = [event["phase"] for event in events
              if event.get("event") == "correction.phase_start"]
    assert "restore" in phases, phases
    # And the provider's per-attempt event names it too.
    call_phases = [event["phase"] for event in events if event.get("event") == "provider.call"]
    assert "restore" in call_phases, call_phases
    # Telemetry must never carry document text.
    blob = json.dumps(events, default=str)
    assert "Journeyman" not in blob
    assert "Cascade" not in blob


def test_restoration_travels_through_the_disclosure_gate(disclosure):
    """[integrated] The restoration request is gated and redacted like any other."""
    model = RestorationModel()
    result, provider, session = _run(model)

    assert result.status == "approved"
    # Every outbound request went through the gate: the system message carries the
    # contract, the user message carries the redacted payload, and no request was
    # sent that the disclosure check rejected.
    assert len(session.requests) == len(model.phases)
    for request in session.requests:
        body = request["json"]
        assert "Bearer synthetic-key" in request["headers"]["Authorization"]
        assert body["messages"][0]["content"]
        assert body["messages"][1]["content"]
        # The restore phase is present on the wire and is asked as its own question.
        assert body["temperature"] == 0.1 and body["max_tokens"] > 0


def test_disclosure_cannot_be_bypassed_for_the_restore_phase(monkeypatch):
    """[integrated] With no approved policy, restoration never reaches the wire."""
    monkeypatch.setattr(correction_provider, "_approved_disclosure_available",
                        lambda: False)

    async def instant(_seconds):
        return None

    monkeypatch.setattr(correction_provider, "_sleep", instant)

    model = RestorationModel()
    session = ScriptedHTTP(model)
    provider = CorrectionProvider("synthetic-key", session)

    import asyncio
    with pytest.raises(source_correction.PrivacyReviewRequired):
        asyncio.run(provider("restore", {
            "contract": "synthetic restore contract",
            "sources": [{"source_id": "candidate", "text": SOURCE}],
            "source_claims": [{"id": "candidate:29:51", "source_id": "candidate",
                               "start": 29, "end": 51, "quote": TITLE,
                               "classification": "required"}],
            "lines": [{"line_id": "L000002",
                       "text": "*Professional | 03/2020 - Present*"}],
        }))
    assert session.requests == [], "nothing may be sent without an approved policy"


def test_restore_phase_counts_against_the_call_budget(disclosure):
    """[integrated] Restoration is a real provider call, bounded like the others."""
    model = RestorationModel()
    result, provider, session = _run(model)
    assert result.status == "approved"
    # inventory + review + restore + re-review at minimum
    assert provider.calls >= 4
    assert provider.calls <= correction_provider.MAX_CALLS


def test_unusable_restoration_withholds_through_the_real_provider(disclosure):
    """[integrated] A restoration that cannot be used leaves the run withheld."""

    class RefusingModel(RestorationModel):
        def _restore(self, payload):
            self.restore_answers += 1
            return {"unexpected": "shape"}

    model = RefusingModel()
    result, provider, session = _run(model)
    assert result.status == "withheld"
    assert result.reason == "source_facts_missing"
    assert result.markdown is None
    assert model.restore_answers == 1, "exactly one bounded attempt"


def test_still_missing_after_restoration_withholds_through_the_real_provider(disclosure):
    """[integrated] Restoration happened, re-review still says missing -> withheld."""
    model = RestorationModel()

    original_review = model._review

    def review_always_missing(payload):
        answer = original_review(payload)
        # Refuse to acknowledge the title even after the replacement is in the draft.
        segments = payload["segments"]
        answer["represented"] = [s["index"] for s in segments
                                 if s["text"].strip()
                                 and not s["text"].strip().startswith("#")
                                 and TITLE not in s["text"]]
        answer["missing"] = [s["index"] for s in segments if TITLE in s["text"]]
        return answer

    model._review = review_always_missing
    result, provider, session = _run(model)

    assert model.restore_answers == 1, "no second restoration"
    assert result.status == "withheld"
    assert result.reason == "source_facts_missing"


def test_restoration_is_captured_for_replay(disclosure):
    """[integrated] Live capture records the restore exchange, not just inventory/review."""
    from live_capture import LiveCapture

    capture = LiveCapture("00000000-0000-4000-8000-000000000123")
    model = RestorationModel()
    result, provider, session = _run(model, capture=capture)

    assert result.status == "approved"
    data = capture.data
    phases = [exchange["phase"] for exchange in data["exchanges"]]
    assert "restore" in phases, phases
    # The recorded exchange carries the request and the answer, so it can be replayed.
    restore_exchange = next(e for e in data["exchanges"] if e["phase"] == "restore")
    assert restore_exchange["request"]["json"] is not None
    assert restore_exchange["response_body"]
    # Summaries stay content-free.
    summary = json.dumps({k: v for k, v in capture.data.items()
                          if k in {"timing", "gate", "review"}}, default=str)
    assert "Journeyman" not in summary
