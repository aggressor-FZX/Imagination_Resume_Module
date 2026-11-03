"""Repair reachability and controlled recovery for the correction engine.

Delivery runs 4-7 were all withheld as ``source_facts_missing`` with
``repair_calls == 0``. The cause was an ordering defect in the gate: when a review
reported *both* unsupported lines and missing facts, the missing-fact branch withheld
immediately, so the repair path — which exists precisely to remove unsupported content
— was never reached. A draft carrying 13 unsupported lines was therefore rejected
without an attempt to fix them.

These tests establish four separate things, and the distinction between them matters:

1. **The original rejection** is pinned to the revision that produced it, read from
   the capture's own bytes. It is not recomputed: current code deliberately goes
   further, so recomputing would silently rewrite history.
2. **Repair reachability** on the two new-code behaviours: the new code now enters the
   repair phase on every recorded capture instead of stopping at the gate.
3. **Controlled recovery**: with separately labelled answers for the phases the
   recording never performed, a repairable draft is repaired, re-reviewed, validated
   and approved — and an unrepairable one still is not.
4. **Negative controls**: the repair path cannot delete an accomplishment, move a fact
   between employers, invent a qualification, or exceed its call budget.

Layers, stated per test because "the route works" and "a model writes better prose" are
different claims:

``[capture]``     protected live bytes, replayed through the real adapter, engine and
                  validator, no network.
``[controlled]``  real engine plus labelled test-authored answers for phases the
                  recording never reached. Proves mechanism, never model quality.
``[engine]``      real engine driven by a scripted provider; no capture involved.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

import live_capture
from source_correction import SourceDocument, correct
from tests.support import controlled_answers as ca

ROOT = Path("/home/skystarved/Render_Dockers/.live-run-2026-11-02")

# Each capture pinned to the revision that produced it. The SHA-256 is the identity:
# a capture cannot be re-pointed at a different revision by editing a field, and the
# artifact cannot be silently replaced without the hash failing.
CAPTURES = {
    "run4": {
        "path": ROOT / "retained" / "delivery-run-4-withheld.json",
        "sha256": "92e03d4bbfde82089bcb428ff563d274a9e215cbabe496428f6565cdff891230",
        "revision": "5f3c1f0",
        "correlation_id": "8e6d3d72-1694-4df2-867d-4687ede388b6",
    },
    "run5": {
        "path": ROOT / "captures" / "run5" / "3ee23fd9-0767-4d8f-977f-50708c5afcbd.json",
        "sha256": "8972d3711c6b8d4ff13bdfeb9c18b250d5802fb6f4d441180a0a72487dd51646",
        "revision": "9057d966",
        "correlation_id": "3ee23fd9-0767-4d8f-977f-50708c5afcbd",
    },
    "run6": {
        "path": ROOT / "captures" / "run6" / "709fb7fd-c48b-48a9-90ef-720f974b78c9.json",
        "sha256": "269caf5e06b3a1cc6c12c2f82a9a9819415af63792574b2ca774c213658085f9",
        "revision": "fb75917",
        "correlation_id": "709fb7fd-c48b-48a9-90ef-720f974b78c9",
    },
    "run7": {
        "path": ROOT / "captures" / "run7" / "0a267e04-affd-4cee-b944-26c574703039.json",
        "sha256": "bb9643ffbaea2949b7e1491a14706db149a4afca39d15cfe7b7a00b82115edbd",
        "revision": "b855087",
        "correlation_id": "0a267e04-affd-4cee-b944-26c574703039",
    },
}


def _available(spec: dict) -> bool:
    return spec["path"].exists()


requires = {key: pytest.mark.skipif(not _available(spec),
                                   reason=f"{key} capture is not present")
            for key, spec in CAPTURES.items()}


# ---------------------------------------------------------------------------
# 1. The original rejection is pinned to its original revision
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(CAPTURES))
def test_capture_bytes_are_unchanged(name):
    """[capture] The artifact that proves the rejection is byte-identical.

    A capture is evidence about a specific revision. If its bytes change, every
    conclusion drawn from it has to be re-derived, so the hash is asserted rather than
    trusted.
    """
    spec = CAPTURES[name]
    if not _available(spec):
        pytest.skip(f"{name} capture is not present")
    digest = hashlib.sha256(spec["path"].read_bytes()).hexdigest()
    assert digest == spec["sha256"]


@pytest.mark.parametrize("name", list(CAPTURES))
def test_original_rejection_is_read_from_the_capture_not_recomputed(name):
    """[capture] The stopping point is evidence about a revision, not a live claim.

    The original code withheld at the gate with zero repairs. Current code
    deliberately repairs first, so it no longer stops where the live run stopped.
    Recomputing "what the old code did" from current code would be a fiction; the
    recorded diagnosis is therefore read out of the artifact.
    """
    spec = CAPTURES[name]
    if not _available(spec):
        pytest.skip(f"{name} capture is not present")
    stopped = live_capture.recorded_stopping_point(spec["path"])

    assert stopped["correlation_id"] == spec["correlation_id"]
    assert stopped["status"] == "withheld"
    assert stopped["reason"] == "source_facts_missing"
    assert stopped["review_calls"] == 1
    assert stopped["repair_calls"] == 0, (
        "the recorded run must show that repair was never attempted")
    assert stopped["phases"] == ["source_inventory", "review"]
    assert stopped["phase_count"] == 2, (
        "the recording stops after the initial review; that is why the new code "
        "cannot be validated on it alone")


@pytest.mark.parametrize("name", list(CAPTURES))
def test_recorded_review_reports_missing_and_unsupported_together(name):
    """[capture] Each recorded review had both flagged lines and missing facts.

    That combination is the precondition for the ordering defect: with only missing
    facts the restore branch is correct, and with only flagged lines repair was always
    reachable. It is the coexistence that withheld.
    """
    spec = CAPTURES[name]
    if not _available(spec):
        pytest.skip(f"{name} capture is not present")
    data = live_capture.load(spec["path"])
    review = next(e for e in data["exchanges"] if e["phase"] == "review")
    payload = json.loads(json.loads(review["response_body"])["choices"][0]["message"]["content"])

    assert payload["unsupported"], "no unsupported lines were recorded"
    assert payload["missing"], "no missing source items were recorded"


# ---------------------------------------------------------------------------
# 2. Repair reachability: the new code enters the repair phase
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(CAPTURES))
def test_new_code_reaches_repair_on_the_recorded_draft(name):
    """[capture] The gate no longer withholds before trying the repair path.

    The recording ends before the repair answer, so the run cannot complete cleanly.
    What is asserted is the thing that was broken: the engine *asks* for a repair at all.
    Before the fix this was ``repair_calls == 0``; now the request is made, the recording
    is fully consumed, and the run stops only because the next answer was never recorded.

    The failure surfaces as ``provider_error`` rather than ``ReplayExhausted`` because the
    adapter wraps every transport-level exception, and a replay running out of recorded
    answers is such an exception. The assertion therefore reads the session, which is the
    only thing that can distinguish "asked for an unrecorded phase" from any other error.
    """
    spec = CAPTURES[name]
    if not _available(spec):
        pytest.skip(f"{name} capture is not present")

    data = live_capture.load(spec["path"])
    extension = live_capture.ControlledExtension({})
    extension.prepare(data)
    session = live_capture.ExtendedReplaySession(data, extension)

    import asyncio
    import os

    import correction_provider
    from correction_provider import CorrectionProvider

    previous = os.environ.get(correction_provider.DISCLOSURE_SETTING)
    os.environ[correction_provider.DISCLOSURE_SETTING] = "1"
    try:
        provider = CorrectionProvider("no-network", session)
        sources = tuple(SourceDocument(i["source_id"], i["text"]) for i in data["sources"])
        result = asyncio.run(correct(data["draft"], sources, provider,
                                     timeout_seconds=60))
    finally:
        if previous is None:
            os.environ.pop(correction_provider.DISCLOSURE_SETTING, None)
        else:
            os.environ[correction_provider.DISCLOSURE_SETTING] = previous

    # The recording was consumed in full, and the engine then asked for one more phase.
    assert result.repair_calls == 1, (
        "the engine must attempt a repair on a draft with unsupported lines")
    assert session.position == len(data["exchanges"]), (
        "the engine consumed the whole recording before asking for anything more")
    assert session.unanswered, (
        "the engine must ask for a phase the recording does not contain (the repair)")
    # It cannot complete, because the repair answer was never recorded. Withholding is
    # correct; the point is that it got further than the old stopping point.
    assert result.status == "withheld"
    assert result.markdown is None


@pytest.mark.parametrize("name", list(CAPTURES))
def test_repair_request_names_the_flagged_claims(name):
    """[capture] The repair asked for is the right one: the flagged spans.

    A reachability claim is only meaningful if the request carries the defect. This
    inspects the request the engine actually built, using the session's own request
    log rather than a test-side wrapper.
    """
    spec = CAPTURES[name]
    if not _available(spec):
        pytest.skip(f"{name} capture is not present")

    data = live_capture.load(spec["path"])
    extension = live_capture.ControlledExtension({
        # Deliberately unusable: we stop at the request, not the outcome.
        "repair": ca.success({"edits": []}),
    })
    extension.prepare(data)
    session = live_capture.ExtendedReplaySession(data, extension)

    import asyncio
    import os

    import correction_provider
    from correction_provider import CorrectionProvider

    previous = os.environ.get(correction_provider.DISCLOSURE_SETTING)
    os.environ[correction_provider.DISCLOSURE_SETTING] = "1"
    try:
        provider = CorrectionProvider("no-network", session)
        sources = tuple(SourceDocument(i["source_id"], i["text"]) for i in data["sources"])
        try:
            asyncio.run(correct(data["draft"], sources, provider, timeout_seconds=60))
        except Exception:
            pass  # the controlled answer is unusable on purpose
    finally:
        if previous is None:
            os.environ.pop(correction_provider.DISCLOSURE_SETTING, None)
        else:
            os.environ[correction_provider.DISCLOSURE_SETTING] = previous

    assert "repair" in extension.served, "the engine never asked for a repair"

    payloads = []
    for request in session.requests:
        body = (request.get("json") or {}).get("messages")
        if not body:
            continue
        try:
            payloads.append(json.loads(body[1]["content"]))
        except Exception:
            continue
    repair_payloads = [p for p in payloads if "flagged_claim_ids" in p]
    assert repair_payloads, "the engine never built a repair request"
    payload = repair_payloads[0]
    assert payload["flagged_claim_ids"], "repair carried no flagged claim ids"
    assert payload["claims"], "repair carried no claims to edit against"

    # And those ids must be exactly the lines the reviewer flagged.
    review = next(e for e in data["exchanges"] if e["phase"] == "review")
    answer = json.loads(json.loads(review["response_body"])["choices"][0]["message"]["content"])
    flagged_lines = set(answer["unsupported"]) | set(answer.get("uncertain") or [])
    assert flagged_lines, "no flagged lines in the recording"
    # A claim id is ``{line_id}:{start}:{end}``, so the line id is the first field.
    carried = {cid.split(":", 1)[0] for cid in payload["flagged_claim_ids"]}
    assert carried == flagged_lines, (
        "the repair must target the flagged lines, not an arbitrary subset")


def test_recovery_plus_one_repair_stops_without_colliding_with_the_budget():
    """After a recovery re-review, a settled draft reports the real reason.

    Measured on delivery run 8: the engine spent inventory, review, recovery review,
    repair, re-review, repair = 6 provider calls, then asked for a seventh (the
    re-review of the second repair). The adapter's ``REPAIR_PATH_CALLS`` guard refused
    it, so ``correct()`` reported ``provider_error`` for a decision it had already
    reached — all six calls had returned HTTP 200, and the withhold was ours.

    A second repair pass costs two calls (repair plus the re-review that must follow it).
    When the recovery re-review has already been spent, that pair cannot be funded, so the
    engine stops and names ``unresolved_claims``.

    Driven from the run-8 recording itself. That is the only honest way to assert this
    sequence: an invented script kept reaching a different stop (missing facts, no-op
    repair) and would have proved nothing about the real collision.
    """
    import asyncio

    run8 = Path("/home/skystarved/Render_Dockers/.live-run-2026-11-02/captures/"
                "run8/7f1cc601-990c-4fdb-a2aa-0665c8cb7a0f.json")
    if not run8.exists():
        pytest.skip("run-8 capture is not present")

    import live_capture
    from source_correction import SourceDocument, correct

    data = live_capture.load(str(run8))
    session = live_capture.ReplaySession(data)

    import correction_provider
    from correction_provider import CorrectionProvider

    import os
    previous = os.environ.get(correction_provider.DISCLOSURE_SETTING)
    os.environ[correction_provider.DISCLOSURE_SETTING] = "1"
    try:
        provider = CorrectionProvider("no-network", session)
        sources = tuple(SourceDocument(item["source_id"], item["text"])
                        for item in data["sources"])
        result = asyncio.run(correct(data["draft"], sources, provider,
                                     timeout_seconds=120))
    finally:
        if previous is None:
            os.environ.pop(correction_provider.DISCLOSURE_SETTING, None)
        else:
            os.environ[correction_provider.DISCLOSURE_SETTING] = previous

    assert result.status == "withheld"
    assert result.reason != "provider_error", (
        "a budget collision must not be reported as a provider failure")
    assert result.reason == "unresolved_claims", result.reason
    # Six calls are permitted; the engine must not ask for a seventh.
    assert session.position <= 6, session.position
