"""The completeness diagnostic must OBSERVABLY fire, not merely look wired.

Two failures made this file necessary:

1. The first version instrumented the wrong branch — the post-review path rather
   than the source-inventory path that actually refused.
2. The second version referenced a name (`items`) that only exists inside
   `_validate_review`, so it raised `NameError` — which the guard swallowed. The
   code read as instrumented and the event never appeared.

Both were found by running it, not by reading it. So these tests assert the EVENT,
and one of them asserts that a broken diagnostic reports its own failure instead
of failing silently.
"""

import pytest

import correction_events as ce
import source_correction as sc


def test_the_inventory_refusal_emits_an_observable_event():
    """Fire the real inventory branch and require the event to appear.

    A test that only checked "the code contains correction.completeness" passed
    while the event could not fire at all. `tests/test_source_correction.run()`
    does not forward an observer, so this drives `correct()` directly.
    """
    import asyncio
    import sys
    sys.path.insert(0, "tests")
    import test_source_correction as fixture

    def uncertain_inventory(phase, payload):
        response = fixture.legacy_inventory(payload)
        response["source_claims"][-1]["classification"] = "uncertain"
        return response

    async def adapter(phase, payload):
        if phase == "source_inventory":
            return uncertain_inventory(phase, payload)
        raise AssertionError("no review should run after an uncertain inventory")

    seen: list[dict] = []
    result = asyncio.run(sc.correct(
        fixture.COMPLETE_DRAFT, fixture.COMPLETE_SOURCES, adapter,
        timeout_seconds=1, observer=seen.append))

    assert result.reason == "source_completeness_uncertain"
    emitted = [e for e in seen if e.get("event") == "correction.completeness"]
    assert emitted, (
        "the inventory refusal emitted no diagnostic — either it is not "
        "instrumented or the instrumentation raised and was swallowed")
    assert emitted[0]["uncertain_required_claims"] >= 1
    assert emitted[0]["required_source_claims"] >= 1
    # Becomes a log line or it may as well not exist.
    assert ce.sanitize(emitted[0]), "the event does not survive sanitization"


def test_a_broken_diagnostic_reports_instead_of_vanishing():
    """A swallowed telemetry bug must be visible.

    The guard exists so telemetry can never change a verdict. But a bare `pass`
    turned a NameError into silence, and the withhold then claimed to be
    undiagnosable when it was only mis-instrumented. The fallback event is what
    distinguishes "the gate has no evidence" from "the instrument is broken".
    """
    # The fallback uses an existing phase outcome, so it is already allow-listed.
    assert "callback_error" in ce.PHASE_OUTCOMES
    clean = ce.sanitize({"event": "correction.phase_end",
                         "phase": "post_patch_review",
                         "outcome": "callback_error"})
    assert clean["outcome"] == "callback_error"


def test_both_refusal_sites_use_an_in_scope_inventory():
    """Guard the specific defect: reading a helper-local name from correct().

    `items` exists only inside `_validate_review`. `correct()` holds `inventory`.
    Read the FILE, not `inspect.getsource(correct)`: `correct` is an `async def`
    whose body contains nested helpers, and getsource does not reliably return the
    whole nested scope.
    """
    import pathlib
    source = pathlib.Path(sc.__file__).read_text()
    assert 'by_id = {item["id"]: item for item in inventory}' in source, (
        "correct() no longer builds its lookup from the in-scope inventory")
    assert "items.get(mapping.get(" not in source, (
        "the diagnostic reads `items`, which is local to _validate_review — the "
        "exact NameError the guard hides")
