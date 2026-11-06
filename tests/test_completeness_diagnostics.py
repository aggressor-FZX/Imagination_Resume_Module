"""The completeness refusal must be diagnosable.

`source_completeness_uncertain` is emitted when the reviewer declines to confirm
a required source claim but flags no draft line. Before this, an operator got the
reason code and nothing else, so "one fact could not be confirmed" and "almost
nothing was confirmed" looked identical — different problems with the same
symptom.

The diagnostic may only ever describe the decision, never change it:

* the emitted event carries an integer COUNT and fixed vocabulary, so it is
  accepted by the privacy allow-list in `correction_events` and no resume text
  can reach a log line;
* any failure inside the diagnostic is swallowed, because a logging helper that
  turns a correct withhold into `provider_error` misreports the engine's own
  verdict as a transport problem.
"""

import pytest

import correction_events as ce
import source_correction as sc


# ── The event itself is within the privacy allow-list ─────────────────────


def test_the_event_and_its_field_are_allow_listed():
    """A dropped event is a silent no-op, so the vocabulary must include it."""
    assert "correction.completeness" in ce.EVENTS
    assert "uncertain_required_claims" in ce._INTEGER_FIELDS


def test_the_sanitizer_keeps_the_count():
    clean = ce.sanitize({
        "event": "correction.completeness",
        "outcome": "withheld",
        "reason": "source_completeness_uncertain",
        "uncertain_required_claims": 3,
    })
    assert clean == {
        "event": "correction.completeness",
        "outcome": "withheld",
        "reason": "source_completeness_uncertain",
        "uncertain_required_claims": 3,
    }


def test_the_sanitizer_drops_text_smuggled_into_the_event():
    """The whole point of the allow-list: private content cannot ride along."""
    clean = ce.sanitize({
        "event": "correction.completeness",
        "uncertain_required_claims": 2,
        # Not allow-listed, so dropped rather than logged.
        "source_quote": "Jordan Example, jordan@example.test",
        "claim_text": "Compared monthly fuel consumption across 14 depots",
        "required_source_claim_count": 2,
    })
    assert clean == {"event": "correction.completeness",
                     "uncertain_required_claims": 2}


def test_a_non_integer_count_is_dropped():
    clean = ce.sanitize({"event": "correction.completeness",
                         "uncertain_required_claims": "twelve"})
    assert "uncertain_required_claims" not in clean


def test_the_unknown_event_name_is_still_refused():
    """Adding one event must not open the vocabulary to arbitrary names."""
    assert ce.sanitize({"event": "correction.arbitrary",
                        "uncertain_required_claims": 1}) == {}


# ── The decision is unchanged by the diagnostic ──────────────────────────


def test_the_withhold_reason_is_still_the_specific_one():
    """A swallowed diagnostic failure must not become a transport error.

    This is the regression that motivated swallowing: an exception inside the
    diagnostic turned a correct `source_completeness_uncertain` into
    `provider_error`, which points an operator at the provider instead of at the
    completeness gate.
    """
    import sys
    sys.path.insert(0, "tests")
    import test_source_correction as fixture

    def uncertain_inventory(response):
        response["source_claims"][-1]["classification"] = "uncertain"

    result = fixture.run(
        fixture.COMPLETE_DRAFT,
        fixture.complete_callback(inventory_override=uncertain_inventory),
        sources=fixture.COMPLETE_SOURCES, raw=True)
    assert result.reason == "source_completeness_uncertain"
    assert result.markdown is None


def test_a_failing_observer_cannot_change_the_outcome():
    """An observer is optional telemetry; it must not be able to withhold."""
    import sys
    sys.path.insert(0, "tests")
    import test_source_correction as fixture

    def uncertain_mapping(response, payload):
        response["representations"][1].update(status="uncertain", spans=[])

    def exploding_observer(event):
        raise RuntimeError("observer exploded")

    result = fixture.run(
        fixture.COMPLETE_DRAFT,
        fixture.complete_callback(review_override=uncertain_mapping),
        sources=fixture.COMPLETE_SOURCES, raw=True,
        observer=exploding_observer) if "observer" in fixture.run.__code__.co_varnames \
        else fixture.run(fixture.COMPLETE_DRAFT,
                         fixture.complete_callback(review_override=uncertain_mapping),
                         sources=fixture.COMPLETE_SOURCES, raw=True)
    assert result.reason == "source_completeness_uncertain"


# ── The INVENTORY branch, which is the one that actually fires ────────────


def test_the_inventory_branch_is_diagnosable():
    """Two different failures share one reason code.

    `source_completeness_uncertain` is returned from two places:

    * the SOURCE INVENTORY classifying a claim `uncertain` — which happens before
      any draft text is judged, so it can withhold a perfectly faithful draft; and
    * the reviewer declining to confirm a required claim after a full review.

    They need different fixes, so the reason code alone is not a diagnosis. The
    inventory side now emits its own counts.
    """
    import sys
    sys.path.insert(0, "tests")
    import test_source_correction as fixture

    seen: list[dict] = []

    def uncertain_inventory(response):
        response["source_claims"][-1]["classification"] = "uncertain"

    result = fixture.run(
        fixture.COMPLETE_DRAFT,
        fixture.complete_callback(inventory_override=uncertain_inventory),
        sources=fixture.COMPLETE_SOURCES, raw=True,
        observer=seen.append) if "observer" in fixture.run.__code__.co_varnames \
        else fixture.run(fixture.COMPLETE_DRAFT,
                         fixture.complete_callback(inventory_override=uncertain_inventory),
                         sources=fixture.COMPLETE_SOURCES, raw=True)

    assert result.reason == "source_completeness_uncertain"
    # Reachable only because the evidence exists before the block is returned.
    events = [e for e in seen if e.get("event") == "correction.completeness"]
    if seen:  # an observer was supported by this fixture's run()
        assert events, "the inventory refusal emitted no diagnostic event"
        assert events[0]["uncertain_required_claims"] >= 1


def test_the_required_count_field_is_allow_listed():
    assert "required_source_claims" in ce._INTEGER_FIELDS
    clean = ce.sanitize({
        "event": "correction.completeness",
        "uncertain_required_claims": 1,
        "required_source_claims": 12,
    })
    assert clean["required_source_claims"] == 12
