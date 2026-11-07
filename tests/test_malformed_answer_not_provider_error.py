"""A served-but-unusable answer is not a transport failure.

Measured 2026-11-07 on a live Imaginative run against ``realistic_candidate``.
The run withheld as ``provider_error``, and the log looked like an outage:

```
"event": "provider.call", "http_status": 200, "outcome": "ok",
"phase": "initial_review", "provider": "DeepInfra"
"event": "correction.phase_end", "outcome": "callback_error"
"event": "correction.end", "reason": "provider_error"
```

Both provider calls returned 200 with ``outcome: ok``. The reviewer had returned
JSON that could not be mapped onto the engine's strict review shape, and the
adapter's generic ``except Exception`` swallowed that as ``provider_error``.

``UnjudgedLines`` was already mapped to ``incomplete_review``. Its sibling
``MalformedIndexedResponse`` — the same base class — was not, so a duplicate line
id or an out-of-range segment index was reported as a network fault. That
misattribution sends an operator to check OpenRouter credentials when the real
answer is "the reviewer returned an unusable answer".

These tests pin the mapping, not the model's behaviour: a malformed answer must
surface as ``malformed_output``, and the phase event must name it.
"""

import pytest

import correction_protocol
import source_correction as sc


def test_malformed_review_answer_is_its_own_exception():
    """It must not be a subclass of the transport exceptions.

    If it were, an adapter that catches ``ProviderUnavailable`` would silently
    treat a validation failure as availability and retry it.
    """
    assert issubclass(sc.MalformedReviewAnswer, ValueError)
    for transport in (sc.ProviderCallTimeout, sc.ProviderRateLimited,
                      sc.ProviderUnavailable, sc.ProviderRejectedRequest):
        assert not issubclass(sc.MalformedReviewAnswer, transport)


def test_malformed_indexed_response_is_the_base_of_unjudged_lines():
    """The two siblings share a base, which is why both needed handling."""
    assert issubclass(correction_protocol.UnjudgedLines,
                      correction_protocol.MalformedIndexedResponse)


def test_engine_maps_a_malformed_answer_to_malformed_output():
    """The user-facing reason must say the output was malformed.

    ``provider_error`` tells the operator to look at the provider; the truth is
    that the provider answered and we rejected the answer.
    """
    import inspect

    source = inspect.getsource(sc._correct_body)
    malformed = source.find("except MalformedReviewAnswer:")
    assert malformed != -1, "the engine no longer maps MalformedReviewAnswer"
    # The handler must precede the generic catch-all that follows it, or the
    # generic one wins and the reason regresses to `provider_error`. Search for
    # the NEXT generic handler after the malformed one, not the first in the
    # function: `_correct_body` has several earlier, unrelated ones.
    generic = source.find("except Exception:", malformed)
    assert generic != -1
    assert malformed < generic, (
        "MalformedReviewAnswer must be caught before the generic handler")
    assert 'blocked("malformed_output")' in source[malformed:generic]


def test_phase_event_names_a_malformed_answer():
    """The phase event must not report `callback_error` for a served answer."""
    import inspect

    source = inspect.getsource(sc._invoke)
    assert 'outcome = "malformed_answer"' in source, (
        "a malformed answer is still reported as callback_error")


def test_provider_raises_the_mapped_exception_not_a_generic_one():
    """The adapter must translate the protocol error, not let it fall through."""
    import inspect

    import correction_provider

    source = inspect.getsource(correction_provider.CorrectionProvider.__call__)
    assert "except correction_protocol.MalformedIndexedResponse" in source
    assert "raise MalformedReviewAnswer(phase)" in source
    # And the re-raise list must include it, or the generic handler below
    # re-labels it as provider_error.
    assert "MalformedReviewAnswer," in source
