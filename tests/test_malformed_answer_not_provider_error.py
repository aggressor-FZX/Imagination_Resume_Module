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
    # Matches both `except MalformedReviewAnswer:` and the `as exc` form, which
    # the bounded shape re-ask needs in order to read `exc.constraint`.
    malformed = source.find("except MalformedReviewAnswer")
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
    assert "raise MalformedReviewAnswer(" in source
    # The constraint identifier must be carried through, so the specific failed
    # check is recorded rather than only "malformed".
    assert 'getattr(exc, "constraint"' in source
    # And the re-raise list must include it, or the generic handler below
    # re-labels it as provider_error.
    assert "MalformedReviewAnswer," in source


def test_the_constraint_identifier_reaches_the_phase_event():
    """The specific failed check must be recorded, not just `malformed_output`.

    Measured 2026-11-07: a run withheld as `provider_error` and the specific
    constraint was unrecoverable after the fact. The reason code alone says only
    that the answer was unusable.
    """
    import inspect

    source = inspect.getsource(sc._invoke)
    assert "malformed_constraint" in source
    assert "constraint=malformed_constraint" in source
    # It must be initialised before the try block, or the `finally` raises
    # NameError on the success path.
    assert "malformed_constraint: str | None = None" in source


def test_the_exception_carries_a_constraint():
    exc = sc.MalformedReviewAnswer("initial_review", "duplicate_line_verdict")
    assert exc.constraint == "duplicate_line_verdict"
    assert exc.phase == "initial_review"
    # A default keeps every existing raise site working.
    assert sc.MalformedReviewAnswer().constraint == "unspecified"


def test_the_constraint_survives_the_event_sanitizer():
    """The identifier must reach the log, or it cannot be diagnosed.

    Measured 2026-11-08: three live runs withheld as `malformed_output` and the
    phase event carried no `constraint` key at all, because the sanitizer is an
    allow-list and neither the new outcome nor the new field was on it. The
    identifier existed in the exception and was dropped on the way to the log.
    """
    import correction_events as ce

    event = {"event": "correction.phase_end", "phase": "initial_review",
             "outcome": "malformed_answer",
             "constraint": "duplicate_line_verdict", "request_index": 2}
    clean = ce.sanitize(event)
    assert clean.get("outcome") == "malformed_answer"
    assert clean.get("constraint") == "duplicate_line_verdict"


def test_a_non_identifier_constraint_is_dropped():
    """The field is a closed-set identifier, never model text."""
    import correction_events as ce

    event = {"event": "correction.phase_end", "phase": "initial_review",
             "outcome": "malformed_answer",
             "constraint": "some model text with spaces", "request_index": 2}
    assert "constraint" not in ce.sanitize(event)



# ── The bridge contract must describe the payload it ships with ────────────

def test_the_bridge_contract_describes_the_fields_actually_sent():
    """A contract that names a field the payload lacks makes the id a guess.

    Measured 2026-11-08: the contract said each bridge came "with the candidate
    segment index that supports it", but the payload sends `id`, `skill` and
    `evidence` — no index. A reviewer reading the contract had to guess what to
    key its answer on, and one answered with an invented id, which the engine
    correctly refused as `bridge_id_unknown`.
    """
    import correction_protocol as cp

    contract = cp.BRIDGE_DECISION_CONTRACT
    # The fields the payload actually carries.
    for field in ("id", "skill", "evidence"):
        assert f"`{field}`" in contract, field
    # The field the payload does NOT carry must not be promised.
    assert "segment index" not in contract
    # And the id rule must be stated plainly.
    assert "Copy each `id` EXACTLY as given" in contract
    assert "use no other id" in contract


def test_the_unknown_id_refusal_is_retained():
    """The contract fix must not weaken the refusal.

    An invented id leaves the real bridge unreviewed, so approving would publish
    a label nobody judged. The refusal is the correct behaviour and stays.
    """
    import correction_protocol as cp

    request = {"bridges": [{"id": "b1", "skill": "trend analysis",
                            "evidence": "x"}]}
    response = {"bridge_decisions": {"bridge-99-invented": "accept"}}
    try:
        cp._bridge_answers(response, request)
    except cp.MalformedIndexedResponse as exc:
        assert exc.constraint == "bridge_id_unknown"
    else:
        raise AssertionError("an invented bridge id must still be refused")


def test_every_offered_bridge_must_be_answered():
    """Completeness is still required: an unanswered bridge is not an approval."""
    import correction_protocol as cp

    request = {"bridges": [{"id": "b1", "skill": "a", "evidence": "x"},
                           {"id": "b2", "skill": "b", "evidence": "y"}]}
    # Answering only one is accepted by the parser (the engine treats the
    # unanswered one as not-accepted), but the contract still demands both.
    answers = cp._bridge_answers({"bridge_decisions": {"b1": "accept"}}, request)
    assert set(answers) == {"b1"}
    assert "Answer for every bridge id you were given, exactly once" in \
        cp.BRIDGE_DECISION_CONTRACT
