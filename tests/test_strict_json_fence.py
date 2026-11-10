"""`_strict_json` must accept a fenced object WITHOUT weakening strictness.

Measured 2026-11-09 against the live contract
(`results/live-contract-probe/probe.json`): `google/gemma-4-31b-it` returned a
complete and correct source inventory with right ids, right offsets and quotes
matching the source slices — wrapped in a ```json fence. `_strict_json` rejected it
at character 0, which would have withheld every correction.

So the fence is stripped. Everything else stays strict: duplicate keys, non-finite
numbers and any surrounding prose still fail. A parser that tolerated those would
convert a malformed answer into an accepted one.
"""

from __future__ import annotations

import pytest

from correction_provider import _strict_json


# ── The tolerance ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    '```json\n{"a": 1}\n```',
    '```\n{"a": 1}\n```',
    '```json\r\n{"a": 1}\r\n```',
    '  ```json\n{"a": 1}\n```  ',
    '{"a": 1}',
])
def test_fenced_and_bare_objects_both_parse(raw):
    assert _strict_json(raw) == {"a": 1}


def test_the_observed_live_shape_parses():
    """The exact failure: a real inventory, fenced."""
    raw = (
        '```json\n'
        '{\n'
        '  "source_claims": [\n'
        '    {\n'
        '      "id": "candidate-main.txt:0:24",\n'
        '      "source_id": "candidate-main.txt",\n'
        '      "start": 0,\n'
        '      "end": 24,\n'
        '      "quote": "## Professional Summary",\n'
        '      "classification": "resume"\n'
        '    }\n'
        '  ]\n'
        '}\n'
        '```'
    )
    parsed = _strict_json(raw)
    assert parsed["source_claims"][0]["start"] == 0
    assert parsed["source_claims"][0]["quote"] == "## Professional Summary"


# ── Strictness preserved ────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    'Here is the JSON: {"a": 1}',          # prose before
    '{"a": 1}\nHope that helps!',          # prose after
    '```json\n{"a": 1}\n``` and more',     # fence plus trailing prose
    '```json\n{"a": 1}\n',                 # unterminated fence
    'not json at all',
    '',
])
def test_surrounding_prose_still_fails(raw):
    with pytest.raises(Exception):
        _strict_json(raw)


def test_duplicate_keys_still_fail():
    with pytest.raises(ValueError, match="duplicate key"):
        _strict_json('```json\n{"a": 1, "a": 2}\n```')


def test_non_finite_numbers_still_fail():
    with pytest.raises(ValueError):
        _strict_json('```json\n{"a": NaN}\n```')


def test_an_array_is_still_not_an_object():
    """The engine requires a dict; the fence strip must not change that."""
    assert _strict_json('```json\n[1, 2]\n```') == [1, 2]
    # ...and the caller's own dict check is what rejects it, not this helper.
    assert not isinstance(_strict_json('```json\n[1, 2]\n```'), dict)


def test_non_string_input_is_passed_through_unchanged():
    """A non-string must fail the same way it always did, not crash the stripper."""
    with pytest.raises(TypeError):
        _strict_json(None)
