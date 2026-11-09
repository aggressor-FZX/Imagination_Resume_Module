"""Regression tests built from the ACTUAL retained answers, not from invented shapes.

Every test here is anchored to a real response from
``results/comparison-20261109-0719/attempts.json``. They exist because the first
comparison reported 21 of 24 attempts as `operational_failure`, and the dominant
label `envelope_not_json` was wrong about almost every one of them:

  * 11 were **truncated** — `max_tokens: 1200` cut the answer mid-string;
  * 0 were a genuine parser failure on a well-formed answer;
  * several were presentation differences (markdown emphasis, the omission marker
    written four different ways) that the parser rejected as if they were content
    errors.

The correction matters for the metric, not just for tidiness: labelling a
truncation as a parser defect, and counting a correct `revise` as a false refusal,
makes a plumbing problem look like a semantic result.

A standing caution, restated because it is easy to forget once tests are green:
**parsing successfully is not judging correctly.** Several tests below assert that
a *correct* answer now parses; none asserts that an answer is semantically right.
That still requires the independent blind audit.
"""

import json
import pathlib

import pytest

import decision_contract as dc
from experiment.replay import (
    CURRENT_FORMAT,
    REVISED_FORMAT,
    classify,
    looks_truncated,
    requested_format,
)

RUN = pathlib.Path(__file__).resolve().parent.parent / "results" / "comparison-20261109-0719"


def _attempts():
    return json.loads((RUN / "attempts.json").read_text())


def _attempt(cell, case_id):
    for attempt in _attempts():
        if attempt["cell"] == cell and attempt["case_id"] == case_id:
            return attempt
    raise AssertionError(f"no retained attempt for {cell}/{case_id}")


def _sources_for(case_id):
    from experiment.cases import CASES
    return dict(next(c for c in CASES if c.case_id == case_id).sources)


# ── The truncation finding ──────────────────────────────────────────────────


def test_the_current_cell_used_the_parser_its_prompt_asked_for():
    """The `current` prompts request JSON; the model answered JSON.

    This is the check that settles the `envelope_not_json` question. If the parser
    had been the wrong one — e.g. the prose decision contract applied to a JSON
    prompt — the response would not begin with `{`. It does.
    """
    attempt = _attempt("current+advisory", "faithful_paraphrase")
    raw = attempt["provider_response"]
    assert raw.lstrip().startswith("{")
    assert '"claims"' in raw
    assert "decision:" not in raw.lower()
    assert requested_format("current+advisory") == CURRENT_FORMAT


def test_those_responses_are_truncated_not_malformed():
    """An unterminated JSON object is an OUTPUT-BUDGET problem.

    `max_tokens: 1200` was set in the runner, and the span envelope for a full
    résumé needs more. Reporting this as `envelope_not_json` blamed the parser for
    the provider's token limit.
    """
    attempt = _attempt("current+advisory", "faithful_paraphrase")
    truncated, why = looks_truncated(attempt["provider_response"], None)
    assert truncated, "the retained response ends mid-string"
    assert "Unterminated" in why or "closing brace" in why


def test_truncation_is_classified_as_truncation_across_the_current_cells():
    """Most `current` answers were cut off — but not all, and the difference matters.

    `current+advisory/missing_fact` returned a complete envelope (3,525 chars,
    closing brace present) and parsed correctly. Asserting that EVERY current
    answer was truncated would be the same over-generalisation that produced the
    original `envelope_not_json` label.
    """
    rows = [(a["cell"], a["case_id"],
             classify(a["provider_response"], a["cell"], None,
                      a.get("draft_after_trim", ""), {}))
            for a in _attempts() if a["cell"].startswith("current")]
    truncations = [r for r in rows if r[2]["category"] == "truncation"]
    complete = [r for r in rows if r[2]["category"] != "truncation"]
    assert len(truncations) == 11, len(truncations)
    assert len(complete) == 1, complete
    cell, case_id, result = complete[0]
    assert (cell, case_id) == ("current+advisory", "missing_fact")
    # The one complete answer is a reference refusal, not an operational failure.
    assert result["category"] == "semantic_refuse"


def test_a_provider_length_reason_is_honoured():
    truncated, why = looks_truncated('{"a": 1', "length")
    assert truncated and "length" in why


def test_complete_json_is_not_mistaken_for_truncated():
    truncated, _ = looks_truncated('{"claims": []}', None)
    assert not truncated


# ── Presentation differences must not read as content errors ────────────────


def test_bold_wrapped_decision_and_problem_words_parse():
    """Retained shape: `Decision: **revise**` / `problem: **unsupported**`."""
    raw = (
        "Decision: **revise**\n"
        "\n"
        "Issues:\n"
        "- passage: Installed panels across 6 properties.\n"
        "  problem: **unsupported**\n"
        "  sources: [doc-0]\n"
        "  why: the property count is not in this record\n"
        "  correction: Installed panels.\n"
    )
    parsed = dc.decide_from_text(raw, "## P\n- Installed panels across 6 properties.\n",
                                 {"doc-0": "Installed panels."})
    assert parsed.decision == dc.DECISION_REVISE
    assert parsed.issues[0].problem == dc.PROBLEM_UNSUPPORTED
    assert parsed.ok, parsed.structural_problems


@pytest.mark.parametrize("marker", ["(omitted)", "*(omitted)*", "**omitted**",
                                    "omitted", "(omit)", "OMITTED"])
def test_every_omission_marker_shape_normalises(marker):
    """All six were observed or are produced by the same decoration rule."""
    raw = (
        "Decision: revise\n"
        "\n"
        "Issues:\n"
        f"- passage: {marker}\n"
        "  problem: omitted\n"
        "  sources: [doc-0]\n"
        "  why: a sourced credential was dropped\n"
        "  correction: (omit)\n"
    )
    parsed = dc.decide_from_text(raw, "## P\n- x\n", {"doc-0": "- x\n"})
    assert parsed.issues[0].passage == dc.OMITTED_MARKER
    assert parsed.ok, parsed.structural_problems


def test_markup_stripping_does_not_alter_interior_emphasis():
    """Only a WHOLE-VALUE emphasis span is unwrapped.

    `*Role | 01/2020*` inside a passage must keep its asterisks, or the passage
    stops matching the draft and a correct review is refused.
    """
    assert dc._strip_markup_wrapping("**unsupported**") == "unsupported"
    # A value that merely CONTAINS emphasis is returned untouched, because the
    # passage must keep matching the draft character for character.
    assert dc._strip_markup_wrapping("*Role | 01/2020* — Fife, WA") == \
        "*Role | 01/2020* — Fife, WA"


# ── Unknown words, invented passages and unknown sources still fail ─────────


def test_an_unknown_problem_word_still_fails():
    raw = (
        "Decision: revise\n"
        "\n"
        "Issues:\n"
        "- passage: Installed panels across 6 properties.\n"
        "  problem: totallybogus\n"
        "  sources: [doc-0]\n"
        "  why: x\n"
        "  correction: Installed panels.\n"
    )
    parsed = dc.decide_from_text(raw, "## P\n- Installed panels across 6 properties.\n",
                                 {"doc-0": "Installed panels."})
    assert not parsed.ok
    assert any(p.startswith("problem_word_unknown") for p in parsed.structural_problems)


def test_an_invented_passage_still_fails():
    raw = (
        "Decision: revise\n"
        "\n"
        "Issues:\n"
        "- passage: A sentence the draft never contained.\n"
        "  problem: unsupported\n"
        "  sources: [doc-0]\n"
        "  why: x\n"
        "  correction: y\n"
    )
    parsed = dc.decide_from_text(raw, "## P\n- Installed panels.\n",
                                 {"doc-0": "Installed panels."})
    assert not parsed.ok
    assert any(p.startswith("passage_not_in_draft") for p in parsed.structural_problems)


def test_an_unknown_source_still_fails():
    raw = (
        "Decision: revise\n"
        "\n"
        "Issues:\n"
        "- passage: Installed panels.\n"
        "  problem: unsupported\n"
        "  sources: [doc-99]\n"
        "  why: x\n"
        "  correction: y\n"
    )
    parsed = dc.decide_from_text(raw, "## P\n- Installed panels.\n",
                                 {"doc-0": "Installed panels."})
    assert not parsed.ok
    assert any(p.startswith("sources_unresolved") for p in parsed.structural_problems)


# ── Omissions: handled explicitly, not by requiring absent text ────────────


def test_a_grounded_omission_is_actionable_without_a_replacement():
    """The intended workflow: the reviewer names the dropped SOURCE fact.

    An omission has no draft passage to quote by construction, so requiring the
    passage to appear in the draft is unsatisfiable. Grounding it in a real source
    is what makes `revise` actionable here.
    """
    raw = (
        "Decision: revise\n"
        "\n"
        "Issues:\n"
        "- passage: Maintain parts availability records for 7 service branches.\n"
        "  problem: omitted\n"
        "  sources: [candidate-supplement.txt]\n"
        "  why: the record is absent from the draft\n"
        "  correction: (omit)\n"
    )
    parsed = dc.decide_from_text(
        raw, "## P\n- something else\n",
        {"candidate-supplement.txt":
         "- Maintain parts availability records for 7 service branches.\n"})
    assert parsed.ok, parsed.structural_problems
    assert parsed.issues[0].reports_omission


def test_an_omission_quoting_text_in_no_source_still_fails():
    """Grounding is the control: invention cannot hide behind `omitted`."""
    raw = (
        "Decision: revise\n"
        "\n"
        "Issues:\n"
        "- passage: An invented requirement nobody wrote.\n"
        "  problem: omitted\n"
        "  sources: [candidate-supplement.txt]\n"
        "  why: x\n"
        "  correction: (omit)\n"
    )
    parsed = dc.decide_from_text(
        raw, "## P\n- something else\n",
        {"candidate-supplement.txt": "- Maintain parts availability records.\n"})
    assert not parsed.ok
    assert any(p.startswith("passage_not_in_draft") for p in parsed.structural_problems)


# ── The corrected metric ───────────────────────────────────────────────────


def test_a_revise_is_not_counted_as_a_false_refusal():
    """The metric correction, pinned.

    A `revise` is the contract WORKING. Counting it as a false refusal would
    penalise the reviewer for finding a real defect, and would make a correct
    system look broken.
    """
    from experiment.replay import _category_for_decision
    assert _category_for_decision(dc.DECISION_REVISE) == "semantic_revise"
    assert _category_for_decision(dc.DECISION_APPROVE) == "approve"
    assert _category_for_decision(dc.DECISION_CANNOT_ASSESS) == "semantic_refuse"
    assert _category_for_decision("") == "operational"


def test_a_reference_refusal_is_a_refusal_not_an_operational_failure():
    envelope = json.dumps({
        "claims": [{"id": "L1:0:5", "line_id": "L1", "start": 0, "end": 5,
                    "verdict": "supported",
                    "citations": [{"source_id": "nope.txt", "start": 0, "end": 5,
                                   "quote": "xxxxx"}]}],
        "representations": [],
    })
    result = classify(envelope, "current+advisory", None, "## P\n- xxxxx\n",
                      {"doc-0": "xxxxx"})
    assert result["category"] == "semantic_refuse"
    assert result["first_failure"] == "reference_invalid"


# ── Case validity, which the first run silently got wrong ──────────────────


def test_every_approve_case_draft_covers_its_sources():
    """Held-out cases must be VALID before their expectations mean anything.

    Discovered while replaying: every v1 case supplied both source documents but a
    draft that omitted the `Ridgeline Equipment Services` record from the
    supplement. A faithful reviewer MUST flag that omission, so the three cases
    expecting `approve` were unsatisfiable and the model was right every time it
    refused them.

    This guard now checks the ACTIVE fixture (`cases_v2`, the corrected complete
    two-document baseline). The v1 fixture is deliberately NOT repaired in place:
    it is the record of what actually ran, and the frozen archive depends on it.
    `test_v1_fixture_is_recorded_as_invalid` below asserts it is still recognised
    as invalid, so the history cannot be quietly rewritten.
    """
    from experiment.cases_v2 import CASES_V2
    import re as _re

    offenders = []
    for case in CASES_V2:
        if case.expectation != "approve":
            continue
        employers = set(_re.findall(r"\*\*([A-Z][A-Za-z ]+)\*\* --",
                                    " ".join(case.sources.values())))
        missing = [e for e in employers if e not in case.draft_before_trim]
        if missing:
            offenders.append((case.case_id, missing))
    assert offenders == [], (
        "these approve-cases drop a source record, so they cannot be approved "
        f"faithfully: {offenders}")


def test_v1_fixture_is_recorded_as_invalid():
    """The v1 cases must still be recognisably invalid, not retro-fixed.

    If someone "repairs" `experiment/cases.py` in place, the frozen archive in
    `results/comparison-20261109-0719` stops corresponding to the fixture that
    produced it and the original findings become unverifiable. The defect belongs
    on record; the correction belongs in `cases_v2`.
    """
    from experiment.cases import CASES as V1
    import re as _re

    offenders = []
    for case in V1:
        if case.expectation != "approve":
            continue
        employers = set(_re.findall(r"\*\*([A-Z][A-Za-z ]+)\*\* --",
                                    " ".join(case.sources.values())))
        missing = [e for e in employers if e not in case.draft_before_trim]
        if missing:
            offenders.append(case.case_id)
    assert sorted(offenders) == ["faithful_paraphrase", "supported_method",
                                 "transferable_skill"]
