"""Offline product regressions for the deliberately literal attribution guard."""
import asyncio
import copy

import pytest

from source_correction import SourceDocument, correct
from tests.test_source_correction import (
    MATRIX_ANSWER_KEY,
    MATRIX_SOURCE,
    complete_inventory,
    complete_review,
)


def lying_review(payload, substitutions):
    """Return valid offsets/citations while reviewing substituted source text."""
    substituted = copy.deepcopy(payload)
    for line in substituted["lines"]:
        line["text"] = substitutions.get(line["text"], line["text"])
    response = complete_review(substituted)
    original = {line["line_id"]: line["text"] for line in payload["lines"]}
    for claim in response["claims"]:
        line_id = claim["line_id"]
        if original[line_id] != next(line["text"] for line in substituted["lines"]
                                     if line["line_id"] == line_id):
            old_end = claim["end"]
            claim["end"] = len(original[line_id])
            claim["id"] = f'{line_id}:{claim["start"]}:{claim["end"]}'
            for mapping in response["representations"]:
                for span in mapping["spans"]:
                    if span["line_id"] == line_id and span["end"] == old_end:
                        span["end"] = claim["end"]
    return response


def run(draft, sources, substitutions=None, *, repair=False):
    phases = []

    async def callback(phase, payload):
        phases.append(phase)
        if phase == "source_inventory":
            return complete_inventory(payload)
        if phase == "repair":
            return {"edits": [{"claim_id": payload["flagged_claim_ids"][0],
                               "replacement": ""}]}
        if repair and phase == "initial_review":
            shortened = copy.deepcopy(payload)
            target = next(line for line in shortened["lines"] if line["text"].endswith(" invented"))
            target["text"] = target["text"][:-len(" invented")]
            response = lying_review(shortened, substitutions or {})
            prefix = next(c for c in response["claims"] if c["line_id"] == target["line_id"])
            end = prefix["end"]
            response["claims"].append({"id": f'{target["line_id"]}:{end}:{end + 9}',
                                       "line_id": target["line_id"], "start": end,
                                       "end": end + 9, "verdict": "unsupported", "citations": []})
            return response
        return lying_review(payload, substitutions or {})

    return asyncio.run(correct(draft, sources, callback, timeout_seconds=1)), phases


def test_original_answer_key_can_approve():
    result, phases = run(MATRIX_ANSWER_KEY, MATRIX_SOURCE)
    assert (result.status, result.markdown, result.reason) == ("approved", MATRIX_ANSWER_KEY, None)
    assert phases == ["source_inventory", "initial_review"]


def test_original_adversarial_wrong_employer_is_corrected():
    original = "Alpha: improved accuracy 30% over 6 months"
    wrong = "Beta: improved accuracy 30% over 6 months"
    draft = MATRIX_ANSWER_KEY.replace(original, wrong)
    result, phases = run(draft, MATRIX_SOURCE, {wrong: original})
    assert (result.status, result.markdown, result.reason) == ("approved", MATRIX_ANSWER_KEY, None)
    assert phases == ["source_inventory", "initial_review"]
    assert "restored 1 source-owned record label" in result.commentary
    assert "30% over 6 months" in result.markdown and wrong not in result.markdown


@pytest.mark.parametrize("source_line,draft_line", [
    ("Alpha: engineer, 2019–2021", "Beta: engineer, 2019–2021"),
    ("Beta: engineer, 2022–Present", "Alpha: engineer, 2022–Present"),
    ("Alpha: improved accuracy 30% over 6 months", "BETA: improved accuracy 30% over 6 months"),
])
def test_exact_records_under_wrong_labels_are_corrected(source_line, draft_line):
    draft = MATRIX_ANSWER_KEY.replace(source_line, draft_line)
    result, _ = run(draft, MATRIX_SOURCE, {draft_line: source_line})
    assert result.status == "approved", result
    assert result.markdown == MATRIX_ANSWER_KEY
    assert "restored 1 source-owned record label" in result.commentary


def test_changed_record_body_is_still_not_silently_reassigned():
    # The label is corrigible, but a changed body is not silently equated with
    # the source record; the scripted reviewer cannot map it and withholds.
    draft = MATRIX_ANSWER_KEY.replace("Alpha: improved accuracy 30% over 6 months",
                                      "Beta: improved  accuracy 30% over 6 months")
    result, _ = run(draft, MATRIX_SOURCE)
    assert (result.status, result.reason, result.markdown) == (
        "withheld", "source_facts_missing", None)


def test_post_patch_review_still_checks_a_corrected_label():
    original = "Alpha: improved accuracy 30% over 6 months"
    wrong = "Beta: improved accuracy 30% over 6 months"
    draft = MATRIX_ANSWER_KEY.replace(original, wrong + " invented")
    result, phases = run(draft, MATRIX_SOURCE, {wrong: original}, repair=True)
    assert (result.status, result.markdown, result.reason) == ("approved", MATRIX_ANSWER_KEY, None)
    assert phases == ["source_inventory", "initial_review", "repair", "post_patch_review"]
    assert (result.review_calls, result.repair_calls) == (2, 1)
    assert "restored 1 source-owned record label" in result.commentary
    assert "revised 1 span(s)" in result.commentary


def test_ambiguous_multi_owner_record_is_withheld():
    sources = (SourceDocument(
        "candidate",
        "Alpha: Completed safety audit\nBeta: Completed safety audit\nGamma: Maintained records\n",
    ),)
    draft = "Gamma: Completed safety audit\nGamma: Maintained records\n"

    async def callback(phase, payload):
        if phase == "source_inventory":
            return complete_inventory(payload)
        # The reviewer lies: it cites a legitimate owner for the Gamma line, so
        # a citation alone cannot resolve which of the two owners is correct.
        claims = []
        for line in payload["lines"]:
            text = line["text"]
            claims.append({"id": f"{line['line_id']}:0:{len(text)}", "line_id": line["line_id"],
                           "start": 0, "end": len(text), "verdict": "supported",
                           "citations": []})
        for claim, text in zip(claims, ("Gamma: Completed safety audit", "Gamma: Maintained records")):
            if text == "Gamma: Completed safety audit":
                wanted = ("Alpha: Completed safety audit", "Beta: Completed safety audit")
            else:
                wanted = ("Gamma: Maintained records",)
            claim["citations"] = [
                {key: item[key] for key in ("source_id", "start", "end", "quote")}
                for item in payload["source_claims"] if item["quote"].strip() in wanted
            ]
        representations = [
            {"source_claim_id": item["id"], "status": "represented",
             "spans": [{"line_id": claims[0]["line_id"], "start": 0, "end": claims[0]["end"]}]}
            if item["quote"].strip().startswith(("Alpha:", "Beta:"))
            else {"source_claim_id": item["id"], "status": "represented",
                  "spans": [{"line_id": claims[1]["line_id"], "start": 0,
                             "end": claims[1]["end"]}]}
            for item in payload["source_claims"]
        ]
        return {"claims": claims, "representations": representations}

    result = asyncio.run(correct(draft, sources, callback, timeout_seconds=1))
    assert (result.status, result.reason, result.markdown) == (
        "withheld", "attribution_conflict", None)
    assert result.repair_calls == 0


def test_shared_exact_record_can_belong_to_both_explicit_labels():
    sources = (SourceDocument("candidate", "Alpha: engineer\nBeta: engineer\n"),)
    draft = "Alpha: engineer\nBeta: engineer\n"
    result, _ = run(draft, sources)
    assert (result.status, result.markdown) == ("approved", draft)


def test_shared_record_across_documents_is_not_restricted_to_first_owner():
    sources = (SourceDocument("a", "Alpha: engineer\n"), SourceDocument("b", "Beta: engineer\n"))
    draft = "Alpha: engineer\nBeta: engineer\n"
    result, _ = run(draft, sources)
    assert (result.status, result.markdown) == ("approved", draft)


def test_generic_section_renaming_is_not_treated_as_employer_attribution():
    sources = (SourceDocument("candidate", "Skills: Python\n"),)
    result, _ = run("Tools: Python\n", sources, {"Tools: Python": "Skills: Python"})
    assert (result.status, result.markdown) == ("approved", "Tools: Python\n")


@pytest.mark.parametrize("source,draft", [
    ("Core Competencies: Python, SQL\n", "Technical Skills: Python, SQL\n"),
    ("Acme Inc: engineer, 2020–2022\n", "Acme: engineer, 2020–2022\n"),
])
def test_unseen_alias_does_not_trigger_literal_attribution_guard(source, draft):
    sources = (SourceDocument("candidate", source),)
    result, _ = run(draft, sources, {draft.strip(): source.strip()})
    assert (result.status, result.markdown) == ("approved", draft)


def test_wrong_label_across_source_documents_is_corrected():
    sources = (SourceDocument("a", "Alpha: improved accuracy 30% over 6 months\n"),
               SourceDocument("b", "Beta: engineer\n"))
    original = "Alpha: improved accuracy 30% over 6 months"
    wrong = "Beta: improved accuracy 30% over 6 months"
    draft = wrong + "\nBeta: engineer\n"
    result, _ = run(draft, sources, {wrong: original})
    assert result.status == "approved"
    assert result.markdown == original + "\nBeta: engineer\n"
    assert "restored 1 source-owned record label" in result.commentary


def test_label_is_corrected_before_the_reviewer_receives_the_draft():
    original = "Alpha: improved accuracy 30% over 6 months"
    wrong = "Beta: improved accuracy 30% over 6 months"
    draft = MATRIX_ANSWER_KEY.replace(original, wrong)
    seen = []

    async def callback(phase, payload):
        if phase == "source_inventory":
            return complete_inventory(payload)
        seen.append([line["text"] for line in payload["lines"]])
        return complete_review(payload)

    result = asyncio.run(correct(draft, MATRIX_SOURCE, callback, timeout_seconds=1))
    assert result.status == "approved" and result.markdown == MATRIX_ANSWER_KEY
    assert seen == [[line for line in MATRIX_ANSWER_KEY.splitlines() if line.strip()]]


def test_paraphrased_wrong_record_is_outside_literal_guard():
    original = "Alpha: improved accuracy 30% over 6 months"
    paraphrased = "Beta: improved accuracy by 30% over six months"
    draft = MATRIX_ANSWER_KEY.replace(original, paraphrased)
    result, _ = run(draft, MATRIX_SOURCE, {paraphrased: original})
    # Characterizes a remaining semantic limitation, not a safe attribution.
    assert (result.status, result.markdown) == ("approved", draft)