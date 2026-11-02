"""Synthetic tests only: callbacks are local test doubles, never providers."""
import asyncio
import copy

import pytest

from source_correction import SourceDocument, correct


SOURCE = (SourceDocument(
    "resume-a",
    "Alex Example led diagnostics at Acme and improved accuracy by 30%. "
    "Alex attended North College. A coworker earned a degree at West University.",
),)
DRAFT = (
    "# Alex Example\r\n\r\n"
    "## Summary\r\n"
    "Improved diagnostic accuracy by 30% in 6 months.\r\n"
    "## Skills\r\n"
    "Diagnostics\r\n"
    "## Education\r\n"
    "Attended North College\r\n"
)


def inventory_item(source_id, start, end, text, classification):
    return {"id": f"{source_id}:{start}:{end}", "source_id": source_id,
            "start": start, "end": end, "quote": text[start:end],
            "classification": classification}


def legacy_inventory(payload):
    # These legacy tests exercise syntax and control flow, not semantic completeness.
    items = []
    for source in payload["sources"]:
        text, sid = source["text"], source["source_id"]
        marker = next((value for value in ("Alex Example", "École North", "José") if value in text), text)
        start, end = text.index(marker), text.index(marker) + len(marker)
        if start:
            items.append(inventory_item(sid, 0, start, text, "non_resume"))
        items.append(inventory_item(sid, start, end, text, "required"))
        if end < len(text):
            items.append(inventory_item(sid, end, len(text), text, "non_resume"))
    return {"source_claims": items}


def add_legacy_representations(response, inventory):
    if type(response) is not dict or type(response.get("claims")) is not list:
        return response
    mappings = []
    for item in inventory:
        status, spans = item["classification"], []
        if status == "required":
            status = "missing"
            for c in response["claims"]:
                if type(c) is not dict or c.get("verdict") != "supported":
                    continue
                for cite in c.get("citations", []):
                    if (type(cite) is dict and cite.get("source_id") == item["source_id"]
                            and type(cite.get("start")) is int and type(cite.get("end")) is int
                            and cite["start"] <= item["start"] and cite["end"] >= item["end"]):
                        status = "represented"
                        spans = [{key: c[key] for key in ("line_id", "start", "end")}]
                        break
                if spans:
                    break
        mappings.append({"source_claim_id": item["id"], "status": status, "spans": spans})
    response["representations"] = mappings
    return response


def run(markdown, callback, *, sources=SOURCE, timeout=1, raw=False):
    async def adapter(phase, payload):
        if phase == "source_inventory":
            return legacy_inventory(payload)
        response = await callback(phase, payload)
        if phase in {"initial_review", "post_patch_review"}:
            return add_legacy_representations(response, payload["source_claims"])
        return response

    return asyncio.run(correct(markdown, sources, callback if raw else adapter, timeout_seconds=timeout))


def citation(quote="Alex Example", source=SOURCE[0]):
    a = source.text.index(quote)
    return {"source_id": source.source_id, "start": a, "end": a + len(quote), "quote": quote}


def claim(line, start, end, verdict, citations=None):
    return {"id": f"{line['line_id']}:{start}:{end}", "line_id": line["line_id"],
            "start": start, "end": end, "verdict": verdict,
            "citations": [] if citations is None else citations}


def review(payload, flagged=None):
    """The test double supplies decisions explicitly; it is not a semantic model."""
    flagged = flagged or {}
    claims = []
    for line in payload["lines"]:
        text = line["text"]
        if text in flagged:
            fragment, verdict = flagged[text]
            start = text.index(fragment)
            if start:
                claims.append(claim(line, 0, start, "supported", [citation()]))
            claims.append(claim(line, start, start + len(fragment), verdict))
            if start + len(fragment) < len(text):
                claims.append(claim(line, start + len(fragment), len(text), "supported", [citation()]))
        elif text in ("## Summary", "## Skills", "## Education"):
            claims.append(claim(line, 0, len(text), "structural"))
        else:
            claims.append(claim(line, 0, len(text), "supported", [citation()]))
    return {"claims": claims}


def flagged_for_duration(payload):
    return review(payload, {"Improved diagnostic accuracy by 30% in 6 months.": (" in 6 months", "unsupported")})


def test_duration_patch_preserves_supported_fact_and_every_untouched_byte():
    calls = []

    async def callback(phase, payload):
        calls.append((phase, copy.deepcopy(payload)))
        if phase == "repair":
            return {"edits": [{"claim_id": payload["flagged_claim_ids"][0], "replacement": ""}]}
        if phase == "initial_review":
            return flagged_for_duration(payload)
        return review(payload)

    result = run(DRAFT, callback)
    assert result.status == "approved"
    assert result.markdown == DRAFT.replace(" in 6 months", "")
    assert "30%" in result.markdown
    assert [phase for phase, _ in calls] == ["initial_review", "repair", "post_patch_review"]
    assert [line["text"] for line in calls[-1][1]["lines"]] == [
        "# Alex Example", "## Summary", "Improved diagnostic accuracy by 30%.",
        "## Skills", "Diagnostics", "## Education", "Attended North College",
    ]
    assert result.coverage.level == "model_assisted_not_proof"
    assert result.coverage.line_count == 7
    assert "not proof" in result.commentary
    assert (result.review_calls, result.repair_calls) == (2, 1)


@pytest.mark.parametrize("draft,fragment", [
    ("# CEO of Acme", "CEO"),
    ("## Summary\nOwned the coworker's degree", "coworker's degree"),
    ("## Skills\nQuantum computing", "Quantum computing"),
    ("## Education\nGraduated from West University", "Graduated from West University"),
])
def test_model_flags_ownership_and_education_claims_even_with_related_source_words(draft, fragment):
    phases = []

    async def callback(phase, payload):
        phases.append(phase)
        if phase == "repair":
            # Unresolvable; incomplete response is not a reason to publish.
            return {"edits": []}
        line = next(line for line in payload["lines"] if fragment in line["text"])
        return review(payload, {line["text"]: (fragment, "uncertain")})

    result = run("# Alex Example\n" + draft, callback)
    assert result.status == "withheld" and result.markdown is None
    assert result.reason == "malformed_output" and phases == ["initial_review", "repair"]


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(source_id="job-ad"),
    lambda c: c.update(start=-1),
    lambda c: c.update(end=99999),
    lambda c: c.update(start=True),
    lambda c: c.update(quote="invented quote"),
    lambda c: c.update(quote=""),
    lambda c: c.update(extra="unexpected"),
])
def test_invalid_citations_fail_closed(mutation):
    async def callback(phase, payload):
        response = review(payload)
        mutation(response["claims"][0]["citations"][0])
        return response

    result = run("# Alex Example", callback)
    assert result.status == "withheld" and result.reason == "malformed_output"
    assert result.markdown is None and result.coverage is None


@pytest.mark.parametrize("mutation", [
    lambda r: r["claims"].pop(),
    lambda r: r["claims"].append(copy.deepcopy(r["claims"][0])),
    lambda r: r["claims"][0].update(id="made-up"),
    lambda r: r["claims"][0].update(start=1),
    lambda r: r["claims"][0].update(end=1),
    lambda r: r["claims"][0].update(verdict="pass"),
    lambda r: r["claims"][0].update(citations=[]),
    lambda r: r.update(extra=True),
])
def test_review_schema_and_exhaustive_unique_span_ids(mutation):
    async def callback(phase, payload):
        response = review(payload)
        mutation(response)
        return response

    result = run("# Alex Example\n## Education\nAttended North College", callback)
    assert result.reason == "malformed_output" and result.markdown is None


@pytest.mark.parametrize("edit_type", ["supported", "missing", "duplicate", "newline", "noop", "extra"])
def test_bad_edits_never_trigger_retry(edit_type):
    calls = []

    async def callback(phase, payload):
        calls.append(phase)
        if phase != "repair":
            return flagged_for_duration(payload)
        cid = payload["flagged_claim_ids"][0]
        edit = {"claim_id": cid, "replacement": ""}
        if edit_type == "supported":
            edit["claim_id"] = next(c["id"] for c in payload["claims"] if c["verdict"] == "supported")
        elif edit_type == "missing":
            return {"edits": []}
        elif edit_type == "duplicate":
            return {"edits": [edit, edit]}
        elif edit_type == "newline":
            edit["replacement"] = "\nnew line"
        elif edit_type == "noop":
            edit["replacement"] = " in 6 months"
        elif edit_type == "extra":
            edit["explanation"] = "private content"
        return {"edits": [edit]}

    result = run(DRAFT, callback)
    assert result.reason == "malformed_output" and result.markdown is None
    assert calls == ["initial_review", "repair"]


def test_repeated_unsupported_edits_exhaust_two_repairs_and_three_reviews():
    calls = []

    async def callback(phase, payload):
        calls.append(phase)
        if phase == "repair":
            return {"edits": [{"claim_id": payload["flagged_claim_ids"][0],
                               "replacement": "invented" if calls.count("repair") == 1 else "still invented"}]}
        text = next(line["text"] for line in payload["lines"] if line["text"].startswith("Improved"))
        fragment = " in 6 months" if phase == "initial_review" else (
            "still invented" if "still invented" in text else "invented")
        return review(payload, {text: (fragment, "unsupported")})

    result = run(DRAFT, callback)
    assert result.reason == "unresolved_claims" and result.markdown is None
    assert calls == ["initial_review", "repair", "post_patch_review", "repair", "post_patch_review"]
    assert (result.review_calls, result.repair_calls) == (3, 2)


@pytest.mark.parametrize("timeout_phase", ["initial_review", "repair", "post_patch_review"])
def test_deadline_stops_all_further_calls(timeout_phase):
    calls = []

    async def callback(phase, payload):
        calls.append(phase)
        if phase == timeout_phase:
            await asyncio.Event().wait()
        if phase == "repair":
            return {"edits": [{"claim_id": payload["flagged_claim_ids"][0], "replacement": ""}]}
        return flagged_for_duration(payload) if phase == "initial_review" else review(payload)

    result = run(DRAFT, callback, timeout=0.02)
    # The reason names the phase so a production timeout is diagnosable.
    assert result.reason == f"timeout_{timeout_phase}" and result.markdown is None
    assert calls[-1] == timeout_phase
    assert calls.count("repair") <= 1


@pytest.mark.parametrize("bad_response", [None, "{}", [], {"claims": None}])
def test_malformed_evaluator_cannot_be_bypassed(bad_response):
    async def callback(phase, payload):
        return bad_response

    result = run(DRAFT, callback)
    assert result.reason == "malformed_output" and result.repair_calls == 0


def test_provider_exception_is_sanitized_and_not_logged(caplog):
    async def callback(phase, payload):
        raise RuntimeError("Alex Example private provider message")

    result = run(DRAFT, callback)
    assert result.reason == "provider_error" and result.markdown is None
    assert "Alex" not in repr(result) and "Alex" not in caplog.text


def test_source_inputs_are_explicit_and_not_job_ads():
    calls = []

    async def callback(phase, payload):
        calls.append(phase)
        return review(payload)

    for sources in [(), (SourceDocument("ad", "Alex", "job_ad"),),
                    (SourceDocument("x", "Alex"), SourceDocument("x", "Alex")),
                    (SourceDocument("", "Alex"),), (SourceDocument("x", " "),)]:
        result = run(DRAFT, callback, sources=sources)
        assert result.reason == "invalid_input" and result.markdown is None
    assert calls == []


def test_callback_mutation_does_not_change_source_or_review_state():
    async def callback(phase, payload):
        response = review(payload)
        payload["sources"][0]["text"] = "Alex Example altered"
        payload["lines"][0]["text"] = "changed"
        return response

    result = run("# Alex Example", callback)
    assert result.status == "approved" and result.markdown == "# Alex Example"
    assert SOURCE[0].text.startswith("Alex Example led")


def test_post_patch_review_is_required_and_schema_checked():
    async def callback(phase, payload):
        if phase == "repair":
            return {"edits": [{"claim_id": payload["flagged_claim_ids"][0], "replacement": ""}]}
        if phase == "post_patch_review":
            return {"claims": []}
        return flagged_for_duration(payload)

    result = run(DRAFT, callback)
    assert result.reason == "malformed_output" and result.markdown is None
    assert (result.review_calls, result.repair_calls) == (2, 1)


def test_all_structural_is_not_accepted_as_a_resume():
    async def callback(phase, payload):
        return {"claims": [claim(line, 0, len(line["text"]), "structural") for line in payload["lines"]]}

    assert run("# Example", callback).reason == "no_supported_claims"


def test_second_patch_must_also_be_completely_re_reviewed():
    calls = []

    async def callback(phase, payload):
        calls.append(phase)
        if phase == "repair":
            replacement = " in a year" if calls.count("repair") == 1 else ""
            return {"edits": [{"claim_id": payload["flagged_claim_ids"][0], "replacement": replacement}]}
        if phase == "initial_review":
            return flagged_for_duration(payload)
        if calls.count("post_patch_review") == 1:
            text = next(line["text"] for line in payload["lines"] if "in a year" in line["text"])
            return review(payload, {text: (" in a year", "uncertain")})
        return review(payload)

    result = run(DRAFT, callback)
    assert result.status == "approved" and result.markdown == DRAFT.replace(" in 6 months", "")
    assert calls == ["initial_review", "repair", "post_patch_review", "repair", "post_patch_review"]
    assert (result.review_calls, result.repair_calls) == (3, 2)


def test_deadline_is_cumulative_across_phases():
    calls = []

    async def callback(phase, payload):
        calls.append(phase)
        if phase == "initial_review":
            await asyncio.sleep(0.03)
            return flagged_for_duration(payload)
        if phase == "repair":
            await asyncio.sleep(0.2)
        raise AssertionError("no other call is allowed")

    result = run(DRAFT, callback, timeout=0.08)
    assert result.reason == "timeout_repair" and result.markdown is None
    assert calls == ["initial_review", "repair"]


def test_self_cancelled_callback_is_withheld():
    async def callback(phase, payload):
        raise asyncio.CancelledError

    result = run(DRAFT, callback)
    assert result.reason == "provider_error" and result.markdown is None


def test_unicode_offsets_in_distinct_source_documents_are_exact():
    sources = (SourceDocument("one", "José Example"), SourceDocument("two", "José attended École North"))

    async def callback(phase, payload):
        line1, line2 = payload["lines"]
        return {"claims": [claim(line1, 0, len(line1["text"]), "supported", [citation("José", sources[0])]),
                           claim(line2, 0, len(line2["text"]), "supported", [citation("École North", sources[1])])]}

    result = run("# José Example\nÉcole North", callback, sources=sources)
    assert result.status == "approved" and result.coverage.line_count == 2


COMPLETE_SOURCES = (
    SourceDocument("work", "Alex Example\nEngineer at Acme\nEngineer at Beta\n"),
    SourceDocument("school", "Alex Example attended North College\nCoworker attended West University\n"),
)
COMPLETE_DRAFT = (
    "# Alex Example\n## Experience\nEngineer at Acme\nEngineer at Beta\n"
    "## Education\nAlex Example attended North College\n"
)


def complete_inventory(payload):
    assert set(payload) == {"contract", "sources"}  # No draft reaches the inventory phase.
    items = []
    for source in payload["sources"]:
        position = 0
        for text in source["text"].splitlines(keepends=True):
            end = position + len(text)
            classification = "non_resume" if text.startswith("Coworker") else "required"
            items.append(inventory_item(source["source_id"], position, end,
                                        source["text"], classification))
            position = end
    return {"source_claims": items}


def complete_review(payload, overrides=None):
    overrides = overrides or {}
    source_by_id = {s["source_id"]: s["text"] for s in payload["sources"]}
    items = payload["source_claims"]
    claims = []
    for line in payload["lines"]:
        text = line["text"]
        if text.startswith("##"):
            claims.append(claim(line, 0, len(text), "structural"))
            continue
        item = next((item for item in items if item["classification"] == "required"
                     and item["quote"].strip() == text.lstrip("# ")), None)
        if item is None:
            claims.append(claim(line, 0, len(text), "unsupported"))
            continue
        cite = {key: item[key] for key in ("source_id", "start", "end", "quote")}
        assert source_by_id[cite["source_id"]][cite["start"]:cite["end"]] == cite["quote"]
        claims.append(claim(line, 0, len(text), "supported", [cite]))
    mappings = []
    for item in items:
        status, spans = item["classification"], []
        if status == "required":
            found = next((c for c in claims if any(
                cite["source_id"] == item["source_id"] and cite["start"] <= item["start"]
                and cite["end"] >= item["end"] for cite in c["citations"])), None)
            if found:
                status = "represented"
                spans = [{key: found[key] for key in ("line_id", "start", "end")}]
            else:
                status = "missing"
        status = overrides.get(item["quote"].strip(), status)
        if status != "represented":
            spans = []
        mappings.append({"source_claim_id": item["id"], "status": status, "spans": spans})
    return {"claims": claims, "representations": mappings}


def complete_callback(review_override=None, inventory_override=None):
    async def callback(phase, payload):
        if phase == "source_inventory":
            result = complete_inventory(payload)
            if inventory_override:
                inventory_override(result)
            return result
        assert phase in {"initial_review", "post_patch_review"}
        result = complete_review(payload)
        if review_override:
            review_override(result, payload)
        return result
    return callback


def test_independent_inventory_and_exact_representations_allow_complete_resume():
    calls = []
    async def callback(phase, payload):
        calls.append(phase)
        if phase == "source_inventory":
            return complete_inventory(payload)
        return complete_review(payload)

    result = run(COMPLETE_DRAFT, callback, sources=COMPLETE_SOURCES, raw=True)
    assert result.status == "approved" and result.markdown == COMPLETE_DRAFT
    assert calls == ["source_inventory", "initial_review"]
    assert (result.review_calls, result.repair_calls) == (1, 0)


@pytest.mark.parametrize("removed", [
    "Engineer at Beta\n", "## Experience\nEngineer at Acme\nEngineer at Beta\n",
    "## Education\nAlex Example attended North College\n",
])
def test_omitted_employer_or_section_is_withheld(removed):
    result = run(COMPLETE_DRAFT.replace(removed, ""), complete_callback(),
                 sources=COMPLETE_SOURCES, raw=True)
    assert (result.status, result.reason, result.markdown) == ("withheld", "source_facts_missing", None)
    assert result.repair_calls == 0


def test_multiple_sources_swapped_education_is_not_candidate_education():
    draft = COMPLETE_DRAFT.replace("Alex Example attended North College", "Coworker attended West University")
    result = run(draft, complete_callback(), sources=COMPLETE_SOURCES, raw=True)
    assert result.reason == "source_facts_missing" and result.markdown is None


@pytest.mark.parametrize("mutation", [
    lambda r: r["source_claims"].pop(),
    lambda r: r["source_claims"].append(copy.deepcopy(r["source_claims"][0])),
    lambda r: r["source_claims"][0].update(id="fake"),
    lambda r: r["source_claims"][0].update(start=True),
    lambda r: r["source_claims"][0].update(end=1),
    lambda r: r["source_claims"][0].update(quote="fabricated"),
    lambda r: r["source_claims"][0].update(source_id="school"),
    lambda r: r["source_claims"][0].update(classification="irrelevant"),
    lambda r: r["source_claims"][0].update(extra=1),
    lambda r: r.update(extra=1),
])
def test_source_inventory_strictly_partitions_each_immutable_document(mutation):
    result = run(COMPLETE_DRAFT, complete_callback(inventory_override=mutation),
                 sources=COMPLETE_SOURCES, raw=True)
    assert result.reason == "malformed_source_inventory" and result.markdown is None
    assert (result.review_calls, result.repair_calls) == (0, 0)


@pytest.mark.parametrize("mutation", [
    lambda r, p: r["representations"].pop(),
    lambda r, p: r["representations"].append(copy.deepcopy(r["representations"][0])),
    lambda r, p: r["representations"][0].update(source_claim_id="invented"),
    lambda r, p: r["representations"][0].update(status="non_resume", spans=[]),
    lambda r, p: r["representations"][0].update(spans=[]),
    lambda r, p: r["representations"][0]["spans"][0].update(line_id="L999999"),
    lambda r, p: r["representations"][0]["spans"][0].update(start=True),
    lambda r, p: r["representations"][0]["spans"].append(copy.deepcopy(r["representations"][0]["spans"][0])),
    lambda r, p: r["representations"][0].update(spans=[{"line_id": "L000002", "start": 0, "end": 13}]),
    lambda r, p: r["representations"][-1].update(status="represented", spans=r["representations"][0]["spans"]),
    lambda r, p: r["claims"][0]["citations"][0].update(source_id="school"),
])
def test_fabricated_or_incomplete_representation_fails_closed(mutation):
    result = run(COMPLETE_DRAFT, complete_callback(review_override=mutation),
                 sources=COMPLETE_SOURCES, raw=True)
    assert result.reason == "malformed_output" and result.markdown is None


def test_uncertain_source_relevance_and_representation_are_withheld():
    def uncertain_inventory(response):
        response["source_claims"][-1]["classification"] = "uncertain"

    result = run(COMPLETE_DRAFT, complete_callback(inventory_override=uncertain_inventory),
                 sources=COMPLETE_SOURCES, raw=True)
    assert result.reason == "source_completeness_uncertain" and result.review_calls == 0

    def uncertain_mapping(response, payload):
        response["representations"][1].update(status="uncertain", spans=[])

    result = run(COMPLETE_DRAFT, complete_callback(review_override=uncertain_mapping),
                 sources=COMPLETE_SOURCES, raw=True)
    assert result.reason == "source_completeness_uncertain" and result.markdown is None


def test_valid_but_unrelated_citation_cannot_satisfy_source_mapping():
    def swap_mapping(response, payload):
        education = next(i for i, item in enumerate(payload["source_claims"]) if "North College" in item["quote"])
        beta = next(i for i, item in enumerate(payload["source_claims"]) if "Beta" in item["quote"])
        response["representations"][education]["spans"] = copy.deepcopy(response["representations"][beta]["spans"])

    result = run(COMPLETE_DRAFT, complete_callback(review_override=swap_mapping),
                 sources=COMPLETE_SOURCES, raw=True)
    assert result.reason == "malformed_output" and result.markdown is None


def test_source_inventory_callback_cannot_mutate_sources_or_saved_inventory():
    async def callback(phase, payload):
        if phase == "source_inventory":
            response = complete_inventory(payload)
            payload["sources"][0]["text"] = "mutated"
            return response
        response = complete_review(payload)
        payload["source_claims"][0]["quote"] = "mutated"
        payload["sources"][0]["text"] = "mutated"
        return response

    result = run(COMPLETE_DRAFT, callback, sources=COMPLETE_SOURCES, raw=True)
    assert result.status == "approved" and result.markdown == COMPLETE_DRAFT
    assert COMPLETE_SOURCES[0].text == "Alex Example\nEngineer at Acme\nEngineer at Beta\n"


def test_source_inventory_timeout_stops_before_any_draft_review():
    calls = []
    async def callback(phase, payload):
        calls.append(phase)
        await asyncio.Event().wait()

    result = run(COMPLETE_DRAFT, callback, sources=COMPLETE_SOURCES, timeout=0.02, raw=True)
    assert result.reason == "timeout_source_inventory" and result.markdown is None
    assert calls == ["source_inventory"] and (result.review_calls, result.repair_calls) == (0, 0)


def test_review_after_patch_must_account_for_original_inventory_again():
    draft = COMPLETE_DRAFT.replace("Engineer at Beta", "Engineer at Beta plus invented detail")
    calls = []

    async def callback(phase, payload):
        calls.append(phase)
        if phase == "source_inventory":
            return complete_inventory(payload)
        if phase == "repair":
            return {"edits": [{"claim_id": payload["flagged_claim_ids"][0],
                               "replacement": ""}]}
        if phase == "initial_review":
            # A supported, properly cited prefix accounts for Beta before repair.
            original = copy.deepcopy(payload)
            line = next(line for line in payload["lines"] if "invented detail" in line["text"])
            line["text"] = "Engineer at Beta"
            response = complete_review(payload)
            target = next(c for c in response["claims"] if c["line_id"] == line["line_id"])
            response["claims"].append(claim(next(l for l in original["lines"] if l["line_id"] == line["line_id"]),
                                              target["end"], len("Engineer at Beta plus invented detail"), "unsupported"))
            return response
        assert [line["text"] for line in payload["lines"]] == COMPLETE_DRAFT.splitlines()
        response = complete_review(payload)
        beta = next(i for i, item in enumerate(payload["source_claims"]) if "Beta" in item["quote"])
        response["representations"][beta].update(status="missing", spans=[])
        return response

    result = run(draft, callback, sources=COMPLETE_SOURCES, raw=True)
    assert result.reason == "source_facts_missing" and result.markdown is None
    assert calls == ["source_inventory", "initial_review", "repair", "post_patch_review"]
    assert (result.review_calls, result.repair_calls) == (2, 1)


# Independent synthetic answer key: each whole source line is required, while the
# job advertisement is never candidate evidence. The scripted reviewer decisions
# below exercise orchestration, not a real model's ability to recognize claims.
MATRIX_SOURCE = (SourceDocument("candidate", (
    "Alex Example\n"
    "Alpha: engineer, 2019–2021\n"
    "Alpha: improved accuracy 30% over 6 months\n"
    "Beta: engineer, 2022–Present\n"
)),)
MATRIX_ANSWER_KEY = (
    "# Alex Example\n## Experience\nAlpha: engineer, 2019–2021\n"
    "Alpha: improved accuracy 30% over 6 months\nBeta: engineer, 2022–Present\n"
)


def matrix_review_with_invented_suffixes(payload, suffixes):
    """Explicitly flag suffixes while independently accounting for source lines."""
    shortened = copy.deepcopy(payload)
    changed = {}
    for line in shortened["lines"]:
        for suffix in suffixes:
            if line["text"].endswith(suffix):
                original = line["text"]
                line["text"] = original[:-len(suffix)]
                changed[line["line_id"]] = original
                break
    response = complete_review(shortened)
    for line_id, original in changed.items():
        prefix = next(c for c in response["claims"] if c["line_id"] == line_id)
        response["claims"].append(claim({"line_id": line_id}, prefix["end"], len(original), "unsupported"))
    return response


@pytest.mark.parametrize("suffix", ["", " within 2 weeks"])
def test_scripted_duration_review_retains_source_duration(suffix):
    calls = []
    draft = MATRIX_ANSWER_KEY.replace("over 6 months", "over 6 months" + suffix)

    async def callback(phase, payload):
        calls.append(phase)
        if phase == "source_inventory":
            return complete_inventory(payload)
        if phase == "repair":
            return {"edits": [{"claim_id": cid, "replacement": ""} for cid in payload["flagged_claim_ids"]]}
        return matrix_review_with_invented_suffixes(payload, [" within 2 weeks"])

    result = run(draft, callback, sources=MATRIX_SOURCE, raw=True)
    assert result.status == "approved" and result.markdown == MATRIX_ANSWER_KEY
    assert (result.review_calls, result.repair_calls) == ((2, 1) if suffix else (1, 0))
    assert calls[-1] == ("post_patch_review" if suffix else "initial_review")


def test_scripted_year_only_review_removes_invented_months():
    draft = MATRIX_ANSWER_KEY.replace("2019–2021", "2019–2021 (January–December)")
    draft = draft.replace("2022–Present", "2022–Present (starting February)")
    phases = []

    async def callback(phase, payload):
        phases.append(phase)
        if phase == "source_inventory":
            return complete_inventory(payload)
        if phase == "repair":
            return {"edits": [{"claim_id": cid, "replacement": ""} for cid in payload["flagged_claim_ids"]]}
        return matrix_review_with_invented_suffixes(payload, [" (January–December)", " (starting February)"])

    result = run(draft, callback, sources=MATRIX_SOURCE, raw=True)
    assert result.status == "approved" and result.markdown == MATRIX_ANSWER_KEY
    assert phases == ["source_inventory", "initial_review", "repair", "post_patch_review"]


def test_scripted_job_ad_requirement_is_not_candidate_evidence():
    job_ad_only = "5 years Kubernetes"
    draft = MATRIX_ANSWER_KEY.replace("Beta: engineer, 2022–Present",
                                      "Beta: engineer, 2022–Present; " + job_ad_only)
    phases = []

    async def callback(phase, payload):
        phases.append(phase)
        assert [source["text"] for source in payload["sources"]] == [MATRIX_SOURCE[0].text]
        assert all(job_ad_only not in source["text"] for source in payload["sources"])
        if phase == "source_inventory":
            return complete_inventory(payload)
        if phase == "repair":
            return {"edits": [{"claim_id": cid, "replacement": ""} for cid in payload["flagged_claim_ids"]]}
        return matrix_review_with_invented_suffixes(payload, ["; " + job_ad_only])

    result = run(draft, callback, sources=MATRIX_SOURCE, raw=True)
    assert result.status == "approved" and result.markdown == MATRIX_ANSWER_KEY
    assert phases == ["source_inventory", "initial_review", "repair", "post_patch_review"]


def test_scripted_swapped_date_ranges_are_withheld():
    # Two labels rewritten into each other is a cycle. Even before the label
    # guard can report it, the reviewer's completeness check detects that the
    # source date records are no longer represented, so publication is withheld.
    draft = (MATRIX_ANSWER_KEY.replace("2019–2021", "SWAP")
             .replace("2022–Present", "2019–2021").replace("SWAP", "2022–Present"))
    result = run(draft, complete_callback(), sources=MATRIX_SOURCE, raw=True)
    assert (result.status, result.reason, result.markdown) == (
        "withheld", "source_facts_missing", None)
    assert result.repair_calls == 0


def test_scripted_wrong_employer_is_corrected_without_losing_the_accomplishment():
    draft = MATRIX_ANSWER_KEY.replace("Alpha: improved", "Beta: improved")
    result = run(draft, complete_callback(), sources=MATRIX_SOURCE, raw=True)
    assert result.status == "approved"
    assert result.markdown == MATRIX_ANSWER_KEY
    assert "restored 1 source-owned record label" in result.commentary
    assert "improved accuracy 30% over 6 months" in result.markdown
    assert "2019–2021" in result.markdown and "2022–Present" in result.markdown


def test_original_counterexample_reviewer_never_receives_wrong_employer_label():
    # Original adversarial fixture: a reviewer approved Beta as the owner of an
    # accomplishment whose source record belongs to Alpha. The deterministic
    # label restoration runs first, so a reviewer cannot approve that wrong
    # label, while the accomplishment and unrelated facts remain intact.
    draft = MATRIX_ANSWER_KEY.replace("Alpha: improved", "Beta: improved")
    wrong_line = "Beta: improved accuracy 30% over 6 months"
    correct_line = "Alpha: improved accuracy 30% over 6 months"
    reviewed = []

    async def dishonest_callback(phase, payload):
        if phase == "source_inventory":
            return complete_inventory(payload)
        lines = [line["text"] for line in payload["lines"]]
        reviewed.append(lines)
        assert wrong_line not in lines and correct_line in lines
        return complete_review(payload)

    result = run(draft, dishonest_callback, sources=MATRIX_SOURCE, raw=True)
    assert result.status == "approved" and result.markdown == MATRIX_ANSWER_KEY
    assert result.reason is None
    assert reviewed == [[line for line in MATRIX_ANSWER_KEY.splitlines() if line.strip()]]
    assert "restored 1 source-owned record label" in result.commentary
    assert "30% over 6 months" in result.markdown
    assert "2019–2021" in result.markdown and "2022–Present" in result.markdown