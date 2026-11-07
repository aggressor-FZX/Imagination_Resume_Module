"""Offline, synthetic regressions for source-skill omission recovery."""
import asyncio

import pytest

from source_correction import (SourceDocument, _restore_missing_skill_lines,
                               _source_skill_line, correct)


SOURCE = ("Professional Experience\n"
          "Harbor Transit -- Tacoma, WA\n"
          "Analyst | 2019 - Present\n"
          "- Reconciled fuel invoices across 14 depots.\n"
          "Skills\n"
          "Spreadsheet modelling, SQL\n"
          "Invoice reconciliation\n")
DRAFT = ("## Professional Experience\n"
         "**Harbor Transit** -- Tacoma, WA\n"
         "*Analyst | 2019 - Present*\n"
         "- Reconciled fuel invoices across 14 depots.\n"
         "## Technical Skills\n"
         "- Spreadsheet modelling\n"
         "- Trend analysis\n")


def run(source, draft, *, always_missing=False, bridge=False, budget_case=None):
    phases, reviews = [], []

    async def callback(phase, payload):
        phases.append(phase)
        if phase == "source_inventory":
            text = payload["sources"][0]["text"]
            items, position = [], 0
            for line in text.splitlines(keepends=True):
                end = position + len(line)
                items.append({"id": f"candidate:{position}:{end}",
                              "source_id": "candidate", "start": position,
                              "end": end, "quote": line,
                              "classification": "required" if (line.startswith((
                                  "Spreadsheet", "Invoice")) or
                                  (budget_case == "generic_after_skill_repair" and
                                   line.startswith("- Reconciled"))) else "non_resume"})
                position = end
            return {"source_claims": items}
        if phase == "restore":
            # If the old selector chooses the experience bullet, demonstrate that
            # the model can overwrite it with the requested source fact.
            return {"restorations": [
                {"line_id": line["line_id"], "source_claim_id": line["source_claim_id"],
                 "before": line["text"],
                 "replacement": next(i["quote"].strip() for i in payload["source_claims"]
                                     if i["id"] == line["source_claim_id"])}
                for line in payload["lines"]]}
        if phase == "repair":
            lines = {line["line_id"]: line["text"] for line in payload["lines"]}
            return {"edits": [
                {"claim_id": claim["id"],
                 "replacement": lines[claim["line_id"]].rstrip(".")
                                + ("." if not lines[claim["line_id"]].endswith(".") else "")}
                for claim in payload["claims"]
                if claim["verdict"] in {"unsupported", "uncertain"}]}
        reviews.append([line["text"] for line in payload["lines"]])
        source_text = payload["sources"][0]["text"]
        cite = {"source_id": "candidate", "start": 0, "end": len(source_text),
                "quote": source_text}
        claims = []
        for line in payload["lines"]:
            text = line["text"]
            structural = text.startswith("##")
            flagged = ("Reconciled fuel invoices" in text and
                       ((budget_case == "repair_after_skill" and phase != "initial_review") or
                        (budget_case == "skill_after_repairs" and phases.count("repair") < 2) or
                        (budget_case == "generic_after_skill_repair" and
                         phase == "post_restore_review")))
            claims.append({"id": f"{line['line_id']}:0:{len(text)}",
                           "line_id": line["line_id"], "start": 0, "end": len(text),
                           "verdict": "structural" if structural else "unsupported" if flagged else "supported",
                           "citations": [] if structural or flagged else [cite]})
        mappings = []
        for item in payload["source_claims"]:
            if item["classification"] != "required":
                mappings.append({"source_claim_id": item["id"], "status": "non_resume", "spans": []})
                continue
            quote = item["quote"].strip()
            line = next((c for c in claims if quote.casefold() in
                         next(l["text"] for l in payload["lines"] if l["line_id"] == c["line_id"]).casefold()), None)
            if always_missing or (budget_case == "generic_after_skill_repair" and
                                  "Reconciled fuel invoices" in quote):
                line = None
            mappings.append({"source_claim_id": item["id"],
                             "status": "represented" if line else "missing",
                             "spans": [{k: line[k] for k in ("line_id", "start", "end")}] if line else []})
        response = {"claims": claims, "representations": mappings}
        if bridge:
            response["bridge_decisions"] = {"b1": {"decision": "accept"}}
        return response

    bridges = [{"id": "b1", "skill": "Trend analysis"}] if bridge else None
    result = asyncio.run(correct(draft, (SourceDocument("candidate", source),),
                                 callback, permitted_bridges=bridges))
    return result, phases, reviews


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_partial_plain_text_skills_are_added_without_replacing_experience(newline):
    source, draft = SOURCE.replace("\n", newline), DRAFT.replace("\n", newline)
    result, phases, reviews = run(source, draft)
    assert result.status == "approved", result.reason
    assert phases == ["source_inventory", "initial_review", "post_restore_review"]
    assert result.restore_calls == 2
    assert "- Reconciled fuel invoices across 14 depots." in result.markdown
    assert "- Spreadsheet modelling" in result.markdown
    skills = result.markdown.split("## Technical Skills", 1)[1]
    assert "Spreadsheet modelling, SQL" in skills
    assert "Invoice reconciliation" in skills
    assert "- Trend analysis" in skills
    assert "- Reconciled fuel invoices across 14 depots." in reviews[-1]


def test_permitted_bridge_is_preserved_while_restoring_partial_skills():
    result, _, _ = run(SOURCE, DRAFT, bridge=True)
    assert result.status == "approved", result.reason
    assert "- Trend analysis" in result.markdown
    assert "Spreadsheet modelling, SQL" in result.markdown


def test_review_still_withholds_if_source_skills_remain_missing():
    result, phases, reviews = run(SOURCE, DRAFT, always_missing=True)
    assert (result.status, result.reason, result.markdown) == ("withheld", "source_facts_missing", None)
    assert phases == ["source_inventory", "initial_review", "post_restore_review"]
    assert "- Reconciled fuel invoices across 14 depots." in reviews[-1]


def test_skill_restoration_reserves_budget_for_repair_and_review():
    result, phases, _ = run(SOURCE, DRAFT, budget_case="repair_after_skill")
    assert (result.status, result.reason, result.markdown) == ("withheld", "unresolved_claims", None)
    assert phases == ["source_inventory", "initial_review", "post_restore_review",
                      "repair", "post_patch_review"]
    assert result.repair_calls == 1


def test_skill_restoration_does_not_edit_when_review_budget_is_exhausted():
    result, phases, _ = run(SOURCE, DRAFT, budget_case="skill_after_repairs")
    assert (result.status, result.reason, result.markdown) == ("withheld", "source_facts_missing", None)
    assert phases == ["source_inventory", "initial_review", "repair", "post_patch_review",
                      "repair", "post_patch_review"]
    assert result.restore_calls == 0


def test_generic_restoration_after_skill_and_repair_reserves_full_review():
    result, phases, _ = run(SOURCE, DRAFT, budget_case="generic_after_skill_repair")
    assert (result.status, result.reason, result.markdown) == ("withheld", "source_facts_missing", None)
    assert phases == ["source_inventory", "initial_review", "post_restore_review",
                      "repair", "post_patch_review"]
    assert result.repair_calls == 1


def test_missing_skill_without_a_draft_skill_section_never_targets_experience():
    draft = DRAFT.split("## Technical Skills")[0]
    result, phases, reviews = run(SOURCE, draft)
    assert result.status == "approved", result.reason
    assert phases == ["source_inventory", "initial_review", "post_restore_review"]
    assert "- Reconciled fuel invoices across 14 depots." in reviews[-1]
    assert "Invoice reconciliation" in result.markdown.split("## Technical Skills", 1)[1]


def item_for(text, fragment):
    start = text.index(fragment)
    return {"id": f"candidate:{start}:{start + len(fragment)}", "source_id": "candidate",
            "start": start, "end": start + len(fragment), "quote": fragment,
            "classification": "required"}


def test_partial_inventory_span_restores_the_complete_original_source_line():
    item = item_for(SOURCE, "SQL")
    result, count, ids = _restore_missing_skill_lines(DRAFT, {"candidate": SOURCE}, [item])
    assert count == 1 and ids == {item["id"]}
    assert "- Spreadsheet modelling, SQL" in result
    assert "- Reconciled fuel invoices across 14 depots." in result


def test_multiline_item_is_not_copied_or_targeted_as_a_generic_fact():
    item = item_for(SOURCE, "2019 - Present\n- Reconciled fuel invoices across 14 depots.\nSkills\nSpreadsheet")
    assert _source_skill_line(SOURCE, item) == (True, None)
    result, count, ids = _restore_missing_skill_lines(DRAFT, {"candidate": SOURCE}, [item])
    assert (result, count, ids) == (DRAFT, 0, {item["id"]})


def test_source_document_separator_ends_skills_section():
    text = "Skills\nSQL\n\n---\n\nAlex Example\n"
    assert _source_skill_line(text, item_for(text, "Alex Example")) == (False, None)


def test_duplicate_draft_skill_sections_are_not_guessed():
    draft = DRAFT + "## Skills\n- SQL\n"
    item = item_for(SOURCE, "Invoice reconciliation")
    result, count, ids = _restore_missing_skill_lines(draft, {"candidate": SOURCE}, [item])
    assert (result, count, ids) == (draft, 0, {item["id"]})


def test_already_present_source_line_is_not_duplicated_or_replaced():
    draft = DRAFT + "- Invoice reconciliation\n"
    item = item_for(SOURCE, "Invoice reconciliation")
    result, count, ids = _restore_missing_skill_lines(draft, {"candidate": SOURCE}, [item])
    assert (result, count, ids) == (draft, 0, {item["id"]})


def test_skill_lines_insert_before_following_section_preserving_unchanged_text():
    draft = DRAFT + "## Education\nSchool\n"
    item = item_for(SOURCE, "Invoice reconciliation")
    result, count, _ = _restore_missing_skill_lines(draft, {"candidate": SOURCE}, [item])
    assert count == 1
    assert result == draft.replace("## Education", "- Invoice reconciliation\n## Education")


@pytest.mark.parametrize("heading,marker", [("Skills:", "• "), ("## Skills", "- ")])
def test_pdf_plain_text_bullets_and_markdown_source_are_restored(heading, marker):
    source = f"{heading}\n{marker}Invoice reconciliation\n"
    item = item_for(source, "Invoice reconciliation")
    result, count, _ = _restore_missing_skill_lines(DRAFT, {"candidate": source}, [item])
    assert count == 1
    assert "- Invoice reconciliation" in result.split("## Technical Skills", 1)[1]
    assert "- Reconciled fuel invoices across 14 depots." in result


def test_same_skill_in_two_sources_is_added_once():
    text = "Skills\nInvoice reconciliation\n"
    first = item_for(text, "Invoice reconciliation")
    second = {**first, "id": first["id"].replace("candidate:", "second:"), "source_id": "second"}
    result, count, ids = _restore_missing_skill_lines(
        DRAFT, {"candidate": text, "second": text}, [first, second])
    assert count == 1
    assert ids == {first["id"], second["id"]}
    assert result.count("- Invoice reconciliation") == 1