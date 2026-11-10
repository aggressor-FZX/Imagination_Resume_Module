"""The full approval path, end to end, with every refusal that must survive it.

These are the eight focused regressions the brief names. Each drives the real
`run_release` with a scripted reviewer, so the path under test is the one that will
run in production — not a helper.

The receipt tests use the REAL `publication_gate.issue_receipt` / `verify_receipt`
with a test signing key, so "invalid receipt is refused" is proven against production
code rather than a stub.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

import decision_contract as dc
from experiment.cases_v2 import SOURCES, by_id
from experiment.release_trace import (
    REQUIRED_SECTIONS, run_release, sections_present, trace_markdown,
)

APPROVE = "Decision: approve\n\nIssues:\n- none\n"


def reviewer(answers: list[str]):
    """An `ask` that records the exact document it was given and replays answers."""
    seen: list[str] = []

    async def ask(document: str, _sources: dict[str, str]) -> str:
        seen.append(document)
        return answers[min(len(seen) - 1, len(answers) - 1)]

    ask.seen = seen  # type: ignore[attr-defined]
    return ask


def run(case_id: str, answers: list[str], **kwargs):
    case = by_id(case_id)
    ask = reviewer(answers)
    trace = asyncio.run(run_release(case_id, case.draft_before_trim,
                                    case.sources, ask=ask, **kwargs))
    return trace, ask


# ── 1. Faithful paraphrase approves without edits ───────────────────────────

def test_1_faithful_paraphrase_approves_without_edits():
    trace, ask = run("faithful_paraphrase", [APPROVE], issue_receipt=None)
    assert trace.first_decision == "approve"
    assert trace.first_valid is True
    assert trace.repair_status == "not_attempted"
    assert trace.proposed_sha == "", "no document may be proposed on an approval"
    # One call only: an approval must not trigger a repair or a re-review.
    assert len(ask.seen) == 1
    assert ask.seen[0] == by_id("faithful_paraphrase").draft_before_trim


# ── 2. Source-stated concurrent employment remains unchanged ────────────────

def test_2_source_backed_concurrency_is_not_edited():
    """Both `Present` roles are source-stated; the document must be released as is.

    If the reviewer approves, no edit is applied at all. The test asserts the
    document is byte-identical, which is what "unchanged" means.
    """
    case = by_id("faithful_paraphrase")
    assert "04/2019 - Present" in case.draft_before_trim
    assert "08/2023 - Present" in case.draft_before_trim
    trace, _ask = run("faithful_paraphrase", [APPROVE])
    assert trace.original == case.draft_before_trim
    assert trace.proposed_document == "" or \
        trace.proposed_document == case.draft_before_trim
    assert trace.release_reason.startswith("initial_llm_approve")


# ── 3. Unsupported-method correction preserves the supported achievement ────

def test_3_unsupported_method_correction_preserves_the_achievement():
    revise = ("Decision: revise\n\nIssues:\n"
              "- passage: Cut rework by 18% through standardizing panel labeling "
              "and targeted quality audits across the terminal fleet.\n"
              "  problem: unsupported\n  sources: [candidate-main.txt]\n"
              "  why: 'targeted quality audits' is not in the source.\n"
              "  correction: Cut rework by 18% through standardizing panel "
              "labeling across the terminal fleet.\n")
    trace, ask = run("unsupported_method", [revise, APPROVE])
    assert trace.repair_status == "applied"
    corrected = trace.proposed_document
    assert "targeted quality audits" not in corrected
    for kept in ("18%", "standardizing panel labeling", "terminal fleet",
                 "Cascade Freight Partners"):
        assert kept in corrected, kept
    # The re-review saw exactly the corrected document.
    assert ask.seen[1] == corrected
    assert trace.re_review_decision == "approve"


# ── 4. Wrong-record correction preserves the fact under its correct owner ───

def test_4_wrong_record_correction_keeps_the_fact_with_its_owner():
    revise = ("Decision: revise\n\nIssues:\n"
              "- passage: Prepared the monthly cost summary that the regional "
              "manager reviewed and reconciled 1,180 fuel invoices each quarter.\n"
              "  problem: misattributed\n  sources: [candidate-main.txt]\n"
              "  why: the 1,180 figure belongs to Harborline.\n"
              "  correction: Prepared the monthly cost summary reviewed by the "
              "regional manager.\n")
    trace, ask = run("wrong_record_attribution", [revise, APPROVE])
    assert trace.repair_status == "applied"
    corrected = trace.proposed_document
    cascade = corrected.split("**Cascade Freight Partners**")[1]
    cascade_bullets = "\n".join(l for l in cascade.splitlines() if l.startswith("- "))
    assert "1,180" not in cascade_bullets
    harborline = corrected.split("**Harborline Transit Authority**")[1].split("**")[0]
    assert "Reconciled 1,180 fuel invoices" in harborline
    assert ask.seen[1] == corrected


# ── 5. Missing-record restoration preserves source employer/title/dates/bullets

def test_5_record_restoration_uses_the_sources_own_fields():
    revise = ("Decision: revise\n\nIssues:\n"
              "- passage: (omitted)\n  problem: omitted\n"
              "  sources: [candidate-supplement.txt]\n"
              "  why: the Ridgeline record is missing.\n"
              "  correction: add the Ridgeline Equipment Services record.\n")
    trace, ask = run("omitted_record", [revise, APPROVE])
    assert trace.repair_status == "applied"
    restored = trace.proposed_document
    for expected in ("Ridgeline Equipment Services", "Inventory Analyst",
                     "08/2023", "Present", "Fife, WA",
                     "Maintain parts availability records for 7 service branches.",
                     "set reorder points."):
        assert expected in restored, expected
    # Nothing invented: no date outside the source.
    assert "08/2022" not in restored and "03/2023" not in restored
    assert ask.seen[1] == restored


# ── 6. Wrong-owner, altered-content and invalid-receipt exports are refused ─

def test_6a_wrong_owner_receipt_is_refused():
    """A receipt minted for one owner must not authorise another."""
    from publication_gate import issue_receipt, verify_receipt
    os.environ.setdefault("PUBLICATION_RECEIPT_SECRET", "test-secret-not-for-production-16")
    markdown = by_id("faithful_paraphrase").draft_before_trim
    from publication_gate import Verdict
    receipt = issue_receipt(markdown=markdown,
                            verdict=Verdict.PASS_TIER_A.value,
                            owner="candidate-a")
    assert receipt, "test signing key must be available"
    from approved_resume import export_payload
    payload = export_payload(markdown)
    # Sanity first: the SAME receipt IS accepted for the owner it was minted for,
    # so the refusal below is the owner check and not a broken signature.
    ok_same, _ = verify_receipt(receipt, markdown=markdown,
                                owner="candidate-a", render_payload=payload)
    assert ok_same is True
    ok, why = verify_receipt(receipt, markdown=markdown, owner="candidate-b",
                             render_payload=payload)
    assert ok is False, "a receipt issued for another owner must be refused"
    assert why == "receipt_owner_mismatch"


def test_6b_altered_content_receipt_is_refused():
    """Editing the approved markdown after signing must invalidate the receipt."""
    from publication_gate import issue_receipt, verify_receipt
    os.environ.setdefault("PUBLICATION_RECEIPT_SECRET", "test-secret-not-for-production-16")
    from approved_resume import export_payload
    markdown = by_id("faithful_paraphrase").draft_before_trim
    payload = export_payload(markdown)
    from publication_gate import Verdict
    receipt = issue_receipt(markdown=markdown, verdict=Verdict.PASS_TIER_A.value,
                            owner="c", approved_payload=payload)
    assert receipt
    tampered = markdown.replace("nine years", "twenty years")
    assert tampered != markdown
    # The export is re-derived from the TAMPERED markdown, which is what an
    # attacker would have to do — the binding is on content, not on the string.
    ok, why = verify_receipt(receipt, markdown=tampered, owner="c",
                             render_payload=export_payload(tampered))
    assert ok is False
    assert why in ("receipt_content_mismatch", "receipt_render_content_mismatch")


def test_6c_a_revise_never_produces_a_receipt():
    """A revise must not mint anything, even if the model would approve later."""
    revise = ("Decision: revise\n\nIssues:\n"
              "- passage: Reconciled 1,180 fuel invoices against depot dispensing "
              "records every quarter.\n"
              "  problem: unsupported\n  sources: [candidate-main.txt]\n"
              "  why: x\n  correction: x\n")
    trace, _ask = run("faithful_paraphrase", [revise, "Decision: revise\n"])
    assert trace.released is False
    assert trace.receipt == ""
    # Either no applicable repair, or a repair whose re-review refused. Both are
    # correct and neither may mint a receipt.
    assert trace.release_reason in ("re_review_not_approved", "revision_unresolved")


# ── 7. Re-review receives exactly the proposed document ────────────────────

def test_7_re_review_receives_exactly_the_proposal():
    revise = ("Decision: revise\n\nIssues:\n"
              "- passage: Cut rework by 18% through standardizing panel labeling "
              "and targeted quality audits across the terminal fleet.\n"
              "  problem: unsupported\n  sources: [candidate-main.txt]\n"
              "  why: unsupported method\n"
              "  correction: Cut rework by 18% through standardizing panel "
              "labeling across the terminal fleet.\n")
    trace, ask = run("unsupported_method", [revise, APPROVE])
    assert len(ask.seen) == 2
    assert ask.seen[1] == trace.proposed_document
    assert trace.re_review_document_sha == trace.proposed_sha
    # ...and it is NOT the original.
    assert ask.seen[1] != trace.original


# ── 8. Required core sections present, without invented filler ─────────────

def test_8_core_sections_are_checked_not_generated():
    case = by_id("faithful_paraphrase")
    present = sections_present(case.draft_before_trim)
    assert present == {name: True for name in REQUIRED_SECTIONS}

    stripped = case.draft_before_trim.replace("## Professional Summary", "")
    assert sections_present(stripped)["Professional Summary"] is False


def test_8b_a_document_missing_a_section_is_not_released():
    """Structure is checked BEFORE a receipt is minted."""
    from publication_gate import issue_receipt, verify_receipt
    os.environ.setdefault("PUBLICATION_RECEIPT_SECRET", "test-secret-not-for-production-16")
    case = by_id("faithful_paraphrase")
    broken = case.draft_before_trim.replace("## Professional Summary\n", "")
    trace = asyncio.run(run_release(
        "faithful_paraphrase", broken, case.sources,
        ask=reviewer([APPROVE]), issue_receipt=issue_receipt,
        verify_receipt=verify_receipt, owner="c"))
    assert trace.released is False
    assert trace.release_reason.startswith("missing_required_sections")
    assert trace.receipt == "", "no receipt may be minted for a broken document"


def test_8c_section_checking_never_adds_text():
    """`sections_present` reports; it does not fill gaps."""
    original = "## Professional Experience\n- x\n"
    assert sections_present(original) == {
        "Professional Summary": False, "Professional Experience": True}
    # The input string is immutable and is not rewritten.
    assert original == "## Professional Experience\n- x\n"


# ── The trace itself ───────────────────────────────────────────────────────

def test_the_trace_records_every_step_and_the_hashes():
    revise = ("Decision: revise\n\nIssues:\n"
              "- passage: Cut rework by 18% through standardizing panel labeling "
              "and targeted quality audits across the terminal fleet.\n"
              "  problem: unsupported\n  sources: [candidate-main.txt]\n"
              "  why: unsupported method\n"
              "  correction: Cut rework by 18% through standardizing panel "
              "labeling across the terminal fleet.\n")
    trace, _ask = run("unsupported_method", [revise, APPROVE])
    assert trace.original_sha and trace.proposed_sha
    assert trace.original_sha != trace.proposed_sha
    assert any(s.startswith("first_review") for s in trace.steps)
    assert any(s.startswith("repair") for s in trace.steps)
    assert any(s.startswith("re_review") for s in trace.steps)
    assert trace.repair_diff, "the applied edit must be diffable"
    text = trace_markdown(trace)
    assert "Release trace" in text and "Repair diff" in text
