"""Integration regressions for the live semantic-review path.

These drive `semantic_review.engine.review_and_release` with a scripted reviewer and
assert the outcome the LIVE caller depends on: a `CorrectionResult` whose `markdown`
is the document that may be published, or `withheld` with an understood reason code.

They also assert the properties the integration must not lose: production imports no
experiment runner, and the emission of a `(omitted)` marker never reaches a document.
"""

from __future__ import annotations

import asyncio
import ast
from pathlib import Path

import pytest

import decision_contract as dc
import source_correction as sc
from experiment.cases_v2 import SOURCES, by_id
from semantic_review.engine import (
    CorrectionResult, review_and_release,
)

APPROVE = "Decision: approve\n\nIssues:\n- none\n"
ROOT = Path(__file__).resolve().parents[1]


def reviewer(answers: list[str]):
    seen: list[str] = []

    async def review(document: str) -> str:
        seen.append(document)
        return answers[min(len(seen) - 1, len(answers) - 1)]

    review.seen = seen  # type: ignore[attr-defined]
    return review


def run(draft: str, answers: list[str], sources=None, **kw):
    r = reviewer(answers)
    result = asyncio.run(review_and_release(draft, sources or SOURCES, r, **kw))
    return result, r


# ── The integration boundary itself ────────────────────────────────────────

def test_the_result_is_field_compatible_with_the_live_caller():
    """`app.py` unpacks this type; a missing field would break the release branch."""
    original_fields = {
        "status", "markdown", "reason", "commentary", "coverage",
        "review_calls", "repair_calls", "restore_calls",
    }
    assert set(CorrectionResult.__dataclass_fields__) == original_fields
    assert set(sc.CorrectionResult.__dataclass_fields__) == original_fields | {"final_bridges"}
    original_args = ("approved", "document", None, "", None, 1, 0, 0)
    semantic = CorrectionResult(*original_args)
    legacy = sc.CorrectionResult(*original_args)
    assert all(getattr(semantic, field) == getattr(legacy, field) for field in original_fields)
    assert legacy.final_bridges is None
    assert sc.CorrectionResult(*original_args, final_bridges=()).final_bridges == ()


def test_production_package_imports_no_experiment_runner():
    """Production must not depend on run harnesses, ledgers or report generators."""
    offenders = []
    for path in (ROOT / "semantic_review").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                if name.startswith("experiment"):
                    offenders.append(f"{path.name}: {name}")
    assert not offenders, offenders


def test_no_span_coverage_is_fabricated():
    """Coverage must name what it is, so it cannot be read as span coverage."""
    case = by_id("faithful_paraphrase")
    result, _ = run(case.draft_before_trim, [APPROVE])
    assert result.status == "approved"
    assert result.coverage is not None
    assert result.coverage.level == "prose_issues"
    # The prose contract has no per-span verdicts; reporting zero is honest.
    assert result.coverage.supported_count == 0
    assert result.coverage.structural_count == 0


# ── 1. Approval ────────────────────────────────────────────────────────────

def test_1_approval_returns_the_document_unchanged():
    case = by_id("faithful_paraphrase")
    result, r = run(case.draft_before_trim, [APPROVE])
    assert result.status == "approved"
    assert result.markdown == case.draft_before_trim
    assert result.reason is None
    assert result.review_calls == 1
    assert result.repair_calls == 0
    assert r.seen == [case.draft_before_trim]


def test_1b_source_stated_concurrency_is_approved_unedited():
    """Both `Present` roles are source-stated; no edit may be applied."""
    case = by_id("faithful_paraphrase")
    assert "04/2019 - Present" in case.draft_before_trim
    assert "08/2023 - Present" in case.draft_before_trim
    result, r = run(case.draft_before_trim, [APPROVE])
    assert result.markdown == case.draft_before_trim
    assert len(r.seen) == 1


# ── 2. Correction followed by fresh approval ───────────────────────────────

def test_2_correction_then_fresh_approval_releases_the_corrected_document():
    case = by_id("unsupported_method")
    revise = ("Decision: revise\n\nIssues:\n"
              "- passage: Cut rework by 18% through standardizing panel labeling "
              "and targeted quality audits across the terminal fleet.\n"
              "  problem: unsupported\n  sources: [candidate-main.txt]\n"
              "  why: 'targeted quality audits' is not in the source.\n"
              "  correction: Cut rework by 18% through standardizing panel "
              "labeling across the terminal fleet.\n")
    result, r = run(case.draft_before_trim, [revise, APPROVE])

    assert result.status == "approved"
    assert "targeted quality audits" not in result.markdown
    for kept in ("18%", "standardizing panel labeling", "terminal fleet",
                 "Cascade Freight Partners"):
        assert kept in result.markdown
    # The re-review saw the CORRECTED document, not the original.
    assert len(r.seen) == 2
    assert r.seen[1] == result.markdown
    assert r.seen[1] != case.draft_before_trim
    assert result.repair_calls == 1


def test_2b_a_revision_that_is_not_re_approved_withholds():
    """`revise` is never permission to release."""
    case = by_id("unsupported_method")
    revise = ("Decision: revise\n\nIssues:\n"
              "- passage: Cut rework by 18% through standardizing panel labeling "
              "and targeted quality audits across the terminal fleet.\n"
              "  problem: unsupported\n  sources: [candidate-main.txt]\n"
              "  why: x\n"
              "  correction: Cut rework by 18% through standardizing panel "
              "labeling across the terminal fleet.\n")
    # The re-review must be a VALID refusal, not an empty one: a bare
    # `Decision: revise` is malformed and would test the parser, not the rule.
    refusal = ("Decision: revise\n\nIssues:\n"
               "- passage: Cut rework by 18% through standardizing panel labeling "
               "across the terminal fleet.\n"
               "  problem: unresolved\n  sources: [candidate-main.txt]\n"
               "  why: still unsure the method is sourced.\n"
               "  correction: leave the clause as written.\n")
    result, _ = run(case.draft_before_trim, [revise, refusal])
    assert result.status == "withheld"
    assert result.markdown is None
    assert result.reason == "unresolved_claims"


# ── 3. Missing-record restoration ──────────────────────────────────────────

def test_3_missing_record_is_restored_from_the_source():
    case = by_id("omitted_record")
    revise = ("Decision: revise\n\nIssues:\n"
              "- passage: (omitted)\n  problem: omitted\n"
              "  sources: [candidate-supplement.txt]\n"
              "  why: the Ridgeline record is missing.\n"
              "  correction: add the Ridgeline Equipment Services record.\n")
    result, r = run(case.draft_before_trim, [revise, APPROVE])

    assert result.status == "approved"
    doc = result.markdown
    for expected in ("Ridgeline Equipment Services", "Inventory Analyst",
                     "08/2023", "Fife, WA",
                     "set reorder points."):
        assert expected in doc, expected
    assert "08/2022" not in doc and "03/2023" not in doc
    assert result.restore_calls == 1
    assert r.seen[1] == doc


# ── 4. Wrong-record attribution ────────────────────────────────────────────

def test_4_wrong_record_correction_keeps_the_fact_with_its_owner():
    case = by_id("wrong_record_attribution")
    revise = ("Decision: revise\n\nIssues:\n"
              "- passage: Prepared the monthly cost summary that the regional "
              "manager reviewed and reconciled 1,180 fuel invoices each quarter.\n"
              "  problem: misattributed\n  sources: [candidate-main.txt]\n"
              "  why: the 1,180 figure belongs to Harborline.\n"
              "  correction: Prepared the monthly cost summary reviewed by the "
              "regional manager.\n")
    result, _ = run(case.draft_before_trim, [revise, APPROVE])

    assert result.status == "approved"
    doc = result.markdown
    cascade = doc.split("**Cascade Freight Partners**")[1]
    cascade_bullets = "\n".join(l for l in cascade.splitlines() if l.startswith("- "))
    assert "1,180" not in cascade_bullets
    harborline = doc.split("**Harborline Transit Authority**")[1].split("**")[0]
    assert "Reconciled 1,180 fuel invoices" in harborline
    assert doc.count("**Cascade Freight Partners**") == 1


# ── 5. Refusals ────────────────────────────────────────────────────────────

def test_5_cannot_assess_withholds():
    case = by_id("faithful_paraphrase")
    result, r = run(case.draft_before_trim,
                    ["Decision: cannot_assess\n\nThe dates cannot be reconciled.\n"])
    assert result.status == "withheld"
    assert result.reason == "unresolved_claims"
    assert len(r.seen) == 1


@pytest.mark.parametrize("answer", [
    "Looks good to me.",
    "Decision: probably_fine\n",
    "Decision: revise\n\nIssues:\n- none\n",
])
def test_5b_malformed_decisions_withhold(answer):
    case = by_id("faithful_paraphrase")
    result, _ = run(case.draft_before_trim, [answer])
    assert result.status == "withheld"
    assert result.markdown is None


def test_5c_an_ambiguous_edit_withholds():
    """Two identical bullets, so no unambiguous target exists."""
    case = by_id("unsupported_method")
    doubled = case.draft_before_trim + "\n" + case.draft_before_trim
    revise = ("Decision: revise\n\nIssues:\n"
              "- passage: Cut rework by 18% through standardizing panel labeling "
              "and targeted quality audits across the terminal fleet.\n"
              "  problem: unsupported\n  sources: [candidate-main.txt]\n"
              "  why: x\n"
              "  correction: Cut rework by 18% through standardizing panel "
              "labeling across the terminal fleet.\n")
    result, _ = run(doubled, [revise])
    assert result.status == "withheld"
    assert result.repair_calls == 0


def test_5d_unresolvable_reference_withholds():
    """A source ID that does not exist is rejected by the PARSER as structural.

    So the accurate reason is `malformed_output`, not a separate attribution code.
    Either way the document is withheld, which is the property that matters.
    """
    case = by_id("faithful_paraphrase")
    answer = ("Decision: revise\n\nIssues:\n"
              "- passage: Reconciled 1,180 fuel invoices against depot dispensing "
              "records every quarter.\n"
              "  problem: unsupported\n  sources: [not-a-source.txt]\n"
              "  why: x\n  correction: Reconciled 1,180 invoices.\n")
    result, _ = run(case.draft_before_trim, [answer])
    assert result.status == "withheld"
    assert result.reason == "malformed_output"
    assert result.markdown is None


def test_5d2_a_reference_that_resolves_to_nothing_still_withholds():
    """An APPROVE whose issue cites a fabricated quote must not release.

    The passage and the source id are valid, so the envelope parses; the quote is
    invented, so `references_resolve` fails and the release is refused.
    """
    case = by_id("faithful_paraphrase")
    answer = ("Decision: revise\n\nIssues:\n"
              "- passage: Reconciled 1,180 fuel invoices against depot dispensing "
              "records every quarter.\n"
              "  problem: unsupported\n"
              "  sources: [candidate-main.txt: \"a sentence that is not in the "
              "source at all\"]\n"
              "  why: x\n  correction: Reconciled 1,180 invoices.\n")
    result, _ = run(case.draft_before_trim, [answer])
    assert result.status == "withheld"
    assert result.markdown is None


def test_5e_a_provider_failure_withholds_without_claiming_a_verdict():
    case = by_id("faithful_paraphrase")

    async def boom(_document: str) -> str:
        raise TimeoutError("provider timed out")

    result = asyncio.run(review_and_release(case.draft_before_trim, SOURCES, boom))
    assert result.status == "withheld"
    assert result.reason == "provider_error"
    assert result.markdown is None


# ── 6. Required structure ──────────────────────────────────────────────────

def test_6_missing_section_withholds():
    case = by_id("faithful_paraphrase")
    broken = case.draft_before_trim.replace("## Professional Summary\n", "")
    result, _ = run(broken, [APPROVE])
    assert result.status == "withheld"
    assert "missing sections" in (result.commentary or "")


def test_6b_structure_is_checked_not_generated():
    """No filler is inserted to satisfy the check."""
    case = by_id("faithful_paraphrase")
    broken = case.draft_before_trim.replace("## Professional Experience\n", "")
    result, _ = run(broken, [APPROVE])
    assert result.status == "withheld"
    # The engine returned nothing rather than a padded document.
    assert result.markdown is None


# ── 7. The (omitted) export marker ─────────────────────────────────────────

def test_7_an_omitted_marker_in_the_final_document_withholds():
    """Reproduced defect: `(omitted)` reached export_payload verbatim.

    It is a control marker, not resume text. It must never be published, and the
    engine must not silently strip it either — stripping would change a document
    after approval, so the receipt would bind text nobody approved.
    """
    from approved_resume import export_payload
    marker_doc = ("## Professional Summary\nOperations analyst with nine years.\n\n"
                  "## Professional Experience\n**Harborline Transit Authority** -- "
                  "Tacoma, WA\n- Reconciled 1,180 fuel invoices.\n- (omitted)\n")

    # The defect is real: the marker survives into the export payload today.
    payload = export_payload(marker_doc)
    assert "(omitted)" in str(payload), "fixture no longer reproduces the defect"

    result, _ = run(marker_doc, [APPROVE])
    assert result.status == "withheld"
    assert result.markdown is None
    assert "marker" in (result.commentary or "").lower()


def test_7b_a_clean_document_is_not_blocked_by_the_marker_check():
    case = by_id("faithful_paraphrase")
    result, _ = run(case.draft_before_trim, [APPROVE])
    assert result.status == "approved"
    assert "(omitted)" not in result.markdown


# ── Bounds ─────────────────────────────────────────────────────────────────

def test_the_call_budget_is_bounded():
    """A refusal loop cannot spend more than one repair plus one re-review."""
    case = by_id("unsupported_method")
    revise = ("Decision: revise\n\nIssues:\n"
              "- passage: Cut rework by 18% through standardizing panel labeling "
              "and targeted quality audits across the terminal fleet.\n"
              "  problem: unsupported\n  sources: [candidate-main.txt]\n"
              "  why: x\n"
              "  correction: Cut rework by 18% through standardizing panel "
              "labeling across the terminal fleet.\n")
    result, r = run(case.draft_before_trim, [revise, revise, revise])
    assert len(r.seen) <= 2, "no third call may be made"
    assert result.status == "withheld"


def test_an_expired_deadline_withholds_before_calling():
    case = by_id("faithful_paraphrase")
    result, r = run(case.draft_before_trim, [APPROVE], deadline_seconds=-1)
    assert result.status == "withheld"
    assert result.reason == "timeout_initial_review"
    assert r.seen == [], "no call may be made past the deadline"
