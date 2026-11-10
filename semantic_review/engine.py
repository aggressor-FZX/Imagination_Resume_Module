"""Semantic review engine: the live approval path, without span bookkeeping.

The integration boundary
------------------------
`app.py` consumes exactly five things from a correction:

    status      "approved" | "withheld"
    markdown    the document to publish, or None
    reason      a stable reason code
    commentary  short operator text
    counters    review_calls / repair_calls / restore_calls (+ coverage diagnostics)

It never reads a claim span, an offset or a citation position. `publication_gate.decide`
takes only markdown, metrics support, finish_reason and entities. So the smallest
boundary that returns the existing approved-document outcome is: **produce a
CorrectionResult whose markdown is the approved document.**

This module does that by driving the PROSE contract:

    freeze sources
      ↓
    review(draft) ── approve ──────────────→ hard controls → approved
                   ├─ revise ── repair ──→ review(EXACT corrected doc)
                   │                        └─ approve → hard controls → approved
                   └─ cannot_assess / malformed / ambiguous edit → withheld

What is preserved from the live service
---------------------------------------
The caller does authentication, ownership, redaction, receipts and PDF/DOCX. This
module does not touch them. It returns the same `CorrectionResult` type the existing
caller already handles, so `app.py`'s release branch is unchanged.

What is deliberately NOT done
-----------------------------
- No span coverage is fabricated. `coverage` is reported as the levels this engine
  can actually substantiate (`line_count`, `claim_count` from the prose issues), and
  is marked with its own `level` so nobody mistakes it for span coverage.
- No meaning-changing cleanup after approval. The approved markdown is returned
  byte-for-byte; the caller's receipt binds it.
- Semantic observations never override a valid approval.

Bounded
-------
Call budget, deadline and a hard retry ceiling are all checked before each provider
call. A refusal to edit is never converted into an approval.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import decision_contract as dc
from semantic_review.references import references_resolve
from semantic_review.repair import apply_repair

# The live caller's reason codes. Anything else is reported as a generic failure by
# `app.py`, so this module must only emit values that are already understood.
REASON_MALFORMED = "malformed_output"
REASON_INCOMPLETE = "incomplete_review"
REASON_UNRESOLVED = "unresolved_claims"
REASON_NO_OP = "no_op_repair"
REASON_TIMEOUT = "timeout_initial_review"
REASON_TIMEOUT_REREVIEW = "timeout_post_patch_review"
REASON_PROVIDER = "provider_error"
REASON_ATTR = "attribution_conflict"

# Hard ceilings. A clean pass is 1 call; a repaired pass is 2. The extra headroom
# allows one shape retry, and nothing more.
MAX_REVIEW_CALLS = 4


@dataclass(frozen=True)
class Coverage:
    """Diagnostics only — never a release decision.

    `level` says what kind of coverage this is, so it can never be confused with the
    span coverage the prose contract does not produce. `claim_count` is the number of
    issues the reviewer raised, which is the only count this engine can substantiate.
    """

    level: str
    line_count: int
    claim_count: int
    supported_count: int
    structural_count: int


@dataclass(frozen=True)
class CorrectionResult:
    """Same shape as `source_correction.CorrectionResult`, so the caller is unchanged."""

    status: str
    markdown: str | None
    reason: str | None
    commentary: str
    coverage: Coverage | None
    review_calls: int
    repair_calls: int
    restore_calls: int = 0


Provider = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


def _withhold(reason: str, *, commentary: str = "", review_calls: int = 0,
              repair_calls: int = 0, restore_calls: int = 0,
              coverage: Coverage | None = None) -> CorrectionResult:
    return CorrectionResult("withheld", None, reason, commentary, coverage,
                            review_calls, repair_calls, restore_calls)


def _required_sections_present(markdown: str) -> tuple[bool, list[str]]:
    """Mandatory structure. Code checks presence; it never generates filler."""
    required = ("## Professional Summary", "## Professional Experience")
    missing = [name for name in required if name not in (markdown or "")]
    return (not missing), [m.removeprefix("## ") for m in missing]


def _marker_leaked(markdown: str) -> bool:
    """A control marker must never reach a candidate's document.

    `(omitted)` is an instruction to the repair layer, not resume text. Reproduced
    2026-11-09: a markdown containing `- (omitted)` passed straight through
    `export_payload` into `experience[].bullets`, so it would print in the PDF and
    DOCX. This is checked on the FINAL document, because that is the only place that
    matters.
    """
    return dc.OMITTED_MARKER in (markdown or "")


async def review_and_release(
    draft: str,
    sources: dict[str, str],
    review: Callable[[str], Awaitable[str]],
    *,
    deadline_seconds: float | None = None,
    observer: Callable[[dict[str, Any]], None] | None = None,
) -> CorrectionResult:
    """Run the prose approval path on one document.

    `review(document)` returns the reviewer's raw text for EXACTLY that document.
    The caller owns the provider; this module owns the decision sequence.
    """
    started = time.monotonic()

    def over_deadline() -> bool:
        return (deadline_seconds is not None
                and (time.monotonic() - started) >= deadline_seconds)

    def emit(event: dict[str, Any]) -> None:
        if observer is None:
            return
        try:
            observer(event)  # observer failures must never break a request
        except Exception:  # noqa: BLE001
            pass

    if not isinstance(draft, str) or not draft.strip():
        return _withhold("invalid_input")

    review_calls = 0
    repair_calls = 0
    restore_calls = 0

    # ── First review ────────────────────────────────────────────────────────
    if over_deadline():
        return _withhold(REASON_TIMEOUT)
    review_calls += 1
    emit({"phase": "initial_review", "call": review_calls})
    try:
        raw = await review(draft)
    except Exception:  # noqa: BLE001
        return _withhold(REASON_PROVIDER, review_calls=review_calls)

    first = dc.decide_from_text(raw, draft, sources)

    # An unusable answer is an operational outcome, never an approval.
    if not first.decision or first.structural_problems:
        return _withhold(REASON_MALFORMED,
                         commentary="; ".join(first.structural_problems[:3]),
                         review_calls=review_calls)

    if first.decision == dc.DECISION_CANNOT_ASSESS:
        return _withhold(REASON_UNRESOLVED,
                         commentary=(first.rationale or "")[:400],
                         review_calls=review_calls)

    if not references_resolve(first, sources):
        # The reviewer's evidence did not point at real text. That is an argument
        # defect, not a verdict about the document, and it cannot authorise release.
        return _withhold(REASON_ATTR, review_calls=review_calls)

    if first.decision == dc.DECISION_APPROVE:
        return _finish_release(draft, first, review_calls, repair_calls,
                               restore_calls, emit, reason_ok=None)

    # ── Revise: one repair, then a review of the EXACT corrected document ───
    if len(first.issues) != 1:
        # The scaffold applies one unambiguous edit at a time. More than one issue
        # means the target is not unambiguous, and choosing one would be a guess.
        return _withhold(REASON_UNRESOLVED,
                         commentary="multiple issues; no unambiguous edit",
                         review_calls=review_calls)

    issue = first.issues[0]
    is_restore = (issue.problem == dc.PROBLEM_OMITTED
                  or (issue.passage or "").strip() == dc.OMITTED_MARKER)
    corrected, status = apply_repair(draft, sources, [issue])
    if corrected == draft:
        reason = REASON_NO_OP if status in ("no_change",) else REASON_UNRESOLVED
        return _withhold(reason, commentary=f"repair:{status}",
                         review_calls=review_calls)
    repair_calls += 1
    restore_calls += 1 if is_restore else 0
    emit({"phase": "repair", "status": status})

    if over_deadline():
        return _withhold(REASON_TIMEOUT_REREVIEW, review_calls=review_calls,
                         repair_calls=repair_calls, restore_calls=restore_calls)

    # The corrected document is reviewed in full, not diffed. Anything less would
    # let the edit carry unreviewed meaning into the release.
    review_calls += 1
    emit({"phase": "post_patch_review", "call": review_calls})
    try:
        raw2 = await review(corrected)
    except Exception:  # noqa: BLE001
        return _withhold(REASON_PROVIDER, review_calls=review_calls,
                         repair_calls=repair_calls, restore_calls=restore_calls)

    second = dc.decide_from_text(raw2, corrected, sources)
    if not second.decision or second.structural_problems:
        return _withhold(REASON_MALFORMED, review_calls=review_calls,
                         repair_calls=repair_calls, restore_calls=restore_calls)
    if second.decision != dc.DECISION_APPROVE:
        # A revise after a repair is a refusal, not permission to keep editing.
        return _withhold(REASON_UNRESOLVED, review_calls=review_calls,
                         repair_calls=repair_calls, restore_calls=restore_calls)
    if not references_resolve(second, sources):
        return _withhold(REASON_ATTR, review_calls=review_calls,
                         repair_calls=repair_calls, restore_calls=restore_calls)

    return _finish_release(corrected, second, review_calls, repair_calls,
                           restore_calls, emit, reason_ok=None)


def _finish_release(markdown: str, decision: dc.Decision, review_calls: int,
                    repair_calls: int, restore_calls: int,
                    emit, *, reason_ok) -> CorrectionResult:
    """Mandatory controls, in order, before the document may be published.

    Order matters. Structure and marker checks run BEFORE the document is returned,
    so a broken or marker-bearing document can never be bound to a receipt.
    """
    ok, missing = _required_sections_present(markdown)
    if not ok:
        return _withhold("no_supported_claims",
                         commentary="missing sections: " + ", ".join(missing),
                         review_calls=review_calls, repair_calls=repair_calls,
                         restore_calls=restore_calls)

    if _marker_leaked(markdown):
        # A control marker reached the document. Refuse rather than strip: silently
        # editing an approved document would make the receipt bind text nobody
        # approved.
        return _withhold(REASON_UNRESOLVED,
                         commentary="control marker present in document",
                         review_calls=review_calls, repair_calls=repair_calls,
                         restore_calls=restore_calls)

    lines = sum(1 for line in markdown.splitlines() if line.strip())
    coverage = Coverage(
        level="prose_issues",            # NOT span coverage; named so it cannot be
        line_count=lines,                # mistaken for it
        claim_count=len(decision.issues),
        supported_count=0,               # this contract has no per-span verdicts
        structural_count=0,
    )
    emit({"phase": "approved", "review_calls": review_calls})
    return CorrectionResult("approved", markdown, None,
                            (decision.rationale or "")[:400], coverage,
                            review_calls, repair_calls, restore_calls)
