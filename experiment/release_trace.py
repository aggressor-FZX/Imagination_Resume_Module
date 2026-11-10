"""Full approval path with an auditable trace: review → repair → re-review → release.

The path this implements, exactly as specified:

    Generated draft + immutable candidate sources
        ↓
    Gemma semantic review
        ├─ approve      → hard controls → release
        ├─ revise       → explicit correction → review the EXACT corrected document
        │                                    └─ release only after a valid approval
        └─ cannot_assess / invalid response → no release

Division of responsibility, enforced here rather than described:

- The MODEL decides faithfulness, permitted paraphrase, material omission and
  attribution. This module never second-guesses a valid approval.
- CODE enforces authorization, reference validity, document identity, required
  structure, receipt integrity and execution limits.
- A `revise` is never permission to release: release requires a NEW valid `approve`
  of the corrected text.
- No meaning-changing cleanup happens after approval. The receipt is bound to the
  exact approved Markdown, its canonical export payload, the owner and the policy
  state, so a later edit invalidates it.

Why a trace object rather than just a boolean
---------------------------------------------
A release is only as trustworthy as its audit trail. Every step is recorded —
the document text at each stage, the decision, the repair status and diff, the
re-review, and the receipt — so a reader can verify that the document which was
exported is the one that was approved, rather than being told so.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import dataclass, field

import decision_contract as dc
from experiment.binding_candidate import references_resolve
from experiment.repair import apply_repair

# Core sections a delivered résumé must contain. Code checks presence; it does not
# generate filler to satisfy itself.
REQUIRED_SECTIONS = ("Professional Summary", "Professional Experience")


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sections_present(markdown: str) -> dict[str, bool]:
    """Which required headings exist. Presence only — never repaired by adding text."""
    return {name: f"## {name}" in (markdown or "") for name in REQUIRED_SECTIONS}


def _diff(before: str, after: str) -> str:
    return "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(),
                                          "before", "after", lineterm="", n=1))


@dataclass
class ReleaseTrace:
    """Every step from the frozen draft to the released document."""

    case_id: str
    original: str
    original_sha: str = ""
    # First review
    first_decision: str = ""
    first_valid: bool = False
    first_problems: list[str] = field(default_factory=list)
    first_response: str = ""
    first_issues: list[dict] = field(default_factory=list)
    # Repair
    repair_status: str = "not_attempted"
    proposed_document: str = ""
    proposed_sha: str = ""
    repair_diff: str = ""
    # Re-review
    re_review_decision: str = ""
    re_review_valid: bool = False
    re_review_problems: list[str] = field(default_factory=list)
    re_review_document_sha: str = ""
    re_review_response: str = ""
    # Controls
    references_valid: bool = True
    sections: dict[str, bool] = field(default_factory=dict)
    sections_ok: bool = True
    # Release
    released: bool = False
    release_reason: str = ""
    released_document_sha: str = ""
    receipt: str = ""
    receipt_reason: str = ""
    provider: str = ""
    # Audit
    steps: list[str] = field(default_factory=list)

    def record(self, step: str) -> None:
        self.steps.append(step)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


async def run_release(case_id: str, draft: str, sources: dict[str, str], *,
                      ask, issue_receipt=None, verify_receipt=None,
                      owner: str | None = None,
                      policy_state: dict | None = None) -> ReleaseTrace:
    """Drive one document through the full path.

    `ask(document, sources)` returns the reviewer's raw text for that exact document.
    `issue_receipt` / `verify_receipt` are injected so production can supply the real
    signing functions and tests can supply stubs; when absent, release is reported
    without a receipt and the reason says so.
    """
    trace = ReleaseTrace(case_id=case_id, original=draft,
                         original_sha=_sha(draft))
    trace.record("start")

    # ── First review ────────────────────────────────────────────────────────
    try:
        trace.first_response = await ask(draft, sources)
    except Exception as exc:  # noqa: BLE001
        trace.release_reason = f"review_error:{type(exc).__name__}"
        trace.record("first_review_error")
        return trace

    first = dc.decide_from_text(trace.first_response, draft, sources)
    trace.first_decision = first.decision
    trace.first_valid = bool(first.decision) and not first.structural_problems
    trace.first_problems = list(first.structural_problems)
    trace.first_issues = [
        {"problem": i.problem, "passage": i.passage, "why": i.why,
         "correction": i.correction, "sources": list(i.sources)}
        for i in first.issues]
    trace.record(f"first_review:{first.decision or 'none'}")

    if not trace.first_valid:
        # An unusable answer is an operational outcome, never an approval.
        trace.release_reason = "initial_review_unverifiable"
        trace.record("withheld:unverifiable")
        return trace

    if first.decision == dc.DECISION_CANNOT_ASSESS:
        trace.release_reason = "llm_cannot_assess"
        trace.record("withheld:cannot_assess")
        return trace

    if not references_resolve(first, sources):
        trace.references_valid = False
        trace.release_reason = "references_unresolved"
        trace.record("withheld:references")
        return trace

    if first.decision == dc.DECISION_APPROVE:
        return _release(trace, draft, owner, policy_state,
                        issue_receipt, verify_receipt,
                        reason="initial_llm_approve")

    # ── Revise: propose a correction ────────────────────────────────────────
    proposed, trace.repair_status = apply_repair(draft, sources, first.issues)
    trace.proposed_document = proposed
    trace.proposed_sha = _sha(proposed)
    trace.record(f"repair:{trace.repair_status}")

    if proposed == draft:
        trace.release_reason = "revision_unresolved"
        trace.record("withheld:no_applicable_repair")
        return trace

    trace.repair_diff = _diff(draft, proposed)

    # ── Re-review the EXACT corrected document ──────────────────────────────
    # The document handed to the reviewer is `proposed` and nothing else.
    try:
        trace.re_review_response = await ask(proposed, sources)
    except Exception as exc:  # noqa: BLE001
        trace.release_reason = f"re_review_error:{type(exc).__name__}"
        trace.record("re_review_error")
        return trace

    trace.re_review_document_sha = _sha(proposed)
    assert trace.re_review_document_sha == trace.proposed_sha, \
        "re-review must receive exactly the proposed document"

    second = dc.decide_from_text(trace.re_review_response, proposed, sources)
    trace.re_review_decision = second.decision
    trace.re_review_valid = bool(second.decision) and not second.structural_problems
    trace.re_review_problems = list(second.structural_problems)
    trace.record(f"re_review:{second.decision or 'none'}")

    if not trace.re_review_valid or second.decision != dc.DECISION_APPROVE:
        trace.release_reason = "re_review_not_approved"
        trace.record("withheld:re_review_not_approved")
        return trace
    if not references_resolve(second, sources):
        trace.references_valid = False
        trace.release_reason = "re_review_references_unresolved"
        trace.record("withheld:re_review_references")
        return trace

    return _release(trace, proposed, owner, policy_state,
                    issue_receipt, verify_receipt,
                    reason="corrected_document_llm_approved")


def _release(trace: ReleaseTrace, document: str, owner, policy_state,
             issue_receipt, verify_receipt, *, reason: str) -> ReleaseTrace:
    """Hard controls, then receipt, then release.

    Order matters: structure is checked BEFORE a receipt is minted, so a document
    missing a core section can never carry a valid approval. And the receipt is
    issued for THIS document only, so a later edit invalidates it.
    """
    trace.sections = sections_present(document)
    trace.sections_ok = all(trace.sections.values())
    trace.record(f"structure:{'ok' if trace.sections_ok else 'missing'}")
    if not trace.sections_ok:
        missing = [k for k, v in trace.sections.items() if not v]
        trace.release_reason = "missing_required_sections:" + ",".join(missing)
        trace.record("withheld:structure")
        return trace

    if issue_receipt is None:
        # No signer available: report the approval without claiming a release that
        # cannot be verified downstream.
        trace.release_reason = reason + ";no_receipt_signer"
        trace.record("refused:no_receipt_signer")
        return trace

    # The receipt binds the RENDERER-PARSED object, not just the markdown, and
    # `verify_receipt` requires that object back (`receipt_render_payload_missing`
    # otherwise). Building it here from the approved markdown guarantees the export
    # is derived from the approved text and not from an earlier draft.
    try:
        from approved_resume import export_payload
        render_payload = export_payload(document)
    except Exception as exc:  # noqa: BLE001
        trace.release_reason = reason + f";export_payload_unavailable:{type(exc).__name__}"
        trace.record("refused:no_export_payload")
        return trace

    # The verdict is a TYPED value from publication_gate, not the literal string
    # "approved". `verify_receipt` compares against `Verdict.PASS_TIER_A.value`;
    # passing "approved" mints a receipt that verifies as `receipt_not_approved`.
    # Imported rather than copied so the two can never drift.
    from publication_gate import Verdict
    receipt = issue_receipt(markdown=document, verdict=Verdict.PASS_TIER_A.value,
                            owner=owner, policy_state=policy_state,
                            approved_payload=render_payload)
    if not receipt:
        trace.release_reason = reason + ";receipt_unavailable"
        trace.record("refused:receipt_unavailable")
        return trace

    trace.receipt = receipt
    ok, why = verify_receipt(receipt, markdown=document, owner=owner,
                             render_payload=render_payload,
                             policy_state=policy_state)
    trace.receipt_reason = why
    if not ok:
        trace.release_reason = reason + f";receipt_invalid:{why}"
        trace.record("refused:receipt_invalid")
        return trace

    trace.released = True
    trace.release_reason = reason
    trace.released_document_sha = _sha(document)
    trace.record("released")
    return trace


def trace_markdown(trace: ReleaseTrace) -> str:
    """Human-readable audit of one document's path."""
    lines = [
        f"# Release trace — `{trace.case_id}`", "",
        f"- original sha256: `{trace.original_sha[:16]}`",
        f"- first decision: `{trace.first_decision or 'none'}` "
        f"(valid: {trace.first_valid})",
        f"- repair: `{trace.repair_status}`",
    ]
    if trace.proposed_sha:
        lines.append(f"- proposed sha256: `{trace.proposed_sha[:16]}`")
    if trace.re_review_decision:
        lines.append(f"- re-review decision: `{trace.re_review_decision}` "
                     f"(valid: {trace.re_review_valid})")
    lines += [
        f"- sections present: `{trace.sections}`",
        f"- **released: {trace.released}** — `{trace.release_reason}`",
    ]
    if trace.released_document_sha:
        lines.append(f"- released document sha256: "
                     f"`{trace.released_document_sha[:16]}`")
    if trace.receipt:
        lines.append(f"- receipt verified: {trace.receipt_reason}")
    lines += ["", "## Steps", ""]
    lines += [f"{n}. {step}" for n, step in enumerate(trace.steps, 1)]
    if trace.repair_diff:
        lines += ["", "## Repair diff", "", "```diff", trace.repair_diff, "```"]
    return "\n".join(lines) + "\n"
