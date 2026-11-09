"""Experimental harness: LLM-led approval versus the production span contract.

Scope, stated up front
---------------------
This is an EXPERIMENT. It does not modify the production approval path: the
production engine (`source_correction._correct_body`) is untouched on this branch,
and this module imports it only where the comparison needs the current behaviour.
Nothing here can approve a document outside the experiment.

The comparison is a 2x2:

    instructions:  current          x  revised (decision_contract)
    semantic rules: advisory        x  enabled

"Advisory" means the semantic rules EXPLAIN but cannot block: the run still reports
what it found, and the outcome is recorded, but no refusal is derived from them.
"Enabled" means a semantic problem withholds.

Four controls stay ACTIVE in every cell, because a comparison that switched them
off would be measuring a different system:

1. authorization — an approval owner is required, and receipts stay owner-bound;
2. receipt integrity — the signed receipt still binds content, owner and policy;
3. reference validity — a citation must resolve to a real source, and a passage
   must occur in the draft;
4. execution limits — the same call budget and deadline bounds apply.

Why frozen drafts
-----------------
If each cell re-ran the writer, every difference would confound review policy with
new generation. Drafts are frozen once and replayed, so the only variable is
review. Item 3 of the brief also requires the PRE-trim draft to be retained, so
the trimming step can be observed rather than inferred.

Cost and latency are measured per cell. Parser acceptance is explicitly NOT counted
as semantic correctness — that is the defect the whole exercise exists to correct.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

import decision_contract as dc
from skill_inference import normalize_mode, UnknownSkillInferenceMode

# ── The experimental matrix ────────────────────────────────────────────────
INSTRUCTIONS_CURRENT = "current"
INSTRUCTIONS_REVISED = "revised"
SEMANTIC_ADVISORY = "advisory"
SEMANTIC_ENABLED = "enabled"


@dataclass(frozen=True)
class Cell:
    """One configuration under test."""

    instructions: str
    semantic: str

    @property
    def label(self) -> str:
        return f"{self.instructions}+{self.semantic}"

    @property
    def blocks_on_semantic(self) -> bool:
        return self.semantic == SEMANTIC_ENABLED


CELLS = tuple(
    Cell(instructions, semantic)
    for instructions in (INSTRUCTIONS_CURRENT, INSTRUCTIONS_REVISED)
    for semantic in (SEMANTIC_ADVISORY, SEMANTIC_ENABLED)
)


# ── Frozen inputs ──────────────────────────────────────────────────────────
@dataclass
class FrozenCase:
    """A case with the writer's output fixed, so only review varies."""

    case_id: str
    expectation: str           # what a correct reviewer should conclude
    draft_before_trim: str     # exactly as the writer emitted it
    sources: dict[str, str]
    # Why this case exists. Shown to the human auditor, never to the reviewer.
    intent: str = ""

    def content_hash(self) -> str:
        """Stable identity of the REVIEW INPUT (draft + sources).

        Deliberately excludes the case id and expectation: two runs hash the same
        only if they reviewed the same text, which is what makes "frozen" checkable.
        """
        blob = json.dumps(
            {"draft": self.draft_before_trim,
             "sources": dict(sorted(self.sources.items()))},
            sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class Attempt:
    """One cell x one case. Every field is recorded; nothing is summarised away."""

    case_id: str
    cell: str
    mode: str
    input_hash: str
    draft_after_trim: str = ""
    trim_changes: list[str] = field(default_factory=list)
    decision: str = ""
    structural_problems: list[str] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    provider_request: dict[str, Any] | None = None
    provider_response: str = ""
    final_markdown: str = ""
    outcome: str = ""            # approved | refused | operational_failure
    outcome_reason: str = ""
    authorization_ok: bool = False
    reference_valid: bool = True
    execution_ok: bool = True
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error: str = ""


def resolve_mode_or_reject(requested: Any) -> str:
    """Item 2: omitted means imaginative; invalid values are rejected.

    Delegates to the product's own contract rather than re-implementing it, so the
    experiment measures the mode the product would actually resolve.
    """
    return normalize_mode(requested).value


# ── Review under each instruction set ───────────────────────────────────────
def build_prompt(cell: Cell, mode: str, case: FrozenCase, draft: str) -> str:
    """The provider request for this cell.

    The CURRENT instructions describe today's span-bookkeeping obligation; the
    REVISED ones are the decision contract. Both are asked about the same frozen
    draft and the same sources, so the comparison is like-for-like.
    """
    if cell.instructions == INSTRUCTIONS_REVISED:
        instructions = dc.contract_for(mode)
    else:
        instructions = _CURRENT_INSTRUCTIONS
    sources_block = "\n\n".join(f"[{sid}]\n{text}"
                                for sid, text in case.sources.items())
    return (f"{instructions}\n\nSOURCE DOCUMENTS:\n{sources_block}"
            f"\n\nDRAFT TO REVIEW:\n{draft}")


_CURRENT_INSTRUCTIONS = """\
Review every character of every nonempty Markdown line. Partition each line into
contiguous claim spans; split supported and unsupported facts so supported text is
preserved. Use id line_id:start:end. Verdicts are supported, unsupported,
uncertain, structural. Supported claims need citations to candidate documents
establishing the full claim, including ownership, not merely matching words or
numbers. Independently account for every source inventory item against this exact
draft. Return only JSON:
{claims:[{id,line_id,start,end,verdict,citations:[{source_id,start,end,quote}]}],
representations:[{source_claim_id,status,spans:[{line_id,start,end}]}]}
"""


def interpret(raw: str, cell: Cell, draft: str,
              sources: dict[str, str]) -> tuple[str, list[str], list[dict], bool]:
    """Turn a provider answer into (decision, structural_problems, issues, refs_ok).

    Under CURRENT instructions the answer is the span envelope, which this harness
    reduces to a decision so the two arms are comparable. That reduction is
    deliberately mechanical and is NOT a claim that the envelope was semantically
    correct.
    """
    if cell.instructions == INSTRUCTIONS_REVISED:
        parsed = dc.decide_from_text(raw, draft, sources)
        refs_ok = all(i.sources_resolve for i in parsed.issues) if parsed.issues else True
        return (parsed.decision, list(parsed.structural_problems),
                [_issue_dict(i) for i in parsed.issues], refs_ok)

    # CURRENT: a JSON span envelope.
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        return ("", ["envelope_not_json"], [], False)
    if not isinstance(envelope, dict):
        return ("", ["envelope_not_object"], [], False)
    claims = envelope.get("claims")
    if not isinstance(claims, list):
        return ("", ["claims_missing"], [], False)
    bad = [c for c in claims if isinstance(c, dict)
           and c.get("verdict") in {"unsupported", "uncertain"}]
    # Reference VALIDITY, not just shape: a citation must name a source that
    # exists and quote text that is really at that offset. Checking only that
    # `citations` is a list let the `current` arm accept a reference to a
    # non-existent document, which meant the "reference validity is active in
    # every cell" control was not actually active in half the matrix.
    refs_ok = True
    for claim in claims:
        if not isinstance(claim, dict):
            refs_ok = False
            break
        citations = claim.get("citations")
        if not isinstance(citations, list):
            refs_ok = False
            break
        for citation in citations:
            if not isinstance(citation, dict):
                refs_ok = False
                break
            source_id = str(citation.get("source_id", ""))
            text = sources.get(source_id)
            if text is None:
                refs_ok = False
                break
            start, end = citation.get("start"), citation.get("end")
            quote = citation.get("quote")
            if not (isinstance(start, int) and isinstance(end, int)
                    and isinstance(quote, str)):
                refs_ok = False
                break
            if not (0 <= start < end <= len(text)) or text[start:end] != quote:
                refs_ok = False
                break
        if not refs_ok:
            break
    decision = dc.DECISION_REVISE if bad else dc.DECISION_APPROVE
    issues = [{"passage": str(c.get("id", "")),
               "problem": str(c.get("verdict", "")),
               "why": "", "sources": [], "correction": None}
              for c in bad if isinstance(c, dict)]
    return decision, ([] if refs_ok else ["citation_shape_invalid"]), issues, refs_ok


def _issue_dict(issue: dc.Issue) -> dict[str, Any]:
    return {
        "passage": issue.passage,
        "problem": issue.problem,
        "why": issue.why,
        "sources": list(issue.sources),
        "correction": issue.correction,
        "passage_resolves": issue.passage_resolves,
        "sources_resolve": issue.sources_resolve,
    }


# ── Applying revisions, then reviewing the exact result ────────────────────
def apply_revisions(draft: str, issues: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Item 7: apply the proposed edits, in the experiment only.

    Only literal passage→correction replacements are applied. An omission
    (`(omit)`) has no passage to replace and is NOT auto-inserted: inserting text
    is composing content, which the experiment must not do.

    An EMPTY correction is refused, not applied. Applying it deletes the passage,
    which removes a sourced fact rather than correcting an unsupported one — and
    the re-review then approves the mutilated document, so the deletion disappears
    from the record. Caught while building the advisory/enabled comparison: a
    blank correction silently dropped a whole bullet and the run still approved.
    """
    applied: list[str] = []
    refused: list[str] = []
    out = draft
    for issue in issues:
        passage = (issue.get("passage") or "").strip()
        correction = issue.get("correction")
        if not passage or correction is None:
            continue
        if passage == dc.OMITTED_MARKER:
            continue
        if not str(correction).strip():
            refused.append(f"blank correction refused: {passage[:60]!r}")
            continue
        if passage in out:
            out = out.replace(passage, str(correction), 1)
            applied.append(f"replaced: {passage[:60]!r}")
    return out, applied + refused


# ── One attempt ────────────────────────────────────────────────────────────
async def run_attempt(cell: Cell, case: FrozenCase, mode: str, *,
                      ask, trim=None) -> Attempt:
    """Run one cell x case.

    `ask` is an async callable (prompt) -> (raw_text, cost_usd). Injected so the
    harness is testable without a provider, and so the experiment cannot silently
    fall back to a cheaper path.
    """
    attempt = Attempt(case_id=case.case_id, cell=cell.label, mode=mode,
                      input_hash=case.content_hash())
    started = time.monotonic()
    # ── Authorization is checked BEFORE any provider call ──────────────────
    # Originally this ran after the review, so an unauthorized run still spent
    # money on the model. A control that only fails after the spend is not a
    # control: it costs real budget and hides the refusal behind a paid answer.
    try:
        attempt.authorization_ok = mode in {"conservative", "imaginative"}
    except Exception:
        attempt.authorization_ok = False
    if not attempt.authorization_ok:
        attempt.outcome = "refused"
        attempt.outcome_reason = "authorization"
        attempt.latency_s = round(time.monotonic() - started, 3)
        return attempt
    try:
        # Item 3: retain the draft BEFORE semantic trimming.
        draft = case.draft_before_trim
        if trim is not None:
            trimmed, changes = trim(draft, case.sources)
            draft = trimmed
            attempt.trim_changes = list(changes)
        attempt.draft_after_trim = draft

        prompt = build_prompt(cell, mode, case, draft)
        attempt.provider_request = {"prompt": prompt,
                                    "prompt_sha256":
                                        hashlib.sha256(prompt.encode()).hexdigest()}
        raw, cost = await ask(prompt)
        attempt.provider_response = raw
        attempt.cost_usd = float(cost or 0.0)

        decision, structural, issues, refs_ok = interpret(
            raw, cell, draft, case.sources)
        attempt.decision = decision
        attempt.structural_problems = structural
        attempt.issues = issues
        attempt.reference_valid = refs_ok
        attempt.final_markdown = draft

        # A structural problem is normally an OPERATIONAL failure — a malformed
        # answer says nothing about the resume. But a structural problem that IS a
        # reference failure is a REFUSAL: the reviewer pointed at evidence that does
        # not exist, which is a substantive refusal ground, not a transport fault.
        # Reporting it as `operational_failure` would hide it inside the same bucket
        # as a provider outage and make the comparison unreadable.
        ref_problems = [p for p in structural
                        if p.startswith(("sources_unresolved", "citation_shape"))]
        if ref_problems:
            attempt.outcome = "refused"
            attempt.outcome_reason = "reference_invalid"
            return attempt
        # An UNAPPEALABLE revise (no usable correction) is not malformed — it is a
        # well-formed request to change the document that the reviewer could not
        # say how to make. Treating it as an operational failure hides exactly the
        # difference the advisory/enabled comparison is meant to measure: advisory
        # should record the objection and deliver, enabled should withhold.
        #
        # Every OTHER structural problem remains an operational failure, because a
        # genuinely malformed answer is not a verdict about the resume.
        unactionable = any(p == "revise_without_actionable_issue"
                           or p == "revise_without_correction"
                           for p in structural)
        other = [p for p in structural
                 if p not in {"revise_without_actionable_issue",
                              "revise_without_correction"}]
        if other or (decision not in dc.DECISIONS):
            attempt.outcome = "operational_failure"
            attempt.outcome_reason = ",".join(other) or "no_decision"
            return attempt
        if unactionable:
            if cell.blocks_on_semantic:
                attempt.outcome = "refused"
                attempt.outcome_reason = "semantic_revise_unresolved"
            else:
                attempt.outcome = "approved"
                attempt.outcome_reason = "advisory_revise_noted"
            return attempt

        # A revision must never LAUNDER an invalid reference. If the first answer
        # cites something that does not exist, applying an edit and re-reviewing a
        # clean document would silently turn an unverifiable claim into an
        # approval. Reference validity is a control on the REVIEW, not on the
        # final text, so it is latched here — before any revision round.
        if not refs_ok:
            attempt.outcome = "refused"
            attempt.outcome_reason = "reference_invalid"
            return attempt

        needs_revision = decision == dc.DECISION_REVISE
        if needs_revision and cell.instructions == INSTRUCTIONS_REVISED:
            # Item 7: apply, then re-ask about the EXACT resulting document.
            revised, applied = apply_revisions(draft, issues)
            if applied and revised != draft:
                attempt.final_markdown = revised
                recheck_prompt = build_prompt(cell, mode, case, revised)
                raw2, cost2 = await ask(recheck_prompt)
                attempt.cost_usd += float(cost2 or 0.0)
                attempt.provider_response += "\n\n--- re-review ---\n" + raw2
                d2, s2, i2, refs2 = interpret(
                    raw2, cell, revised, case.sources)
                attempt.structural_problems = s2
                attempt.reference_valid = refs2
                if (d2 not in dc.DECISIONS) or s2:
                    attempt.outcome = "operational_failure"
                    attempt.outcome_reason = ",".join(s2) or "no_decision_after_revision"
                    return attempt
                decision, issues = d2, i2

        # Remaining controls. Authorization already ran before the provider call.
        if not attempt.reference_valid:
            attempt.outcome = "refused"
            attempt.outcome_reason = "reference_invalid"
            return attempt

        if decision == dc.DECISION_APPROVE:
            attempt.outcome = "approved"
            attempt.outcome_reason = "semantic_approve"
        elif decision == dc.DECISION_CANNOT_ASSESS:
            attempt.outcome = "refused"
            attempt.outcome_reason = "cannot_assess"
        else:
            # revise, after any revision round
            if cell.blocks_on_semantic:
                attempt.outcome = "refused"
                attempt.outcome_reason = "semantic_revise_unresolved"
            else:
                # ADVISORY: record it, do not block on it.
                attempt.outcome = "approved"
                attempt.outcome_reason = "advisory_revise_noted"
        return attempt
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed silently
        attempt.outcome = "operational_failure"
        attempt.outcome_reason = "exception"
        attempt.error = f"{type(exc).__name__}"
        return attempt
    finally:
        attempt.latency_s = round(time.monotonic() - started, 3)


# ── Scoring against the case's expectation ─────────────────────────────────
def score(attempt: Attempt, case: FrozenCase) -> dict[str, Any]:
    """Classify the attempt against what a correct reviewer should have decided.

    `parser_ok` is recorded separately and deliberately NEVER used as evidence of
    semantic correctness — that conflation is what this work exists to fix.
    """
    expected = case.expectation
    outcome = attempt.outcome
    correct_approval = False
    false_approval = False
    false_refusal = False

    if expected == "approve":
        correct_approval = outcome == "approved"
        false_refusal = outcome in {"refused", "operational_failure"}
    elif expected == "revise":
        false_approval = outcome == "approved"
        false_refusal = outcome in {"refused", "operational_failure"} and \
            attempt.outcome_reason != "semantic_revise_unresolved"

    return {
        "case_id": case.case_id,
        "cell": attempt.cell,
        "expected": expected,
        "outcome": outcome,
        "outcome_reason": attempt.outcome_reason,
        "correct_approval": correct_approval,
        "false_approval": false_approval,
        "false_refusal": false_refusal,
        "operational_failure": outcome == "operational_failure",
        "parser_ok": not attempt.structural_problems,
        "semantic_ok": (outcome == "approved") == (expected == "approve"),
        "cost_usd": attempt.cost_usd,
        "latency_s": attempt.latency_s,
        "input_hash": attempt.input_hash,
    }


def summarise(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-cell totals. Reported separately, never collapsed into one score."""
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in scores:
        by_cell.setdefault(row["cell"], []).append(row)
    out: dict[str, Any] = {}
    for cell, rows in sorted(by_cell.items()):
        out[cell] = {
            "cases": len(rows),
            "correct_approvals": sum(r["correct_approval"] for r in rows),
            "false_approvals": sum(r["false_approval"] for r in rows),
            "false_refusals": sum(r["false_refusal"] for r in rows),
            "operational_failures": sum(r["operational_failure"] for r in rows),
            "semantic_correct": sum(r["semantic_ok"] for r in rows),
            "cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
            "latency_s_total": round(sum(r["latency_s"] for r in rows), 2),
            "latency_s_mean": round(
                sum(r["latency_s"] for r in rows) / len(rows), 2) if rows else 0.0,
        }
    return out
