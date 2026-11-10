"""Runner v2 — same 2x2 matrix, corrected omission handling and telemetry.

Relationship to v1
------------------
`experiment/runner.py` is unchanged and remains the v1 record. This module is the
versioned correction. It shares the *design* (frozen drafts, 2x2 matrix, four
always-active controls) but differs in six specific, listed ways, each traceable to
a measured defect:

1. **Omission handling identifies source AND destination without editing.** v1's
   `apply_revisions` skipped the omitted marker entirely, so an omission report was
   recorded and then dropped. v2 records an `OmissionReport` with the cited source,
   the anchor in the draft (the passage the record should follow), and the missing
   text — and applies NOTHING. Item 6 requires exactly this: name it, do not
   auto-insert. Ambiguous text (a passage that occurs more than once, or an anchor
   that cannot be located unambiguously) is reported as `ambiguous` and also not
   applied.

2. **`finish_reason`, usage, latency and cost are retained per call** (see
   `provider_v2`). v1 kept cost and latency only, so truncation had to be inferred
   from byte length after the fact.

3. **Truncation is its own outcome.** `TruncatedResponse` becomes
   `operational:truncation` and the attempt is marked `complete=False`. It is never
   scored as a semantic result, satisfying item 5.

4. **Reference outcomes are three-valued.** `matches` / `offsets_off` /
   `fabricated` / `unknown_source`, per `replay_v2.resolve_citation`. An offset
   mismatch is a convention defect, not a refusal.

5. **Re-review is bounded and recorded.** A revision round is a second paid call;
   its result and its telemetry are kept separately so a "correct after re-review"
   is distinguishable from "correct first time".

6. **Budget is enforced in the runner**, not by convention. `RunBudget` refuses to
   start a call that would exceed the agreed ceiling, so the ceiling cannot be
   overrun by a retry loop.

Nothing here changes production behaviour. Production approval remains untouched.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import decision_contract as dc
from experiment.case_audit import employers_in
from experiment.replay_v2 import (
    CAT_REF_CITES_DRAFT,
    CAT_REF_CONVENTION,
    CAT_REF_FABRICATED,
    CAT_REF_UNKNOWN_SOURCE,
)
from experiment.runner import (
    CELLS,
    INSTRUCTIONS_CURRENT,
    INSTRUCTIONS_REVISED,
    SEMANTIC_ADVISORY,
    Cell,
    FrozenCase,
    SEMANTIC_ENABLED,
    _CURRENT_INSTRUCTIONS,
    resolve_mode_or_reject,
)
from experiment.replay_v2 import resolve_citation

# ── Outcomes. Recorded verbatim; no retry-until-lucky. ──────────────────────
OUTCOME_APPROVED = "approved"
OUTCOME_REFUSED = "refused"
OUTCOME_OPERATIONAL = "operational_failure"
# A distinct outcome for "the reviewer's evidence did not verify".
#
# This is NOT a semantic refusal. The reviewer may have been right about the
# document and simply cited it badly; the delivered outcome is 'withheld', but the
# reason is evidence integrity, not a judgement. Item 3 of the brief forbids filing
# these as semantic refusals, and v1 did exactly that (its only completed legacy
# answer was recorded `refused/reference_invalid` when all 10 of its quotes were
# real source text at wrong offsets).
#
# Keeping it separate means a reader can see whether an arm withheld because it
# judged the document defective, or because its citation machinery is unreliable.
OUTCOME_REFERENCE_FAILURE = "reference_failure"

REASON_APPROVE = "semantic_approve"
REASON_CANNOT_ASSESS = "cannot_assess"
REASON_REVISE_UNRESOLVED_ENABLED = "semantic_revise_unresolved"
REASON_REVISE_NOTED_ADVISORY = "advisory_revise_noted"
REASON_REVISED_AND_APPROVED = "revised_then_approved"
REASON_TRUNCATED = "truncation"
REASON_REFERENCE_FABRICATED = "reference_fabricated"
REASON_REFERENCE_UNKNOWN = "reference_unknown_source"
REASON_REFERENCE_CITES_DRAFT = "reference_cites_draft"
REASON_REFERENCE_CONVENTION = "reference_convention"
REASON_EXCEPTION = "exception"
# Transport/provider failures are NOT harness defects and must not be confused
# with them. A 504 says nothing about the adapter; the same call can succeed a
# minute later. Measured on probe f (2026-11-09): one 504 was reported as a probe
# failure, which would have stopped a run for a reason unrelated to the experiment.
REASON_TRANSPORT = "transport_error"


@dataclass
class RunBudget:
    """A hard ceiling on provider spend. Enforced BEFORE each call, not after.

    v1 had no ceiling; a 3-attempt retry loop per call x 24+ attempts could in
    principle spend several times the intended amount with nothing in the record
    saying so. A budget that is only checked afterwards is a report, not a limit.
    """

    max_calls: int
    max_cost_usd: float
    calls_made: int = 0
    cost_usd: float = 0.0

    def can_call(self) -> tuple[bool, str]:
        if self.calls_made >= self.max_calls:
            return False, f"call budget exhausted ({self.calls_made}/{self.max_calls})"
        if self.cost_usd >= self.max_cost_usd:
            return False, (f"cost budget exhausted "
                           f"({self.cost_usd:.4f}/{self.max_cost_usd:.2f} USD)")
        return True, ""

    def record(self, cost: float) -> None:
        self.calls_made += 1
        self.cost_usd += float(cost or 0.0)

    def to_dict(self) -> dict:
        return {"max_calls": self.max_calls, "max_cost_usd": self.max_cost_usd,
                "calls_made": self.calls_made, "cost_usd": round(self.cost_usd, 6)}


@dataclass
class OmissionReport:
    """A missing source-backed fact. Identified, never auto-inserted.

    Item 6 asks for "the cited source and destination". `source_id` is where the
    fact came from; `destination` says where it belongs, with `destination_basis`
    recording HOW that was determined so a reader can judge the inference instead
    of taking it on trust:

        anchor_in_draft     the reviewer pointed at real draft text
        source_order        inferred from the neighbouring records in the source
        undeterminable      the answer and the sources do not say

    `applied` is ALWAYS False. The harness does not compose résumé content: an
    automatic insertion would mean the experiment fabricated text and then graded
    its own fabrication as fidelity.
    """

    source_id: str = ""
    missing_text: str = ""
    anchor_in_draft: str = ""
    destination: str = ""
    destination_basis: str = "undeterminable"
    cites_real_source: bool = False
    anchor_resolves: bool = False
    anchor_occurrences: int = 0
    ambiguous: bool = False
    applied: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _destination_from_source_order(missing_text: str, sources: dict[str, str],
                                   draft: str) -> tuple[str, str]:
    """Where does the missing record belong, judged by its NEIGHBOURS in the sources?

    A missing record has no draft position by definition, so "the exact draft text
    affected" cannot locate it. The sources do record its position, though: if the
    record sits between two employers, it belongs between them in the draft too.

    Neighbours are collected across ALL supplied documents, not just the one the
    fact was cited from. A supplement often holds a single record, so its only
    neighbours are in the main document — restricting the search to the cited
    document made the destination unknowable for the very case that needs it.

    Returns (destination, basis). Names real neighbouring records rather than an
    offset, because a fabricated offset would look like evidence.
    """
    from experiment.case_audit import sections_in
    ordered: list[str] = []
    for source_text in sources.values():
        for body in sections_in(source_text).values():
            for name in employers_in(body):
                if name not in ordered:
                    ordered.append(name)
    draft_employers = list(employers_in(draft))
    if not ordered or not draft_employers:
        return "", "undeterminable"

    # Which record does the missing text describe? Markdown is optional here: a
    # reviewer restating a record often drops the `**` emphasis, so the bold-only
    # regex would miss the very text it needs to locate. Fall back to a name match.
    target = ""
    for name in employers_in(missing_text or ""):
        if name in ordered:
            target = name
            break
    if not target:
        for name in ordered:
            if name.casefold() in (missing_text or "").casefold():
                target = name
                break
    if not target:
        return "", "undeterminable"
    index = ordered.index(target)

    before = next((name for name in reversed(ordered[:index]) if name in draft), "")
    after = next((name for name in ordered[index + 1:] if name in draft), "")
    if before and after:
        return f"between {before!r} and {after!r}", "source_order"
    if before:
        return f"after {before!r}", "source_order"
    if after:
        return f"before {after!r}", "source_order"
    return "", "undeterminable"


_TOKEN_RE = None


def _content_tokens(text: str) -> set[str]:
    """Distinctive words of a passage, for a grounding test that ignores markup.

    Markup and inflections are stripped so that

        - Maintained parts availability records for 7 service branches.

    and the source's

        - Maintain parts availability records for 7 service branches.

    compare as the same evidence. A byte comparison would call a correctly grounded
    omission ungrounded purely because the reviewer wrote "Maintained" for
    "Maintain" — and the guard's job is to detect INVENTION, not tense.
    """
    global _TOKEN_RE
    if _TOKEN_RE is None:
        import re
        _TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9/%.]*")
    raw = _TOKEN_RE.findall(text or "")
    out = set()
    for token in raw:
        cleaned = token.strip(".-").lower()
        if len(cleaned) < 4 and not any(ch.isdigit() for ch in cleaned):
            continue
        # Strip a trailing inflection so maintain/maintained/maintaining agree.
        for suffix in ("ing", "ed", "es", "s"):
            if cleaned.endswith(suffix) and len(cleaned) - len(suffix) >= 4:
                cleaned = cleaned[: -len(suffix)]
                break
        if cleaned:
            out.add(cleaned)
    return out


# How much of a quoted omission must be traceable to the cited source.
#
# 0.7 is a deliberately conservative floor. A reviewer quoting real evidence in
# its own words scores near 1.0; an invented sentence about something the source
# never discusses scores near 0. Anything in between is genuinely unclear and is
# reported as ungrounded, which is the safe direction: the note says the harness
# could not verify the citation, and the omission is still not applied.
_GROUNDING_FLOOR = 0.7


def _grounded_in_source(missing: str, source_text: str) -> bool:
    """Is the quoted omission actually traceable to this source?

    Token overlap rather than substring equality, because a reviewer restates the
    missing record rather than copying it byte-for-byte. The floor is set high
    enough that an invented claim cannot pass simply by sharing function words
    with the source.
    """
    wanted = _content_tokens(missing)
    if not wanted:
        return False
    available = _content_tokens(source_text)
    overlap = len(wanted & available) / len(wanted)
    if overlap < _GROUNDING_FLOOR:
        return False
    # A quoted FIGURE must appear in the source. Numbers are the one kind of token
    # a writer cannot paraphrase, so a digit that is absent is strong evidence the
    # quoted fact is not in this document.
    missing_digits = {t for t in wanted if any(ch.isdigit() for ch in t)}
    if missing_digits and not missing_digits <= available:
        return False
    return True


def describe_omission(issue: dict[str, Any], draft: str,
                      sources: dict[str, str]) -> OmissionReport:
    """Identify where a missing fact came from and where it belongs.

    Deliberately does NOT edit the draft. It answers three checkable questions:

      * is the missing text really in a cited source?   -> cites_real_source
      * does the anchor occur in the draft, and once?   -> anchor_resolves, count
      * is the situation ambiguous?                    -> ambiguous

    Ambiguity is reported rather than resolved. If the anchor text occurs twice,
    inserting after "the" match would silently pick one; the honest output is
    "I cannot tell where this goes", which is a question for the writer, not a
    substitution the harness should make on its own.
    """
    report = OmissionReport()
    missing = (issue.get("correction") or "").strip()
    passage = (issue.get("passage") or "").strip()
    references = list(issue.get("sources") or [])

    # The missing text: a real correction, or (for a bare omission) the passage the
    # reviewer quoted from the source.
    missing_text = missing if missing and missing != dc.OMITTED_MARKER else ""
    if not missing_text and passage and passage != dc.OMITTED_MARKER:
        missing_text = passage
    report.missing_text = missing_text

    # Which source actually contains it? Verification, not trust.
    for reference in references:
        ref_id = str(reference).split(":")[0].strip()
        text = sources.get(ref_id)
        if text is None:
            continue
        if report.source_id == "":
            report.source_id = ref_id  # named, resolution not yet proven
        if _grounded_in_source(missing_text, text):
            report.source_id = ref_id
            report.cites_real_source = True
            break

    # The destination. Three ways to know it, in order of directness.
    if passage and passage != dc.OMITTED_MARKER:
        flat_passage = " ".join(passage.split())
        flat_draft = " ".join(draft.split())
        occurrences = flat_draft.count(flat_passage) if flat_passage else 0
        report.anchor_in_draft = passage
        report.anchor_occurrences = occurrences
        if occurrences == 1:
            report.anchor_resolves = True
            report.destination = passage
            report.destination_basis = "anchor_in_draft"
        elif occurrences > 1:
            report.ambiguous = True
            report.note = (f"anchor occurs {occurrences} times; destination "
                           f"ambiguous, not applied")
        # A passage that occurs ZERO times is usually the reviewer quoting SOURCE
        # text for a missing fact — the normal shape for an omission. Fall through
        # to the source-order inference rather than calling it a failure.

    if not report.anchor_resolves:
        destination, basis = _destination_from_source_order(
            missing_text, sources, draft)
        if destination:
            report.destination = destination
            report.destination_basis = basis
            report.anchor_resolves = True
            note = f"destination inferred from {basis}: {destination}"
            report.note = f"{report.note}; {note}" if report.note else note
        elif not report.note:
            report.note = ("no anchor in draft; destination not determinable "
                           "from answer")

    if not report.cites_real_source:
        report.note = (report.note + "; " if report.note else "") + (
            "cited source does not contain the quoted text")
    report.applied = False
    return report


def apply_revisions_v2(draft: str, issues: list[dict[str, Any]],
                       sources: dict[str, str]) -> tuple[str, list[dict]]:
    """Apply unambiguous corrections; report omissions without applying them.

    Differences from v1, each deliberate:

    * An omission produces an `OmissionReport` and leaves the draft untouched.
    * A correction is located with the emphasis-tolerant `locate_passage`, so a
      reviewer that restated a bolded record without its Markdown still produces an
      applicable edit. The edit always targets REAL draft text.
    * A passage that occurs more than once is NOT applied: the harness cannot know
      which occurrence the reviewer meant, and choosing one would be inventing an
      edit.
    * A blank correction is refused (v1 behaviour kept — applying it deletes a
      sourced fact and the re-review then approves the mutilation).
    """
    log: list[dict] = []
    out = draft
    for issue in issues:
        passage = (issue.get("passage") or "").strip()
        correction = issue.get("correction")

        if issue.get("problem") == dc.PROBLEM_OMITTED or passage == dc.OMITTED_MARKER:
            report = describe_omission(issue, out, sources)
            log.append({"kind": "omission", **report.to_dict()})
            continue  # NEVER applied. See docstring.
        if not passage or correction is None:
            continue
        if not str(correction).strip():
            log.append({"kind": "refused", "reason": "blank correction",
                        "passage": passage[:80]})
            continue
        # A correction that is an omit marker (with or without trailing commentary)
        # is an OMIT, never replacement text. Writing "(omit)" or "(omit) (Note: …)"
        # into the résumé would overwrite a sourced bullet with a note about it.
        # Measured on the v2 probe (2026-11-09) before this guard existed.
        if dc.is_omit_marker(str(correction)):
            report = describe_omission(issue, out, sources)
            log.append({"kind": "omission", "reason": "correction was an omit "
                        "marker, not replacement text", **report.to_dict()})
            continue

        # Resolve to text that really exists in the draft.
        located = dc.locate_passage(passage, out)
        if located is None:
            log.append({"kind": "refused", "reason": "passage not in draft",
                        "passage": passage[:80]})
            continue
        occurrences = out.count(located)
        if occurrences > 1:
            log.append({"kind": "refused",
                        "reason": f"passage occurs {occurrences} times",
                        "passage": passage[:80]})
            continue
        out = out.replace(located, str(correction), 1)
        log.append({"kind": "replaced", "passage": located[:80],
                    "correction": str(correction)[:80]})
    return out, log


# ── Attempt record ─────────────────────────────────────────────────────────
@dataclass
class AttemptV2:
    case_id: str
    cell: str
    mode: str
    input_hash: str = ""
    draft_before_trim: str = ""
    draft_after_trim: str = ""
    trim_changes: list[str] = field(default_factory=list)
    decision: str = ""
    structural_problems: list[str] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    omissions: list[dict[str, Any]] = field(default_factory=list)
    revision_log: list[dict[str, Any]] = field(default_factory=list)
    provider_request: dict[str, Any] | None = None
    provider_response: str = ""
    recheck_request: dict[str, Any] | None = None
    recheck_response: str = ""
    final_markdown: str = ""
    outcome: str = ""
    outcome_reason: str = ""
    complete: bool = True           # False when truncated / never finished
    authorization_ok: bool = False
    reference_verdicts: dict[str, int] = field(default_factory=dict)
    reference_valid: bool = True
    execution_ok: bool = True
    telemetry: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _issue_dict(issue: dc.Issue) -> dict[str, Any]:
    return {
        "passage": issue.passage,
        "problem": issue.problem,
        "why": issue.why,
        "sources": list(issue.sources),
        "correction": issue.correction,
        "passage_resolves": issue.passage_resolves,
        "sources_resolve": issue.sources_resolve,
        "reports_omission": issue.reports_omission,
    }


def build_prompt(cell: Cell, mode: str, case: FrozenCase, draft: str) -> str:
    """Identical to v1's prompt construction, kept bit-compatible on purpose.

    A prompt change between versions would make the two runs incomparable for a
    reason unrelated to the hypothesis. Only the OUTPUT BUDGET changed.
    """
    if cell.instructions == INSTRUCTIONS_REVISED:
        instructions = dc.contract_for(mode)
    else:
        instructions = _CURRENT_INSTRUCTIONS
    sources_block = "\n\n".join(f"[{sid}]\n{text}"
                                for sid, text in case.sources.items())
    return (f"{instructions}\n\nSOURCE DOCUMENTS:\n{sources_block}"
            f"\n\nDRAFT TO REVIEW:\n{draft}")


def interpret_v2(raw: str, cell: Cell, draft: str,
                 sources: dict[str, str]) -> tuple[str, list[str], list[dict],
                                                   dict[str, int], bool]:
    """(decision, structural_problems, issues, reference_verdicts, refs_valid)."""
    if cell.instructions == INSTRUCTIONS_REVISED:
        parsed = dc.decide_from_text(raw, draft, sources)
        verdicts: dict[str, int] = {}
        for issue in parsed.issues:
            for reference in issue.sources or []:
                ref_id = reference.split(":")[0].strip()
                if ref_id in sources:
                    verdicts["resolves"] = verdicts.get("resolves", 0) + 1
                elif len(reference) > 8 and reference.casefold() in " ".join(
                        sources.values()).casefold():
                    verdicts["quote_resolves"] = verdicts.get("quote_resolves", 0) + 1
                else:
                    verdicts["unresolved"] = verdicts.get("unresolved", 0) + 1
        refs_ok = not verdicts.get("unresolved")
        return (parsed.decision, list(parsed.structural_problems),
                [_issue_dict(i) for i in parsed.issues], verdicts, refs_ok)

    # CURRENT: JSON span envelope, with three-valued citation outcomes.
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        return "", ["envelope_not_json"], [], {}, False
    if not isinstance(envelope, dict):
        return "", ["envelope_not_object"], [], {}, False
    claims = envelope.get("claims")
    if not isinstance(claims, list):
        return "", ["claims_missing"], [], {}, False

    verdicts: dict[str, int] = {}
    bad: list[dict[str, Any]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            verdicts["malformed"] = verdicts.get("malformed", 0) + 1
            continue
        if claim.get("verdict") in {"unsupported", "uncertain"}:
            bad.append(claim)
        for citation in claim.get("citations") or []:
            outcome = resolve_citation(citation, sources, draft)
            verdicts[outcome] = verdicts.get(outcome, 0) + 1

    refs_ok = not (verdicts.get("fabricated") or verdicts.get("unknown_source")
                   or verdicts.get("malformed") or verdicts.get("cites_draft"))
    decision = dc.DECISION_REVISE if bad else dc.DECISION_APPROVE
    issues = [{"passage": str(c.get("id", "")),
               "problem": str(c.get("verdict", "")),
               "why": "", "sources": [], "correction": None,
               "passage_resolves": True, "sources_resolve": True,
               "reports_omission": False}
              for c in bad]
    structural = [] if refs_ok else ["citation_shape_invalid"]
    return decision, structural, issues, verdicts, refs_ok


async def run_attempt_v2(cell: Cell, case: FrozenCase, mode: str, *,
                         ask, budget: RunBudget | None = None,
                         trim=None) -> AttemptV2:
    """One cell x one case, with telemetry and an enforced budget."""
    attempt = AttemptV2(case_id=case.case_id, cell=cell.label, mode=mode,
                        input_hash=case.content_hash())
    attempt.draft_before_trim = case.draft_before_trim
    started = time.monotonic()
    try:
        attempt.authorization_ok = mode in {"conservative", "imaginative"}
        if not attempt.authorization_ok:
            attempt.outcome = OUTCOME_REFUSED
            attempt.outcome_reason = "authorization"
            return attempt

        draft = case.draft_before_trim
        if trim is not None:
            trimmed, changes = trim(draft, case.sources)
            draft = trimmed
            attempt.trim_changes = list(changes)
        attempt.draft_after_trim = draft

        if budget is not None:
            allowed, why = budget.can_call()
            if not allowed:
                attempt.outcome = OUTCOME_OPERATIONAL
                attempt.outcome_reason = f"budget_refused:{why}"
                attempt.complete = False
                return attempt

        prompt = build_prompt(cell, mode, case, draft)
        attempt.provider_request = {
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }
        result = await ask(prompt, label=f"{cell.label}/{case.case_id}")
        if budget is not None:
            budget.record(result.cost_usd)
        attempt.provider_response = result.text
        attempt.cost_usd = result.cost_usd
        attempt.telemetry.append({"phase": "review", **result.to_dict()})

        decision, structural, issues, verdicts, refs_ok = interpret_v2(
            result.text, cell, draft, case.sources)
        attempt.decision = decision
        attempt.structural_problems = structural
        attempt.issues = issues
        attempt.reference_verdicts = verdicts
        attempt.reference_valid = refs_ok
        attempt.final_markdown = draft

        # Evidence that does not exist is a REFUSAL ground. Wrong offsets are NOT:
        # the evidence was real, the convention differed (see replay_v2).
        #
        # `cites_draft` is also a refusal ground, and deliberately a DISTINCT one:
        # the reviewer quoted the document under review and called it source
        # evidence. The text exists, so this is not fabrication; the pointer is
        # simply not evidence, so it cannot approve. Measured on the v2 probe:
        # 5 of 20 citations in one answer were draft text (2026-11-09).
        if verdicts.get("fabricated") or verdicts.get("unknown_source") \
                or verdicts.get("cites_draft"):
            attempt.outcome = OUTCOME_REFERENCE_FAILURE
            if verdicts.get("fabricated"):
                attempt.outcome_reason = REASON_REFERENCE_FABRICATED
            elif verdicts.get("cites_draft"):
                attempt.outcome_reason = REASON_REFERENCE_CITES_DRAFT
            else:
                attempt.outcome_reason = REASON_REFERENCE_UNKNOWN
            return attempt
        if verdicts.get("offsets_off") and not verdicts.get("resolves"):
            attempt.outcome_reason = REASON_REFERENCE_CONVENTION

        # ── Classifying structural problems ────────────────────────────────
        #
        # Three genuinely different failures used to share one bucket:
        #
        #   UNREADABLE    — code could not parse the answer at all.
        #                   -> operational_failure. Not a verdict on anything.
        #   EVIDENCE      — the reviewer cited something that does not exist.
        #                   -> reference_failure. A distinct outcome.
        #   UNAPPLICABLE  — the reviewer found a real problem but the passage it
        #                   named cannot be located exactly, so no edit can be
        #                   made safely.
        #                   -> NOT an operational failure, and NOT an approval.
        #
        # `passage_not_in_draft` belongs in UNAPPLICABLE, and that is a change from
        # v1 where it was an operational failure. Measured across five v2 probes
        # (2026-11-09): the reviewer repeatedly described the affected record
        # ("Included Ridgeline Equipment Services as a separate role with dates")
        # instead of transcribing it. The objection is about the document; only the
        # pointer is imprecise. Calling that an operational failure discards a
        # correct review, while applying the correction to a guessed span would be
        # worse — so advisory records it, enabled withholds, and nothing is edited.
        _UNAPPLICABLE = {"passage_not_in_draft", "revise_without_actionable_issue",
                         "revise_without_correction", "no_applicable_correction"}
        unapplicable = [p for p in structural
                        if p.split(":")[0] in _UNAPPLICABLE]
        other = [p for p in structural
                 if p.split(":")[0] not in _UNAPPLICABLE]
        if other or decision not in dc.DECISIONS:
            attempt.outcome = OUTCOME_OPERATIONAL
            attempt.outcome_reason = ",".join(other) or "no_decision"
            attempt.complete = bool(decision)
            return attempt
        if unapplicable:
            # Recorded verbatim so the reader can see the pointer was imprecise,
            # and never applied. Advisory: note and deliver. Enabled: withhold.
            attempt.outcome = (OUTCOME_REFUSED if cell.blocks_on_semantic
                               else OUTCOME_APPROVED)
            attempt.outcome_reason = (REASON_REVISE_UNRESOLVED_ENABLED
                                      if cell.blocks_on_semantic
                                      else REASON_REVISE_NOTED_ADVISORY)
            attempt.structural_problems = list(unapplicable)
            return attempt

        if decision == dc.DECISION_REVISE and cell.instructions == INSTRUCTIONS_REVISED:
            revised, log = apply_revisions_v2(draft, issues, case.sources)
            attempt.revision_log = log
            attempt.omissions = [e for e in log if e.get("kind") == "omission"]
            if revised != draft:
                attempt.final_markdown = revised
                if budget is not None:
                    allowed, why = budget.can_call()
                    if not allowed:
                        # A partial revision with no re-review is NOT an approval.
                        attempt.outcome = OUTCOME_OPERATIONAL
                        attempt.outcome_reason = f"budget_refused_recheck:{why}"
                        attempt.complete = False
                        return attempt
                recheck = build_prompt(cell, mode, case, revised)
                attempt.recheck_request = {
                    "prompt": recheck,
                    "prompt_sha256": hashlib.sha256(recheck.encode()).hexdigest(),
                }
                result2 = await ask(recheck, label=f"{cell.label}/{case.case_id}/recheck")
                if budget is not None:
                    budget.record(result2.cost_usd)
                attempt.cost_usd += result2.cost_usd
                attempt.recheck_response = result2.text
                attempt.telemetry.append({"phase": "recheck", **result2.to_dict()})
                d2, s2, i2, v2, refs2 = interpret_v2(
                    result2.text, cell, revised, case.sources)
                attempt.reference_verdicts = v2
                attempt.reference_valid = refs2
                if (d2 not in dc.DECISIONS) or s2:
                    attempt.outcome = OUTCOME_OPERATIONAL
                    attempt.outcome_reason = ",".join(s2) or "no_decision_after_revision"
                    return attempt
                decision, issues = d2, i2
                if decision == dc.DECISION_APPROVE:
                    attempt.outcome = OUTCOME_APPROVED
                    attempt.outcome_reason = REASON_REVISED_AND_APPROVED
                    return attempt

        if decision == dc.DECISION_APPROVE:
            attempt.outcome = OUTCOME_APPROVED
            attempt.outcome_reason = REASON_APPROVE
        elif decision == dc.DECISION_CANNOT_ASSESS:
            attempt.outcome = OUTCOME_REFUSED
            attempt.outcome_reason = REASON_CANNOT_ASSESS
        else:
            attempt.outcome = (OUTCOME_REFUSED if cell.blocks_on_semantic
                               else OUTCOME_APPROVED)
            attempt.outcome_reason = (REASON_REVISE_UNRESOLVED_ENABLED
                                      if cell.blocks_on_semantic
                                      else REASON_REVISE_NOTED_ADVISORY)
        return attempt
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
        truncated = type(exc).__name__ == "TruncatedResponse"
        message = str(exc)
        transport = (not truncated) and any(
            marker in message for marker in
            ("provider exhausted retries", "http_429", "http_500", "http_502",
             "http_503", "http_504", "TimeoutError", "ClientConnectorError",
             "ServerDisconnectedError", "no choices in response"))
        attempt.outcome = OUTCOME_OPERATIONAL
        if truncated:
            attempt.outcome_reason = REASON_TRUNCATED
        elif transport:
            attempt.outcome_reason = REASON_TRANSPORT
        else:
            attempt.outcome_reason = REASON_EXCEPTION
        attempt.complete = False
        attempt.execution_ok = False
        attempt.error = f"{type(exc).__name__}: {message[:300]}"
        return attempt
    finally:
        attempt.latency_s = round(time.monotonic() - started, 3)


# ── Scoring ────────────────────────────────────────────────────────────────
def score_v2(attempt: AttemptV2, case: FrozenCase) -> dict[str, Any]:
    """Score against the preregistered expectation, keeping axes separate.

    A truncated or otherwise incomplete attempt is NOT scored as a false refusal.
    That was the mistake item 5 forbids: the legacy arm looked worse than it was
    purely because its answers were cut off.

    A REFERENCE FAILURE is likewise not a false refusal. The reviewer's evidence
    did not verify, which is a defect in how it argued rather than a verdict about
    the document. It is counted in its own column so the two cannot be added
    together — item 3's requirement, and the specific error that made v1's single
    completed legacy answer look like a semantic refusal.
    """
    expected = case.expectation
    outcome = attempt.outcome
    complete = attempt.complete
    reference_failure = outcome == OUTCOME_REFERENCE_FAILURE

    is_approval = outcome == OUTCOME_APPROVED
    correct_approval = False
    false_approval = False
    false_refusal = False

    if not complete:
        pass  # no semantic score at all
    elif reference_failure:
        pass  # evidence did not verify: reported separately, not a judgement
    elif expected == "approve":
        correct_approval = is_approval
        # Only an actual REFUSAL is a false refusal. An operational failure is its
        # own column and is not evidence about faithfulness in either direction.
        #
        # Measured while reporting the v2 matrix (2026-11-09): counting an
        # operational failure as a false refusal put the same attempt in two
        # columns and inflated the failure count for the arm that hit more parser
        # problems. That is precisely the conflation item 5 of the brief forbids.
        false_refusal = outcome == OUTCOME_REFUSED
    elif expected == "revise":
        false_approval = is_approval
        false_refusal = (outcome == OUTCOME_REFUSED
                         and attempt.outcome_reason
                         != REASON_REVISE_UNRESOLVED_ENABLED)

    return {
        "case_id": case.case_id,
        "cell": attempt.cell,
        "expected": expected,
        "outcome": outcome,
        "outcome_reason": attempt.outcome_reason,
        "complete": complete,
        "scored": complete and not reference_failure,
        "reference_failure": reference_failure,
        "correct_approval": correct_approval,
        "false_approval": false_approval,
        "false_refusal": false_refusal,
        "operational_failure": outcome == OUTCOME_OPERATIONAL,
        "parser_ok": not attempt.structural_problems,
        # A verdict was actually reached only if the attempt completed, avoided a
        # reference failure, AND produced a semantic outcome.
        #
        # Measured on the v2 matrix (2026-11-09): `unsupported_method` expects
        # `revise` and came back `operational_failure`, which scored as
        # semantically CORRECT because "not approved" was compared against
        # "expects revise". The harness had failed to parse the answer; crediting
        # it with a correct judgement is the same category conflation this work
        # exists to remove. An operational failure reaches NO verdict, so its
        # semantic score is None, not True.
        "semantic_ok": ((is_approval == (expected == "approve"))
                        if (complete and not reference_failure
                            and outcome in {OUTCOME_APPROVED, OUTCOME_REFUSED})
                        else None),
        "reference_verdicts": attempt.reference_verdicts,
        "cost_usd": attempt.cost_usd,
        "latency_s": attempt.latency_s,
        "input_hash": attempt.input_hash,
    }


def summarise_v2(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-cell totals, with incomplete attempts counted apart from judgements."""
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in scores:
        by_cell.setdefault(row["cell"], []).append(row)
    out: dict[str, Any] = {}
    for cell, rows in sorted(by_cell.items()):
        scored = [r for r in rows if r["scored"]]
        out[cell] = {
            "cases": len(rows),
            "scored": len(scored),
            "incomplete": sum(1 for r in rows if not r["complete"]),
            "reference_failures": sum(1 for r in rows if r["reference_failure"]),
            "correct_approvals": sum(r["correct_approval"] for r in rows),
            "false_approvals": sum(r["false_approval"] for r in rows),
            "false_refusals": sum(r["false_refusal"] for r in rows),
            "operational_failures": sum(r["operational_failure"] for r in rows),
            "semantic_correct": sum(1 for r in scored if r["semantic_ok"]),
            "cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
            "latency_s_total": round(sum(r["latency_s"] for r in rows), 2),
            "latency_s_mean": round(
                sum(r["latency_s"] for r in rows) / len(rows), 2) if rows else 0.0,
        }
    return out
