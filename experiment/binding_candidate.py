"""Experimental binding LLM review; never used by the production gate.

Rules observe only. A repair is merely a proposal until an independent review of
the exact changed document approves it; attribution and support are not inferred
from a source ID or from the parser accepting a response.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Awaitable, Callable

import decision_contract as dc

_HEAD = re.compile(r'^\*\*(.+?)\*\*\s*--')


@dataclass
class BindingResult:
    original: str
    rule_observations: list[str] = field(default_factory=list)
    initial_response: str = ""
    initial_decision: str = ""
    initial_problems: list[str] = field(default_factory=list)
    proposed_document: str = ""
    repair_status: str = "not_attempted"
    re_review_response: str = ""
    re_review_decision: str = ""
    re_review_problems: list[str] = field(default_factory=list)
    release: bool = False
    release_reason: str = "not_reviewed"
    reference_level: str = "source identifiers only; semantic support unassessed"


def _bullets_by_owner(text: str) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    owner = None
    for line in text.splitlines():
        if line.startswith('## '):
            owner = None
        match = _HEAD.match(line.strip())
        if match:
            owner = match.group(1)
            if owner in owners:
                # Repeated employer names are ambiguous, not safely attributable.
                owners[owner] = []
                owner = None
            else:
                owners[owner] = []
        elif owner is not None and line.strip().startswith('- '):
            owners[owner].append(line.strip()[2:].strip())
    return owners


def _bullet_text(value: str) -> str:
    return value.strip().removeprefix('- ').strip()


def _omit_marker(value: str) -> bool:
    text = value.strip().lstrip('*_` ').lstrip('"\'“‘').lstrip('*_` ')
    return dc.is_omit_marker(text)


def references_resolve(decision: dc.Decision, sources: dict[str, str]) -> bool:
    """Check every offered ID or bare quote, not whether evidence supports a claim."""
    for issue in decision.issues:
        for reference in issue.sources:
            ref_id = reference.split(':', 1)[0].strip()
            if ref_id in sources:
                continue
            quote = ' '.join(reference.split()).casefold()
            if len(quote) > 8 and any(quote in ' '.join(text.split()).casefold()
                                      for text in sources.values()):
                continue
            return False
    return True


def safe_repair(draft: str, sources: dict[str, str], issues: list[dc.Issue]) -> tuple[str, str]:
    """Only a single exact source bullet for the uniquely identified owner may replace a single bullet.

    No multiline/source-ambiguous/omission/deletion edits. Refusing an edit is
    not a semantic judgment about its correctness. All issues must be repaired.
    """
    if len(issues) != 1:
        return draft, "ambiguous_or_multiple_issues"
    issue = issues[0]
    passage = (issue.passage or '').strip()
    correction = (issue.correction or '').strip()
    if issue.problem == dc.PROBLEM_OMITTED or passage == dc.OMITTED_MARKER:
        return draft, "omission_requires_explicit_destination"
    if not correction or not passage or _omit_marker(correction):
        return draft, "no_safe_replacement"
    # Must be one entire literal bullet in a single unambiguous owner record.
    if '\n' in passage or '\n' in correction:
        return draft, "multiline_or_commentary"
    lines = draft.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if line.strip() in {passage, '- ' + passage}]
    if len(matches) != 1:
        return draft, "passage_not_unique_literal_bullet"
    index = matches[0]
    if not lines[index].strip().startswith('- '):
        return draft, "passage_not_unique_literal_bullet"
    heads = []
    for line in lines[:index]:
        if line.startswith('## '):
            heads = []
        match = _HEAD.match(line.strip())
        if match:
            heads.append(match.group(1))
    if not heads:
        return draft, "owner_unresolved"
    owner = heads[-1]
    if sum(1 for line in lines if (m := _HEAD.match(line.strip())) and m.group(1) == owner) != 1:
        return draft, "owner_ambiguous"
    source_records = [_bullets_by_owner(text) for text in sources.values()]
    records = [record[owner] for record in source_records if owner in record]
    if len(records) != 1 or _bullet_text(correction) not in records[0]:
        return draft, "replacement_not_exact_owned_source_bullet"
    new = '- ' + _bullet_text(correction)
    line = lines[index]
    lines[index] = line[:len(line) - len(line.lstrip())] + new + ('\n' if line.endswith('\n') else '')
    result = ''.join(lines)
    if result == draft:
        return draft, "no_change"
    return result, "proposed_exact_owned_source_bullet"


async def run_binding(draft: str, sources: dict[str, str], *,
                      ask: Callable[[str, dict[str, str]], Awaitable[str]],
                      observe_rules: Callable[[str, dict[str, str]], list[str]] | None = None) -> BindingResult:
    result = BindingResult(original=draft, proposed_document=draft)
    if observe_rules is not None:
        result.rule_observations = list(observe_rules(draft, sources))
    try:
        result.initial_response = await ask(draft, sources)
        first = dc.decide_from_text(result.initial_response, draft, sources)
    except Exception as exc:
        result.release_reason = 'review_error:' + type(exc).__name__
        return result
    result.initial_decision = first.decision
    result.initial_problems = list(first.structural_problems)
    if not first.ok or not references_resolve(first, sources):
        result.release_reason = 'initial_review_unverifiable'
        return result
    if first.decision == dc.DECISION_APPROVE:
        result.release, result.release_reason = True, 'initial_llm_approve'
        return result
    if first.decision == dc.DECISION_CANNOT_ASSESS:
        result.release_reason = 'llm_cannot_assess'
        return result
    proposed, result.repair_status = safe_repair(draft, sources, first.issues)
    result.proposed_document = proposed
    if proposed == draft:
        result.release_reason = 'revision_unresolved'
        return result
    try:
        result.re_review_response = await ask(proposed, sources)
        second = dc.decide_from_text(result.re_review_response, proposed, sources)
    except Exception as exc:
        result.release_reason = 're_review_error:' + type(exc).__name__
        return result
    result.re_review_decision = second.decision
    result.re_review_problems = list(second.structural_problems)
    if second.ok and references_resolve(second, sources) and second.decision == dc.DECISION_APPROVE:
        result.release, result.release_reason = True, 'corrected_document_llm_approved'
    else:
        result.release_reason = 're_review_not_approved'
    return result