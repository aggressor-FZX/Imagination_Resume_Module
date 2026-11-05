"""Bridge decision lifecycle: accept / narrow / remove (Phase C.7).

A bridge is never approved by publishing a verdict alongside unchanged text. The
only content that gets signed is the content that survived, and every decision
leaves a compact, auditable record:

    accept   the exact reviewed label stands, in this draft, at this anchor
    narrow   a weaker neutral replacement is authorised; the original label is NOT
    remove   the label must go; the surrounding facts must stay

Two rules make this safe rather than decorative:

1. **Narrowing cannot escalate.** The replacement must pass the same general-label
   validation as an original proposal, so a reviewer cannot "narrow" a label into
   ``ARIMA forecasting``, a tool name or a proficiency claim.
2. **Removal never destroys source facts.** A decision is applied to a *line*, and
   removing a standalone skill item is safe; removing an embedded phrase from a
   line that also carries candidate facts is refused, because the repair path that
   would rewrite it is the place that must preserve them.

Nothing here calls a model, and nothing here grants publication. It records the
decision, validates the wording, and yields the text the existing review will see
next. Publication remains the receipt's job (Phase D).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from skill_bridge import _is_general_label  # shared validator, one definition


# Decision values, deliberately few. "uncertain" is not a decision: an unresolved
# bridge keeps its ordinary line-level handling (repair or withhold) instead of
# silently becoming a removal.
ACCEPT = "accept"
NARROW = "narrow"
REMOVE = "remove"

_VALID_DECISIONS = frozenset({ACCEPT, NARROW, REMOVE})

# Reason codes, fixed strings only. Free text from a reviewer or a caller is never
# echoed into a log line.
REASON_CODES = frozenset({
    "supported_by_anchor",
    "too_specific",
    "adds_tool",
    "adds_credential",
    "adds_proficiency",
    "overstates_activity",
    "not_anchored",
    "duplicate",
})

MAX_LABEL_CHARS = 120


class BridgeDecisionError(ValueError):
    """A decision could not be applied without violating the policy."""


@dataclass(frozen=True)
class BridgeDecision:
    """One recorded decision about one bridge occurrence."""

    bridge_id: str
    decision: str
    skill: str
    reason: str = ""
    replacement: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "bridge_id": self.bridge_id,
            "decision": self.decision,
            "skill": self.skill,
            "reason": self.reason,
            "replacement": self.replacement,
        }


def _as_decision(raw: Any) -> BridgeDecision:
    """Coerce one decision payload, refusing anything that is not a mapping.

    Callers pass a mix of dicts and ``BridgeDecision`` objects; anything else is a
    caller error, not a document error, and must be reported as such rather than
    raising ``TypeError`` out of the middle of a text transform.
    """
    if isinstance(raw, BridgeDecision):
        return raw
    if not isinstance(raw, dict):
        raise BridgeDecisionError("decision must be an object")
    return validate_decision(**raw)


def _normalise_decision(value: Any) -> str:
    text = value.strip().lower() if isinstance(value, str) else ""
    if text not in _VALID_DECISIONS:
        raise BridgeDecisionError("decision must be accept, narrow or remove")
    return text


def validate_decision(
    *,
    bridge_id: Any,
    decision: Any,
    skill: Any,
    reason: Any = "",
    replacement: Any = "",
) -> BridgeDecision:
    """Validate one decision without applying it.

    Rejecting here (rather than at apply time) keeps the failure attributable to
    the decision instead of to the document.
    """
    if not isinstance(bridge_id, str) or not bridge_id.strip():
        raise BridgeDecisionError("bridge_id is required")
    if not isinstance(skill, str) or not skill.strip():
        raise BridgeDecisionError("skill is required")

    chosen = _normalise_decision(decision)
    code = reason.strip() if isinstance(reason, str) else ""
    if code and code not in REASON_CODES:
        # A caller-invented reason is not an error worth failing the request over,
        # but it must not be recorded as if it were a known code.
        code = ""

    replacement_text = replacement.strip() if isinstance(replacement, str) else ""

    if chosen == NARROW:
        if not replacement_text:
            raise BridgeDecisionError("narrow requires a replacement label")
        if len(replacement_text) > MAX_LABEL_CHARS:
            raise BridgeDecisionError("replacement label is too long")
        # The decisive guard: a narrowing cannot introduce a specific algorithm,
        # a named product, a metric or a proficiency claim.
        if not _is_general_label(replacement_text):
            raise BridgeDecisionError(
                "replacement must be a general transferable-skill label"
            )
        # A "narrowing" that is not narrower is an escalation attempt.
        if replacement_text.casefold() == skill.strip().casefold():
            raise BridgeDecisionError("replacement is identical to the original")
    elif replacement_text:
        raise BridgeDecisionError("only narrow may supply a replacement")

    return BridgeDecision(
        bridge_id=bridge_id.strip(),
        decision=chosen,
        skill=skill.strip(),
        reason=code,
        replacement=replacement_text,
    )


def _word_bounded(label: str) -> str:
    return r"(?<![a-z0-9])" + re.escape(" ".join(label.casefold().split())) + r"(?![a-z0-9])"


def find_occurrences(text: str, label: str) -> List[Tuple[int, int]]:
    """Every word-bounded occurrence of ``label`` in ``text``, case-folded.

    Word-bounded on purpose: ``"R"`` must not match inside ``"arc flash"`` and
    ``"analysis"`` must not match inside ``"analyses"``.
    """
    if not isinstance(text, str) or not label or not label.strip():
        return []
    folded = " ".join(text.casefold().split())
    if not folded:
        return []
    # Searching the folded view would misreport offsets for the original string,
    # so the caller gets offsets in the FOLDED text. Application callers that need
    # original offsets resolve the label within one line, where the two views have
    # the same shape because folding only collapses runs of whitespace.
    return [(m.start(), m.end()) for m in re.finditer(_word_bounded(label), folded)]


def _line_is_only_the_label(line: str, label: str) -> bool:
    """True when the visible content of ``line`` is exactly the label.

    Removing or rewriting such a line cannot destroy a candidate fact, because
    there is nothing else on it.
    """
    residue = re.sub(r"[*_`#]", " ", line or "")
    residue = re.sub(r"^\s*[-\u2022*+]\s*", " ", residue)
    residue = " ".join(residue.casefold().split())
    residue = re.sub(_word_bounded(label), " ", residue)
    residue = re.sub(r"[^a-z0-9]+", " ", residue).strip()
    return not residue


def _has_embedded_occurrence(lines: Sequence[str], label: str) -> bool:
    """True when the label appears on a line that carries other content too."""
    for line in lines:
        if not find_occurrences(line, label):
            continue
        if not _line_is_only_the_label(line, label):
            return True
    return False


def apply_decisions(
    markdown: str,
    decisions: Sequence[Any],
) -> Tuple[str, List[BridgeDecision], List[Dict[str, str]]]:
    """Apply decisions to a draft, returning (text, applied, rejected).

    ``rejected`` carries only a bridge id and a fixed reason code — never document
    text — so a refusal is loggable without leaking the candidate's material.

    Rules:

    * **accept** is a no-op on text; it only records that the label stands.
    * **narrow** rewrites the label in place, and only on lines whose remainder is
      not a candidate fact (a standalone skill item). A narrowing inside prose is
      refused rather than silently rewriting a sentence the reviewer did not
      author.
    * **remove** deletes a standalone skill line. A label embedded in a line that
      carries other content is refused: the repair path must preserve those facts,
      and this function cannot know what they are.

    A refused decision leaves the text untouched. It is NOT a publication
    decision: unresolved content still goes through the existing review, which is
    what withholds or repairs it.
    """
    if not isinstance(markdown, str):
        raise BridgeDecisionError("markdown must be a string")

    applied: List[BridgeDecision] = []
    rejected: List[Dict[str, str]] = []
    lines = markdown.splitlines(keepends=True)
    text = markdown

    for raw in decisions or []:
        try:
            decision = _as_decision(raw)
        except (BridgeDecisionError, TypeError):
            bridge_id = ""
            if isinstance(raw, dict):
                bridge_id = str(raw.get("bridge_id") or "")
            rejected.append({"bridge_id": bridge_id, "reason": "invalid_decision"})
            continue

        target = decision.skill if decision.decision != NARROW else decision.replacement
        matches = [
            (index, line) for index, line in enumerate(lines)
            if _line_is_only_the_label(line, decision.skill)
        ]
        # A "removal" that left the label in prose would not have removed it. So a
        # decision is only applicable when EVERY occurrence is standalone; if any
        # occurrence is embedded in a sentence that also carries facts, the whole
        # decision is refused and left to the existing repair path.
        embedded_elsewhere = _has_embedded_occurrence(lines, decision.skill)

        if decision.decision == ACCEPT:
            if not matches:
                # Nothing to accept: the label is not a standalone item. That is
                # not an error, but the decision authorises nothing either.
                rejected.append({"bridge_id": decision.bridge_id,
                                 "reason": "label_not_standalone"})
                continue
            applied.append(decision)
            continue

        if not matches:
            rejected.append({"bridge_id": decision.bridge_id,
                             "reason": "embedded_label_requires_repair"})
            continue

        if embedded_elsewhere:
            # Refuse the whole decision rather than applying it partially: a
            # partial removal reports success while the label survives.
            rejected.append({"bridge_id": decision.bridge_id,
                             "reason": "embedded_label_requires_repair"})
            continue

        if decision.decision == NARROW:
            for index, line in matches:
                ending = line[len(line.rstrip("\r\n")):]
                body = line.rstrip("\r\n")
                marker = re.match(r"^(\s*[-*\u2022+]\s*)", body)
                prefix = marker.group(1) if marker else ""
                new_body = prefix + decision.replacement + ending
                lines[index] = new_body
            applied.append(decision)
            continue

        # REMOVE
        for index in sorted((i for i, _ in matches), reverse=True):
            del lines[index]
        applied.append(decision)

    text = "".join(lines)
    return text, applied, rejected


def surviving_labels(decisions: Sequence[Any]) -> List[str]:
    """The labels a decision set leaves authorised, in stable order.

    Used by the approval path: an accepted or narrowed bridge survives (with the
    narrowed wording), a removed one does not, and an invalid one contributes
    nothing.
    """
    out: List[str] = []
    seen: set[str] = set()
    for raw in decisions or []:
        try:
            decision = _as_decision(raw)
        except (BridgeDecisionError, TypeError):
            continue
        if decision.decision == REMOVE:
            continue
        label = decision.replacement if decision.decision == NARROW else decision.skill
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(label)
    return out
