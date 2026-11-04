"""Typed, source-anchored skill-bridge records (Phase B).

A bridge is a **proposal**, not a fact. It says: "this candidate activity
plausibly supports this general skill label". Everything here exists to keep that
distinction enforceable:

* the record is created and numbered by the server, never by the model;
* the quoted evidence must appear verbatim in the immutable candidate source;
* a client-supplied ``origin`` never grants anything — provenance is assigned
  here, after validation;
* a proposal is *structurally* valid, not *semantically* approved. Only the
  correction reviewer can permit its use (Phase C).

The policy authority is ``docs/Plans/CogitoMetric_Official_Skill_Bridging_Plan_v1.md``
(v1.1). This module deliberately contains no network access and no taxonomy
client: grounding is supplied by the caller and merely recorded.

Policy on specificity (v1.1 clarification):

    Trend analysis from comparing sales trends   -> allowed (general label)
    Time-series analysis from measurements/time  -> allowed (general label)
    ARIMA forecasting from unspecified analysis  -> NOT established
    Python from a physics/analyst job title      -> NOT established
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from skill_inference import (
    POLICY_VERSION,
    SkillInferenceMode,
    normalize_mode,
    policy_version_for,
)

ORIGIN_RESEARCHER_INFERENCE = "researcher_inference"

# Bounds. These are design defaults, not measured provider capacity: they exist
# so one malformed response cannot become an unbounded payload, and so a record
# is rejected rather than silently truncated into a different meaning.
MAX_BRIDGES = 6
MAX_SKILL_CHARS = 120
MAX_QUOTE_CHARS = 400
MAX_RATIONALE_CHARS = 500

# A label must look like a skill name, not a sentence. Rejected early so the
# writer never receives "I also used Python extensively" as a "skill".
_SENTENCE_PUNCT = re.compile(r"[.;!?]\s")
_WHITESPACE = re.compile(r"\s+")


class BridgeValidationError(ValueError):
    """A proposed bridge could not be tied to real candidate evidence."""


@dataclass(frozen=True)
class SkillBridge:
    """One server-validated transferable-skill proposal.

    ``id`` is assigned by the server after validation, so a model or client
    cannot nominate its own identity and later claim approval for it.
    """

    id: str
    skill: str
    origin: str
    source_id: str
    evidence_quote: str
    rationale: str
    policy_version: str
    experience_id: str = ""
    taxonomy_refs: Tuple[str, ...] = field(default_factory=tuple)
    # Where the quote was found, for auditability. Offsets are code points in
    # the immutable source text, matching the correction engine's contract.
    quote_start: int = 0
    quote_end: int = 0
    # Compact record of a narrowing/removal decision (Phase C fills this in).
    decision: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "skill": self.skill,
            "origin": self.origin,
            "experience_id": self.experience_id,
            "source_id": self.source_id,
            "evidence_quote": self.evidence_quote,
            "quote_start": self.quote_start,
            "quote_end": self.quote_end,
            "rationale": self.rationale,
            "taxonomy_refs": list(self.taxonomy_refs),
            "policy_version": self.policy_version,
            "decision": self.decision,
        }


def _clean(value: Any) -> str:
    """Collapse whitespace and strip control characters from a model string."""
    if not isinstance(value, str):
        return ""
    return _WHITESPACE.sub(" ", value.replace("\x00", " ")).strip()


def _slice_candidates(source_text: str, quote: str) -> Iterable[int]:
    """Yield every offset where ``quote`` occurs in ``source_text``."""
    if not quote:
        return
    start = source_text.find(quote)
    while start != -1:
        yield start
        start = source_text.find(quote, start + 1)


def _resolve_quote(
    quote: str,
    sources: Dict[str, str],
    preferred_source_id: Optional[str],
    experience_id: str = "",
) -> Optional[Tuple[str, int, int]]:
    """Locate ``quote`` verbatim in the candidate sources.

    The preferred source wins. Otherwise the first source that contains the
    quote is used, but a quote found in *no* source is rejected: this is the rule
    that stops an inferred skill from being anchored to text the candidate never
    wrote.

    A quote that spans source documents is rejected, because a single evidence
    anchor cannot be attributed to two owners.
    """
    normalized = _clean(quote)
    if not normalized:
        return None

    order: List[str] = []
    if preferred_source_id and preferred_source_id in sources:
        order.append(preferred_source_id)
    order.extend(sid for sid in sources if sid not in order)

    for source_id in order:
        text = sources[source_id]
        # Try the cleaned quote first, then the raw form (a model may keep the
        # original line breaks in a multi-line quote).
        for candidate in (normalized, quote.strip()):
            for start in _slice_candidates(text, candidate):
                end = start + len(candidate)
                if _crosses_document_boundary(text, start, end):
                    continue
                return source_id, start, end
    return None


def _crosses_document_boundary(source_text: str, start: int, end: int) -> bool:
    """True when the span spans the multi-file join marker.

    The orchestrator joins uploads with ``\\n\\n---\\n\\n``. An anchor that
    straddles that marker would merge two documents' evidence into one claim.
    """
    return "\n\n---\n\n" in source_text[start:end]


def _is_general_label(skill: str) -> bool:
    """Heuristic: is this a skill label rather than a smuggled claim?

    This is not semantic verification — it cannot tell a good bridge from a bad
    one. It rejects shapes that are *never* a general transferable-skill label:

    * a full sentence, or a label with a category colon;
    * anything containing a digit (a metric or a rank wearing a skill's name);
    * a proficiency/rank word, which turns a label into a claim about level;
    * a single word, because a transferable-skill bridge is a *phrase*
      ("trend analysis"), while bare single words are almost always tool or
      language names ("Python", "SQL", "Kubernetes");
    * an ALL-CAPS token, which is how acronym-valued products and algorithms
      appear ("ARIMA", "AWS", "SAP").

    Anything this misses is caught later by the reviewer, which is the only
    component that can judge meaning.
    """
    if not skill or len(skill) > MAX_SKILL_CHARS:
        return False
    if _SENTENCE_PUNCT.search(skill):
        return False
    if ":" in skill:
        return False
    # A digit anywhere is a metric, a version, a team size or a rank.
    if any(ch.isdigit() for ch in skill):
        return False
    # Proficiency/rank words turn a label into an assertion about level.
    if re.search(
        r"\b(expert|advanced|proficient|proven|mastery|senior|lead|principal)\b",
        skill,
        re.IGNORECASE,
    ):
        return False
    words = [w for w in re.split(r"[\s/]+", skill) if w]
    if len(words) < 2:
        return False
    # An all-caps token of length >= 2 is an acronym: a named product,
    # algorithm or platform, not a general capability.
    if any(len(w) >= 2 and w.isupper() for w in words):
        return False
    return True


def _bridge_id(skill: str, source_id: str, quote_start: int, ordinal: int) -> str:
    """Deterministic server ID from the anchor, not from model-chosen text.

    Deterministic IDs make a re-run of the same analysis produce the same
    identities (useful for tests and for idempotent approval binding), while the
    digest keeps the ID opaque and short.
    """
    digest = hashlib.sha256(
        f"{skill.casefold()}|{source_id}|{quote_start}".encode("utf-8")
    ).hexdigest()[:10]
    return f"bridge-{ordinal:02d}-{digest}"


def build_bridges(
    proposals: Sequence[Dict[str, Any]],
    *,
    sources: Dict[str, str],
    mode: Any = None,
    policy_version: Optional[str] = None,
    taxonomy: Optional[Dict[str, Sequence[str]]] = None,
) -> Tuple[List[SkillBridge], List[Dict[str, Any]]]:
    """Validate model proposals into server-owned records.

    Returns ``(accepted, rejected)``. ``rejected`` carries only a reason code and
    an index — never the model's own unjustified text — so a rejection can be
    logged without leaking private evidence.

    In Conservative mode this returns no bridges at all: the permission does not
    exist, so a proposal is not validated into a record merely because it arrived.

    ``taxonomy`` is optional grounding: ``{skill_label_lower: [ref, ...]}``. A
    reference is only attached when the caller actually supplied it. Missing
    grounding produces an empty ``taxonomy_refs`` and is never invented.
    """
    effective = normalize_mode(mode)
    if effective is SkillInferenceMode.CONSERVATIVE:
        return [], [
            {"index": i, "reason": "mode_forbids_bridging"}
            for i, _ in enumerate(proposals or [])
        ]

    version = policy_version or policy_version_for(effective)
    accepted: List[SkillBridge] = []
    rejected: List[Dict[str, Any]] = []
    seen_anchors: set[Tuple[str, int, str]] = set()

    for index, raw in enumerate(proposals or []):
        if len(accepted) >= MAX_BRIDGES:
            rejected.append({"index": index, "reason": "too_many_bridges"})
            continue
        if not isinstance(raw, dict):
            rejected.append({"index": index, "reason": "not_an_object"})
            continue

        # The client/model may send `origin`, but it is never trusted: the value
        # is overwritten below. Reading it only to ignore it makes that explicit.
        skill = _clean(raw.get("skill"))
        quote = _clean(raw.get("evidence_quote"))
        rationale = _clean(raw.get("rationale"))
        preferred_source = _clean(raw.get("source_id")) or None
        experience_id = _clean(raw.get("experience_id"))
        # Optional, caller-supplied model citation. Discarded unless it names a
        # label the caller actually supplied in `taxonomy`.
        claimed_refs = raw.get("taxonomy_refs")

        if not skill:
            rejected.append({"index": index, "reason": "missing_skill"})
            continue
        if not _is_general_label(skill):
            rejected.append({"index": index, "reason": "not_a_general_skill_label"})
            continue
        if not quote:
            rejected.append({"index": index, "reason": "missing_evidence_quote"})
            continue
        if len(quote) > MAX_QUOTE_CHARS:
            rejected.append({"index": index, "reason": "evidence_quote_too_long"})
            continue
        if not rationale:
            rejected.append({"index": index, "reason": "missing_rationale"})
            continue
        if len(rationale) > MAX_RATIONALE_CHARS:
            rejected.append({"index": index, "reason": "rationale_too_long"})
            continue

        located = _resolve_quote(quote, sources, preferred_source, experience_id)
        if located is None:
            # The decisive rejection: an inferred skill with no verbatim
            # candidate anchor is not a bridge, it is a guess.
            rejected.append({"index": index, "reason": "quote_not_in_candidate_source"})
            continue
        source_id, start, end = located

        anchor = (skill.casefold(), start, source_id)
        if anchor in seen_anchors:
            rejected.append({"index": index, "reason": "duplicate_bridge"})
            continue
        seen_anchors.add(anchor)

        refs: Tuple[str, ...] = ()
        if taxonomy and isinstance(claimed_refs, (list, tuple)):
            allowed = {
                str(ref)
                for ref in (taxonomy.get(skill.casefold()) or ())
            }
            # Intersection only: a model may cite a ref that exists, never create
            # one.
            refs = tuple(sorted(str(ref) for ref in claimed_refs if str(ref) in allowed))
        elif taxonomy:
            refs = tuple(sorted(str(r) for r in (taxonomy.get(skill.casefold()) or ())))

        accepted.append(
            SkillBridge(
                id=_bridge_id(skill, source_id, start, len(accepted) + 1),
                skill=skill,
                origin=ORIGIN_RESEARCHER_INFERENCE,
                source_id=source_id,
                evidence_quote=sources[source_id][start:end],
                rationale=rationale,
                policy_version=version,
                experience_id=experience_id,
                taxonomy_refs=refs,
                quote_start=start,
                quote_end=end,
            )
        )

    return accepted, rejected


def bridges_for_writer(bridges: Sequence[SkillBridge]) -> List[Dict[str, Any]]:
    """Compact view handed to the writer/reviewer.

    Deliberately excludes offsets and policy plumbing: the writer needs to know
    which label is permitted *for which quoted activity*, not how the anchor is
    encoded. Offsets stay server-side for verification.
    """
    return [
        {
            "skill": bridge.skill,
            "evidence_quote": bridge.evidence_quote,
            "rationale": bridge.rationale,
        }
        for bridge in bridges
    ]


def policy_record(
    mode: Any,
    bridges: Sequence[SkillBridge],
) -> Dict[str, Any]:
    """Deterministic policy state for the approval binding (Phase D consumes it).

    Stable key order and record order make this safe to fingerprint: two runs of
    the same analysis over the same content produce an identical structure.
    """
    effective = normalize_mode(mode)
    return {
        "skill_inference_mode": effective.value,
        "skill_inference_policy_version": POLICY_VERSION
        if effective is SkillInferenceMode.IMAGINATIVE
        else policy_version_for(effective),
        "surviving_bridges": [
            {
                "id": b.id,
                "skill": b.skill,
                "origin": b.origin,
                "source_id": b.source_id,
                "policy_version": b.policy_version,
            }
            for b in sorted(bridges, key=lambda item: item.id)
        ],
    }


def source_map(source_documents: Optional[Sequence[Dict[str, Any]]]) -> Dict[str, str]:
    """Normalise the request's source documents into ``{id: text}``.

    Never merges documents: a bridge anchored in document A must remain
    attributable to A.
    """
    out: Dict[str, str] = {}
    for doc in source_documents or []:
        if not isinstance(doc, dict):
            continue
        sid = doc.get("source_id")
        text = doc.get("text")
        if isinstance(sid, str) and sid.strip() and isinstance(text, str) and text.strip():
            out[sid] = text
    return out
