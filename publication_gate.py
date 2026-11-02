"""Single authoritative publication gate.

ONE decision: may generated resume content be shown, exported, cached or saved?

Every route must call this. Before this module existed the decision logic was
duplicated across three response-assembly sites in `app.py`, and only one of them
was guarded — `POST /api/analysis/multi-file` returned
`processing_status = COMPLETED` unconditionally, so a resume containing invented
figures was publishable from that route.

Design contract (milestone 2 — fail closed):

    publish is allowed ONLY when verification positively succeeded.

Anything else — absent, unchecked, raising, malformed, empty, conversational,
truncated, fallback-shaped, missing a required fact, or carrying an unsupported
claim — blocks publication. A checker that cannot run is not a pass.

Deliberate non-goals:
  * does not mutate the generated text (never silently deletes a claim)
  * does not invent a replacement resume
  * does not touch the candidate's own source facts
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Verdict(str, Enum):
    """Typed verification outcome.

    The wire format stays a string (existing consumers and saved reports compare
    strings), but the DECISION must never be made by string matching. Previously
    `verdict.startswith("PASS")` accepted both the qualified and a bare "PASS",
    which is how the fail-open shipped. Typed comparison removes that class.
    """
    PASS_TIER_A = "PASS (tier A only)"
    FAIL = "FAIL"


class Reason(str, Enum):
    """Machine-readable block reasons. Presentation lives elsewhere."""
    FALLBACK_OUTPUT = "fallback_output"
    TRUNCATED_OUTPUT = "truncated_output"
    EMPTY_OUTPUT = "empty_output"
    CONVERSATIONAL_OUTPUT = "conversational_output"
    BELOW_MIN_WORDS = "below_min_words"
    MISSING_SECTION = "missing_section"
    PLACEHOLDER_LEAK = "placeholder_leak"
    VERIFICATION_MISSING = "verification_missing"
    VERIFICATION_EMPTY = "verification_empty"
    VERIFICATION_NOT_RUN = "verification_not_run"
    VERIFICATION_INCONCLUSIVE = "verification_inconclusive"
    UNSUPPORTED_CLAIMS = "unsupported_claims"
    ENTITY_EVIDENCE_MISSING = "entity_evidence_missing"
    ENTITY_EVIDENCE_EMPTY = "entity_evidence_empty"
    REQUIRED_FACTS_MISSING = "required_facts_missing"


class EntityEvidence(str, Enum):
    """Three-state evidence availability.

    Collapsing 'missing' and 'empty' into one value loses the distinction the
    owner called out: a route that forgot to pass evidence is a BUG, whereas a
    candidate who genuinely has no projects is a legitimate empty case. They must
    not produce the same result.
    """
    MISSING = "missing"     # route did not supply the parameter at all -> BUG
    EMPTY = "empty"         # explicitly supplied, candidate has none
    POPULATED = "populated"  # supplied with entities to check against

# Field names that can carry generated resume content to a client. A blocked
# publication must null ALL of them: measured, withholding only
# final_written_section* still left the rejected claims readable in
# `rewritten_resume`, a backward-compatibility alias the frontend consumes.
DELIVERABLE_FIELDS = (
    "final_written_section_markdown",
    "final_written_section",
    "rewritten_resume",
    "resume_text",
)

# Minimum viable shapes for a publishable resume.
_META_REPLY_STARTS = (
    "i'm ready to", "i am ready to", "please provide", "i need the details",
    "could you provide", "i'd be happy to", "as an ai", "i cannot", "i'm unable",
    "i'm sorry",
)
# Floor is deliberately low: this guard exists to reject stubs and chat replies,
# not to enforce a length policy. Measured false positives at 120 words on a
# legitimate sparse resume (113 words with 2 employers, 4 bullets, education).
# Length policy belongs to the product, not the publication gate.
_MIN_WORDS = 80


@dataclass
class PublishDecision:
    """The one authoritative answer, plus the reasons behind it."""
    publishable: bool
    status: str                       # matches models.ProcessingStatus values
    reasons: List[Reason] = field(default_factory=list)
    coverage: str = "none"            # 'full' | 'partial' | 'none'
    review_message: Optional[str] = None
    entity_evidence: str = EntityEvidence.MISSING.value

    def blocked(self) -> bool:
        return not self.publishable

    @property
    def reason_codes(self) -> List[str]:
        """Presentation form: stable strings for the wire."""
        return [r.value for r in self.reasons]


def _looks_conversational(text: str) -> bool:
    head = (text or "").strip().lower()[:220]
    return any(head.startswith(m) or m in head for m in _META_REPLY_STARTS)


def _has_section(text: str, patterns) -> bool:
    return any(p.search(text) for p in patterns)


_EXPERIENCE_SECTION = [
    re.compile(r'^#{0,6}\s*(?:professional|work|employment|relevant|career)?\s*'
               r'experience\b', re.IGNORECASE | re.MULTILINE),
]
_EDUCATION_SECTION = [
    re.compile(r'^#{0,6}\s*(?:education|academic)\b', re.IGNORECASE | re.MULTILINE),
]


def decide(
    *,
    resume_markdown: Optional[str],
    metrics_support: Optional[Dict[str, Any]],
    fallback_used: bool = False,
    finish_reason: Optional[str] = None,
    required_facts_missing: Optional[List[str]] = None,
    approved_entities: Any = None,
) -> PublishDecision:
    """Decide whether generated resume content may be published.

    Order matters: structural checks run BEFORE the truth verdict, because a
    truth check on vacuous content trivially passes (measured: a 12-word
    header-only document and three "I'm ready to craft..." replies all scored
    PASS until schema gating was added).

    ``approved_entities`` is three-state and MANDATORY at this boundary:

        None            -> MISSING. The caller forgot to supply evidence.
                           This is a wiring bug, so publication is BLOCKED.
        [] (empty list) -> EMPTY. The candidate genuinely has no projects /
                           employers to check. Legitimate; the entity check is
                           a no-op, not a block.
        [...]           -> POPULATED. Entities are checked against this list.

    Collapsing the first two into one value is how a route could silently skip
    entity verification and still publish.
    """
    reasons: List[Reason] = []
    section_reasons: Dict[str, str] = {}

    # ── 0. entity evidence presence (mandatory) ─────────────────────────
    if approved_entities is None:
        evidence_state = EntityEvidence.MISSING
        reasons.append(Reason.ENTITY_EVIDENCE_MISSING)
    elif isinstance(approved_entities, (list, tuple, set)):
        evidence_state = (EntityEvidence.POPULATED if len(approved_entities)
                          else EntityEvidence.EMPTY)
    else:
        evidence_state = EntityEvidence.MISSING
        reasons.append(Reason.ENTITY_EVIDENCE_MISSING)

    # ── 1. generic fallback / truncation ────────────────────────────────
    if fallback_used:
        reasons.append(Reason.FALLBACK_OUTPUT)
    if finish_reason == "length":
        reasons.append(Reason.TRUNCATED_OUTPUT)

    # ── 2. content shape ────────────────────────────────────────────────
    text = (resume_markdown or "").strip()
    if re.match(r'^\s*(?:\{|\[|```json\b)', text, re.IGNORECASE):
        # A structured model envelope is not the approved Markdown document.
        # In particular, literal "\\n## Education" inside JSON is not a section.
        reasons.append(Reason.CONVERSATIONAL_OUTPUT)
    if not text:
        reasons.append(Reason.EMPTY_OUTPUT)
    else:
        if _looks_conversational(text):
            reasons.append(Reason.CONVERSATIONAL_OUTPUT)
        wc = len(text.split())
        if wc < _MIN_WORDS:
            reasons.append(Reason.BELOW_MIN_WORDS)
            section_reasons["words"] = f"{wc}<{_MIN_WORDS}"
        if not _has_section(text, _EXPERIENCE_SECTION):
            reasons.append(Reason.MISSING_SECTION)
            section_reasons["missing"] = "experience"
        if not _has_section(text, _EDUCATION_SECTION):
            reasons.append(Reason.MISSING_SECTION)
            section_reasons.setdefault("missing_2", "education")
        if re.search(r'\[[A-Za-z ]{3,}\]|\{[A-Za-z ]{3,}\}', text):
            reasons.append(Reason.PLACEHOLDER_LEAK)

    # ── 3. verification must have positively succeeded (TYPED) ──────────
    support = metrics_support or {}
    raw_verdict = support.get("verdict")
    checked = support.get("checked")
    coverage = "none"

    if not support:
        reasons.append(Reason.VERIFICATION_MISSING)
    elif raw_verdict is None or raw_verdict == "":
        reasons.append(Reason.VERIFICATION_EMPTY)
    else:
        try:
            verdict = Verdict(str(raw_verdict))
        except ValueError:
            verdict = None

        if verdict is Verdict.FAIL:
            reasons.append(Reason.UNSUPPORTED_CLAIMS)
            coverage = "partial" if checked else "none"
        elif verdict is Verdict.PASS_TIER_A:
            if checked is not True:
                # PASS with checked=false = the checker never ran. Observed in
                # production: this exact combination is the fail-open.
                reasons.append(Reason.VERIFICATION_NOT_RUN)
            else:
                coverage = "partial"   # Tier A only
        else:
            # Anything that is not the exact typed value is not a pass.
            reasons.append(Reason.VERIFICATION_INCONCLUSIVE)

    # ── 4. required facts ───────────────────────────────────────────────
    if required_facts_missing:
        reasons.append(Reason.REQUIRED_FACTS_MISSING)

    # de-duplicate while preserving order
    seen = set()
    deduped: List[Reason] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            deduped.append(r)

    publishable = not deduped
    decision = PublishDecision(
        publishable=publishable,
        status="completed" if publishable else "needs_review",
        reasons=deduped,
        coverage=coverage,
        review_message=None if publishable else (
            "We couldn't verify some generated claims. Your original "
            "information is preserved; this draft needs review."
        ),
        entity_evidence=evidence_state.value,
    )
    decision.section_detail = section_reasons  # type: ignore[attr-defined]
    return decision


def apply_to_payload(payload: Dict[str, Any],
                     decision: PublishDecision) -> Dict[str, Any]:
    """Apply a decision to a response payload, in place, and return it.

    A blocked payload retains unrelated candidate facts, review reasons and run
    metrics, and withholds deliverables. Experience aliases in pipeline output
    are also withheld when their source versus generated provenance is ambiguous.
    The generated text is never edited — only withheld.
    """
    out = dict(payload or {})
    out["metrics_support"] = payload.get("metrics_support") if payload else None
    out["requires_review"] = decision.blocked()
    # Presentation strings are derived FROM the typed reasons, never the reverse.
    codes = decision.reason_codes
    out["review_reason"] = codes[0] if codes else None
    out["review_reasons"] = codes
    out["review_message"] = decision.review_message
    out["verification_coverage"] = decision.coverage
    out["entity_evidence"] = decision.entity_evidence
    out["processing_status"] = decision.status

    if decision.blocked():
        for f in DELIVERABLE_FIELDS:
            if f in out:
                out[f] = None
        out["final_written_section_provenance"] = []
        # The 3.0 integration constructs both aliases from rewritten Drafter
        # experiences when available, otherwise from parsed source experiences.
        # Their provenance is not recorded, so a blocked response cannot safely
        # distinguish them. Preserve explicitly source-named fields and unrelated
        # candidate facts; source-only payloads are not treated as 3.0 output.
        if (out.get("pipeline_version") == "3.0"
                or ("experience" in out and "experiences" in out)
                or ("pipeline_status" in out and any(
                    key in out for key in DELIVERABLE_FIELDS))):
            for key in ("experience", "experiences"):
                if key in out:
                    out[key] = None
        if "rewritten_experiences" in out:
            out["rewritten_experiences"] = None
        stages = out.get("stages")
        if isinstance(stages, dict) and isinstance(stages.get("drafter"), dict):
            stages = dict(stages)
            drafter = dict(stages["drafter"])
            if "rewritten_experiences" in drafter:
                drafter["rewritten_experiences"] = None
            stages["drafter"] = drafter
            out["stages"] = stages
        # Diagnostics must not re-publish what was withheld. `metrics_support`
        # carries `unsupported_metrics`, whose `surface` field is literally the
        # rejected claim ("87%"). Leaving it intact means a blocked response
        # still ships the invented figure in a side channel — the client was
        # measured reconstructing text from exactly such fields.
        out["metrics_support"] = _redact_metric_surfaces(out.get("metrics_support"))
        # Belt-and-braces against reconstruction: a blocked payload must not
        # carry the generated text inside ANY nested object either, because a
        # client extractor may walk nested fields (measured: the frontend
        # extractor parses a JSON blob out of whatever string it finds).
        for key in list(out.keys()):
            if key in DELIVERABLE_FIELDS:
                continue
            val = out[key]
            if isinstance(val, dict):
                val = dict(val)
                for sub in DELIVERABLE_FIELDS:
                    if sub in val:
                        val[sub] = None
                out[key] = val
        if isinstance(out.get("final_output"), dict):
            out["final_output"]["metrics_support"] = (
                _redact_metric_surfaces(out["final_output"].get("metrics_support")))
    return out


def _redact_metric_surfaces(support: Any) -> Any:
    """Drop the literal claim text from a blocked result's diagnostics.

    Keeps the machine-readable shape (value/unit/reason) so an operator can still
    see WHAT kind of violation occurred, but not the prose that was withheld.
    """
    if not isinstance(support, dict):
        return support
    out = dict(support)
    unsupported = out.get("unsupported_metrics")
    if isinstance(unsupported, list):
        redacted = []
        for item in unsupported:
            if isinstance(item, dict):
                kept = {k: v for k, v in item.items() if k != "surface"}
                kept["redacted"] = True
                redacted.append(kept)
            else:
                redacted.append({"redacted": True})
        out["unsupported_metrics"] = redacted
    return out


# ============================================================================
# SERVER-AUTHORITATIVE APPROVAL RECEIPTS (owner item 2)
# ============================================================================
# Problem: /resume/export read `metrics_support` straight out of the request
# body. A caller could POST {"metrics_support": {"verdict": "PASS (tier A only)",
# "checked": true}} alongside invented content and the gate would approve it.
# Client-supplied approval metadata is not approval.
#
# Fix: /analyze issues a signed receipt binding (content hash, owner, verdict).
# /resume/export verifies the receipt against the content it is about to render.
# A forged verdict string no longer matters, and modified content no longer
# matches the hash it was approved under.
#
# The secret comes from the environment. If it is absent, receipts cannot be
# issued or verified — export then FAILS CLOSED rather than trusting the caller.

RECEIPT_TTL_SECONDS = 60 * 30

# Receipt wire version. Bump when the payload shape or signing scheme changes.
# A verifier that does not recognise the version rejects rather than guesses:
# an unknown version may mean a field it needs is absent, and "verify what you
# understand, reject the rest" is the only fail-closed option.
#
#   v1 — bound `(fp, verdict, owner, iat, ttl, cl)` where `cl` was a FLAT,
#        field-keyed claim set. Superseded: a flat set cannot tell "approved
#        bullet at employer A" from "approved bullet at employer B".
#   v2 — `sb` asserts the legacy Markdown-segment co-membership check.
#   v3 — `render_fp` commits a whitespace-normalized career object but omits
#        contact fields; the Markdown fingerprint also collapses whitespace.
#   v4 — exact UTF-8 Markdown and the complete renderer-parsed object, including
#        identity/contact fields and the parsed strings' exact whitespace.
RECEIPT_VERSION = 4
_SUPPORTED_RECEIPT_VERSIONS = frozenset({4})


def _receipt_secret() -> Optional[bytes]:
    """Signing key. Absent => receipts unavailable => export fails closed.

    Dedicated secret only (owner item 4). This previously fell back to
    `IMAGINATOR_AUTH_TOKEN` — the shared bearer token used for request auth.
    Reusing it means anyone who learns the request token can also MINT approval
    receipts, so auth leakage silently becomes approval forgery. An approval key
    must be distinct from an access key. If it is unset, no receipt can be
    issued, and export fails closed instead of borrowing a weaker credential.
    """
    val = os.environ.get("PUBLICATION_RECEIPT_SECRET")
    if val and len(val) >= 16:
        return val.encode()
    return None


def content_fingerprint(markdown: str) -> str:
    """Hash the exact UTF-8 Markdown, without reflow or Unicode normalization."""
    if not isinstance(markdown, str):
        raise TypeError("Markdown must be a string")
    return hashlib.sha256(markdown.encode("utf-8")).hexdigest()


def canonical_render_content(payload: Any) -> Optional[str]:
    """Serialize the complete parsed renderer input, preserving its exact strings.

    The parser applies its own aliases and normalization, including stripping
    some input whitespace. This binds the resulting object, not the raw payload
    or the eventual PDF/DOCX bytes. Nested ``data`` follows the renderer's
    precedence and no independent headline override is applied.
    """
    if not isinstance(payload, dict):
        return None
    try:
        from resume_builder.build_resume_from_export import parse_resume_payload
    except Exception:  # pragma: no cover - renderer unavailable
        return None

    try:
        resume = parse_resume_payload({"data": payload} if "data" not in payload else payload)
        structure = asdict(resume)
        has_content = any(
            bool(structure[k]) for k in
            ("experience", "projects", "education", "certifications", "summary",
             "skills")
        )
        if not has_content:
            return None
        return json.dumps(structure, sort_keys=True, separators=(",", ":"))
    except Exception:  # noqa: BLE001 - malformed payload => no canonical form
        return None


# ── Legacy render-claim diagnostics (not the v4 receipt binding) ──────────
# The receipt binds the approved MARKDOWN. Export, however, is rendered from the
# STRUCTURED payload, which is a different set of fields. A caller could
# therefore hold a valid receipt for the approved markdown and swap the
# `experience` bullets underneath it: the approved fingerprint still matched,
# and the PDF changed.
#
# Why this is a consistency check rather than a second signature over the
# structure: the server that issues the receipt does not see the caller's
# profile metadata (name, phone, email), while the export request does. A
# fingerprint over the render structure would therefore never match between the
# two sides — it would reject every legitimate export. Instead, every substantive
# structured string the renderer will emit must APPEAR in the approved markdown.
# Swapped or injected content is by definition absent from it.
_IDENTITY_FIELDS = frozenset({
    "name", "phone", "email", "linkedin", "github", "location", "headline",
})


def _squash(text: str) -> str:
    """Lowercase alphanumerics only, for layout-insensitive containment tests."""
    return re.sub(r'[^a-z0-9]+', '', (text or "").lower())


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _iter_render_claims(payload: Dict[str, Any]):
    """Yield (field, text) for each substantive string the export payload supplies.

    Deliberately reads the RAW payload rather than the renderer's parsed output.
    `parse_resume_payload` fills in defaults (a placeholder name, a boilerplate
    summary) and composes titles as "Title -- Company"; those strings were never
    approved by anyone, so treating them as claims made every legitimate export
    fail. What matters is the content the CALLER supplied.

    Identity and contact fields (name, phone, email, links, location, headline)
    are excluded on purpose. They come from the signed-in user's own profile at
    export time, which the analysis service never sees — so they cannot be
    pre-approved, and rejecting them would refuse every honest export. They are
    still scoped: the receipt is bound to the owner, so one candidate's approval
    cannot be spent on another candidate's document.
    """
    def _emit(field: str, value: Any):
        if isinstance(value, str) and value.strip():
            yield field, value
        elif isinstance(value, dict):
            for key in ("text", "name", "title", "value", "description"):
                inner = value.get(key)
                if isinstance(inner, str) and inner.strip():
                    yield field, inner
                    return
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from _emit(field, item)

    summary = payload.get("summary")
    if isinstance(summary, str) and summary.strip():
        yield "summary", summary

    # Narrowed identity exemption (revision 5). A populated headline or
    # top-level location asserts a qualification / market claim, so the APPROVED
    # side must publish them here for export to be able to accept them.
    # Vacuous values are skipped: there is nothing to vouch for.
    for field in ("headline", "location"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            yield field, value

    for value in _as_list(payload.get("skills")):
        yield from _emit("skills", value)

    for entry in _as_list(payload.get("experience") or payload.get("experiences")):
        if isinstance(entry, dict):
            for key in ("bullets", "highlights", "achievements",
                        "responsibilities"):
                for bullet in _as_list(entry.get(key)):
                    yield from _emit("bullet", bullet)
            for key in ("title", "role", "title_line"):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    yield "role", value
            company = entry.get("company")
            if isinstance(company, str) and company.strip():
                yield "company", company
        else:
            yield from _emit("bullet", entry)

    for entry in _as_list(payload.get("projects")):
        if isinstance(entry, dict):
            for key in ("bullets", "highlights", "description"):
                for bullet in _as_list(entry.get(key)):
                    yield from _emit("project", bullet)
            for key in ("title", "name", "role"):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    yield "project_title", value
        else:
            yield from _emit("project", entry)

    for value in _as_list(payload.get("education")):
        yield from _emit("education", value)

    for value in _as_list(payload.get("certifications")):
        yield from _emit("certifications", value)

    publication = payload.get("publication")
    if isinstance(publication, str) and publication.strip():
        yield "publication", publication


# Minimum length for a claim to be asserted on. Below this, fragments such as
# "Python" or "SQL" appear coincidentally in unrelated prose, which would make
# the check a source of false positives rather than a guard.
_MIN_CLAIM_CHARS = 12


def approved_render_claims(payload: Any) -> Dict[str, List[str]]:
    """Normalised strings the renderer would emit for `payload`, keyed by field.

    Computed on the issuing side from the server's own result and embedded in
    the receipt. Keying by field is what lets export distinguish "the server
    approved this name" from "the client supplied a name the server never saw".
    """
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, set] = {}
    for field, text in _iter_render_claims(payload):
        squashed = _squash(text)
        if len(squashed) >= _MIN_CLAIM_CHARS:
            out.setdefault(field, set()).add(squashed)
    return {field: sorted(vals) for field, vals in out.items()}


def render_content_violations(
    payload: Any,
    markdown: Optional[str],
    approved_claims: Optional[Dict[str, List[str]]] = None,
    structural_binding: bool = True,
) -> List[Dict[str, str]]:
    """Structured render content that was NOT part of the approved document.

    Three checks, in order of specificity:

        1. BLOCK BINDING — uniquely scoped claims within a career block must not
            span different approved blocks. This catches some REASSIGNMENT: moving an
         approved bullet to another employer, moving a project into employment,
         or restating a role's dates keeps every string approved while changing
         which block it belongs to.
      2. flat claim containment — every substantive string must appear in the
         approved markdown or claim set. Catches INJECTION of new content.
      3. identity claims — a populated headline / top-level location must appear
         in the approved document (narrowed exemption).

    Career achievements are also checked for preservation. This is not a
    canonical-object equality check: ambiguous/shared claims and rearrangements
    inside one approved block remain known limits.
    """
    if not isinstance(payload, dict):
        return [{"field": "*", "text": "", "reason": "payload_not_structured"}]

    approved_md = _squash(markdown)
    bound: Dict[str, List[str]] = approved_claims or {}
    bound_all = {claim for claims in bound.values() for claim in claims}

    if not approved_md and not bound_all and not structural_binding:
        return [{"field": "*", "text": "", "reason": "approved_content_empty"}]

    violations: List[Dict[str, str]] = []

    # ── 1. relational block binding ─────────────────────────────────────
    # Caught here rather than by hashing a "canonical career object": the
    # analysis route and the export client canonicalise the same resume
    # differently, so content hashes never match across the two. What is shared
    # is the approved MARKDOWN, so blocks are scoped against it.
    if structural_binding:
        segments = approved_segments(markdown)
        if not segments:
            violations.append({
                "field": "*", "text": "",
                "reason": "approved_content_empty",
            })
        else:
            violations.extend(
                _block_binding_violations(payload, markdown, segments))
            violations.extend(_preservation_violations(payload, segments))

    # ── 2. flat claim containment (injection) ───────────────────────────
    for field, text in _iter_render_claims(payload):
        squashed = _squash(text)
        if len(squashed) < _MIN_CLAIM_CHARS:
            continue
        if squashed in approved_md or squashed in bound_all:
            continue
        violations.append({
            "field": field,
            "text": text[:120],
            "reason": "render_content_not_approved",
        })

    # ── 3. narrowed identity claims ─────────────────────────────────────
    violations.extend(_identity_violations(payload, approved_md, bound))

    return violations


# ============================================================================
# STRUCTURED CAREER-CONTENT BINDING (owner item 2, revision 5)
# ============================================================================
# A flat "every substantive string must appear somewhere in the approval" check
# stops INJECTION but not REASSIGNMENT. Measured before this change:
#
#   Approved:  Employer A -> "reduced diagnostic time by 30%"
#              Employer B -> "maintained equipment"
#   Export:    Employer B -> "reduced diagnostic time by 30%"
#
# Every string is approved and the flat check accepts it, but the claim has
# moved to a different employer. The same holds for moving a project into
# employment, restating a role's dates from another role's dates, and dropping
# a required achievement.
#
# The fix is a RELATIONAL binding: the receipt commits to career BLOCKS
# (employer/role/dates/ordered bullets, project facts), and export must present
# the same blocks. Membership is per block, never global.
#
# What is deliberately NOT bound, and why:
#
#   * identity/contact fields — name, phone, email, linkedin, github. These live
#     in the caller's own profile, are never sent to the writer, and cannot be
#     pre-approved server-side. They are scoped by the RECEIPT OWNER instead.
#
#   * `headline` and top-level `location` — NARROWED in this revision. These were
#     previously exempt with the contact fields, but a headline can carry a
#     qualification claim ("Licensed Master Electrician") and a top-level location
#     can be a career claim. They are now EXEMPT ONLY WHEN VACUOUS: an empty or
#     absent value is ignored, a populated one must have appeared in the
#     approved document. Callers that never populate them are unaffected.
#
#   * per-role `location` — a role location is part of employment history, so it
#     is bound as a block field rather than exempted.

# Block fields that identify an employment block. Any of these changing is a
# different block, not a reformat of the same one.
_DATE_KEYS = ("start_date", "end_date", "date_string", "dates", "period")


def _canonical_date(value: Any) -> str:
    """Reduce a date to a comparable token so Mar 2020 == 03/2020 == 2020-03.

    Only the month and year matter for the binding: the renderer receives
    reformatted dates from the client (the frontend maps ISO dates to
    "03/2020"), and a genuine content change still alters the token.
    """
    text = (value or "").lower() if isinstance(value, str) else str(value or "")
    if not text.strip():
        return ""
    if "present" in text or "current" in text:
        return "present"
    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11,
        "dec": 12,
    }
    year = re.search(r'(19|20)\d{2}', text)
    if not year:
        return _squash(text)
    year_int = int(year.group(0))
    month = None
    # ISO 8601 (2020-03, 2020-03-01) — checked first, because the MM/YYYY
    # pattern would otherwise read the "-03" as a day-less fragment and fall
    # back to a year-only token, making "2020-03-01" != "03/2020".
    iso = re.search(r'\b((?:19|20)\d{2})[-/.](\d{1,2})\b', text)
    if iso:
        month = int(iso.group(2))
        if not 1 <= month <= 12:
            month = None
    if month is None:
        numeric = re.search(r'\b(0?[1-9]|1[0-2])\s*[/.-]\s*(?:19|20)\d{2}', text)
        if numeric:
            month = int(numeric.group(1))
    if month is None:
        for name, number in months.items():
            if re.search(rf'\b{name}[a-z]*\b', text):
                month = number
                break
    return f"{year_int:04d}-{month:02d}" if month else f"{year_int:04d}"


def _renderer_sections(payload: Any):
    """Experience/project sections as the RENDERER will emit them.

    Returns ``(experience_sections, project_sections)`` as
    ``(title, date_string, body_lines)`` tuples, or ``(None, None)`` when the
    renderer is unavailable — in which case the caller must not treat the
    absence as "approved".
    """
    if not isinstance(payload, dict):
        return None, None
    try:
        from resume_builder.build_resume_from_export import parse_resume_payload
    except Exception:  # pragma: no cover - renderer unavailable
        return None, None
    try:
        resume = parse_resume_payload(payload)
    except Exception:  # noqa: BLE001 - malformed payload
        return None, None

    def _trip(sections):
        return [
            (s.title or "", s.date_string or "",
             [str(line) for line in (s.body_lines or [])])
            for s in sections or []
        ]

    return _trip(resume.experience), _trip(resume.projects)


# ── relational binding, anchored on the approved document ────────────────
# Hashing "the canonical career object" was the first attempt and it does not
# survive contact with the real pipeline. Measured: the analysis route emits
# `experiences[]` shaped `{title_line, snippet}`, while export sends
# `experience[]` shaped `{title, bullets}` built by the CLIENT's parser. The two
# canonicalise differently, so block hashes never matched and the check bound
# nothing — it passed for the wrong reason.
#
# What IS identical on both sides is the approved MARKDOWN. So the binding is
# expressed as: split the approved document into blocks at each heading, and
# require every structured block the export supplies to be consistent with a
# SINGLE approved block. Concretely, for each pair of substantive claims inside
# one export block, some approved block must contain both.
#
# That single rule catches all four reassignments:
#   * bullet to another employer  -> the pair spans two approved blocks
#   * project into employment     -> ditto
#   * another role's dates        -> ditto (date tokens are claims)
#   * dropped achievement         -> handled by the preservation check
# and it still accepts legitimate reshaping, because it never compares labels,
# key names or ordering — only co-membership.

_HEADING_RE = re.compile(r'^\s{0,3}(?:#{1,6}\s*|\*\*.+\*\*[^*]*)\S')

_DATE_TEXT_RE = re.compile(
    r'\b(?:(?:19|20)\d{2}[-/.]\d{1,2}(?:[-/.]\d{1,2})?'
    r'|\d{1,2}\s*[/.-]\s*(?:19|20)\d{2}'
    r'|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+(?:19|20)\d{2}'
    r'|(?:19|20)\d{2})\b',
    re.IGNORECASE,
)


def _date_tokens(text: str) -> set:
    """Canonical month/year tokens mentioned in a piece of text."""
    tokens = set()
    lowered = (text or "").lower()
    if "present" in lowered or "current" in lowered:
        tokens.add("date:present")
    for match in _DATE_TEXT_RE.finditer(text or ""):
        token = _canonical_date(match.group(0))
        if token:
            tokens.add(f"date:{token}")
    return tokens


def _text_claims(text: str) -> set:
    """Substantive claims in one piece of text: prose lines plus date tokens."""
    claims = _date_tokens(text)
    squashed = _squash(text)
    if len(squashed) >= _MIN_CLAIM_CHARS:
        claims.add(squashed)
    return claims


def approved_segments(markdown: Optional[str]) -> List[Dict[str, Any]]:
    """Split the approved document into blocks at each heading.

    A block boundary is either a markdown heading ("## Professional Experience")
    or a bold leading run ("**Cascade Power Solutions, LLC** - Tacoma, WA"), the
    two forms the Imaginator emits. Each segment's claim set is everything the
    segment asserts: its heading, every body line, and every date token.

    Section headers ("Professional Experience", "Education") are boundaries but
    are not career blocks, so a claim that only appears in a header never scopes
    anything.
    """
    text = markdown or ""
    if not text.strip():
        return []

    segments: List[Dict[str, Any]] = []
    current = {"heading": "", "claims": set(), "bullets": [],
               "is_block": False, "section": None}
    section = None

    def flush():
        if current["heading"] or current["claims"]:
            # Joined wrapped bullets must scope together, not only as fragments.
            for bullet in current["bullets"]:
                current["claims"] |= _text_claims(bullet)
            segments.append(dict(current))

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Remove emphasis markers from the heading text itself so the stored
        # heading is comparable prose, not "Company** - City".
        heading = line.replace("#", "").replace("**", "").strip()
        is_markdown_heading = bool(re.match(r'^\s{0,3}#{1,6}\s', raw))
        is_bold_block = _is_block_heading(line)
        is_section_label = _squash(heading) in _SECTION_LABELS

        if is_markdown_heading or is_bold_block or is_section_label:
            flush()
            if is_section_label:
                label = _squash(heading)
                section = ("experience" if label in {
                    "professionalexperience", "workexperience",
                    "employmentexperience", "experience", "relevantexperience",
                    "careerhistory",
                } else "project" if label == "projects" else "other")
            current = {
                "heading": heading,
                "claims": _text_claims(heading),
                "bullets": [],
                "is_block": not is_section_label,
                "section": section,
            }
        else:
            if not current["heading"]:
                current["heading"] = heading
            claims = _text_claims(line)
            current["claims"] |= claims
            # A bullet is a body line that asserts a substantive claim. These
            # are the "achievements" a preservation promise is about.
            #
            # Continuation lines are JOINED to the bullet they belong to. The
            # Imaginator hard-wraps long bullets across lines ("... panels and\n
            # motor control centers, reducing ..."), and treating each wrapped
            # line as a separate achievement made every wrapped bullet look
            # omitted from the rendered document.
            starts_bullet = bool(re.match(r'^\s*[-•·▪▸*]\s+', raw))
            if starts_bullet:
                current["bullets"].append("")
            if current["bullets"]:
                current["bullets"][-1] += " " + line
            elif claims - {c for c in claims if c.startswith("date:")}:
                current["bullets"].append(line)

    flush()
    return segments


_SECTION_LABELS = frozenset({
    "professionalexperience", "workexperience", "employmentexperience",
    "experience", "education", "academic", "projects", "skills",
    "technicalskills", "certifications", "publications", "summary",
    "professionalsummary", "awards", "achievements", "relevantexperience",
    "careerhistory",
})


def _is_block_heading(line: str) -> bool:
    """A bold leading run, e.g. "**Cascade Power Solutions, LLC** - Tacoma, WA".

    NB: do not strip '*' before testing — `**` IS the marker. Only a leading
    bullet marker ("- ", "• ") may be removed first.
    """
    # Bold emphasis inside a bullet is body text, NEVER an employer boundary.
    if re.match(r'^[-•·▪▸*]\s+', line.strip()):
        return False
    stripped = line.strip()
    return stripped.startswith("**") and stripped.count("**") >= 2


def _block_claim_groups(payload: Any) -> List[Dict[str, Any]]:
    """Per-block claim sets for every structured career block in the export."""
    experience, projects = _renderer_sections(payload)
    groups: List[Dict[str, Any]] = []
    for kind, sections in (("experience", experience), ("project", projects)):
        for title, date_string, body_lines in sections or []:
            claims = set()
            claims |= _text_claims(title or "")
            claims |= _date_tokens(date_string or "")
            for line in body_lines:
                claims |= _text_claims(line)
            if not claims:
                continue
            groups.append({"kind": kind, "label": title, "claims": claims})
    return groups


def _segment_containing(token: str,
                        segments: List[Dict[str, Any]]) -> List[int]:
    """Indices of approved segments that contain `token`."""
    return [i for i, seg in enumerate(segments) if token in seg["claims"]]


def _block_binding_violations(payload: Any, markdown: Optional[str],
                              segments: Optional[List[Dict[str, Any]]]
                              ) -> List[Dict[str, str]]:
    """Career blocks whose internal relationships were never approved together.

    THE RULE: no two substantive claims inside ONE export block may be scoped to
    two DIFFERENT approved blocks. Equivalently, some single approved block must
    account for everything the export block asserts.

    Why this and not a content hash: the analysis route and the export client
    canonicalise the same resume differently (`experiences[]` with `title_line`
    vs `experience[]` with `title`), so identical-content hashes never match and
    the check would bind nothing. Comparing WHICH claims travel TOGETHER is
    label-independent and survives that reshape.

    Only segments flagged `is_block` scope a claim: a section label like
    "Professional Experience" is a boundary, not an employer, and must not
    compete for scope. A claim appearing in more than one approved block scopes
    nothing (its membership is genuinely ambiguous); a claim appearing nowhere is
    left to the flat containment check, which reports it as unapproved content.
    """
    if segments is None:
        return []
    blocks = [i for i, seg in enumerate(segments) if seg.get("is_block")]
    violations: List[Dict[str, str]] = []
    for seg in segments:
        if (seg.get("section") in {"experience", "project"}
                and not seg.get("is_block") and seg.get("bullets")):
            # An unrecognised heading must not collapse all employers into one
            # section-wide scope and silently turn this guard off.
            violations.append({"field": seg["section"], "text": "",
                               "reason": "career_segmentation_ambiguous"})
    for group in _block_claim_groups(payload):
        scopes = set()
        for token in group["claims"]:
            hits = [i for i in blocks if token in segments[i]["claims"]]
            if len(hits) == 1:
                scopes.add(hits[0])
        if len(scopes) > 1:
            violations.append({
                "field": group["kind"],
                "text": (group["label"] or "")[:120],
                "reason": "career_block_reassigned",
            })
    return violations


def _preservation_violations(payload: Any,
                             segments: List[Dict[str, Any]]
                             ) -> List[Dict[str, str]]:
    """Approved achievements missing from the rendered structure.

    Export renders the APPROVED resume, so it promises to preserve what was
    approved. Deleting a bullet leaves every remaining claim correctly scoped —
    reassignment checks pass — yet the document no longer says what was signed
    off. This closes that gap.

    Only bullets are checked, and only bullets that an export block could have
    carried: this compares the set of approved prose claims against the set the
    payload renders. Headings, dates and section labels are excluded because a
    legitimate export may legitimately re-derive them.

    Only named non-career sections are exempt. Missing all blocks or deleting
    an entire employer is NOT evidence that the renderer preserves the Markdown.
    """
    groups = _block_claim_groups(payload)

    # Containment, not set equality. The live analysis route emits
    # `experiences[]` entries whose `snippet` concatenates whole sections, so a
    # rendered block can legitimately carry several approved bullets as one
    # string. Requiring exact membership rejected that legitimate passthrough;
    # requiring containment still catches a DELETED bullet, because a bullet that
    # is gone from the payload cannot appear in the rendered prose.
    rendered_blob = "".join(
        c for group in groups for c in group["claims"] if not c.startswith("date:")
    )

    approved_bullets = []
    for seg in segments:
        # The exemption is based on SECTION TYPE, not surviving overlap. A
        # deleted employer/project (including every bullet) is still required.
        # Education/certifications render as flat entries, not career bullets.
        if not seg.get("is_block") or seg.get("section") not in {
                "experience", "project"}:
            continue
        for bullet in seg.get("bullets") or []:
            squashed = _squash(bullet)
            if len(squashed) >= _MIN_CLAIM_CHARS:
                approved_bullets.append(squashed)

    missing = [b for b in approved_bullets if b not in rendered_blob]
    if not missing:
        return []
    return [{
        "field": "experience",
        "text": missing[0][:120],
        "reason": "approved_achievement_omitted",
    }]



# ============================================================================
# NARROWED IDENTITY EXEMPTION (owner item 2, revision 5)
# ============================================================================
# `headline` and top-level `location` may be restored from the caller's profile
# at export time, but only while they assert nothing. A populated headline is a
# qualification claim; a populated top-level location is a job-market claim.
# Both must therefore have appeared in the approved document.
#
# name/phone/email/linkedin/github stay exempt unconditionally: they are contact
# detail, the writer never sees them, and they are scoped by owner binding.
_CONTACT_ONLY_FIELDS = ("name", "phone", "email", "linkedin", "github")
_CLAIMING_IDENTITY_FIELDS = ("headline", "location")


def _identity_violations(payload: Any, approved_md: str,
                         bound: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """Populated headline/location that the approved document never contained."""
    if not isinstance(payload, dict):
        return []
    bound_all = {claim for claims in (bound or {}).values() for claim in claims}
    violations: List[Dict[str, str]] = []
    for field in _CLAIMING_IDENTITY_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            continue          # vacant -> nothing asserted -> exempt
        squashed = _squash(value)
        if len(squashed) < _MIN_CLAIM_CHARS:
            continue          # too short to be a falsifiable claim
        if squashed in approved_md or squashed in bound_all:
            continue
        violations.append({
            "field": field,
            "text": value[:120],
            "reason": "identity_claim_not_approved",
        })
    return violations


def _normalise_owner(owner: Optional[str]) -> str:
    return (owner or "").strip().lower()


def issue_receipt(*, markdown: Optional[str], verdict: str,
                  owner: Optional[str] = None,
                  ttl: int = RECEIPT_TTL_SECONDS,
                  approved_claims: Optional[Dict[str, List[str]]] = None,
                  approved_payload: Optional[Dict[str, Any]] = None,
                  ) -> Optional[str]:
    """Issue a signed approval receipt, or None when signing is unavailable.

    Version 4 binds the exact Markdown and the complete renderer-parsed object.
    ``approved_claims`` and ``sb`` remain in the signed envelope for existing
    callers, but are not alternatives to the render-object fingerprint.
    """
    secret = _receipt_secret()
    if secret is None:
        return None
    try:
        fingerprint = content_fingerprint(markdown)
    except (TypeError, UnicodeEncodeError):
        return None
    payload = {
        "v": RECEIPT_VERSION,
        "fp": fingerprint,
        "verdict": verdict,
        "owner": _normalise_owner(owner),
        "iat": int(time.time()),
        "ttl": int(ttl),
    }
    if approved_claims:
        payload["cl"] = {
            field: sorted(set(claims))
            for field, claims in approved_claims.items() if claims
        }
    payload["sb"] = 1
    if approved_payload is None:
        from approved_resume import export_payload
        try:
            approved_payload = export_payload(markdown)
        except ValueError:
            return None
    career = canonical_render_content(approved_payload)
    if not career:
        return None
    payload["render_fp"] = hashlib.sha256(career.encode("utf-8")).hexdigest()
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    sig = hmac.new(secret, body.encode("utf-8"), hashlib.sha256).hexdigest()
    token = base64.urlsafe_b64encode(body.encode("utf-8")).decode("ascii")
    return f"{token}.{sig}"


def verify_receipt(receipt: Optional[str], *, markdown: Optional[str],
                   owner: Optional[str] = None,
                   render_payload: Any = None) -> tuple:
    """Verify a receipt against the content being exported.

    Returns (ok: bool, reason: str).

    Failure reasons are specific so a caller and an operator can distinguish
    tampering from expiry from a stale-content binding.

    Version 4 requires the structured payload and compares the complete parsed
    object, not a claim set or an assertion about the generated PDF/DOCX bytes.
    """
    if not receipt or not isinstance(receipt, str) or "." not in receipt:
        return False, "receipt_missing"
    secret = _receipt_secret()
    if secret is None:
        return False, "receipt_signing_unavailable"

    token, _, sig = receipt.rpartition(".")
    try:
        body = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
    except Exception:  # noqa: BLE001
        return False, "receipt_malformed"

    expected = hmac.new(secret, body.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return False, "receipt_signature_invalid"

    try:
        payload = json.loads(body)
    except Exception:  # noqa: BLE001
        return False, "receipt_malformed"
    if not isinstance(payload, dict):
        return False, "receipt_malformed"

    # ── version gate (item 4) ───────────────────────────────────────────
    # Absent or unknown version => reject. Guessing at an unfamiliar layout is
    # how a verifier ends up validating a field that no longer means what it
    # did when the receipt was minted.
    version = payload.get("v")
    if version is None:
        return False, "receipt_version_missing"
    if type(version) is not int:
        return False, "receipt_version_unknown"
    if version == 3:
        return False, "receipt_version_3_reanalysis_required"
    if version not in _SUPPORTED_RECEIPT_VERSIONS:
        return False, "receipt_version_unknown"

    if payload.get("verdict") != Verdict.PASS_TIER_A.value:
        return False, "receipt_not_approved"

    issued = int(payload.get("iat") or 0)
    ttl = int(payload.get("ttl") or 0)
    if ttl <= 0 or (time.time() - issued) > ttl:
        return False, "receipt_expired"

    try:
        fingerprint = content_fingerprint(markdown)
    except (TypeError, UnicodeEncodeError):
        return False, "receipt_content_mismatch"
    if payload.get("fp") != fingerprint:
        return False, "receipt_content_mismatch"

    # ── owner binding ───────────────────────────────────────────────────
    # Checked BEFORE the render-content check: presenting someone else's
    # approval and presenting tampered content can both be true, and the owner
    # mismatch is the more specific and more important diagnosis. Reporting
    # "content mismatch" for what is actually an attempt to spend another
    # candidate's approval would send an operator looking in the wrong place.
    #
    # An EMPTY bound owner is rejected outright. This was previously skipped
    # (`if bound_owner and ...`), which meant a receipt minted by a route that
    # forgot to forward the owner header was spendable by ANY authenticated
    # caller — the approval was real, just unattributed. Measured cause: the
    # multi-file analyze proxy never forwarded X-Cogito-Owner-Id, so every
    # multi-file approval was minted unowned.
    bound_owner = _normalise_owner(payload.get("owner"))
    presented_owner = _normalise_owner(owner)
    if not bound_owner:
        return False, "receipt_owner_unbound"
    if presented_owner != bound_owner:
        return False, "receipt_owner_mismatch"

    # Bind the exact normalized object passed to the renderer, not raw aliases
    # or a whitespace-collapsed subset. A missing object must not downgrade v4.
    if render_payload is None:
        return False, "receipt_render_payload_missing"
    if payload.get("sb") != 1 or not payload.get("render_fp"):
        return False, "receipt_structure_unbound"
    career = canonical_render_content(render_payload)
    if not career or payload["render_fp"] != hashlib.sha256(career.encode("utf-8")).hexdigest():
        return False, "receipt_render_content_mismatch"

    return True, "ok"
