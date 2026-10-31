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
from dataclasses import dataclass, field
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

    A blocked payload keeps every non-generated field (the candidate's own facts,
    the review reasons, run metrics) and nulls every deliverable representation.
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
                for sub in DELIVERABLE_FIELDS:
                    if sub in val:
                        val[sub] = None
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
RECEIPT_VERSION = 1
_SUPPORTED_RECEIPT_VERSIONS = frozenset({RECEIPT_VERSION})


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


def content_fingerprint(markdown: Optional[str]) -> str:
    """Stable hash of exactly what would be rendered.

    Whitespace-normalised so a cosmetic reflow does not invalidate an otherwise
    identical approval, but any substantive edit does.
    """
    text = (markdown or "").strip()
    normalised = re.sub(r'\s+', ' ', text)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def canonical_render_content(payload: Any) -> Optional[str]:
    """Deterministic text view of what the resume renderer will put in the document.

    Built from the renderer's OWN parser (`parse_resume_payload`) so this cannot
    drift from what is actually rendered.
    """
    if not isinstance(payload, dict):
        return None
    try:
        from resume_builder.build_resume_from_export import parse_resume_payload
    except Exception:  # pragma: no cover - renderer unavailable
        return None

    try:
        resume = parse_resume_payload(payload)
    except Exception:  # noqa: BLE001 - malformed payload => no canonical form
        return None

    def _norm(value: Any) -> Any:
        if isinstance(value, str):
            return re.sub(r'\s+', ' ', value).strip()
        if isinstance(value, (list, tuple)):
            return [_norm(v) for v in value]
        return value

    structure = {
        "name": _norm(resume.name),
        "location": _norm(resume.location),
        "phone": _norm(resume.phone),
        "email": _norm(resume.email),
        "linkedin": _norm(resume.linkedin),
        "github": _norm(resume.github),
        "headline": _norm(resume.headline),
        "summary": _norm(resume.summary),
        "skills": _norm([[g, items] for g, items in (resume.skills or [])]),
        "experience": _norm([
            [s.title, s.body_lines, s.start_date, s.end_date, s.date_string]
            for s in (resume.experience or [])
        ]),
        "projects": _norm([
            [s.title, s.body_lines, s.start_date, s.end_date, s.date_string]
            for s in (resume.projects or [])
        ]),
        "education": _norm(list(resume.education or [])),
        "certifications": _norm(list(resume.certifications or [])),
        "publication": _norm(resume.publication),
    }

    has_content = any(
        bool(structure[k]) for k in
        ("experience", "projects", "education", "certifications", "summary",
         "skills")
    )
    if not has_content:
        return None
    return json.dumps(structure, sort_keys=True, separators=(",", ":"))


# ── Render-content binding (owner item 2) ────────────────────────────────
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
) -> List[Dict[str, str]]:
    """Structured render content that was NOT part of the approved document.

    Two acceptance routes, because the client legitimately reshapes data (it
    maps `role` to `title`, reformats dates, and adds its own profile metadata,
    which would fail a strict equality test against the server's own structure):

      1. the claim appears in the approved markdown, or
      2. the claim appears in the approved claim set for that field.

    New or substituted content is in neither route, so it still fails.
    """
    if not isinstance(payload, dict):
        return [{"field": "*", "text": "", "reason": "payload_not_structured"}]

    approved_md = _squash(markdown)
    bound: Dict[str, List[str]] = approved_claims or {}
    bound_all = {claim for claims in bound.values() for claim in claims}
    if not approved_md and not bound_all:
        return [{"field": "*", "text": "", "reason": "approved_content_empty"}]

    violations: List[Dict[str, str]] = []
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
    return violations


def _normalise_owner(owner: Optional[str]) -> str:
    return (owner or "").strip().lower()


def issue_receipt(*, markdown: Optional[str], verdict: str,
                  owner: Optional[str] = None,
                  ttl: int = RECEIPT_TTL_SECONDS,
                  approved_claims: Optional[Dict[str, List[str]]] = None
                  ) -> Optional[str]:
    """Issue a signed approval receipt, or None when signing is unavailable.

    ``approved_claims`` is the normalised, field-keyed set of strings the
    server's own renderer would emit (see `approved_render_claims`). Embedding it
    lets export verify that the structured fields it is about to render were part
    of the approved document, instead of trusting a markdown-only binding.
    """
    secret = _receipt_secret()
    if secret is None:
        return None
    payload = {
        "v": RECEIPT_VERSION,
        "fp": content_fingerprint(markdown),
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

    When ``render_payload`` is supplied, the structured fields the renderer
    will emit must also appear in the approved markdown. That is what stops a
    caller from keeping an approved markdown and swapping the experience
    bullets underneath it (owner item 2).
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
    try:
        version_int = int(version)
    except (TypeError, ValueError):
        return False, "receipt_version_unknown"
    if version_int not in _SUPPORTED_RECEIPT_VERSIONS:
        return False, "receipt_version_unknown"

    if payload.get("verdict") != Verdict.PASS_TIER_A.value:
        return False, "receipt_not_approved"

    issued = int(payload.get("iat") or 0)
    ttl = int(payload.get("ttl") or 0)
    if ttl <= 0 or (time.time() - issued) > ttl:
        return False, "receipt_expired"

    if payload.get("fp") != content_fingerprint(markdown):
        return False, "receipt_content_mismatch"

    # ── owner binding ───────────────────────────────────────────────────
    # Checked BEFORE the render-content check: presenting someone else's
    # approval and presenting tampered content can both be true, and the owner
    # mismatch is the more specific and more important diagnosis. Reporting
    # "content mismatch" for what is actually an attempt to spend another
    # candidate's approval would send an operator looking in the wrong place.
    bound_owner = _normalise_owner(payload.get("owner"))
    if bound_owner and _normalise_owner(owner) != bound_owner:
        return False, "receipt_owner_mismatch"

    # ── render-content binding (item 2) ─────────────────────────────────
    # The receipt binds the approved MARKDOWN. Export, however, is rendered from
    # the STRUCTURED payload. Without this check a caller could hold a valid
    # receipt for the approved markdown and swap `experience` bullets, projects,
    # education or certifications underneath it: the approved fingerprint still
    # matches while the produced PDF says something different.
    if render_payload is not None:
        bound_claims = payload.get("cl")
        if not isinstance(bound_claims, dict):
            bound_claims = None
        if render_content_violations(render_payload, markdown, bound_claims):
            return False, "receipt_render_content_mismatch"

    return True, "ok"
