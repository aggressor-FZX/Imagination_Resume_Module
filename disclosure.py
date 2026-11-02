"""Correspondence-preserving redaction for the correction disclosure gate.

The correction engine validates citations by exact code-point offsets into the
source documents (``quote == sources[sid][start:end]``). Any redaction that
changes the length of the source therefore invalidates every offset, so the
usual "replace with ``[email removed]``" approach cannot be used here.

This module redacts the same PII categories that the normal generation path
already removes (``imaginator_new_integration.sanitize_resume_pii``) but pads
each replacement to the exact length of the matched span. Offsets are preserved
byte-for-byte while the sensitive characters are removed.

The redaction is *not* a proof that arbitrary text is safe: it covers the same
unambiguous categories as the existing pipeline, and it deliberately leaves
names, employers, schools, city/state and portfolio URLs intact because those
are resume content the reviewer needs. The attestation it produces is verified
per request by ``correction_provider`` before any provider call.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

POLICY_ID = "length_preserving_pii_redaction_v1"

# Same unambiguous categories as the normal pipeline. Deliberately no name
# detection: standalone names are kept by the existing pipeline too, and a
# false positive here would corrupt a legitimate record label.
_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.IGNORECASE)),
    ("phone", re.compile(r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b")),
    ("ssn", re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b")),
    ("address", re.compile(
        r"\b\d+\s+[A-Za-z]+(?:\s+[A-Za-z]+)*\s+"
        r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Circle|Cir|Place|Pl|Loop)\b",
        re.IGNORECASE,
    )),
    ("linkedin", re.compile(r"linkedin\.com/in/[^\s,;|]+", re.IGNORECASE)),
    # Full contact header lines: name + email/phone/address. Replaced in place so
    # the line still exists (the draft often keeps a matching placeholder line).
    ("contact_line", re.compile(
        r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\|\s*"
        r"(?:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        r"|(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}"
        r"|\d+\s+[A-Za-z]+(?:\s+[A-Za-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd))",
        re.MULTILINE | re.IGNORECASE,
    )),
)

# Padding character: not whitespace (never truncates a line), not alphanumeric
# (cannot synthesise a new email/phone/address), and single-codepoint so the
# character length equals the text length for the engine's offsets.
_PAD = "\u2588"


@dataclass(frozen=True)
class Attestation:
    """Evidence that a source set was redacted with the approved policy."""

    policy: str
    sources: tuple[tuple[str, int], ...]  # (source_id, character length)
    redactions: int

    def as_payload(self) -> dict[str, Any]:
        return {"policy": self.policy,
                "sources": [{"source_id": sid, "length": length}
                            for sid, length in self.sources],
                "redactions": self.redactions}


def _placeholder(label: str, width: int) -> str:
    """Return a label token padded to exactly ``width`` characters."""
    token = f"[{label} removed]"
    if len(token) > width:
        token = f"[{label[:2]}]"
    if len(token) > width:
        token = "[" * min(width, 1)
    return token + _PAD * (width - len(token))


def _redact_value(value: str, label: str) -> str:
    if not value:
        return value
    return _placeholder(label, len(value))


def redact_text(text: str) -> tuple[str, int]:
    """Return ``(redacted, count)`` with identical character length."""
    if not isinstance(text, str) or not text:
        return text, 0
    output = text
    count = 0
    # Longest/most specific first: a contact header line contains an email or
    # phone, so replacing the whole line first avoids a double substitution that
    # would still be length-preserving but less readable.
    for label, pattern in _PII_PATTERNS:
        def substitute(match: re.Match[str]) -> str:
            nonlocal count
            span = match.group(0)
            replacement = _redact_value(span, label)
            if replacement != span:
                count += 1
            return replacement
        output = pattern.sub(substitute, output)
    return output, count


def contains_pii(text: str) -> bool:
    """True when any covered, unambiguous PII pattern still matches."""
    if not isinstance(text, str) or not text:
        return False
    return any(pattern.search(text) for _, pattern in _PII_PATTERNS)


def redact_sources(
    documents: Iterable[tuple[str, str]],
) -> tuple[tuple[tuple[str, str], ...], Attestation | None]:
    """Redact a source set, returning redacted documents and an attestation.

    Returns ``(documents, None)`` when correspondence cannot be proven: the
    caller must then withhold rather than send anything. Length equality is
    checked per document *and* the result is scanned for residual PII.
    """
    redacted: list[tuple[str, str]] = []
    lengths: list[tuple[str, int]] = []
    total = 0
    for source_id, text in documents:
        if not isinstance(source_id, str) or not isinstance(text, str):
            return (), None
        cleaned, count = redact_text(text)
        if len(cleaned) != len(text) or contains_pii(cleaned):
            return (), None
        redacted.append((source_id, cleaned))
        lengths.append((source_id, len(text)))
        total += count
    if not redacted:
        return (), None
    return tuple(redacted), Attestation(POLICY_ID, tuple(lengths), total)


def verify_disclosure(
    sources: Iterable[dict[str, Any]],
    *,
    approved_policy: str | None,
) -> bool:
    """Verify the payload about to be sent satisfies the approved policy.

    This is the per-request check behind the production gate. It passes only
    when (a) the approved policy is enabled and (b) every source document
    contains none of the covered, unambiguous PII categories. Because redaction
    is length-preserving, no offset comparison is needed: the offset contract is
    guaranteed by construction, and this check independently confirms the
    sensitive characters are actually gone.
    """
    if approved_policy != POLICY_ID:
        return False
    items = list(sources) if isinstance(sources, (list, tuple)) else []
    if not items:
        return False
    for item in items:
        if type(item) is not dict or set(item) != {"source_id", "text"}:
            return False
        if not isinstance(item["source_id"], str) or not isinstance(item["text"], str):
            return False
        if not item["text"].strip() or contains_pii(item["text"]):
            return False
    return True
