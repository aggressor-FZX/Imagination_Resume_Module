"""Skill-inference permission policy: the two supported modes and their contract.

This module is the single normalisation point for `skill_inference_mode`. It is
deliberately separate from the prose-style `creativity_mode` control: creativity
changes temperature and tone, while this value is a **permission policy** that
decides whether inferred transferable skills may appear at all.

Policy authority: ``docs/Plans/CogitoMetric_Official_Skill_Bridging_Plan_v1.md``
(v1.1 on ``resumeDraft_fix``).

Two modes only:

* ``imaginative`` (default) — the Researcher may propose candidate-anchored
  transferable-skill bridges and the writer/editor may use them after review.
* ``conservative`` — no skill bridging. Supplied facts and faithful rewriting
  only. The bridging-only Researcher call is skipped entirely.

Unknown values are a validation error, never a silent fallback: choosing a mode
on the caller's behalf is how a permission policy becomes an accident.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

# Policy version recorded with the analysis and bound into the approval state.
# Bump only when the permission rules change; receipt wire version is separate.
POLICY_VERSION = "imaginative-v1"
CONSERVATIVE_POLICY_VERSION = "conservative-v1"


class SkillInferenceMode(str, Enum):
    """The two supported skill-inference modes."""

    IMAGINATIVE = "imaginative"
    CONSERVATIVE = "conservative"


DEFAULT_SKILL_INFERENCE_MODE = SkillInferenceMode.IMAGINATIVE

# Request fields a client may use for this setting. Aliases exist only for
# transport compatibility; the value itself is never inferred from a different
# concept (for example `creativity_mode`).
_MODE_ALIASES = ("skill_inference_mode", "skillInferenceMode")


class UnknownSkillInferenceMode(ValueError):
    """Raised when a caller supplies a value outside the two permitted modes."""

    def __init__(self, value: Any):
        super().__init__(f"unknown skill_inference_mode: {value!r}")
        # Deliberately does not echo arbitrary caller text back to a response.
        self.value = value


def normalize_mode(value: Any) -> SkillInferenceMode:
    """Return the effective mode, rejecting unknown values.

    ``None``/absent means the default (imaginative) for this new policy. An
    empty string is treated as absent because JSON payloads and multipart forms
    both commonly send "" for an unset field. Anything else must match one of
    the two values exactly (case-insensitively, surrounding whitespace trimmed).
    """
    if value is None:
        return DEFAULT_SKILL_INFERENCE_MODE
    if isinstance(value, SkillInferenceMode):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if not cleaned:
            return DEFAULT_SKILL_INFERENCE_MODE
        try:
            return SkillInferenceMode(cleaned)
        except ValueError:
            raise UnknownSkillInferenceMode(value) from None
    raise UnknownSkillInferenceMode(value)


def policy_version_for(mode: SkillInferenceMode) -> str:
    """The policy version recorded alongside an effective mode."""
    return (
        POLICY_VERSION
        if mode is SkillInferenceMode.IMAGINATIVE
        else CONSERVATIVE_POLICY_VERSION
    )


def is_imaginative(mode: Any) -> bool:
    """True when bridging inference is permitted for this effective mode."""
    return normalize_mode(mode) is SkillInferenceMode.IMAGINATIVE


def mode_contract(mode: Any) -> Dict[str, str]:
    """Request-local contract handed to stages, review and export.

    Only server-normalised values appear here. A caller cannot say it is
    "already approved" or nominate a policy version.
    """
    effective = normalize_mode(mode)
    return {
        "skill_inference_mode": effective.value,
        "skill_inference_policy_version": policy_version_for(effective),
    }


def from_payload(payload: Optional[Dict[str, Any]]) -> SkillInferenceMode:
    """Read the mode from a request body, preferring the canonical field name.

    Mirrors the model's aliases so non-Pydantic callers (tests, internal tools,
    legacy routes) resolve the same way. Unknown values still raise.
    """
    if isinstance(payload, SkillInferenceMode):
        return payload
    if not isinstance(payload, dict):
        return DEFAULT_SKILL_INFERENCE_MODE
    for key in _MODE_ALIASES:
        if key in payload and payload.get(key) is not None:
            return normalize_mode(payload.get(key))
    return DEFAULT_SKILL_INFERENCE_MODE
