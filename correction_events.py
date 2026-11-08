"""Privacy-safe phase events for source correction.

Events describe timing, budgets, outcomes and token counts only. They are
sanitized at the logging boundary: unknown keys are dropped, strings must match
a fixed vocabulary or a short identifier pattern, and anything else is dropped.
Document text, prompts, model output and exception messages therefore cannot
reach a log line even if a caller passes them.
"""
import json
import math
import re

_INTEGER_FIELDS = {
    "attempt", "budget_ms", "budget_left_ms", "call_timeout_ms", "deadline_ms", "elapsed_ms",
    "max_tokens", "pipeline_elapsed_ms", "prompt_tokens", "completion_tokens",
    "repairs", "request_index", "reserve_ms", "reviews", "total_tokens",
    "upstream_timeout_ms", "http_status",
    # Source-inventory counts at a completeness refusal. Counts, never the
    # claims: they separate "one ambiguous span" from "almost nothing was
    # usable", which are different diagnoses for the same reason code. Text is
    # dropped by the allow-list below regardless.
    #
    # Names state exactly what is counted. An earlier `uncertain_required_claims`
    # counted spans classified `uncertain`, NOT required spans that turned
    # uncertain — a field that lied about its own meaning is worse than no field.
    "uncertain_source_claims",
    "required_source_claims",
    "source_claim_total",
}
_FLOAT_FIELDS = {"cost_usd"}
_IDENTIFIER = re.compile(r"[a-z0-9_]{1,64}\Z")
_CORRELATION = re.compile(r"[A-Za-z0-9-]{1,64}\Z")
_MODEL = re.compile(r"[a-z0-9_./-]{1,80}\Z")
_PROVIDER = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,39}\Z")

EVENTS = frozenset({"correction.budget", "correction.phase_start", "correction.phase_end",
                    "correction.end", "provider.call",
                    "correction.completeness"})
PHASES = frozenset({"source_inventory", "initial_review", "repair", "post_patch_review"})
# Transport/callback outcomes only; validation results are reported as reasons.
# ``provider_rate_limited`` and ``provider_unavailable`` were missing from this set
# once they existed, so the sanitizer silently dropped the very events that explain
# a retry sequence. Keep this list in step with what the adapter can emit.
PHASE_OUTCOMES = frozenset({"ok", "deadline", "provider_call_timeout", "callback_timeout",
                            "callback_error", "privacy", "cancelled",
                            # The transport succeeded and the model answered;
                            # our translation rejected the answer. Distinct
                            # from `callback_error`, which made a
                            # served-but-unusable answer look like a transport
                            # fault.
                            "malformed_answer"})
FINISH_REASONS = frozenset({"stop", "length", "other"})
PROVIDER_OUTCOMES = frozenset({"ok", "provider_call_timeout", "provider_error",
                               "provider_rate_limited", "provider_unavailable",
                               "provider_rejected", "cancelled"})

_FIELDS = (_INTEGER_FIELDS | _FLOAT_FIELDS
           | {"event", "correlation_id", "phase", "outcome", "status", "reason",
              "model", "finish_reason", "timeout_category", "provider",
              # Which check a malformed answer failed, as a fixed identifier
              # from a closed set. `malformed_output` alone says only that the
              # answer was unusable, so a live failure could only be
              # re-diagnosed by re-running it.
              "constraint"})


def _bounded_int(value):
    return type(value) is int and 0 <= value <= 10 ** 9


def sanitize(event):
    """Return a log-safe copy of ``event``, or ``{}`` if the event is not recognized."""
    if type(event) is not dict or event.get("event") not in EVENTS:
        return {}
    clean = {"event": event["event"]}
    for key, value in event.items():
        if key == "event" or key not in _FIELDS:
            continue
        if key in _INTEGER_FIELDS:
            if _bounded_int(value):
                clean[key] = value
        elif key in _FLOAT_FIELDS:
            if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1000:
                clean[key] = round(float(value), 6)
        elif key == "correlation_id":
            if type(value) is str and _CORRELATION.fullmatch(value):
                clean[key] = value
        elif key == "model":
            if type(value) is str and _MODEL.fullmatch(value):
                clean[key] = value
        elif key == "provider":
            # Provider names are short identifiers ("Venice", "DeepInfra"). Anything
            # that does not fit the pattern is dropped rather than logged.
            if type(value) is str and _PROVIDER.fullmatch(value):
                clean[key] = value
        elif key == "phase":
            if value in PHASES:
                clean[key] = value
        elif key == "outcome":
            if value in PHASE_OUTCOMES | PROVIDER_OUTCOMES | {"approved", "withheld"}:
                clean[key] = value
        elif key == "constraint":
            # A fixed identifier from a closed set, never model text. The
            # pattern is the same one used for other short identifiers, so
            # anything that is not a plain lowercase token is dropped rather
            # than logged.
            if type(value) is str and _IDENTIFIER.fullmatch(value):
                clean[key] = value
        elif key == "finish_reason":
            if value in FINISH_REASONS:
                clean[key] = value
        elif key in {"status", "reason", "timeout_category"}:
            if type(value) is str and _IDENTIFIER.fullmatch(value):
                clean[key] = value
    return clean


def log_event(logger, event):
    """Emit one sanitized JSON line; never raise into the correction path."""
    try:
        clean = sanitize(event)
        if clean:
            logger.info("correction_event %s", json.dumps(clean, sort_keys=True))
    except Exception:
        pass
