"""Single-request, no-retry correction adapter. Never log request or response data."""
import asyncio
import json
import re
import time

import aiohttp

from disclosure import POLICY_ID, verify_disclosure
from pipeline_config import OR_SLUG_DRAFTER, estimate_cost
from source_correction import PrivacyReviewRequired, ProviderCallTimeout, SourceDocument


MAX_REQUEST_BYTES = 96_000
MAX_RESPONSE_BYTES = 512_000
# Output allowance per provider call. Inventory and review must copy or cite every
# span, so output grows with input; see docs/CORRECTION_TIMEOUT_DIAGNOSIS_2026-11-02.md.
MAX_OUTPUT_TOKENS = 12000
# Per-request read timeout. Live runs timed out at 30s: the inventory call for
# a ~1KB source returns 12000 max_tokens of JSON from a reasoning model, which
# regularly exceeds 30s. The orchestrator allows 180s for the whole analyze
# request, and source_correction.MAX_TIMEOUT_SECONDS bounds the operation, so
# this must be well below that but comfortably above observed call latency.
PROVIDER_TIMEOUT_SECONDS = 60.0

# The disclosure gate. It is enabled only by an explicit operator setting; the
# default (unset/false) keeps external correction disabled. Enabling it does not
# skip the checks: `verify_disclosure` still runs against the exact payload that
# would be sent, per request, and any uncovered PII withholds the result.
#
# Scope of what this protects: candidate source documents (the raw uploaded
# text), which the normal generation path never sends verbatim. Detection is the
# same unambiguous category set already applied to the normal pipeline
# (email/phone/SSN/street address/LinkedIn/contact header lines); names,
# employers, schools and city/state are intentionally retained as resume
# content. It is not a general proof that arbitrary text is non-identifying.
DISCLOSURE_SETTING = "CORRECTION_PROVIDER_DISCLOSURE"
_ENABLED_VALUES = {"1", "true", "yes", "on"}


def _approved_disclosure_available():
    """True only when an operator explicitly enabled the approved policy."""
    import os

    enabled = (os.getenv(DISCLOSURE_SETTING, "") or "").strip().lower()
    return enabled in _ENABLED_VALUES


def approved_disclosure_policy():
    """Return the enforceable policy id, or None when disclosure is not approved."""
    return POLICY_ID if _approved_disclosure_available() else None


def snapshot_sources(resume_text, documents, supplied):
    """Validate the Node join exactly; source designation is not proof of ownership."""
    if type(resume_text) is not str or not resume_text.strip():
        return None
    if not supplied:
        return (SourceDocument("resume", resume_text),)
    if type(documents) is not list or not documents:
        return None
    sources, ids = [], set()
    for doc in documents:
        if (type(doc) is not dict or set(doc) != {"source_id", "text", "kind"}
                or type(doc["source_id"]) is not str
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", doc["source_id"])
                or doc["source_id"] in ids or type(doc["text"]) is not str
                or not doc["text"].strip() or doc["kind"] != "candidate_document"):
            return None
        ids.add(doc["source_id"])
        sources.append(SourceDocument(doc["source_id"], doc["text"], doc["kind"]))
    if "\n\n---\n\n".join(source.text for source in sources) != resume_text:
        return None
    return tuple(sources)


def _strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject_constant(_):
        raise ValueError("non-finite JSON number")

    return json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant)


class CorrectionProvider:
    def __init__(self, api_key, session, *, model=OR_SLUG_DRAFTER, observer=None):
        self.api_key = api_key
        self.session = session
        self.model = model
        # Optional sanitized event sink (see correction_events); receives no text.
        self.observer = observer
        self.calls = 0
        self.usage = {"calls": [], "total_prompt_tokens": 0,
                      "total_completion_tokens": 0, "total_tokens": 0,
                      "estimated_cost_usd": 0.0, "failures": []}

    async def __call__(self, phase, payload):
        if phase not in {"source_inventory", "initial_review", "repair", "post_patch_review"} or self.calls >= 6:
            raise ValueError("correction call budget exhausted")
        if not self.api_key or self.session is None:
            raise ValueError("correction provider unavailable")
        serialized_payload = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        system_content = payload["contract"] + " Return only a JSON object."
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": serialized_payload},
            ],
            "temperature": 0.1,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "stream": False,
            "provider": {"allow_fallbacks": False},
        }
        # Measure the complete JSON envelope, including JSON escaping.
        if len(json.dumps(request, ensure_ascii=True, allow_nan=False).encode("utf-8")) > MAX_REQUEST_BYTES:
            raise ValueError("correction request too large")
        # Never submit the original evidence or draft under an unspecified
        # disclosure policy. The check runs against the EXACT sources about to be
        # serialized, not against a caller's claim that they were redacted.
        policy = approved_disclosure_policy()
        if policy is None:
            raise PrivacyReviewRequired
        if not verify_disclosure(payload.get("sources"), approved_policy=policy):
            raise PrivacyReviewRequired
        self.calls += 1
        # POST is not retried by aiohttp; redirects and model fallbacks are disabled.
        recorded = False
        started = time.monotonic()
        outcome = "provider_error"
        finish = None
        request_index = self.calls
        try:
            async with self.session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=request, timeout=aiohttp.ClientTimeout(total=PROVIDER_TIMEOUT_SECONDS),
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise ValueError("provider HTTP failure")
                chunks, size = [], 0
                async for chunk in response.content.iter_chunked(65536):
                    size += len(chunk)
                    if size > MAX_RESPONSE_BYTES:
                        raise ValueError("provider response too large")
                    chunks.append(chunk)
                raw = b"".join(chunks)
                data = _strict_json(raw.decode("utf-8"))
            if type(data) is not dict:
                raise ValueError("invalid provider envelope")
            self._record_usage(data, phase)
            recorded = True
            if data.get("model") != self.model:
                raise ValueError("unexpected provider model")
            choices = data.get("choices")
            if type(choices) is not list or len(choices) != 1:
                raise ValueError("invalid provider choices")
            choice = choices[0]
            finish = choice.get("finish_reason") if type(choice) is dict else None
            if type(choice) is not dict or finish != "stop":
                raise ValueError("incomplete provider response")
            message = choice.get("message")
            if type(message) is not dict or type(message.get("content")) is not str:
                raise ValueError("missing provider content")
            parsed = _strict_json(message["content"])
            if type(parsed) is not dict:
                raise ValueError("invalid provider response")
            outcome = "ok"
            return parsed
        except asyncio.CancelledError:
            outcome = "cancelled"
            self.usage["failures"].append({"stage": "source_correction", "phase": phase,
                                            "reason": "cancelled", "usage_unknown": not recorded})
            raise
        except (asyncio.TimeoutError, TimeoutError) as exc:
            # A single-call read timeout is NOT the overall deadline expiring.
            # Tag it so `correct()` can report which failure actually happened.
            outcome = "provider_call_timeout"
            self.usage["failures"].append({"stage": "source_correction", "phase": phase,
                                            "reason": "provider_call_timeout",
                                            "usage_unknown": not recorded})
            raise ProviderCallTimeout(phase) from exc
        except Exception:
            self.usage["failures"].append({"stage": "source_correction", "phase": phase,
                                            "reason": "provider_error", "usage_unknown": not recorded})
            raise
        finally:
            self._emit(phase, request_index, outcome, started, recorded, finish)

    def _emit(self, phase, request_index, outcome, started, recorded, finish):
        """Report one call's timing and token counts. Never includes request or response text."""
        if self.observer is None:
            return
        event = {"event": "provider.call", "phase": phase, "outcome": outcome,
                 "model": self.model, "request_index": request_index,
                 "elapsed_ms": int(round((time.monotonic() - started) * 1000)),
                 "max_tokens": MAX_OUTPUT_TOKENS,
                 "call_timeout_ms": int(PROVIDER_TIMEOUT_SECONDS * 1000)}
        if finish is not None:
            event["finish_reason"] = finish if finish in {"stop", "length"} else "other"
        if recorded and self.usage["calls"]:
            last = self.usage["calls"][-1]
            event.update(prompt_tokens=last["prompt_tokens"],
                         completion_tokens=last["completion_tokens"],
                         total_tokens=last["total_tokens"], cost_usd=last["cost_usd"])
        try:
            self.observer(event)
        except Exception:
            pass

    def _record_usage(self, data, phase):
        if not isinstance(data, dict) or not isinstance(data.get("usage"), dict):
            raise ValueError("provider usage missing")
        usage = data["usage"]
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if type(prompt) is not int or type(completion) is not int or prompt < 0 or completion < 0:
            raise ValueError("invalid provider usage")
        total = usage.get("total_tokens")
        if type(total) is not int or total < prompt + completion:
            raise ValueError("invalid provider usage")
        cost = estimate_cost(self.model, prompt, completion)
        self.usage["calls"].append({"stage": "source_correction", "phase": phase,
            "model": self.model,
            "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": total, "cost_usd": cost})
        self.usage["total_prompt_tokens"] += prompt
        self.usage["total_completion_tokens"] += completion
        self.usage["total_tokens"] += total
        self.usage["estimated_cost_usd"] += cost