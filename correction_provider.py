"""Single-request, no-retry correction adapter. Never log request or response data."""
import asyncio
import json
import math
import re
import time

import aiohttp

import correction_protocol
from disclosure import (
    POLICY_ID,
    payload_disclosure_errors,
    redact_payload,
    verify_disclosure,
)
from pipeline_config import OR_SLUG_DRAFTER, estimate_cost
from source_correction import (
    MalformedReviewAnswer,
    PrivacyReviewRequired,
    ProviderCallTimeout,
    ProviderRateLimited,
    ProviderRejectedRequest,
    ProviderUnavailable,
    SourceDocument,
    UnjudgedReview,
)


MAX_REQUEST_BYTES = 96_000
MAX_RESPONSE_BYTES = 512_000
# Output allowance per provider call. Inventory and review must copy or cite every
# span, so output grows with input; see docs/CORRECTION_TIMEOUT_DIAGNOSIS_2026-11-02.md.
MAX_OUTPUT_TOKENS = 12000
# Ceiling for one provider request. The orchestrator allows 180s (JSON) or 300s
# (SSE) for the whole analyze request and the engine caps the operation itself, so
# this must sit below those.
#
# Measured 2026-11-02: the initial review of a 4KB draft is a 13.6KB request that
# answered in 38.3s warm, and the live run spent ~48s on inventory first. A fixed 60s
# ceiling therefore killed a review that was still working, with ~12s of the granted
# ^120s budget unspent, and reported provider_call_timeout_initial_review as if the
# request were too slow. The ceiling is now the larger of its own floor and the
# budget actually remaining, so a call is only abandoned when the correction could
# not have finished anyway.
PROVIDER_TIMEOUT_SECONDS = 60.0
# Never give a single call more than this, however much budget is left: one slow
# phase must not be able to consume the whole correction and leave no time for the
# review-and-repair cycle that the answer exists to enable.
PROVIDER_TIMEOUT_CEILING_SECONDS = 120.0
# Hard ceiling on provider calls for one correction. Six is the documented
# two-repair path (inventory + review + repair + re-review + repair + re-review).
REPAIR_PATH_CALLS = 6
# Seven allows the single bounded completion re-review that finishes a review the
# model answered only partially. That headroom is reachable only by a request
# carrying the `must_judge` hint, so it cannot be spent by a second repair pass or
# by a direct caller.
MAX_CALLS = 7

# --- Transient-failure retry allowance -------------------------------------
# These are the statuses that mean "this endpoint cannot serve the request *right
# now*". Measured 2026-11-02: the same valid 13.6KB review request returned 504 in
# ~8.5s on some attempts and 200 in 26-39s on others, so they are transient by
# nature and repeating them is the correct response.
RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})
# These mean the request itself is wrong (auth, schema, unknown model). Repeating an
# identical request produces an identical refusal, and a 401 must reach an operator
# instead of being retried into silence.
REJECTED_STATUSES = frozenset({400, 401, 403, 404, 422})
# One initial attempt plus at most this many additional ones. Deliberately small: a
# retry is for a transient blip, not for waiting out an outage.
RETRY_MAX_ATTEMPTS = 3
# Backoff when the provider does not say how long to wait. Chosen so two retries fit
# inside a normal phase budget without crowding out the work itself.
RETRY_BASE_DELAY_SECONDS = 1.5
RETRY_MAX_DELAY_SECONDS = 15.0
# Do not start another attempt with less than this left: it would mostly burn budget
# that a later phase (or the engine's own deadline check) needs.
RETRY_MIN_REMAINING_SECONDS = 10.0


async def _sleep(seconds):
    """Indirection so tests can prove retry behaviour without waiting it out."""
    await asyncio.sleep(seconds)

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
        # Remaining overall budget offered by the engine, and the timeout actually
        # applied to the last call (for the sanitized event).
        self._remaining_seconds = None
        self._last_call_timeout = PROVIDER_TIMEOUT_SECONDS
        # Engine phase calls. This is what the call budget (MAX_CALLS) and the
        # usage-accounting cross-check both count, so transport retries must NOT
        # increment it: a retried inventory is still one phase, and letting retries
        # eat the phase budget would discard the completed work they are meant to
        # protect.
        self.calls = 0
        # Transport attempts, including retries. Diagnostic only.
        self.attempts = 0
        self.usage = {"calls": [], "total_prompt_tokens": 0,
                      "total_completion_tokens": 0, "total_tokens": 0,
                      "estimated_cost_usd": 0.0, "failures": []}

    def set_budget(self, remaining_seconds):
        """Offer the engine's remaining overall budget for the next call.

        The engine calls this before each phase. Spending the whole remaining budget
        on one call is only correct when that call is the last thing that can
        usefully happen, and it is not: a review that finishes late still needs a
        repair pass and a re-review. So a call may use the remaining budget up to
        ``PROVIDER_TIMEOUT_CEILING_SECONDS`` and never less than the fixed floor.
        """
        if (type(remaining_seconds) in (int, float)
                and math.isfinite(remaining_seconds) and remaining_seconds > 0):
            self._remaining_seconds = float(remaining_seconds)

    def _call_timeout(self):
        """Timeout for this call, sized from the budget the engine offered."""
        remaining = self._remaining_seconds
        if remaining is None:
            return PROVIDER_TIMEOUT_SECONDS
        return max(PROVIDER_TIMEOUT_SECONDS,
                   min(remaining, PROVIDER_TIMEOUT_CEILING_SECONDS))

    def _retry_budget(self, phase_started):
        """Seconds still available for retrying, or None when no budget was offered.

        The engine's offer covers the whole phase, so time already spent counts
        against it. Leaving ``RETRY_MIN_REMAINING_SECONDS`` in hand means an attempt
        is only started when it has a realistic chance of finishing.
        """
        if self._remaining_seconds is None:
            return None
        return self._remaining_seconds - (time.monotonic() - phase_started)

    @staticmethod
    def _retry_delay(response, attempt):
        """How long to wait before the next attempt, and whether the provider named it.

        Returns ``(delay_seconds, named_by_provider)``.

        An explicit ``Retry-After`` is returned **uncapped**. Shortening it to something
        that fits would disobey the instruction the provider just gave, which in practice
        earns another refusal and spends budget that cannot be recovered. The caller
        compares the delay against the budget and stops cleanly when it does not fit, so
        an absurd advertised value fails safe instead of pinning the request.

        Only the integer-seconds form counts as an instruction. An HTTP-date or a missing
        header falls back to backoff, which is our own choice and so may be capped.
        """
        raw = None
        try:
            raw = response.headers.get("Retry-After") if response is not None else None
        except Exception:
            raw = None
        if isinstance(raw, str) and raw.strip().isdigit():
            return float(raw.strip()), True
        return (min(RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)), RETRY_MAX_DELAY_SECONDS),
                False)

    def _record_transport_failure(self, phase, reason, attempt, status, usage_unknown=True):
        """Record one non-successful transport attempt without any content.

        ``usage_unknown`` stays True: a refused or incomplete request may still have
        been billed for its prompt, so treating it as free would understate spend.
        """
        entry = {"stage": "source_correction", "phase": phase, "reason": reason,
                 "attempt": attempt, "usage_unknown": usage_unknown}
        if type(status) is int:
            entry["http_status"] = status
        self.usage["failures"].append(entry)

    async def __call__(self, phase, payload):
        # A direct caller past the ceiling is rejected before the disclosure gate so
        # "budget exhausted" is the reason it sees, not a privacy error.
        if self.calls >= MAX_CALLS:
            raise ValueError("correction call budget exhausted")
        # The engine's phases. `restore` and `post_restore_review` were missing: the
        # engine could ask for a restoration and the adapter rejected it as an
        # unknown phase, so the bounded restoration added for delivery run 4 could
        # never run through the real provider. Both are ordinary provider calls and
        # are treated as such below: same disclosure gate, same budget, same retry
        # classification, same telemetry, same capture.
        if phase not in {"source_inventory", "initial_review", "repair",
                         "post_patch_review", "restore", "post_restore_review"}:
            raise ValueError("correction call budget exhausted")
        # Recovery headroom is not spendable by an arbitrary caller: it exists only
        # for the engine's single completion re-review, which is the only request
        # that carries the `must_judge` hint naming the lines it must answer.
        if self.calls >= REPAIR_PATH_CALLS and "must_judge" not in payload:
            raise ValueError("correction call budget exhausted")
        if not self.api_key or self.session is None:
            raise ValueError("correction provider unavailable")
        # The engine asks for exact offsets and gapless partitions; live models
        # cannot produce those reliably. Translate to index-only questions here,
        # and translate the answers back to the strict shape the engine verifies.
        # A payload without the engine's shape is passed through untouched so
        # direct callers and their error paths behave as before.
        original_payload = payload
        call_timeout = self._call_timeout()
        self._last_call_timeout = call_timeout
        translated = None
        try:
            if phase == "source_inventory" and "sources" in original_payload:
                translated = correction_protocol.indexed_inventory_request(original_payload)
            elif (phase in {"initial_review", "post_patch_review", "post_restore_review"}
                    and "sources" in original_payload and "source_claims" in original_payload):
                # A post-restore review is a review: same question, same translation,
                # same validation. It differs only in that the draft has changed.
                translated = correction_protocol.indexed_review_request(original_payload)
            elif phase == "restore" and "source_claims" in original_payload:
                translated = correction_protocol.indexed_restore_request(original_payload)
        except correction_protocol.MalformedIndexedResponse:
            # Never send an ambiguous restore request without the indexed
            # line-to-source mapping; the unindexed fallback cannot recover it.
            if phase == "restore":
                raise
            translated = None
        if translated is not None:
            payload = translated
        outgoing = {key: value for key, value in payload.items() if not key.startswith("_")}
        # Bound the request before anything else looks at it, so an oversized
        # payload is rejected for being oversized regardless of its content.
        if len(json.dumps(outgoing, ensure_ascii=True, allow_nan=False).encode("utf-8")) > MAX_REQUEST_BYTES:
            raise ValueError("correction request too large")
        # Never submit the original evidence or draft under an unspecified
        # disclosure policy. The check runs against the EXACT structure about to be
        # serialized, not against a caller's claim that it was redacted.
        policy = approved_disclosure_policy()
        if policy is None:
            raise PrivacyReviewRequired
        if not verify_disclosure(original_payload.get("sources"), approved_policy=policy):
            raise PrivacyReviewRequired
        # The caller redacts source documents, but draft lines, segment text and
        # claim quotes travel in the same request and were never covered. Apply the
        # same length-preserving substitution to every free-text value actually
        # transmitted, then verify the result. Width is preserved, so the engine's
        # code-point offsets stay valid; the reply carries only line ids and
        # segment indices, never offsets or document text.
        try:
            outgoing, _redactions = redact_payload(outgoing)
        except ValueError:
            # Text that cannot be made PII-free without changing length is not sent
            # at all: a shifted offset would invalidate every citation.
            raise PrivacyReviewRequired
        if payload_disclosure_errors(outgoing, approved_policy=policy):
            raise PrivacyReviewRequired
        serialized_payload = json.dumps(outgoing, ensure_ascii=False, allow_nan=False)
        system_content = outgoing["contract"] + " Return only a JSON object."
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": serialized_payload},
            ],
            "temperature": 0.1,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "stream": False,
            # Fallbacks must stay enabled. The correction model is served by two
            # endpoints; pinning one meant a saturated provider (429
            # ``engine_overloaded``) failed the whole correction instead of routing
            # to the other. See ProviderRateLimited. Redirects remain disabled, so
            # this cannot move the request to a third-party URL.
            "provider": {"allow_fallbacks": True},
        }
        # The complete envelope is measured again after redaction, including
        # JSON escaping, because the padded placeholders can change the byte size.
        if len(json.dumps(request, ensure_ascii=True, allow_nan=False).encode("utf-8")) > MAX_REQUEST_BYTES:
            raise ValueError("correction request too large")
        self.calls += 1
        # POST is not retried by aiohttp; redirects and model fallbacks are disabled.
        # A transient availability status is the ONE thing retried, and only here, so
        # a late failure repeats just this phase instead of discarding the stages
        # that already succeeded.
        recorded = False
        started = time.monotonic()
        finish = None
        provider_name = None
        request_index = self.calls
        attempt = 0
        try:
            while True:
                attempt += 1
                self.attempts += 1
                # An attempt is only "ok" once the whole envelope has been
                # validated. Setting it right after the JSON parse would report a
                # 200 carrying an unusable envelope as a success.
                attempt_outcome = "provider_error"
                attempt_status = None
                attempt_recorded = False
                attempt_started = time.monotonic()
                retry_delay = None
                delay_named = False
                # True once a 200 body has parsed as JSON. Distinct from
                # ``attempt_outcome == "ok"``, which additionally requires the whole
                # envelope to validate: conflating the two made a served-but-unusable
                # 200 look like a transport failure.
                served = False
                try:
                    async with self.session.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=request, timeout=aiohttp.ClientTimeout(total=call_timeout),
                        allow_redirects=False,
                    ) as response:
                        attempt_status = response.status if type(response.status) is int else None
                        if response.status in RETRYABLE_STATUSES:
                            # Availability, not a malformed request. Retry inside the
                            # deadline; if the allowance runs out the phase fails as
                            # rate-limited (429) or unavailable (502/503/504) rather
                            # than as a generic error.
                            attempt_outcome = ("provider_rate_limited"
                                               if response.status == 429
                                               else "provider_unavailable")
                            retry_delay, delay_named = self._retry_delay(response, attempt)
                            retryable = True
                        elif response.status in REJECTED_STATUSES:
                            attempt_outcome = "provider_rejected"
                            retryable = False
                        elif response.status != 200:
                            attempt_outcome = "provider_error"
                            retryable = False
                        else:
                            chunks, size = [], 0
                            async for chunk in response.content.iter_chunked(65536):
                                size += len(chunk)
                                if size > MAX_RESPONSE_BYTES:
                                    raise ValueError("provider response too large")
                                chunks.append(chunk)
                            raw = b"".join(chunks)
                            data = _strict_json(raw.decode("utf-8"))
                            retryable = False
                            served = True

                    if not served:
                        # Record the attempt, then decide whether to repeat it.
                        if attempt_outcome == "provider_rejected":
                            self._record_transport_failure(
                                phase, "provider_rejected_request", attempt, attempt_status)
                            raise ProviderRejectedRequest(phase, attempt_status)
                        remaining = self._retry_budget(started)
                        fits = (remaining is None
                                or remaining - retry_delay >= RETRY_MIN_REMAINING_SECONDS)
                        if retryable and attempt < RETRY_MAX_ATTEMPTS and fits:
                            self._record_transport_failure(
                                phase, attempt_outcome, attempt, attempt_status)
                            await _sleep(retry_delay)
                            continue
                        # Allowance spent, or the wait does not fit the remaining
                        # budget. A delay the provider named is never shortened to make
                        # it fit: waiting less than instructed usually earns another
                        # refusal and burns budget that cannot be recovered, so stopping
                        # cleanly with an accurate reason is the better outcome.
                        self._record_transport_failure(
                            phase, attempt_outcome, attempt, attempt_status)
                        if retryable and delay_named and not fits:
                            # Record why no further attempt was made, without content.
                            self._record_transport_failure(
                                phase, "retry_after_exceeded_budget", attempt,
                                attempt_status, usage_unknown=False)
                        if attempt_outcome == "provider_rate_limited":
                            raise ProviderRateLimited(phase)
                        if attempt_outcome == "provider_unavailable":
                            raise ProviderUnavailable(phase, attempt_status)
                        raise ValueError("provider HTTP failure")

                    if type(data) is not dict:
                        raise ValueError("invalid provider envelope")
                    self._record_usage(data, phase)
                    attempt_recorded = recorded = True
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
                    # Fully validated: this is the one place an attempt becomes "ok".
                    attempt_outcome = "ok"
                    if type(data.get("provider")) is str:
                        provider_name = data["provider"]
                except asyncio.CancelledError:
                    attempt_outcome = "cancelled"
                    self.usage["failures"].append({"stage": "source_correction", "phase": phase,
                                                    "reason": "cancelled",
                                                    "attempt": attempt,
                                                    "usage_unknown": not recorded})
                    raise
                except (asyncio.TimeoutError, TimeoutError) as exc:
                    # A single-call read timeout is NOT the overall deadline expiring.
                    # Tag it so `correct()` can report which failure actually happened.
                    #
                    # Deliberately NOT retried, even though it is availability-shaped.
                    # A read timeout has already consumed most of a phase's budget, so
                    # repeating it would usually guarantee the correction fails with
                    # nothing left, instead of stopping cleanly with a reason that says
                    # what happened. Only the statuses the endpoint itself reports as
                    # transient (429/502/503/504) are worth another attempt.
                    self._record_transport_failure(
                        phase, "provider_call_timeout", attempt, attempt_status)
                    raise ProviderCallTimeout(phase) from exc
                finally:
                    self._emit(phase, request_index, attempt_outcome, attempt_started,
                               attempt_recorded, finish if attempt_outcome == "ok" else None,
                               attempt, attempt_status,
                               provider_name if attempt_outcome == "ok" else None)
                break

            # Translate the index-list answer into the strict shape the engine
            # verifies. A model (or a transport double) may answer in that strict
            # shape directly; prefer it verbatim, the engine still validates it.
            #
            # A restoration is the exception: its wire form deliberately omits the
            # internal source_claim_id and the `before` echo, so a wire answer whose
            # only key is "restorations" is NOT the strict form even though the key
            # matches. Skipping translation there returned answers the engine could
            # never accept, and every restoration failed as source_facts_missing.
            if translated is None:
                return parsed
            if phase == "restore":
                return correction_protocol.strict_restore_from_indexed(parsed, payload)
            strict = ("source_claims" if phase == "source_inventory"
                      else {"claims", "representations"})
            if type(parsed) is dict and set(parsed) == strict:
                return parsed
            if phase == "source_inventory":
                return correction_protocol.strict_inventory_from_indexed(parsed, payload)
            try:
                return correction_protocol.strict_review_from_indexed(parsed, payload)
            except correction_protocol.UnjudgedLines as exc:
                # The answer parsed but skipped part of the draft. Report it as an
                # incomplete review so the reason is accurate; a generic malformed
                # output hid this live and sent the draft into a wasted repair.
                # Deliberately NOT retried: the answer was valid, just incomplete.
                raise UnjudgedReview(exc.line_ids) from exc
            except correction_protocol.MalformedIndexedResponse as exc:
                # The answer parsed as JSON but could not be mapped onto the
                # engine's strict shape: a duplicate line id, an out-of-range
                # segment index, an unknown key. That is OUR validation
                # rejecting the model's answer, not a transport failure.
                #
                # Measured 2026-11-07: this fell through to the generic handler
                # below and was reported as `provider_error`, so a live run
                # looked like an OpenRouter outage when the reviewer had simply
                # returned an unusable answer. `UnjudgedLines` was already
                # handled; its sibling was not, even though it subclasses the
                # same base. The engine maps `_Invalid` to `malformed_output`,
                # so raise that instead of mislabelling our own check.
                raise MalformedReviewAnswer(phase) from exc
        except (ProviderRateLimited, ProviderUnavailable, ProviderRejectedRequest,
                ProviderCallTimeout, UnjudgedReview, MalformedReviewAnswer,
                PrivacyReviewRequired):
            raise
        except Exception:
            self.usage["failures"].append({"stage": "source_correction", "phase": phase,
                                            "reason": "provider_error", "attempt": max(attempt, 1),
                                            "usage_unknown": not recorded})
            raise

    def _emit(self, phase, request_index, outcome, started, recorded, finish,
              attempt=None, http_status=None, provider_name=None):
        """Report one attempt's timing, status and token counts. Never any content.

        ``attempt`` and ``http_status`` make a retry sequence legible after the fact:
        the live 2026-11-02 runs could only be explained by re-sending captured
        requests because the log did not say "this was attempt 2, it returned 504".
        """
        if self.observer is None:
            return
        event = {"event": "provider.call", "phase": phase, "outcome": outcome,
                 "model": self.model, "request_index": request_index,
                 "elapsed_ms": int(round((time.monotonic() - started) * 1000)),
                 "max_tokens": MAX_OUTPUT_TOKENS,
                 "call_timeout_ms": int(self._last_call_timeout * 1000)}
        if type(attempt) is int and attempt > 0:
            event["attempt"] = attempt
        if type(http_status) is int:
            event["http_status"] = http_status
        if type(provider_name) is str and provider_name:
            # Which endpoint actually served the attempt. The 2026-11-02 runs could
            # not be explained without this: the same request succeeded on one
            # endpoint and 504'd on the other.
            event["provider"] = provider_name
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