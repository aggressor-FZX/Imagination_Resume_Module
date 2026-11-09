"""Provider adapter v2 — sufficient output budget, full per-call telemetry.

Why this is a separate module
-----------------------------
v1's adapter (`experiment/run_comparison.py::make_asker`) hard-coded
`max_tokens: 1200` and returned `(text, cost)`. It also discarded
`finish_reason` and `usage`, so after the run there was no way to tell a truncated
answer from a refused one without re-deriving it from the byte length.

Measured consequence: 11 of 12 legacy answers were severed mid-string at
~3,700-4,900 characters. The legacy contract was therefore never evaluated. Item 5
of the brief forbids comparing a truncated legacy answer with a complete revised
answer as a judgment-quality result, and the only way to honour that is to make
truncation *impossible to miss* and *impossible to record silently*.

What v2 does differently
------------------------
1. **A sufficient budget.** `DEFAULT_MAX_TOKENS = 8192`. A complete span envelope
   over a seven-bullet résumé needs roughly 2-3k output tokens once every line is
   partitioned into claims AND every source inventory item is accounted for in
   `representations`; 1,200 was never enough even in principle. The value used is
   carried in the result so a later reader never has to guess.

2. **Truncation is a recorded, first-class outcome.** `finish_reason` is kept
   verbatim. A `length` finish raises `TruncatedResponse` rather than returning a
   plausible-looking fragment, so an incomplete answer cannot enter the results as
   if it were a judgment.

3. **Usage is retained.** Prompt/completion/total tokens and cost per call. Without
   usage a cost comparison between arms is not reproducible.

4. **Retries are recorded, never disguised.** Every retry is appended to a ledger
   with its reason and the attempt number, so a retry cannot appear as a
   first-try success.

The adapter refuses to guess. If the provider returns something it cannot read, the
error is raised and recorded by the runner — it is never converted into an empty
answer that later looks like a refusal.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# A full span envelope does not fit in 1200 tokens. See module docstring.
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TIMEOUT_S = 240
DEFAULT_MODEL = "qwen/qwen3-coder"


class TruncatedResponse(RuntimeError):
    """The provider stopped because the output budget ran out.

    Raised instead of returning the fragment, because a fragment that happens to
    parse would be recorded as a real judgment. The runner records this as
    `operational:truncation` and the attempt stays in the record.
    """


@dataclass
class CallResult:
    """Everything one provider call produced. Nothing is summarised away."""

    text: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    model: str = ""
    provider: str = ""
    latency_s: float = 0.0
    retries: list[dict] = field(default_factory=list)
    raw_usage: dict = field(default_factory=dict)
    truncated: bool = False
    # How many HTTP requests this ONE logical call actually sent, retries
    # included. Without it, a report that sums `cost_usd` over logical calls
    # silently omits every retried attempt.
    requests_sent: int = 1
    # True when the provider answered 2xx but reported no usage, so the real cost
    # of that request is unknown rather than zero.
    usage_missing: bool = False

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RequestMeter:
    """Per-REQUEST accounting, so no billable attempt can go unrecorded.

    Why this is separate from a per-CASE budget
    -------------------------------------------
    The previous budget counted one unit per logical `ask()`. A logical call can
    send up to three HTTP requests (the retry loop), and each of those is billed
    by the provider whether or not it produced usable text. Counting logical calls
    therefore under-reported both requests and spend, and a `$1.00` ceiling
    described in a report could be exceeded without the record showing it.

    What is HARD and what is NOT
    ----------------------------
    * `max_requests` is a **hard** ceiling: it is checked before every single HTTP
      attempt, including retries, so the run cannot send more than that number of
      requests no matter how many fail.
    * `max_cost_usd` is only as hard as the provider's reporting. It is enforced
      before each attempt using cost observed SO FAR, and every 2xx response that
      omits usage increments `unmetered_requests`. If that counter is non-zero, the
      true spend is **unknown and may exceed the ceiling** — this class reports
      that rather than asserting a limit it cannot prove.
    """

    max_requests: int
    max_cost_usd: float
    requests_sent: int = 0
    billable_requests: int = 0
    unmetered_requests: int = 0
    cost_usd: float = 0.0
    logical_calls: int = 0
    attempts: list[dict] = field(default_factory=list)

    def can_send(self) -> tuple[bool, str]:
        if self.requests_sent >= self.max_requests:
            return False, (f"request ceiling reached "
                           f"({self.requests_sent}/{self.max_requests})")
        if self.cost_usd >= self.max_cost_usd:
            return False, (f"observed-cost ceiling reached "
                           f"({self.cost_usd:.4f}/{self.max_cost_usd:.2f} USD)")
        return True, ""

    def record_attempt(self, *, label: str, attempt: int, status: str,
                       cost: float | None, usage_reported: bool) -> None:
        """Meter one HTTP attempt. Called for FAILED attempts too.

        Any attempt without reported usage makes total spend UNKNOWN, not zero.
        This deliberately includes 429/5xx and transport errors: no provider
        guarantees that a 5xx went unbilled, and a timeout may have been billed
        before the connection dropped. Claiming `cost_is_known` after a failure
        would be a guess presented as a fact.
        """
        self.requests_sent += 1
        if usage_reported:
            self.billable_requests += 1
            self.cost_usd += float(cost or 0.0)
        else:
            self.unmetered_requests += 1
        self.attempts.append({
            "label": label,
            "attempt": attempt,
            "status": status,
            "cost_usd": cost,
            "usage_reported": usage_reported,
        })

    @property
    def cost_is_known(self) -> bool:
        """Whether total spend is knowable from this record."""
        return self.unmetered_requests == 0

    def to_dict(self) -> dict:
        return {
            "max_requests": self.max_requests,
            "max_cost_usd": self.max_cost_usd,
            "requests_sent": self.requests_sent,
            "billable_requests": self.billable_requests,
            "unmetered_requests": self.unmetered_requests,
            "logical_calls": self.logical_calls,
            "cost_usd": round(self.cost_usd, 6),
            "cost_is_known": self.cost_is_known,
            "ceiling_statement": (
                "The request ceiling is hard: it is checked before every HTTP "
                "attempt including retries. The cost ceiling is enforced against "
                "observed cost only; it is NOT a proof of maximum spend when any "
                "response omits usage."),
            "unmetered_caveat": (
                None if self.cost_is_known else
                f"{self.unmetered_requests} response(s) omitted usage, so true "
                "spend is unknown and may exceed the stated cost ceiling."),
        }


def _usage_cost(usage: dict) -> tuple[float, bool]:
    """Cost from a usage object, and whether usage was reported at all.

    OpenRouter's chat-completions `usage` carries `cost`. Reading only
    `total_cost` reported $0.00 for every call and made the cost ceiling
    unenforceable. Both keys are honoured, in order of authority.

    A missing/empty usage object is NOT a free request — it is an UNKNOWN one.
    Returning 0.0 with `reported=False` lets the meter say so instead of quietly
    recording a zero.
    """
    if not usage:
        return 0.0, False
    if usage.get("cost") is not None:
        return float(usage["cost"]), True
    if usage.get("total_cost") is not None:
        return float(usage["total_cost"]), True
    return 0.0, False


def make_asker(api_key: str | None = None, *,
               model: str | None = None,
               max_tokens: int = DEFAULT_MAX_TOKENS,
               timeout_s: int = DEFAULT_TIMEOUT_S,
               ledger: list[dict] | None = None,
               meter: RequestMeter | None = None,
               max_attempts: int = 3,
               sleep_between_retries: bool = True):
    """Build an async `ask` callable returning a `CallResult`.

    Every HTTP attempt — including retries for 429/5xx and for transport errors —
    is metered through `meter` BEFORE it is sent and recorded AFTER, so the request
    count is a real ceiling rather than a description. `sleep_between_retries` is
    exposed so offline tests can exercise the retry path without waiting.
    """
    import aiohttp  # imported lazily so offline tests can import this module

    key = api_key or os.environ.get("OPENROUTER_API_KEY_1") or os.environ.get(
        "OPENROUTER_API_KEY_2")
    if not key:
        raise SystemExit("no OPENROUTER_API_KEY_1/_2 in the environment")
    chosen_model = model or os.environ.get("EXPERIMENT_MODEL", DEFAULT_MODEL)
    calls: list[dict] = ledger if ledger is not None else []

    async def ask(prompt: str, *, label: str = "") -> CallResult:
        body = {
            "model": chosen_model,
            "messages": [{"role": "user", "content": prompt}],
            # Deterministic: the comparison is about review policy, so sampling
            # variance would be noise on the variable under test.
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {key}",
                   "Content-Type": "application/json"}
        if meter is not None:
            meter.logical_calls += 1
        retries: list[dict] = []
        last_error = ""
        requests_this_call = 0

        async def record(status: str, cost: float | None, reported: bool,
                         attempt_no: int) -> None:
            if meter is not None:
                meter.record_attempt(label=label, attempt=attempt_no,
                                     status=status, cost=cost,
                                     usage_reported=reported)

        for attempt_no in range(1, max_attempts + 1):
            # ── Checked BEFORE the request, so the ceiling binds the request that
            #    would cross it, not the one after. ────────────────────────────
            if meter is not None:
                allowed, why = meter.can_send()
                if not allowed:
                    raise RuntimeError(f"meter refused request: {why}")

            requests_this_call += 1
            loop_started = asyncio.get_event_loop().time()
            # Every dispatched request must be metered EXACTLY ONCE, whichever
            # branch handles it. This flag is the enforcement: a branch that sets
            # it before raising guarantees the `except` blocks do not record the
            # same request a second time, and a branch that forgets to record is
            # caught by the safety net below rather than vanishing from the count.
            recorded = False
            # Distinguishes "we got an HTTP response and could not read it" from
            # "we never got a response at all": both are unmetered cost, but they
            # are different operational facts and must not share a status label.
            got_response = False
            try:
                async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=timeout_s)) as session:
                    async with session.post(ENDPOINT, headers=headers,
                                            json=body) as resp:
                        got_response = True
                        if resp.status in (429, 500, 502, 503, 504):
                            last_error = f"http_{resp.status}"
                            retries.append({"attempt": attempt_no,
                                            "reason": last_error,
                                            "label": label})
                            # A 5xx/429 is sent and may be billed; it is recorded
                            # as a request even though no usage came back.
                            await record(f"http_{resp.status}", None, False,
                                         attempt_no)
                            recorded = True
                            if sleep_between_retries:
                                await asyncio.sleep(2 * attempt_no)
                            continue
                        if resp.status >= 400:
                            detail = (await resp.text())[:400]
                            last_error = f"http_{resp.status}: {detail}"
                            await record(f"http_{resp.status}", None, False,
                                         attempt_no)
                            recorded = True
                            raise RuntimeError(last_error)
                        data = await resp.json()
                latency = round(asyncio.get_event_loop().time() - loop_started, 3)

                if "choices" not in data or not data["choices"]:
                    # A 2xx that carries no completion. The request WAS dispatched
                    # and may be billed, but this branch used to raise a plain
                    # RuntimeError that no `record()` call covered, so the retry
                    # loop could send up to three uncounted HTTP requests. Metered
                    # here, once, before raising.
                    #
                    # Usage is often still present on such a response. If it is,
                    # the cost is known and is charged to the meter; if it is not,
                    # the request is recorded as unknown spend, never as free.
                    partial_usage = (data.get("usage") or {}) if isinstance(
                        data, dict) else {}
                    partial_cost, partial_reported = _usage_cost(partial_usage)
                    await record("ok_no_choices", partial_cost, partial_reported,
                                 attempt_no)
                    recorded = True
                    raise RuntimeError(
                        f"no choices in response: {json.dumps(data)[:400]}")

                choice = data["choices"][0]
                text = (choice.get("message") or {}).get("content") or ""
                finish_reason = str(choice.get("finish_reason") or "")
                usage = data.get("usage") or {}
                cost, usage_reported = _usage_cost(usage)

                result = CallResult(
                    text=text,
                    finish_reason=finish_reason,
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    total_tokens=int(usage.get("total_tokens") or 0),
                    cost_usd=cost,
                    max_tokens=max_tokens,
                    model=str(data.get("model") or chosen_model),
                    provider=str(data.get("provider") or ""),
                    latency_s=latency,
                    retries=list(retries),
                    raw_usage=usage,
                    truncated=(finish_reason == "length"),
                    requests_sent=requests_this_call,
                    usage_missing=not usage_reported,
                )
                await record("ok" if usage_reported else "ok_no_usage", cost,
                             usage_reported, attempt_no)
                recorded = True
                calls.append({"label": label, **result.to_dict()})
                if result.truncated:
                    # Recorded AND raised: the attempt is kept, but marked as
                    # incomplete so it can never be scored as a judgment.
                    raise TruncatedResponse(
                        f"finish_reason=length at max_tokens={max_tokens} "
                        f"({result.completion_tokens} completion tokens)")
                return result
            except TruncatedResponse:
                raise
            except RuntimeError as exc:
                # An HTTP >=400 already recorded above; re-raise without a second
                # retry so a deterministic client error is not billed three times.
                if str(exc).startswith("http_"):
                    raise
                if not recorded:
                    # Safety net. If a new branch raises without metering, the
                    # request is still counted here exactly once rather than
                    # disappearing from the ceiling.
                    await record("error_unmetered_branch", None, False, attempt_no)
                    recorded = True
                last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
                retries.append({"attempt": attempt_no, "reason": last_error,
                                "label": label})
                if sleep_between_retries:
                    await asyncio.sleep(2 * attempt_no)
            except Exception as exc:  # noqa: BLE001
                # Transport errors (timeout, connection reset) were SENT and may
                # have been billed. They are metered as requests with unknown cost.
                last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
                retries.append({"attempt": attempt_no, "reason": last_error,
                                "label": label})
                if not recorded:
                    # A response we could not read still carries unknown cost; a
                    # connection that never produced one is a different fact.
                    await record(
                        "unreadable_response" if got_response else "transport_error",
                        None, False, attempt_no)
                    recorded = True
                if sleep_between_retries:
                    await asyncio.sleep(2 * attempt_no)
        raise RuntimeError(f"provider exhausted retries: {last_error}")

    ask.calls = calls  # type: ignore[attr-defined]
    ask.model = chosen_model  # type: ignore[attr-defined]
    ask.max_tokens = max_tokens  # type: ignore[attr-defined]
    ask.meter = meter  # type: ignore[attr-defined]
    return ask
