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

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def make_asker(api_key: str | None = None, *,
               model: str | None = None,
               max_tokens: int = DEFAULT_MAX_TOKENS,
               timeout_s: int = DEFAULT_TIMEOUT_S,
               ledger: list[dict] | None = None):
    """Build an async `ask` callable returning a `CallResult`.

    The signature differs from v1 on purpose: v2 returns the whole result rather
    than `(text, cost)`, because cost alone was not enough to audit a run.
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
        retries: list[dict] = []
        last_error = ""
        for attempt_no in range(1, 4):
            loop_started = asyncio.get_event_loop().time()
            try:
                async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=timeout_s)) as session:
                    async with session.post(ENDPOINT, headers=headers,
                                            json=body) as resp:
                        if resp.status in (429, 500, 502, 503, 504):
                            last_error = f"http_{resp.status}"
                            retries.append({"attempt": attempt_no,
                                            "reason": last_error,
                                            "label": label})
                            await asyncio.sleep(2 * attempt_no)
                            continue
                        if resp.status >= 400:
                            detail = (await resp.text())[:400]
                            raise RuntimeError(
                                f"http_{resp.status}: {detail}")
                        data = await resp.json()
                latency = round(asyncio.get_event_loop().time() - loop_started, 3)

                if "choices" not in data or not data["choices"]:
                    raise RuntimeError(
                        f"no choices in response: {json.dumps(data)[:400]}")

                choice = data["choices"][0]
                text = (choice.get("message") or {}).get("content") or ""
                finish_reason = str(choice.get("finish_reason") or "")
                usage = data.get("usage") or {}

                result = CallResult(
                    text=text,
                    finish_reason=finish_reason,
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    total_tokens=int(usage.get("total_tokens") or 0),
                    # OpenRouter's chat-completions `usage` object carries `cost`.
                    # Reading only `total_cost` reported $0.00 for every call and
                    # made the cost ceiling unenforceable. Both keys are accepted,
                    # in order of authority, so a provider change cannot silently
                    # zero the budget again.
                    cost_usd=float(usage.get("cost")
                                   if usage.get("cost") is not None
                                   else (usage.get("total_cost") or 0.0)),
                    max_tokens=max_tokens,
                    model=str(data.get("model") or chosen_model),
                    provider=str(data.get("provider") or ""),
                    latency_s=latency,
                    retries=list(retries),
                    raw_usage=usage,
                    truncated=(finish_reason == "length"),
                )
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
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
                retries.append({"attempt": attempt_no, "reason": last_error,
                                "label": label})
                await asyncio.sleep(2 * attempt_no)
        raise RuntimeError(f"provider exhausted retries: {last_error}")

    ask.calls = calls  # type: ignore[attr-defined]
    ask.model = chosen_model  # type: ignore[attr-defined]
    ask.max_tokens = max_tokens  # type: ignore[attr-defined]
    return ask
