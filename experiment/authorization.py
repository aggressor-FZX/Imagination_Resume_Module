"""Once-only run authorization, and a hard boundary between render and execute.

Why this module exists
----------------------
A run of the model screen was executed twice. The second time was unintentional: a
report-regeneration command was passed `--authorized`, and the CLI had no concept of
"this authorization was already used". The per-run request ceiling held for each
invocation (20 and 21 requests against a stated 40), while the aggregate was 41.

Two defects made that possible, and both are fixed here:

1. **Report rendering and execution were the same code path.** Regenerating a report
   required the module entry point that also executes. `render_from_artifact()`
   reads a recorded file and writes markdown; it takes no `ask` callable, imports no
   provider path, and cannot dispatch a request.

2. **`--authorized` was stateless.** Every invocation was treated as the first.
   `consume_authorization()` writes a token file BEFORE the run starts and refuses a
   second consumption of the same token. It fails closed: a crashed run still leaves
   the token consumed, so a silent retry is blocked rather than permitted.

The token records cumulative requests across invocations, so 20 + 21 = 41 is visible
as one number instead of two individually-compliant ones.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone


class AuthorizationError(RuntimeError):
    """An authorization was missing, already consumed, or invalid."""


DEFAULT_LEDGER = pathlib.Path("results/.authorizations.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        # A corrupt ledger must not silently authorise anything.
        raise AuthorizationError(
            f"authorization ledger {path} is unreadable or corrupt; refusing to run")


def _save(path: pathlib.Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def consume_authorization(token: str, *, expected_requests: int,
                          ledger: pathlib.Path | None = None) -> dict:
    """Claim `token` for one run. Raises if it has already been consumed.

    The record is written BEFORE the run begins, so a crash mid-run still blocks a
    retry with the same token. That ordering is the point: a guard that only records
    on success can be bypassed by failing.
    """
    if not token or not token.strip():
        raise AuthorizationError("no authorization token supplied")
    path = ledger or DEFAULT_LEDGER
    data = _load(path)

    if token in data:
        prior = data[token]
        raise AuthorizationError(
            f"authorization {token!r} already consumed at {prior.get('consumed_at')} "
            f"({prior.get('status')}). Refusing to run again: a new run needs a new "
            f"authorization token, not a repeated command.")

    data[token] = {
        "consumed_at": _now(),
        "status": "in_progress",
        "expected_requests": expected_requests,
        "requests_sent": 0,
        "cost_usd": 0.0,
    }
    _save(path, data)
    return data[token]


def record_outcome(token: str, *, requests_sent: int, cost_usd: float,
                   status: str = "completed",
                   ledger: pathlib.Path | None = None) -> dict:
    """Close out a consumed authorization with what it actually spent."""
    path = ledger or DEFAULT_LEDGER
    data = _load(path)
    if token not in data:
        raise AuthorizationError(f"authorization {token!r} was never consumed")
    entry = data[token]
    entry.update({"status": status, "finished_at": _now(),
                  "requests_sent": int(requests_sent),
                  "cost_usd": round(float(cost_usd), 9)})
    _save(path, data)
    return entry


def cumulative(ledger: pathlib.Path | None = None) -> dict:
    """Totals across every authorization ever consumed, so overruns are visible."""
    data = _load(ledger or DEFAULT_LEDGER)
    return {
        "authorizations": len(data),
        "requests_sent": sum(v.get("requests_sent", 0) for v in data.values()),
        "cost_usd": round(sum(v.get("cost_usd", 0.0) for v in data.values()), 9),
    }


def render_from_artifact(artifact: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    """Rewrite the markdown report from a RECORDED artifact. Never dispatches.

    Deliberately takes no `ask` callable and imports no provider module, so this
    function cannot make a request even if misused. Regenerating a report must not
    require the execution path — that equivalence is what produced the second run.
    """
    from experiment.model_screen import MODELS, render, summarise

    if not artifact.is_file():
        raise FileNotFoundError(f"no recorded artifact at {artifact}")
    data = json.loads(artifact.read_text())
    rows = data.get("rows")
    if not rows:
        raise ValueError(f"{artifact} contains no rows to render")

    plan = data.get("plan") or {}
    summary = data.get("summary") or summarise(rows, MODELS)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(plan, summary, rows))
    return out
