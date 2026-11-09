#!/usr/bin/env python3
"""Run the 2x2 comparison against the live provider and write the report.

Every attempt is recorded. No retry-until-lucky: each case is asked once per cell,
and a failure stays in the record.

Reports SUBMISSION-level categories separately (correct approvals, false approvals,
false refusals, operational failures, cost, latency). Parser acceptance is kept in
its own column and is never used as evidence of semantic correctness.

Usage:
    OPENROUTER_API_KEY_1=... python -m experiment.run_comparison --out results/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import aiohttp

from experiment.cases import CASES, by_id  # noqa: E402
from experiment.runner import (  # noqa: E402
    CELLS,
    run_attempt,
    score,
    summarise,
)

MODEL = os.environ.get("EXPERIMENT_MODEL", "qwen/qwen3-coder")
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def make_asker(api_key: str):
    """One provider call per prompt, returning (text, cost).

    Adds a small, explicit backoff on 429/5xx and RECORDS every retry, so a retry
    cannot masquerade as a first attempt in the results.
    """
    retries: list[dict] = []

    async def ask(prompt: str) -> tuple[str, float]:
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            # Deterministic: the comparison is about review policy, so sampling
            # variance would be noise on the variable under test.
            "temperature": 0,
            "max_tokens": 1200,
        }
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type": "application/json"}
        last_error = ""
        for attempt in range(1, 4):
            try:
                async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=150)) as session:
                    async with session.post(ENDPOINT, headers=headers,
                                            json=body) as resp:
                        if resp.status in (429, 500, 502, 503, 504):
                            last_error = f"http_{resp.status}"
                            retries.append({"attempt": attempt,
                                            "reason": last_error})
                            await asyncio.sleep(2 * attempt)
                            continue
                        data = await resp.json()
                if "choices" not in data:
                    return (json.dumps(data)[:800], 0.0)
                text = data["choices"][0]["message"]["content"]
                cost = float((data.get("usage") or {}).get("total_cost") or 0.0)
                return (text, cost)
            except Exception as exc:  # noqa: BLE001
                last_error = type(exc).__name__
                retries.append({"attempt": attempt, "reason": last_error})
                await asyncio.sleep(2 * attempt)
        raise RuntimeError(f"provider exhausted retries: {last_error}")

    return ask, retries


async def main_async(out_dir: pathlib.Path, model: str) -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY_1") or os.environ.get(
        "OPENROUTER_API_KEY_2")
    if not api_key:
        raise SystemExit("no OPENROUTER_API_KEY_1/_2 in the environment")
    ask, retries = make_asker(api_key)

    attempts = []
    scores = []
    for cell in CELLS:
        for case in CASES:
            attempt = await run_attempt(cell, case, "imaginative", ask=ask)
            attempts.append(attempt)
            row = score(attempt, case)
            row["expectation"] = case.expectation
            scores.append(row)
            print(f"  {cell.label:22} {case.case_id:26} -> {attempt.outcome:20} "
                  f"({attempt.outcome_reason})")

    summary = summarise(scores)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "attempts.json").write_text(json.dumps(
        [{**a.__dict__, "issues": a.issues} for a in attempts], indent=2,
        default=str))
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "retries.json").write_text(json.dumps(retries, indent=2))
    (out_dir / "REPORT.md").write_text(render_report(summary, scores, model))
    return {"summary": summary, "scores": scores, "attempts": attempts,
            "out_dir": str(out_dir)}


def render_report(summary: dict, scores: list[dict], model: str) -> str:
    """A report that keeps every category separate and states its own limits."""
    lines = [
        "# Decision-contract comparison",
        "",
        f"Model: `{model}` · 6 frozen cases × 4 cells = {len(scores)} attempts.",
        "",
        "## Per-cell totals",
        "",
        "| Cell | Cases | Correct approvals | False approvals | False refusals | "
        "Operational failures | Cost (USD) | Mean latency (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cell, t in summary.items():
        lines.append(
            f"| `{cell}` | {t['cases']} | {t['correct_approvals']} | "
            f"{t['false_approvals']} | {t['false_refusals']} | "
            f"{t['operational_failures']} | {t['cost_usd']:.5f} | "
            f"{t['latency_s_mean']:.1f} |")

    lines += [
        "",
        "## Per-case outcomes",
        "",
        "| Case | Expected | Cell | Outcome | Reason |",
        "|---|---|---|---|---|",
    ]
    for row in scores:
        lines.append(f"| `{row['case_id']}` | {row['expected']} | "
                     f"`{row['cell']}` | {row['outcome']} | {row['outcome_reason']} |")

    correct = sum(r["correct_approval"] for r in scores)
    false_appr = sum(r["false_approval"] for r in scores)
    false_ref = sum(r["false_refusal"] for r in scores)
    op = sum(r["operational_failure"] for r in scores)
    lines += [
        "",
        "## Honest limits of this comparison",
        "",
        f"- {len(scores)} attempts. That is enough to compare cells, not enough to "
        "estimate a production failure rate.",
        f"- {op} operational failure(s) are counted separately and are NOT evidence "
        "about faithfulness in either direction.",
        "- Parser acceptance is not reported as semantic correctness. A well-formed "
        "approval of a defective document is a false approval and is counted as one.",
        f"- Totals across all cells: {correct} correct approvals, {false_appr} false "
        f"approvals, {false_ref} false refusals.",
        "- Expectations were fixed before any run. They are the author's judgement of "
        "what a correct reviewer should conclude, and they need the independent "
        "blind audit before they can be treated as labels.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = asyncio.run(main_async(pathlib.Path(args.out), MODEL))
    print()
    print((pathlib.Path(args.out) / "REPORT.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
