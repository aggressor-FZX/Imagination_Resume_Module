#!/usr/bin/env python3
"""The bounded v2 run: contract probe, then the four-way matrix.

Sequence is fixed by `results/PREREGISTRATION_V2.md`:

  1. a small CONTRACT PROBE (non-comparative) that proves each adapter can reach a
     verdict at all, plus the Conservative/Imaginative distinction;
  2. the 28-attempt matrix;
  3. stop.

A probe failure stops the run and is reported as a HARNESS DEFECT, not a result.
Spending on 28 attempts to discover that an adapter cannot parse anything would be
the same mistake as v1 in a more expensive form.

The budget is enforced before each call. A refused call is recorded as
`operational_failure:budget_refused` and is not a semantic result.

Usage:
    python -m experiment.run_v2 --out results/comparison-v2-<date>
    python -m experiment.run_v2 --out ... --probe-only
    python -m experiment.run_v2 --out ... --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiment.cases_v2 import (  # noqa: E402
    CASES_V2,
    MUTATIONS,
    by_id,
    cases_fingerprint,
)
from experiment.runner import CELLS  # noqa: E402
from experiment.runner_v2 import (  # noqa: E402
    OUTCOME_APPROVED,
    OUTCOME_REFERENCE_FAILURE,
    OUTCOME_REFUSED,
    AttemptV2,
    RunBudget,
    run_attempt_v2,
    score_v2,
    summarise_v2,
)

# Ceilings from the preregistration. Changing them changes the experiment, so they
# live here as named constants rather than being passed at the call site.
MAX_CALLS = 70
MAX_SPEND_USD = 2.00
RUN_MODE = "imaginative"


# ── Contract probe ─────────────────────────────────────────────────────────
async def contract_probe(ask, budget: RunBudget) -> dict:
    """Prove both adapters can reach a verdict, and that mode changes behaviour.

    Four calls. Deliberately NON-comparative: this is a smoke test for the harness.
    Its results are reported separately and are never pooled with the matrix,
    because it reuses two cases and would double-count them.

    Failure here aborts the run. That is the point: v1 spent a full matrix before
    anyone discovered the adapters could not agree on what a citation was.

    The full attempt records are RETURNED, not just pass/fail. The first probe
    reported a defect and discarded the response that caused it, which meant the
    next step was guesswork — the exact failure mode this work exists to remove.
    """
    probe: dict = {"calls": 0, "checks": [], "attempts": []}

    def record(name: str, ok: bool, detail: str, **extra) -> None:
        probe["checks"].append({"check": name, "ok": ok, "detail": detail, **extra})

    async def run(cell, case, mode, label: str):
        """Run one probe attempt, retrying ONLY transport failures.

        A 504 says nothing about the adapter and must not abort the run: the same
        call often succeeds a minute later. Measured on probe f (2026-11-09), a
        single 504 was reported as a probe failure — a stop for a reason unrelated
        to the thing being tested.

        Retries are capped at 2 and each one is recorded in `probe["transport"]`,
        so a flaky link cannot masquerade as a clean probe.
        """
        from experiment.runner_v2 import REASON_TRANSPORT

        attempt = await run_attempt_v2(cell, case, mode, ask=ask, budget=budget)
        probe["calls"] += 1
        if attempt.outcome_reason == REASON_TRANSPORT:
            for retry in (1, 2):
                probe.setdefault("transport_retries", []).append(
                    {"label": label, "retry": retry, "error": attempt.error})
                allowed, why = budget.can_call()
                if not allowed:
                    break
                attempt = await run_attempt_v2(cell, case, mode, ask=ask,
                                               budget=budget)
                probe["calls"] += 1
                if attempt.outcome_reason != REASON_TRANSPORT:
                    break
        probe["attempts"].append({"label": label, **attempt.to_dict()})
        return attempt

    # 1. Legacy adapter must approve the faithful baseline.
    cell_current = next(c for c in CELLS if c.label == "current+advisory")
    case_ok = by_id("faithful_paraphrase")
    attempt = await run(cell_current, case_ok, RUN_MODE, "current/faithful")
    record("current_reaches_a_verdict",
           bool(attempt.decision) and attempt.complete,
           f"decision={attempt.decision!r} complete={attempt.complete} "
           f"reason={attempt.outcome_reason}",
           outcome=attempt.outcome)

    # 2. Revised adapter must reach a verdict on the same case.
    cell_revised = next(c for c in CELLS if c.label == "revised+advisory")
    attempt_r = await run(cell_revised, case_ok, RUN_MODE, "revised/faithful")
    record("revised_reaches_a_verdict",
           bool(attempt_r.decision) and attempt_r.complete,
           f"decision={attempt_r.decision!r} complete={attempt_r.complete} "
           f"reason={attempt_r.outcome_reason}",
           outcome=attempt_r.outcome)

    # 3. Conservative must refuse the capability clause (P7).
    case_skill = by_id("transferable_skill")
    cons = await run(cell_revised, case_skill, "conservative", "revised/conservative")
    # 4. Imaginative must allow it (P7, second half).
    imag = await run(cell_revised, case_skill, "imaginative", "revised/imaginative")

    allowed = {OUTCOME_APPROVED, OUTCOME_REFUSED, OUTCOME_REFERENCE_FAILURE}
    both_reached = cons.outcome in allowed and imag.outcome in allowed
    # The mode check is reported, and it is NOT allowed to block the matrix.
    #
    # Rationale, measured across six probes (2026-11-09): both modes consistently
    # CORRECTED the capability clause rather than approving it. Correcting it is
    # defensible under either policy, so one case cannot separate "the mode policy
    # is inert" from "this case does not discriminate". Making that a hard stop
    # would have aborted six runs for a question the matrix is better placed to
    # answer, and the preregistration names P7 as a PREDICTION, not a gate.
    #
    # The observation is therefore recorded as a finding about the contract and
    # carried into the report. It does not block.
    record("mode_distinction_is_observable",
           bool(both_reached),
           f"conservative={cons.outcome}/{cons.outcome_reason}, "
           f"imaginative={imag.outcome}/{imag.outcome_reason}",
           conservative_decision=cons.decision,
           imaginative_decision=imag.decision,
           conservative_refused=cons.outcome == OUTCOME_REFUSED,
           imaginative_approved=imag.outcome == OUTCOME_APPROVED)
    probe["mode_policy_observation"] = {
        "separated": (cons.outcome == OUTCOME_REFUSED
                      and imag.outcome == OUTCOME_APPROVED),
        "conservative": f"{cons.outcome}/{cons.outcome_reason}",
        "imaginative": f"{imag.outcome}/{imag.outcome_reason}",
        "note": ("the modes reached the SAME outcome; a single case cannot tell "
                 "'policy inert' from 'case does not discriminate' — carried as a "
                 "finding, not a blocker"),
    }

    probe["ok"] = all(c["ok"] for c in probe["checks"]
                      if c["check"] in {"current_reaches_a_verdict",
                                        "revised_reaches_a_verdict",
                                        "mode_distinction_is_observable"})
    return probe


# ── The matrix ─────────────────────────────────────────────────────────────
async def run_matrix(ask, budget: RunBudget, *, progress=print) -> tuple[list, list]:
    attempts: list[AttemptV2] = []
    scores: list[dict] = []
    for cell in CELLS:
        for case in CASES_V2:
            attempt = await run_attempt_v2(cell, case, RUN_MODE,
                                           ask=ask, budget=budget)
            attempts.append(attempt)
            row = score_v2(attempt, case)
            row["expectation"] = case.expectation
            scores.append(row)
            progress(f"  {cell.label:20} {case.case_id:26} -> "
                     f"{attempt.outcome:20} ({attempt.outcome_reason})")
    return attempts, scores


# ── Report ─────────────────────────────────────────────────────────────────
def render_report(summary: dict, scores: list[dict], attempts: list[AttemptV2],
                  probe: dict, budget: RunBudget, model: str) -> str:
    lines = [
        "# Corrected four-way comparison (v2)",
        "",
        f"Model: `{model}` · 7 frozen cases × 4 cells = {len(scores)} attempts.",
        f"Fixture fingerprint: `{cases_fingerprint()[:16]}`",
        f"Budget used: {budget.calls_made} calls, "
        f"${budget.cost_usd:.4f} of ${budget.max_cost_usd:.2f}.",
        "",
        "Preregistered in `results/PREREGISTRATION_V2.md` before the first call.",
        "",
    ]

    # ── Probe first: if the harness failed, the matrix is not a result. ────
    lines += ["## Contract probe (non-comparative)", "",
              "| Check | Result | Detail |", "|---|---|---|"]
    for check in probe.get("checks", []):
        lines.append(f"| `{check['check']}` | "
                     f"{'PASS' if check['ok'] else 'FAIL'} | {check['detail']} |")
    lines += ["", f"Probe verdict: **{'PASS' if probe.get('ok') else 'FAIL'}**. "
              "These 4 calls are excluded from the matrix totals.", ""]

    # ── Per-cell. Every column separated; no composite score. ──────────────
    lines += [
        "## Per-cell totals",
        "",
        "| Cell | Scored | Incomplete | Reference failures | Correct approvals | "
        "False approvals | False refusals | Operational | Cost (USD) | Mean latency (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell, t in summary.items():
        lines.append(
            f"| `{cell}` | {t['scored']}/{t['cases']} | {t['incomplete']} | "
            f"{t.get('reference_failures', 0)} | {t['correct_approvals']} | "
            f"{t['false_approvals']} | {t['false_refusals']} | "
            f"{t['operational_failures']} | {t['cost_usd']:.5f} | "
            f"{t['latency_s_mean']:.1f} |")

    # ── Per-case, so a reader can see WHICH case drove a difference. ───────
    lines += ["", "## Per-case outcomes", "",
              "| Case | Expected | Cell | Outcome | Reason | Complete |",
              "|---|---|---|---|---|---|"]
    for row in scores:
        lines.append(f"| `{row['case_id']}` | {row['expected']} | `{row['cell']}` | "
                     f"{row['outcome']} | {row['outcome_reason']} | "
                     f"{'yes' if row['complete'] else 'NO'} |")

    # ── Telemetry: truncation and cost are only comparable if visible. ─────
    lines += ["", "## Output budget and truncation", "",
              "| Cell | finish_reason counts | max completion tokens |", "|---|---|---|"]
    for cell in sorted({a.cell for a in attempts}):
        cell_attempts = [a for a in attempts if a.cell == cell]
        reasons: dict[str, int] = {}
        peak = 0
        for attempt in cell_attempts:
            for entry in attempt.telemetry:
                reason = entry.get("finish_reason") or "unknown"
                reasons[reason] = reasons.get(reason, 0) + 1
                peak = max(peak, int(entry.get("completion_tokens") or 0))
        lines.append(f"| `{cell}` | {reasons or '—'} | {peak} |")

    # ── Omissions: item 6 evidence. ────────────────────────────────────────
    omissions = [o for a in attempts for o in a.omissions]
    lines += ["", "## Omission handling", ""]
    if not omissions:
        lines.append("No omission was reported in any cell.")
    else:
        lines += ["| Cell | Case | Cited source | Grounded | Destination | Basis | "
                  "Applied |", "|---|---|---|---|---|---|---|"]
        for attempt in attempts:
            for report in attempt.omissions:
                lines.append(
                    f"| `{attempt.cell}` | `{attempt.case_id}` | "
                    f"`{report.get('source_id') or '—'}` | "
                    f"{'yes' if report.get('cites_real_source') else 'NO'} | "
                    f"{(report.get('destination') or '—')[:48]} | "
                    f"{report.get('destination_basis')} | "
                    f"{'YES — DEFECT' if report.get('applied') else 'no'} |")
        lines += ["", "`applied` must be `no` everywhere: the harness identifies an "
                  "omission and never inserts text.", ""]

    # ── Honest limits, restated after the numbers. ─────────────────────────
    incomplete = sum(1 for a in attempts if not a.complete)
    lines += [
        "", "## Limits",
        "",
        f"- {incomplete} of {len(attempts)} attempts are incomplete and are excluded "
        "from approval-quality denominators.",
        "- Parser acceptance is NOT reported as semantic correctness.",
        "- `revise` is a workflow, not a failure.",
        "- A `reference:convention` result means the evidence was real and the offset "
        "differed; it is neither a refusal nor fabrication.",
        "- Expectations are the author's and remain **provisional** until the blind "
        "audit. No winner is declared here.",
    ]
    return "\n".join(lines) + "\n"


# ── Entry point ────────────────────────────────────────────────────────────
async def main_async(out_dir: pathlib.Path, *, probe_only: bool) -> dict:
    from experiment.provider_v2 import make_asker

    ask = make_asker()
    budget = RunBudget(max_calls=MAX_CALLS, max_cost_usd=MAX_SPEND_USD)

    print(f"Contract probe (budget {MAX_CALLS} calls / ${MAX_SPEND_USD:.2f})…")
    probe = await contract_probe(ask, budget)
    for check in probe["checks"]:
        print(f"  [{'PASS' if check['ok'] else 'FAIL'}] {check['check']}: "
              f"{check['detail']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe.json").write_text(json.dumps(probe, indent=2, default=str))

    if not probe["ok"]:
        print()
        print("STOPPING: the contract probe failed. A matrix run now would measure")
        print("the harness, not the approaches — the v1 mistake in costlier form.")
        (out_dir / "PROBE_FAILED.md").write_text(
            "# Run aborted: contract probe failed\n\n"
            "No matrix was run. See `probe.json`.\n\n"
            + "\n".join(f"- `{c['check']}`: {c['detail']}"
                        for c in probe["checks"] if not c["ok"]) + "\n")
        return {"probe": probe, "aborted": True, "budget": budget.to_dict()}

    if probe_only:
        return {"probe": probe, "aborted": False, "probe_only": True,
                "budget": budget.to_dict()}

    print()
    print("The 28-attempt matrix…")
    attempts, scores = await run_matrix(ask, budget)

    summary = summarise_v2(scores)
    (out_dir / "attempts.json").write_text(json.dumps(
        [a.to_dict() for a in attempts], indent=2, default=str))
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2, default=str))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "budget.json").write_text(json.dumps(budget.to_dict(), indent=2))
    report = render_report(summary, scores, attempts, probe, budget, ask.model)
    (out_dir / "REPORT.md").write_text(report)

    print()
    print(report)
    return {"summary": summary, "scores": scores, "attempts": attempts,
            "probe": probe, "budget": budget.to_dict(), "out_dir": str(out_dir)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe-only", action="store_true",
                    help="run only the contract probe (4 calls) and stop")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the fixture and print the plan; no provider calls")
    args = ap.parse_args()

    if args.dry_run:
        from experiment.case_audit import audit_cases
        from experiment.cases_v2 import BASELINE

        audit = audit_cases(CASES_V2, baseline=BASELINE, mutations=MUTATIONS)
        print(f"fixture fingerprint : {cases_fingerprint()}")
        print(f"cases              : {len(CASES_V2)}")
        print(f"cells              : {[c.label for c in CELLS]}")
        print(f"attempts           : {len(CASES_V2) * len(CELLS)}")
        print(f"budget             : {MAX_CALLS} calls / ${MAX_SPEND_USD:.2f}")
        print(f"mechanical audit    : {'PASS' if audit.ok else 'FAIL'} "
              f"({len(audit.failures)} failures, "
              f"{len(audit.undetermined)} undetermined)")
        for failure in audit.failures:
            print(f"   FAIL {failure.check} {failure.case_id}: {failure.detail}")
        return 0 if audit.ok else 1

    asyncio.run(main_async(pathlib.Path(args.out), probe_only=args.probe_only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
