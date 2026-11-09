#!/usr/bin/env python3
"""Rescore a completed run with the current scoring rules. NO provider calls.

Why this exists
---------------
Scoring is code, and code can be wrong. Measured while reporting the v2 matrix
(2026-11-09): an operational failure on a `revise` case was counted in BOTH the
`false_refusals` and `operational_failures` columns. That inflated the failure
count for whichever arm happened to hit more parser problems — the same
category-conflation the whole exercise exists to remove, in the reporting layer
rather than the parsing layer.

The attempts themselves are correct and are left untouched. Only the derived table
is rebuilt, which is why this can run offline and cheaply. If a rescore disagrees
with the published REPORT.md, the rescore wins and the disagreement is the finding.

Usage:
    python -m experiment.rescore results/comparison-v2b-20261109
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiment.cases_v2 import by_id  # noqa: E402
from experiment.runner_v2 import (  # noqa: E402
    AttemptV2,
    score_v2,
    summarise_v2,
)


def load_attempts(path: pathlib.Path) -> list[AttemptV2]:
    """Rebuild AttemptV2 objects from the recorded dicts."""
    raw = json.loads(path.read_text())
    out: list[AttemptV2] = []
    for item in raw:
        known = {k: v for k, v in item.items()
                 if k in AttemptV2.__dataclass_fields__}
        out.append(AttemptV2(**known))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("--write", action="store_true",
                    help="overwrite scores.json/summary.json in the run directory")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run)
    attempts = load_attempts(run_dir / "attempts.json")
    scores = []
    for attempt in attempts:
        case = by_id(attempt.case_id)
        row = score_v2(attempt, case)
        row["expectation"] = case.expectation
        scores.append(row)
    summary = summarise_v2(scores)

    # ── Column integrity: every attempt lands in exactly one column. ───────
    problems = []
    for row in scores:
        columns = sum(bool(row[k]) for k in
                      ("correct_approval", "false_approval", "false_refusal",
                       "operational_failure", "reference_failure"))
        if columns > 1:
            problems.append((row["cell"], row["case_id"], columns,
                             row["outcome_reason"]))

    print(f"{'cell':20} {'scored':>7} {'incompl':>8} {'ref':>4} {'correct':>8} "
          f"{'false_appr':>11} {'false_ref':>10} {'op':>3} {'sem_correct':>12}")
    print("-" * 92)
    for cell, t in summary.items():
        print(f"{cell:20} {t['scored']:>4}/{t['cases']:<2} {t['incomplete']:>8} "
              f"{t.get('reference_failures', 0):>4} {t['correct_approvals']:>8} "
              f"{t['false_approvals']:>11} {t['false_refusals']:>10} "
              f"{t['operational_failures']:>3} {t['semantic_correct']:>12}")

    print()
    if problems:
        print("COLUMN DOUBLE-COUNTING:")
        for cell, case, count, reason in problems:
            print(f"  {cell} / {case}: in {count} columns ({reason})")
    else:
        print("Column integrity: every attempt is in exactly one column.")

    print()
    print("Outcome reasons by cell:")
    for cell in sorted({a.cell for a in attempts}):
        reasons = Counter(a.outcome_reason.split(",")[0]
                          for a in attempts if a.cell == cell)
        print(f"  {cell}:")
        for reason, count in reasons.most_common():
            print(f"    {count}  {reason}")

    if args.write:
        (run_dir / "scores.json").write_text(json.dumps(scores, indent=2,
                                                        default=str))
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print()
        print(f"rewrote {run_dir}/scores.json and summary.json")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
