#!/usr/bin/env python3
"""Report the ACTUAL withhold mix from live ``correction.end`` events.

Why this exists
---------------
A remediation that says "4 of 9 withholds removed" is a claim about what the code
now permits, not about what production does. This script reads real events so the
two can be compared, and it deliberately refuses to estimate: every number it
prints is derived from an event it actually saw, and every gap is reported as a
gap.

It also does NOT infer that a withdraw reason disappeared because the code changed.
Absence of a reason is reported as absence-of-evidence, because a reason may simply
not have been exercised in the window.

Usage
-----
    python scripts/measure_withhold_mix.py                 # last 24h
    python scripts/measure_withhold_mix.py --hours 72
    python scripts/measure_withhold_mix.py --json

Requires the Render CLI to be authenticated (``render whoami``).
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import shutil
import subprocess
import sys

IMAGINATOR_SERVICE = "srv-d3nf73ur433s73bh9j00"

# Reasons that describe a failure of THIS SERVICE rather than of the candidate's
# document. Mirrors app._INFRASTRUCTURE_REASONS; kept as a separate literal so a
# drift in either is visible in a diff rather than silently agreeing.
INFRASTRUCTURE = {
    "provider_rate_limited", "provider_unavailable", "provider_rejected_request",
    "provider_error", "privacy_redaction_failed", "privacy_review_required",
    "timeout", "pipeline_budget_exhausted",
    "timeout_source_inventory", "timeout_initial_review", "timeout_repair",
    "timeout_post_patch_review", "timeout_restore", "timeout_post_restore_review",
    "provider_call_timeout_source_inventory", "provider_call_timeout_initial_review",
    "provider_call_timeout_repair", "provider_call_timeout_post_patch_review",
    "provider_call_timeout_restore", "provider_call_timeout_post_restore_review",
    "malformed_output", "malformed_source_inventory", "no_op_repair",
    "incomplete_review", "correction_failed", "analysis_failed",
    "receipt_signing_unavailable", "render_contract_ambiguous",
    "section_contract_unavailable", "source_documents_mismatch",
}
ACTIONABLE = {"receipt_unavailable"}
CONTENT = {
    "source_facts_missing", "source_completeness_uncertain", "source_section_missing",
    "attribution_conflict", "unresolved_claims", "no_supported_claims",
    "no_required_source_claims", "invalid_input",
}


def classify(reason: str) -> str:
    if reason in CONTENT:
        return "content"
    if reason in ACTIONABLE:
        return "actionable"
    return "infrastructure"


def fetch_logs(hours: int) -> list[dict]:
    """All ``correction.end`` events the CLI will return for the window."""
    if shutil.which("render") is None:
        sys.exit("render CLI not found on PATH")
    start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours))
    cmd = [
        "render", "logs", "--resources", IMAGINATOR_SERVICE,
        "--text", "correction.end", "--limit", "300",
        "--start", start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "--output", "text",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        sys.exit("render logs timed out")
    if out.returncode != 0:
        sys.exit(f"render logs failed: {out.stderr.strip()[:400]}")
    events = []
    for line in out.stdout.splitlines():
        brace = line.find("{")
        if brace < 0:
            continue
        try:
            payload = json.loads(line[brace:])
        except json.JSONDecodeError:
            continue
        if payload.get("event") != "correction.end":
            continue
        stamp = re.match(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})", line)
        events.append({
            "at": stamp.group(1).replace("T", " ") if stamp else "",
            "status": payload.get("status"),
            "reason": payload.get("reason"),
            "reviews": payload.get("reviews"),
            "repairs": payload.get("repairs"),
        })
    # De-duplicate: the log API can return the same event across windows.
    seen, unique = set(), []
    for event in events:
        key = (event["at"], event["status"], event["reason"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
    return sorted(unique, key=lambda e: e["at"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    events = fetch_logs(args.hours)
    approved = [e for e in events if e["status"] == "approved"]
    withheld = [e for e in events if e["status"] == "withheld"]

    by_reason = collections.Counter(e["reason"] for e in withheld)
    by_class = collections.Counter(classify(e["reason"] or "") for e in withheld)

    report = {
        "window_hours": args.hours,
        "events": len(events),
        "first": events[0]["at"] if events else None,
        "last": events[-1]["at"] if events else None,
        "approved": len(approved),
        "withheld": len(withheld),
        "withhold_rate": (round(len(withheld) / len(events), 4) if events else None),
        "by_reason": dict(by_reason.most_common()),
        "by_class": dict(by_class.most_common()),
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    if not events:
        print("No correction.end events in the window.")
        print("This is absence of evidence, NOT evidence the failures are gone.")
        return 0

    print(f"window           : last {args.hours}h")
    print(f"first / last     : {report['first']} -> {report['last']}")
    print(f"correction runs  : {report['events']}")
    print(f"  approved       : {report['approved']}")
    print(f"  withheld       : {report['withheld']}  ({report['withhold_rate']:.0%})")
    print()
    print("withheld by reason:")
    for reason, count in by_reason.most_common():
        print(f"  {reason:42} {count:>3}  [{classify(reason or '')}]")
    print()
    print("withheld by class (whose failure it was):")
    for cls, count in by_class.most_common():
        print(f"  {cls:42} {count:>3}")

    if report["events"] < 20:
        print()
        print(f"CAVEAT: only {report['events']} events. A reason's absence here is "
              "not proof it cannot occur.")

    # Explicitly refuse the tempting inference.
    empty = [r for r in sorted(INFRASTRUCTURE | ACTIONABLE | CONTENT)
             if r not in by_reason]
    if empty:
        print()
        print(f"{len(empty)} known reason(s) did not occur in this window; that is "
              "absence of evidence, not a fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
