"""Read-only, stage-separated replay of the retained v2b comparison.

No provider calls. This module does not infer semantic correctness from a
withhold, a citation check, a release, or a provisional case expectation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import difflib
import hashlib
import json
from pathlib import Path
import re

DECISION = re.compile(r"(?im)^\s*decision\s*:\s*[*_` ]*(approve|revise|cannot_assess)\b")


def explicit_decision(raw: str) -> str | None:
    match = DECISION.search(raw or "")
    return match.group(1).lower() if match else None


def replay(attempts: list[dict]) -> list[dict]:
    rows = []
    for index, a in enumerate(attempts):
        first = explicit_decision(a.get("provider_response", ""))
        recheck = explicit_decision(a.get("recheck_response", ""))
        before = a.get("draft_before_trim", "")
        after = a.get("draft_after_trim", "")
        final = a.get("final_markdown", "")
        edits = a.get("revision_log") or []
        changed = bool(final) and final != after
        outcomes = a.get("reference_verdicts") or {}
        legacy = a.get("cell", "").startswith("current")
        rows.append({
            "index": index,
            "case_id": a.get("case_id"), "cell": a.get("cell"),
            "first_raw_model_decision": first,
            "legacy_harness_inferred_decision": a.get("decision") if legacy else None,
            "first_response_available": bool(a.get("provider_response")),
            "first_review_raw": a.get("provider_response") or None,
            "rule_actions_reported": a.get("trim_changes") or [],
            "pre_rule_draft": before, "post_rule_draft": after,
            "rule_changed_draft": before != after,
            "repair_log": edits,
            "repair_logged_replacements": sum(e.get("kind") == "replaced" for e in edits),
            "repair_changed_document": changed,
            "repair_change_unified_diff": "".join(difflib.unified_diff(
                after.splitlines(keepends=True), final.splitlines(keepends=True),
                fromfile="reviewed", tofile="final")) if changed else "",
            "final_document": final or None,
            "re_review_present": bool(a.get("recheck_response")),
            "re_review_raw_model_decision": recheck,
            "re_review_raw": a.get("recheck_response") or None,
            "harness_release_outcome": a.get("outcome"),
            "harness_release_reason": a.get("outcome_reason"),
            "release_policy_override": a.get("outcome") == "approved" and first == "revise" and recheck != "approve",
            "first_review_structure": a.get("structural_problems") or [],
            "reference_counts_as_recorded": outcomes,
            "reference_stage": ("re-review if performed; first review otherwise; overwritten by runner"),
            "reference_validation_level": ("legacy: quote and offset checks under v2 adapter" if legacy
                                           else "prose: source IDs or bare quotes only; linked quotes not validated"),
            "claim_support_assessed": False,
            "semantic_correctness": "not assessed by replay",
            "finish_and_usage": a.get("telemetry") or None,
            "error": a.get("error") or None,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempts", type=Path)
    args = parser.parse_args()
    raw = args.attempts.read_bytes()
    rows = replay(json.loads(raw))
    print(json.dumps({
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "attempts": len(rows),
        "overrides": sum(row["release_policy_override"] for row in rows),
        "outcomes": dict(Counter(row["harness_release_outcome"] for row in rows)),
        "rows": rows,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()