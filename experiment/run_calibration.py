#!/usr/bin/env python3
"""Bounded calibration: seven known cases + ten unfamiliar paired controls.

Scope. This runner is deliberately the smallest thing that can test whether the
revised instructions changed reviewer behaviour:

  * it reviews drafts only — no writer, no trimming, no production path;
  * one call per case for the first judgement;
  * a second call ONLY where the scaffold drew a proposal, and that second call is
    handed the scaffold's exact proposed document, never a different text;
  * a hard call and cost ceiling, enforced BEFORE each call.

Why a separate runner rather than another matrix
------------------------------------------------
The instruction is "do not rerun the full matrix". A 2x2x7 matrix buys cells that
are not needed to answer the current question. The question is narrower:

    with the refined instructions frozen, does the reviewer separate faithful
    documents from defective ones on content it has not seen?

That needs 17 first calls (7 familiar + 10 unfamiliar) plus at most one re-review
per case where a proposal exists. `DEFAULT_MAX_CALLS` is set from the fixture size
so the ceiling is a property of the design, not a guess.

Why freeze the instructions
---------------------------
Item 6 requires the instructions frozen before the calls. `freeze()` records the
exact prompt text and its sha256 per mode in the output, and every call is built
from that frozen text. If the text were edited mid-run, the recorded hash would no
longer match the prompt that produced the answers, and the run would be
uninterpretable — so the hash is written before the first call and re-checked
after the last.

Why the re-review is guarded
----------------------------
Item 7: "a retained response for a different document cannot validate it." The
document handed to the re-review call is therefore passed in explicitly and its
identity is asserted against the scaffold's proposal at the call site. A mismatch
raises rather than being recorded as a re-review.

Usage:
    python -m experiment.run_calibration --out results/calibration-v1
    python -m experiment.run_calibration --out ... --dry-run   # no provider calls
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import decision_contract as dc  # noqa: E402
from experiment.binding_candidate import BindingResult, safe_repair  # noqa: E402
from experiment.calibration_cases import (  # noqa: E402
    CALIBRATION_CASES, CALIBRATION_MUTATIONS, check as check_calibration,
    cases_fingerprint as calibration_fingerprint,
)
from experiment.cases_v2 import CASES_V2, SOURCES, by_id as by_v2_id  # noqa: E402
from experiment.runner_v2 import RunBudget  # noqa: E402

# The retained matrix ran every cell under ONE mode: `imaginative`
# (verified from `results/comparison-v2b-20261109/attempts.json`, 28/28 cells).
# Calibration must use the same mode or the familiar cases stop being comparable
# for a reason unrelated to the instructions: `transferable_skill` is approvable
# only under Imaginative, so reviewing it under Conservative would turn a correct
# approval into a correct refusal and the preregistered label into a contradiction.
MODE = "imaginative"

# The seven familiar cases plus the ten unfamiliar controls. Exactly the fixture,
# nothing held back, so the denominator is the whole design.
FAMILIAR = CASES_V2
UNFAMILIAR = CALIBRATION_CASES

# One first call each, plus one re-review each at most. Computed, not guessed.
DEFAULT_MAX_CALLS = len(FAMILIAR) + len(UNFAMILIAR) * 2
DEFAULT_MAX_SPEND_USD = 1.00


def build_prompt(instructions: str, sources: dict[str, str], draft: str) -> str:
    """The review prompt. Same shape as the matrix runner, so results are comparable."""
    sources_block = "\n\n".join(f"[{sid}]\n{text}" for sid, text in sources.items())
    return (f"{instructions}\n\nSOURCE DOCUMENTS:\n{sources_block}"
            f"\n\nDRAFT TO REVIEW:\n{draft}")


def document_from_prompt(prompt: str) -> str:
    marker = "DRAFT TO REVIEW:\n"
    return prompt[prompt.index(marker) + len(marker):]


def freeze() -> dict:
    """Freeze the instructions and record their identity BEFORE any call."""
    text = dc.contract_for(MODE)
    return {
        "mode": MODE,
        "instructions": text,
        "instructions_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "contract_fingerprint": dc.contract_fingerprint(MODE),
        "calibration_fixture_fingerprint": calibration_fingerprint(),
        "familiar_cases": [c.case_id for c in FAMILIAR],
        "unfamiliar_cases": [c.case_id for c in UNFAMILIAR],
        "max_calls": DEFAULT_MAX_CALLS,
        "max_spend_usd": DEFAULT_MAX_SPEND_USD,
    }


async def review_one(ask, budget: RunBudget, frozen_instructions: str,
                     case, *, familiar: bool) -> dict:
    """Review one draft and, only where a proposal exists, re-review it exactly."""
    row: dict = {
        "case_id": case.case_id,
        "group": "familiar" if familiar else "unfamiliar",
        "mode": MODE,
        "expected": case.expectation,
        "mutation": (CALIBRATION_MUTATIONS.get(case.case_id) if not familiar
                     else None),
        "instructions_sha256": hashlib.sha256(
            frozen_instructions.encode("utf-8")).hexdigest(),
        "draft": case.draft_before_trim,
        "sources": dict(sorted(case.sources.items())),
        "first_decision": None,
        "first_response": "",
        "first_finish_reason": "",
        "first_cost_usd": 0.0,
        "first_latency_s": 0.0,
        "first_error": "",
        "repair_status": "not_attempted",
        "proposed_document": case.draft_before_trim,
        "proposal_drawn": False,
        "re_review_decision": None,
        "re_review_response": "",
        "re_review_document_sha256": "",
        "re_review_document_is_proposal": None,
        "re_review_error": "",
        "release": False,
        "release_reason": "not_reviewed",
    }

    allowed, why = budget.can_call()
    if not allowed:
        row["first_error"] = f"budget:{why}"
        row["release_reason"] = "budget_exhausted"
        return row

    try:
        call = await ask(build_prompt(frozen_instructions, case.sources,
                                      case.draft_before_trim),
                         label=f"{case.case_id}:first")
    except Exception as exc:  # noqa: BLE001
        row["first_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        row["release_reason"] = "operational:" + type(exc).__name__
        return row

    budget.record(call.cost_usd)
    row |= {"first_response": call.text,
            "first_finish_reason": call.finish_reason,
            "first_cost_usd": call.cost_usd,
            "first_latency_s": call.latency_s}

    first = dc.decide_from_text(call.text, case.draft_before_trim, case.sources)
    row["first_decision"] = first.decision or None
    row["first_structural_problems"] = list(first.structural_problems)

    if not first.ok:
        row["release_reason"] = "initial_review_unverifiable"
        return row
    if first.decision == dc.DECISION_APPROVE:
        row["release"], row["release_reason"] = True, "initial_llm_approve"
        return row
    if first.decision == dc.DECISION_CANNOT_ASSESS:
        row["release_reason"] = "llm_cannot_assess"
        return row

    proposed, row["repair_status"] = safe_repair(
        case.draft_before_trim, case.sources, first.issues)
    row["proposed_document"] = proposed
    row["proposal_drawn"] = proposed != case.draft_before_trim
    if not row["proposal_drawn"]:
        row["release_reason"] = "revision_unresolved"
        return row

    # ── The guard item 7 asks for ──────────────────────────────────────────────
    # The re-review prompt is built from `proposed` and nothing else, and the
    # document recovered from that prompt is asserted to BE `proposed`. If a later
    # change ever passed a different text here, this raises instead of silently
    # attributing an approval to the wrong document.
    recheck_prompt = build_prompt(frozen_instructions, case.sources, proposed)
    assert document_from_prompt(recheck_prompt) == proposed, \
        "re-review document does not match the proposal"
    row["re_review_document_sha256"] = hashlib.sha256(
        proposed.encode("utf-8")).hexdigest()
    row["re_review_document_is_proposal"] = True

    allowed, why = budget.can_call()
    if not allowed:
        row["re_review_error"] = f"budget:{why}"
        row["release_reason"] = "budget_exhausted_before_re_review"
        return row

    try:
        call = await ask(recheck_prompt, label=f"{case.case_id}:re_review")
    except Exception as exc:  # noqa: BLE001
        row["re_review_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        row["release_reason"] = "re_review_error:" + type(exc).__name__
        return row

    budget.record(call.cost_usd)
    row["re_review_response"] = call.text
    second = dc.decide_from_text(call.text, proposed, case.sources)
    row["re_review_decision"] = second.decision or None
    row["re_review_structural_problems"] = list(second.structural_problems)

    if second.ok and second.decision == dc.DECISION_APPROVE:
        row["release"], row["release_reason"] = True, "corrected_document_llm_approved"
    else:
        row["release_reason"] = "re_review_not_approved"
    return row


def summarise(rows: list[dict]) -> dict:
    """Report judgement, applicability, approval and release as SEPARATE counts."""
    def group(name: str) -> list[dict]:
        return [r for r in rows if r["group"] == name]

    out: dict = {}
    for name in ("familiar", "unfamiliar"):
        selected = group(name)
        decided = [r for r in selected if r["first_decision"]]
        agree = [r for r in decided
                 if (r["first_decision"] == "approve") == (r["expected"] == "approve")]
        out[name] = {
            "cases": len(selected),
            "operational_failures": sum(1 for r in selected if r["first_error"]),
            "decisions": {d: sum(1 for r in decided if r["first_decision"] == d)
                          for d in ("approve", "revise", "cannot_assess")},
            "agrees_with_expectation": len(agree),
            "disagrees_with_expectation": len(decided) - len(agree),
            "no_decision": len(selected) - len(decided),
            "proposals_drawn": sum(1 for r in selected if r["proposal_drawn"]),
            "scaffold_refusals": {
                s: sum(1 for r in selected if r["repair_status"] == s)
                for s in sorted({r["repair_status"] for r in selected})
                if s != "proposed_exact_owned_source_bullet"},
            "corrected_document_approved": sum(
                1 for r in selected if r["re_review_decision"] == "approve"),
            "re_reviews_valid": sum(
                1 for r in selected if r["re_review_document_is_proposal"]),
            "released": sum(1 for r in selected if r["release"]),
            "release_reasons": {
                s: sum(1 for r in selected if r["release_reason"] == s)
                for s in sorted({r["release_reason"] for r in selected})},
            "cost_usd": round(sum(r["first_cost_usd"] for r in selected), 6),
        }
    out["pair_separation"] = _pair_separation(rows)
    out["total_cost_usd"] = round(sum(r["first_cost_usd"] for r in rows), 6)
    return out


def _pair_separation(rows: list[dict]) -> dict:
    """Did the reviewer separate each unfamiliar faithful/destructive pair?

    This is not a composite score. It is a per-pair statement of which member got
    which decision, so a reader can see the split instead of a tally.
    """
    index = {r["case_id"]: r for r in rows if r["group"] == "unfamiliar"}
    pairs: dict[str, dict] = {}
    for case_id, mutation in CALIBRATION_MUTATIONS.items():
        if case_id not in index:
            continue
        baseline = mutation["baseline"]
        key = f"{baseline} > {case_id}"
        pairs[key] = {
            "baseline_decision": (index.get(baseline) or {}).get("first_decision"),
            "variant_decision": index[case_id]["first_decision"],
            "expected_baseline": "approve",
            "expected_variant": "revise",
            "separated": ((index.get(baseline) or {}).get("first_decision") == "approve"
                          and index[case_id]["first_decision"] in
                          ("revise", "cannot_assess")),
        }
    return pairs


def render(frozen: dict, summary: dict, rows: list[dict]) -> str:
    lines = [
        "# Bounded calibration — refined instructions, unfamiliar paired controls",
        "",
        "**Instructions were frozen before these calls.**",
        "",
        f"- mode: `{frozen['mode']}`",
        f"- instructions sha256: `{frozen['instructions_sha256']}`",
        f"- contract fingerprint: `{frozen['contract_fingerprint']}`",
        f"- fixture fingerprint: `{frozen['calibration_fixture_fingerprint']}`",
        f"- call ceiling: {frozen['max_calls']} · spend ceiling: "
        f"${frozen['max_spend_usd']:.2f}",
        "",
        "The four axes are reported separately and are never summed.",
        "",
    ]
    for name in ("familiar", "unfamiliar"):
        s = summary[name]
        lines += [f"## {name.capitalize()} cases ({s['cases']})", "",
                  f"- **initial judgement** — decisions `{s['decisions']}`, agrees "
                  f"{s['agrees_with_expectation']}, disagrees "
                  f"{s['disagrees_with_expectation']}, no decision "
                  f"{s['no_decision']}, operational failures "
                  f"{s['operational_failures']}",
                  f"- **edit applicability** — proposals drawn "
                  f"{s['proposals_drawn']}; scaffold refusals "
                  f"`{s['scaffold_refusals']}`",
                  f"- **corrected-document approval** — approvals of the exact "
                  f"proposal: {s['corrected_document_approved']} of "
                  f"{s['re_reviews_valid']} valid re-reviews",
                  f"- **release** — {s['released']} released; reasons "
                  f"`{s['release_reasons']}`",
                  f"- cost: ${s['cost_usd']:.6f}", ""]

    lines += ["## Per-pair separation (unfamiliar controls only)", "",
              "| pair (baseline > variant) | baseline decision | variant decision | "
              "separated |", "|---|---|---|---|"]
    for key, value in summary["pair_separation"].items():
        lines.append(f"| `{key}` | `{value['baseline_decision']}` | "
                     f"`{value['variant_decision']}` | "
                     f"{'yes' if value['separated'] else '**NO**'} |")

    lines += ["", "## Source-linked judgments", "",
              "Every row below cites the exact passage the reviewer named, so the "
              "judgement can be checked against the source rather than taken on "
              "trust. `—` means the reviewer named no passage.", "",
              "| case | expected | decision | first named passage | why |",
              "|---|---|---|---|---|"]
    for row in rows:
        passage, why = _first_objection(row)
        lines.append(f"| `{row['case_id']}` | `{row['expected']}` | "
                     f"`{row['first_decision'] or row['first_error'] or '—'}` | "
                     f"{_cell(passage)} | {_cell(why)} |")
    lines += ["", f"Total cost: ${summary['total_cost_usd']:.6f}", ""]
    return "\n".join(lines) + "\n"


def _first_objection(row: dict) -> tuple[str, str]:
    """The first objection the reviewer actually wrote, as cited text."""
    raw = row.get("first_response") or ""
    if not raw:
        return "—", row.get("first_error") or "no response"
    decision = dc.decide_from_text(raw, row["draft"], row["sources"])
    if not decision.issues:
        return "—", "no issue reported"
    issue = decision.issues[0]
    return (issue.passage or "—"), (issue.why or "—")


def _cell(text: str) -> str:
    return (text or "—").replace("|", "\\|").replace("\n", " ")[:220]


async def main_async(out_dir: pathlib.Path, *, dry_run: bool) -> dict:
    frozen = freeze()
    out_dir.mkdir(parents=True, exist_ok=True)

    verdict = check_calibration()
    if verdict["problems"]:
        raise SystemExit("calibration fixture is invalid: "
                         + "; ".join(verdict["problems"]))

    # The instructions are frozen and written BEFORE any call, so a later edit
    # cannot be mistaken for the text that produced these answers.
    (out_dir / "frozen-instructions.json").write_text(
        json.dumps(frozen, indent=2) + "\n")

    if dry_run:
        return {"dry_run": True, "frozen": frozen,
                "would_review": [c.case_id for c in FAMILIAR + UNFAMILIAR],
                "max_calls": frozen["max_calls"]}

    from experiment.provider_v2 import make_asker

    budget = RunBudget(max_calls=frozen["max_calls"],
                       max_cost_usd=frozen["max_spend_usd"])
    ledger: list[dict] = []
    ask = make_asker(ledger=ledger)

    rows: list[dict] = []
    for case in FAMILIAR:
        rows.append(await review_one(ask, budget, frozen["instructions"], case,
                                     familiar=True))
        print(f"  familiar {case.case_id:26s} -> {rows[-1]['first_decision'] or rows[-1]['first_error']}")
    for case in UNFAMILIAR:
        rows.append(await review_one(ask, budget, frozen["instructions"], case,
                                     familiar=False))
        print(f"  unfamiliar {case.case_id:24s} -> {rows[-1]['first_decision'] or rows[-1]['first_error']}")

    # Re-check that the text used for every row is the frozen text.
    used = {r["instructions_sha256"] for r in rows}
    if used != {frozen["instructions_sha256"]}:
        raise SystemExit("instructions changed during the run: "
                         f"{sorted(used)} != {frozen['instructions_sha256']}")

    summary = summarise(rows)
    payload = {"frozen": frozen, "budget": budget.to_dict(),
               "summary": summary, "rows": rows, "ledger": ledger}
    (out_dir / "calibration.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out_dir / "CALIBRATION.md").write_text(render(frozen, summary, rows))
    return {"budget": budget.to_dict(), "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/calibration-v1")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and validate the frozen plan without any provider call")
    args = ap.parse_args()
    result = asyncio.run(main_async(pathlib.Path(args.out), dry_run=args.dry_run))
    print(json.dumps(result, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
