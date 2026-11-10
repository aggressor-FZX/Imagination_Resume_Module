#!/usr/bin/env python3
"""Matched cross-model comparison: is `qwen/qwen3-coder` the bottleneck?

Why this and not another prompt iteration
-----------------------------------------
The instructions now say, explicitly, that source-stated overlapping roles are a
fact and that ordinary synonymy is permitted. The model still objects to both. Two
explanations remain and they call for opposite responses:

    (a) the instructions are still not doing the work, so iterate on wording; or
    (b) this MODEL's behaviour is the limit, so nothing about the wording is the
        problem.

A second reviewer on the SAME frozen inputs separates those. Nothing changes except
the model, so any difference is attributable to the model.

Design
------
Six frozen drafts, chosen to cover the four behaviours the brief names:

    clean paraphrase            faithful_paraphrase, cal_ops_faithful
    source-backed concurrency   cal_lab_faithful  (Tidewater 03/2021-, Cascade 09/2023-)
    unsupported method          unsupported_method, cal_ops_unsupported
    omission / wrong record     omitted_record, wrong_record_attribution

Every draft and every source document is the EXACT text already frozen for the
calibration run — no re-authoring, no paraphrase drift. The prompt is built by
`run_calibration.build_prompt` from the same frozen instruction text, in the same
mode. Only `model` differs.

What is compared
----------------
Per case, per model: the decision, whether the envelope was usable, and the
objections themselves — quoted, not scored. The question the brief asks is whether
the *objections* are defensible, so the output is a side-by-side of the actual
objections plus a small number of named checks:

    concurrency_invented   did it claim the source-backed overlap is impossible?
    synonymy_objected      did it object to "through"/"by" style paraphrase?
    defect_named           did it name the planted defect?
    approved_clean         did it approve a clean draft?

Usage:
    python -m experiment.cross_model --model openai/gpt-4o-mini --dry-run
    python -m experiment.cross_model --model openai/gpt-4o-mini --out results/cross-model-v1
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import decision_contract as dc  # noqa: E402
from experiment.calibration_cases import by_id as cal_by_id  # noqa: E402
from experiment.cases_v2 import by_id as fam_by_id  # noqa: E402
from experiment.detection_rubric import TRUTH, grade_case  # noqa: E402
from experiment.provider_v2 import RequestMeter, make_asker  # noqa: E402
from experiment.run_calibration import (  # noqa: E402
    MODE, build_prompt, document_from_prompt,
)

# The frozen cast. `probe` names the behaviour this draft is in the comparison for.
CAST = (
    ("faithful_paraphrase", "familiar", "clean_paraphrase"),
    ("cal_ops_faithful", "unfamiliar", "clean_paraphrase"),
    ("cal_lab_faithful", "unfamiliar", "source_backed_concurrency"),
    ("unsupported_method", "familiar", "unsupported_method"),
    ("cal_ops_unsupported", "unfamiliar", "unsupported_method"),
    ("omitted_record", "familiar", "omission_or_wrong_record"),
    ("wrong_record_attribution", "familiar", "omission_or_wrong_record"),
)

# One first call per case. No re-review: this comparison is about the JUDGEMENT,
# and a repair path would add a second variable (scaffold applicability) that is
# not what is being compared.
ATTEMPTS_PER_CALL = 2
MAX_REQUESTS = len(CAST) * ATTEMPTS_PER_CALL
MAX_SPEND_USD = 0.75


def case(case_id: str):
    try:
        return cal_by_id(case_id)
    except KeyError:
        return fam_by_id(case_id)


def frozen_instructions() -> str:
    """The same frozen prompt the calibration used. Not re-authored here."""
    return dc.contract_for(MODE)


# ── Named checks, applied to the objections themselves ───────────────────────

# Words that treat a source-stated overlap as a defect.
#
# On a CLEAN draft, any reference to overlap/concurrency at all is an invented
# objection, because the sources themselves state the roles as concurrent. The
# observed phrasings are wider than "impossible":
#
#   "making concurrent employment impossible"
#   "an overlap or inconsistency in timeline that is not clarified or supported"
#   "the concurrent 'Present' roles without clarification, which violates …"
#   "misrepresents the nature of the roles as sequential rather than concurrent"
#
# All four are the same error: importing an exclusivity rule the evidence does not
# establish. The check therefore keys on the TOPIC on a clean draft rather than
# trying to enumerate every way of framing it as a defect.
EXCLUSIVITY_WORDS = (
    "impossible", "cannot simultaneously", "cannot hold", "contradiction",
    "overlap", "concurrent", "simultaneous", "two full-time", "sequential rather",
)
# The paraphrase pairs the instructions explicitly permit.
SYNONYMY_PHRASES = (
    "causal mechanism", "through", "changes the meaning", "imply",
    "specificity", "synonymous with", "alters the meaning",
)


def _objection_text(decision: dc.Decision) -> str:
    return " ".join(
        f"{i.passage or ''} {i.why or ''} {i.correction or ''}"
        for i in decision.issues).lower()


def named_checks(case_id: str, friendly: str, decision: dc.Decision) -> dict:
    text = _objection_text(decision)
    clean = friendly in ("clean_paraphrase", "source_backed_concurrency")

    # Only meaningful on a CLEAN draft: there, the sources state the roles as
    # concurrent, so any overlap objection is an invented exclusivity rule. On a
    # defective draft the same words may be a legitimate finding, so the check is
    # deliberately not applied there rather than being applied and then explained.
    concurrency = bool(clean and any(w in text for w in EXCLUSIVITY_WORDS))
    synonymy = bool(clean and any(p in text for p in SYNONYMY_PHRASES))

    defect = None
    if case_id in TRUTH:
        graded = grade_case(case_id, decision)
        defect = bool(graded["all_complete"])

    return {
        "concurrency_claimed_impossible": concurrency,
        "synonymy_objected": synonymy,
        "defect_fully_detected": defect,
        "approved_clean_draft": (decision.decision == "approve") if clean else None,
        "clean_refused": (decision.decision != "approve") if clean else None,
    }


async def review(ask, meter: RequestMeter, label: str, case_id: str,
                 instructions: str) -> dict:
    c = case(case_id)
    row = {"case_id": case_id, "model_label": label,
           "instructions_sha256": hashlib.sha256(
               instructions.encode()).hexdigest(),
           "decision": None, "valid": None, "structural": [],
           "response": "", "cost_usd": 0.0, "error": "",
           "objections": []}

    allowed, why = meter.can_send()
    if not allowed:
        row["error"] = f"meter:{why}"
        return row

    try:
        call = await ask(build_prompt(instructions, c.sources, c.draft_before_trim),
                         label=f"{label}:{case_id}")
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return row

    decision = dc.decide_from_text(call.text, c.draft_before_trim, c.sources)
    row |= {
        "response": call.text,
        "cost_usd": call.cost_usd,
        "decision": decision.decision or None,
        "valid": bool(decision.decision) and not decision.structural_problems,
        "structural": list(decision.structural_problems),
        "objections": [{"problem": i.problem, "passage": i.passage,
                        "why": i.why, "correction": i.correction,
                        "passage_resolves": i.passage_resolves}
                       for i in decision.issues],
    }
    return row


def compare(rows: list[dict]) -> dict:
    """Side-by-side per case. Two models over the same frozen draft."""
    by_case: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], {})[row["model_label"]] = row

    out: dict = {"per_case": {}}
    labels = sorted({r["model_label"] for r in rows})
    for case_id, cells in by_case.items():
        entry = {"models": {}}
        for label in labels:
            row = cells.get(label)
            if not row:
                continue
            friendly = next((f for cid, _g, f in CAST if cid == case_id), "?")
            entry["models"][label] = {
                "decision": row["decision"],
                "valid": row["valid"],
                "objections": len(row["objections"]),
                "checks": named_checks(case_id, friendly, _as_decision(row)),
                "first_objection": _first(row),
            }
        entry["probe"] = next((f for cid, _g, f in CAST if cid == case_id), "?")
        out["per_case"][case_id] = entry

    for label in labels:
        mine = [r for r in rows if r["model_label"] == label]
        out[label] = {
            "cases": len(mine),
            "usable_answers": sum(1 for r in mine if r["valid"]),
            "errors": sum(1 for r in mine if r["error"]),
            "concurrency_claims": sum(
                1 for r in mine if named_checks(
                    r["case_id"],
                    next((f for cid, _g, f in CAST if cid == r["case_id"]), "?"),
                    _as_decision(r))["concurrency_claimed_impossible"]),
            "synonymy_objections": sum(
                1 for r in mine if named_checks(
                    r["case_id"],
                    next((f for cid, _g, f in CAST if cid == r["case_id"]), "?"),
                    _as_decision(r))["synonymy_objected"]),
            "clean_approved": sum(
                1 for r in mine if named_checks(
                    r["case_id"],
                    next((f for cid, _g, f in CAST if cid == r["case_id"]), "?"),
                    _as_decision(r))["approved_clean_draft"] is True),
            "defect_fully_detected": sum(
                1 for r in mine if named_checks(
                    r["case_id"],
                    next((f for cid, _g, f in CAST if cid == r["case_id"]), "?"),
                    _as_decision(r))["defect_fully_detected"] is True),
            "cost_usd": round(sum(r["cost_usd"] for r in mine), 6),
        }
    return out


def _as_decision(row: dict) -> dc.Decision:
    """Rebuild a Decision from the recorded objections, for the named checks."""
    issues = [dc.Issue(passage=o["passage"], problem=o["problem"], why=o["why"],
                       correction=o["correction"])
              for o in row["objections"]]
    return dc.Decision(decision=row["decision"] or "", issues=issues)


def _first(row: dict) -> str:
    if row["error"]:
        return f"error: {row['error']}"
    if not row["objections"]:
        return "(no issues raised)"
    first = row["objections"][0]
    return " ".join(f"[{first['problem']}] {first['why'] or ''}".split())[:260]


def render(payload: dict) -> str:
    labels = payload["models"]
    lines = ["# Matched cross-model comparison", "",
             "Same frozen drafts, same frozen sources, same mode and policy. "
             "**Only the model differs.**", "",
             f"- mode: `{payload['mode']}`",
             f"- instructions sha256: `{payload['instructions_sha256']}`",
             f"- requests: {payload['meter']['requests_sent']} / "
             f"{payload['meter']['max_requests']}",
             f"- cost: ${payload['meter']['cost_usd']:.6f}"
             + ("" if payload["meter"]["cost_is_known"]
                else " (UNKNOWN: some responses omitted usage)"), "",
             "## Per-model summary", "",
             "| model | usable | concurrency claimed impossible | synonymy "
             "objected | clean approved | defect fully detected | cost |",
             "|---|---|---|---|---|---|---|"]
    for label in labels:
        s = payload[label]
        lines.append(f"| `{label}` | {s['usable_answers']}/{s['cases']} | "
                     f"**{s['concurrency_claims']}** | "
                     f"**{s['synonymy_objections']}** | "
                     f"{s['clean_approved']} | {s['defect_fully_detected']} | "
                     f"${s['cost_usd']:.6f} |")

    lines += ["", "## The objections themselves", "",
              "Quoted, not scored. This is the material an independent assessor "
              "needs.", ""]
    for case_id, entry in payload["per_case"].items():
        lines += [f"### `{case_id}` — probe: `{entry['probe']}`", ""]
        for label, cell in entry["models"].items():
            checks = cell["checks"]
            flags = []
            if checks["concurrency_claimed_impossible"]:
                flags.append("**claims concurrency impossible**")
            if checks["synonymy_objected"]:
                flags.append("**objects to synonymy**")
            if checks["defect_fully_detected"]:
                flags.append("detects defect")
            if checks["approved_clean_draft"]:
                flags.append("approves")
            lines.append(f"- `{label}` → `{cell['decision']}` "
                         f"({'usable' if cell['valid'] else 'UNUSABLE'}), "
                         f"{cell['objections']} objection(s)"
                         + (f" — {', '.join(flags)}" if flags else ""))
            lines.append(f"  - first objection: {cell['first_objection']}")
        lines.append("")
    return "\n".join(lines) + "\n"


async def main_async(model: str, out_dir: pathlib.Path, *, dry_run: bool) -> dict:
    instructions = frozen_instructions()
    plan = {
        "mode": MODE,
        "instructions_sha256": hashlib.sha256(instructions.encode()).hexdigest(),
        "cases": [c for c, _g, _f in CAST],
        "max_requests": MAX_REQUESTS,
        "max_spend_usd": MAX_SPEND_USD,
        "attempts_per_call": ATTEMPTS_PER_CALL,
        "model": model,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    if dry_run:
        return {"dry_run": True, "plan": plan}

    meter = RequestMeter(max_requests=MAX_REQUESTS, max_cost_usd=MAX_SPEND_USD)
    ledger: list[dict] = []
    ask = make_asker(model=model, ledger=ledger, meter=meter)

    rows = []
    for case_id, _group, _friendly in CAST:
        row = await review(ask, meter, model, case_id, instructions)
        rows.append(row)
        print(f"  {model} {case_id:26s} -> {row['decision'] or row['error']}")
    ask.meter = meter

    payload = {"model": model, "mode": MODE,
               "instructions_sha256": plan["instructions_sha256"],
               "meter": meter.to_dict(), "rows": rows,
               "ledger": ledger, "models": [model],
               **compare(rows)}
    (out_dir / f"cross-model-{model.replace('/', '_')}.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    (out_dir / f"CROSS_MODEL_{model.replace('/', '_')}.md").write_text(render(payload))
    return {"meter": meter.to_dict(), "summary": {model: payload[model]}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="results/cross-model-v1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    result = asyncio.run(main_async(args.model, pathlib.Path(args.out),
                                    dry_run=args.dry_run))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
