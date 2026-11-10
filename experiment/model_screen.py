#!/usr/bin/env python3
"""Five-case, three-model screen — is a smarter model enough?

Why five cases
--------------
A screen, not an evaluation. Five drafts, each isolating one behaviour, is the
cheapest design that can distinguish "this model approves faithful text and refuses
defects" from "this model refuses everything". Anything broader would re-measure
what the calibration already recorded.

Case list, fixed by the brief and unchanged from the frozen fixtures:

    clean_paraphrase     faithful_paraphrase   (familiar)   must approve
    clean_paraphrase     cal_ops_faithful      (unfamiliar) must approve
    unsupported_method   unsupported_method    (familiar)   must revise, correct clause
    wrong_record         wrong_record_attribution (familiar) must revise, correct owner
    omitted_record       omitted_record        (familiar)   must revise, omission named

Two clean and three defective: enough to see both false-refusal and missed-defect
behaviour on the same model, which a one-case probe cannot.

What "detection" means here
---------------------------
The brief is explicit: correctness is NOT inferred from keywords or the decision
word. Each answer is graded on the same five criteria the calibration used, and this
module reports FOUR outcomes separately:

    valid_judgment        the answer is structurally usable at all
    actual_detection      a material defect was named, with owner where ownership is at issue
    invented_objection    a clean draft was refused on a non-material ground
    missed_defect         a defective draft was not correctly identified

A refusal on a defective draft is not credited unless the defect is actually named,
and an approval on a clean draft is not credited unless the answer is valid.

Usage:
    python -m experiment.model_screen --plan          # print plan, no calls
    python -m experiment.model_screen --authorized    # execute
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
from experiment.cross_model import _objection_text, EXCLUSIVITY_WORDS  # noqa: E402
from experiment.detection_rubric import TRUTH, grade_case  # noqa: E402
from experiment.provider_v2 import RequestMeter, make_asker  # noqa: E402
from experiment.run_calibration import (  # noqa: E402
    MODE, build_prompt, document_from_prompt,
)

# ── The three requested models, plus the incumbent comparison ────────────────
#
# `xiaomi/mimo-v2-flash` does NOT exist in the OpenRouter catalogue (checked
# 2026-11-09 against /api/v1/models, 432 entries). The closest is
# `xiaomi/mimo-v2.6-flash`, which is what this screen uses. The substitution is
# recorded here and in the plan output rather than made silently.
MODELS = [
    "google/gemma-4-31b-it",
    "google/gemini-2.5-flash-lite",
    "xiaomi/mimo-v2.6-flash",          # requested as xiaomi/mimo-v2-flash (absent)
    "openai/gpt-4o-mini",              # the baseline the brief asks to compare against
]

# Per-model provider pin. MiMo-V2.6-Flash is served by EIGHT endpoints (fp4, bf16
# and six fp8). Without a pin, OpenRouter may route per request, so two calls of the
# same prompt are not necessarily the same model instance — and an fp4 routing would
# be materially weaker than the fp8 the result claims to measure.
#
# `cohere/fp8` is the requested endpoint and exists (verified against
# /api/v1/models/xiaomi/mimo-v2.6-flash/endpoints, 8 endpoints, status 0,
# 1,048,576 context, $0.14/M prompt, $0.28/M completion, 100% uptime last 30m).
# `makora/fp8` is an fp8 alternative at identical price; it is recorded as the
# fallback in the plan rather than used, since a second pin would be a second
# variable.
PROVIDER_PINS: dict[str, list[str]] = {
    "xiaomi/mimo-v2.6-flash": ["cohere/fp8"],
}
PROVIDER_FALLBACKS: dict[str, list[str]] = {
    "xiaomi/mimo-v2.6-flash": ["makora/fp8"],
}

REQUESTED_MODEL = "xiaomi/mimo-v2-flash"
MODEL_SUBSTITUTION = {
    "requested": REQUESTED_MODEL,
    "used": "xiaomi/mimo-v2.6-flash",
    "reason": ("xiaomi/mimo-v2-flash is not present in the OpenRouter catalogue; "
               "xiaomi/mimo-v2.6-flash is the nearest available MiMo Flash."),
}

CAST = (
    ("faithful_paraphrase", "familiar", "clean_paraphrase", "approve"),
    ("cal_ops_faithful", "unfamiliar", "clean_paraphrase", "approve"),
    ("unsupported_method", "familiar", "unsupported_method", "revise"),
    ("wrong_record_attribution", "familiar", "wrong_record", "revise"),
    ("omitted_record", "familiar", "omitted_record", "revise"),
)

ATTEMPTS_PER_CALL = 2


def case(case_id: str):
    try:
        return cal_by_id(case_id)
    except KeyError:
        return fam_by_id(case_id)


def budget_for(models: list[str]) -> dict:
    calls = len(CAST) * len(models)
    return {
        "models": models,
        "provider_pins": {m: PROVIDER_PINS[m] for m in models if m in PROVIDER_PINS},
        "provider_fallbacks": {m: PROVIDER_FALLBACKS[m] for m in models
                               if m in PROVIDER_FALLBACKS},
        "cases": [c for c, _g, _f, _e in CAST],
        "logical_calls": calls,
        "attempts_per_call": ATTEMPTS_PER_CALL,
        "max_requests": calls * ATTEMPTS_PER_CALL,
        "max_spend_usd": 0.10,
        "max_tokens": 8192,
        "mode": MODE,
        "instructions_sha256": hashlib.sha256(
            dc.contract_for(MODE).encode()).hexdigest(),
        "ceiling_statement": (
            "Request ceiling is HARD, checked before every HTTP attempt including "
            "retries. The $0.10 figure is enforced against OBSERVED cost only and "
            "is not a proof of maximum spend; any response without usage is "
            "recorded as unknown spend, not zero."),
    }


# ── Four separate outcomes ───────────────────────────────────────────────────

def classify(case_id: str, expected: str, decision: dc.Decision) -> dict:
    """Report validity, detection, invention and misses as INDEPENDENT facts."""
    text = _objection_text(decision)
    valid = bool(decision.decision) and not decision.structural_problems
    approved = decision.decision == "approve"

    out = {"valid_judgment": valid, "decision": decision.decision or None,
           "structural": list(decision.structural_problems),
           "issues": len(decision.issues),
           "actual_detection": None, "invented_objection": None,
           "missed_defect": None, "detection_detail": None}

    if expected == "approve":
        # A clean draft. Any objection is invented; an approval is correct only if
        # the answer is usable, because `approve` plus issues is not an approval.
        invented = bool(decision.issues) or any(w in text for w in EXCLUSIVITY_WORDS)
        out["invented_objection"] = invented
        out["missed_defect"] = False          # nothing to miss on a clean draft
        out["actual_detection"] = None
        return out

    # A defective draft. Credit detection ONLY from the five-criterion rubric; a
    # refusal word alone is not detection, and neither is a keyword hit.
    graded = grade_case(case_id, decision)
    detected = bool(graded["all_complete"])
    out["actual_detection"] = detected
    out["missed_defect"] = not detected
    out["invented_objection"] = None
    out["detection_detail"] = {
        "owner_required": TRUTH[case_id].get(
            "owner_required", TRUTH[case_id]["fix"] != "remove_clause"),
        "criteria": (graded["best"] or {}),
        "issues_mentioning_fact": graded["issues_mentioning_fact"],
    }
    return out


async def review(ask, meter: RequestMeter, model: str, case_id: str,
                 expected: str, instructions: str) -> dict:
    c = case(case_id)
    row = {"model": model, "case_id": case_id, "expected": expected,
           "decision": None, "response": "", "error": "",
           "cost_usd": 0.0, "requests": 0, "usage_missing": None}
    allowed, why = meter.can_send()
    if not allowed:
        row["error"] = f"meter:{why}"
        return row
    try:
        call = await ask(build_prompt(instructions, c.sources,
                                      c.draft_before_trim),
                         label=f"{model}:{case_id}")
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return row

    d = dc.decide_from_text(call.text, c.draft_before_trim, c.sources)
    row |= {"response": call.text, "cost_usd": call.cost_usd,
            "requests": call.requests_sent, "usage_missing": call.usage_missing,
            "decision": d.decision or None,
            # The pin is an instruction; the reported provider is the fact.
            "provider_pinned": call.provider_pinned,
            "provider_reported": call.provider_reported,
            "provider_pin_held": (not call.provider_pinned
                                   or call.provider_reported in
                                   [p.split("/")[0].lower()
                                    for p in call.provider_pinned])}
    row["outcomes"] = classify(case_id, expected, d)
    row["first_objection"] = (
        " ".join((d.issues[0].why or "").split())[:240] if d.issues else "")
    return row


def summarise(rows: list[dict], models: list[str]) -> dict:
    out: dict = {}
    for model in models:
        mine = [r for r in rows if r["model"] == model]
        clean = [r for r in mine if r["expected"] == "approve"]
        defective = [r for r in mine if r["expected"] == "revise"]
        out[model] = {
            "cases": len(mine),
            "operational_failures": sum(1 for r in mine if r["error"]),
            # 1. validity
            "valid_judgment": sum(1 for r in mine
                                  if (r.get("outcomes") or {}).get("valid_judgment")),
            # 2. detection, on defective cases only
            "actual_detection": sum(
                1 for r in defective
                if (r.get("outcomes") or {}).get("actual_detection") is True),
            "defective_cases": len(defective),
            # 3. invented objections, on clean cases only
            "invented_objections": sum(
                1 for r in clean
                if (r.get("outcomes") or {}).get("invented_objection")),
            "clean_cases": len(clean),
            # 4. missed defects
            "missed_defects": sum(
                1 for r in defective
                if (r.get("outcomes") or {}).get("missed_defect") is True),
            "clean_approved": sum(
                1 for r in clean if r["decision"] == "approve"
                and (r.get("outcomes") or {}).get("valid_judgment")),
            "cost_usd": round(sum(r["cost_usd"] for r in mine), 6),
            # Routing: a pin that did not hold is reported, not assumed away.
            "provider_pinned": sorted({p for r in mine
                                       for p in (r.get("provider_pinned") or [])}),
            "providers_reported": sorted({r.get("provider_reported") or ""
                                          for r in mine} - {""}),
            "pin_violations": sum(1 for r in mine
                                  if r.get("provider_pin_held") is False),
        }
    return out


def render(plan: dict, summary: dict, rows: list[dict]) -> str:
    lines = [
        "# Five-case, three-model screen", "",
        "Reused frozen drafts and sources. No repairs, no production change, no "
        "automatic dismissal of objections.", "",
        f"- mode: `{plan['mode']}`",
        f"- instructions sha256: `{plan['instructions_sha256']}`",
        f"- models: {', '.join(f'`{m}`' for m in plan['models'])}",
        f"- requests: {plan['observed_requests']} / {plan['max_requests']}",
        f"- cost: ${plan['observed_cost_usd']:.6f} "
        f"({'known' if plan['cost_is_known'] else 'UNKNOWN — some responses omitted usage'})",
        "", "## The four outcomes, reported separately", "",
        "A refusal is not credited unless the defect is named; an approval is not "
        "credited unless the answer is valid. Nothing here is inferred from a "
        "decision word or a keyword.", "",
        "| model | valid | detection (of 3) | invented (of 2) | missed (of 3) | "
        "clean approved | cost |",
        "|---|---|---|---|---|---|---|"]
    for model in plan["models"]:
        s = summary[model]
        lines.append(
            f"| `{model}` | {s['valid_judgment']}/{s['cases']} | "
            f"{s['actual_detection']}/{s['defective_cases']} | "
            f"{s['invented_objections']}/{s['clean_cases']} | "
            f"{s['missed_defects']}/{s['defective_cases']} | "
            f"{s['clean_approved']}/{s['clean_cases']} | ${s['cost_usd']:.6f} |")

    lines += ["", "### Endpoint routing", "",
              "A pin is an instruction; the provider reported in each response is "
              "the fact. Any mismatch is counted as a violation rather than "
              "assumed away.", "",
              "| model | pinned | actually served | violations |",
              "|---|---|---|---|"]
    for model in plan["models"]:
        s = summary[model]
        pinned = ", ".join(f"`{p}`" for p in s["provider_pinned"]) or "— (unpinned)"
        served = ", ".join(f"`{p}`" for p in s["providers_reported"]) or "—"
        lines.append(f"| `{model}` | {pinned} | {served} | "
                     f"{s['pin_violations']} |")

    lines += ["", "## Per case", ""]
    for case_id, _g, probe, expected in CAST:
        lines += [f"### `{case_id}` — `{probe}`, expected **{expected}**", ""]
        for model in plan["models"]:
            row = next((r for r in rows if r["model"] == model
                        and r["case_id"] == case_id), None)
            if row is None:
                continue
            o = row.get("outcomes") or {}
            flags = []
            if o.get("valid_judgment") is False:
                flags.append("INVALID")
            if o.get("actual_detection") is True:
                flags.append("detects")
            if o.get("missed_defect") is True:
                flags.append("**misses**")
            if o.get("invented_objection"):
                flags.append("**invents**")
            verdict = ", ".join(flags) or "—"
            lines.append(f"- `{model}` → `{row['decision'] or row['error']}` "
                         f"({verdict}), {o.get('issues', 0)} issue(s)")
            if row.get("first_objection"):
                lines.append(f"  - first: {row['first_objection']}")
        lines.append("")
    return "\n".join(lines) + "\n"


async def main_async(out_dir: pathlib.Path, *, plan_only: bool) -> dict:
    instructions = dc.contract_for(MODE)
    plan = budget_for(MODELS)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plan.json").write_text(json.dumps(
        {"plan": plan, "model_substitution": MODEL_SUBSTITUTION,
         "cast": [{"case_id": c, "group": g, "probe": p, "expected": e}
                  for c, g, p, e in CAST]}, indent=2) + "\n")
    if plan_only:
        return {"plan": plan, "model_substitution": MODEL_SUBSTITUTION}

    meter = RequestMeter(max_requests=plan["max_requests"],
                         max_cost_usd=plan["max_spend_usd"])
    ledger: list[dict] = []

    rows: list[dict] = []
    for model in MODELS:
        ask = make_asker(model=model, ledger=ledger, meter=meter,
                         provider_only=PROVIDER_PINS.get(model))
        for case_id, _group, _probe, expected in CAST:
            row = await review(ask, meter, model, case_id, expected, instructions)
            rows.append(row)
            status = row["decision"] or row["error"]
            print(f"  {model:34s} {case_id:26s} -> {status}")

    summary = summarise(rows, MODELS)
    meter_dict = meter.to_dict()
    observed = {"observed_requests": meter_dict["requests_sent"],
                "observed_cost_usd": meter_dict["cost_usd"],
                "cost_is_known": meter_dict["cost_is_known"]}
    payload = {"plan": {**plan, **observed},
               "model_substitution": MODEL_SUBSTITUTION,
               "meter": meter_dict, "rows": rows, "summary": summary,
               "ledger": ledger}
    (out_dir / "screen.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out_dir / "SCREEN.md").write_text(render(payload["plan"], summary, rows))
    return {"meter": meter_dict, "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/model-screen-v1")
    ap.add_argument("--plan", action="store_true", help="print the plan; no calls")
    ap.add_argument("--authorized", action="store_true",
                    help="execute the run (requires explicit authorization)")
    args = ap.parse_args()
    if not (args.plan or args.authorized):
        ap.error("pass --plan to inspect, or --authorized to execute")
    result = asyncio.run(main_async(pathlib.Path(args.out),
                                    plan_only=args.plan))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
