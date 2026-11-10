#!/usr/bin/env python3
"""Authorized Gemma repair-and-release test — live output through the real path.

Scope: 10 logical calls / 20 requests / $0.05 observed, one token, no reruns.

What this adds over the offline tests
-------------------------------------
The offline regressions replay Gemma's RETAINED answers. They prove the repair
logic handles those exact corrections. They do not prove that a LIVE Gemma call
produces a correction the repair can apply — the model could phrase it differently
and the repair could refuse.

So this run answers one question: **does the repair path work on live output?**

Cases, with two untouched controls:

    faithful_paraphrase       control (familiar)     must approve, 0 edits
    cal_ops_faithful          control (unfamiliar)   must approve, 0 edits
    unsupported_method        clause removal         must revise -> repair -> release
    wrong_record_attribution  wrong-owner correction must revise -> repair -> release
    omitted_record            record restoration     must revise -> restore -> release

The restoration case is expected to succeed now: `apply_repair` reads the block from
the SOURCE, so a model that proposes invented dates is refused rather than followed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import decision_contract as dc  # noqa: E402
from experiment.authorization import (  # noqa: E402
    AuthorizationError, consume_authorization, record_outcome,
)
from experiment.calibration_cases import by_id as cal_by_id  # noqa: E402
from experiment.cases_v2 import SOURCES as FAM_SOURCES, by_id as fam_by_id  # noqa: E402
from experiment.provider_v2 import RequestMeter, make_asker  # noqa: E402
from experiment.release_trace import run_release, trace_markdown  # noqa: E402
from experiment.run_calibration import MODE, build_prompt  # noqa: E402

MODEL = "google/gemma-4-31b-it"
TOKEN = "gemma-repair-2026-11-09-01"
MAX_CALLS = 10
MAX_REQUESTS = 20
MAX_SPEND = 0.05

CAST = (
    ("faithful_paraphrase", "control_familiar", "approve", None),
    ("cal_ops_faithful", "control_unfamiliar", "approve", None),
    ("unsupported_method", "clause_removal", "revise", None),
    ("wrong_record_attribution", "wrong_owner", "revise", None),
    ("omitted_record", "record_restoration", "revise", None),
)


def case(case_id: str):
    try:
        return cal_by_id(case_id)
    except KeyError:
        return fam_by_id(case_id)


async def main_async(out_dir: pathlib.Path, *, render_only: bool) -> dict:
    if render_only:
        data = json.loads((out_dir / "repair.json").read_text())
        (out_dir / "REPAIR.md").write_text(
            "\n\n".join(trace_markdown_dict(t) for t in data["traces"]))
        print(f"rendered {out_dir / 'REPAIR.md'} from {out_dir / 'repair.json'}")
        return {"rendered": True}

    out_dir.mkdir(parents=True, exist_ok=True)
    meter = RequestMeter(max_requests=MAX_REQUESTS, max_cost_usd=MAX_SPEND)
    ledger: list[dict] = []
    ask = make_asker(model=MODEL, ledger=ledger, meter=meter,
                     max_attempts=2)

    instructions = dc.contract_for(MODE)

    async def review(document: str, sources: dict[str, str]) -> str:
        call = await ask(build_prompt(instructions, sources, document),
                         label=f"repair:{len(ledger)}")
        return call.text

    from publication_gate import issue_receipt, verify_receipt

    traces = []
    for case_id, role, expected, _ in CAST:
        c = case(case_id)
        trace = await run_release(case_id, c.draft_before_trim, c.sources,
                                  ask=review, issue_receipt=issue_receipt,
                                  verify_receipt=verify_receipt,
                                  owner="gemma-repair-test")
        d = trace.to_dict()
        d["role"] = role
        d["expected"] = expected
        d["edits_applied"] = trace.proposed_sha != "" and \
            trace.proposed_sha != trace.original_sha
        traces.append(d)
        print(f"  {case_id:26s} {trace.first_decision or 'none':8s} "
              f"repair={trace.repair_status:28s} released={trace.released}")

    payload = {"model": MODEL, "token": TOKEN, "mode": MODE,
               "instructions_sha256": dc.contract_fingerprint(MODE),
               "meter": meter.to_dict(), "traces": traces, "ledger": ledger}
    (out_dir / "repair.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out_dir / "REPAIR.md").write_text(
        "\n\n".join(trace_markdown_dict(t) for t in traces))
    return {"meter": meter.to_dict(), "traces": len(traces)}


def trace_markdown_dict(d: dict) -> str:
    """Render a recorded trace dict without rebuilding a ReleaseTrace."""
    from experiment.release_trace import ReleaseTrace
    return trace_markdown(ReleaseTrace(**{k: v for k, v in d.items()
                                          if k in ReleaseTrace.__annotations__
                                          or k in ReleaseTrace.__dataclass_fields__}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/gemma-repair-v1")
    ap.add_argument("--render", action="store_true",
                    help="re-render from the recorded artifact; no calls")
    ap.add_argument("--ledger", default="results/.authorizations.json")
    args = ap.parse_args()
    out = pathlib.Path(args.out)

    if args.render:
        result = asyncio.run(main_async(out, render_only=True))
        print(json.dumps(result, indent=2))
        return 0

    try:
        consume_authorization(TOKEN, expected_requests=MAX_REQUESTS,
                              ledger=pathlib.Path(args.ledger))
    except AuthorizationError as exc:
        print(f"REFUSED: {exc}")
        return 2

    try:
        result = asyncio.run(main_async(out, render_only=False))
    except BaseException:
        record_outcome(TOKEN, requests_sent=0, cost_usd=0.0, status="failed",
                       ledger=pathlib.Path(args.ledger))
        raise

    meter = result.get("meter", {})
    record_outcome(TOKEN, requests_sent=meter.get("requests_sent", 0),
                   cost_usd=meter.get("cost_usd", 0.0),
                   ledger=pathlib.Path(args.ledger))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
