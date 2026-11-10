#!/usr/bin/env python3
"""Can a candidate reviewer speak the LIVE contract? Probe before any merge.

Why this must come first
------------------------
The live service does not use `decision_contract`. `app.py` imports
`source_correction`, whose reviewer must answer a JSON span envelope:

    {"claims": [{id, line_id, start, end, verdict, citations}], "representations": [...]}

with exact code-point offsets and quotes that must equal the source slice byte for
byte. Gemma was selected on the PROSE contract. Those are different protocols.

If Gemma cannot produce a valid span envelope, pointing the imaginator at it would
make **every real résumé withhold** — a worse outcome than the current reviewer.
So this probe runs the real `_INVENTORY_CONTRACT` and `_REVIEW_CONTRACT` prompts
against the candidate model, through the real provider adapter, and reports whether
each phase produced something the engine would accept.

It calls the engine's own validators rather than re-implementing them, so a pass
here means the engine would accept the answer.

Usage:
    python -m experiment.probe_live_contract --model google/gemma-4-31b-it --plan
    python -m experiment.probe_live_contract --model ... --authorized TOKEN
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiment.authorization import (  # noqa: E402
    AuthorizationError, consume_authorization, record_outcome,
)
from correction_provider import _strict_json  # noqa: E402
from experiment.provider_v2 import RequestMeter, make_asker  # noqa: E402

MAX_CALLS = 4
MAX_REQUESTS = 8
MAX_SPEND = 0.05


def build_payloads():
    """The engine's own contracts and a small real draft/source pair."""
    import source_correction as sc
    from experiment.cases_v2 import SOURCES, by_id

    case = by_id("unsupported_method")
    draft = case.draft_before_trim
    sources = [sc.SourceDocument(name, text) for name, text in SOURCES.items()]
    source_payload = [{"source_id": s.source_id, "text": s.text} for s in sources]
    return sc, draft, sources, source_payload


async def probe(model: str, out_dir: pathlib.Path) -> dict:
    sc, draft, sources, source_payload = build_payloads()
    meter = RequestMeter(max_requests=MAX_REQUESTS, max_cost_usd=MAX_SPEND)
    ledger: list[dict] = []
    ask = make_asker(model=model, ledger=ledger, meter=meter, max_attempts=2)

    results: dict = {"model": model, "phases": {}}

    # ── Phase 1: source_inventory ───────────────────────────────────────────
    inv_payload = {"contract": sc._INVENTORY_CONTRACT, "sources": source_payload}
    inv_request = {"model": model, "messages": [
        {"role": "system",
         "content": sc._INVENTORY_CONTRACT + " Return only a JSON object."},
        {"role": "user", "content": json.dumps(inv_payload, ensure_ascii=False)},
    ]}
    call = await ask(json.dumps(inv_request), label="probe:inventory")
    inv_text = call.text
    inventory = None
    try:
        inventory = _strict_json(inv_text)
        inv_ok = isinstance(inventory, dict) and bool(inventory)
    except Exception as exc:  # noqa: BLE001
        inv_ok = False
        results["phases"]["inventory_error"] = f"{type(exc).__name__}: {exc}"
    results["phases"]["source_inventory"] = {
        "ok": inv_ok, "raw_full": inv_text,
        "keys": sorted(inventory)[:8] if isinstance(inventory, dict) else None}

    if not inv_ok:
        results["verdict"] = "candidate_cannot_speak_live_contract"
        return results

    # ── Phase 2: initial_review, validated by the ENGINE's validator ────────
    # Exactly the shape the engine sends: `lines`, not a draft blob.
    lines_payload = [{"line_id": line["line_id"], "text": line["text"]}
                     for line in sc._lines(draft)]
    review_payload = {"contract": sc._REVIEW_CONTRACT, "sources": source_payload,
                      "source_claims": inventory, "lines": lines_payload}
    rev_request = {"model": model, "messages": [
        {"role": "system",
         "content": sc._REVIEW_CONTRACT + " Return only a JSON object."},
        {"role": "user", "content": json.dumps(review_payload, ensure_ascii=False)},
    ]}
    call = await ask(json.dumps(rev_request), label="probe:review")
    rev_text = call.text
    parsed = None
    try:
        parsed = _strict_json(rev_text)
    except Exception as exc:  # noqa: BLE001
        results["phases"]["review_parse_error"] = f"{type(exc).__name__}: {exc}"

    lines = sc._lines(draft)
    validate_error = None
    if parsed is not None:
        try:
            sc._validate_review(parsed, lines,
                                {s.source_id: s.text for s in sources},
                                inventory)
        except Exception as exc:  # noqa: BLE001
            validate_error = f"{type(exc).__name__}: {exc}"

    results["phases"]["initial_review"] = {
        "ok": parsed is not None and validate_error is None,
        "validate_error": validate_error,
        "raw_full": rev_text,
        "claim_count": len((parsed or {}).get("claims") or [])
        if isinstance(parsed, dict) else None,
        "representation_count": len((parsed or {}).get("representations") or [])
        if isinstance(parsed, dict) else None,
    }
    results["verdict"] = (
        "candidate_speaks_live_contract"
        if results["phases"]["initial_review"]["ok"]
        else "candidate_cannot_speak_live_contract")
    results["meter"] = meter.to_dict()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="results/live-contract-probe")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--authorized", metavar="TOKEN")
    ap.add_argument("--ledger", default="results/.authorizations.json")
    args = ap.parse_args()

    if args.plan:
        print(json.dumps({"model": args.model, "logical_calls": MAX_CALLS,
                          "max_requests": MAX_REQUESTS,
                          "max_spend_usd": MAX_SPEND,
                          "phases": ["source_inventory", "initial_review"]},
                         indent=2))
        return 0
    if not args.authorized:
        ap.error("pass --plan or --authorized TOKEN")

    try:
        consume_authorization(args.authorized, expected_requests=MAX_REQUESTS,
                              ledger=pathlib.Path(args.ledger))
    except AuthorizationError as exc:
        print(f"REFUSED: {exc}")
        return 2

    results = asyncio.run(probe(args.model, pathlib.Path(args.out)))
    meter = results.get("meter") or {}
    record_outcome(args.authorized, requests_sent=meter.get("requests_sent", 0),
                   cost_usd=meter.get("cost_usd", 0.0),
                   ledger=pathlib.Path(args.ledger))
    print(json.dumps({k: v for k, v in results.items() if k != "meter"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
