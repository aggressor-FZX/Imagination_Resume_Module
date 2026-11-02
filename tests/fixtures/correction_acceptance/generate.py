#!/usr/bin/env python3
"""Generate inspectable, synthetic-only evidence through the real correction engine."""
from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import json
import socket
import sys
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from source_correction import SourceDocument, correct  # noqa: E402

SOURCES = (
    SourceDocument("identity", "Alex Example\n"),
    SourceDocument("work", "Alpha: engineer, 2019–2021\n"
                   "Alpha: improved accuracy 30% over 6 months\n"
                   "Beta: engineer, 2022–Present\n"),
    SourceDocument("education", "North College: BA Mathematics, 2014\n"
                   "West University: MS Physics, 2016\n"),
    SourceDocument("skills", "Diagnostics\n"),
)
ANSWER = ("# Alex Example\n## Experience\n"
          "Alpha: engineer, 2019–2021\n"
          "Alpha: improved accuracy 30% over 6 months\n"
          "Beta: engineer, 2022–Present\n## Education\n"
          "North College: BA Mathematics, 2014\n"
          "West University: MS Physics, 2016\n## Skills\nDiagnostics\n")
DURATION = "Alpha: improved accuracy 30% over 6 months"
EXTRA = " within 2 weeks"
WRONG_EMPLOYER = "Beta: improved accuracy 30% over 6 months"
NORTH = "North College: BA Mathematics, 2014"
WEST = "West University: MS Physics, 2016"
SWAPPED_NORTH = "North College: MS Physics, 2016"
SWAPPED_WEST = "West University: BA Mathematics, 2014"


def item(source_id, start, end, text):
    return {"id": f"{source_id}:{start}:{end}", "source_id": source_id,
            "start": start, "end": end, "quote": text[start:end],
            "classification": "required"}


def inventory(payload):
    assert set(payload) == {"contract", "sources"}, "inventory must not receive a draft"
    entries = []
    for source in payload["sources"]:
        position = 0
        for line in source["text"].splitlines(keepends=True):
            end = position + len(line)
            entries.append(item(source["source_id"], position, end, source["text"]))
            position = end
    return {"source_claims": entries}


def reviewed(payload, suffixes=(), substitutions=None, mark_missing=None):
    """Build scripted decisions, not a semantic reviewer.

    substitutions deliberately assert false equivalences for adversarial cases.
    Suffixes identify whole unsupported suffixes, leaving exact source prefixes.
    """
    substitutions = substitutions or {}
    claims, mappings = [], []
    matched = {}
    for line in payload["lines"]:
        text, line_id = line["text"], line["line_id"]
        if text.startswith("## "):
            parts = [(0, len(text), "structural", [])]
        else:
            suffix = next((s for s in suffixes if text.endswith(s)), "")
            prefix = text[:-len(suffix)] if suffix else text
            desired = substitutions.get(prefix, prefix.lstrip("# "))
            source_item = next((entry for entry in payload["source_claims"]
                                if entry["quote"].strip() == desired), None)
            if source_item is None:
                raise ValueError(f"Unscripted synthetic line: {text!r}")
            citation = {k: source_item[k] for k in ("source_id", "start", "end", "quote")}
            parts = [(0, len(prefix), "supported", [citation])]
            if suffix:
                parts.append((len(prefix), len(text), "unsupported", []))
            matched[source_item["id"]] = {"line_id": line_id, "start": 0, "end": len(prefix)}
        for start, end, verdict, citations in parts:
            claims.append({"id": f"{line_id}:{start}:{end}", "line_id": line_id,
                           "start": start, "end": end, "verdict": verdict,
                           "citations": citations})
    for entry in payload["source_claims"]:
        span = matched.get(entry["id"])
        if entry["quote"].strip() == mark_missing:
            span = None
        mappings.append({"source_claim_id": entry["id"],
                         "status": "represented" if span else "missing",
                         "spans": [span] if span else []})
    return {"claims": claims, "representations": mappings}


def case(name, draft, mode, expected, description):
    return {"name": name, "draft": draft, "mode": mode, "expected": expected,
            "description": description}


REPAIRED = ["source_inventory", "initial_review", "repair", "post_patch_review"]
CASES = [
    case("supported_duration_correct_employer", ANSWER, "honest",
         {"status": "approved", "markdown": ANSWER, "phases": ["source_inventory", "initial_review"],
          "review_calls": 1, "repair_calls": 0},
         "Supported duration and employer remain unchanged; both education records remain associated."),
    case("unsupported_duration_repair", ANSWER.replace(DURATION, DURATION + EXTRA), "remove_suffix",
         {"status": "approved", "markdown": ANSWER, "phases": REPAIRED,
          "review_calls": 2, "repair_calls": 1},
         "Remove the scripted unsupported duration while retaining the supported six months and all unrelated facts."),
    case("repeated_invalid_repairs", ANSWER.replace(DURATION, DURATION + EXTRA), "repeat",
         {"status": "withheld", "reason": "unresolved_claims", "markdown": None,
          "phases": ["source_inventory", "initial_review", "repair", "post_patch_review", "repair", "post_patch_review"],
          "review_calls": 3, "repair_calls": 2},
         "Two syntactically valid repairs introduce new unsupported facts; further repair is bounded."),
    case("revalidation_missing_fact", ANSWER.replace(DURATION, DURATION + EXTRA), "missing_after_repair",
         {"status": "withheld", "reason": "source_facts_missing", "markdown": None,
          "phases": REPAIRED, "review_calls": 2, "repair_calls": 1},
         "The post-patch reviewer marks an unrelated original education fact missing; publication is withheld."),
    case("wrong_employer_misleading_review", ANSWER.replace(DURATION, WRONG_EMPLOYER), "wrong_employer",
         {"status": "approved", "markdown": ANSWER, "reason": None,
          "phases": ["source_inventory", "initial_review"], "review_calls": 1, "repair_calls": 0},
         "A literal record under the wrong explicit label is repaired to its single source owner; "
         "the accomplishment, dates and every unrelated fact are preserved."),
    case("unresolved_attribution_negative_control",
         ANSWER.replace(NORTH, SWAPPED_NORTH).replace(WEST, SWAPPED_WEST), "swapped_education",
         {"status": "withheld", "reason": "attribution_conflict", "markdown": None,
          "phases": ["source_inventory", "initial_review"], "review_calls": 1, "repair_calls": 0},
         "Negative control: the reviewer approves exchanged records, but two labels rewritten into "
         "each other are not a unique repair, so the draft is rejected rather than guessed."),
]


def blocked(*args, **kwargs):
    raise RuntimeError("network disabled in synthetic acceptance generator")


async def execute(spec):
    trace = []
    errors = []
    repairs = 0

    async def callback(phase, payload):
        nonlocal repairs
        record = {"phase": phase, "payload": copy.deepcopy(payload)}
        trace.append(record)
        try:
            if phase == "source_inventory":
                response = inventory(payload)
            elif phase == "repair":
                repairs += 1
                replacement = (" invented period" if repairs == 1 else " another invented period") if spec["mode"] == "repeat" else ""
                response = {"edits": [{"claim_id": cid, "replacement": replacement}
                                      for cid in payload["flagged_claim_ids"]]}
            elif phase in {"initial_review", "post_patch_review"}:
                substitutions = {}
                if spec["mode"] == "swapped_education":
                    substitutions = {SWAPPED_NORTH: NORTH, SWAPPED_WEST: WEST}
                suffixes = (EXTRA, " another invented period", " invented period")
                missing = NORTH if spec["mode"] == "missing_after_repair" and phase == "post_patch_review" else None
                response = reviewed(payload, suffixes, substitutions, missing)
            else:
                raise ValueError(f"Unscripted callback phase: {phase}")
            record["response"] = copy.deepcopy(response)
            return response
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            record["script_error"] = errors[-1]
            raise

    result = await correct(spec["draft"], SOURCES, callback, timeout_seconds=5)
    actual = dataclasses.asdict(result)
    actual["phases"] = [entry["phase"] for entry in trace]
    checks = []
    for key, expected in spec["expected"].items():
        value = actual.get(key)
        checks.append({"assertion": key, "expected": expected, "actual": value,
                       "pass": value == expected})
    checks.append({"assertion": "callback_script_errors", "expected": [], "actual": errors, "pass": not errors})
    if spec["mode"] == "wrong_employer":
        restored = ("restored 1 source-owned record label" in actual["commentary"].lower()
                    and actual["markdown"] is not None
                    and WRONG_EMPLOYER not in actual["markdown"]
                    and DURATION in actual["markdown"])
        checks.append({"assertion": "wrong_employer_corrected_to_single_owner",
                       "expected": True, "actual": restored, "pass": restored})
    if spec["mode"] == "swapped_education":
        # A broken callback, malformed response or timeout is not attribution protection.
        valid_rejection = (actual["reason"] == "attribution_conflict"
                           and actual["phases"] == ["source_inventory", "initial_review"]
                           and not errors)
        checks.append({"assertion": "unresolved_attribution_bounded_rejection",
                       "expected": True, "actual": valid_rejection, "pass": valid_rejection})
    commentary_ok = ("independent review" in actual["commentary"].lower() and
                     (actual["status"] != "approved" or "not proof" in actual["commentary"].lower()))
    checks.append({"assertion": "commentary_discloses_limitation", "expected": True,
                   "actual": commentary_ok, "pass": commentary_ok})
    if spec["mode"] == "remove_suffix":
        explanatory = "revised 1 span(s) flagged by the reviewer" in actual["commentary"].lower()
        checks.append({"assertion": "correction_commentary_identifies_revision", "expected": True,
                       "actual": explanatory, "pass": explanatory})
    if spec["mode"] == "repeat":
        useful = "two correction attempts" in actual["commentary"].lower()
        checks.append({"assertion": "bounded_stop_commentary", "expected": True,
                       "actual": useful, "pass": useful})
    for index, entry in enumerate(trace):
        if entry["phase"] == "post_patch_review":
            lines = [line["text"] for line in entry["payload"]["lines"]]
            retained = all(fact in lines for fact in (NORTH, WEST, "Diagnostics", "Beta: engineer, 2022–Present"))
            checks.append({"assertion": f"post_patch_{index}_unrelated_facts_retained",
                           "expected": True, "actual": retained, "pass": retained})
    return {"name": spec["name"], "description": spec["description"],
            "draft": spec["draft"], "expected": spec["expected"],
            "actual": actual, "checks": checks, "trace": trace}


async def run():
    with patch.object(socket.socket, "connect", blocked), patch.object(socket.socket, "connect_ex", blocked), patch.object(socket, "getaddrinfo", blocked):
        return [await execute(spec) for spec in CASES]


def main():
    results = asyncio.run(run())
    failed = sum(not check["pass"] for result in results for check in result["checks"])
    bundle = {"notice": "Synthetic data and scripted callback decisions only; not semantic proof or provider evaluation.",
              "source_correction_sha256": hashlib.sha256((ROOT / "source_correction.py").read_bytes()).hexdigest(),
              "sources": [dataclasses.asdict(source) for source in SOURCES],
              "answer_key": ANSWER, "cases": results,
              "summary": {"cases": len(results), "failed_assertions": failed}}
    output = HERE / "results.json"
    output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for result in results:
        failures = [check["assertion"] for check in result["checks"] if not check["pass"]]
        print(f"{result['name']}: {result['actual']['status']}" + (f"; UNMET: {', '.join(failures)}" if failures else "; matched"))
    print(f"Wrote {output}; {failed} unmet assertions")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())