#!/usr/bin/env python3
"""Replay the retained responses OFFLINE and classify every failure precisely.

Why this exists
---------------
The first comparison reported 21 of 24 attempts as `operational_failure`, and the
dominant reason was `envelope_not_json`. That single label concealed at least three
different things:

* a response **truncated mid-string** — the model ran out of output tokens;
* a response in the **format the prompt asked for**, which the harness then failed
  to accept;
* a response that was never JSON because that cell's contract is prose.

None of those is a semantic judgement about the résumé, and none is a JSON syntax
error. Reporting them together as one parser failure made the table look like a
measurement of approval quality when it was measuring plumbing.

This script re-runs interpretation over the RETAINED bytes, with no provider calls,
and prints for each attempt:

    exact outbound prompt (its sha256, and the answer format it requested)
    raw response, finish reason if recorded
    which parser the cell is entitled to use
    parsed decision
    the FIRST failing validation step, named specifically

and then a corrected classification that separates:

    truncation            output budget exhausted, response incomplete
    format_mismatch       answered in a format the prompt did not request
    parser_defect         correct format for the cell, rejected by the parser
    semantic_revise       the reviewer proposed a change (an EXPECTED workflow,
                          not a false refusal)
    semantic_refuse       the reviewer declined for a semantic reason
    approve               the reviewer approved
    operational           genuinely broken (auth, transport, empty answer)

Usage:
    python experiment/replay.py results/comparison-20261109-0719

No network. No provider. Safe to run repeatedly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import decision_contract as dc  # noqa: E402


# ── The answer format each cell's prompt actually requests ──────────────────
CURRENT_FORMAT = "json_span_envelope"
REVISED_FORMAT = "prose_decision"


def requested_format(cell: str) -> str:
    """What the OUTBOUND prompt asked for, read from the cell label.

    `current` cells are given the legacy span envelope contract and `revised`
    cells the prose decision contract. This is derived from the cell rather than
    guessed from the response, so a response in the wrong format is detectable.
    """
    return CURRENT_FORMAT if cell.startswith("current") else REVISED_FORMAT


def looks_truncated(raw: str, finish_reason: str | None) -> tuple[bool, str]:
    """Is the answer incomplete because the model ran out of output budget?

    Truncation is checked structurally, not by trusting `finish_reason` alone:
    an unterminated JSON object/string, or a response that ends mid-token, is
    incomplete regardless of what the provider reported. The provider's own
    `length` reason is honoured when present.
    """
    if finish_reason == "length":
        return True, "provider reported finish_reason=length"
    stripped = raw.rstrip()
    if not stripped:
        return False, ""
    if stripped.startswith("{"):
        try:
            json.loads(stripped)
            return False, ""
        except json.JSONDecodeError as exc:
            if "Unterminated string" in str(exc) or exc.pos >= len(stripped) - 2:
                return True, f"JSON ends before its closing brace at char {exc.pos}"
            return False, ""
    # A prose answer that stops without terminal punctuation mid-identifier is
    # also suspicious, but only a provider `length` can settle it.
    return False, ""


def classify(raw: str, cell: str, finish_reason: str | None,
             draft: str, sources: dict[str, str]) -> dict:
    """Interpret one retained answer and name the FIRST failing step."""
    fmt = requested_format(cell)
    out: dict = {
        "requested_format": fmt,
        "decision": "",
        "parsed": None,
        "first_failure": None,
        "category": None,
        "detail": "",
    }

    truncated, why = looks_truncated(raw, finish_reason)
    if truncated:
        out["category"] = "truncation"
        out["first_failure"] = "response_incomplete"
        out["detail"] = why
        return out

    if not raw.strip():
        out["category"] = "operational"
        out["first_failure"] = "empty_response"
        return out

    if fmt == REVISED_FORMAT:
        parsed = dc.decide_from_text(raw, draft, sources)
        out["decision"] = parsed.decision
        out["parsed"] = {
            "decision": parsed.decision,
            "issues": [
                {"problem": i.problem, "passage_resolves": i.passage_resolves,
                 "sources_resolve": i.sources_resolve, "why": i.why[:200],
                 "correction": (i.correction or "")[:200]}
                for i in parsed.issues
            ],
            "structural_problems": parsed.structural_problems,
        }
        if not parsed.ok:
            # A prose contract asked for prose; a prose answer that fails is a
            # PARSER or CONTRACT problem, never a JSON-format problem.
            out["category"] = "parser_defect"
            out["first_failure"] = parsed.structural_problems[0] if parsed.structural_problems else "unknown"
            out["detail"] = "; ".join(parsed.structural_problems[:3])
            return out
        out["category"] = _category_for_decision(parsed.decision)
        return out

    # CURRENT cell: the prompt asked for the JSON span envelope.
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Not JSON at all. Was it prose? Then the model ignored the format — a
        # format mismatch, not a parser defect and not a semantic verdict.
        if raw.lstrip().lower().startswith("decision:"):
            out["category"] = "format_mismatch"
            out["first_failure"] = "answered_prose_to_json_prompt"
            out["detail"] = "prompt requested the JSON span envelope"
            return out
        out["category"] = "parser_defect"
        out["first_failure"] = "envelope_not_json"
        out["detail"] = f"{exc.msg} at char {exc.pos}"
        return out

    if not isinstance(envelope, dict) or not isinstance(envelope.get("claims"), list):
        out["category"] = "parser_defect"
        out["first_failure"] = "claims_missing"
        return out

    claims = envelope["claims"]
    flagged = [c for c in claims
               if isinstance(c, dict) and c.get("verdict") in {"unsupported", "uncertain"}]
    # Reference validity, mirroring the runner's control.
    refs_ok = True
    ref_problem = ""
    for claim in claims:
        if not isinstance(claim, dict):
            refs_ok, ref_problem = False, "claim_not_object"
            break
        citations = claim.get("citations")
        if not isinstance(citations, list):
            refs_ok, ref_problem = False, "citations_not_list"
            break
        for citation in citations:
            if not isinstance(citation, dict):
                refs_ok, ref_problem = False, "citation_not_object"
                break
            text = sources.get(str(citation.get("source_id", "")))
            if text is None:
                refs_ok, ref_problem = False, f"unknown_source:{citation.get('source_id')}"
                break
            start, end, quote = citation.get("start"), citation.get("end"), citation.get("quote")
            if not (isinstance(start, int) and isinstance(end, int) and isinstance(quote, str)):
                refs_ok, ref_problem = False, "citation_malformed"
                break
            if not (0 <= start < end <= len(text)) or text[start:end] != quote:
                refs_ok, ref_problem = False, "quote_not_at_offset"
                break
        if not refs_ok:
            break

    out["parsed"] = {
        "claims": len(claims),
        "flagged": len(flagged),
        "references_valid": refs_ok,
        "reference_problem": ref_problem,
        "representations": len(envelope.get("representations") or []),
    }
    out["decision"] = dc.DECISION_REVISE if flagged else dc.DECISION_APPROVE
    if not refs_ok:
        # A reference control failing is a REFUSAL, not an operational failure.
        out["category"] = "semantic_refuse"
        out["first_failure"] = "reference_invalid"
        out["detail"] = ref_problem
        return out
    out["category"] = _category_for_decision(out["decision"])
    return out


def _category_for_decision(decision: str) -> str:
    if decision == dc.DECISION_APPROVE:
        return "approve"
    if decision == dc.DECISION_REVISE:
        # REVISE is the expected workflow, not a failure. Counting it as a false
        # refusal would penalise the contract for doing its job.
        return "semantic_revise"
    if decision == dc.DECISION_CANNOT_ASSESS:
        return "semantic_refuse"
    return "operational"


def find_finish_reason(attempt: dict) -> str | None:
    """The provider's finish reason, wherever the capture put it."""
    for key in ("finish_reason", "provider_finish_reason"):
        if attempt.get(key):
            return str(attempt[key])
    for container in ("provider_request", "provider_meta", "provider_response_meta"):
        value = attempt.get(container)
        if isinstance(value, dict) and value.get("finish_reason"):
            return str(value["finish_reason"])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", help="directory containing attempts.json")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--verbose", action="store_true",
                    help="print the outbound prompt and raw response per attempt")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run)
    attempts = json.loads((run_dir / "attempts.json").read_text())

    # Frozen drafts and sources come from the case set, so the replay judges the
    # SAME text the first run did.
    from experiment.cases import CASES  # noqa: WPS433
    by_id = {c.case_id: c for c in CASES}

    rows = []
    for attempt in attempts:
        case = by_id.get(attempt["case_id"])
        draft = attempt.get("draft_after_trim") or (case.draft if case else "")
        sources = dict(case.sources) if case else {}
        raw = attempt.get("provider_response") or ""
        finish = find_finish_reason(attempt)
        result = classify(raw, attempt["cell"], finish, draft, sources)
        prompt = (attempt.get("provider_request") or {}).get("prompt", "")
        rows.append({
            "cell": attempt["cell"],
            "case_id": attempt["case_id"],
            "mode": attempt.get("mode"),
            "prompt_sha256": hashlib.sha256(str(prompt).encode()).hexdigest()[:16],
            "prompt_format_requested": result["requested_format"],
            "finish_reason": finish,
            "response_chars": len(raw),
            "decision": result["decision"],
            "parsed": result["parsed"],
            "first_failure": result["first_failure"],
            "category": result["category"],
            "detail": result["detail"],
            "recorded_outcome": attempt.get("outcome"),
            "recorded_reason": attempt.get("outcome_reason"),
        })

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    if args.verbose:
        for attempt, row in zip(attempts, rows):
            print("=" * 78)
            print(f"{row['cell']} / {row['case_id']}  (mode={row['mode']})")
            print(f"  prompt sha256[:16] : {row['prompt_sha256']}")
            print(f"  format requested   : {row['prompt_format_requested']}")
            print(f"  finish_reason      : {row['finish_reason']}")
            print(f"  response chars     : {row['response_chars']}")
            print("  --- outbound prompt (tail) ---")
            tail = str((attempt.get("provider_request") or {}).get("prompt", ""))[-400:]
            print("   " + tail.replace("\n", "\n   "))
            print("  --- raw response ---")
            print("   " + str(attempt.get("provider_response", ""))[:600].replace("\n", "\n   "))
            print(f"  parsed decision    : {row['decision']!r}")
            print(f"  first failure      : {row['first_failure']}  [{row['category']}]")
            if row["detail"]:
                print(f"  detail             : {row['detail']}")
        print()

    # ── The table the request asked for ────────────────────────────────────
    header = (f"{'cell':22} {'case':26} {'fmt':6} {'fin':7} {'len':>5} "
              f"{'decision':13} {'first failing step':40} category")
    print(header)
    print("-" * len(header))
    for row in rows:
        fmt = "json" if row["prompt_format_requested"] == CURRENT_FORMAT else "prose"
        print(f"{row['cell']:22} {row['case_id']:26} {fmt:6} "
              f"{str(row['finish_reason'] or '-'):7} {row['response_chars']:>5} "
              f"{str(row['decision'] or '-'):13} "
              f"{str(row['first_failure'] or '-'):40} {row['category']}")

    # ── Corrected metrics ──────────────────────────────────────────────────
    print()
    print("RECORDED vs CORRECTED")
    from collections import Counter
    recorded = Counter(r["recorded_reason"] for r in rows)
    corrected = Counter(r["category"] for r in rows)
    print("  recorded outcome_reason :")
    for reason, count in recorded.most_common():
        print(f"    {count:3}  {reason}")
    print("  corrected category      :")
    for cat, count in corrected.most_common():
        print(f"    {count:3}  {cat}")

    print()
    print("INTERPRETATION")
    trunc = corrected.get("truncation", 0)
    mismatch = corrected.get("format_mismatch", 0)
    defect = corrected.get("parser_defect", 0)
    revise = corrected.get("semantic_revise", 0)
    print(f"  output-budget truncation : {trunc}  (raise max_tokens; response was cut off)")
    print(f"  format mismatch          : {mismatch}  (answered prose to a JSON prompt)")
    print(f"  parser / contract defect : {defect}  (correct format, rejected)")
    print(f"  expected revisions       : {revise}  (a WORKFLOW, not a false refusal)")
    print()
    print("  None of the above is a semantic judgement. Semantic correctness needs")
    print("  the independent blind audit; parser acceptance is not correctness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
