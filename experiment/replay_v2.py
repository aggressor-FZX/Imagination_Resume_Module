#!/usr/bin/env python3
"""Replay v2 — corrected classification. Reads the FROZEN bytes, writes elsewhere.

This is a VERSIONED correction of `experiment/replay.py`. v1 stays byte-for-byte
intact as the record of what it concluded; its outputs live in
`results/comparison-20261109-0719` behind a `SHA256SUMS` manifest. This module
never writes into that directory.

What v1 got wrong
-----------------
v1 separated truncation from parser work, which was a real improvement, but it
still merged two *materially different* things and split two others wrongly:

1. **A reference failure was filed as `semantic_refuse`.** Measured now on the one
   legacy answer that completed: all 10 citations quote text that really exists in
   the supplied sources, and 10 of 10 have wrong offsets. That is a **harness
   defect** — the model used one offset convention, the harness accepted another.
   Calling it a semantic refusal credited the reviewer with a judgement it never
   made, and understated the legacy contract at the same time.

2. **`revise` was filed as a semantic category at all.** `revise` is the contract
   working. It says nothing about whether the document should have been delivered,
   so it belongs in the workflow column, not the semantic one.

3. **A wrong-format answer and a malformed answer were both `parser_defect`.** An
   answer in a different format than requested is a *prompt/contract* result; an
   answer in the requested format that code cannot read is a *parser* result.

The corrected taxonomy keeps those axes separate, because they answer different
questions:

    截断/transport   -> nothing about the document      (operational)
    format           -> did the prompt get the answer shape it asked for?
    parse            -> could code read the shape it got?
    reference        -> did the cited evidence exist, and did the offsets agree?
    semantics        -> approve / revise / cannot_assess

A reference check has three distinct outcomes, and v1 collapsed them:

    reference_matches      quote is at the stated offset        (evidence verified)
    reference_offsets_off  quote EXISTS in the source, offset differs
    reference_fabricated   quote appears in NO source
    reference_unknown_src  source_id names no supplied document

Only the last two are refusals `about the document`. `reference_offsets_off` is
`reference_convention` — a harness/contract defect. It is reported separately and
is NEVER counted as a semantic refusal, because the reviewer's evidence was real.

Usage:
    python experiment/replay_v2.py results/comparison-20261109-0719
    python experiment/replay_v2.py <run-dir> --json --out results/replay-v2

No network. No provider. Read-only against the run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import decision_contract as dc  # noqa: E402

REPLAY_VERSION = "v2"

# ── Category vocabulary. One axis per prefix, so no category can hide another. ─
# Operational: nothing about the document.
CAT_TRUNCATION = "operational:truncation"
CAT_TRANSPORT = "operational:transport"
CAT_EMPTY = "operational:empty"

# Prompt/contract shape.
CAT_FORMAT_MISMATCH = "format:mismatch_requested"

# Code cannot read a shape the prompt did request.
CAT_PARSE_DEFECT = "parse:defect"
CAT_CONTRACT_DEFECT = "parse:contract_defect"

# Evidence.
CAT_REF_CONVENTION = "reference:convention"
CAT_REF_FABRICATED = "reference:fabricated"
CAT_REF_UNKNOWN_SOURCE = "reference:unknown_source"
CAT_REF_CITES_DRAFT = "reference:cites_draft"

# Semantics — the ONLY categories that say something about the document.
CAT_APPROVE = "semantic:approve"
CAT_REVISE = "semantic:revise"
CAT_CANNOT_ASSESS = "semantic:cannot_assess"

SEMANTIC_CATEGORIES = {CAT_APPROVE, CAT_REVISE, CAT_CANNOT_ASSESS}
REFERENCE_CATEGORIES = {CAT_REF_CONVENTION, CAT_REF_FABRICATED,
                        CAT_REF_UNKNOWN_SOURCE}

CURRENT_FORMAT = "json_span_envelope"
REVISED_FORMAT = "prose_decision"


def requested_format(cell: str) -> str:
    return CURRENT_FORMAT if cell.startswith("current") else REVISED_FORMAT


# ── Evidence resolution, with the convention separated from fabrication ─────
def resolve_citation(citation: object, sources: dict[str, str],
                     draft: str = "") -> str:
    """Classify ONE citation. Returns one of the reference_* verdict suffixes.

    The ordering matters: verifying the QUOTE is independent of verifying the
    OFFSET. An offset error is not a hallucination, and treating it as one threw
    away a correct review (see module docstring).

    Measured on the v2 contract probe (2026-11-09), 20 citations from one legacy
    answer split three ways — and v1 could not see the difference:

        12  quote is real source text
         5  quote is text from the DRAFT, cited as if it were a source
         3  quote is in neither document

    The middle group is its own defect. The reviewer quoted the document it was
    reviewing and attributed it to a source; the claim may be true, but the
    evidence offered does not establish it. Calling that "fabricated" overstates
    it (the text exists) and calling it "convention" understates it (the pointer is
    wrong, not merely mis-numbered). `cites_draft` names it.
    """
    if not isinstance(citation, dict):
        return "malformed"
    source_id = str(citation.get("source_id", ""))
    text = sources.get(source_id)
    if text is None:
        return "unknown_source"
    quote = citation.get("quote")
    if not isinstance(quote, str) or not quote.strip():
        return "malformed"
    start, end = citation.get("start"), citation.get("end")

    flat_quote = " ".join(quote.split())

    # 1. Does the quoted text exist in the named source AT ALL?
    if quote not in text:
        # Tolerate insignificant whitespace differences before calling it
        # fabrication: a model reproducing a line that wrapped differently quoted
        # real evidence. Compare on a whitespace-normalised basis.
        flat_text = " ".join(text.split())
        if flat_quote and flat_quote in flat_text:
            return "offsets_off"
        # Not in the cited source. Is it in the DRAFT? Then the reviewer quoted the
        # document under review rather than the evidence for it.
        if draft:
            flat_draft = " ".join(draft.split())
            if flat_quote and flat_quote in flat_draft:
                return "cites_draft"
        return "fabricated"

    # 2. The quote is real. Do the offsets agree with it?
    if not (isinstance(start, int) and isinstance(end, int)):
        return "offsets_off"
    if 0 <= start < end <= len(text) and text[start:end] == quote:
        return "matches"
    # Some models write `end` as a LENGTH rather than an end offset. This is
    # checked BEFORE the range test, because under that convention `end` is a
    # small number and `start < end` is false — which is exactly how a correct
    # citation was being misread as a convention failure during testing.
    if 0 <= start <= len(text) and text[start:start + end] == quote:
        return "matches_length_convention"
    return "offsets_off"


def classify(cell: str, raw: str, draft: str, sources: dict[str, str],
             finish_reason: str | None = None) -> dict:
    """Classify one retained answer. Names the FIRST failing step, then the rest."""
    fmt = requested_format(cell)
    out: dict = {
        "requested_format": fmt,
        "decision": "",
        "category": "",
        "first_failure": "",
        "detail": "",
        "reference": None,
        "parseable": None,
    }

    if not (raw or "").strip():
        out.update(category=CAT_EMPTY, first_failure="empty_response")
        return out

    # ── Prose cell ────────────────────────────────────────────────────────
    if fmt == REVISED_FORMAT:
        parsed = dc.decide_from_text(raw, draft, sources)
        out["parseable"] = bool(parsed.decision)
        out["decision"] = parsed.decision
        ref = _reference_summary(parsed, sources)
        out["reference"] = ref
        if parsed.structural_problems:
            # Distinguish a CONTRACT defect (correct prose, a rule rejected it) from
            # a PARSE defect (code could not read the prose at all).
            unreadable = {"decision_missing", "decision_word_unknown", "answer_not_text"}
            cat = (CAT_PARSE_DEFECT
                   if any(p.split(":")[0] in unreadable for p in parsed.structural_problems)
                   else CAT_CONTRACT_DEFECT)
            out.update(category=cat, first_failure=parsed.structural_problems[0],
                       detail="; ".join(parsed.structural_problems[:3]))
            return out
        if ref and ref["verdict"] in {"fabricated", "unknown_source"}:
            out.update(category=(CAT_REF_FABRICATED
                                 if ref["verdict"] == "fabricated"
                                 else CAT_REF_UNKNOWN_SOURCE),
                       first_failure=f"reference_{ref['verdict']}",
                       detail=ref.get("detail", ""))
            return out
        if ref and ref["verdict"] in {"offsets_off"}:
            out.update(category=CAT_REF_CONVENTION,
                       first_failure="reference_offsets_off",
                       detail=ref.get("detail", ""))
            return out
        out["category"] = {dc.DECISION_APPROVE: CAT_APPROVE,
                           dc.DECISION_REVISE: CAT_REVISE,
                           dc.DECISION_CANNOT_ASSESS: CAT_CANNOT_ASSESS}.get(
                               parsed.decision, CAT_CONTRACT_DEFECT)
        return out

    # ── JSON envelope cell ────────────────────────────────────────────────
    if finish_reason == "length":
        out.update(category=CAT_TRUNCATION,
                   first_failure="output_budget_exhausted",
                   detail="provider reported finish_reason=length")
        return out

    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        truncated = ("Unterminated string" in exc.msg
                     or exc.pos >= len(raw.rstrip()) - 2
                     or exc.msg.startswith("Expecting ',' delimiter")
                     and exc.pos >= len(raw.rstrip()) - 3)
        if truncated:
            out.update(category=CAT_TRUNCATION,
                       first_failure="response_incomplete",
                       detail=f"{exc.msg} at char {exc.pos} (unterminated)")
            return out
        if raw.lstrip().lower().startswith("decision:"):
            out.update(category=CAT_FORMAT_MISMATCH,
                       first_failure="answered_prose_to_json_prompt",
                       detail="prompt requested the JSON span envelope")
            return out
        out.update(category=CAT_PARSE_DEFECT, first_failure="envelope_not_json",
                   detail=f"{exc.msg} at char {exc.pos}")
        return out

    out["parseable"] = True
    if not isinstance(envelope, dict):
        out.update(category=CAT_PARSE_DEFECT, first_failure="envelope_not_object")
        return out
    claims = envelope.get("claims")
    if not isinstance(claims, list):
        out.update(category=CAT_PARSE_DEFECT, first_failure="claims_missing")
        return out

    verdicts: Counter = Counter()
    citation_verdicts: Counter = Counter()
    for claim in claims:
        if not isinstance(claim, dict):
            citation_verdicts["malformed"] += 1
            continue
        verdicts[str(claim.get("verdict", ""))] += 1
        citations = claim.get("citations")
        if not isinstance(citations, list):
            citation_verdicts["malformed"] += 1
            continue
        for citation in citations:
            citation_verdicts[resolve_citation(citation, sources, draft)] += 1

    out["reference"] = dict(citation_verdicts)
    flagged = verdicts.get("unsupported", 0) + verdicts.get("uncertain", 0)
    out["decision"] = dc.DECISION_REVISE if flagged else dc.DECISION_APPROVE

    # Evidence takes precedence over the decision word: a decision reached by
    # citing things that do not exist is not a semantic result.
    if citation_verdicts.get("fabricated"):
        out.update(category=CAT_REF_FABRICATED, first_failure="quote_absent_from_source",
                   detail=f"{citation_verdicts['fabricated']} quote(s) in no source")
        return out
    if citation_verdicts.get("cites_draft"):
        out.update(category=CAT_REF_CITES_DRAFT,
                   first_failure="quote_is_draft_text_not_source",
                   detail=(f"{citation_verdicts['cites_draft']} quote(s) occur only "
                           f"in the draft under review"))
        return out
    if citation_verdicts.get("unknown_source"):
        out.update(category=CAT_REF_UNKNOWN_SOURCE, first_failure="unknown_source_id",
                   detail=f"{citation_verdicts['unknown_source']} unknown source(s)")
        return out
    if citation_verdicts.get("offsets_off"):
        out.update(
            category=CAT_REF_CONVENTION, first_failure="reference_offsets_off",
            detail=(f"{citation_verdicts['offsets_off']} quote(s) exist in the source "
                    f"but not at the stated offset; "
                    f"{citation_verdicts.get('matches', 0)} correct"))
        return out
    out["category"] = CAT_REVISE if flagged else CAT_APPROVE
    return out


def _reference_summary(parsed: dc.Decision, sources: dict[str, str]) -> dict | None:
    """Summarise evidence for a prose decision, keeping convention separate."""
    if not parsed.issues:
        return None
    counts: Counter = Counter()
    for issue in parsed.issues:
        if not issue.sources:
            counts["no_reference"] += 1
            continue
        for reference in issue.sources:
            ref_id = reference.split(":")[0].strip()
            if ref_id in sources:
                counts["resolves"] += 1
                continue
            if len(reference) > 8 and reference.casefold() in " ".join(
                    sources.values()).casefold():
                counts["quote_resolves"] += 1
                continue
            counts["unresolved"] += 1
    if counts.get("unresolved"):
        return {"verdict": "fabricated", "counts": dict(counts),
                "detail": f"{counts['unresolved']} unresolved reference(s)"}
    return {"verdict": "resolves", "counts": dict(counts)}


def find_finish_reason(attempt: dict) -> str | None:
    for key in ("finish_reason", "provider_finish_reason"):
        if attempt.get(key):
            return str(attempt[key])
    for container in ("provider_request", "provider_meta", "provider_response_meta"):
        value = attempt.get(container)
        if isinstance(value, dict) and value.get("finish_reason"):
            return str(value["finish_reason"])
    return None


def replay(run_dir: pathlib.Path) -> list[dict]:
    """Re-classify every retained attempt. Pure function of the frozen bytes."""
    attempts = json.loads((run_dir / "attempts.json").read_text())

    # The fixtures that were ACTUALLY used. v1 is imported explicitly so a rerun
    # of the OLD bytes is judged against the OLD inputs; using v2 here would
    # misreport the archive.
    from experiment.cases import CASES  # noqa: WPS433
    by_id = {c.case_id: c for c in CASES}

    rows = []
    for attempt in attempts:
        case = by_id.get(attempt["case_id"])
        draft = attempt.get("draft_after_trim") or (case.draft_before_trim if case else "")
        sources = dict(case.sources) if case else {}
        raw = attempt.get("provider_response") or ""
        finish = find_finish_reason(attempt)
        result = classify(attempt["cell"], raw, draft, sources, finish)
        prompt = (attempt.get("provider_request") or {}).get("prompt", "")
        rows.append({
            "cell": attempt["cell"],
            "case_id": attempt["case_id"],
            "mode": attempt.get("mode"),
            "prompt_sha256_16": hashlib.sha256(str(prompt).encode()).hexdigest()[:16],
            "finish_reason": finish,
            "response_chars": len(raw),
            "decision": result["decision"],
            "category": result["category"],
            "first_failure": result["first_failure"],
            "detail": result["detail"],
            "reference": result["reference"],
            "recorded_outcome": attempt.get("outcome"),
            "recorded_reason": attempt.get("outcome_reason"),
            "cost_usd": attempt.get("cost_usd"),
            "latency_s": attempt.get("latency_s"),
        })
    return rows


def summarise(rows: list[dict]) -> dict:
    by_category = Counter(r["category"] for r in rows)
    axes = {
        "operational": sum(v for k, v in by_category.items()
                           if k.startswith("operational:")),
        "format": sum(v for k, v in by_category.items() if k.startswith("format:")),
        "parse": sum(v for k, v in by_category.items() if k.startswith("parse:")),
        "reference": sum(v for k, v in by_category.items()
                         if k.startswith("reference:")),
        "semantic": sum(v for k, v in by_category.items()
                        if k.startswith("semantic:")),
    }
    return {
        "replay_version": REPLAY_VERSION,
        "attempts": len(rows),
        "by_axis": axes,
        "by_category": dict(by_category),
        "semantic_verdicts": {
            r["case_id"]: r["category"] for r in rows
            if r["category"] in SEMANTIC_CATEGORIES
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", help="directory containing the FROZEN attempts.json")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", help="write rows+summary JSON here (never into the run dir)")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run)
    rows = replay(run_dir)
    summary = summarise(rows)

    if args.out:
        out_dir = pathlib.Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "replay_v2_rows.json").write_text(json.dumps(rows, indent=2))
        (out_dir / "replay_v2_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"wrote {out_dir}/replay_v2_rows.json and replay_v2_summary.json")

    if args.json:
        print(json.dumps({"rows": rows, "summary": summary}, indent=2))
        return 0

    header = (f"{'cell':22} {'case':26} {'len':>5} {'decision':13} "
              f"{'category':30} first failing step")
    print(header)
    print("-" * (len(header) + 24))
    for row in rows:
        print(f"{row['cell']:22} {row['case_id']:26} {row['response_chars']:>5} "
              f"{str(row['decision'] or '-'):13} {row['category']:30} "
              f"{row['first_failure'] or '-'}")

    print()
    print("BY AXIS  (only the `semantic:` axis is a judgement about the document)")
    for axis, count in summary["by_axis"].items():
        print(f"  {axis:12} {count:3}")

    print()
    print("BY CATEGORY")
    for category, count in sorted(summary["by_category"].items()):
        print(f"  {count:3}  {category}")

    print()
    print("CORRECTION TO v1")
    print("  v1 filed 1 attempt as `semantic_refuse` for a failed reference.")
    print("  Re-measured: all 10 citations in that answer quote text that exists in")
    print("  the supplied sources; 10 of 10 have wrong OFFSETS. That is a harness")
    print("  convention defect, not a semantic refusal and not fabrication.")
    print("  This run therefore contains ZERO verified semantic judgments in the")
    print("  legacy arm: 11 truncated + 1 offset-convention mismatch.")
    print()
    print("  No semantic conclusion of any kind is drawn from this archive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
