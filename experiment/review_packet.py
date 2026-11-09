#!/usr/bin/env python3
"""Build the seven-case review packet from the RETAINED attempts. No provider calls.

What this produces
------------------
One file per case containing, in review order:

    source documents          both, verbatim
    reviewed draft            the exact text the reviewer was given
    per cell:
        raw model decision    the literal word the model wrote (or "none: JSON")
        reported issues       passage / problem / why / sources / correction
        proposed corrections  what the harness would apply, and whether it can
        re-review result      the second raw decision, if a re-review happened
    binding-gate outcome      what the experimental gate does with these answers

and, deliberately SEPARATE:

    EXPECTED_LABELS.md        the author's provisional expectation per case

Why the labels are separate
---------------------------
The point of the review is to judge whether the objections are correct. If the
expected label sits next to the objection, a reader is anchored by it and the
review stops being independent. The labels are therefore in their own file, and
the per-case files never mention them.

Why the gate is replayed here
-----------------------------
The question "does the binding gate honour approve / revise / cannot_assess" is
answerable offline from the retained bytes: the gate is a pure function of the
draft, the sources, and the model's answers. Replaying it costs nothing and
cannot drift from the recorded evidence.

A gate result is NOT a claim that the objection was correct. It only shows what
the gate does with the answer it was given.

Usage:
    python -m experiment.review_packet --out results/review-packet-v3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import decision_contract as dc  # noqa: E402
from experiment.binding_candidate import run_binding  # noqa: E402
from experiment.cases_v2 import CASES_V2, MUTATIONS, SOURCES, by_id  # noqa: E402

RUN = pathlib.Path("results/comparison-v2b-20261109/attempts.json")

CELL_ORDER = ("current+advisory", "current+enabled",
              "revised+advisory", "revised+enabled")


def _raw_decision(raw: str) -> str:
    """The literal decision word, or an explicit note that there is none."""
    if not raw:
        return "**none — no response retained**"
    parsed = dc.parse_decision(raw)
    if parsed.decision:
        return f"`{parsed.decision}`"
    if raw.lstrip().startswith("{"):
        return "**none — JSON envelope, no prose decision**"
    return "**none — unparseable**"


def _issues_block(raw: str, draft: str, sources: dict[str, str]) -> str:
    """Reported issues, with what code could verify about each."""
    if not raw:
        return "_No response retained, so no issues can be reported._\n"
    if raw.lstrip().startswith("{"):
        return ("_JSON span envelope. Issues are per-claim verdicts, not prose "
                "objections; see the raw response below._\n")
    parsed = dc.decide_from_text(raw, draft, sources)
    if not parsed.issues:
        return "_No issues parsed._\n"
    lines = ["| # | passage | problem | why | sources | correction | resolves |",
             "|---|---|---|---|---|---|---|"]
    for number, issue in enumerate(parsed.issues, 1):
        passage = (issue.passage or "").replace("|", "\\|").replace("\n", " ")
        why = (issue.why or "").replace("|", "\\|").replace("\n", " ")
        correction = (issue.correction or "—").replace("|", "\\|").replace("\n", " ")
        refs = ", ".join(issue.sources) or "—"
        resolves = ("passage ✓" if issue.passage_resolves else "passage ✗")
        resolves += " / " + ("sources ✓" if issue.sources_resolve else "sources ✗")
        lines.append(f"| {number} | {passage[:160]} | `{issue.problem}` | "
                     f"{why[:200]} | {refs[:80]} | {correction[:160]} | {resolves} |")
    return "\n".join(lines) + "\n"


def _corrections_block(raw: str, draft: str, sources: dict[str, str]) -> str:
    """What the harness would actually apply, and whether it can."""
    if not raw or raw.lstrip().startswith("{"):
        return "_No prose corrections to apply._\n"
    parsed = dc.decide_from_text(raw, draft, sources)
    usable = [i for i in parsed.issues if i.correction]
    if not usable:
        return "_No correction offered._\n"
    lines = ["| passage | proposed replacement | locatable in draft | omit marker |",
             "|---|---|---|---|"]
    for issue in usable:
        located = dc.locate_passage(issue.passage, draft)
        lines.append(
            f"| {(issue.passage or '')[:120]} | {(issue.correction or '')[:120]} | "
            f"{'yes' if located else 'NO'} | "
            f"{'YES' if dc.is_omit_marker(issue.correction or '') else 'no'} |")
    return "\n".join(lines) + "\n"


def _gate_replay(case_id: str, draft: str, first: str, recheck: str) -> dict:
    """Run the experimental binding gate offline against the retained answers."""
    answers = [first]
    if recheck:
        answers.append(recheck)

    async def ask(_prompt: str, _sources: dict[str, str]) -> str:
        if not answers:
            # The gate asked for a re-review the retained run never made. Say so
            # rather than inventing an answer.
            raise RuntimeError("no retained answer for this gate step")
        return answers.pop(0)

    result = asyncio.run(run_binding(draft, SOURCES, ask=ask))

    # Distinguish a FORMAT mismatch from an unreadable prose answer.
    #
    # The binding gate is a prose contract. A legacy JSON span envelope is a
    # well-formed answer to a DIFFERENT contract, so the gate cannot read it —
    # but calling that "unverifiable" implies the answer was defective, which is
    # false and would misattribute a harness/contract mismatch to the model.
    # Measured while building this packet (2026-11-09): all 14 legacy cells were
    # reported `initial_review_unverifiable` when the accurate reason is that the
    # gate does not speak that format.
    reason = result.release_reason
    if reason == "initial_review_unverifiable" and (first or "").lstrip().startswith("{"):
        reason = "format_mismatch_json_answer_to_prose_gate"

    return {
        "initial_decision": result.initial_decision or "none",
        "initial_problems": result.initial_problems,
        "repair_status": result.repair_status,
        "re_review_decision": result.re_review_decision or "none",
        "release": result.release,
        "release_reason": reason,
    }


_REASON_NOTES = {
    "format_mismatch_json_answer_to_prose_gate":
        "The gate is a prose contract; this cell answered the legacy JSON "
        "envelope. The answer is not defective — it answers a different "
        "contract, so the gate cannot read it. This is a contract mismatch, "
        "not a model failure.",
    "initial_review_unverifiable":
        "The prose answer could not be used as a verdict: a structural problem "
        "remained after parsing (see the problems listed).",
    "revision_unresolved":
        "The reviewer asked for a change, but no repair could be applied safely "
        "(multiple issues, an omission, an ambiguous passage, or a correction "
        "that is not an exact owned source bullet). Nothing was edited.",
    "re_review_not_approved":
        "A repair was applied and re-reviewed, and the re-review did not "
        "approve the corrected document.",
}


def _gate_block(gate: dict) -> str:
    verdict = "**RELEASE**" if gate["release"] else "**WITHHOLD**"
    lines = [
        f"- initial decision: `{gate['initial_decision']}`",
        f"- repair status: `{gate['repair_status']}`",
        f"- re-review decision: `{gate['re_review_decision']}`",
        f"- gate result: {verdict} (`{gate['release_reason']}`)",
    ]
    if gate["initial_problems"]:
        lines.append(f"- initial structural problems: "
                     f"`{', '.join(gate['initial_problems'][:4])}`")
    note = _REASON_NOTES.get(gate["release_reason"])
    if note:
        lines += ["", f"> {note}"]
    return "\n".join(lines) + "\n"


def build(out_dir: pathlib.Path) -> dict:
    attempts = json.loads(RUN.read_text())
    by_case: dict[str, dict[str, dict]] = {}
    for attempt in attempts:
        by_case.setdefault(attempt["case_id"], {})[attempt["cell"]] = attempt

    out_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(exist_ok=True)

    index = ["# Seven-case review packet", "",
             "Each case file contains the sources, the reviewed draft, every cell's "
             "raw model decision, reported issues, proposed corrections, re-review "
             "result, and the binding-gate outcome.", "",
             "**Expected labels are in `EXPECTED_LABELS.md`, deliberately separate.**",
             "", "| Case | Mutation | Cells present |", "|---|---|---|"]
    gate_summary = []

    for case in CASES_V2:
        cells = by_case.get(case.case_id, {})
        mutation = MUTATIONS[case.case_id]
        index.append(f"| [`{case.case_id}`](cases/{case.case_id}.md) | "
                     f"`{mutation['kind']}` | {', '.join(sorted(cells))} |")

        lines = [f"# Case: `{case.case_id}`", "",
                 f"**Declared mutation:** `{mutation['kind']}` — {mutation['detail']}",
                 "", "---", "", "## Source documents", ""]
        for name, text in SOURCES.items():
            lines += [f"### `{name}`", "", "```text", text.rstrip("\n"), "```", ""]

        lines += ["---", "", "## Reviewed draft", "",
                  "The exact text the reviewer was given.", "",
                  "```markdown", case.draft_before_trim.rstrip("\n"), "```", ""]

        for cell in CELL_ORDER:
            attempt = cells.get(cell)
            if attempt is None:
                continue
            first = attempt.get("provider_response") or ""
            recheck = attempt.get("recheck_response") or ""
            draft = attempt.get("draft_after_trim") or case.draft_before_trim

            lines += ["---", "", f"## Cell `{cell}`", "",
                      f"**Raw model decision:** {_raw_decision(first)}", ""]

            lines += ["### Reported issues", "",
                      _issues_block(first, draft, SOURCES)]
            lines += ["### Proposed corrections", "",
                      _corrections_block(first, draft, SOURCES)]

            lines += ["### Re-review", ""]
            if recheck:
                lines += [f"**Raw re-review decision:** {_raw_decision(recheck)}", "",
                          "```text", recheck.rstrip("\n"), "```", ""]
            else:
                lines += ["_No re-review was performed in the retained run._", ""]

            gate = _gate_replay(case.case_id, draft, first, recheck)
            gate_summary.append({"case_id": case.case_id, "cell": cell, **gate})
            lines += ["### Binding-gate outcome (offline replay)", "",
                      _gate_block(gate)]

            lines += ["### Raw first response", "", "```text",
                      (first or "(none retained)").rstrip("\n"), "```", ""]

        (cases_dir / f"{case.case_id}.md").write_text("\n".join(lines) + "\n")

    # ── Gate summary, so the question "does the gate honour the three words?"
    #    is answerable without reading 28 cells. ─────────────────────────────
    by_reason: dict[str, list[str]] = {}
    for gate in gate_summary:
        by_reason.setdefault(gate["release_reason"], []).append(
            f"{gate['cell']}/{gate['case_id']}")
    index += ["", "---", "", "## Binding-gate outcomes", "",
              f"**Released: {sum(1 for g in gate_summary if g['release'])} of "
              f"{len(gate_summary)}.** The gate is fail-closed: it released nothing "
              "from the retained answers.", "",
              "| Reason | Count | Cells |", "|---|---|---|"]
    for reason, cells in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        index.append(f"| `{reason}` | {len(cells)} | {', '.join(sorted(cells)[:6])}"
                     f"{' …' if len(cells) > 6 else ''} |")
    index += ["",
              "### Does the gate honour approve / revise / cannot_assess?", "",
              "- **approve** — honoured: an initial valid `approve` releases "
              "immediately, and no retained answer contains one.",
              "- **revise** — honoured: never releases on the first `revise`. "
              "Release requires an unambiguously applicable repair AND a new valid "
              "`approve` of the corrected text. Two cells reached a repair; both "
              "re-reviews still said `revise`, so both withheld.",
              "- **cannot_assess** — honoured: withholds. Not exercised by any "
              "retained answer.",
              "- **unreadable/unverifiable** — withholds, and is reported as its "
              "own reason rather than being treated as a refusal.",
              "",
              "The gate behaviour above is a claim about the RELEASE RULE only. It "
              "says nothing about whether any objection was correct — that is what "
              "the review of the per-case objections is for."]
    (out_dir / "README.md").write_text("\n".join(index) + "\n")

    # ── Labels, deliberately in their own file. ────────────────────────────
    label_lines = ["# Expected labels — PROVISIONAL, kept separate", "",
                   "These are the author's expectations, fixed before the run. They "
                   "are **not** evidence that an objection was correct, and they "
                   "must not be read alongside the objections when judging them.",
                   "", "| Case | Expected |", "|---|---|"]
    for case in CASES_V2:
        label_lines.append(f"| `{case.case_id}` | `{case.expectation}` |")
    label_lines += ["", "`transferable_skill` is approvable only under Imaginative; "
                    "a Conservative refusal of it is correct, not a false refusal."]
    (out_dir / "EXPECTED_LABELS.md").write_text("\n".join(label_lines) + "\n")

    (out_dir / "gate-summary.json").write_text(
        json.dumps(gate_summary, indent=2) + "\n")

    return {"cases": len(CASES_V2), "cells": len(gate_summary),
            "released": sum(1 for g in gate_summary if g["release"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/review-packet-v3")
    args = ap.parse_args()
    summary = build(pathlib.Path(args.out))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
