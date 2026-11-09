#!/usr/bin/env python3
"""Source-linked judgments: the real defect vs the objection the reviewer wrote.

Why this is not a score
-----------------------
A single number ("the reviewer got 5 of 8") hides the only thing worth knowing:
WHICH defect it named. A reviewer can reach `revise` on a defective document for
the wrong reason, and a reviewer can reach `revise` on a faithful document for an
invented reason. Both are failures a count cannot distinguish, and both are
invisible unless the objection text is quoted next to the ground truth.

So this module prints, per cell, in plain language:

    the declared defect      what the fixture actually changed (from MUTATIONS)
    the ground-truth needle  the literal text that must appear in an objection for
                             it to be a detection of THAT defect
    the objections written   each one, verbatim, with what code could verify
    verdict                  detected / not_detected / asserted_the_opposite
    basis                    the words in the objection that decided the verdict

The needles are literal strings whose presence/absence was verified against the
sources, so `assert_needles()` fails loudly if a fixture ever changes under them.
A `not_detected` verdict is not "the reviewer was wrong to say revise" — the case
is genuinely defective. It means the reviewer did not name the defect that exists.

Usage:
    python -m experiment.judgments                # all prose cells, both arms
    python -m experiment.judgments --out FILE
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import decision_contract as dc  # noqa: E402
from experiment.cases_v2 import MUTATIONS, SOURCES  # noqa: E402

RUN = pathlib.Path("results/comparison-v2b-20261109/attempts.json")

# For each defective case: the literal text a correct objection must mention, plus
# WHERE that text lives. Declaring the location is what makes the check meaningful
# rather than decorative, because the four defect kinds put the same needle in
# opposite places:
#
#   unsourced_method   needle is IN THE DRAFT and NOT in the sources (that is what
#                      makes the clause unsupported)
#   move_fact          needle is in the sources AND still in the draft (under the
#                      wrong employer)
#   delete_bullet      needle is in the sources and NOT in the draft
#   delete_record      needle is in the sources and NOT in the draft
DETECTION = {
    "unsupported_method": {
        "needle": "targeted quality audits",
        "in_sources": False,
        "in_draft": True,
        "why": "names the unsourced method clause",
    },
    "wrong_record_attribution": {
        "needle": "1,180",
        "in_sources": True,
        "in_draft": True,
        "why": "names the figure that was moved to the wrong employer",
    },
    "missing_fact": {
        "needle": "1,180",
        "in_sources": True,
        "in_draft": False,
        "why": "names the deleted quantified bullet",
    },
    "omitted_record": {
        "needle": "Ridgeline",
        "in_sources": True,
        "in_draft": False,
        "why": "names the employer record missing from the draft",
    },
}

# Approve cases: the real defect is nothing. Any issue is an invented objection, and
# the specific inventions are checked here so the claim is not impressionistic.
APPROVE_CASES = ("faithful_paraphrase", "supported_method", "transferable_skill")


def assert_needles() -> None:
    """Verify every needle against the actual fixture. Fails loudly, never silently.

    If a fixture ever changes so that a needle no longer sits where the check
    assumes, every verdict computed from it becomes meaningless — so this raises
    rather than producing a confident-looking report about the wrong text.
    """
    attempts = json.loads(RUN.read_text())
    drafts = {a["case_id"]: a.get("draft_after_trim") or "" for a in attempts}
    for case_id, spec in DETECTION.items():
        needle = spec["needle"].lower()
        in_sources = any(needle in text.lower() for text in SOURCES.values())
        in_draft = needle in drafts.get(case_id, "").lower()
        if in_sources != spec["in_sources"]:
            raise AssertionError(
                f"{case_id}: needle {spec['needle']!r} in sources = {in_sources}, "
                f"expected {spec['in_sources']}")
        if in_draft != spec["in_draft"]:
            raise AssertionError(
                f"{case_id}: needle {spec['needle']!r} in draft = {in_draft}, "
                f"expected {spec['in_draft']}")


def classify(case_id: str, decision: dc.Decision) -> dict:
    """Decide whether the objections name the defect that actually exists."""
    if case_id in APPROVE_CASES:
        if not decision.issues and decision.decision == "approve":
            return {"verdict": "correct_approval", "basis": "no issues raised"}
        passages = [i.passage for i in decision.issues if i.passage]
        unresolvable = [i.passage for i in decision.issues
                        if i.passage and not i.passage_resolves]
        basis = (f"{len(decision.issues)} issue(s) on a document with no declared "
                 f"defect")
        if unresolvable:
            # Quoting text that is not in the draft is the strongest form of
            # invented objection, and it is verifiable rather than a judgement.
            basis += f"; {len(unresolvable)} quote text absent from the draft"
        return {"verdict": "invented_objection", "basis": basis,
                "passages": passages}

    spec = DETECTION[case_id]
    needle = spec["needle"].lower()

    # ORDER MATTERS, and getting it wrong flips a failure into a success.
    #
    # When the defect is that text is MISSING from the draft, an objection whose
    # quoted passage contains that text is not a detection: it is quoting absent
    # text as though it were present. Checking "does the needle appear anywhere in
    # the objection?" first would score exactly that failure as a success, because
    # the needle appears in the miscitation. So the miscitation is tested first.
    asserted_present = [
        i for i in decision.issues
        if spec["in_draft"] is False and i.passage
        and not i.passage_resolves
        and needle in (i.passage or "").lower()]

    # A detection must NAME the defect in the reviewer's own reasoning: the `why`
    # or the `correction`, or a passage that really is in the draft. A properly
    # formed omission claim marks the issue `omitted` and names the missing text in
    # `why`, which is exactly the shape the contract asks for.
    def names_defect(issue) -> bool:
        quote_ok = bool(issue.passage) and issue.passage_resolves
        return (needle in (issue.why or "").lower()
                or needle in (issue.correction or "").lower()
                or (quote_ok and needle in (issue.passage or "").lower()))

    detected = [i for i in decision.issues if names_defect(i)]

    if asserted_present and detected:
        # Both a miscitation AND a grounded claim. A single bucket would either
        # credit the miscitation or discard the real catch; the honest report says
        # the reviewer got there while also inventing a passage.
        return {"verdict": "detected_with_a_miscitation",
                "basis": (f"one objection correctly names {spec['needle']!r} as "
                          "missing, while another quotes it as draft text"),
                "passages": [i.passage for i in asserted_present]}
    if asserted_present:
        return {"verdict": "asserted_the_opposite",
                "basis": (f"quoted {spec['needle']!r} as draft text, but it is "
                          "absent from the draft"),
                "passages": [i.passage for i in asserted_present]}
    if detected:
        return {"verdict": "detected",
                "basis": f"reviewer's reasoning names {spec['needle']!r}",
                "why": spec["why"]}

    return {"verdict": "not_detected",
            "basis": f"no objection mentions {spec['needle']!r}; the defect exists",
            "issues_raised": [(i.problem, (i.passage or "")[:80])
                              for i in decision.issues]}


def build() -> dict:
    assert_needles()
    attempts = json.loads(RUN.read_text())
    rows = []
    for attempt in attempts:
        raw = attempt.get("provider_response") or ""
        kind = ("no_response" if not raw
                else "legacy_json_envelope" if raw.lstrip().startswith("{")
                else "prose")
        if kind != "prose":
            continue
        draft = attempt.get("draft_after_trim") or ""
        decision = dc.decide_from_text(raw, draft, SOURCES)
        verdict = classify(attempt["case_id"], decision)
        rows.append({
            "case_id": attempt["case_id"],
            "cell": attempt["cell"],
            "declared_defect": MUTATIONS[attempt["case_id"]]["kind"],
            "declared_detail": MUTATIONS[attempt["case_id"]]["detail"],
            "decision": decision.decision or None,
            "issue_count": len(decision.issues),
            "objections": [{"problem": i.problem, "passage": i.passage,
                            "why": i.why, "correction": i.correction,
                            "passage_resolves": i.passage_resolves,
                            "sources_resolve": i.sources_resolve}
                           for i in decision.issues],
            "verdict": verdict["verdict"],
            "basis": verdict["basis"],
        })
    return {"rows": rows, "summary": _summarise(rows)}


def _summarise(rows: list[dict]) -> dict:
    by_cell: dict[str, dict[str, str]] = {}
    for row in rows:
        by_cell.setdefault(row["cell"], {})[row["case_id"]] = row["verdict"]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    return {
        "prose_cells": len(rows),
        "verdict_counts": counts,
        "by_cell": by_cell,
        "caveat": ("A `detected` verdict means the objection NAMED the defect. It "
                   "does not mean the proposed correction was safe to apply, nor "
                   "that the corrected document would be approvable. Those are the "
                   "separate axes in gate_axes."),
    }


def render(payload: dict) -> str:
    lines = ["# Source-linked judgments", "",
             "Each row quotes the objection the reviewer wrote against the defect "
             "the fixture actually contains. The verdict is about NAMING, not about "
             "whether the correction would be safe to apply.", "",
             "| case | cell | declared defect | decision | verdict | basis |",
             "|---|---|---|---|---|---|"]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['case_id']}` | `{row['cell']}` | `{row['declared_defect']}` | "
            f"`{row['decision']}` | **{row['verdict']}** | "
            f"{row['basis'].replace('|', chr(92) + '|')[:170]} |")

    lines += ["", "## The objections themselves", ""]
    for row in payload["rows"]:
        lines += [f"### `{row['case_id']}` / `{row['cell']}` — "
                  f"**{row['verdict']}**", "",
                  f"Declared defect: `{row['declared_defect']}` — "
                  f"{row['declared_detail']}", ""]
        if not row["objections"]:
            lines += ["_No issues were reported._", ""]
            continue
        lines += ["| # | problem | passage (as quoted) | in draft? | why |",
                  "|---|---|---|---|---|"]
        for number, issue in enumerate(row["objections"], 1):
            passage = (issue["passage"] or "—").replace("|", "\\|").replace("\n", " ")
            why = (issue["why"] or "—").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {number} | `{issue['problem']}` | {passage[:150]} | "
                         f"{'yes' if issue['passage_resolves'] else '**NO**'} | "
                         f"{why[:180]} |")
        lines.append("")
    lines += ["## Summary", "",
              f"Prose cells judged: {payload['summary']['prose_cells']}", "",
              "| verdict | count |", "|---|---|"]
    for verdict, count in sorted(payload["summary"]["verdict_counts"].items()):
        lines.append(f"| `{verdict}` | {count} |")
    lines += ["", f"> {payload['summary']['caveat']}", ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    payload = build()
    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(payload))
        print(json.dumps(payload["summary"], indent=2))
    else:
        print(render(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
