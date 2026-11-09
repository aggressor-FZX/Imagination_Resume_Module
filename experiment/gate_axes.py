#!/usr/bin/env python3
"""Four separate reporting axes for the binding gate. Offline; no provider calls.

Why this module exists
----------------------
The earlier packet reported one thing per cell — `release` — and a composite count.
That conflates four different events, and the conflation is the same error class
this whole line of work exists to remove: a withhold is not a correct detection, a
resolvable citation is not supported evidence, and a repair the scaffold could apply
is not a repair that is right.

So each cell is reported on four axes that are computed independently and are never
added together:

  1. initial_judgment        what the reviewer decided, and whether that agrees with
                             the preregistered expectation. Agreement is NOT proof
                             the objection is correct; it is one mechanical fact.
  2. edit_applicability      could the narrow repair scaffold construct a proposal
                             from this reviewer's answer? This is a fact about the
                             SCAFFOLD, not about the reviewer's judgement.
  3. corrected_document_approval
                             did a re-review approve the EXACT document the gate
                             proposed? `None` when no valid re-review exists — and a
                             retained re-review of a DIFFERENT document is `None`,
                             not an approval and not a refusal.
  4. release                 did the gate release, and on what rule.

The denominator rule (item 2)
-----------------------------
Only contract-compatible PROSE answers with a retained response are in the
candidate's success/failure denominator. A legacy JSON span-envelope answer is a
correct answer to a DIFFERENT contract: the prose gate cannot read it, so it can
neither pass nor fail here. A truncated answer with no response at all is an
operational failure and is also outside the denominator. Both are reported, in their
own columns, with their own counts.

The re-review identity rule (item 7)
------------------------------------
A re-review is a judgement about one exact document. The retained run asked its
re-review about the harness's own `final_markdown`, not about the gate's proposal.
Where those two texts differ, the retained re-review says nothing about the
proposal, and scoring it as approval or refusal would attribute a judgement to the
wrong document. This module therefore computes the identity check before it reports
any re-review result.

Usage:
    python -m experiment.gate_axes --out results/review-packet-v3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiment.binding_candidate import run_binding  # noqa: E402
from experiment.cases_v2 import CASES_V2, SOURCES, by_id  # noqa: E402

RUN = pathlib.Path("results/comparison-v2b-20261109/attempts.json")
DRAFT_MARKER = "DRAFT TO REVIEW:\n"

FORMAT_MISMATCH = "format_mismatch_json_answer_to_prose_gate"


def document_from_prompt(prompt: str) -> str | None:
    """The exact document a review call was given, taken from its own prompt.

    Recovered from the recorded bytes rather than assumed, because the whole point
    of the identity check is that the document a call saw may not be the document a
    later reader expects it saw.
    """
    index = prompt.find(DRAFT_MARKER)
    return prompt[index + len(DRAFT_MARKER):] if index >= 0 else None


def _replay(draft: str, first: str,
            recheck: str = "") -> tuple[object, list[str]]:
    """Run the gate offline, recording the document handed to each call."""
    answers = [first]
    if recheck:
        answers.append(recheck)
    seen: list[str] = []

    async def ask(prompt: str, _sources: dict[str, str]) -> str:
        seen.append(document_from_prompt(prompt) or "")
        if not answers:
            raise RuntimeError("no retained answer for this gate step")
        return answers.pop(0)

    return asyncio.run(run_binding(draft, SOURCES, ask=ask)), seen


def classify_answer(raw: str) -> str:
    """Which contract the answer belongs to, before any parsing happens."""
    if not raw:
        return "no_response"
    if raw.lstrip().startswith("{"):
        return "legacy_json_envelope"
    return "prose"


def build() -> dict:
    attempts = json.loads(RUN.read_text())
    rows: list[dict] = []

    for attempt in attempts:
        case = by_id(attempt["case_id"])
        draft = attempt.get("draft_after_trim") or case.draft_before_trim
        first = attempt.get("provider_response") or ""
        answer_kind = classify_answer(first)
        in_denominator = answer_kind == "prose"

        row: dict = {
            "case_id": attempt["case_id"],
            "cell": attempt["cell"],
            "answer_kind": answer_kind,
            "in_candidate_denominator": in_denominator,
            "expected": case.expectation,
        }

        if not in_denominator:
            # Outside the denominator. Record why, and report nothing on the
            # semantic axes rather than a default of "failed".
            row |= {
                "excluded_reason": ("legacy answer to the JSON span contract; the "
                                    "prose gate cannot read it")
                                   if answer_kind == "legacy_json_envelope"
                                   else "no response retained (truncated)",
                "axis_initial_judgment": None,
                "axis_initial_decision": None,
                "axis_agrees_with_expectation": None,
                "axis_edit_applicability": None,
                "axis_repair_status": None,
                "axis_repair_refusals": [],
                "axis_corrected_document_approval": None,
                "axis_corrected_document_note": "outside the denominator",
                "axis_re_review_document_identity": None,
                "axis_release": None,
                "axis_release_reason": None,
            }
            rows.append(row)
            continue

        recheck = attempt.get("recheck_response") or ""

        # Pass 1: the first answer alone. This settles every decision that does not
        # require a re-review, and it establishes the exact proposal text.
        first_pass, first_seen = _replay(draft, first)
        decision = first_pass.initial_decision or None
        proposal_drawn = first_pass.proposed_document != draft
        result, docs_seen = first_pass, first_seen

        # ── Axis 1: the decision, and whether it matches the preregistration ──
        agrees = (None if decision is None
                  else (decision == "approve") == (case.expectation == "approve"))
        row |= {
            "excluded_reason": None,
            "axis_initial_judgment": decision,
            "axis_initial_decision": decision,
            "axis_agrees_with_expectation": agrees,
        }

        # ── Axis 2: could the scaffold build a proposal at all? ──────────────
        # A single refusal reason per cell: the scaffold stops at the first one.
        refusals = [result.repair_status] if not proposal_drawn and result.repair_status != "not_attempted" else []
        row |= {
            "axis_edit_applicability": "proposal_drawn" if proposal_drawn else "no_proposal",
            "axis_repair_status": result.repair_status,
            "axis_repair_refusals": refusals,
        }

        # ── Axis 3: approval of the EXACT proposed document ──────────────────
        # The retained re-review was asked about the harness's own corrected text,
        # NOT about the gate's proposal. Where the two differ, that answer is a
        # judgement about a different document and is reported as no observation.
        retained_reviewed = None
        if attempt.get("recheck_request"):
            retained_reviewed = document_from_prompt(
                attempt["recheck_request"]["prompt"])
        identity = None
        if retained_reviewed is not None:
            identity = ("same_document"
                        if retained_reviewed == result.proposed_document
                        else "different_document")

        # Every reason that applies is stated, rather than only the first one found.
        # Earlier the "no proposal" branch pre-empted the identity note, which hid
        # the fact that the retained re-review had answered a different document —
        # exactly the attribution error this axis exists to prevent.
        reasons: list[str] = []
        if not proposal_drawn:
            reasons.append("no proposal was drawn, so there was nothing to approve")
        if not recheck:
            reasons.append("no retained re-review exists for this cell")
        if identity == "different_document":
            reasons.append("the retained re-review answered about a DIFFERENT "
                           "document; it cannot validate or invalidate this proposal")

        if not proposal_drawn or not recheck or identity != "same_document":
            approval, note = None, "; ".join(reasons)
        else:
            # Only here is the retained answer a judgement about this proposal.
            result, docs_seen = _replay(draft, first, recheck)
            approval = "approved" if result.release else "not_approved"
            note = "validated: the re-review saw exactly the proposed document"

        row |= {
            "axis_corrected_document_approval": approval,
            "axis_corrected_document_note": note,
            "axis_re_review_document_identity": identity,
        }

        # ── Axis 4: release, and the rule that produced it ───────────────────
        # A gate that never drew a proposal never needed a re-review, so its
        # withhold is complete on the retained evidence. Only a DRAWN proposal
        # whose re-review is missing leaves the rule unfinished, and that is
        # reported as undetermined rather than as a refusal.
        if proposal_drawn and identity == "different_document":
            release, reason = None, "re_review_not_available_for_this_proposal"
        elif not proposal_drawn:
            release, reason = False, result.release_reason
        else:
            release, reason = bool(result.release), result.release_reason
        row |= {"axis_release": release, "axis_release_reason": reason}
        rows.append(row)

    denominator = [r for r in rows if r["in_candidate_denominator"]]

    def count(field: str, source: list[dict]) -> dict:
        return dict(Counter(r[field] for r in source))

    summary = {
        "cells_total": len(rows),
        "denominator_cells": len(denominator),
        "excluded": {
            "legacy_json_envelope": sum(
                1 for r in rows if r["answer_kind"] == "legacy_json_envelope"),
            "no_response": sum(1 for r in rows if r["answer_kind"] == "no_response"),
        },
        "denominator_rule": (
            "Only contract-compatible prose answers with a retained response are in "
            "the success/failure denominator. Legacy JSON envelopes answer a "
            "different contract; a truncated cell has no answer at all."),
        "axis_1_initial_judgment": {
            "decisions": count("axis_initial_decision", denominator),
            "agrees_with_expectation": count("axis_agrees_with_expectation", denominator),
            "caveat": ("Agreement with the preregistered expectation is one "
                       "mechanical fact, not evidence the objection was correct."),
        },
        "axis_2_edit_applicability": {
            "status": count("axis_edit_applicability", denominator),
            "scaffold_refusals": count("axis_repair_status", denominator),
            "caveat": ("A refusal here is a limitation of the narrow repair scaffold, "
                       "not a finding about the reviewer's judgement."),
        },
        "axis_3_corrected_document_approval": {
            "status": count("axis_corrected_document_approval", denominator),
            "re_review_document_identity": count(
                "axis_re_review_document_identity", denominator),
            "caveat": ("A retained re-review of a different document is reported as "
                       "no observation, never as an approval or a refusal."),
        },
        "axis_4_release": {
            "released": sum(1 for r in denominator if r["axis_release"] is True),
            "undetermined": sum(1 for r in denominator if r["axis_release"] is None),
            "withheld": sum(1 for r in denominator if r["axis_release"] is False),
            "reasons": count("axis_release_reason", denominator),
            "caveat": ("Undetermined means the gate could not complete its own rule "
                       "with the evidence retained. It is neither a release nor a "
                       "refusal."),
        },
        "rows": rows,
    }
    return summary


def render(summary: dict) -> str:
    d = summary["denominator_cells"]
    a1 = summary["axis_1_initial_judgment"]
    a2 = summary["axis_2_edit_applicability"]
    a3 = summary["axis_3_corrected_document_approval"]
    a4 = summary["axis_4_release"]

    lines = [
        "## Four axes, reported separately",
        "",
        f"**Candidate denominator: {d} prose cells.** "
        f"{summary['excluded']['legacy_json_envelope']} legacy JSON-envelope cells and "
        f"{summary['excluded']['no_response']} truncated cell are excluded and reported "
        "in their own columns below.",
        "",
        "> " + summary["denominator_rule"],
        "",
        "### Axis 1 — initial judgment",
        "",
        f"- decisions: `{a1['decisions']}`",
        f"- agrees with preregistered expectation: `{a1['agrees_with_expectation']}`",
        f"- *{a1['caveat']}*",
        "",
        "### Axis 2 — edit applicability (a property of the scaffold)",
        "",
        f"- `{a2['status']}`",
        f"- scaffold refusal reasons: `{a2['scaffold_refusals']}`",
        f"- *{a2['caveat']}*",
        "",
        "### Axis 3 — approval of the exact proposed document",
        "",
        f"- `{a3['status']}`",
        f"- re-review document identity: `{a3['re_review_document_identity']}`",
        f"- *{a3['caveat']}*",
        "",
        "### Axis 4 — release",
        "",
        f"- released: **{a4['released']} of {d}**",
        f"- withheld: {a4['withheld']} · undetermined: {a4['undetermined']}",
        f"- reasons: `{a4['reasons']}`",
        f"- *{a4['caveat']}*",
        "",
    ]

    lines += ["| Case | Cell | Decision | Edit | Corrected-doc approval | Release |",
              "|---|---|---|---|---|---|"]
    for row in summary["rows"]:
        if not row["in_candidate_denominator"]:
            continue
        lines.append(
            f"| `{row['case_id']}` | `{row['cell']}` | "
            f"`{row['axis_initial_decision']}` | `{row['axis_repair_status']}` | "
            f"`{row['axis_corrected_document_approval']}` | "
            f"{'**RELEASE**' if row['axis_release'] else 'withhold'} "
            f"(`{row['axis_release_reason']}`) |")
    lines += ["", "| Excluded cell | Kind | Why |", "|---|---|---|"]
    for row in summary["rows"]:
        if row["in_candidate_denominator"]:
            continue
        lines.append(f"| `{row['cell']}`/{row['case_id']} | `{row['answer_kind']}` | "
                     f"{row['excluded_reason']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/review-packet-v3")
    args = ap.parse_args()
    summary = build()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate-axes.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "GATE_AXES.md").write_text(render(summary))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
