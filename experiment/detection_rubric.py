#!/usr/bin/env python3
"""Detection graded on five criteria, not on whether a needle was mentioned.

Why the needle check was not enough
-----------------------------------
`experiment/judgments.py` asks "does the objection text contain this string?". That
is a LOCATOR: it finds passages worth reading. It cannot distinguish:

    "the 24-hour reporting bullet is missing"          -> correct
    "the 24-hour reporting bullet is wrong"            -> wrong claim about the draft
    "the 24-hour bullet should move to Cascade"        -> right fact, wrong owner
    "the 24-hour window is fine but 2025 is suspect"   -> right fact, irrelevant defect

All four contain the needle. Only the first is correct detection. So each defective
cell is graded on the five things a correct detection has to establish:

    1. FACT      the relevant fact is identified
    2. IN_SOURCE is it in the source documents?      (and the answer is stated)
    3. IN_DRAFT  is it in the reviewed draft?        (and the answer is stated)
    4. OWNER     which employer/project owns it
    5. CORRECTION what actually needs correcting

Criteria 2, 3 and 5 are graded by looking for an ASSERTION of the right proposition,
not merely for a keyword. A reviewer that says "the draft includes X" and a reviewer
that says "the draft omits X" both mention X; only one is right for a given case.

This module is deliberately conservative: where a criterion cannot be established
mechanically it returns `None` (undetermined) rather than guessing. An undetermined
criterion is not a pass.

Usage:
    python -m experiment.detection_rubric                     # retained matrix
    python -m experiment.detection_rubric --run FILE --out OUT
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import decision_contract as dc  # noqa: E402

# ── Ground truth per defective case ───────────────────────────────────────────
#
# `fact`       literal text identifying the fact (used for criterion 1)
# `in_source`  the true answer for criterion 2
# `in_draft`   the true answer for criterion 3
# `owner`      the employer that owns the fact
# `wrong_owner` the employer it must NOT be attributed to (wrong_record only)
# `fix`        what a correct correction does
#
# Ownership and draft-presence are the two things the needle check could not see.
TRUTH: dict[str, dict] = {
    # ── familiar (cases_v2) ──────────────────────────────────────────────────
    "unsupported_method": {
        "owner_required": False,
        "fact": ["targeted quality audits"],
        "in_source": False, "in_draft": True,
        "owner": "Cascade Freight Partners",
        "fix": "remove_clause",
        "fix_words": ["remov", "delet", "drop", "strip", "without"],
    },
    "missing_fact": {
        "fact": ["1,180", "1,180 fuel invoices", "invoice"],
        "in_source": True, "in_draft": False,
        "owner": "Harborline Transit Authority",
        "fix": "restore_fact",
        "fix_words": ["add", "restor", "reinstate", "reinsert", "include"],
    },
    "wrong_record_attribution": {
        "fact": ["1,180"],
        "in_source": True, "in_draft": True,
        "owner": "Harborline Transit Authority",
        "wrong_owner": "Cascade Freight Partners",
        "fix": "move_to_owner",
        "fix_words": ["harborline", "move", "belong", "restore", "separat"],
    },
    "omitted_record": {
        "fact": ["Ridgeline", "Inventory Analyst"],
        "in_source": True, "in_draft": False,
        "owner": "Ridgeline Equipment Services",
        "fix": "restore_record",
        "fix_words": ["add", "restor", "reinstate", "include"],
    },
    # ── unfamiliar (calibration_cases) ───────────────────────────────────────
    "cal_lab_unsupported": {
        "owner_required": False,
        "fact": ["EPA Method 1604", "membrane filtration"],
        "in_source": False, "in_draft": True,
        "owner": "Tidewater Analytical Group",
        "fix": "remove_clause",
        "fix_words": ["remov", "delet", "drop", "strip", "without"],
    },
    "cal_lab_missing": {
        "fact": ["24-hour", "regulatory window"],
        "in_source": True, "in_draft": False,
        "owner": "Tidewater Analytical Group",
        "fix": "restore_fact",
        "fix_words": ["add", "restor", "reinstate", "reinsert", "include"],
    },
    "cal_lab_wrong_record": {
        "fact": ["640"],
        "in_source": True, "in_draft": True,
        "owner": "Tidewater Analytical Group",
        "wrong_owner": "Cascade Microbiology Services",
        "fix": "move_to_owner",
        "fix_words": ["tidewater", "move", "belong", "restore", "separat"],
    },
    "cal_lab_omitted": {
        "fact": ["Cascade Microbiology", "heterotrophic", "Contract Analyst"],
        "in_source": True, "in_draft": False,
        "owner": "Cascade Microbiology Services",
        "fix": "restore_record",
        "fix_words": ["add", "restor", "reinstate", "include"],
    },
    "cal_ops_unsupported": {
        "owner_required": False,
        "fact": ["LTL", "consolidat", "two regional carriers"],
        "in_source": False, "in_draft": True,
        "owner": "Pioneer Cold Storage",
        "fix": "remove_clause",
        "fix_words": ["remov", "delet", "drop", "strip", "without"],
    },
    "cal_ops_missing": {
        "fact": ["94%", "missed delivery"],
        "in_source": True, "in_draft": False,
        "owner": "Gorge Logistics Partners",
        "fix": "restore_fact",
        "fix_words": ["add", "restor", "reinstate", "reinsert", "include"],
    },
    "cal_ops_wrong_record": {
        "fact": ["260"],
        "in_source": True, "in_draft": True,
        "owner": "Pioneer Cold Storage",
        "wrong_owner": "Gorge Logistics Partners",
        "fix": "move_to_owner",
        "fix_words": ["pioneer", "move", "belong", "restore", "separat"],
    },
    "cal_ops_omitted": {
        "fact": ["Gorge Logistics", "Scheduling Analyst", "28 routes"],
        "in_source": True, "in_draft": False,
        "owner": "Gorge Logistics Partners",
        "fix": "restore_record",
        "fix_words": ["add", "restor", "reinstate", "include"],
    },
}

# Absence assertions, SCOPED. A reviewer saying "absent from the source" has
# answered criterion 2, not criterion 3 — treating the two as one test made a
# correct unsourced-method objection look like a false claim about the draft.
ABSENT_FROM_SOURCE = ("not mentioned", "not stated", "not present in the source",
                      "absent from the source", "not in the source",
                      "does not state", "does not mention", "neither source",
                      "no source", "unsourced", "not provided by the source",
                      "not supported by the source", "unauthorized addition",
                      "not evidence", "invent")
ABSENT_FROM_DRAFT = ("omits", "omitted from the draft", "missing from the draft",
                     "not in the draft", "absent from the draft",
                     "does not include", "draft does not", "removed from the draft",
                     "deleted from the draft", "not reflected in the draft",
                     "no longer in the draft")
# Unscoped absence words: they assert absence somewhere and are counted for the
# draft only when no source-scoped phrase already claimed the sentence.
ABSENT_GENERIC = ("missing", "absent", "omit", "dropped")


def _text_of(issue: dc.Issue) -> str:
    return " ".join([issue.passage or "", issue.why or "", issue.correction or ""])


_WORD_RE = re.compile(r"[a-z0-9]+")


def _contains_phrase(text: str, phrase: str) -> bool:
    """Substring match that will not fire inside a longer word.

    Measured while building this rubric: bare `in` matching fired "invent" on the
    word "inventor**y**" and scored a correct omission claim as a denial that the
    fact is source-backed. Single-word phrases therefore require a word boundary;
    multi-word phrases are matched on their words rather than raw characters.
    """
    hay = " ".join(text.lower().split())
    needle = " ".join(phrase.lower().split())
    if not needle:
        return False
    if " " in needle:
        return needle in hay
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", hay) is not None


def _any_phrase(text: str, phrases: tuple[str, ...] | list[str]) -> list[str]:
    return [p for p in phrases if _contains_phrase(text, p)]


def _mentions(text: str, needles: list[str]) -> bool:
    return bool(_any_phrase(text, needles))


def _strip_needle(text: str, needles: list[str]) -> str:
    """Remove every needle from `text`, so a correction can be checked by CONTENT."""
    out = text.lower()
    for needle in needles:
        out = out.replace(needle.lower(), "")
    return out


def grade_issue(case_id: str, issue: dc.Issue) -> dict:
    """Grade ONE issue against the five criteria. `None` means undetermined.

    Criterion 4 (`owner`) is `None` where ownership is not at issue. Appending an
    unsourced method to a bullet that is already under the right employer is a
    local removal; demanding the reviewer restate the employer would fail a correct
    answer for brevity. Ownership is graded where the defect IS about ownership:
    a moved fact or a dropped record.
    """
    truth = TRUTH[case_id]
    text = _text_of(issue)
    low = text.lower()

    fact = _mentions(text, truth["fact"])
    if not fact:
        return {"fact": False, "in_source": None, "in_draft": None,
                "owner": None, "correction": None, "complete": False,
                "problem_used": issue.problem}

    # ── 4. ownership ─────────────────────────────────────────────────────────
    if not truth.get("owner_required", truth["fix"] != "remove_clause"):
        owner = None          # not applicable, NOT a failure
    else:
        owner = _contains_phrase(low, truth["owner"])
        if truth["fix"] == "move_to_owner":
            # A move is only described completely by naming where the fact is and
            # where it belongs.
            owner = owner and _contains_phrase(low, truth["wrong_owner"])

    # ── 2. does it exist in the source, and is that stated? ─────────────────
    asserts_unsourced = bool(_any_phrase(low, ABSENT_FROM_SOURCE))
    if truth["in_source"] is False:
        in_source = True if asserts_unsourced else None
    else:
        # Source-backed facts: an `omitted` issue, or any answer that does not
        # claim the source lacks them, treats them as sourced. Explicitly denying
        # a real source fact is a failure, not an unknown.
        in_source = False if asserts_unsourced else True

    # ── 3. does it exist in the draft, and is that stated? ──────────────────
    if truth["in_draft"] is False:
        # The fact must be reported as missing FROM THE DRAFT. `omitted` implies
        # it; otherwise an explicit draft-scoped phrase, or an unscoped absence
        # word, is required. A source-scoped phrase alone does NOT satisfy this:
        # "not mentioned in the source" answers criterion 2, not criterion 3.
        if _any_phrase(low, ABSENT_FROM_DRAFT) or issue.problem == "omitted":
            in_draft = True
        elif _any_phrase(low, ABSENT_GENERIC) and not asserts_unsourced:
            in_draft = True
        elif _any_phrase(low, ABSENT_GENERIC) and asserts_unsourced:
            # Ambiguous: "missing" alongside a source claim. Undetermined rather
            # than assumed, because guessing here would manufacture a pass.
            in_draft = None
        else:
            in_draft = None
    else:
        claims_absent_from_draft = bool(_any_phrase(low, ABSENT_FROM_DRAFT))
        in_draft = False if claims_absent_from_draft else True

    # ── 5. what actually needs correcting ───────────────────────────────────
    # Graded on the CONTENT of the correction, not on its vocabulary. "remove"
    # is one way to strip a clause; restating the sentence without it is another,
    # and the second is what the model actually does.
    corrected = " ".join((issue.correction or "").split())
    if not corrected:
        correction = None                      # nothing offered: undetermined
    else:
        still_has_fact = _mentions(corrected, truth["fact"])
        if truth["fix"] == "remove_clause":
            # Correct: the clause is gone. Wrong: it survives the "correction".
            correction = not still_has_fact
        elif truth["fix"] in ("restore_fact", "restore_record"):
            # Correct: the missing text is supplied. `(omit)` deletes and is the
            # opposite of restoring.
            correction = still_has_fact and not dc.is_omit_marker(corrected)
        elif truth["fix"] == "move_to_owner":
            correction = (_contains_phrase(corrected, truth["owner"])
                          or not _contains_phrase(corrected, truth["wrong_owner"]))
        else:                                   # pragma: no cover - exhaustive
            correction = None

    criteria = {"fact": fact, "in_source": in_source, "in_draft": in_draft,
                "owner": owner, "correction": correction}
    # `owner` being None counts as satisfied-by-not-applicable, so a local clause
    # removal is not failed for omitting an employer it did not need to name.
    scored = [v for k, v in criteria.items() if not (k == "owner" and v is None)]
    return criteria | {"complete": all(scored) and fact,
                       "owner_applicable": truth.get(
                           "owner_required", truth["fix"] != "remove_clause"),
                       "problem_used": issue.problem}


def grade_case(case_id: str, decision: dc.Decision) -> dict:
    """Grade a whole answer: the best-matching issue, plus every issue's score."""
    graded = [grade_issue(case_id, issue) for issue in decision.issues]
    scoring = [g for g in graded if g["fact"]]
    best = max(scoring, key=lambda g: sum(
        bool(g[k]) for k in ("in_source", "in_draft", "owner", "correction")),
        default=None)
    return {
        "case_id": case_id,
        "issues_total": len(graded),
        "issues_mentioning_fact": len(scoring),
        "best": best,
        "all_complete": any(g["complete"] for g in graded),
        "per_issue": graded,
    }


# ── Fixture integrity: the truth table must match the fixtures ───────────────

def assert_truth_matches_fixtures() -> None:
    """Fail loudly if a fixture edit invalidates the truth table."""
    from experiment.calibration_cases import by_id as cal_by_id
    from experiment.cases_v2 import by_id as fam_by_id

    def lookup(case_id: str):
        try:
            return cal_by_id(case_id)
        except KeyError:
            return fam_by_id(case_id)

    for case_id, truth in TRUTH.items():
        case = lookup(case_id)
        # Whitespace-normalised, because both sources and drafts are hard-wrapped
        # mid-phrase ("targeted quality\naudits"), and a raw substring test would
        # report a wrapped phrase as absent and reject a correct truth entry.
        sources = " ".join(
            " ".join(text.split()) for text in case.sources.values()).lower()
        draft = " ".join(case.draft_before_trim.split()).lower()

        for needle in truth["fact"]:
            n = needle.lower()
            if n not in sources and n not in draft:
                raise AssertionError(
                    f"{case_id}: fact {needle!r} is in neither source nor draft")

        # The DECLARED presence must hold for the needles as a group: at least one
        # needle in the source when in_source, and the same for the draft. A
        # defect is a single change, so a per-needle requirement would fail on a
        # synonym supplied for the reviewer's convenience.
        present = [n for n in truth["fact"] if n.lower() in sources]
        absent = [n for n in truth["fact"] if n.lower() not in sources]
        if truth["in_source"] and not present:
            raise AssertionError(f"{case_id}: claims in_source but no needle is")
        if not truth["in_source"] and present:
            raise AssertionError(
                f"{case_id}: claims NOT in_source but {present} is present")

        in_draft = [n for n in truth["fact"] if n.lower() in draft]
        if truth["in_draft"] and not in_draft:
            raise AssertionError(f"{case_id}: claims in_draft but no needle is")
        if not truth["in_draft"] and in_draft:
            raise AssertionError(
                f"{case_id}: claims NOT in draft but {in_draft} is present")


def grade_rows(rows: list[dict]) -> dict:
    """Grade an arbitrary set of run rows, using each row's own draft and sources."""
    out = []
    for row in rows:
        raw = row.get("first_response") or ""
        case_id = row["case_id"]
        if case_id not in TRUTH or not raw:
            continue
        decision = dc.decide_from_text(raw, row["draft"], row["sources"])
        graded = grade_case(case_id, decision)
        out.append({**graded, "group": row.get("group", "?"),
                    "decision": decision.decision,
                    "valid": bool(decision.decision)
                    and not decision.structural_problems})
    return summarise(out, grouped=len({r.get("group") for r in out}) > 1)


def summarise(results: list[dict], *, grouped: bool = False) -> dict:
    def tally(subset: list[dict]) -> dict:
        return {
            "cases": len(subset),
            "complete_detection": sum(1 for r in subset if r["all_complete"]),
            "fact_identified": sum(1 for r in subset
                                   if r["issues_mentioning_fact"]),
            "no_fact_identified": sum(1 for r in subset
                                      if not r["issues_mentioning_fact"]),
            "criteria": {
                key: sum(1 for r in subset
                         if (r["best"] or {}).get(key) is True)
                for key in ("in_source", "in_draft", "owner", "correction")},
            "undetermined": {
                key: sum(1 for r in subset
                         if (r["best"] or {}).get(key) is None)
                for key in ("in_source", "in_draft", "owner", "correction")},
        }

    out = {"all": tally(results), "cases": results}
    if grouped:
        out["by_group"] = {group: tally([r for r in results
                                         if r.get("group") == group])
                           for group in sorted({r.get("group") for r in results})}
    return out


def render(summary: dict) -> str:
    lines = ["# Detection rubric — five criteria per defective case", "",
             "`fact` = the relevant fact identified. `source` = whether it is in the "
             "source, and the answer stated. `draft` = whether it is in the draft, "
             "and the answer stated. `owner` = the right employer named. "
             "`fix` = the correction does the right kind of thing.", "",
             "`~` means **undetermined** — not a pass. `·` means failed.", ""]

    def table(title: str, t: dict, rows: list[dict]) -> list[str]:
        block = [f"## {title}", "",
                 f"- complete detection (all five): "
                 f"**{t['complete_detection']} / {t['cases']}**",
                 f"- fact identified, but not all five: "
                 f"{t['fact_identified'] - t['complete_detection']}",
                 f"- no relevant fact identified at all: {t['no_fact_identified']}",
                 f"- criteria met: `{t['criteria']}`",
                 f"- undetermined: `{t['undetermined']}`", "",
                 "| case | decision | fact | source | draft | owner | fix | all five |",
                 "|---|---|---|---|---|---|---|---|"]
        for row in rows:
            best = row["best"] or {}
            mark = lambda k: ("✓" if best.get(k) is True
                              else "~" if best.get(k) is None else "·")
            block.append(
                f"| `{row['case_id']}` | `{row['decision']}` | "
                f"{'✓' if row['issues_mentioning_fact'] else '·'} | "
                f"{mark('in_source')} | {mark('in_draft')} | {mark('owner')} | "
                f"{mark('correction')} | "
                f"**{'YES' if row['all_complete'] else 'no'}** |")
        return block + [""]

    lines += table("All defective cases", summary["all"], summary["cases"])
    for group, t in (summary.get("by_group") or {}).items():
        lines += table(f"Group: {group}", t,
                       [r for r in summary["cases"] if r.get("group") == group])
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="results/calibration-v1/calibration.json",
                    help="a calibration-style artifact with rows[].first_response")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    assert_truth_matches_fixtures()
    payload = json.loads(pathlib.Path(args.run).read_text())
    summary = grade_rows(payload["rows"])
    if args.out:
        pathlib.Path(args.out).write_text(render(summary))
        print(json.dumps({k: v for k, v in summary.items() if k != "cases"},
                         indent=2))
    else:
        print(render(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
