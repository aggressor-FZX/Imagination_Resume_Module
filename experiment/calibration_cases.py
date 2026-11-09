"""Bounded calibration fixture: the seven existing cases plus unfamiliar PAIRS.

Why pairs
---------
The seven existing cases share one source set and one writer. A reviewer can score
well on them by learning that fixture's quirks — the shared `Present` dates, the
1,180-invoice figure, the `targeted quality audits` clause. That would be
memorisation, not judgement.

This fixture therefore adds **paired controls on unfamiliar content**: for each
domain, a faithful draft and a defective draft built from the SAME source, differing
in exactly one declared way. A reviewer that approves both has not read either; one
that refuses both has not read either; only a reviewer that separates the pair has
made a judgement about the difference. Paired controls are also what makes a false
approval and a false refusal visible without needing a composite score.

The defect kinds are the four the calibration is meant to exercise:
    unsupported_activity, missing_fact, wrong_record, omitted_record
plus the faithful counterpart of each.

What is deliberately NOT here
-----------------------------
No new defect kinds, no redesigned contract, no production path. This fixture only
supplies unfamiliar text; the gate and the review instructions are unchanged.

Isolation, as in `cases_v2`
---------------------------
Every draft is one complete source plus exactly one declared transformation.
`MUTATIONS` records it and `_once` fails loudly if a transformation does not apply
exactly once, so "one declared change" is checked, not claimed.

Usage (offline only; no provider calls in this module):
    python -m experiment.calibration_cases --check
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiment.runner import FrozenCase  # noqa: E402

# ── Pair 1: a laboratory technician. Unfamiliar employer, unfamiliar metrics. ──
LAB_MAIN = """\
## Professional Summary
Laboratory technician with six years of experience in water quality analysis for
municipal and industrial clients.

## Professional Experience
**Tidewater Analytical Group** -- Everett, WA
*Laboratory Technician II | 03/2021 - Present*
- Ran total coliform and E. coli assays on 640 drinking-water samples in 2025.
- Maintained the lab's chain-of-custody log for all client samples.
- Trained two incoming technicians on the autoclave and the IDEXX workflow.
- Reported results to the state drinking-water program within the 24-hour
  regulatory window.

**Northgate Environmental Labs** -- Bellingham, WA
*Laboratory Assistant | 06/2018 - 02/2021*
- Prepared reagents and calibrated pH meters each morning before sample intake.
- Logged 1,400 sample receipts into the laboratory information management system.

## Certifications
- Washington State Drinking Water Laboratory Analyst certification
"""

LAB_SUPPLEMENT = """\
## Professional Experience
**Cascade Microbiology Services** -- Renton, WA
*Contract Analyst | 09/2023 - Present*
- Performed heterotrophic plate counts on industrial cooling-tower samples.
- Reviewed 90 quarterly quality-control charts for the contract laboratory.

## Certifications
- Certified Hazardous Materials Manager (CHMM)
"""

LAB_SOURCES = {"lab-main.txt": LAB_MAIN, "lab-supplement.txt": LAB_SUPPLEMENT}

LAB_BASELINE = """\
## Professional Summary
Laboratory technician with six years of experience in water quality testing for
municipal and industrial clients.

## Professional Experience
**Tidewater Analytical Group** -- Everett, WA
*Laboratory Technician II | 03/2021 - Present*
- Completed total coliform and E. coli assays on 640 drinking-water samples during
  2025.
- Kept the laboratory's chain-of-custody log for every client sample.
- Trained two new technicians on the autoclave and the IDEXX workflow.
- Sent results to the state drinking-water program inside the 24-hour regulatory
  window.

**Cascade Microbiology Services** -- Renton, WA
*Contract Analyst | 09/2023 - Present*
- Ran heterotrophic plate counts on industrial cooling-tower samples.
- Reviewed 90 quarterly quality-control charts for the contract laboratory.

**Northgate Environmental Labs** -- Bellingham, WA
*Laboratory Assistant | 06/2018 - 02/2021*
- Prepared reagents and calibrated pH meters each morning before samples arrived.
- Entered 1,400 sample receipts into the laboratory information management system.

## Certifications
- Washington State Drinking Water Laboratory Analyst certification
- Certified Hazardous Materials Manager (CHMM)
"""

# ── Pair 2: a warehouse/scheduling coordinator. Different figures entirely. ────
OPS_MAIN = """\
## Professional Summary
Warehouse operations coordinator with five years of experience in shift
scheduling and inbound freight for regional distributors.

## Professional Experience
**Pioneer Cold Storage** -- Pasco, WA
*Operations Coordinator | 01/2022 - Present*
- Scheduled 46 warehouse associates across three shifts each week.
- Reduced dock-to-stock time from 5.5 hours to 3.2 hours over eight months.
- Negotiated a carrier contract that lowered inbound freight cost by 11%.
- Audited 260 inbound shipments in 2024 for temperature compliance.

**Baker Valley Distribution** -- Yakima, WA
*Inventory Clerk | 04/2019 - 12/2021*
- Counted cycle inventory across 12,000 stocked SKUs each month.
- Flagged 130 purchase orders that arrived short against the vendor invoice.

## Certifications
- Certified Supply Chain Professional (CSCP)
"""

OPS_SUPPLEMENT = """\
## Professional Experience
**Gorge Logistics Partners** -- The Dalles, OR
*Scheduling Analyst | 06/2023 - Present*
- Built the weekly driver schedule for 28 routes.
- Recovered 94% of missed delivery windows during the 2024 peak season.

## Certifications
- OSHA 30-Hour General Industry certification
"""

OPS_SOURCES = {"ops-main.txt": OPS_MAIN, "ops-supplement.txt": OPS_SUPPLEMENT}

OPS_BASELINE = """\
## Professional Summary
Warehouse operations coordinator with five years of experience in shift
scheduling and inbound freight for regional distributors.

## Professional Experience
**Pioneer Cold Storage** -- Pasco, WA
*Operations Coordinator | 01/2022 - Present*
- Scheduled 46 warehouse associates across three shifts every week.
- Brought dock-to-stock time down from 5.5 hours to 3.2 hours in eight months.
- Negotiated a carrier contract that cut inbound freight cost by 11%.
- Audited 260 inbound shipments during 2024 for temperature compliance.

**Gorge Logistics Partners** -- The Dalles, OR
*Scheduling Analyst | 06/2023 - Present*
- Built the weekly driver schedule covering 28 routes.
- Recovered 94% of missed delivery windows in the 2024 peak season.

**Baker Valley Distribution** -- Yakima, WA
*Inventory Clerk | 04/2019 - 12/2021*
- Counted cycle inventory across 12,000 stocked SKUs every month.
- Flagged 130 purchase orders that arrived short against the vendor invoice.

## Certifications
- Certified Supply Chain Professional (CSCP)
- OSHA 30-Hour General Industry certification
"""


def _once(old: str, new: str, text: str) -> str:
    """Apply a replacement that must match exactly once, or fail loudly."""
    count = text.count(old)
    if count != 1:
        raise AssertionError(
            f"mutation target must occur exactly once, found {count}: {old[:70]!r}")
    return text.replace(old, new, 1)


# ══ Laboratory pair: one declared change each ══════════════════════════════════
LAB_FAITHFUL = LAB_BASELINE

LAB_UNSUPPORTED = _once(
    "- Completed total coliform and E. coli assays on 640 drinking-water samples during\n  2025.",
    "- Completed total coliform and E. coli assays on 640 drinking-water samples "
    "during\n  2025 using EPA Method 1604 membrane filtration.",
    LAB_BASELINE)

LAB_MISSING = _once(
    "- Sent results to the state drinking-water program inside the 24-hour regulatory\n"
    "  window.\n",
    "", LAB_BASELINE)

LAB_WRONG_RECORD = _once(
    "- Reviewed 90 quarterly quality-control charts for the contract laboratory.",
    "- Reviewed 90 quarterly quality-control charts for the contract laboratory and\n"
    "  completed 640 drinking-water assays during 2025.",
    LAB_BASELINE)

LAB_OMITTED = _once(
    """\
**Cascade Microbiology Services** -- Renton, WA
*Contract Analyst | 09/2023 - Present*
- Ran heterotrophic plate counts on industrial cooling-tower samples.
- Reviewed 90 quarterly quality-control charts for the contract laboratory.

""",
    "", LAB_BASELINE)

# ══ Operations pair: one declared change each ══════════════════════════════════
OPS_FAITHFUL = OPS_BASELINE

OPS_UNSUPPORTED = _once(
    "- Negotiated a carrier contract that cut inbound freight cost by 11%.",
    "- Negotiated a carrier contract that cut inbound freight cost by 11% by "
    "consolidating\n  LTL lanes onto two regional carriers.",
    OPS_BASELINE)

OPS_MISSING = _once(
    "- Recovered 94% of missed delivery windows in the 2024 peak season.\n",
    "", OPS_BASELINE)

OPS_WRONG_RECORD = _once(
    "- Built the weekly driver schedule covering 28 routes.",
    "- Built the weekly driver schedule covering 28 routes and audited 260 inbound\n"
    "  shipments for temperature compliance.",
    OPS_BASELINE)

OPS_OMITTED = _once(
    """\
**Gorge Logistics Partners** -- The Dalles, OR
*Scheduling Analyst | 06/2023 - Present*
- Built the weekly driver schedule covering 28 routes.
- Recovered 94% of missed delivery windows in the 2024 peak season.

""",
    "", OPS_BASELINE)


# `baseline` names the complete, faithful draft a case was derived from. Isolation
# is measured against THAT, not against the sibling control: `cal_lab_missing` and
# `cal_lab_wrong_record` are two different edits of the same baseline, so comparing
# them to each other shows two regions and would wrongly look like a double change.
CALIBRATION_MUTATIONS: dict[str, dict[str, str]] = {
    "cal_lab_faithful": {
        "kind": "reword_only",
        "detail": "Complete two-document baseline, reworded. No fact changed.",
        "baseline": "cal_lab_faithful",
        "contrast": "cal_lab_unsupported",
    },
    "cal_lab_unsupported": {
        "kind": "append_unsourced_method",
        "detail": "Adds 'using EPA Method 1604 membrane filtration'. Neither source "
                  "names a method.",
        "baseline": "cal_lab_faithful",
        "contrast": "cal_lab_faithful",
    },
    "cal_lab_missing": {
        "kind": "delete_sourced_bullet",
        "detail": "Deletes the 24-hour regulatory reporting bullet.",
        "baseline": "cal_lab_faithful",
        "contrast": "cal_lab_wrong_record",
    },
    "cal_lab_wrong_record": {
        "kind": "move_fact_to_other_employer",
        "detail": "Attaches Tidewater's 640-sample assay figure to Cascade "
                  "Microbiology, where it does not belong.",
        "baseline": "cal_lab_faithful",
        "contrast": "cal_lab_missing",
    },
    "cal_lab_omitted": {
        "kind": "delete_whole_record",
        "detail": "Removes the entire Cascade Microbiology record while "
                  "lab-supplement.txt still supplies it.",
        "baseline": "cal_lab_faithful",
        "contrast": "cal_lab_faithful",
    },
    "cal_ops_faithful": {
        "kind": "reword_only",
        "detail": "Complete two-document baseline, reworded. No fact changed.",
        "baseline": "cal_ops_faithful",
        "contrast": "cal_ops_unsupported",
    },
    "cal_ops_unsupported": {
        "kind": "append_unsourced_method",
        "detail": "Adds 'by consolidating LTL lanes onto two regional carriers'. "
                  "Neither source states how the cost was cut.",
        "baseline": "cal_ops_faithful",
        "contrast": "cal_ops_faithful",
    },
    "cal_ops_missing": {
        "kind": "delete_sourced_bullet",
        "detail": "Deletes the 94% missed-window recovery bullet.",
        "baseline": "cal_ops_faithful",
        "contrast": "cal_ops_wrong_record",
    },
    "cal_ops_wrong_record": {
        "kind": "move_fact_to_other_employer",
        "detail": "Attaches Pioneer Cold Storage's 260-shipment audit to Gorge "
                  "Logistics, where it does not belong.",
        "baseline": "cal_ops_faithful",
        "contrast": "cal_ops_missing",
    },
    "cal_ops_omitted": {
        "kind": "delete_whole_record",
        "detail": "Removes the entire Gorge Logistics record while "
                  "ops-supplement.txt still supplies it.",
        "baseline": "cal_ops_faithful",
        "contrast": "cal_ops_faithful",
    },
}

DEFECTIVE_IDS = ("cal_lab_unsupported", "cal_lab_missing", "cal_lab_wrong_record",
                 "cal_lab_omitted", "cal_ops_unsupported", "cal_ops_missing",
                 "cal_ops_wrong_record", "cal_ops_omitted")
FAITHFUL_IDS = ("cal_lab_faithful", "cal_ops_faithful")


CALIBRATION_CASES: tuple[FrozenCase, ...] = (
    FrozenCase("cal_lab_faithful", "approve", LAB_FAITHFUL, LAB_SOURCES,
               "Unfamiliar domain, complete two-document source, reworded only. "
               "Approving this is the floor for a reviewer that has not memorised "
               "the original fixture."),
    FrozenCase("cal_lab_unsupported", "revise", LAB_UNSUPPORTED, LAB_SOURCES,
               "An assay method is named that neither source states. Plausible for "
               "the role; still an invented method."),
    FrozenCase("cal_lab_missing", "revise", LAB_MISSING, LAB_SOURCES,
               "A regulatory reporting obligation with a stated deadline is "
               "dropped. The document is weaker than the source."),
    FrozenCase("cal_lab_wrong_record", "revise", LAB_WRONG_RECORD, LAB_SOURCES,
               "A real Tidewater figure is asserted under Cascade Microbiology."),
    FrozenCase("cal_lab_omitted", "revise", LAB_OMITTED, LAB_SOURCES,
               "A whole employer record is missing while its source is supplied."),
    FrozenCase("cal_ops_faithful", "approve", OPS_FAITHFUL, OPS_SOURCES,
               "Second unfamiliar domain, complete source, reworded only."),
    FrozenCase("cal_ops_unsupported", "revise", OPS_UNSUPPORTED, OPS_SOURCES,
               "A carrier-consolidation method is invented for a sourced cost cut."),
    FrozenCase("cal_ops_missing", "revise", OPS_MISSING, OPS_SOURCES,
               "A quantified peak-season recovery figure is dropped."),
    FrozenCase("cal_ops_wrong_record", "revise", OPS_WRONG_RECORD, OPS_SOURCES,
               "A real Pioneer shipment-audit figure is asserted under Gorge."),
    FrozenCase("cal_ops_omitted", "revise", OPS_OMITTED, OPS_SOURCES,
               "A whole employer record is missing while its source is supplied."),
)


def by_id(case_id: str) -> FrozenCase:
    for case in CALIBRATION_CASES:
        if case.case_id == case_id:
            return case
    raise KeyError(case_id)


def expectations() -> dict[str, str]:
    return {c.case_id: c.expectation for c in CALIBRATION_CASES}


def cases_fingerprint() -> str:
    """Identity of this fixture, recorded with any calibration run."""
    import hashlib
    blob = [{"case_id": c.case_id, "expectation": c.expectation,
             "draft": c.draft_before_trim,
             "sources": dict(sorted(c.sources.items())),
             "mutation": CALIBRATION_MUTATIONS[c.case_id]}
            for c in CALIBRATION_CASES]
    payload = json.dumps(blob, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check() -> dict:
    """Prove the fixture is isolated and that each mutation really applied.

    Returns a dict rather than printing, so a test can assert on it. Any failure
    here means the fixture is not measuring what it claims.
    """
    problems: list[str] = []
    regions: dict[str, int] = {}
    for case in CALIBRATION_CASES:
        mutation = CALIBRATION_MUTATIONS[case.case_id]
        baseline = by_id(mutation["baseline"])
        # Isolation is measured against the complete faithful draft: one declared
        # transformation means exactly ONE changed region there. A rewritten bullet
        # deletes a line and inserts another in the same opcode, so that is one.
        regions[case.case_id] = _change_regions(baseline.draft_before_trim,
                                                case.draft_before_trim)
        if case.case_id in DEFECTIVE_IDS and regions[case.case_id] == 0:
            problems.append(f"{case.case_id}: identical to its baseline "
                            f"({mutation['baseline']})")
        if regions[case.case_id] > 1:
            problems.append(f"{case.case_id}: {regions[case.case_id]} separate "
                            f"regions differ from its baseline; the declared "
                            "change is not isolated")
        if not case.draft_before_trim.strip():
            problems.append(f"{case.case_id}: empty draft")
        for name, text in case.sources.items():
            if not text.strip():
                problems.append(f"{case.case_id}: empty source {name}")

    present = {c.case_id for c in CALIBRATION_CASES}
    contrast_regions: dict[str, int] = {}
    for case_id, mutation in CALIBRATION_MUTATIONS.items():
        if case_id not in present:
            problems.append(f"dangling mutation for absent case: {case_id}")
        # The APPROVAL control is the baseline. Every defective draft must have a
        # faithful counterpart on the same sources, or there is nothing to show the
        # reviewer can tell approve from refuse.
        if mutation["baseline"] not in FAITHFUL_IDS:
            problems.append(f"{case_id}: baseline {mutation['baseline']} is not a "
                            "faithful control")
        # The CONTRAST control is a different defect on the same baseline. It is
        # what tests discrimination between defect kinds, so it must be a genuinely
        # different edit, not a second copy of the same one.
        contrast = mutation["contrast"]
        if contrast not in present:
            problems.append(f"dangling contrast reference: {case_id} -> {contrast}")
            continue
        contrast_regions[case_id] = _change_regions(
            by_id(contrast).draft_before_trim, by_id(case_id).draft_before_trim)
        if case_id in DEFECTIVE_IDS and contrast_regions[case_id] == 0:
            problems.append(f"{case_id}: identical to its contrast control "
                            f"({contrast})")
    return {
        "cases": len(CALIBRATION_CASES),
        "faithful": len(FAITHFUL_IDS),
        "defective": len(DEFECTIVE_IDS),
        "regions_from_baseline": regions,
        "regions_from_contrast": contrast_regions,
        "fingerprint": cases_fingerprint(),
        "problems": sorted(set(problems)),
    }


def _diff_lines(left: str, right: str) -> list[str]:
    import difflib
    return [line for line in
            difflib.unified_diff(left.splitlines(), right.splitlines(),
                                 lineterm="", n=0)
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]


def _change_regions(left: str, right: str) -> int:
    """How many separate regions of `right` differ from `left`.

    Two adjacent changed lines are ONE edit; two changes separated by unchanged
    lines are TWO. That distinction is what "exactly one declared transformation"
    actually means, and it is not the same as a changed-line count.
    """
    import difflib
    matcher = difflib.SequenceMatcher(None, left.splitlines(),
                                      right.splitlines(), autojunk=False)
    return sum(1 for tag, *_ in matcher.get_opcodes() if tag != "equal")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify the fixture (the only supported operation here)")
    args = ap.parse_args()
    if not args.check:
        ap.error("this module only builds fixtures; use --check (no provider calls)")
    result = check()
    print(json.dumps(result, indent=2))
    return 1 if result["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
