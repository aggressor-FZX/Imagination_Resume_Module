"""The frozen case set.

Every case states what a CORRECT reviewer should conclude, before any run. Fixing
the expectation first is what stops the experiment from grading the model against
itself: if the expectation were derived from the output, a false approval would
become a "correct" one.

The categories are the ones the brief names, and they are deliberately balanced so
that a reviewer which simply approves everything, or simply refuses everything,
scores badly in both directions:

  faithful_paraphrase      same fact, different wording        -> approve
  supported_method         a method the source DOES state       -> approve
  unsupported_method       a plausible method the source does NOT state -> revise
  missing_fact             a sourced fact dropped from the draft -> revise
  wrong_record_attribution a real fact on the wrong employer    -> revise
  transferable_skill       a legitimate capability reading      -> approve

The two positive-but-subtle cases matter most. A comparison that only contains
obvious defects rewards a paranoid reviewer; one that only contains clean text
rewards a permissive one.
"""

from __future__ import annotations

from experiment.runner import FrozenCase

MAIN = """\
## Professional Summary
Operations analyst with 9 years of experience in depot reporting and fuel cost
control across public transit operations.

## Professional Experience
**Harborline Transit Authority** -- Tacoma, WA
*Fuel Reporting Analyst | 04/2019 - Present*
- Tracked monthly fuel consumption across 14 depots and compared each month
  against the same month in the prior year.
- Reconciled 1,180 fuel invoices against depot dispensing records each quarter.
- Built the depot scorecard that supervisors review in the Monday operations
  meeting.

**Cascade Freight Partners** -- Kent, WA
*Cost Reporting Coordinator | 07/2015 - 03/2019*
- Cut rework by 18% by standardizing panel labeling across the terminal fleet.
- Prepared the monthly cost summary reviewed by the regional manager.
"""

SUPPLEMENT = """\
## Professional Experience
**Ridgeline Equipment Services** -- Fife, WA
*Inventory Analyst | 08/2023 - Present*
- Maintain parts availability records for 7 service branches.
- Compared quarterly parts usage against the prior four quarters to set reorder
  points.

## Certifications
- Certified Inventory Control Specialist (CICS)
"""

SOURCES = {"candidate-main.txt": MAIN, "candidate-supplement.txt": SUPPLEMENT}

# A draft that is faithful throughout: the baseline for false refusals.
FAITHFUL_DRAFT = """\
## Professional Summary
Operations analyst with nine years in depot reporting and fuel cost control for
public transit operations.

## Professional Experience
**Harborline Transit Authority** -- Tacoma, WA
*Fuel Reporting Analyst | 04/2019 - Present*
- Monitored monthly fuel use across 14 depots, comparing each month to the same
  month a year earlier.
- Reconciled 1,180 fuel invoices against depot dispensing records every quarter.
- Produced the depot scorecard supervisors review in the Monday operations
  meeting.

**Cascade Freight Partners** -- Kent, WA
*Cost Reporting Coordinator | 07/2015 - 03/2019*
- Cut rework by 18% by standardizing panel labeling across the terminal fleet.
- Prepared the monthly cost summary the regional manager reviewed.

## Certifications
- Certified Inventory Control Specialist (CICS)
"""

# The defect the previous contract approved (retained-case shape).
UNSUPPORTED_METHOD_DRAFT = FAITHFUL_DRAFT.replace(
    "- Cut rework by 18% by standardizing panel labeling across the terminal fleet.",
    "- Cut rework by 18% by standardizing panel labeling across the terminal fleet "
    "through targeted quality audits.")

# A method the source DOES state, re-worded. Must approve.
SUPPORTED_METHOD_DRAFT = FAITHFUL_DRAFT.replace(
    "- Cut rework by 18% by standardizing panel labeling across the terminal fleet.",
    "- Reduced rework by 18% through standardizing panel labeling.")

# A sourced fact silently dropped.
MISSING_FACT_DRAFT = FAITHFUL_DRAFT.replace(
    "- Reconciled 1,180 fuel invoices against depot dispensing records every quarter.\n",
    "")

# A real figure moved to the wrong employer.
WRONG_RECORD_DRAFT = FAITHFUL_DRAFT.replace(
    "- Prepared the monthly cost summary the regional manager reviewed.",
    "- Prepared the monthly cost summary the regional manager reviewed and "
    "reconciled 1,180 fuel invoices each quarter.")

# A capability reading of a demonstrated activity (Imaginative-permitted).
TRANSFERABLE_SKILL_DRAFT = FAITHFUL_DRAFT.replace(
    "- Reconciled 1,180 fuel invoices against depot dispensing records every quarter.",
    "- Reconciled 1,180 fuel invoices against depot dispensing records each "
    "quarter, demonstrating audit and reconciliation capability.")

CASES: tuple[FrozenCase, ...] = (
    FrozenCase(
        case_id="faithful_paraphrase",
        expectation="approve",
        draft_before_trim=FAITHFUL_DRAFT,
        sources=SOURCES,
        intent="Same facts, different wording throughout. Approving this is the "
               "baseline: a reviewer that refuses it is unusable regardless of "
               "how well it catches defects."),
    FrozenCase(
        case_id="supported_method",
        expectation="approve",
        draft_before_trim=SUPPORTED_METHOD_DRAFT,
        sources=SOURCES,
        intent="The source states the method ('by standardizing panel labeling'). "
               "Re-wording it is a paraphrase, not an addition."),
    FrozenCase(
        case_id="unsupported_method",
        expectation="revise",
        draft_before_trim=UNSUPPORTED_METHOD_DRAFT,
        sources=SOURCES,
        intent="'through targeted quality audits' is plausible for the role and "
               "absent from the source. This is the exact clause the previous "
               "contract approved after calling it a reasonable interpretation."),
    FrozenCase(
        case_id="missing_fact",
        expectation="revise",
        draft_before_trim=MISSING_FACT_DRAFT,
        sources=SOURCES,
        intent="A quantified, source-backed achievement was dropped. Losing it "
               "makes the document weaker but not false; the brief treats this as "
               "correctable, not a reason to deliver silently."),
    FrozenCase(
        case_id="wrong_record_attribution",
        expectation="revise",
        draft_before_trim=WRONG_RECORD_DRAFT,
        sources=SOURCES,
        intent="The 1,180-invoice figure is real, but it belongs to Harborline. "
               "Attaching it to Cascade Freight is the misattribution the "
               "same-document-wrong-record rule exists for."),
    FrozenCase(
        case_id="transferable_skill",
        expectation="approve",
        draft_before_trim=TRANSFERABLE_SKILL_DRAFT,
        sources=SOURCES,
        intent="Generalising a demonstrated activity into a stated capability is "
               "permitted under Imaginative. Refusing it would make the mode "
               "pointless."),
)


def by_id(case_id: str) -> FrozenCase:
    for case in CASES:
        if case.case_id == case_id:
            return case
    raise KeyError(case_id)


def expectations() -> dict[str, str]:
    return {case.case_id: case.expectation for case in CASES}
