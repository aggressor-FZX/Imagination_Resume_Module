"""The corrected case set — v2. Complete two-document baseline.

Why v2 exists
-------------
v1 (`experiment/cases.py`) supplies BOTH source documents to every case but gives
every draft only the content of `candidate-main.txt`. `candidate-supplement.txt`
carries the `Ridgeline Equipment Services` record and the CICS certification, and
no v1 draft contains either.

A faithful reviewer MUST flag a dropped employer record. So the three v1 cases
expecting `approve` were unsatisfiable, and the model's refusal of them was correct
every time. That is what invalidated the first comparison: it did not measure
approval quality, it measured the harness's missing omission handling.

v1 is left byte-for-byte intact as the record of what was actually run
(`results/comparison-20261109-0719`, frozen with `SHA256SUMS`). v2 is the corrected
fixture and lives beside it.

The repair chosen
-----------------
**Include the supplement. Do not remove it.** The brief is explicit: "Do not remove
the supplement to make existing approval cases pass." Deleting a source document
would make the approve cases pass by shrinking the problem, and would also destroy
the only case that can exercise omission handling.

So `BASELINE` below is a faithful draft of the COMPLETE two-document source. It
names all three employers, carries all seven source bullets and the certification.

Isolation
---------
Every case is `BASELINE` plus EXACTLY ONE declared transformation. That is what
makes a wrong verdict attributable: if `missing_fact` is refused and
`faithful_paraphrase` is approved, the only difference between them is the deleted
bullet. Cases built by independent hand-editing would differ in several ways at
once and no result would be interpretable.

`MUTATIONS` records each transformation, so a test can prove the isolation rather
than trusting this docstring.
"""

from __future__ import annotations

from experiment.runner import FrozenCase

# ── The complete source, verbatim ──────────────────────────────────────────
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

# Both documents. Neither is optional, and neither is ever withheld from a case.
SOURCES = {"candidate-main.txt": MAIN, "candidate-supplement.txt": SUPPLEMENT}

# ── The baseline: the complete two-document source, faithfully written ─────
# Paraphrased throughout (that is the writer's job) but every record, bullet and
# credential from BOTH documents is present. This is the draft the approve cases
# are built from, so it must be genuinely approvable.
#
# Numerals are kept exact (14, 1,180, 18%, 7) rather than spelled out. Paraphrase
# rewords; it does not degrade a figure into an approximation. Keeping the figures
# literal also lets the validity guard check them mechanically without needing a
# number-word parser that could itself be wrong.
BASELINE = """\
## Professional Summary
Operations analyst with nine years of experience in depot reporting and fuel cost
control across public transit operations.

## Professional Experience
**Harborline Transit Authority** -- Tacoma, WA
*Fuel Reporting Analyst | 04/2019 - Present*
- Monitored monthly fuel use across 14 depots, comparing each month with the same
  month a year earlier.
- Reconciled 1,180 fuel invoices against depot dispensing records every quarter.
- Produced the depot scorecard that supervisors review in the Monday operations
  meeting.

**Ridgeline Equipment Services** -- Fife, WA
*Inventory Analyst | 08/2023 - Present*
- Maintained parts availability records for 7 service branches.
- Compared quarterly parts usage with the prior four quarters to set reorder points.

**Cascade Freight Partners** -- Kent, WA
*Cost Reporting Coordinator | 07/2015 - 03/2019*
- Cut rework by 18% by standardizing panel labeling across the terminal fleet.
- Prepared the monthly cost summary that the regional manager reviewed.

## Certifications
- Certified Inventory Control Specialist (CICS)
"""

# ── One transformation per case ────────────────────────────────────────────
_REWORK_BULLET = (
    "- Cut rework by 18% by standardizing panel labeling across the terminal fleet.")
_INVOICE_BULLET = (
    "- Reconciled 1,180 fuel invoices against depot dispensing records every quarter.")
_SUMMARY_BULLET = (
    "- Prepared the monthly cost summary that the regional manager reviewed.")
_RIDGELINE_RECORD = """\
**Ridgeline Equipment Services** -- Fife, WA
*Inventory Analyst | 08/2023 - Present*
- Maintained parts availability records for 7 service branches.
- Compared quarterly parts usage with the prior four quarters to set reorder points.

"""


def _once(old: str, new: str, text: str = BASELINE) -> str:
    """Apply a replacement that must match exactly once.

    A mutation that silently matched zero times would leave the case identical to
    the baseline while its expectation still said `revise` — an unsatisfiable case,
    which is exactly the v1 defect. Matching exactly once makes the substitution
    fail loudly instead.
    """
    count = text.count(old)
    if count != 1:
        raise AssertionError(
            f"mutation target must occur exactly once, found {count}: {old[:60]!r}")
    return text.replace(old, new, 1)


# approve — the whole baseline, as written. Nothing changed.
FAITHFUL_DRAFT = BASELINE

# approve — the source states the method ("by standardizing panel labeling").
# Only the framing verb changes, so the method is still the sourced one.
SUPPORTED_METHOD_DRAFT = _once(
    _REWORK_BULLET,
    "- Reduced rework by 18% through standardizing panel labeling.")

# revise — a plausible method appended to a specific achievement. The clause
# "through targeted quality audits" appears in neither source document.
UNSUPPORTED_METHOD_DRAFT = _once(
    _REWORK_BULLET,
    "- Cut rework by 18% through standardizing panel labeling and targeted "
    "quality audits across the terminal fleet.")

# revise — a quantified, source-backed achievement silently dropped. The document
# is weaker but not false, which is why this is correctable rather than withheld.
MISSING_FACT_DRAFT = _once(_INVOICE_BULLET + "\n", "")

# revise — a real figure attached to the wrong employer. The 1,180-invoice figure
# belongs to Harborline; here it is asserted for Cascade Freight.
WRONG_RECORD_DRAFT = _once(
    _SUMMARY_BULLET,
    "- Prepared the monthly cost summary that the regional manager reviewed and "
    "reconciled 1,180 fuel invoices each quarter.")

# approve (Imaginative only) — a general capability generalised from a
# demonstrated activity. No specific method, tool or metric is added.
TRANSFERABLE_SKILL_DRAFT = _once(
    _INVOICE_BULLET,
    "- Reconciled 1,180 fuel invoices against depot dispensing records every "
    "quarter, demonstrating reconciliation and audit capability.")

# revise — the entire Ridgeline record is gone while its source is supplied. This
# is the v1 defect turned into a case on purpose: it is the ONLY case whose
# expectation is "the reviewer must notice a whole omitted record", and it is what
# exercises omission handling (cited source + destination, no auto-insert).
OMITTED_RECORD_DRAFT = _once(_RIDGELINE_RECORD, "")


# ── The declared transformations ───────────────────────────────────────────
# case_id -> (kind, description). `kind` is what a test asserts against, so the
# isolation claim is checkable rather than documentary.
MUTATIONS: dict[str, dict[str, str]] = {
    "faithful_paraphrase": {
        "kind": "reword_only",
        "detail": "The baseline itself, paraphrased throughout. No fact changed.",
    },
    "supported_method": {
        "kind": "reword_sourced_method",
        "detail": "'Cut ... by' -> 'Reduced ... through'. The method 'standardizing "
                  "panel labeling' is the one the source states.",
    },
    "unsupported_method": {
        "kind": "append_unsourced_method",
        "detail": "Appends 'and targeted quality audits'. Neither source states it.",
    },
    "missing_fact": {
        "kind": "delete_sourced_bullet",
        "detail": "Deletes the 1,180-invoice reconciliation bullet entirely.",
    },
    "wrong_record_attribution": {
        "kind": "move_fact_to_other_employer",
        "detail": "Attaches the 1,180-invoice figure to Cascade Freight, where it "
                  "does not belong.",
    },
    "transferable_skill": {
        "kind": "add_capability_clause",
        "detail": "Adds 'demonstrating reconciliation and audit capability' — a "
                  "capability, not an activity, and permitted only under Imaginative.",
    },
    "omitted_record": {
        "kind": "delete_whole_record",
        "detail": "Removes the entire Ridgeline Equipment Services record while "
                  "candidate-supplement.txt still supplies it.",
    },
}


CASES_V2: tuple[FrozenCase, ...] = (
    FrozenCase(
        case_id="faithful_paraphrase",
        expectation="approve",
        draft_before_trim=FAITHFUL_DRAFT,
        sources=SOURCES,
        intent="Complete two-document baseline, reworded throughout. Approving "
               "this is the floor: a reviewer that refuses a faithful draft of a "
               "document it was given in full is unusable at any mode."),
    FrozenCase(
        case_id="supported_method",
        expectation="approve",
        draft_before_trim=SUPPORTED_METHOD_DRAFT,
        sources=SOURCES,
        intent="The source states how the rework was cut ('by standardizing panel "
               "labeling'). Changing the framing verb is paraphrase, not an "
               "addition. Refusing this would punish the writer for rewriting."),
    FrozenCase(
        case_id="unsupported_method",
        expectation="revise",
        draft_before_trim=UNSUPPORTED_METHOD_DRAFT,
        sources=SOURCES,
        intent="'and targeted quality audits' is plausible for the role and absent "
               "from both sources. This is the exact clause class the previous "
               "contract approved after calling it a reasonable interpretation."),
    FrozenCase(
        case_id="missing_fact",
        expectation="revise",
        draft_before_trim=MISSING_FACT_DRAFT,
        sources=SOURCES,
        intent="A quantified, source-backed achievement was dropped. The document "
               "is weaker, not false, so the right answer is to restore it, not to "
               "deliver silently."),
    FrozenCase(
        case_id="wrong_record_attribution",
        expectation="revise",
        draft_before_trim=WRONG_RECORD_DRAFT,
        sources=SOURCES,
        intent="The 1,180-invoice figure is real but belongs to Harborline. "
               "Attaching it to Cascade Freight is the misattribution the "
               "same-document-wrong-record rule exists for."),
    FrozenCase(
        case_id="transferable_skill",
        expectation="approve",
        draft_before_trim=TRANSFERABLE_SKILL_DRAFT,
        sources=SOURCES,
        intent="A general capability generalised from a demonstrated activity. "
               "Permitted under Imaginative; a Conservative reviewer would "
               "correctly refuse it, so this case is only valid when the run mode "
               "is Imaginative."),
    FrozenCase(
        case_id="omitted_record",
        expectation="revise",
        draft_before_trim=OMITTED_RECORD_DRAFT,
        sources=SOURCES,
        intent="A whole employer record is missing from the draft while its source "
               "is supplied. The reviewer must name the missing record and the "
               "document it came from, and must NOT be given an automatic "
               "insertion by the harness. This case exists because v1 had it by "
               "accident."),
)


def by_id(case_id: str) -> FrozenCase:
    for case in CASES_V2:
        if case.case_id == case_id:
            return case
    raise KeyError(case_id)


def expectations() -> dict[str, str]:
    return {case.case_id: case.expectation for case in CASES_V2}


def cases_fingerprint() -> str:
    """Identity of the FIXTURE, so a preregistration cannot drift from it."""
    import hashlib
    import json as _json
    blob = [{"case_id": c.case_id,
             "expectation": c.expectation,
             "draft": c.draft_before_trim,
             "sources": dict(sorted(c.sources.items())),
             "mutation": MUTATIONS[c.case_id]}
            for c in CASES_V2]
    payload = _json.dumps(blob, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
