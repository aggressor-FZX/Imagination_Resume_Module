# Compact source-linked judgment table

Retained prose cells only (14 of 28). Generated from
`experiment/judgments.py`, which verifies each needle's location in the fixture
before reporting. **Expected labels are in `EXPECTED_LABELS.md`** and are not part
of the verdict column.

Legend for `verdict`:
- `detected` — the reviewer's own reasoning names the defect that exists.
- `detected_with_a_miscitation` — it names the defect **and** quotes absent text as
  present in another objection.
- `not_detected` — the defect exists; no objection mentions it.
- `invented_objection` — the draft has no declared defect, yet issues were raised.

---

## The table

| # | case | cell | defect that exists | decision | verdict | the objection, in its own words |
|---|---|---|---|---|---|---|
| 1 | `faithful_paraphrase` | advisory | none (reworded baseline) | revise | **invented_objection** | "*Ridgeline … 08/2023 - Present*" called misattributed; source states this exact role and date |
| 2 | `supported_method` | advisory | none (sourced method kept) | revise | **invented_objection** | "adds a causal mechanism ('through')"; source says "by standardizing panel labeling" |
| 3 | `unsupported_method` | advisory | "targeted quality audits" in neither source | revise | **detected** | "adds 'targeted quality audits' which is not mentioned in the source" |
| 4 | `missing_fact` | advisory | 1,180-invoice bullet deleted | revise | **not_detected** | argues verb tense "Maintained"/"Maintain" and "with"/"against"; never mentions 1,180 |
| 5 | `wrong_record_attribution` | advisory | 1,180 moved under Cascade | revise | **detected** | "merges two separate responsibilities from different employers … under Cascade Freight Partners" |
| 6 | `transferable_skill` | advisory | none (capability clause, Imaginative) | revise | **invented_objection** | "conflicting timelines" — each source states its own role as `Present`, so the overlap is sourced |
| 7 | `omitted_record` | advisory | whole Ridgeline record missing | revise | **detected_with_a_miscitation** | one objection correctly reports the record missing; another quotes `**Ridgeline …**` as if in the draft |
| 8 | `faithful_paraphrase` | enabled | none | revise | **invented_objection** | "cannot simultaneously hold two full-time positions marked as 'Present'" |
| 9 | `supported_method` | enabled | none | revise | **invented_objection** | "'through standardizing panel labeling' implies a causal mechanism not present in the source" |
| 10 | `unsupported_method` | enabled | "targeted quality audits" | revise | **detected** | "introduces an unsupported [method]" — names the clause |
| 11 | `missing_fact` | enabled | 1,180 bullet deleted | revise | **detected** | "removed a key factual detail … 'Reconciled 1,180 fuel invoices'" |
| 12 | `wrong_record_attribution` | enabled | 1,180 moved under Cascade | revise | **detected** | "these two actions were separate … merged" under Cascade |
| 13 | `transferable_skill` | enabled | none | revise | **invented_objection** | "08/2023 - Present" declared inconsistent with the source, which states it |
| 14 | `omitted_record` | enabled | whole Ridgeline record missing | revise | **detected_with_a_miscitation** | same pattern as #7 |

### Totals

| Verdict | advisory | enabled | total |
|---|---|---|---|
| `detected` | 2 | 3 | 5 |
| `detected_with_a_miscitation` | 1 | 1 | 2 |
| `not_detected` | 1 | 0 | 1 |
| `invented_objection` | 3 | 3 | 6 |

### What the table shows

1. **The reviewer is not blind to defects, but it is not reliable.** It names the
   planted defect in 7 of 8 defect cells (5 cleanly, 2 alongside a miscitation). The
   one miss (`missing_fact`, advisory) is real but not systematic — the same case is
   caught in the enabled arm.
2. **The dominant failure is refusal of faithful documents.** All 6 `invented_objection`
   verdicts are on three drafts the fixture declares clean, in both arms. The
   recurring invention is a `Present`/`Present` overlap called "impossible" when the
   sources themselves state it: `candidate-main.txt` gives
   `Fuel Reporting Analyst | 04/2019 - Present` and `candidate-supplement.txt` gives
   `Inventory Analyst | 08/2023 - Present`. The reviewer infers a contradiction the
   sources do not contain.
3. **The second recurring failure is quoting absent text as present** (#7, #14).
   Mechanically verified: neither "Ridgeline" nor "Inventory Analyst" occurs anywhere
   in either of those drafts.
4. **Agreement is not the same as correctness.** Axis 1 says 4 exact agreements; this
   table shows only 5 clean detections of 8 defects. The two numbers measure different
   things and neither is a quality score.

## Unfamiliar-case results

**Not yet available.** The calibration fixture is frozen and validated offline
(`experiment/calibration_cases.py --check`: 10 cases, 2 faithful, 8 defective, each
one declared change from its baseline) but **no provider calls have been made**, so
there are no unfamiliar-case results to report. Any figure here would be invented.
