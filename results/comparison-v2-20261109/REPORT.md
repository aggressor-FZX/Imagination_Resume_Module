# Corrected four-way comparison (v2)

Model: `qwen/qwen3-coder` · 7 frozen cases × 4 cells = 28 attempts.
Fixture fingerprint: `1aa800a3b14f22fd`
Budget used: 40 calls, $0.0000 of $2.00.

Preregistered in `results/PREREGISTRATION_V2.md` before the first call.

## Contract probe (non-comparative)

| Check | Result | Detail |
|---|---|---|
| `current_reaches_a_verdict` | PASS | decision='approve' complete=True reason=reference_fabricated |
| `revised_reaches_a_verdict` | PASS | decision='revise' complete=True reason=advisory_revise_noted |
| `mode_distinction_is_observable` | PASS | conservative=approved/advisory_revise_noted, imaginative=approved/advisory_revise_noted |

Probe verdict: **PASS**. These 4 calls are excluded from the matrix totals.

## Per-cell totals

| Cell | Scored | Incomplete | Reference failures | Correct approvals | False approvals | False refusals | Operational | Cost (USD) | Mean latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| `current+advisory` | 4/7 | 0 | 3 | 2 | 1 | 1 | 1 | 0.00000 | 17.6 |
| `current+enabled` | 5/7 | 0 | 2 | 1 | 2 | 1 | 0 | 0.00000 | 13.3 |
| `revised+advisory` | 7/7 | 0 | 0 | 2 | 2 | 3 | 3 | 0.00000 | 4.8 |
| `revised+enabled` | 7/7 | 0 | 0 | 0 | 0 | 4 | 1 | 0.00000 | 3.3 |

## Per-case outcomes

| Case | Expected | Cell | Outcome | Reason | Complete |
|---|---|---|---|---|---|
| `faithful_paraphrase` | approve | `current+advisory` | approved | semantic_approve | yes |
| `supported_method` | approve | `current+advisory` | approved | advisory_revise_noted | yes |
| `unsupported_method` | revise | `current+advisory` | reference_failure | reference_unknown_source | yes |
| `missing_fact` | revise | `current+advisory` | operational_failure | citation_shape_invalid | yes |
| `wrong_record_attribution` | revise | `current+advisory` | reference_failure | reference_fabricated | yes |
| `transferable_skill` | approve | `current+advisory` | reference_failure | reference_fabricated | yes |
| `omitted_record` | revise | `current+advisory` | approved | semantic_approve | yes |
| `faithful_paraphrase` | approve | `current+enabled` | approved | semantic_approve | yes |
| `supported_method` | approve | `current+enabled` | reference_failure | reference_cites_draft | yes |
| `unsupported_method` | revise | `current+enabled` | reference_failure | reference_fabricated | yes |
| `missing_fact` | revise | `current+enabled` | refused | semantic_revise_unresolved | yes |
| `wrong_record_attribution` | revise | `current+enabled` | approved | semantic_approve | yes |
| `transferable_skill` | approve | `current+enabled` | refused | semantic_revise_unresolved | yes |
| `omitted_record` | revise | `current+enabled` | approved | semantic_approve | yes |
| `faithful_paraphrase` | approve | `revised+advisory` | approved | advisory_revise_noted | yes |
| `supported_method` | approve | `revised+advisory` | operational_failure | passage_not_in_draft:-Reduced rework by 18% through standardizing panel labeling. | yes |
| `unsupported_method` | revise | `revised+advisory` | operational_failure | revise_without_actionable_issue | yes |
| `missing_fact` | revise | `revised+advisory` | approved | advisory_revise_noted | yes |
| `wrong_record_attribution` | revise | `revised+advisory` | operational_failure | problem_word_unknown:omitted/misplaced | yes |
| `transferable_skill` | approve | `revised+advisory` | approved | advisory_revise_noted | yes |
| `omitted_record` | revise | `revised+advisory` | approved | advisory_revise_noted | yes |
| `faithful_paraphrase` | approve | `revised+enabled` | refused | semantic_revise_unresolved | yes |
| `supported_method` | approve | `revised+enabled` | refused | semantic_revise_unresolved | yes |
| `unsupported_method` | revise | `revised+enabled` | refused | semantic_revise_unresolved | yes |
| `missing_fact` | revise | `revised+enabled` | refused | semantic_revise_unresolved | yes |
| `wrong_record_attribution` | revise | `revised+enabled` | operational_failure | passage_not_in_draft:"08/2023 - Present" under Ridgeline Equipment Services,revise_without_actionable_issue | yes |
| `transferable_skill` | approve | `revised+enabled` | refused | semantic_revise_unresolved | yes |
| `omitted_record` | revise | `revised+enabled` | refused | semantic_revise_unresolved | yes |

## Output budget and truncation

| Cell | finish_reason counts | max completion tokens |
|---|---|---|
| `current+advisory` | {'stop': 7} | 3340 |
| `current+enabled` | {'stop': 7} | 2730 |
| `revised+advisory` | {'stop': 10} | 507 |
| `revised+enabled` | {'stop': 10} | 353 |

## Omission handling

| Cell | Case | Cited source | Grounded | Destination | Basis | Applied |
|---|---|---|---|---|---|---|
| `revised+advisory` | `unsupported_method` | `candidate-supplement.txt` | NO | *Inventory Analyst | 08/2023 - Present* | anchor_in_draft | no |
| `revised+enabled` | `missing_fact` | `candidate-main.txt` | yes | — | undeterminable | no |
| `revised+enabled` | `missing_fact` | `candidate-supplement.txt` | NO | before 'Cascade Freight Partners' | source_order | no |

`applied` must be `no` everywhere: the harness identifies an omission and never inserts text.


## Limits

- 0 of 28 attempts are incomplete and are excluded from approval-quality denominators.
- Parser acceptance is NOT reported as semantic correctness.
- `revise` is a workflow, not a failure.
- A `reference:convention` result means the evidence was real and the offset differed; it is neither a refusal nor fabrication.
- Expectations are the author's and remain **provisional** until the blind audit. No winner is declared here.
