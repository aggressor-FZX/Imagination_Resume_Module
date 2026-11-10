# Decision-contract comparison

Model: `qwen/qwen3-coder` · 6 frozen cases × 4 cells = 24 attempts.

## Per-cell totals

| Cell | Cases | Correct approvals | False approvals | False refusals | Operational failures | Cost (USD) | Mean latency (s) |
|---|---|---|---|---|---|---|---|
| `current+advisory` | 6 | 0 | 0 | 6 | 5 | 0.00000 | 13.2 |
| `current+enabled` | 6 | 0 | 0 | 6 | 6 | 0.00000 | 9.5 |
| `revised+advisory` | 6 | 0 | 0 | 6 | 5 | 0.00000 | 9.7 |
| `revised+enabled` | 6 | 0 | 0 | 6 | 5 | 0.00000 | 6.0 |

## Per-case outcomes

| Case | Expected | Cell | Outcome | Reason |
|---|---|---|---|---|
| `faithful_paraphrase` | approve | `current+advisory` | operational_failure | envelope_not_json |
| `supported_method` | approve | `current+advisory` | operational_failure | envelope_not_json |
| `unsupported_method` | revise | `current+advisory` | operational_failure | envelope_not_json |
| `missing_fact` | revise | `current+advisory` | refused | reference_invalid |
| `wrong_record_attribution` | revise | `current+advisory` | operational_failure | envelope_not_json |
| `transferable_skill` | approve | `current+advisory` | operational_failure | envelope_not_json |
| `faithful_paraphrase` | approve | `current+enabled` | operational_failure | envelope_not_json |
| `supported_method` | approve | `current+enabled` | operational_failure | envelope_not_json |
| `unsupported_method` | revise | `current+enabled` | operational_failure | envelope_not_json |
| `missing_fact` | revise | `current+enabled` | operational_failure | envelope_not_json |
| `wrong_record_attribution` | revise | `current+enabled` | operational_failure | envelope_not_json |
| `transferable_skill` | approve | `current+enabled` | operational_failure | envelope_not_json |
| `faithful_paraphrase` | approve | `revised+advisory` | operational_failure | passage_not_in_draft:**Ridgeline Equipment Services** -- Fife, WA *Inventory Anal |
| `supported_method` | approve | `revised+advisory` | operational_failure | revise_without_issues |
| `unsupported_method` | revise | `revised+advisory` | operational_failure | passage_not_in_draft:**Ridgeline Equipment Services** -- Fife, WA *Inventory Anal,revise_without_actionable_issue |
| `missing_fact` | revise | `revised+advisory` | refused | reference_invalid |
| `wrong_record_attribution` | revise | `revised+advisory` | operational_failure | passage_not_in_draft:**Ridgeline Equipment Services** -- Fife, WA *Inventory Anal,sources_unresolved:candidate-main.txt, candidate-supplement.txt |
| `transferable_skill` | approve | `revised+advisory` | operational_failure | problem_word_unknown:**misattributed**,passage_not_in_draft:*Certified Inventory Control Specialist (CICS)* |
| `faithful_paraphrase` | approve | `revised+enabled` | operational_failure | passage_not_in_draft:**Ridgeline Equipment Services** -- Fife, WA *Inventory Anal |
| `supported_method` | approve | `revised+enabled` | operational_failure | problem_word_unknown:none |
| `unsupported_method` | revise | `revised+enabled` | operational_failure | passage_not_in_draft:- Maintain parts availability records for 7 service branches,passage_not_in_draft:- Compared quarterly parts usage against the prior four quar |
| `missing_fact` | revise | `revised+enabled` | refused | reference_invalid |
| `wrong_record_attribution` | revise | `revised+enabled` | operational_failure | problem_word_unknown:**misattributed**,problem_word_unknown:**unsupported**,passage_not_in_draft:**Ridgeline Equipment Services** -- Fife, WA *Inventory Anal,sources_unresolved:candidate-main.txt, candidate-supplement.txt,passage_not_in_draft:*Certified Inventory Control Specialist (CICS)*,revise_without_actionable_issue |
| `transferable_skill` | approve | `revised+enabled` | operational_failure | problem_word_unknown:**unsupported**,problem_word_unknown:**unsupported**,problem_word_unknown:**misattributed**,passage_not_in_draft:*Reconciled 1,180 fuel invoices against depot dispensing rec,passage_not_in_draft:*Produced the depot scorecard supervisors review in the Mond,passage_not_in_draft:*(omitted)* |

## Honest limits of this comparison

- 24 attempts. That is enough to compare cells, not enough to estimate a production failure rate.
- 21 operational failure(s) are counted separately and are NOT evidence about faithfulness in either direction.
- Parser acceptance is not reported as semantic correctness. A well-formed approval of a defective document is a false approval and is counted as one.
- Totals across all cells: 0 correct approvals, 0 false approvals, 24 false refusals.
- Expectations were fixed before any run. They are the author's judgement of what a correct reviewer should conclude, and they need the independent blind audit before they can be treated as labels.
