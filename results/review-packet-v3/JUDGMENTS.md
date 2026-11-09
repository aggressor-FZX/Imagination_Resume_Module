# Source-linked judgments

Each row quotes the objection the reviewer wrote against the defect the fixture actually contains. The verdict is about NAMING, not about whether the correction would be safe to apply.

| case | cell | declared defect | decision | verdict | basis |
|---|---|---|---|---|---|
| `faithful_paraphrase` | `revised+advisory` | `reword_only` | `revise` | **invented_objection** | 3 issue(s) on a document with no declared defect |
| `supported_method` | `revised+advisory` | `reword_sourced_method` | `revise` | **invented_objection** | 2 issue(s) on a document with no declared defect |
| `unsupported_method` | `revised+advisory` | `append_unsourced_method` | `revise` | **detected** | reviewer's reasoning names 'targeted quality audits' |
| `missing_fact` | `revised+advisory` | `delete_sourced_bullet` | `revise` | **not_detected** | no objection mentions '1,180'; the defect exists |
| `wrong_record_attribution` | `revised+advisory` | `move_fact_to_other_employer` | `revise` | **detected** | reviewer's reasoning names '1,180' |
| `transferable_skill` | `revised+advisory` | `add_capability_clause` | `revise` | **invented_objection** | 3 issue(s) on a document with no declared defect |
| `omitted_record` | `revised+advisory` | `delete_whole_record` | `revise` | **detected_with_a_miscitation** | one objection correctly names 'Ridgeline' as missing, while another quotes it as draft text |
| `faithful_paraphrase` | `revised+enabled` | `reword_only` | `revise` | **invented_objection** | 2 issue(s) on a document with no declared defect |
| `supported_method` | `revised+enabled` | `reword_sourced_method` | `revise` | **invented_objection** | 2 issue(s) on a document with no declared defect |
| `unsupported_method` | `revised+enabled` | `append_unsourced_method` | `revise` | **detected** | reviewer's reasoning names 'targeted quality audits' |
| `missing_fact` | `revised+enabled` | `delete_sourced_bullet` | `revise` | **detected** | reviewer's reasoning names '1,180' |
| `wrong_record_attribution` | `revised+enabled` | `move_fact_to_other_employer` | `revise` | **detected** | reviewer's reasoning names '1,180' |
| `transferable_skill` | `revised+enabled` | `add_capability_clause` | `revise` | **invented_objection** | 2 issue(s) on a document with no declared defect |
| `omitted_record` | `revised+enabled` | `delete_whole_record` | `revise` | **detected_with_a_miscitation** | one objection correctly names 'Ridgeline' as missing, while another quotes it as draft text |

## The objections themselves

### `faithful_paraphrase` / `revised+advisory` — **invented_objection**

Declared defect: `reword_only` — The baseline itself, paraphrased throughout. No fact changed.

| # | problem | passage (as quoted) | in draft? | why |
|---|---|---|---|---|
| 1 | `misattributed` | Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present | yes | The draft assigns Ridgeline Equipment Services to Fife, WA, but candidate-main.txt lists "Ridgeline Equipment Services -- Fife, WA" under a different role and timeframe than what i |
| 2 | `unsupported` | - Maintained parts availability records for 7 service branches. | yes | While this statement paraphrases the source accurately, it shifts from active verb (“Maintain”) to passive (“Maintained”), which changes the tense without clear justification based |
| 3 | `omitted` | - Compared quarterly parts usage with the prior four quarters to set reorder points. | yes | The phrasing of this bullet matches the source closely, but because the role itself appears misplaced due to conflicting timelines between documents, the entire section should be r |

### `supported_method` / `revised+advisory` — **invented_objection**

Declared defect: `reword_sourced_method` — 'Cut ... by' -> 'Reduced ... through'. The method 'standardizing panel labeling' is the one the source states.

| # | problem | passage (as quoted) | in draft? | why |
|---|---|---|---|---|
| 1 | `misattributed` | Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present | yes | The draft correctly lists Ridgeline Equipment Services and aligns with the dates and duties listed in the supplement. However, the main resume omits this role entirely, which could |
| 2 | `unsupported` | Reduced rework by 18% through standardizing panel labeling. | yes | While the source supports that rework was reduced by 18% via standardization, the draft adds a causal mechanism (“through”) that implies a direct cause-effect relationship beyond w |

### `unsupported_method` / `revised+advisory` — **detected**

Declared defect: `append_unsourced_method` — Appends 'and targeted quality audits'. Neither source states it.

| # | problem | passage (as quoted) | in draft? | why |
|---|---|---|---|---|
| 1 | `unsupported` | Cut rework by 18% through standardizing panel labeling and targeted quality audits across the terminal fleet. | yes | The draft adds "targeted quality audits" which is not mentioned in the source and constitutes an unauthorized addition of a specific method. |

### `missing_fact` / `revised+advisory` — **not_detected**

Declared defect: `delete_sourced_bullet` — Deletes the 1,180-invoice reconciliation bullet entirely.

| # | problem | passage (as quoted) | in draft? | why |
|---|---|---|---|---|
| 1 | `misattributed` | Inventory Analyst \| 08/2023 - Present | yes | The draft correctly lists Ridgeline Equipment Services and the role of Inventory Analyst, but candidate-supplement.txt shows the dates as 08/2023 - Present, while main source (cand |
| 2 | `unsupported` | - Maintained parts availability records for 7 service branches. | yes | The verb tense differs ("Maintained" vs "Maintain"), which changes the implication about ongoing action. Since the employment period includes "Present," maintaining past tense is a |
| 3 | `unsupported` | - Compared quarterly parts usage with the prior four quarters to set reorder points. | yes | The preposition used ("with" instead of "against") alters the meaning slightly and unnecessarily. It's better to preserve the original phrasing for accuracy. |

### `wrong_record_attribution` / `revised+advisory` — **detected**

Declared defect: `move_fact_to_other_employer` — Attaches the 1,180-invoice figure to Cascade Freight, where it does not belong.

| # | problem | passage (as quoted) | in draft? | why |
|---|---|---|---|---|
| 1 | `misattributed` | Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present | yes | The employment dates for Ridgeline Equipment Services (08/2023 – Present) overlap with Harborline Transit Authority (04/2019 – Present), but the source lists Ridgeline as a current |
| 2 | `misattributed` | Prepared the monthly cost summary that the regional manager reviewed and reconciled 1,180 fuel invoices each quarter. | yes | The draft merges two separate responsibilities from different employers into one bullet under Cascade Freight Partners. Preparing a monthly cost summary belongs to Cascade Freight  |

### `transferable_skill` / `revised+advisory` — **invented_objection**

Declared defect: `add_capability_clause` — Adds 'demonstrating reconciliation and audit capability' — a capability, not an activity, and permitted only under Imaginative.

| # | problem | passage (as quoted) | in draft? | why |
|---|---|---|---|---|
| 1 | `misattributed` | Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present | yes | The draft places Ridgeline Equipment Services as a current role starting in August 2023, but candidate-main.txt lists Harborline Transit Authority as the current role ("04/2019 - P |
| 2 | `unsupported` | - Compared quarterly parts usage with the prior four quarters to set reorder points. | yes | While the action is listed in the supplement, there's no indication of employment dates or job title associated with this specific task in the main document, so its attribution to  |
| 3 | `unsupported` | - Produced the depot scorecard that supervisors review in the Monday operations meeting. | yes | The phrase “demonstrating reconciliation and audit capability” was added in the draft and not present in the source. It introduces an interpretation beyond what is stated. |

### `omitted_record` / `revised+advisory` — **detected_with_a_miscitation**

Declared defect: `delete_whole_record` — Removes the entire Ridgeline Equipment Services record while candidate-supplement.txt still supplies it.

| # | problem | passage (as quoted) | in draft? | why |
|---|---|---|---|---|
| 1 | `misattributed` | **Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present* - Maintain parts availability records for 7 service branches. -  | **NO** | The employment dates in the draft (08/2023 – Present) conflict with the previously listed roles, making concurrent employment impossible without explicit overlap indication. Additi |

### `faithful_paraphrase` / `revised+enabled` — **invented_objection**

Declared defect: `reword_only` — The baseline itself, paraphrased throughout. No fact changed.

| # | problem | passage (as quoted) | in draft? | why |
|---|---|---|---|---|
| 1 | `misattributed` | *Inventory Analyst \| 08/2023 - Present | yes | The draft places Ridgeline Equipment Services in Fife, WA, but according to the source document, this role overlaps temporally with the current role at Harborline Transit Authority |
| 2 | `unsupported` | **Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present | yes | The candidate cannot simultaneously hold two full-time positions marked as “Present.” The supplemental file lists Ridgeline as “08/2023 – Present,” while the main file has Harborli |

### `supported_method` / `revised+enabled` — **invented_objection**

Declared defect: `reword_sourced_method` — 'Cut ... by' -> 'Reduced ... through'. The method 'standardizing panel labeling' is the one the source states.

| # | problem | passage (as quoted) | in draft? | why |
|---|---|---|---|---|
| 1 | `misattributed` | Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present* - Maintained parts availability records for 7 service branches. -  | yes | The employment dates for Ridgeline Equipment Services are inconsistent between the draft and source. The draft says "08/2023 - Present", but candidate-main.txt shows a gap from 03/ |
| 2 | `unsupported` | Cascade Freight Partners** -- Kent, WA *Cost Reporting Coordinator \| 07/2015 - 03/2019* - Reduced rework by 18% through standardizing panel labeling. | yes | The phrase “through standardizing panel labeling” implies a causal mechanism not present in the source. While the action of standardizing labeling is supported, attaching the outco |

### `unsupported_method` / `revised+enabled` — **detected**

Declared defect: `append_unsourced_method` — Appends 'and targeted quality audits'. Neither source states it.

| # | problem | passage (as quoted) | in draft? | why |
|---|---|---|---|---|
| 1 | `unsupported` | Cut rework by 18% through standardizing panel labeling and targeted quality audits across the terminal fleet. | yes | The draft adds "targeted quality audits" which is not mentioned in the source. This introduces an unsupported method to explain the rework reduction. |

### `missing_fact` / `revised+enabled` — **detected**

Declared defect: `delete_sourced_bullet` — Deletes the 1,180-invoice reconciliation bullet entirely.

| # | problem | passage (as quoted) | in draft? | why |
|---|---|---|---|---|
| 1 | `omitted` | Reconciled 1,180 fuel invoices against depot dispensing records each quarter. | yes | The draft removed a key factual detail present in the source that reflects a core responsibility and measurable activity. |
| 2 | `unsupported` | Built the depot scorecard that supervisors review in the Monday operations meeting. | **NO** | While the draft says "Produced the depot scorecard...", the original wording "Built the depot scorecard..." is more precise and directly attributable. Changing the verb alters the  |
| 3 | `misattributed` | Compared quarterly parts usage against the prior four quarters to set reorder points. | **NO** | The draft changes “against” to “with,” which while minor, introduces unnecessary variation from the source phrasing without benefit and could affect clarity or searchability. |
| 4 | `unsupported` | Cut rework by 18% by standardizing panel labeling across the terminal fleet. | yes | The draft changes “Cut rework” to “Reduced rework,” which alters the specificity of the accomplishment unnecessarily. |
| 5 | `unsupported` | Prepared the monthly cost summary reviewed by the regional manager. | **NO** | The draft adds “that the regional manager reviewed,” changing the passive construction to active. This addition implies direct involvement not stated in the source. |

### `wrong_record_attribution` / `revised+enabled` — **detected**

Declared defect: `move_fact_to_other_employer` — Attaches the 1,180-invoice figure to Cascade Freight, where it does not belong.

| # | problem | passage (as quoted) | in draft? | why |
|---|---|---|---|---|
| 1 | `unsupported` | "08/2023 - Present"** under Ridgeline Equipment Services | **NO** | The candidate-supplement.txt states employment at Ridgeline Equipment Services began in August 2023, but candidate-main.txt lists Harborline Transit Authority as "04/2019 – Present |
| 2 | `misattributed` | Compared quarterly parts usage with the prior four quarters to set reorder points. | yes | While the action described matches the source, it's attributed to Ridgeline Equipment Services in the draft, but candidate-supplement.txt also contains other responsibilities liste |
| 3 | `misattributed` | Prepared the monthly cost summary that the regional manager reviewed and reconciled 1,180 fuel invoices each quarter. | yes | These two actions were separate bullet points under Cascade Freight Partners in the source and have been merged into one sentence in a way that implies the regional manager reviewe |

### `transferable_skill` / `revised+enabled` — **invented_objection**

Declared defect: `add_capability_clause` — Adds 'demonstrating reconciliation and audit capability' — a capability, not an activity, and permitted only under Imaginative.

| # | problem | passage (as quoted) | in draft? | why |
|---|---|---|---|---|
| 1 | `misattributed` | Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present* - Maintained parts availability records for 7 service branches. -  | yes | The draft lists Ridgeline Equipment Services as "08/2023 - Present", but candidate-supplement.txt indicates the role started in August 2023 and is ongoing; however, the main resume |
| 2 | `unsupported` | Operations analyst with nine years of experience in depot reporting and fuel cost control across public transit operations. | yes | While the phrasing matches the source, the total experience spans more than just public transit operations due to inclusion of Ridgeline Equipment Services (inventory analysis) and |

### `omitted_record` / `revised+enabled` — **detected_with_a_miscitation**

Declared defect: `delete_whole_record` — Removes the entire Ridgeline Equipment Services record while candidate-supplement.txt still supplies it.

| # | problem | passage (as quoted) | in draft? | why |
|---|---|---|---|---|
| 1 | `misattributed` | **Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present* - Maintain parts availability records for 7 service branches. -  | **NO** | The employment dates in the draft (08/2023 – Present) conflict with the Harborline Transit Authority role listed as "04/2019 – Present". This misattributes concurrent roles to the  |
| 2 | `omitted` | (omitted) | yes | The draft does not include the candidate’s current position at Ridgeline Equipment Services, despite being listed as “Present” in the source document. |

## Summary

Prose cells judged: 14

| verdict | count |
|---|---|
| `detected` | 5 |
| `detected_with_a_miscitation` | 2 |
| `invented_objection` | 6 |
| `not_detected` | 1 |

> A `detected` verdict means the objection NAMED the defect. It does not mean the proposed correction was safe to apply, nor that the corrected document would be approvable. Those are the separate axes in gate_axes.

