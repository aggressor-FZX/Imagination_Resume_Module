# Bounded calibration — refined instructions, unfamiliar paired controls

**Instructions were frozen before these calls.**

- mode: `imaginative`
- instructions sha256: `4261e9cb5b031230aa2a3adf3f818346eab02318dd229b430eca31f166d17e7b`
- contract fingerprint: `4261e9cb5b031230aa2a3adf3f818346eab02318dd229b430eca31f166d17e7b`
- fixture fingerprint: `b4b73e68b4d69b394216ccf5b68be6c5239719d31809b4fa9f5c394eff25a224`
- logical-call ceiling: 27 (3 attempts each)
- **hard request ceiling: 81**
- observed-cost ceiling: $1.00

The four axes are reported separately and are never summed.

## Familiar cases (7)

- **initial judgement** — classes `{'agree': 2, 'disagree': 2, 'cannot_assess': 0, 'invalid_response': 3, 'operational_failure': 0}`, rate denominator **4** (agrees 2, disagrees 2)
- **edit applicability** — proposals drawn 1; scaffold refusals `{'ambiguous_or_multiple_issues': 1, 'not_attempted': 3, 'passage_not_unique_literal_bullet': 1, 'replacement_not_exact_owned_source_bullet': 1}`
- **corrected-document approval** — 0 approved of 1 valid re-reviews (1 attempted, 0 invalid)
- **release** — 0 released; reasons `{'initial_review_unverifiable': 3, 're_review_not_approved': 1, 'revision_unresolved': 3}`
- requests: 7 first + 1 re-review
- cost: $0.003195 first + $0.000562 re-review = **$0.003757**

## Unfamiliar cases (10)

- **initial judgement** — classes `{'agree': 5, 'disagree': 1, 'cannot_assess': 0, 'invalid_response': 4, 'operational_failure': 0}`, rate denominator **6** (agrees 5, disagrees 1)
- **edit applicability** — proposals drawn 0; scaffold refusals `{'ambiguous_or_multiple_issues': 3, 'not_attempted': 5, 'passage_not_unique_literal_bullet': 2}`
- **corrected-document approval** — 0 approved of 0 valid re-reviews (0 attempted, 0 invalid)
- **release** — 1 released; reasons `{'initial_llm_approve': 1, 'initial_review_unverifiable': 4, 'revision_unresolved': 5}`
- requests: 10 first + 0 re-review
- cost: $0.004731 first + $0.000000 re-review = **$0.004731**

## Per-pair separation (unfamiliar controls only)

Separation requires the faithful member to be APPROVED and the defective member to receive a VALID refusal. Two refusals are not a separation.

| pair (baseline > variant) | baseline decision | variant decision | separated |
|---|---|---|---|
| `cal_lab_faithful > cal_lab_faithful` | `revise` | `revise` | **NO** |
| `cal_lab_faithful > cal_lab_unsupported` | `revise` | `revise` | **NO** |
| `cal_lab_faithful > cal_lab_missing` | `revise` | `revise` | **NO** |
| `cal_lab_faithful > cal_lab_wrong_record` | `revise` | `revise` | **NO** |
| `cal_lab_faithful > cal_lab_omitted` | `revise` | `revise` | **NO** |
| `cal_ops_faithful > cal_ops_faithful` | `approve` | `approve` | **NO** |
| `cal_ops_faithful > cal_ops_unsupported` | `approve` | `revise` | yes |
| `cal_ops_faithful > cal_ops_missing` | `approve` | `revise` | **NO** |
| `cal_ops_faithful > cal_ops_wrong_record` | `approve` | `revise` | yes |
| `cal_ops_faithful > cal_ops_omitted` | `approve` | `revise` | **NO** |

## Source-linked judgments

Every row below cites the exact passage the reviewer named, so the judgement can be checked against the source rather than taken on trust. `—` means the reviewer named no passage.

| case | group | expected | decision | first named passage | why |
|---|---|---|---|---|---|
| `faithful_paraphrase` | familiar | `approve` | `revise` | *Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present | The draft lists Ridgeline Equipment Services as a current role starting in August 2023. However, the main source document [candidate-main.txt] shows the candidate was employed at Harborline Transit Authority from 04/2019 |
| `supported_method` | familiar | `approve` | `revise` | Reduced rework by 18% through standardizing panel labeling. | The draft adds a causal mechanism ("through standardizing") not present in the source, which says "by standardizing". While similar, this introduces a subtle but unsupported inference about *how* the reduction was achiev |
| `unsupported_method` | familiar | `revise` | `revise` | - Cut rework by 18% through standardizing panel labeling and targeted quality audits across the terminal fleet. | The source mentions cutting rework by 18% through standardizing panel labeling but does not mention "targeted quality audits" as a method contributing to that result. Adding this detail constitutes an unauthorized additi |
| `missing_fact` | familiar | `revise` | `revise` | — | no issue reported |
| `wrong_record_attribution` | familiar | `revise` | `revise` | Prepared the monthly cost summary that the regional manager reviewed and reconciled 1,180 fuel invoices each quarter. | The act of reconciling 1,180 fuel invoices is attributed in the source to the role at Harborline Transit Authority, not Cascade Freight Partners. Combining unrelated duties from different roles into a single bullet creat |
| `transferable_skill` | familiar | `approve` | `revise` | Produced the depot scorecard that supervisors review in the Monday operations meeting. | The draft changes "built" to "produced," which is a synonymous wording and normally acceptable; however, it also adds interpretive phrasing "demonstrating reconciliation and audit capability" that is not present in the s |
| `omitted_record` | familiar | `revise` | `revise` | **Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present* - Maintain parts availability records for 7 service branches. - Compared quarterly parts usage against the prior four quarters to set  | The draft places the Ridgeline Equipment Services role as starting in 08/2023, but the Harborline Transit Authority role is listed as "04/2019 - Present" in both the source and draft. This creates an impossible timeline  |
| `cal_lab_faithful` | unfamiliar | `approve` | `revise` | *Cascade Microbiology Services** -- Renton, WA *Contract Analyst \| 09/2023 - Present | The draft implies that Cascade Microbiology Services is a separate employer following Tidewater Analytical Group, but the source documents show concurrent employment (overlapping dates). This misrepresents the nature of  |
| `cal_lab_unsupported` | unfamiliar | `revise` | `revise` | Completed total coliform and E. coli assays on 640 drinking-water samples during 2025 using EPA Method 1604 membrane filtration. | The source does not mention the use of EPA Method 1604 membrane filtration. Adding this specific method is an unauthorized addition of technical detail not present in the source. |
| `cal_lab_missing` | unfamiliar | `revise` | `revise` | Reported results to the state drinking-water program within the 24-hour regulatory window. | The draft omits a key detail about timely reporting of results which demonstrates compliance with regulatory requirements. |
| `cal_lab_wrong_record` | unfamiliar | `revise` | `revise` | Reviewed 90 quarterly quality-control charts for the contract laboratory and completed 640 drinking-water assays during 2025. | The 640 drinking-water assays are attributed to Cascade Microbiology Services in the draft but were performed at Tidewater Analytical Group according to lab-main.txt. |
| `cal_lab_omitted` | unfamiliar | `revise` | `revise` | Completed total coliform and E. coli assays on 640 drinking-water samples during 2025. | The draft says "during 2025", but the source specifies "in 2025", which is more precise and tied to a calendar year. Since the present date is not given, this could imply future activity unless clarified. |
| `cal_ops_faithful` | unfamiliar | `approve` | `approve` | — | no issue reported |
| `cal_ops_unsupported` | unfamiliar | `revise` | `revise` | Negotiated a carrier contract that cut inbound freight cost by 11% by consolidating LTL lanes onto two regional carriers. | The method ("by consolidating LTL lanes onto two regional carriers") is not stated in the source and constitutes an unauthorized addition of specific activity details. |
| `cal_ops_missing` | unfamiliar | `revise` | `revise` | **Gorge Logistics Partners** -- The Dalles, OR *Scheduling Analyst \| 06/2023 - Present* - Built the weekly driver schedule covering 28 routes. - Recovered 94% of missed delivery windows during the 2024 peak season. | The draft includes a statement about recovering 94% of missed delivery windows, but this detail is not present in the source file [ops-supplement.txt]. While the phrase "built the weekly driver schedule for 28 routes" is |
| `cal_ops_wrong_record` | unfamiliar | `revise` | `revise` | Built the weekly driver schedule covering 28 routes and audited 260 inbound shipments for temperature compliance. | The audit of 260 inbound shipments for temperature compliance was performed at Pioneer Cold Storage, not at Gorge Logistics Partners where the route scheduling occurred. Combining these into a single bullet misattributes |
| `cal_ops_omitted` | unfamiliar | `revise` | `revise` | *Scheduling Analyst \| 06/2023 - Present*\n- Built the weekly driver schedule for 28 routes.\n- Recovered 94% of missed delivery windows during the 2024 peak season. | This role and its achievements are not reflected in the draft. The draft omits a confirmed position and key accomplishments from `ops-supplement.txt`. |

## Accounting

- first-call requests: 18 total requests across all cases (retries included)
- cost, first calls: $0.007926
- cost, re-reviews: $0.000562
- **cost, total: $0.008488**

- requests sent / ceiling: 18 / 81
- billable responses: 18, unmetered: 0
- cost known: **True**

> The request ceiling is hard: it is checked before every HTTP attempt including retries. The cost ceiling is enforced against observed cost only; it is NOT a proof of maximum spend when any response omits usage.

