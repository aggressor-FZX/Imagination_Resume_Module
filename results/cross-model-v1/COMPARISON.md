# Matched cross-model comparison

Same frozen drafts, same frozen sources, same mode and policy. **Only the model differs.**

- mode: `imaginative`
- instructions sha256: `4261e9cb5b031230aa2a3adf3f818346eab02318dd229b430eca31f166d17e7b`
- requests: 14 / 28
- cost: $0.010737

## Per-model summary

| model | usable | concurrency claimed impossible | synonymy objected | clean approved | defect fully detected | cost |
|---|---|---|---|---|---|---|
| `qwen/qwen3-coder` | 5/7 | **2** | **2** | 1 | 4 | $0.000000 |
| `openai/gpt-4o-mini` | 5/7 | **0** | **2** | 0 | 3 | $0.002249 |

## The objections themselves

Quoted, not scored. This is the material an independent assessor needs.

### `faithful_paraphrase` — probe: `clean_paraphrase`

- `openai/gpt-4o-mini` → `revise` (usable), 3 objection(s)
  - first objection: [unsupported] The term "monitored" does not accurately reflect the original wording "tracked," which is a more precise description of the activity performed.
- `qwen/qwen3-coder` → `revise` (UNUSABLE), 3 objection(s) — **claims concurrency impossible**, **objects to synonymy**
  - first objection: [unsupported] The draft lists Ridgeline Equipment Services as a current role starting in August 2023. However, the main source document [candidate-main.txt] shows the candidate was employed at Harborline Transit Authority from 04/2019 to Present, indicating an

### `cal_ops_faithful` — probe: `clean_paraphrase`

- `openai/gpt-4o-mini` → `revise` (UNUSABLE), 3 objection(s) — **objects to synonymy**
  - first objection: [misattributed] The draft states "during 2024," which implies the auditing occurred throughout the entire year, while the source specifies it was for the year 2024 without indicating it was ongoing.
- `qwen/qwen3-coder` → `approve` (usable), 0 objection(s) — approves
  - first objection: (no issues raised)

### `cal_lab_faithful` — probe: `source_backed_concurrency`

- `openai/gpt-4o-mini` → `revise` (usable), 4 objection(s) — **objects to synonymy**
  - first objection: [misattributed] The original source states that the assays were run in 2025, but the draft incorrectly states "completed," which implies the work was finished in that year rather than conducted.
- `qwen/qwen3-coder` → `revise` (usable), 4 objection(s) — **claims concurrency impossible**, **objects to synonymy**
  - first objection: [misattributed] The draft implies that Cascade Microbiology Services is a separate employer following Tidewater Analytical Group, but the source documents show concurrent employment (overlapping dates). This misrepresents the nature of the roles as sequential 

### `unsupported_method` — probe: `unsupported_method`

- `openai/gpt-4o-mini` → `revise` (usable), 3 objection(s) — detects defect
  - first objection: [unsupported] The draft adds "targeted quality audits" as a method for cutting rework, which is not supported by the source. The source only states that rework was cut by standardizing panel labeling.
- `qwen/qwen3-coder` → `revise` (usable), 1 objection(s) — detects defect
  - first objection: [unsupported] The source mentions cutting rework by 18% through standardizing panel labeling but does not mention "targeted quality audits" as a method contributing to that result. Adding this detail constitutes an unauthorized addition of a specific activity 

### `cal_ops_unsupported` — probe: `unsupported_method`

- `openai/gpt-4o-mini` → `revise` (usable), 1 objection(s) — detects defect
  - first objection: [unsupported] The draft adds specific details about how the cost was reduced ("by consolidating LTL lanes onto two regional carriers") that are not supported by the source document.
- `qwen/qwen3-coder` → `revise` (usable), 1 objection(s) — detects defect
  - first objection: [unsupported] The method ("by consolidating LTL lanes onto two regional carriers") is not stated in the source and constitutes an unauthorized addition of specific activity details.

### `omitted_record` — probe: `omission_or_wrong_record`

- `openai/gpt-4o-mini` → `approve` (UNUSABLE), 2 objection(s)
  - first objection: [unsupported] The term "monitored" does not fully capture the specific action of "tracked" as stated in the source. While synonymous, "tracked" conveys a more precise meaning in this context.
- `qwen/qwen3-coder` → `revise` (UNUSABLE), 2 objection(s) — detects defect
  - first objection: [misattributed] The draft places the Ridgeline Equipment Services role as starting in 08/2023, but the Harborline Transit Authority role is listed as "04/2019 - Present" in both the source and draft. This creates an impossible timeline overlap. The draft incor

### `wrong_record_attribution` — probe: `omission_or_wrong_record`

- `openai/gpt-4o-mini` → `revise` (usable), 1 objection(s) — detects defect
  - first objection: [misattributed] The reconciliation of 1,180 fuel invoices is attributed to the wrong role; it belongs to the Fuel Reporting Analyst position at Harborline Transit Authority, not the Cost Reporting Coordinator position at Cascade Freight Partners.
- `qwen/qwen3-coder` → `revise` (usable), 1 objection(s) — detects defect
  - first objection: [misattributed] The act of reconciling 1,180 fuel invoices is attributed in the source to the role at Harborline Transit Authority, not Cascade Freight Partners. Combining unrelated duties from different roles into a single bullet creates a misleading implicat

