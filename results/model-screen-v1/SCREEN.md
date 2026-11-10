# Five-case screen — 4 models

Cases: 5 · models: 4 · logical calls: 20

Reused frozen drafts and sources. No repairs, no production change, no automatic dismissal of objections.

- mode: `imaginative`
- instructions sha256: `4261e9cb5b031230aa2a3adf3f818346eab02318dd229b430eca31f166d17e7b`
- models: `google/gemma-4-31b-it`, `google/gemini-2.5-flash-lite`, `xiaomi/mimo-v2.6-flash`, `openai/gpt-4o-mini`
- requests: 20 / 40
- cost: $0.005143 (known)

## The four outcomes, reported separately

A refusal is not credited unless the defect is named; an approval is not credited unless the answer is valid. Nothing here is inferred from a decision word or a keyword.

| model | valid | detection (of 3) | invented (of 2) | missed (of 3) | clean approved | cost |
|---|---|---|---|---|---|---|
| `google/gemma-4-31b-it` | 5/5 | 3/3 | 0/2 | 0/3 | 2/2 | $0.000828 |
| `google/gemini-2.5-flash-lite` | 4/5 | 2/3 | 1/2 | 1/3 | 1/2 | $0.001220 |
| `xiaomi/mimo-v2.6-flash` | 2/5 | 1/3 | 0/2 | 2/3 | 2/2 | $0.001430 |
| `openai/gpt-4o-mini` | 1/5 | 2/3 | 2/2 | 1/3 | 0/2 | $0.001665 |

### Endpoint routing

A pin is an instruction; the provider reported in each response is the fact. Any mismatch is counted as a violation rather than assumed away.

| model | pinned | actually served | violations |
|---|---|---|---|
| `google/gemma-4-31b-it` | — (unpinned) | `CoreWeave` | 0 |
| `google/gemini-2.5-flash-lite` | — (unpinned) | `Google AI Studio` | 0 |
| `xiaomi/mimo-v2.6-flash` | `cohere/fp8` | `Cohere` | 0 |
| `openai/gpt-4o-mini` | — (unpinned) | `OpenAI` | 0 |

## Per case

### `faithful_paraphrase` — `clean_paraphrase`, expected **approve**

- `google/gemma-4-31b-it` → `approve` (—), 0 issue(s)
- `google/gemini-2.5-flash-lite` → `approve` (INVALID, **invents**), 7 issue(s)
  - first: The source uses "Tracked" while the draft uses "Monitored". While synonymous, the source also states "fuel consumption" while the draft states "fuel use". This is a minor change in wording but the core meaning is preserved.
- `xiaomi/mimo-v2.6-flash` → `approve` (—), 0 issue(s)
- `openai/gpt-4o-mini` → `revise` (**invents**), 3 issue(s)
  - first: The term "monitored" does not accurately reflect the original wording "tracked," which is a more precise description of the activity performed.

### `cal_ops_faithful` — `clean_paraphrase`, expected **approve**

- `google/gemma-4-31b-it` → `approve` (—), 0 issue(s)
- `google/gemini-2.5-flash-lite` → `approve` (—), 0 issue(s)
- `xiaomi/mimo-v2.6-flash` → `approve` (—), 0 issue(s)
- `openai/gpt-4o-mini` → `revise` (INVALID, **invents**), 3 issue(s)
  - first: The draft states "during 2024," which implies the auditing occurred throughout the entire year, while the source specifies it was for the year 2024 without indicating it was ongoing.

### `unsupported_method` — `unsupported_method`, expected **revise**

- `google/gemma-4-31b-it` → `revise` (detects), 1 issue(s)
  - first: The draft adds "targeted quality audits" as a method for the achievement, which is not present in the source.
- `google/gemini-2.5-flash-lite` → `revise` (detects), 2 issue(s)
  - first: The draft adds "and targeted quality audits" which is not supported by the source document.
- `xiaomi/mimo-v2.6-flash` → `` (INVALID, **misses**), 0 issue(s)
- `openai/gpt-4o-mini` → `revise` (INVALID, detects), 3 issue(s)
  - first: The draft adds "targeted quality audits" as a method for cutting rework, which is not supported by the source. The source only states that rework was cut by standardizing panel labeling.

### `wrong_record_attribution` — `wrong_record`, expected **revise**

- `google/gemma-4-31b-it` → `revise` (detects), 1 issue(s)
  - first: The draft attributes the fuel invoice reconciliation (which occurred at Harborline Transit Authority) to the role at Cascade Freight Partners.
- `google/gemini-2.5-flash-lite` → `revise` (detects), 2 issue(s)
  - first: The draft attributes the reconciliation of 1,180 fuel invoices to Cascade Freight Partners, but the source document states this achievement occurred at Harborline Transit Authority.
- `xiaomi/mimo-v2.6-flash` → `` (INVALID, **misses**), 0 issue(s)
- `openai/gpt-4o-mini` → `revise` (INVALID, detects), 2 issue(s)
  - first: The reconciliation of 1,180 fuel invoices is attributed to the wrong role; it belongs to the Fuel Reporting Analyst position at Harborline Transit Authority, not the Cost Reporting Coordinator position at Cascade Freight Partners.

### `omitted_record` — `omitted_record`, expected **revise**

- `google/gemma-4-31b-it` → `revise` (detects), 1 issue(s)
  - first: The draft completely omits a current professional role and its associated achievements.
- `google/gemini-2.5-flash-lite` → `revise` (**misses**), 3 issue(s)
  - first: The draft omits the specific metric of "14 depots" which was present in the source document.
- `xiaomi/mimo-v2.6-flash` → `revise` (INVALID, detects), 1 issue(s)
  - first: The supplement documents a current employer, title, location, dates, and two substantive bullets. Dropping it removes a whole concurrent role (08/2023 – Present alongside the Harborline role) and the record that supports the CICS certificat
- `openai/gpt-4o-mini` → `approve` (INVALID, **misses**), 3 issue(s)
  - first: The term "monitored" does not fully capture the specific activity of "tracked monthly fuel consumption," which is more precise and supported by the source.

