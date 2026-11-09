# Detection rubric — five criteria per defective case

`fact` = the relevant fact identified. `source` = whether it is in the source, and the answer stated. `draft` = whether it is in the draft, and the answer stated. `owner` = the right employer named. `fix` = the correction does the right kind of thing.

`~` means **undetermined** — not a pass. `·` means failed.

## All defective cases

- complete detection (all five): **10 / 12**
- fact identified, but not all five: 2
- no relevant fact identified at all: 0
- criteria met: `{'in_source': 11, 'in_draft': 11, 'owner': 8, 'correction': 11}`
- undetermined: `{'in_source': 0, 'in_draft': 1, 'owner': 3, 'correction': 1}`

| case | decision | fact | source | draft | owner | fix | all five |
|---|---|---|---|---|---|---|---|
| `unsupported_method` | `revise` | ✓ | ✓ | ✓ | ~ | ✓ | **YES** |
| `missing_fact` | `revise` | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| `wrong_record_attribution` | `revise` | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| `omitted_record` | `revise` | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| `cal_lab_unsupported` | `revise` | ✓ | ✓ | ✓ | ~ | ✓ | **YES** |
| `cal_lab_missing` | `revise` | ✓ | ✓ | ✓ | · | ✓ | **no** |
| `cal_lab_wrong_record` | `revise` | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| `cal_lab_omitted` | `revise` | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| `cal_ops_unsupported` | `revise` | ✓ | ✓ | ✓ | ~ | ✓ | **YES** |
| `cal_ops_missing` | `revise` | ✓ | · | ~ | ✓ | ~ | **no** |
| `cal_ops_wrong_record` | `revise` | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| `cal_ops_omitted` | `revise` | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |

## Group: familiar

- complete detection (all five): **4 / 4**
- fact identified, but not all five: 0
- no relevant fact identified at all: 0
- criteria met: `{'in_source': 4, 'in_draft': 4, 'owner': 3, 'correction': 4}`
- undetermined: `{'in_source': 0, 'in_draft': 0, 'owner': 1, 'correction': 0}`

| case | decision | fact | source | draft | owner | fix | all five |
|---|---|---|---|---|---|---|---|
| `unsupported_method` | `revise` | ✓ | ✓ | ✓ | ~ | ✓ | **YES** |
| `missing_fact` | `revise` | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| `wrong_record_attribution` | `revise` | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| `omitted_record` | `revise` | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |

## Group: unfamiliar

- complete detection (all five): **6 / 8**
- fact identified, but not all five: 2
- no relevant fact identified at all: 0
- criteria met: `{'in_source': 7, 'in_draft': 7, 'owner': 5, 'correction': 7}`
- undetermined: `{'in_source': 0, 'in_draft': 1, 'owner': 2, 'correction': 1}`

| case | decision | fact | source | draft | owner | fix | all five |
|---|---|---|---|---|---|---|---|
| `cal_lab_unsupported` | `revise` | ✓ | ✓ | ✓ | ~ | ✓ | **YES** |
| `cal_lab_missing` | `revise` | ✓ | ✓ | ✓ | · | ✓ | **no** |
| `cal_lab_wrong_record` | `revise` | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| `cal_lab_omitted` | `revise` | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| `cal_ops_unsupported` | `revise` | ✓ | ✓ | ✓ | ~ | ✓ | **YES** |
| `cal_ops_missing` | `revise` | ✓ | · | ~ | ✓ | ~ | **no** |
| `cal_ops_wrong_record` | `revise` | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| `cal_ops_omitted` | `revise` | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |

