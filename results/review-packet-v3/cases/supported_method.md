# Case: `supported_method`

**Declared mutation:** `reword_sourced_method` — 'Cut ... by' -> 'Reduced ... through'. The method 'standardizing panel labeling' is the one the source states.

---

## Source documents

### `candidate-main.txt`

```text
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
```

### `candidate-supplement.txt`

```text
## Professional Experience
**Ridgeline Equipment Services** -- Fife, WA
*Inventory Analyst | 08/2023 - Present*
- Maintain parts availability records for 7 service branches.
- Compared quarterly parts usage against the prior four quarters to set reorder
  points.

## Certifications
- Certified Inventory Control Specialist (CICS)
```

---

## Reviewed draft

The exact text the reviewer was given.

```markdown
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
- Reduced rework by 18% through standardizing panel labeling.
- Prepared the monthly cost summary that the regional manager reviewed.

## Certifications
- Certified Inventory Control Specialist (CICS)
```

---

## Cell `current+advisory`

**Raw model decision:** **none — JSON envelope, no prose decision**

### Reported issues

_JSON span envelope. Issues are per-claim verdicts, not prose objections; see the raw response below._

### Proposed corrections

_No prose corrections to apply._

### Re-review

_No re-review was performed in the retained run._

### Binding-gate outcome (offline replay)

- initial decision: `none`
- repair status: `not_attempted`
- re-review decision: `none`
- gate result: **WITHHOLD** (`format_mismatch_json_answer_to_prose_gate`)
- initial structural problems: `decision_missing`

> The gate is a prose contract; this cell answered the legacy JSON envelope. The answer is not defective — it answers a different contract, so the gate cannot read it. This is a contract mismatch, not a model failure.

### Raw first response

```text
{
  "claims": [
    {
      "id": "1:1:0:85",
      "line_id": "1",
      "start": 0,
      "end": 85,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 25,
          "end": 110,
          "quote": "Operations analyst with 9 years of experience in depot reporting and fuel cost"
        }
      ]
    },
    {
      "id": "6:1:0:75",
      "line_id": "6",
      "start": 0,
      "end": 75,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 175,
          "end": 250,
          "quote": "- Tracked monthly fuel consumption across 14 depots and compared each month"
        }
      ]
    },
    {
      "id": "7:1:0:80",
      "line_id": "7",
      "start": 0,
      "end": 80,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 252,
          "end": 332,
          "quote": "- Reconciled 1,180 fuel invoices against depot dispensing records each quarter."
        }
      ]
    },
    {
      "id": "8:1:0:86",
      "line_id": "8",
      "start": 0,
      "end": 86,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 334,
          "end": 420,
          "quote": "- Built the depot scorecard that supervisors review in the Monday operations"
        }
      ]
    },
    {
      "id": "12:1:0:63",
      "line_id": "12",
      "start": 0,
      "end": 63,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 175,
          "end": 238,
          "quote": "- Maintain parts availability records for 7 service branches."
        }
      ]
    },
    {
      "id": "13:1:0:85",
      "line_id": "13",
      "start": 0,
      "end": 85,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 240,
          "end": 325,
          "quote": "- Compared quarterly parts usage against the prior four quarters to set reorder"
        }
      ]
    },
    {
      "id": "17:1:0:56",
      "line_id": "17",
      "start": 0,
      "end": 56,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 486,
          "end": 542,
          "quote": "- Cut rework by 18% by standardizing panel labeling"
        }
      ]
    },
    {
      "id": "18:1:0:67",
      "line_id": "18",
      "start": 0,
      "end": 67,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 544,
          "end": 611,
          "quote": "- Prepared the monthly cost summary reviewed by the regional manager."
        }
      ]
    },
    {
      "id": "21:1:0:50",
      "line_id": "21",
      "start": 0,
      "end": 50,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 328,
          "end": 378,
          "quote": "- Certified Inventory Control Specialist (CICS)"
        }
      ]
    }
  ],
  "representations": []
}
```

---

## Cell `current+enabled`

**Raw model decision:** **none — JSON envelope, no prose decision**

### Reported issues

_JSON span envelope. Issues are per-claim verdicts, not prose objections; see the raw response below._

### Proposed corrections

_No prose corrections to apply._

### Re-review

_No re-review was performed in the retained run._

### Binding-gate outcome (offline replay)

- initial decision: `none`
- repair status: `not_attempted`
- re-review decision: `none`
- gate result: **WITHHOLD** (`format_mismatch_json_answer_to_prose_gate`)
- initial structural problems: `decision_missing`

> The gate is a prose contract; this cell answered the legacy JSON envelope. The answer is not defective — it answers a different contract, so the gate cannot read it. This is a contract mismatch, not a model failure.

### Raw first response

```text
{
  "claims": [
    {
      "id": "1:1:0:85",
      "line_id": "1",
      "start": 0,
      "end": 85,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 25,
          "end": 110,
          "quote": "Operations analyst with 9 years of experience in depot reporting and fuel cost"
        }
      ]
    },
    {
      "id": "6:1:0:75",
      "line_id": "6",
      "start": 0,
      "end": 75,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 175,
          "end": 250,
          "quote": "- Tracked monthly fuel consumption across 14 depots and compared each month"
        }
      ]
    },
    {
      "id": "7:1:0:80",
      "line_id": "7",
      "start": 0,
      "end": 80,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 252,
          "end": 332,
          "quote": "- Reconciled 1,180 fuel invoices against depot dispensing records each quarter."
        }
      ]
    },
    {
      "id": "8:1:0:86",
      "line_id": "8",
      "start": 0,
      "end": 86,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 334,
          "end": 420,
          "quote": "- Built the depot scorecard that supervisors review in the Monday operations"
        }
      ]
    },
    {
      "id": "12:1:0:61",
      "line_id": "12",
      "start": 0,
      "end": 61,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 178,
          "end": 239,
          "quote": "- Maintain parts availability records for 7 service branches."
        }
      ]
    },
    {
      "id": "13:1:0:83",
      "line_id": "13",
      "start": 0,
      "end": 83,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 241,
          "end": 324,
          "quote": "- Compared quarterly parts usage against the prior four quarters to set reorder"
        }
      ]
    },
    {
      "id": "17:1:0:56",
      "line_id": "17",
      "start": 0,
      "end": 56,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 486,
          "end": 542,
          "quote": "- Cut rework by 18% by standardizing panel labeling"
        }
      ]
    },
    {
      "id": "18:1:0:69",
      "line_id": "18",
      "start": 0,
      "end": 69,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 544,
          "end": 613,
          "quote": "- Prepared the monthly cost summary reviewed by the regional manager."
        }
      ]
    },
    {
      "id": "21:1:0:55",
      "line_id": "21",
      "start": 0,
      "end": 55,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 388,
          "end": 443,
          "quote": "- Certified Inventory Control Specialist (CICS)"
        }
      ]
    }
  ],
  "representations": []
}
```

---

## Cell `revised+advisory`

**Raw model decision:** `revise`

### Reported issues

| # | passage | problem | why | sources | correction | resolves |
|---|---|---|---|---|---|---|
| 1 | Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present | `misattributed` | The draft correctly lists Ridgeline Equipment Services and aligns with the dates and duties listed in the supplement. However, the main resume omits this role entirely, which could imply it's new or u | candidate-supplement.txt | No change needed for content, but formatting should match other entries. | passage ✓ / sources ✓ |
| 2 | Reduced rework by 18% through standardizing panel labeling. | `unsupported` | While the source supports that rework was reduced by 18% via standardization, the draft adds a causal mechanism (“through”) that implies a direct cause-effect relationship beyond what is stated. This  | candidate-main.txt | Reduced rework by 18% by standardizing panel labeling. Approved as corrected. | passage ✓ / sources ✓ |

### Proposed corrections

| passage | proposed replacement | locatable in draft | omit marker |
|---|---|---|---|
| Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst | 08/2023 - Present | No change needed for content, but formatting should match other entries. | yes | no |
| Reduced rework by 18% through standardizing panel labeling. | Reduced rework by 18% by standardizing panel labeling. Approved as corrected. | yes | no |

### Re-review

**Raw re-review decision:** `revise`

```text
Decision: revise

Issues:
- passage: **Ridgeline Equipment Services** -- Fife, WA  
  *Inventory Analyst | 08/2023 - Present*  
  problem: misattributed  
  sources: [candidate-supplement.txt: "Ridgeline Equipment Services", "Inventory Analyst", "08/2023 - Present"]  
  why: The draft lists Ridgeline Equipment Services as currently employing the candidate ("Present"), but candidate-main.txt shows the candidate was employed at Harborline Transit Authority from 04/2019 to Present. This creates a contradiction in current employment.  
  correction: *Inventory Analyst | 08/2023 - 03/2024*  

- passage: - *Reduced rework by 18% by standardizing panel labeling.* Approved as corrected.  
  problem: unsupported  
  sources: [candidate-main.txt: "Cut rework by 18% by standardizing panel labeling across the terminal fleet."]  
  why: While the metric and outcome are supported, the phrase “Approved as corrected” has no basis in the source documents and implies an endorsement not present in the data.  
  correction: - Reduced rework by 18% by standardizing panel labeling across the terminal fleet.
```

### Binding-gate outcome (offline replay)

- initial decision: `revise`
- repair status: `ambiguous_or_multiple_issues`
- re-review decision: `none`
- gate result: **WITHHOLD** (`revision_unresolved`)

> The reviewer asked for a change, but no repair could be applied safely (multiple issues, an omission, an ambiguous passage, or a correction that is not an exact owned source bullet). Nothing was edited.

### Raw first response

```text
Decision: revise

Issues:
- passage: **Ridgeline Equipment Services** -- Fife, WA  
  *Inventory Analyst | 08/2023 - Present*  
  problem: misattributed  
  sources: [candidate-supplement.txt: "Ridgeline Equipment Services", "08/2023 - Present", "Maintain parts availability records for 7 service branches.", "Compared quarterly parts usage against the prior four quarters to set reorder points."]  
  why: The draft correctly lists Ridgeline Equipment Services and aligns with the dates and duties listed in the supplement. However, the main resume omits this role entirely, which could imply it's new or unverified unless explicitly included. Since both documents now include this role, it’s acceptable but must be consistent in structure and tense.  

  correction: No change needed for content, but formatting should match other entries.

- passage: *Reduced rework by 18% through standardizing panel labeling.*  
  problem: unsupported method attribution  
  sources: [candidate-main.txt: "Cut rework by 18% by standardizing panel labeling across the terminal fleet."]  
  why: While the source supports that rework was reduced by 18% via standardization, the draft adds a causal mechanism (“through”) that implies a direct cause-effect relationship beyond what is stated. This turns a result into an interpreted process, violating the rule against attaching specific methods to achievements.  
  correction: *Reduced rework by 18% by standardizing panel labeling.*

Approved as corrected.
```

---

## Cell `revised+enabled`

**Raw model decision:** `revise`

### Reported issues

| # | passage | problem | why | sources | correction | resolves |
|---|---|---|---|---|---|---|
| 1 | Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present* - Maintained parts availability records for 7 service branches. - Compared q | `misattributed` | The employment dates for Ridgeline Equipment Services are inconsistent between the draft and source. The draft says "08/2023 - Present", but candidate-main.txt shows a gap from 03/2019 to at least 04/ | candidate-supplement.txt | — | passage ✓ / sources ✓ |
| 2 | Cascade Freight Partners** -- Kent, WA *Cost Reporting Coordinator \| 07/2015 - 03/2019* - Reduced rework by 18% through standardizing panel labeling. | `unsupported` | The phrase “through standardizing panel labeling” implies a causal mechanism not present in the source. While the action of standardizing labeling is supported, attaching the outcome directly to that  | candidate-main.txt | Reduced rework by 18% while standardizing panel labeling across the terminal fleet. | passage ✓ / sources ✓ |

### Proposed corrections

| passage | proposed replacement | locatable in draft | omit marker |
|---|---|---|---|
| Cascade Freight Partners** -- Kent, WA *Cost Reporting Coordinator | 07/2015 - 03/2019* - Reduced rework by 18% through  | Reduced rework by 18% while standardizing panel labeling across the terminal fleet. | yes | no |

### Re-review

_No re-review was performed in the retained run._

### Binding-gate outcome (offline replay)

- initial decision: `revise`
- repair status: `ambiguous_or_multiple_issues`
- re-review decision: `none`
- gate result: **WITHHOLD** (`revision_unresolved`)

> The reviewer asked for a change, but no repair could be applied safely (multiple issues, an omission, an ambiguous passage, or a correction that is not an exact owned source bullet). Nothing was edited.

### Raw first response

```text
Decision: revise

Issues:
- passage: **Ridgeline Equipment Services** -- Fife, WA  
  *Inventory Analyst | 08/2023 - Present*  
  - Maintained parts availability records for 7 service branches.  
  - Compared quarterly parts usage with the prior four quarters to set reorder points.  
  problem: misattributed  
  sources: [candidate-supplement.txt: "Maintain parts availability records for 7 service branches.", "Compared quarterly parts usage against the prior four quarters to set reorder points."]  
  why: The employment dates for Ridgeline Equipment Services are inconsistent between the draft and source. The draft says "08/2023 - Present", but candidate-main.txt shows a gap from 03/2019 to at least 04/2019 at Harborline Transit Authority, making it impossible to start at Ridgeline in 2023 without a timeline conflict.  
  correction: (omit)

- passage: **Cascade Freight Partners** -- Kent, WA  
  *Cost Reporting Coordinator | 07/2015 - 03/2019*  
  - Reduced rework by 18% through standardizing panel labeling.  
  problem: unsupported  
  sources: [candidate-main.txt: "Cut rework by 18% by standardizing panel labeling across the terminal fleet."]  
  why: The phrase “through standardizing panel labeling” implies a causal mechanism not present in the source. While the action of standardizing labeling is supported, attaching the outcome directly to that cause constitutes an unauthorized activity claim.  
  correction: Reduced rework by 18% while standardizing panel labeling across the terminal fleet.
```

