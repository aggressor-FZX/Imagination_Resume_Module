# Case: `transferable_skill`

**Declared mutation:** `add_capability_clause` — Adds 'demonstrating reconciliation and audit capability' — a capability, not an activity, and permitted only under Imaginative.

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
- Reconciled 1,180 fuel invoices against depot dispensing records every quarter, demonstrating reconciliation and audit capability.
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
      "id": "1:1:65",
      "line_id": 1,
      "start": 0,
      "end": 65,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 24,
          "end": 89,
          "quote": "Operations analyst with 9 years of experience in depot reporting and fuel cost"
        }
      ]
    },
    {
      "id": "2:1:66",
      "line_id": 1,
      "start": 66,
      "end": 116,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 90,
          "end": 140,
          "quote": "control across public transit operations."
        }
      ]
    },
    {
      "id": "6:6:58",
      "line_id": 6,
      "start": 0,
      "end": 58,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 172,
          "end": 230,
          "quote": "- Tracked monthly fuel consumption across 14 depots and compared each month"
        }
      ]
    },
    {
      "id": "6:59:115",
      "line_id": 6,
      "start": 59,
      "end": 115,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 231,
          "end": 287,
          "quote": "  against the same month in the prior year."
        }
      ]
    },
    {
      "id": "7:1:60",
      "line_id": 7,
      "start": 1,
      "end": 60,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 291,
          "end": 350,
          "quote": "- Reconciled 1,180 fuel invoices against depot dispensing records each quarter."
        }
      ]
    },
    {
      "id": "7:61:116",
      "line_id": 7,
      "start": 61,
      "end": 116,
      "verdict": "unsupported",
      "citations": []
    },
    {
      "id": "8:1:56",
      "line_id": 8,
      "start": 1,
      "end": 56,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 354,
          "end": 409,
          "quote": "- Built the depot scorecard that supervisors review in the Monday operations"
        }
      ]
    },
    {
      "id": "10:10:66",
      "line_id": 10,
      "start": 10,
      "end": 66,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 102,
          "end": 158,
          "quote": "*Inventory Analyst | 08/2023 - Present*"
        }
      ]
    },
    {
      "id": "11:1:47",
      "line_id": 11,
      "start": 1,
      "end": 47,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 162,
          "end": 208,
          "quote": "- Maintain parts availability records for 7 service branches."
        }
      ]
    },
    {
      "id": "12:1:65",
      "line_id": 12,
      "start": 1,
      "end": 65,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 212,
          "end": 276,
          "quote": "- Compared quarterly parts usage against the prior four quarters to set reorder"
        }
      ]
    },
    {
      "id": "14:14:68",
      "line_id": 14,
      "start": 14,
      "end": 68,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 440,
          "end": 494,
          "quote": "*Cost Reporting Coordinator | 07/2015 - 03/2019*"
        }
      ]
    },
    {
      "id": "15:1:51",
      "line_id": 15,
      "start": 1,
      "end": 51,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 498,
          "end": 548,
          "quote": "- Cut rework by 18% by standardizing panel labeling across the terminal fleet."
        }
      ]
    },
    {
      "id": "16:1:57",
      "line_id": 16,
      "start": 1,
      "end": 57,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 552,
          "end": 608,
          "quote": "- Prepared the monthly cost summary reviewed by the regional manager."
        }
      ]
    },
    {
      "id": "18:1:47",
      "line_id": 18,
      "start": 1,
      "end": 47,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 284,
          "end": 330,
          "quote": "## Certifications\n-Certified Inventory Control Specialist (CICS)"
        }
      ]
    }
  ],
  "representations": [
    {
      "source_claim_id": "1:1:65",
      "status": "represented",
      "spans": [
        {
          "line_id": 1,
          "start": 0,
          "end": 65
        }
      ]
    },
    {
      "source_claim_id": "2:1:66",
      "status": "represented",
      "spans": [
        {
          "line_id": 1,
          "start": 66,
          "end": 116
        }
      ]
    },
    {
      "source_claim_id": "6:6:58",
      "status": "represented",
      "spans": [
        {
          "line_id": 6,
          "start": 0,
          "end": 58
        }
      ]
    },
    {
      "source_claim_id": "6:59:115",
      "status": "represented",
      "spans": [
        {
          "line_id": 6,
          "start": 59,
          "end": 115
        }
      ]
    },
    {
      "source_claim_id": "7:1:60",
      "status": "represented",
      "spans": [
        {
          "line_id": 7,
          "start": 1,
          "end": 60
        }
      ]
    },
    {
      "source_claim_id": "7:61:116",
      "status": "represented",
      "spans": [
        {
          "line_id": 7,
          "start": 61,
          "end": 116
        }
      ]
    },
    {
      "source_claim_id": "8:1:56",
      "status": "represented",
      "spans": [
        {
          "line_id": 8,
          "start": 1,
          "end": 56
        }
      ]
    },
    {
      "source_claim_id": "10:10:66",
      "status": "represented",
      "spans": [
        {
          "line_id": 10,
          "start": 10,
          "end": 66
        }
      ]
    },
    {
      "source_claim_id": "11:1:47",
      "status": "represented",
      "spans": [
        {
          "line_id": 11,
          "start": 1,
          "end": 47
        }
      ]
    },
    {
      "source_claim_id": "12:1:65",
      "status": "represented",
      "spans": [
        {
          "line_id": 12,
          "start": 1,
          "end": 65
        }
      ]
    },
    {
      "source_claim_id": "14:14:68",
      "status": "represented",
      "spans": [
        {
          "line_id": 14,
          "start": 14,
          "end": 68
        }
      ]
    },
    {
      "source_claim_id": "15:1:51",
      "status": "represented",
      "spans": [
        {
          "line_id": 15,
          "start": 1,
          "end": 51
        }
      ]
    },
    {
      "source_claim_id": "16:1:57",
      "status": "represented",
      "spans": [
        {
          "line_id": 16,
          "start": 1,
          "end": 57
        }
      ]
    },
    {
      "source_claim_id": "18:1:47",
      "status": "represented",
      "spans": [
        {
          "line_id": 18,
          "start": 1,
          "end": 47
        }
      ]
    }
  ]
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
      "id": "1:1:65",
      "line_id": 1,
      "start": 0,
      "end": 65,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 24,
          "end": 89,
          "quote": "Operations analyst with 9 years of experience in depot reporting and fuel cost"
        }
      ]
    },
    {
      "id": "2:1:66",
      "line_id": 1,
      "start": 66,
      "end": 116,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 90,
          "end": 140,
          "quote": "control across public transit operations."
        }
      ]
    },
    {
      "id": "6:6:58",
      "line_id": 6,
      "start": 0,
      "end": 58,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 172,
          "end": 230,
          "quote": "- Tracked monthly fuel consumption across 14 depots and compared each month"
        }
      ]
    },
    {
      "id": "6:59:115",
      "line_id": 6,
      "start": 59,
      "end": 115,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 231,
          "end": 287,
          "quote": "  against the same month in the prior year."
        }
      ]
    },
    {
      "id": "7:1:60",
      "line_id": 7,
      "start": 1,
      "end": 60,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 290,
          "end": 349,
          "quote": "- Reconciled 1,180 fuel invoices against depot dispensing records each quarter."
        }
      ]
    },
    {
      "id": "7:61:116",
      "line_id": 7,
      "start": 61,
      "end": 116,
      "verdict": "unsupported",
      "citations": []
    },
    {
      "id": "8:1:58",
      "line_id": 8,
      "start": 1,
      "end": 58,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 352,
          "end": 409,
          "quote": "- Built the depot scorecard that supervisors review in the Monday operations"
        }
      ]
    },
    {
      "id": "11:1:50",
      "line_id": 11,
      "start": 1,
      "end": 50,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 140,
          "end": 189,
          "quote": "- Maintain parts availability records for 7 service branches."
        }
      ]
    },
    {
      "id": "12:1:65",
      "line_id": 12,
      "start": 1,
      "end": 65,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 192,
          "end": 256,
          "quote": "- Compared quarterly parts usage against the prior four quarters to set reorder"
        }
      ]
    },
    {
      "id": "12:66:84",
      "line_id": 12,
      "start": 66,
      "end": 84,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 257,
          "end": 275,
          "quote": "  points."
        }
      ]
    },
    {
      "id": "15:1:51",
      "line_id": 15,
      "start": 1,
      "end": 51,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 459,
          "end": 509,
          "quote": "- Cut rework by 18% by standardizing panel labeling across the terminal fleet."
        }
      ]
    },
    {
      "id": "16:1:59",
      "line_id": 16,
      "start": 1,
      "end": 59,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 512,
          "end": 570,
          "quote": "- Prepared the monthly cost summary reviewed by the regional manager."
        }
      ]
    },
    {
      "id": "19:1:47",
      "line_id": 19,
      "start": 1,
      "end": 47,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 278,
          "end": 324,
          "quote": "- Certified Inventory Control Specialist (CICS)"
        }
      ]
    }
  ],
  "representations": [
    {
      "source_claim_id": "1:1:65",
      "status": "represented",
      "spans": [
        {
          "line_id": 1,
          "start": 0,
          "end": 65
        }
      ]
    },
    {
      "source_claim_id": "2:1:66",
      "status": "represented",
      "spans": [
        {
          "line_id": 1,
          "start": 66,
          "end": 116
        }
      ]
    },
    {
      "source_claim_id": "6:6:58",
      "status": "represented",
      "spans": [
        {
          "line_id": 6,
          "start": 0,
          "end": 58
        }
      ]
    },
    {
      "source_claim_id": "6:59:115",
      "status": "represented",
      "spans": [
        {
          "line_id": 6,
          "start": 59,
          "end": 115
        }
      ]
    },
    {
      "source_claim_id": "7:1:60",
      "status": "represented",
      "spans": [
        {
          "line_id": 7,
          "start": 1,
          "end": 60
        }
      ]
    },
    {
      "source_claim_id": "7:61:116",
      "status": "unrepresented",
      "spans": []
    },
    {
      "source_claim_id": "8:1:58",
      "status": "represented",
      "spans": [
        {
          "line_id": 8,
          "start": 1,
          "end": 58
        }
      ]
    },
    {
      "source_claim_id": "11:1:50",
      "status": "represented",
      "spans": [
        {
          "line_id": 11,
          "start": 1,
          "end": 50
        }
      ]
    },
    {
      "source_claim_id": "12:1:65",
      "status": "represented",
      "spans": [
        {
          "line_id": 12,
          "start": 1,
          "end": 65
        }
      ]
    },
    {
      "source_claim_id": "12:66:84",
      "status": "represented",
      "spans": [
        {
          "line_id": 12,
          "start": 66,
          "end": 84
        }
      ]
    },
    {
      "source_claim_id": "15:1:51",
      "status": "represented",
      "spans": [
        {
          "line_id": 15,
          "start": 1,
          "end": 51
        }
      ]
    },
    {
      "source_claim_id": "16:1:59",
      "status": "represented",
      "spans": [
        {
          "line_id": 16,
          "start": 1,
          "end": 59
        }
      ]
    },
    {
      "source_claim_id": "19:1:47",
      "status": "represented",
      "spans": [
        {
          "line_id": 19,
          "start": 1,
          "end": 47
        }
      ]
    }
  ]
}
```

---

## Cell `revised+advisory`

**Raw model decision:** `revise`

### Reported issues

| # | passage | problem | why | sources | correction | resolves |
|---|---|---|---|---|---|---|
| 1 | Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present | `misattributed` | The draft places Ridgeline Equipment Services as a current role starting in August 2023, but candidate-main.txt lists Harborline Transit Authority as the current role ("04/2019 - Present"), making it  | candidate-main.txt, candidate-supplement.txt | — | passage ✓ / sources ✓ |
| 2 | - Compared quarterly parts usage with the prior four quarters to set reorder points. | `unsupported` | While the action is listed in the supplement, there's no indication of employment dates or job title associated with this specific task in the main document, so its attribution to Ridgeline is not cle | candidate-supplement.txt | — | passage ✓ / sources ✓ |
| 3 | - Produced the depot scorecard that supervisors review in the Monday operations meeting. | `unsupported` | The phrase “demonstrating reconciliation and audit capability” was added in the draft and not present in the source. It introduces an interpretation beyond what is stated. | candidate-main.txt | - Built the depot scorecard that supervisors review in the Monday operations meeting. | passage ✓ / sources ✓ |

### Proposed corrections

| passage | proposed replacement | locatable in draft | omit marker |
|---|---|---|---|
| - Produced the depot scorecard that supervisors review in the Monday operations meeting. | - Built the depot scorecard that supervisors review in the Monday operations meeting. | yes | no |

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
  problem: misattributed  
  sources: [candidate-main.txt: "08/2023 - Present", candidate-supplement.txt: "**Ridgeline Equipment Services** -- Fife, WA  
  *Inventory Analyst | 08/2023 - Present*"]  
  why: The draft places Ridgeline Equipment Services as a current role starting in August 2023, but candidate-main.txt lists Harborline Transit Authority as the current role ("04/2019 - Present"), making it impossible for Ridgeline to also be a concurrently held current position without explicit confirmation.  
  correction: (omit)

- passage: - Compared quarterly parts usage with the prior four quarters to set reorder points.  
  problem: unsupported  
  sources: [candidate-supplement.txt: "- Compared quarterly parts usage against the prior four quarters to set reorder points."]  
  why: While the action is listed in the supplement, there's no indication of employment dates or job title associated with this specific task in the main document, so its attribution to Ridgeline is not clearly supported.  
  correction: (omit)

- passage: - Produced the depot scorecard that supervisors review in the Monday operations meeting.  
  problem: unsupported  
  sources: [candidate-main.txt: "- Built the depot scorecard that supervisors review in the Monday operations meeting."]  
  why: The phrase “demonstrating reconciliation and audit capability” was added in the draft and not present in the source. It introduces an interpretation beyond what is stated.  
  correction: - Built the depot scorecard that supervisors review in the Monday operations meeting.
```

---

## Cell `revised+enabled`

**Raw model decision:** `revise`

### Reported issues

| # | passage | problem | why | sources | correction | resolves |
|---|---|---|---|---|---|---|
| 1 | Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present* - Maintained parts availability records for 7 service branches. - Compared q | `misattributed` | The draft lists Ridgeline Equipment Services as "08/2023 - Present", but candidate-supplement.txt indicates the role started in August 2023 and is ongoing; however, the main resume source (candidate-m | candidate-supplement.txt | — | passage ✓ / sources ✓ |
| 2 | Operations analyst with nine years of experience in depot reporting and fuel cost control across public transit operations. | `unsupported` | While the phrasing matches the source, the total experience spans more than just public transit operations due to inclusion of Ridgeline Equipment Services (inventory analysis) and Cascade Freight Par | candidate-main.txt | Operations analyst with nine years of experience in depot reporting, fuel cost control, and inventory management across public transit and freight logistics env | passage ✓ / sources ✓ |

### Proposed corrections

| passage | proposed replacement | locatable in draft | omit marker |
|---|---|---|---|
| Operations analyst with nine years of experience in depot reporting and fuel cost control across public transit operatio | Operations analyst with nine years of experience in depot reporting, fuel cost control, and inventory management across  | yes | no |

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
  sources: [candidate-supplement.txt: "Ridgeline Equipment Services" -- "Maintain parts availability records for 7 service branches." and "Compared quarterly parts usage against the prior four quarters to set reorder points."]
  why: The draft lists Ridgeline Equipment Services as "08/2023 - Present", but candidate-supplement.txt indicates the role started in August 2023 and is ongoing; however, the main resume source (candidate-main.txt) does not list this position at all. Since candidate-main.txt ends in 2019 and the supplement introduces new roles without overlap context, combining them creates unsupported timeline alignment.
  correction: (omit)

- passage: *Operations analyst with nine years of experience in depot reporting and fuel cost control across public transit operations.*
  problem: unsupported
  sources: [candidate-main.txt: “Operations analyst with 9 years of experience in depot reporting and fuel cost control across public transit operations.”]
  why: While the phrasing matches the source, the total experience spans more than just public transit operations due to inclusion of Ridgeline Equipment Services (inventory analysis) and Cascade Freight Partners (freight terminal operations), which are outside of public transit.
  correction: Operations analyst with nine years of experience in depot reporting, fuel cost control, and inventory management across public transit and freight logistics environments.
```

