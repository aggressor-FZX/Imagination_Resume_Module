# Case: `missing_fact`

**Declared mutation:** `delete_sourced_bullet` — Deletes the 1,180-invoice reconciliation bullet entirely.

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
      "id": "1:1:0:79",
      "line_id": "1",
      "start": 0,
      "end": 79,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 0,
          "end": 86,
          "quote": "Operations analyst with 9 years of experience in depot reporting and fuel cost control across public transit operations."
        }
      ]
    },
    {
      "id": "5:1:0:41",
      "line_id": "5",
      "start": 0,
      "end": 41,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 130,
          "end": 171,
          "quote": "**Harborline Transit Authority** -- Tacoma, WA"
        }
      ]
    },
    {
      "id": "6:1:0:45",
      "line_id": "6",
      "start": 0,
      "end": 45,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 172,
          "end": 217,
          "quote": "*Fuel Reporting Analyst | 04/2019 - Present*"
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
          "start": 218,
          "end": 310,
          "quote": "- Tracked monthly fuel consumption across 14 depots and compared each month against the same month in the prior year."
        }
      ]
    },
    {
      "id": "8:1:0:82",
      "line_id": "8",
      "start": 0,
      "end": 82,
      "verdict": "unsupported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 396,
          "end": 488,
          "quote": "- Built the depot scorecard that supervisors review in the Monday operations meeting."
        }
      ]
    },
    {
      "id": "10:1:0:39",
      "line_id": "10",
      "start": 0,
      "end": 39,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 32,
          "end": 71,
          "quote": "**Ridgeline Equipment Services** -- Fife, WA"
        }
      ]
    },
    {
      "id": "11:1:0:43",
      "line_id": "11",
      "start": 0,
      "end": 43,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 72,
          "end": 115,
          "quote": "*Inventory Analyst | 08/2023 - Present*"
        }
      ]
    },
    {
      "id": "12:1:0:54",
      "line_id": "12",
      "start": 0,
      "end": 54,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 116,
          "end": 170,
          "quote": "- Maintain parts availability records for 7 service branches."
        }
      ]
    },
    {
      "id": "13:1:0:81",
      "line_id": "13",
      "start": 0,
      "end": 81,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 171,
          "end": 264,
          "quote": "- Compared quarterly parts usage against the prior four quarters to set reorder points."
        }
      ]
    },
    {
      "id": "15:1:0:37",
      "line_id": "15",
      "start": 0,
      "end": 37,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 490,
          "end": 527,
          "quote": "**Cascade Freight Partners** -- Kent, WA"
        }
      ]
    },
    {
      "id": "16:1:0:53",
      "line_id": "16",
      "start": 0,
      "end": 53,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 528,
          "end": 581,
          "quote": "*Cost Reporting Coordinator | 07/2015 - 03/2019*"
        }
      ]
    },
    {
      "id": "17:1:0:76",
      "line_id": "17",
      "start": 0,
      "end": 76,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 582,
          "end": 658,
          "quote": "- Cut rework by 18% by standardizing panel labeling across the terminal fleet."
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
          "start": 659,
          "end": 726,
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
          "start": 266,
          "end": 316,
          "quote": "## Certifications\n-Certified Inventory Control Specialist (CICS)"
        }
      ]
    }
  ],
  "representations": [
    {
      "source_claim_id": "candidate-main.txt:1:0:86",
      "status": "represented",
      "spans": [
        {
          "line_id": "1",
          "start": 0,
          "end": 79
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:5:0:41",
      "status": "represented",
      "spans": [
        {
          "line_id": "5",
          "start": 0,
          "end": 41
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:6:0:45",
      "status": "represented",
      "spans": [
        {
          "line_id": "6",
          "start": 0,
          "end": 45
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:7:0:92",
      "status": "represented",
      "spans": [
        {
          "line_id": "7",
          "start": 0,
          "end": 80
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:9:0:86",
      "status": "unrepresented",
      "spans": []
    },
    {
      "source_claim_id": "candidate-main.txt:11:0:37",
      "status": "represented",
      "spans": [
        {
          "line_id": "15",
          "start": 0,
          "end": 37
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:12:0:53",
      "status": "represented",
      "spans": [
        {
          "line_id": "16",
          "start": 0,
          "end": 53
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:13:0:76",
      "status": "represented",
      "spans": [
        {
          "line_id": "17",
          "start": 0,
          "end": 76
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:14:0:67",
      "status": "represented",
      "spans": [
        {
          "line_id": "18",
          "start": 0,
          "end": 67
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:1:0:39",
      "status": "represented",
      "spans": [
        {
          "line_id": "10",
          "start": 0,
          "end": 39
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:2:0:43",
      "status": "represented",
      "spans": [
        {
          "line_id": "11",
          "start": 0,
          "end": 43
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:3:0:54",
      "status": "represented",
      "spans": [
        {
          "line_id": "12",
          "start": 0,
          "end": 54
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:4:0:81",
      "status": "represented",
      "spans": [
        {
          "line_id": "13",
          "start": 0,
          "end": 81
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:6:0:50",
      "status": "represented",
      "spans": [
        {
          "line_id": "21",
          "start": 0,
          "end": 50
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
      "id": "1:1:0:67",
      "line_id": "1",
      "start": 0,
      "end": 67,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 0,
          "end": 67,
          "quote": "Operations analyst with 9 years of experience in depot reporting and fuel cost"
        }
      ]
    },
    {
      "id": "1:1:68:71",
      "line_id": "1",
      "start": 68,
      "end": 71,
      "verdict": "unsupported",
      "citations": []
    },
    {
      "id": "4:4:0:41",
      "line_id": "4",
      "start": 0,
      "end": 41,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 120,
          "end": 161,
          "quote": "**Harborline Transit Authority** -- Tacoma, WA"
        }
      ]
    },
    {
      "id": "5:5:0:47",
      "line_id": "5",
      "start": 0,
      "end": 47,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 162,
          "end": 209,
          "quote": "*Fuel Reporting Analyst | 04/2019 - Present*"
        }
      ]
    },
    {
      "id": "6:6:0:77",
      "line_id": "6",
      "start": 0,
      "end": 77,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 210,
          "end": 287,
          "quote": "- Tracked monthly fuel consumption across 14 depots and compared each month"
        }
      ]
    },
    {
      "id": "7:7:0:60",
      "line_id": "7",
      "start": 0,
      "end": 60,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 288,
          "end": 348,
          "quote": "  against the same month in the prior year."
        }
      ]
    },
    {
      "id": "8:8:0:69",
      "line_id": "8",
      "start": 0,
      "end": 69,
      "verdict": "unsupported",
      "citations": []
    },
    {
      "id": "9:9:0:71",
      "line_id": "9",
      "start": 0,
      "end": 71,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 391,
          "end": 462,
          "quote": "- Built the depot scorecard that supervisors review in the Monday operations"
        }
      ]
    },
    {
      "id": "10:10:0:23",
      "line_id": "10",
      "start": 0,
      "end": 23,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 463,
          "end": 486,
          "quote": "  meeting."
        }
      ]
    },
    {
      "id": "12:12:0:43",
      "line_id": "12",
      "start": 0,
      "end": 43,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 37,
          "end": 80,
          "quote": "**Ridgeline Equipment Services** -- Fife, WA"
        }
      ]
    },
    {
      "id": "13:13:0:45",
      "line_id": "13",
      "start": 0,
      "end": 45,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 81,
          "end": 126,
          "quote": "*Inventory Analyst | 08/2023 - Present*"
        }
      ]
    },
    {
      "id": "14:14:0:53",
      "line_id": "14",
      "start": 0,
      "end": 53,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 127,
          "end": 180,
          "quote": "- Maintain parts availability records for 7 service branches."
        }
      ]
    },
    {
      "id": "15:15:0:76",
      "line_id": "15",
      "start": 0,
      "end": 76,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 181,
          "end": 257,
          "quote": "- Compared quarterly parts usage against the prior four quarters to set reorder"
        }
      ]
    },
    {
      "id": "16:16:0:26",
      "line_id": "16",
      "start": 0,
      "end": 26,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 258,
          "end": 284,
          "quote": "  points."
        }
      ]
    },
    {
      "id": "18:18:0:40",
      "line_id": "18",
      "start": 0,
      "end": 40,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 488,
          "end": 528,
          "quote": "**Cascade Freight Partners** -- Kent, WA"
        }
      ]
    },
    {
      "id": "19:19:0:55",
      "line_id": "19",
      "start": 0,
      "end": 55,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 529,
          "end": 584,
          "quote": "*Cost Reporting Coordinator | 07/2015 - 03/2019*"
        }
      ]
    },
    {
      "id": "20:20:0:69",
      "line_id": "20",
      "start": 0,
      "end": 69,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 585,
          "end": 654,
          "quote": "- Cut rework by 18% by standardizing panel labeling across the terminal fleet."
        }
      ]
    },
    {
      "id": "21:21:0:61",
      "line_id": "21",
      "start": 0,
      "end": 61,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 655,
          "end": 716,
          "quote": "- Prepared the monthly cost summary reviewed by the regional manager."
        }
      ]
    },
    {
      "id": "23:23:0:50",
      "line_id": "23",
      "start": 0,
      "end": 50,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 286,
          "end": 336,
          "quote": "## Certifications\n-Certified Inventory Control Specialist (CICS)"
        }
      ]
    }
  ],
  "representations": [
    {
      "source_claim_id": "candidate-main.txt:1",
      "status": "represented",
      "spans": [
        {
          "line_id": "1",
          "start": 0,
          "end": 67
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:4",
      "status": "represented",
      "spans": [
        {
          "line_id": "4",
          "start": 0,
          "end": 41
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:5",
      "status": "represented",
      "spans": [
        {
          "line_id": "5",
          "start": 0,
          "end": 47
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:6",
      "status": "represented",
      "spans": [
        {
          "line_id": "6",
          "start": 0,
          "end": 77
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:7",
      "status": "represented",
      "spans": [
        {
          "line_id": "7",
          "start": 0,
          "end": 60
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:9",
      "status": "represented",
      "spans": [
        {
          "line_id": "9",
          "start": 0,
          "end": 71
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:10",
      "status": "represented",
      "spans": [
        {
          "line_id": "10",
          "start": 0,
          "end": 23
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:1",
      "status": "represented",
      "spans": [
        {
          "line_id": "12",
          "start": 0,
          "end": 43
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:2",
      "status": "represented",
      "spans": [
        {
          "line_id": "13",
          "start": 0,
          "end": 45
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:3",
      "status": "represented",
      "spans": [
        {
          "line_id": "14",
          "start": 0,
          "end": 53
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:4",
      "status": "represented",
      "spans": [
        {
          "line_id": "15",
          "start": 0,
          "end": 76
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:5",
      "status": "represented",
      "spans": [
        {
          "line_id": "16",
          "start": 0,
          "end": 26
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:18",
      "status": "represented",
      "spans": [
        {
          "line_id": "18",
          "start": 0,
          "end": 40
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:19",
      "status": "represented",
      "spans": [
        {
          "line_id": "19",
          "start": 0,
          "end": 55
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:20",
      "status": "represented",
      "spans": [
        {
          "line_id": "20",
          "start": 0,
          "end": 69
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:21",
      "status": "represented",
      "spans": [
        {
          "line_id": "21",
          "start": 0,
          "end": 61
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:7",
      "status": "represented",
      "spans": [
        {
          "line_id": "23",
          "start": 0,
          "end": 50
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
| 1 | Inventory Analyst \| 08/2023 - Present | `misattributed` | The draft correctly lists Ridgeline Equipment Services and the role of Inventory Analyst, but candidate-supplement.txt shows the dates as 08/2023 - Present, while main source (candidate-main.txt) does | candidate-supplement.txt | (no change needed for header; tasks below are addressed separately) | passage ✓ / sources ✓ |
| 2 | - Maintained parts availability records for 7 service branches. | `unsupported` | The verb tense differs ("Maintained" vs "Maintain"), which changes the implication about ongoing action. Since the employment period includes "Present," maintaining past tense is acceptable only if de | candidate-supplement.txt | - Maintain parts availability records for 7 service branches. | passage ✓ / sources ✓ |
| 3 | - Compared quarterly parts usage with the prior four quarters to set reorder points. | `unsupported` | The preposition used ("with" instead of "against") alters the meaning slightly and unnecessarily. It's better to preserve the original phrasing for accuracy. | candidate-supplement.txt | - Compared quarterly parts usage against the prior four quarters to set reorder points. | passage ✓ / sources ✓ |

### Proposed corrections

| passage | proposed replacement | locatable in draft | omit marker |
|---|---|---|---|
| Inventory Analyst | 08/2023 - Present | (no change needed for header; tasks below are addressed separately) | yes | no |
| - Maintained parts availability records for 7 service branches. | - Maintain parts availability records for 7 service branches. | yes | no |
| - Compared quarterly parts usage with the prior four quarters to set reorder points. | - Compared quarterly parts usage against the prior four quarters to set reorder points. | yes | no |

### Re-review

**Raw re-review decision:** `revise`

```text
Decision: revise

Issues:
- passage: Reconciled 1,180 fuel invoices against depot dispensing records each quarter.
  problem: omitted
  sources: [candidate-main.txt: "Reconciled 1,180 fuel invoices against depot dispensing records each quarter."]
  why: The draft omits a key accomplishment explicitly listed in the source. While other statements are faithful paraphrases, this specific activity is missing without justification.
  correction: Reconciled 1,180 fuel invoices against depot dispensing records each quarter.

- passage: Built the depot scorecard that supervisors review in the Monday operations meeting.
  problem: unsupported
  sources: [candidate-main.txt: "Built the depot scorecard that supervisors review in the Monday operations meeting."]
  why: This statement appears to be an unsupported extension because it implies a formal construction ("built") rather than reporting or analysis. The source says "Built the depot scorecard", but the draft's phrasing ("Produced the depot scorecard") changes the meaning slightly and lacks support for implying authorship of a tool versus contributing content.
  correction: Created the depot scorecard used by supervisors during the Monday operations meeting.

- passage: Compared quarterly parts usage against the prior four quarters to set reorder points.
  problem: misattributed
  sources: [candidate-supplement.txt: "- Compared quarterly parts usage against the prior four quarters to set reorder points."]
  why: Although the action is accurately described, it is attributed to Ridgeline Equipment Services, where there is no indication that reorder point setting was part of the role based on provided documents.
  correction: (omit)

Note: The certification section and professional summary are acceptable as they align with the source materials. However, these issues affect material accuracy and attribution.
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
- passage: *Inventory Analyst | 08/2023 - Present*
  problem: misattributed
  sources: [candidate-supplement.txt: "Ridgeline Equipment Services -- Fife, WA", "Inventory Analyst | 08/2023 - Present"]
  why: The draft correctly lists Ridgeline Equipment Services and the role of Inventory Analyst, but candidate-supplement.txt shows the dates as 08/2023 - Present, while main source (candidate-main.txt) does not list this employer or role at all. However, since the supplemental document is treated as valid data, this entry should remain but needs alignment with its tasks.
  correction: (no change needed for header; tasks below are addressed separately)

- passage: - Maintained parts availability records for 7 service branches.
  problem: unsupported
  sources: [candidate-supplement.txt: "- Maintain parts availability records for 7 service branches."]
  why: The verb tense differs ("Maintained" vs "Maintain"), which changes the implication about ongoing action. Since the employment period includes "Present," maintaining past tense is acceptable only if describing completed actions. Here, the original uses present tense consistently.
  correction: - Maintain parts availability records for 7 service branches.

- passage: - Compared quarterly parts usage with the prior four quarters to set reorder points.
  problem: unsupported
  sources: [candidate-supplement.txt: "- Compared quarterly parts usage against the prior four quarters to set reorder points."]
  why: The preposition used ("with" instead of "against") alters the meaning slightly and unnecessarily. It's better to preserve the original phrasing for accuracy.
  correction: - Compared quarterly parts usage against the prior four quarters to set reorder points.
```

---

## Cell `revised+enabled`

**Raw model decision:** `revise`

### Reported issues

| # | passage | problem | why | sources | correction | resolves |
|---|---|---|---|---|---|---|
| 1 | Reconciled 1,180 fuel invoices against depot dispensing records each quarter. | `omitted` | The draft removed a key factual detail present in the source that reflects a core responsibility and measurable activity. | candidate-main.txt | **Reconciled 1,180 fuel invoices against depot dispensing records each quarter.** | passage ✓ / sources ✓ |
| 2 | Built the depot scorecard that supervisors review in the Monday operations meeting. | `unsupported` | While the draft says "Produced the depot scorecard...", the original wording "Built the depot scorecard..." is more precise and directly attributable. Changing the verb alters the implied action witho | candidate-main.txt | **Built the depot scorecard that supervisors review in the Monday operations meeting.** | passage ✗ / sources ✓ |
| 3 | Compared quarterly parts usage against the prior four quarters to set reorder points. | `misattributed` | The draft changes “against” to “with,” which while minor, introduces unnecessary variation from the source phrasing without benefit and could affect clarity or searchability. | candidate-supplement.txt | **Compared quarterly parts usage against the prior four quarters to set reorder points.** | passage ✗ / sources ✓ |
| 4 | Cut rework by 18% by standardizing panel labeling across the terminal fleet. | `unsupported` | The draft changes “Cut rework” to “Reduced rework,” which alters the specificity of the accomplishment unnecessarily. | candidate-main.txt | **Cut rework by 18% by standardizing panel labeling across the terminal fleet.** | passage ✓ / sources ✓ |
| 5 | Prepared the monthly cost summary reviewed by the regional manager. | `unsupported` | The draft adds “that the regional manager reviewed,” changing the passive construction to active. This addition implies direct involvement not stated in the source. | candidate-main.txt | **Prepared the monthly cost summary reviewed by the regional manager.** | passage ✗ / sources ✓ |

### Proposed corrections

| passage | proposed replacement | locatable in draft | omit marker |
|---|---|---|---|
| Reconciled 1,180 fuel invoices against depot dispensing records each quarter. | **Reconciled 1,180 fuel invoices against depot dispensing records each quarter.** | NO | no |
| Built the depot scorecard that supervisors review in the Monday operations meeting. | **Built the depot scorecard that supervisors review in the Monday operations meeting.** | NO | no |
| Compared quarterly parts usage against the prior four quarters to set reorder points. | **Compared quarterly parts usage against the prior four quarters to set reorder points.** | NO | no |
| Cut rework by 18% by standardizing panel labeling across the terminal fleet. | **Cut rework by 18% by standardizing panel labeling across the terminal fleet.** | yes | no |
| Prepared the monthly cost summary reviewed by the regional manager. | **Prepared the monthly cost summary reviewed by the regional manager.** | NO | no |

### Re-review

_No re-review was performed in the retained run._

### Binding-gate outcome (offline replay)

- initial decision: `revise`
- repair status: `not_attempted`
- re-review decision: `none`
- gate result: **WITHHOLD** (`initial_review_unverifiable`)
- initial structural problems: `passage_not_in_draft:Built the depot scorecard that supervisors review in the Mon, passage_not_in_draft:Compared quarterly parts usage against the prior four quarte, passage_not_in_draft:Prepared the monthly cost summary reviewed by the regional m`

> The prose answer could not be used as a verdict: a structural problem remained after parsing (see the problems listed).

### Raw first response

```text
Decision: **revise**

Issues:
- passage: *Reconciled 1,180 fuel invoices against depot dispensing records each quarter.*
  problem: **omitted**
  sources: [candidate-main.txt: "Reconciled 1,180 fuel invoices against depot dispensing records each quarter."]
  why: The draft removed a key factual detail present in the source that reflects a core responsibility and measurable activity.
  correction: **Reconciled 1,180 fuel invoices against depot dispensing records each quarter.**

- passage: *Built the depot scorecard that supervisors review in the Monday operations meeting.*
  problem: **unsupported**
  sources: [candidate-main.txt: "Built the depot scorecard that supervisors review in the Monday operations meeting."]
  why: While the draft says "Produced the depot scorecard...", the original wording "Built the depot scorecard..." is more precise and directly attributable. Changing the verb alters the implied action without justification.
  correction: **Built the depot scorecard that supervisors review in the Monday operations meeting.**

- passage: *Compared quarterly parts usage against the prior four quarters to set reorder points.*
  problem: **misattributed**
  sources: [candidate-supplement.txt: "Compared quarterly parts usage against the prior four quarters to set reorder points."]
  why: The draft changes “against” to “with,” which while minor, introduces unnecessary variation from the source phrasing without benefit and could affect clarity or searchability.
  correction: **Compared quarterly parts usage against the prior four quarters to set reorder points.**

- passage: *Cut rework by 18% by standardizing panel labeling across the terminal fleet.*
  problem: **unsupported**
  sources: [candidate-main.txt: "Cut rework by 18% by standardizing panel labeling across the terminal fleet."]
  why: The draft changes “Cut rework” to “Reduced rework,” which alters the specificity of the accomplishment unnecessarily.
  correction: **Cut rework by 18% by standardizing panel labeling across the terminal fleet.**

- passage: *Prepared the monthly cost summary reviewed by the regional manager.*
  problem: **unsupported**
  sources: [candidate-main.txt: "Prepared the monthly cost summary reviewed by the regional manager."]
  why: The draft adds “that the regional manager reviewed,” changing the passive construction to active. This addition implies direct involvement not stated in the source.
  correction: **Prepared the monthly cost summary reviewed by the regional manager.**
```

