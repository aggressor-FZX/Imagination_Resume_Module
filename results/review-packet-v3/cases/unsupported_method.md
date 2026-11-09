# Case: `unsupported_method`

**Declared mutation:** `append_unsourced_method` — Appends 'and targeted quality audits'. Neither source states it.

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
- Cut rework by 18% through standardizing panel labeling and targeted quality audits across the terminal fleet.
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
      "id": "1:1:0:86",
      "line_id": "1",
      "start": 0,
      "end": 86,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 25,
          "end": 110,
          "quote": "Operations analyst with 9 years of experience in depot reporting and fuel cost control across public transit operations."
        }
      ]
    },
    {
      "id": "5:5:0:55",
      "line_id": "5",
      "start": 0,
      "end": 55,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 142,
          "end": 197,
          "quote": "Fuel Reporting Analyst | 04/2019 - Present"
        }
      ]
    },
    {
      "id": "6:6:0:93",
      "line_id": "6",
      "start": 0,
      "end": 93,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 199,
          "end": 291,
          "quote": "- Tracked monthly fuel consumption across 14 depots and compared each month against the same month in the prior year."
        }
      ]
    },
    {
      "id": "7:7:0:86",
      "line_id": "7",
      "start": 0,
      "end": 86,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 293,
          "end": 378,
          "quote": "- Reconciled 1,180 fuel invoices against depot dispensing records each quarter."
        }
      ]
    },
    {
      "id": "8:8:0:85",
      "line_id": "8",
      "start": 0,
      "end": 85,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 380,
          "end": 464,
          "quote": "- Built the depot scorecard that supervisors review in the Monday operations meeting."
        }
      ]
    },
    {
      "id": "10:10:0:59",
      "line_id": "10",
      "start": 0,
      "end": 59,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 25,
          "end": 84,
          "quote": "Inventory Analyst | 08/2023 - Present"
        }
      ]
    },
    {
      "id": "11:11:0:60",
      "line_id": "11",
      "start": 0,
      "end": 60,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 86,
          "end": 146,
          "quote": "- Maintain parts availability records for 7 service branches."
        }
      ]
    },
    {
      "id": "12:12:0:94",
      "line_id": "12",
      "start": 0,
      "end": 94,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 148,
          "end": 242,
          "quote": "- Compared quarterly parts usage against the prior four quarters to set reorder points."
        }
      ]
    },
    {
      "id": "14:14:0:59",
      "line_id": "14",
      "start": 0,
      "end": 59,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 493,
          "end": 552,
          "quote": "Cost Reporting Coordinator | 07/2015 - 03/2019"
        }
      ]
    },
    {
      "id": "15:15:0:110",
      "line_id": "15",
      "start": 0,
      "end": 110,
      "verdict": "unsupported",
      "citations": []
    },
    {
      "id": "16:16:0:76",
      "line_id": "16",
      "start": 0,
      "end": 76,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 582,
          "end": 657,
          "quote": "- Prepared the monthly cost summary reviewed by the regional manager."
        }
      ]
    },
    {
      "id": "18:18:0:50",
      "line_id": "18",
      "start": 0,
      "end": 50,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 244,
          "end": 294,
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
          "end": 86
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:2",
      "status": "represented",
      "spans": [
        {
          "line_id": "5",
          "start": 0,
          "end": 55
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:3",
      "status": "represented",
      "spans": [
        {
          "line_id": "6",
          "start": 0,
          "end": 93
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:4",
      "status": "represented",
      "spans": [
        {
          "line_id": "7",
          "start": 0,
          "end": 86
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:5",
      "status": "represented",
      "spans": [
        {
          "line_id": "8",
          "start": 0,
          "end": 85
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:1",
      "status": "represented",
      "spans": [
        {
          "line_id": "10",
          "start": 0,
          "end": 59
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:2",
      "status": "represented",
      "spans": [
        {
          "line_id": "11",
          "start": 0,
          "end": 60
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:3",
      "status": "represented",
      "spans": [
        {
          "line_id": "12",
          "start": 0,
          "end": 94
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:6",
      "status": "represented",
      "spans": [
        {
          "line_id": "14",
          "start": 0,
          "end": 59
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:7",
      "status": "partially_represented",
      "spans": [
        {
          "line_id": "15",
          "start": 0,
          "end": 110
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:8",
      "status": "represented",
      "spans": [
        {
          "line_id": "16",
          "start": 0,
          "end": 76
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:4",
      "status": "represented",
      "spans": [
        {
          "line_id": "18",
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
      "id": "1:1:0:86",
      "line_id": "1",
      "start": 0,
      "end": 86,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 25,
          "end": 111,
          "quote": "Operations analyst with 9 years of experience in depot reporting and fuel cost control across public transit operations."
        }
      ]
    },
    {
      "id": "5:5:0:50",
      "line_id": "5",
      "start": 0,
      "end": 50,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 141,
          "end": 191,
          "quote": "**Harborline Transit Authority** -- Tacoma, WA"
        }
      ]
    },
    {
      "id": "6:6:0:48",
      "line_id": "6",
      "start": 0,
      "end": 48,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 194,
          "end": 240,
          "quote": "*Fuel Reporting Analyst | 04/2019 - Present*"
        }
      ]
    },
    {
      "id": "7:7:0:88",
      "line_id": "7",
      "start": 0,
      "end": 88,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 243,
          "end": 331,
          "quote": "- Tracked monthly fuel consumption across 14 depots and compared each month against the same month in the prior year."
        }
      ]
    },
    {
      "id": "8:8:0:84",
      "line_id": "8",
      "start": 0,
      "end": 84,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 334,
          "end": 416,
          "quote": "- Reconciled 1,180 fuel invoices against depot dispensing records each quarter."
        }
      ]
    },
    {
      "id": "9:9:0:85",
      "line_id": "9",
      "start": 0,
      "end": 85,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 419,
          "end": 502,
          "quote": "- Built the depot scorecard that supervisors review in the Monday operations meeting."
        }
      ]
    },
    {
      "id": "11:11:0:49",
      "line_id": "11",
      "start": 0,
      "end": 49,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 93,
          "end": 142,
          "quote": "**Ridgeline Equipment Services** -- Fife, WA"
        }
      ]
    },
    {
      "id": "12:12:0:45",
      "line_id": "12",
      "start": 0,
      "end": 45,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 145,
          "end": 188,
          "quote": "*Inventory Analyst | 08/2023 - Present*"
        }
      ]
    },
    {
      "id": "13:13:0:57",
      "line_id": "13",
      "start": 0,
      "end": 57,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 191,
          "end": 246,
          "quote": "- Maintain parts availability records for 7 service branches."
        }
      ]
    },
    {
      "id": "14:14:0:90",
      "line_id": "14",
      "start": 0,
      "end": 90,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 249,
          "end": 337,
          "quote": "- Compared quarterly parts usage against the prior four quarters to set reorder points."
        }
      ]
    },
    {
      "id": "16:16:0:50",
      "line_id": "16",
      "start": 0,
      "end": 50,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 443,
          "end": 493,
          "quote": "**Cascade Freight Partners** -- Kent, WA"
        }
      ]
    },
    {
      "id": "17:17:0:54",
      "line_id": "17",
      "start": 0,
      "end": 54,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 496,
          "end": 548,
          "quote": "*Cost Reporting Coordinator | 07/2015 - 03/2019*"
        }
      ]
    },
    {
      "id": "18:18:0:100",
      "line_id": "18",
      "start": 0,
      "end": 100,
      "verdict": "unsupported",
      "citations": []
    },
    {
      "id": "19:19:0:74",
      "line_id": "19",
      "start": 0,
      "end": 74,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 578,
          "end": 650,
          "quote": "- Prepared the monthly cost summary reviewed by the regional manager."
        }
      ]
    },
    {
      "id": "21:21:0:51",
      "line_id": "21",
      "start": 0,
      "end": 51,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 340,
          "end": 391,
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
          "end": 86
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:2",
      "status": "represented",
      "spans": [
        {
          "line_id": "1",
          "start": 0,
          "end": 86
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:4",
      "status": "represented",
      "spans": [
        {
          "line_id": "5",
          "start": 0,
          "end": 50
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:5",
      "status": "represented",
      "spans": [
        {
          "line_id": "6",
          "start": 0,
          "end": 48
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:6",
      "status": "represented",
      "spans": [
        {
          "line_id": "7",
          "start": 0,
          "end": 88
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:7",
      "status": "represented",
      "spans": [
        {
          "line_id": "8",
          "start": 0,
          "end": 84
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:8",
      "status": "represented",
      "spans": [
        {
          "line_id": "9",
          "start": 0,
          "end": 85
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:10",
      "status": "represented",
      "spans": [
        {
          "line_id": "16",
          "start": 0,
          "end": 50
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:11",
      "status": "represented",
      "spans": [
        {
          "line_id": "17",
          "start": 0,
          "end": 54
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:12",
      "status": "represented",
      "spans": [
        {
          "line_id": "19",
          "start": 0,
          "end": 74
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:1",
      "status": "represented",
      "spans": [
        {
          "line_id": "11",
          "start": 0,
          "end": 49
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:2",
      "status": "represented",
      "spans": [
        {
          "line_id": "12",
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
          "line_id": "13",
          "start": 0,
          "end": 57
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:4",
      "status": "represented",
      "spans": [
        {
          "line_id": "14",
          "start": 0,
          "end": 90
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:6",
      "status": "represented",
      "spans": [
        {
          "line_id": "21",
          "start": 0,
          "end": 51
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
| 1 | Cut rework by 18% through standardizing panel labeling and targeted quality audits across the terminal fleet. | `unsupported` | The draft adds "targeted quality audits" which is not mentioned in the source and constitutes an unauthorized addition of a specific method. | candidate-main.txt | Cut rework by 18% by standardizing panel labeling across the terminal fleet. | passage ✓ / sources ✓ |

### Proposed corrections

| passage | proposed replacement | locatable in draft | omit marker |
|---|---|---|---|
| Cut rework by 18% through standardizing panel labeling and targeted quality audits across the terminal fleet. | Cut rework by 18% by standardizing panel labeling across the terminal fleet. | yes | no |

### Re-review

**Raw re-review decision:** `revise`

```text
Decision: revise

Issues:
- passage: **Ridgeline Equipment Services** -- Fife, WA  
  *Inventory Analyst | 08/2023 - Present*  
  problem: misattributed  
  sources: [candidate-main.txt, candidate-supplement.txt]  
  why: The draft assigns Ridgeline Equipment Services to Fife, WA, but candidate-main.txt lists "Ridgeline Equipment Services -- Fife, WA" under a different role and timeframe than what is presented in the draft. Specifically, the draft implies current employment (08/2023 – Present), while candidate-main.txt indicates past employment with Harborline Transit Authority during overlapping periods. This creates confusion about timeline accuracy and role attribution.  
  correction: (omit)

- passage: - Maintained parts availability records for 7 service branches.  
  problem: unsupported  
  sources: [candidate-supplement.txt: "Maintain parts availability records for 7 service branches."]  
  why: While this statement appears in candidate-supplement.txt, it is listed under Ridgeline Equipment Services, which conflicts with the employment history provided in candidate-main.txt. Therefore, its inclusion without resolving the conflicting employer attribution makes it misleading.  
  correction: (omit)

- passage: - Compared quarterly parts usage with the prior four quarters to set reorder points.  
  problem: unsupported  
  sources: [candidate-supplement.txt: "Compared quarterly parts usage against the prior four quarters to set reorder points."]  
  why: Same as above—this detail originates from candidate-supplement.txt but is tied to Ridgeline Equipment Services, whose placement in the draft conflicts with other verified roles. Without clarification on accurate dates or separation of roles, this line cannot be confirmed as correctly attributed.  
  correction: (omit)
```

### Binding-gate outcome (offline replay)

- initial decision: `revise`
- repair status: `proposed_exact_owned_source_bullet`
- re-review decision: `revise`
- gate result: **WITHHOLD** (`re_review_not_approved`)

> A repair was applied and re-reviewed, and the re-review did not approve the corrected document.

### Raw first response

```text
Decision: revise

Issues:
- passage: "Cut rework by 18% through standardizing panel labeling and targeted quality audits across the terminal fleet."
  problem: unsupported
  sources: [candidate-main.txt: "Cut rework by 18% by standardizing panel labeling across the terminal fleet."]
  why: The draft adds "targeted quality audits" which is not mentioned in the source and constitutes an unauthorized addition of a specific method.
  correction: "Cut rework by 18% by standardizing panel labeling across the terminal fleet."
```

---

## Cell `revised+enabled`

**Raw model decision:** `revise`

### Reported issues

| # | passage | problem | why | sources | correction | resolves |
|---|---|---|---|---|---|---|
| 1 | Cut rework by 18% through standardizing panel labeling and targeted quality audits across the terminal fleet. | `unsupported` | The draft adds "targeted quality audits" which is not mentioned in the source. This introduces an unsupported method to explain the rework reduction. | candidate-main.txt | Cut rework by 18% by standardizing panel labeling across the terminal fleet. | passage ✓ / sources ✓ |

### Proposed corrections

| passage | proposed replacement | locatable in draft | omit marker |
|---|---|---|---|
| Cut rework by 18% through standardizing panel labeling and targeted quality audits across the terminal fleet. | Cut rework by 18% by standardizing panel labeling across the terminal fleet. | yes | no |

### Re-review

**Raw re-review decision:** `revise`

```text
Decision: revise

Issues:
- passage: **Ridgeline Equipment Services** -- Fife, WA  
  *Inventory Analyst | 08/2023 - Present*  
  problem: unsupported  
  sources: [candidate-supplement.txt: "08/2023 - Present"]  
  why: The draft lists Ridgeline Equipment Services as "08/2023 - Present", but candidate-main.txt shows Harborline Transit Authority as "04/2019 - Present". This creates a timeline conflict where the candidate would be working two jobs simultaneously without indication of兼职 (part-time) or contract roles.  
  correction: (omit)

- passage: - Maintained parts availability records for 7 service branches.  
  problem: omitted  
  sources: [candidate-supplement.txt: "- Maintain parts availability records for 7 service branches."]  
  why: The verb tense should match the rest of the resume (past tense), and the original uses present tense ("Maintain") which is inconsistent.  
  correction: - Maintained parts availability records for 7 service branches.

- passage: - Compared quarterly parts usage with the prior four quarters to set reorder points.  
  problem: omitted  
  sources: [candidate-supplement.txt: "- Compared quarterly parts usage against the prior four quarters to set reorder points."]  
  why: Minor wording variation acceptable, but phrasing should align more closely to preserve clarity and voice.  
  correction: - Compared quarterly parts usage against the prior four quarters to establish reorder points.
```

### Binding-gate outcome (offline replay)

- initial decision: `revise`
- repair status: `proposed_exact_owned_source_bullet`
- re-review decision: `revise`
- gate result: **WITHHOLD** (`re_review_not_approved`)

> A repair was applied and re-reviewed, and the re-review did not approve the corrected document.

### Raw first response

```text
Decision: revise

Issues:
- passage: Cut rework by 18% through standardizing panel labeling and targeted quality audits across the terminal fleet.
  problem: unsupported
  sources: [candidate-main.txt: "Cut rework by 18% by standardizing panel labeling across the terminal fleet."]
  why: The draft adds "targeted quality audits" which is not mentioned in the source. This introduces an unsupported method to explain the rework reduction.
  correction: Cut rework by 18% by standardizing panel labeling across the terminal fleet.
```

