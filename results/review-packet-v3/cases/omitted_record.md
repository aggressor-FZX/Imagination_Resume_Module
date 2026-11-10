# Case: `omitted_record`

**Declared mutation:** `delete_whole_record` — Removes the entire Ridgeline Equipment Services record while candidate-supplement.txt still supplies it.

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
      "id": "1:1:0:80",
      "line_id": "1",
      "start": 0,
      "end": 80,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 24,
          "end": 104,
          "quote": "Operations analyst with 9 years of experience in depot reporting and fuel cost control across public transit operations."
        }
      ]
    },
    {
      "id": "4:4:0:46",
      "line_id": "4",
      "start": 0,
      "end": 46,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 133,
          "end": 179,
          "quote": "**Harborline Transit Authority** -- Tacoma, WA"
        }
      ]
    },
    {
      "id": "5:5:0:48",
      "line_id": "5",
      "start": 0,
      "end": 48,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 180,
          "end": 228,
          "quote": "*Fuel Reporting Analyst | 04/2019 - Present*"
        }
      ]
    },
    {
      "id": "6:6:0:76",
      "line_id": "6",
      "start": 0,
      "end": 76,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 229,
          "end": 305,
          "quote": "- Tracked monthly fuel consumption across 14 depots and compared each month"
        }
      ]
    },
    {
      "id": "7:7:0:58",
      "line_id": "7",
      "start": 0,
      "end": 58,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 306,
          "end": 364,
          "quote": "  against the same month in the prior year."
        }
      ]
    },
    {
      "id": "8:8:0:75",
      "line_id": "8",
      "start": 0,
      "end": 75,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 365,
          "end": 440,
          "quote": "- Reconciled 1,180 fuel invoices against depot dispensing records each quarter."
        }
      ]
    },
    {
      "id": "9:9:0:76",
      "line_id": "9",
      "start": 0,
      "end": 76,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 441,
          "end": 517,
          "quote": "- Built the depot scorecard that supervisors review in the Monday operations"
        }
      ]
    },
    {
      "id": "10:10:0:21",
      "line_id": "10",
      "start": 0,
      "end": 21,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 518,
          "end": 539,
          "quote": "  meeting."
        }
      ]
    },
    {
      "id": "12:12:0:44",
      "line_id": "12",
      "start": 0,
      "end": 44,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 543,
          "end": 587,
          "quote": "**Cascade Freight Partners** -- Kent, WA"
        }
      ]
    },
    {
      "id": "13:13:0:52",
      "line_id": "13",
      "start": 0,
      "end": 52,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 588,
          "end": 640,
          "quote": "*Cost Reporting Coordinator | 07/2015 - 03/2019*"
        }
      ]
    },
    {
      "id": "14:14:0:71",
      "line_id": "14",
      "start": 0,
      "end": 71,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 641,
          "end": 712,
          "quote": "- Cut rework by 18% by standardizing panel labeling across the terminal fleet."
        }
      ]
    },
    {
      "id": "15:15:0:69",
      "line_id": "15",
      "start": 0,
      "end": 69,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 713,
          "end": 782,
          "quote": "- Prepared the monthly cost summary reviewed by the regional manager."
        }
      ]
    },
    {
      "id": "17:17:0:49",
      "line_id": "17",
      "start": 0,
      "end": 49,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 176,
          "end": 225,
          "quote": "## Certifications\n- Certified Inventory Control Specialist (CICS)"
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
      "id": "1:1:0:80",
      "line_id": "1",
      "start": 0,
      "end": 80,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 24,
          "end": 104,
          "quote": "Operations analyst with 9 years of experience in depot reporting and fuel cost control across public transit operations."
        }
      ]
    },
    {
      "id": "4:4:0:46",
      "line_id": "4",
      "start": 0,
      "end": 46,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 133,
          "end": 179,
          "quote": "**Harborline Transit Authority** -- Tacoma, WA"
        }
      ]
    },
    {
      "id": "5:5:0:48",
      "line_id": "5",
      "start": 0,
      "end": 48,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 180,
          "end": 228,
          "quote": "*Fuel Reporting Analyst | 04/2019 - Present*"
        }
      ]
    },
    {
      "id": "6:6:0:78",
      "line_id": "6",
      "start": 0,
      "end": 78,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 229,
          "end": 307,
          "quote": "- Tracked monthly fuel consumption across 14 depots and compared each month"
        }
      ]
    },
    {
      "id": "7:7:0:70",
      "line_id": "7",
      "start": 0,
      "end": 70,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 332,
          "end": 402,
          "quote": "- Reconciled 1,180 fuel invoices against depot dispensing records each quarter."
        }
      ]
    },
    {
      "id": "8:8:0:81",
      "line_id": "8",
      "start": 0,
      "end": 81,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 403,
          "end": 484,
          "quote": "- Built the depot scorecard that supervisors review in the Monday operations meeting."
        }
      ]
    },
    {
      "id": "10:10:0:45",
      "line_id": "10",
      "start": 0,
      "end": 45,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 486,
          "end": 531,
          "quote": "**Cascade Freight Partners** -- Kent, WA"
        }
      ]
    },
    {
      "id": "11:11:0:54",
      "line_id": "11",
      "start": 0,
      "end": 54,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 532,
          "end": 586,
          "quote": "*Cost Reporting Coordinator | 07/2015 - 03/2019*"
        }
      ]
    },
    {
      "id": "12:12:0:74",
      "line_id": "12",
      "start": 0,
      "end": 74,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 587,
          "end": 661,
          "quote": "- Cut rework by 18% by standardizing panel labeling across the terminal fleet."
        }
      ]
    },
    {
      "id": "13:13:0:68",
      "line_id": "13",
      "start": 0,
      "end": 68,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 662,
          "end": 730,
          "quote": "- Prepared the monthly cost summary reviewed by the regional manager."
        }
      ]
    },
    {
      "id": "15:15:0:50",
      "line_id": "15",
      "start": 0,
      "end": 50,
      "verdict": "unsupported",
      "citations": []
    },
    {
      "id": "16:16:0:52",
      "line_id": "16",
      "start": 0,
      "end": 52,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 103,
          "end": 155,
          "quote": "- Certified Inventory Control Specialist (CICS)"
        }
      ]
    }
  ],
  "representations": [
    {
      "source_claim_id": "candidate-main.txt:229:307",
      "status": "paraphrased",
      "spans": [
        {
          "line_id": "6",
          "start": 0,
          "end": 78
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:403:484",
      "status": "paraphrased",
      "spans": [
        {
          "line_id": "8",
          "start": 0,
          "end": 81
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:662:730",
      "status": "paraphrased",
      "spans": [
        {
          "line_id": "13",
          "start": 0,
          "end": 68
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:24:76",
      "status": "missing",
      "spans": []
    },
    {
      "source_claim_id": "candidate-supplement.txt:77:102",
      "status": "missing",
      "spans": []
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
| 1 | **Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present* - Maintain parts availability records for 7 service branches. - Compared q | `misattributed` | The employment dates in the draft (08/2023 – Present) conflict with the previously listed roles, making concurrent employment impossible without explicit overlap indication. Additionally, the role at  | candidate-supplement.txt | (omit) Note: If the candidate can confirm concurrent employment or clarify the timeline, this section may be reassessed. As presented, it introduces an unverifi | passage ✗ / sources ✓ |

### Proposed corrections

| passage | proposed replacement | locatable in draft | omit marker |
|---|---|---|---|
| **Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst | 08/2023 - Present* - Maintain parts availability recor | (omit) Note: If the candidate can confirm concurrent employment or clarify the timeline, this section may be reassessed. | NO | YES |

### Re-review

_No re-review was performed in the retained run._

### Binding-gate outcome (offline replay)

- initial decision: `revise`
- repair status: `not_attempted`
- re-review decision: `none`
- gate result: **WITHHOLD** (`initial_review_unverifiable`)
- initial structural problems: `passage_not_in_draft:**Ridgeline Equipment Services** -- Fife, WA *Inventory Anal, revise_without_actionable_issue`

> The prose answer could not be used as a verdict: a structural problem remained after parsing (see the problems listed).

### Raw first response

```text
Decision: **revise**

Issues:
- passage: "**Ridgeline Equipment Services** -- Fife, WA  
*Inventory Analyst | 08/2023 - Present*  
- Maintain parts availability records for 7 service branches.  
- Compared quarterly parts usage against the prior four quarters to set reorder points."
  problem: **misattributed**
  sources: [candidate-supplement.txt: "Ridgeline Equipment Services", "Inventory Analyst | 08/2023 - Present", "Maintain parts availability records for 7 service branches.", "Compared quarterly parts usage against the prior four quarters to set reorder points."]
  why: The employment dates in the draft (08/2023 – Present) conflict with the previously listed roles, making concurrent employment impossible without explicit overlap indication. Additionally, the role at Ridgeline is dated after the current role at Harborline Transit Authority, which ends in “Present”, creating a timeline inconsistency.
  correction: (omit)

Note: If the candidate can confirm concurrent employment or clarify the timeline, this section may be reassessed. As presented, it introduces an unverifiable structure that conflicts with other clearly dated roles.
```

---

## Cell `revised+enabled`

**Raw model decision:** `revise`

### Reported issues

| # | passage | problem | why | sources | correction | resolves |
|---|---|---|---|---|---|---|
| 1 | **Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present* - Maintain parts availability records for 7 service branches. - Compared q | `misattributed` | The employment dates in the draft (08/2023 – Present) conflict with the Harborline Transit Authority role listed as "04/2019 – Present". This misattributes concurrent roles to the same time period wit | candidate-supplement.txt | (omit) Note: The content from *candidate-supplement.txt* regarding Ridgeline Equipment Services is not reflected in the draft. Since the draft omits this employ | passage ✗ / sources ✓ |
| 2 | (omitted) | `omitted` | The draft does not include the candidate’s current position at Ridgeline Equipment Services, despite being listed as “Present” in the source document. | candidate-supplement.txt | **Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present* - Maintain parts availability records for 7 service branches. - Compared q | passage ✓ / sources ✓ |

### Proposed corrections

| passage | proposed replacement | locatable in draft | omit marker |
|---|---|---|---|
| **Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst | 08/2023 - Present* - Maintain parts availability recor | (omit) Note: The content from *candidate-supplement.txt* regarding Ridgeline Equipment Services is not reflected in the  | NO | YES |
| (omitted) | **Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst | 08/2023 - Present* - Maintain parts availability recor | NO | no |

### Re-review

_No re-review was performed in the retained run._

### Binding-gate outcome (offline replay)

- initial decision: `revise`
- repair status: `not_attempted`
- re-review decision: `none`
- gate result: **WITHHOLD** (`initial_review_unverifiable`)
- initial structural problems: `passage_not_in_draft:**Ridgeline Equipment Services** -- Fife, WA *Inventory Anal`

> The prose answer could not be used as a verdict: a structural problem remained after parsing (see the problems listed).

### Raw first response

```text
Decision: **revise**

Issues:
- passage: "**Ridgeline Equipment Services** -- Fife, WA  
*Inventory Analyst | 08/2023 - Present*  
- Maintain parts availability records for 7 service branches.  
- Compared quarterly parts usage against the prior four quarters to set reorder points."
- problem: **misattributed**
- sources: [candidate-supplement.txt]  
  > "**Ridgeline Equipment Services** -- Fife, WA  
  *Inventory Analyst | 08/2023 - Present*  
  - Maintain parts availability records for 7 service branches.  
  - Compared quarterly parts usage against the prior four quarters to set reorder points."
- why: The employment dates in the draft (08/2023 – Present) conflict with the Harborline Transit Authority role listed as "04/2019 – Present". This misattributes concurrent roles to the same time period without clarification.
- correction: (omit)

Note: The content from *candidate-supplement.txt* regarding Ridgeline Equipment Services is not reflected in the draft. Since the draft omits this employer entirely, and there's no clear reason to exclude it based on date conflicts or other issues, its absence constitutes an **omission**.

However, because including it would introduce a timeline contradiction that isn't resolved in the source materials, we must treat the attempt to reconcile these roles as unresolved.

Revised Issue Entry:
- passage: (omitted)
- problem: **omitted**
- sources: [candidate-supplement.txt]
- why: The draft does not include the candidate’s current position at Ridgeline Equipment Services, despite being listed as “Present” in the source document.
- correction: "**Ridgeline Equipment Services** -- Fife, WA  
*Inventory Analyst | 08/2023 - Present*  
- Maintain parts availability records for 7 service branches.  
- Compared quarterly parts usage against the prior four quarters to set reorder points."
```

