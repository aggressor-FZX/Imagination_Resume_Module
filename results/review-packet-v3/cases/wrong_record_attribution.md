# Case: `wrong_record_attribution`

**Declared mutation:** `move_fact_to_other_employer` — Attaches the 1,180-invoice figure to Cascade Freight, where it does not belong.

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
- Cut rework by 18% by standardizing panel labeling across the terminal fleet.
- Prepared the monthly cost summary that the regional manager reviewed and reconciled 1,180 fuel invoices each quarter.

## Certifications
- Certified Inventory Control Specialist (CICS)
```

---

## Cell `current+advisory`

**Raw model decision:** **none — no response retained**

### Reported issues

_No response retained, so no issues can be reported._

### Proposed corrections

_No prose corrections to apply._

### Re-review

_No re-review was performed in the retained run._

### Binding-gate outcome (offline replay)

- initial decision: `none`
- repair status: `not_attempted`
- re-review decision: `none`
- gate result: **WITHHOLD** (`initial_review_unverifiable`)
- initial structural problems: `decision_missing`

> The prose answer could not be used as a verdict: a structural problem remained after parsing (see the problems listed).

### Raw first response

```text
(none retained)
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
      "id": "1:1:1",
      "line_id": 1,
      "start": 0,
      "end": 69,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 0,
          "end": 69,
          "quote": "Operations analyst with 9 years of experience in depot reporting and fuel cost control across public transit operations."
        }
      ]
    },
    {
      "id": "2:2:1",
      "line_id": 2,
      "start": 0,
      "end": 69,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 0,
          "end": 69,
          "quote": "Operations analyst with 9 years of experience in depot reporting and fuel cost control across public transit operations."
        }
      ]
    },
    {
      "id": "3:1:1",
      "line_id": 3,
      "start": 0,
      "end": 36,
      "verdict": "structural",
      "citations": []
    },
    {
      "id": "4:1:1",
      "line_id": 4,
      "start": 0,
      "end": 50,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 71,
          "end": 121,
          "quote": "**Harborline Transit Authority** -- Tacoma, WA"
        }
      ]
    },
    {
      "id": "5:1:1",
      "line_id": 5,
      "start": 0,
      "end": 46,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 122,
          "end": 168,
          "quote": "*Fuel Reporting Analyst | 04/2019 - Present*"
        }
      ]
    },
    {
      "id": "6:1:1",
      "line_id": 6,
      "start": 0,
      "end": 80,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 169,
          "end": 249,
          "quote": "- Tracked monthly fuel consumption across 14 depots and compared each month against the same month in the prior year."
        }
      ]
    },
    {
      "id": "7:1:1",
      "line_id": 7,
      "start": 0,
      "end": 79,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 250,
          "end": 329,
          "quote": "- Reconciled 1,180 fuel invoices against depot dispensing records each quarter."
        }
      ]
    },
    {
      "id": "8:1:1",
      "line_id": 8,
      "start": 0,
      "end": 76,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 330,
          "end": 406,
          "quote": "- Built the depot scorecard that supervisors review in the Monday operations meeting."
        }
      ]
    },
    {
      "id": "9:1:1",
      "line_id": 9,
      "start": 0,
      "end": 48,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 71,
          "end": 119,
          "quote": "**Ridgeline Equipment Services** -- Fife, WA"
        }
      ]
    },
    {
      "id": "10:1:1",
      "line_id": 10,
      "start": 0,
      "end": 42,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 120,
          "end": 162,
          "quote": "*Inventory Analyst | 08/2023 - Present*"
        }
      ]
    },
    {
      "id": "11:1:1",
      "line_id": 11,
      "start": 0,
      "end": 58,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 163,
          "end": 221,
          "quote": "- Maintain parts availability records for 7 service branches."
        }
      ]
    },
    {
      "id": "12:1:1",
      "line_id": 12,
      "start": 0,
      "end": 82,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 222,
          "end": 304,
          "quote": "- Compared quarterly parts usage against the prior four quarters to set reorder points."
        }
      ]
    },
    {
      "id": "13:1:1",
      "line_id": 13,
      "start": 0,
      "end": 52,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 353,
          "end": 405,
          "quote": "**Cascade Freight Partners** -- Kent, WA"
        }
      ]
    },
    {
      "id": "14:1:1",
      "line_id": 14,
      "start": 0,
      "end": 56,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 406,
          "end": 462,
          "quote": "*Cost Reporting Coordinator | 07/2015 - 03/2019*"
        }
      ]
    },
    {
      "id": "15:1:1",
      "line_id": 15,
      "start": 0,
      "end": 74,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 463,
          "end": 537,
          "quote": "- Cut rework by 18% by standardizing panel labeling across the terminal fleet."
        }
      ]
    },
    {
      "id": "16:1:1",
      "line_id": 16,
      "start": 0,
      "end": 94,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 538,
          "end": 632,
          "quote": "- Prepared the monthly cost summary reviewed by the regional manager."
        }
      ]
    },
    {
      "id": "17:1:1",
      "line_id": 17,
      "start": 0,
      "end": 50,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 250,
          "end": 300,
          "quote": "- Reconciled 1,180 fuel invoices against depot dispensing records each quarter."
        }
      ]
    },
    {
      "id": "18:1:1",
      "line_id": 18,
      "start": 0,
      "end": 20,
      "verdict": "structural",
      "citations": []
    },
    {
      "id": "19:1:1",
      "line_id": 19,
      "start": 0,
      "end": 52,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 306,
          "end": 358,
          "quote": "- Certified Inventory Control Specialist (CICS)"
        }
      ]
    }
  ],
  "representations": [
    {
      "source_claim_id": "candidate-main.txt:1:1",
      "status": "represented",
      "spans": [
        {
          "line_id": 1,
          "start": 0,
          "end": 69
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:2:1",
      "status": "represented",
      "spans": [
        {
          "line_id": 2,
          "start": 0,
          "end": 69
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:4:1",
      "status": "represented",
      "spans": [
        {
          "line_id": 4,
          "start": 0,
          "end": 50
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:5:1",
      "status": "represented",
      "spans": [
        {
          "line_id": 5,
          "start": 0,
          "end": 46
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:6:1",
      "status": "represented",
      "spans": [
        {
          "line_id": 6,
          "start": 0,
          "end": 80
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:7:1",
      "status": "represented",
      "spans": [
        {
          "line_id": 7,
          "start": 0,
          "end": 79
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:8:1",
      "status": "represented",
      "spans": [
        {
          "line_id": 8,
          "start": 0,
          "end": 76
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:1:1",
      "status": "represented",
      "spans": [
        {
          "line_id": 9,
          "start": 0,
          "end": 48
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:2:1",
      "status": "represented",
      "spans": [
        {
          "line_id": 10,
          "start": 0,
          "end": 42
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:3:1",
      "status": "represented",
      "spans": [
        {
          "line_id": 11,
          "start": 0,
          "end": 58
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:4:1",
      "status": "represented",
      "spans": [
        {
          "line_id": 12,
          "start": 0,
          "end": 82
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:9:1",
      "status": "represented",
      "spans": [
        {
          "line_id": 13,
          "start": 0,
          "end": 52
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:10:1",
      "status": "represented",
      "spans": [
        {
          "line_id": 14,
          "start": 0,
          "end": 56
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:11:1",
      "status": "represented",
      "spans": [
        {
          "line_id": 15,
          "start": 0,
          "end": 74
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:12:1",
      "status": "represented",
      "spans": [
        {
          "line_id": 16,
          "start": 0,
          "end": 94
        }
      ]
    },
    {
      "source_claim_id": "candidate-main.txt:7:1",
      "status": "represented",
      "spans": [
        {
          "line_id": 17,
          "start": 0,
          "end": 50
        }
      ]
    },
    {
      "source_claim_id": "candidate-supplement.txt:6:1",
      "status": "represented",
      "spans": [
        {
          "line_id": 19,
          "start": 0,
          "end": 52
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
| 1 | *Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present | `misattributed` | The employment dates for Ridgeline Equipment Services (08/2023 – Present) overlap with Harborline Transit Authority (04/2019 – Present), but the source lists Ridgeline as a current role without indica | candidate-supplement.txt | — | passage ✓ / sources ✓ |
| 2 | Prepared the monthly cost summary that the regional manager reviewed and reconciled 1,180 fuel invoices each quarter. | `misattributed` | The draft merges two separate responsibilities from different employers into one bullet under Cascade Freight Partners. Preparing a monthly cost summary belongs to Cascade Freight Partners, while reco | candidate-main.txt | Prepared the monthly cost summary reviewed by the regional manager. Approved version would reflect these corrections accordingly. | passage ✓ / sources ✓ |

### Proposed corrections

| passage | proposed replacement | locatable in draft | omit marker |
|---|---|---|---|
| Prepared the monthly cost summary that the regional manager reviewed and reconciled 1,180 fuel invoices each quarter. | Prepared the monthly cost summary reviewed by the regional manager. Approved version would reflect these corrections acc | yes | no |

### Re-review

**Raw re-review decision:** `revise`

```text
Decision: revise

Issues:
- passage: **Ridgeline Equipment Services** -- Fife, WA  
  *Inventory Analyst | 08/2023 - Present*  
  problem: misattributed  
  sources: [candidate-supplement.txt: "Ridgeline Equipment Services" listed under Inventory Analyst role]  
  why: The draft lists Ridgeline Equipment Services as starting in 08/2023, but candidate-supplement.txt does not provide a start date. It also conflicts with Harborline Transit Authority role being "Present". Since "Present" implies current employment, and Ridgeline is also marked as "Present", this creates an inconsistency unless clarified.  
  correction:  
  **Ridgeline Equipment Services** -- Fife, WA  
  *Inventory Analyst | Dates unspecified (see context)*  

(Note: If the intention was to show concurrent roles, that should be clarified with verifiable information.)

- passage: - Maintained parts availability records for 7 service branches.  
  problem: unsupported  
  sources: [candidate-supplement.txt: “Maintain parts availability records for 7 service branches.”]  
  why: "Maintain" (present tense) contradicts the past-tense narrative used elsewhere and is inconsistent with "Present" timeframe without clarification.  
  correction: - Maintained parts availability records for 7 service branches.

- passage: - Compared quarterly parts usage with the prior four quarters to set reorder points.  
  problem: unsupported  
  sources: [candidate-supplement.txt: “Compared quarterly parts usage against the prior four quarters to set reorder points.”]  
  why: Minor wording shift from "against" to "with" changes preposition usage but doesn't affect meaning; however, preserving original phrasing improves faithfulness.  
  correction: - Compared quarterly parts usage against the prior four quarters to set reorder points.
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
  sources: [candidate-supplement.txt: "Ridgeline Equipment Services" — "Inventory Analyst | 08/2023 - Present"]  
  why: The employment dates for Ridgeline Equipment Services (08/2023 – Present) overlap with Harborline Transit Authority (04/2019 – Present), but the source lists Ridgeline as a current role without indicating part-time or contract status. This creates a factual conflict unless clarified.  
  correction: (omit)  

- passage: Prepared the monthly cost summary that the regional manager reviewed and reconciled 1,180 fuel invoices each quarter.  
  problem: misattributed  
  sources: [candidate-main.txt: Cascade Freight Partners — "Prepared the monthly cost summary reviewed by the regional manager."; candidate-main.txt: Harborline Transit Authority — "Reconciled 1,180 fuel invoices against depot dispensing records each quarter."]  
  why: The draft merges two separate responsibilities from different employers into one bullet under Cascade Freight Partners. Preparing a monthly cost summary belongs to Cascade Freight Partners, while reconciling fuel invoices belongs to Harborline Transit Authority. Combining them misattributes work.  
  correction: Prepared the monthly cost summary reviewed by the regional manager.  

Approved version would reflect these corrections accordingly.
```

---

## Cell `revised+enabled`

**Raw model decision:** `revise`

### Reported issues

| # | passage | problem | why | sources | correction | resolves |
|---|---|---|---|---|---|---|
| 1 | **"08/2023 - Present"** under Ridgeline Equipment Services | `unsupported` | The candidate-supplement.txt states employment at Ridgeline Equipment Services began in August 2023, but candidate-main.txt lists Harborline Transit Authority as "04/2019 – Present", indicating curren | candidate-main.txt, candidate-supplement.txt | — | passage ✗ / sources ✓ |
| 2 | Compared quarterly parts usage with the prior four quarters to set reorder points. | `misattributed` | While the action described matches the source, it's attributed to Ridgeline Equipment Services in the draft, but candidate-supplement.txt also contains other responsibilities listed without explicit d | candidate-supplement.txt | — | passage ✓ / sources ✓ |
| 3 | Prepared the monthly cost summary that the regional manager reviewed and reconciled 1,180 fuel invoices each quarter. | `misattributed` | These two actions were separate bullet points under Cascade Freight Partners in the source and have been merged into one sentence in a way that implies the regional manager reviewed both summaries and | candidate-main.txt | "- Prepared the monthly cost summary reviewed by the regional manager.\n- Reconciled 1,180 fuel invoices against depot dispensing records each quarter." Final n | passage ✓ / sources ✓ |

### Proposed corrections

| passage | proposed replacement | locatable in draft | omit marker |
|---|---|---|---|
| Prepared the monthly cost summary that the regional manager reviewed and reconciled 1,180 fuel invoices each quarter. | "- Prepared the monthly cost summary reviewed by the regional manager.\n- Reconciled 1,180 fuel invoices against depot d | yes | no |

### Re-review

_No re-review was performed in the retained run._

### Binding-gate outcome (offline replay)

- initial decision: `revise`
- repair status: `not_attempted`
- re-review decision: `none`
- gate result: **WITHHOLD** (`initial_review_unverifiable`)
- initial structural problems: `passage_not_in_draft:**"08/2023 - Present"** under Ridgeline Equipment Services`

> The prose answer could not be used as a verdict: a structural problem remained after parsing (see the problems listed).

### Raw first response

```text
Decision: revise

Issues:
- passage: **"08/2023 - Present"** under Ridgeline Equipment Services
  problem: unsupported
  sources: [candidate-main.txt, candidate-supplement.txt]
  why: The candidate-supplement.txt states employment at Ridgeline Equipment Services began in August 2023, but candidate-main.txt lists Harborline Transit Authority as "04/2019 – Present", indicating current employment there. This creates a timeline conflict suggesting simultaneous roles, which is not clarified or supported.
  correction: (omit)

- passage: **"Compared quarterly parts usage with the prior four quarters to set reorder points."**
  problem: misattributed
  sources: [candidate-supplement.txt: "Compared quarterly parts usage against the prior four quarters to set reorder points."]
  why: While the action described matches the source, it's attributed to Ridgeline Equipment Services in the draft, but candidate-supplement.txt also contains other responsibilities listed without explicit date linkage beyond "quarterly". However, since both documents confirm this activity occurred during the Ridgeline role, this attribution appears acceptable unless contradicted by another clearer source. No contradiction found upon close reading; retain original judgment on this point.

- passage: **"Prepared the monthly cost summary that the regional manager reviewed and reconciled 1,180 fuel invoices each quarter."**
  problem: misattributed
  sources: [candidate-main.txt: "- Prepared the monthly cost summary reviewed by the regional manager.\n- Reconciled 1,180 fuel invoices against depot dispensing records each quarter."]
  why: These two actions were separate bullet points under Cascade Freight Partners in the source and have been merged into one sentence in a way that implies the regional manager reviewed both summaries and reconciliations, which isn't stated.
  correction: "- Prepared the monthly cost summary reviewed by the regional manager.\n- Reconciled 1,180 fuel invoices against depot dispensing records each quarter."

Final note: The professional summary remains accurate based on information provided in the source documents.
```

