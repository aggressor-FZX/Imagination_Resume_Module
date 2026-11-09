# Case: `faithful_paraphrase`

**Declared mutation:** `reword_only` — The baseline itself, paraphrased throughout. No fact changed.

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
          "end": 80,
          "quote": "Operations analyst with 9 years of experience in depot reporting and fuel cost control across public transit operations."
        }
      ]
    },
    {
      "id": "5:1:0:69",
      "line_id": "5",
      "start": 0,
      "end": 69,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 138,
          "end": 207,
          "quote": "Fuel Reporting Analyst | 04/2019 - Present"
        }
      ]
    },
    {
      "id": "6:1:0:77",
      "line_id": "6",
      "start": 0,
      "end": 77,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 211,
          "end": 287,
          "quote": "- Tracked monthly fuel consumption across 14 depots and compared each month"
        }
      ]
    },
    {
      "id": "7:1:0:68",
      "line_id": "7",
      "start": 0,
      "end": 68,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 317,
          "end": 385,
          "quote": "- Reconciled 1,180 fuel invoices against depot dispensing records each quarter."
        }
      ]
    },
    {
      "id": "8:1:0:80",
      "line_id": "8",
      "start": 0,
      "end": 80,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 386,
          "end": 466,
          "quote": "- Built the depot scorecard that supervisors review in the Monday operations meeting."
        }
      ]
    },
    {
      "id": "10:1:0:65",
      "line_id": "10",
      "start": 0,
      "end": 65,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 140,
          "end": 205,
          "quote": "*Inventory Analyst | 08/2023 - Present*"
        }
      ]
    },
    {
      "id": "11:1:0:61",
      "line_id": "11",
      "start": 0,
      "end": 61,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 209,
          "end": 270,
          "quote": "- Maintain parts availability records for 7 service branches."
        }
      ]
    },
    {
      "id": "12:1:0:85",
      "line_id": "12",
      "start": 0,
      "end": 85,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 271,
          "end": 356,
          "quote": "- Compared quarterly parts usage against the prior four quarters to set reorder points."
        }
      ]
    },
    {
      "id": "14:1:0:71",
      "line_id": "14",
      "start": 0,
      "end": 71,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 470,
          "end": 541,
          "quote": "*Cost Reporting Coordinator | 07/2015 - 03/2019*"
        }
      ]
    },
    {
      "id": "15:1:0:76",
      "line_id": "15",
      "start": 0,
      "end": 76,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 545,
          "end": 621,
          "quote": "- Cut rework by 18% by standardizing panel labeling across the terminal fleet."
        }
      ]
    },
    {
      "id": "16:1:0:65",
      "line_id": "16",
      "start": 0,
      "end": 65,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 622,
          "end": 687,
          "quote": "- Prepared the monthly cost summary reviewed by the regional manager."
        }
      ]
    },
    {
      "id": "19:1:0:53",
      "line_id": "19",
      "start": 0,
      "end": 53,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 360,
          "end": 413,
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
      "id": "1:1:0:89",
      "line_id": "1",
      "start": 0,
      "end": 89,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 0,
          "end": 90,
          "quote": "Operations analyst with 9 years of experience in depot reporting and fuel cost control across public transit operations."
        }
      ]
    },
    {
      "id": "4:1:0:46",
      "line_id": "4",
      "start": 0,
      "end": 46,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 36,
          "end": 82,
          "quote": "Harborline Transit Authority -- Tacoma, WA"
        }
      ]
    },
    {
      "id": "5:1:0:50",
      "line_id": "5",
      "start": 0,
      "end": 50,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 83,
          "end": 133,
          "quote": "Fuel Reporting Analyst | 04/2019 - Present"
        }
      ]
    },
    {
      "id": "6:1:0:86",
      "line_id": "6",
      "start": 0,
      "end": 86,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 136,
          "end": 222,
          "quote": "- Tracked monthly fuel consumption across 14 depots and compared each month against the same month in the prior year."
        }
      ]
    },
    {
      "id": "7:1:0:83",
      "line_id": "7",
      "start": 0,
      "end": 83,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 225,
          "end": 308,
          "quote": "- Reconciled 1,180 fuel invoices against depot dispensing records each quarter."
        }
      ]
    },
    {
      "id": "8:1:0:85",
      "line_id": "8",
      "start": 0,
      "end": 85,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 311,
          "end": 396,
          "quote": "- Built the depot scorecard that supervisors review in the Monday operations meeting."
        }
      ]
    },
    {
      "id": "10:1:0:48",
      "line_id": "10",
      "start": 0,
      "end": 48,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 36,
          "end": 84,
          "quote": "Ridgeline Equipment Services -- Fife, WA"
        }
      ]
    },
    {
      "id": "11:1:0:46",
      "line_id": "11",
      "start": 0,
      "end": 46,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 85,
          "end": 131,
          "quote": "Inventory Analyst | 08/2023 - Present"
        }
      ]
    },
    {
      "id": "12:1:0:60",
      "line_id": "12",
      "start": 0,
      "end": 60,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 134,
          "end": 194,
          "quote": "- Maintain parts availability records for 7 service branches."
        }
      ]
    },
    {
      "id": "13:1:0:90",
      "line_id": "13",
      "start": 0,
      "end": 90,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 197,
          "end": 287,
          "quote": "- Compared quarterly parts usage against the prior four quarters to set reorder points."
        }
      ]
    },
    {
      "id": "15:1:0:44",
      "line_id": "15",
      "start": 0,
      "end": 44,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 325,
          "end": 369,
          "quote": "Cascade Freight Partners -- Kent, WA"
        }
      ]
    },
    {
      "id": "16:1:0:54",
      "line_id": "16",
      "start": 0,
      "end": 54,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 370,
          "end": 424,
          "quote": "Cost Reporting Coordinator | 07/2015 - 03/2019"
        }
      ]
    },
    {
      "id": "17:1:0:77",
      "line_id": "17",
      "start": 0,
      "end": 77,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-main.txt",
          "start": 427,
          "end": 504,
          "quote": "- Cut rework by 18% by standardizing panel labeling across the terminal fleet."
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
          "start": 507,
          "end": 576,
          "quote": "- Prepared the monthly cost summary reviewed by the regional manager."
        }
      ]
    },
    {
      "id": "20:1:0:50",
      "line_id": "20",
      "start": 0,
      "end": 50,
      "verdict": "supported",
      "citations": [
        {
          "source_id": "candidate-supplement.txt",
          "start": 290,
          "end": 340,
          "quote": "## Certifications\n- Certified Inventory Control Specialist (CICS)"
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
| 1 | *Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present | `misattributed` | The draft assigns Ridgeline Equipment Services to Fife, WA, but candidate-main.txt lists "Ridgeline Equipment Services -- Fife, WA" under a different role and timeframe than what is presented in the d | candidate-main.txt, candidate-supplement.txt | — | passage ✓ / sources ✓ |
| 2 | - Maintained parts availability records for 7 service branches. | `unsupported` | While this statement paraphrases the source accurately, it shifts from active verb (“Maintain”) to passive (“Maintained”), which changes the tense without clear justification based on separation from  | candidate-supplement.txt | Maintain parts availability records for 7 service branches. | passage ✓ / sources ✓ |
| 3 | - Compared quarterly parts usage with the prior four quarters to set reorder points. | `omitted` | The phrasing of this bullet matches the source closely, but because the role itself appears misplaced due to conflicting timelines between documents, the entire section should be reassigned or removed | candidate-supplement.txt | — | passage ✓ / sources ✓ |

### Proposed corrections

| passage | proposed replacement | locatable in draft | omit marker |
|---|---|---|---|
| - Maintained parts availability records for 7 service branches. | Maintain parts availability records for 7 service branches. | yes | no |

### Re-review

**Raw re-review decision:** `revise`

```text
Decision: revise

Issues:
- passage: **Ridgeline Equipment Services** -- Fife, WA  
  *Inventory Analyst | 08/2023 - Present*  
  Maintain parts availability records for 7 service branches.  
  - Compared quarterly parts usage with the prior four quarters to set reorder points.  
  problem: misattributed  
  sources: [candidate-supplement.txt: "Ridgeline Equipment Services" entry correctly lists duties but implies current role; candidate-main.txt shows no overlap in employment dates]  
  why: The draft presents Ridgeline as a concurrent ("Present") role starting Aug 2023, while Harborline is also listed as "Present", creating an inconsistency not present in the source. The candidate cannot hold two jobs concurrently marked as “Present” without clarification.  
  correction:  
  **Ridgeline Equipment Services** -- Fife, WA  
  *Inventory Analyst | 08/2023 - 04/2024*  
  - Maintained parts availability records for 7 service branches.  
  - Compared quarterly parts usage against the prior four quarters to establish reorder points.

(Note: Adjusted end date assumes role ended before returning to Harborline; if actual dates differ, candidate must clarify.)
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
  sources: [candidate-main.txt, candidate-supplement.txt]  
  why: The draft assigns Ridgeline Equipment Services to Fife, WA, but candidate-main.txt lists "Ridgeline Equipment Services -- Fife, WA" under a different role and timeframe than what is presented in the draft. Specifically, the draft implies current employment (08/2023 – Present), while candidate-main.txt indicates past employment with Harborline Transit Authority during overlapping periods. This creates confusion about timeline accuracy and role attribution.  
  correction: (omit)

- passage: - Maintained parts availability records for 7 service branches.  
  problem: unsupported  
  sources: [candidate-supplement.txt: “Maintain parts availability records for 7 service branches.”]  
  why: While this statement paraphrases the source accurately, it shifts from active verb (“Maintain”) to passive (“Maintained”), which changes the tense without clear justification based on separation from the role. Since the role is listed as “Present,” maintaining present-tense action verbs better reflects ongoing duties.  
  correction: Maintain parts availability records for 7 service branches.

- passage: - Compared quarterly parts usage with the prior four quarters to set reorder points.  
  problem: omitted  
  sources: [candidate-supplement.txt: “Compared quarterly parts usage against the prior four quarters to set reorder points.”]  
  why: The phrasing of this bullet matches the source closely, but because the role itself appears misplaced due to conflicting timelines between documents, the entire section should be reassigned or removed unless verified otherwise.  
  correction: (omit)
```

---

## Cell `revised+enabled`

**Raw model decision:** `revise`

### Reported issues

| # | passage | problem | why | sources | correction | resolves |
|---|---|---|---|---|---|---|
| 1 | *Inventory Analyst \| 08/2023 - Present* | `misattributed` | The draft places Ridgeline Equipment Services in Fife, WA, but according to the source document, this role overlaps temporally with the current role at Harborline Transit Authority (which is listed as | candidate-supplement.txt | *(omit)* | passage ✓ / sources ✓ |
| 2 | **Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst \| 08/2023 - Present* | `unsupported` | The candidate cannot simultaneously hold two full-time positions marked as “Present.” The supplemental file lists Ridgeline as “08/2023 – Present,” while the main file has Harborline Transit also list | candidate-main.txt, candidate-supplement.txt | *(omit)* or request clarification on actual work period or concurrent roles Note: If the candidate confirms that Ridgeline was a contract/temporary role overlap | passage ✓ / sources ✓ |

### Proposed corrections

| passage | proposed replacement | locatable in draft | omit marker |
|---|---|---|---|
| *Inventory Analyst | 08/2023 - Present* | *(omit)* | yes | no |
| **Ridgeline Equipment Services** -- Fife, WA *Inventory Analyst | 08/2023 - Present* | *(omit)* or request clarification on actual work period or concurrent roles Note: If the candidate confirms that Ridgeli | yes | no |

### Re-review

**Raw re-review decision:** `revise`

```text
Decision: revise

Issues:
- passage: **Ridgeline Equipment Services** -- Fife, WA  
  *(omit)*  
  - Maintained parts availability records for 7 service branches.  
  - Compared quarterly parts usage with the prior four quarters to set reorder points.
  problem: misattributed
  sources: [candidate-supplement.txt]: "## Professional Experience  
  **Ridgeline Equipment Services** -- Fife, WA  
  *Inventory Analyst | 08/2023 - Present*  
  - Maintain parts availability records for 7 service branches.  
  - Compared quarterly parts usage against the prior four quarters to set reorder points."
  why: The draft omits the role title ("Inventory Analyst") and changes the tense from present ("Maintain", "Compared") to past ("Maintained", "Compared"), which alters the meaning regarding current employment status. Additionally, the role is incorrectly placed without a position title.
  correction:  
  **Ridgeline Equipment Services** -- Fife, WA  
  *Inventory Analyst | 08/2023 - Present*  
  - Maintain parts availability records for 7 service branches.  
  - Compare quarterly parts usage against the prior four quarters to set reorder points.
```

### Binding-gate outcome (offline replay)

- initial decision: `revise`
- repair status: `ambiguous_or_multiple_issues`
- re-review decision: `none`
- gate result: **WITHHOLD** (`revision_unresolved`)

> The reviewer asked for a change, but no repair could be applied safely (multiple issues, an omission, an ambiguous passage, or a correction that is not an exact owned source bullet). Nothing was edited.

### Raw first response

```text
Decision: **revise**

Issues:
- passage: "*Inventory Analyst | 08/2023 - Present*"
  problem: **misattributed**
  sources: [candidate-supplement.txt: "Ridgeline Equipment Services -- Fife, WA  
Inventory Analyst | 08/2023 - Present"]
  why: The draft places Ridgeline Equipment Services in Fife, WA, but according to the source document, this role overlaps temporally with the current role at Harborline Transit Authority (which is listed as "Present"), creating a conflict in employment timeline.
  correction: *(omit)*

- passage: "**Ridgeline Equipment Services** -- Fife, WA  
*Inventory Analyst | 08/2023 - Present*"
  problem: **unsupported**
  sources: [candidate-main.txt and candidate-supplement.txt do not support concurrent employment]
  why: The candidate cannot simultaneously hold two full-time positions marked as “Present.” The supplemental file lists Ridgeline as “08/2023 – Present,” while the main file has Harborline Transit also listed as “Present” starting from 04/2019. This creates an inconsistency unless clarified.
  correction: *(omit)* or request clarification on actual work period or concurrent roles

Note: If the candidate confirms that Ridgeline was a contract/temporary role overlapping with their ongoing position at Harborline, then it could be retained with appropriate context. Without such confirmation, it's not faithful to include both as “Present.”
```

