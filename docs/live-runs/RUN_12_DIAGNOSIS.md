# Run 12 — instrumented diagnosis

**Correlation:** `83e2eb3f-0fec-4e3d-adc9-c20b43981be0`
**Revision:** `7cc415ba` (unchanged; env-only redeploys)
**Outcome:** withheld — **`source_facts_missing`** (a *different* reason from run 11's `unresolved_claims`)
**Credits:** 902 → 869 (33). **Correction calls:** 3 (inventory, review, restore). `repair_calls: 0`.
**Raw capture:** `83e2eb3f-….json`, 35,984 bytes, sha256 `b2d4b11e25222764b957af70cee6651b24e9b089cf24cd0764b3c439dbc9e67f`

Capture retained **everything** this time: the draft (1,877 chars), both redacted sources (1,380 / 820), and all three exchange bodies.

---

# 1. What withheld the run

The reviewer declared **nine source segments missing**:

```
missing: [0, 1, 2, 19, 21, 22, 23, 24, 25]
```

Resolved to text:

| idx | source | text |
|---|---|---|
| 0 | doc-0 | `Professional Summary` |
| 1 | doc-0 | `SYNTHETIC-ONLY Journeyman electrician with 12 years of electrical experience spanning` |
| 2 | doc-0 | `SYNTHETIC-ONLY commercial construction, industrial maintenance, and facility operations.` |
| 19 | doc-0 | `Associate of Applied Science, Electrical Systems Technology \| 2015` |
| 21 | doc-0 | `SYNTHETIC-ONLY Conduit bending, switchgear troubleshooting, thermographic inspection` |
| 22 | doc-0 | `SYNTHETIC-ONLY Lockout/tagout, NFPA 70E` |
| 23 | doc-1 | `Professional Summary` |
| 24 | doc-1 | `SYNTHETIC-ONLY Maintenance electrician focused on electrical distribution and` |
| 25 | doc-1 | `SYNTHETIC-ONLY building automation systems in occupied commercial properties.` |

The engine then attempted **one** bounded restoration, which is the correct next step. The
restoration failed and the run stopped with `source_facts_missing`.

# 2. The draft — what was actually generated

```markdown
## Professional Experience
**Northline Fabrication, Inc.** -- Spokane, WA
*Welder / Maintenance Helper | 03/2012 – 09/2015*
- Performed preventive maintenance on shop welding and cutting equipment, ensuring consistent uptime.
- Fabricated and repaired structural steel assemblies for 14 industrial contracts using SMAW and GMAW processes.

**Kettle Falls Electrical Contractors** -- Spokane, WA
*Electrical Apprentice | 10/2015 – 06/2019*
- Pulled and terminated branch circuits on 31 commercial buildouts, supporting NEC-compliant installations.
- Completed 8,000 apprenticeship hours under a licensed journeyman, mastering conduit bending and circuit termination.

**Pend Oreille Power Systems** -- Spokane Valley, WA
*Journeyman Electrician | 07/2019 – 02/2023*
- Diagnosed and repaired switchgear faults across 9 substations, maintaining grid reliability.
- Trained 4 apprentices on lockout/tagout procedures, improving safety compliance adherence.
- Cut average fault-response time from 4.5 hours to 2.75 hours through predictive diagnostics.

**Riverbend Facilities Group** -- Coeur d'Alene, ID
*Maintenance Electrician | 03/2023 – Present*
- Maintain electrical distribution and BAS controls for 6 commercial properties, ensuring system uptime.
- Reduced after-hours callouts by 27% by implementing thermographic inspections and preventive scheduling.

## Projects

## Education
**Associate of Applied Science, Spokane Falls Institute of Technology** -- 2015
**Certificate, Inland Northwest Safety Academy** -- 2021

## Certifications
- SYNTHETIC-ONLY Washington 01 Journey Level Electrician License

## Technical Skills
**Electrical:** NFPA 70E, lockout/tagout, switchgear, motor controls, building automation
systems, thermographic inspections, preventive maintenance, electrical distribution,
panels, branch circuits, SMAW, GMAW, conduit bending, circuit termination
```

## 2.1 Five defects, classified

| # | Defect | Class | Evidence |
|---|---|---|---|
| D1 | **`motor controls` appears in Technical Skills** — the one thing the job ad requested and the record never mentions | **Unsupported addition** (`P2`) | `motor control` in sources: **False**; in draft: **True** |
| D2 | **`Professional Summary` heading and both summaries absent.** `12 years` absent. `Electrical Systems Technology` absent (field dropped) | **Missing required fact** (`R1`, `R3`) | `'Professional Summary' in draft: False`, `'12 years' in draft: False` |
| D3 | 6 bullets append an outcome clause: `ensuring consistent uptime`, `supporting NEC-compliant installations`, `mastering conduit bending…`, `maintaining grid reliability`, `improving safety compliance adherence`, `ensuring system uptime` | **Unsupported addition** (`P4`) | Same shape as run 10 — **the defect persists** |
| D4 | `Program Summary` segments declared missing when the writer emitted no summary section at all | **Missing required fact** (correct objection) | Segments 1, 2, 24, 25 |
| D5 | `SYNTHETIC-ONLY` marker left in the licence line | **Protocol/validation error** (writer hygiene) | Draft line `- SYNTHETIC-ONLY Washington 01 Journey Level Electrician License` |

**No wrong attribution.** Every employer, role, date, metric and institution keeps its own owner. Checked positionally.

# 3. Root cause of the failure — a target-selection bug, not a model error

The restoration payload offered four lines with **six segments**, paired by
`_pick_target_line`. Two of the four pairings are nonsensical:

| line offered | segment paired | shares content? |
|---|---|---|
| `*Journeyman Electrician \| 07/2019 – 02/2023*` | `Professional Summary` | **False** |
| `*Welder / Maintenance Helper \| 03/2012 – 09/2015*` | `…12 years of electrical experience…` | **False** |
| `**Associate of Applied Science…**` | `Associate of Applied Science, Electrical Systems Technology \| 2015` | True |
| `**Electrical:** NFPA 70E…` | `Conduit bending, switchgear troubleshooting…` | True |

The model answered correctly for what it was shown — it returned each line unchanged,
because the line and its "source fact" had nothing to do with each other. `_apply_restorations`
then rejected the answer on `_quote_shares_content(replacement, targets[line_id]["quote"])`,
and `_correct_body` converted that exception into `source_facts_missing` (the documented
best-effort path).

**Why the pairing went wrong.** `_pick_target_line` ranks candidate lines by
`(date_bonus, shared_words)`. The summary segments (`Professional Summary`, `…12 years…`)
contain **no dates**, so `wanted_dates` is empty and `date_bonus` is 0 for every line. With
no date signal, it picks by shared-word count, and the seed text
`SYNTHETIC-ONLY Journeyman electrician with 12 years of electrical experience spanning`
shares `electrician`, `electrical` and `experience` with the **role header lines**. Hence
`*Journeyman Electrician | 07/2019 – 02/2023*` — the highest-overlap line — became the
target for a summary sentence.

The two facts that *could* have been restored (`AAS … Electrical Systems Technology | 2015`,
skills) were paired correctly, but restoration is all-or-nothing: one invalid pairing
rejects the whole answer, including the two good ones.

# 4. Root cause and the smallest justified fix

I first proposed a fix to `_pick_target_line` and then **disproved it against the captured
data** — see §4.3. The real cause is elsewhere.

## 4.1 Root cause — the source has no `#` markers, so every preservation rule is blind

`source_preservation._split_sections` recognises a heading only via:

```
_SECTION_HEADING = re.compile(r"^#{1,6}\s*(?P<name>.+?)\s*$")
```

The source this run actually reviewed begins:

```
Professional Summary
SYNTHETIC-ONLY Journeyman electrician with 12 years of electrical experience spanning
```

**No `#`.** Parsed with that regex the whole document is one section with
`heading=None`, and `_find_sections(...)` matches nothing:

```
=== parsed sections ===
  heading=None   lines=42
summary matches: 0
source_summary(combined) -> ''
```

Run 10's source *did* carry `## Professional Summary`, which is why this never surfaced
before. The difference is not the fixture's content — it is that the run-10 sources were
hand-written Markdown while these were extracted from real `.docx` files by python-docx,
where section titles are **styled paragraphs, not `#` prefixes**. Every heading in the
extracted text is therefore invisible to the preservation layer.

### Consequence chain

1. `preserve_source_facts` changed **nothing**: `changed: False`, `notes: []`.
2. With no summary section visible, the summary rule never enforced `Professional Summary`
   or `12 years`.
3. The reviewer correctly reported those segments **missing** (§1).
4. The restoration path picked the wrong target for segment 1 (§4.2).
5. `_apply_restorations` rejected the answer; `_correct_body` mapped that to
   `source_facts_missing` and withheld.

So the withhold itself is **correct** — required facts genuinely were absent. The failure is
that the deterministic layer never noticed the summary it was supposed to preserve.

## 4.2 The one bad pairing

`_pick_target_line(segment 1 → some line)` chose `*Journeyman Electrician | 07/2019 – 02/2023*`.
Neither the segment nor its enclosing source line contains a date, so `wanted_dates` is
empty, `date_bonus` is 0 for every line, and the tie broke on shared-word count — the
summary sentence "Journeyman electrician with 12 years…" shares `journeyman` and
`electrician` with the role header. The other three pairings were sound (AAS→AAS, skills→skills).

## 4.3 A fix I proposed and disproved

I proposed refusing a pairing where
`not _quote_shares_content(line_text, item_quote)`. Measured on the captured data:

```
_quote_shares_content("*Journeyman Electrician | 07/2019 – 02/2023*",
                      "SYNTHETIC-ONLY Journeyman electrician with 12 years…")
  -> True      shared words: ['electrician', 'journeyman']
```

**The guard does not fire.** It would change nothing for the bug that actually occurred. It is
also already the validator's own rule — the code cannot construct a pairing the validator
would reject, because that predicate is already true. Recorded here because it was wrong.

## 4.4 The fix

**Make the section splitter recognise bare, styled headings.** A line is a heading when the
existing `#` form matches **or** the stripped line is a short, title-case phrase that is a
known resume section name.

Minimal shape, in `_split_sections`:

```python
_BARE_SECTION = re.compile(
    r"^(?P<name>(?:Professional\s+)?(?:Summary|Profile|Objective|Experience|"
    r"Professional\s+Experience|Education|Certifications?|Skills|Technical\s+Skills|"
    r"Projects|Awards|Publications))\s*:?$", re.IGNORECASE)
```

anchored to the whole line, with a length bound, so a prose sentence that happens to contain
the word "education" is not turned into a heading.

**Why this and not something larger:**

* It repairs the input to every downstream rule at once — summary, education, skills — rather
  than special-casing each. The `_find_sections` work in `770f2d4` already assumed headings
  were visible; this makes that assumption true for extracted text.
* It **cannot loosen publication**: it makes preservation *see more source*, which can only
  add enforcement. A draft that omits the summary is still withheld.
* It is local: one regex and one branch in one function. No contract change, no reviewer
  change, no gate change, no extra provider call.
* It is **verifiable offline** against the captured run-12 source: after the change,
  `source_summary(combined)` must return both summaries and `preserve_source_facts` must
  reinsert `Professional Summary`.

**What it does not fix:** D1 (`motor controls`), D3 (six appended clauses) and D5 (marker
leak) are writer-side and remain. It also does not fix the restoration target choice; with
the summary restored the reviewer would no longer report segment 1 missing, so that pairing
would not arise — but the underlying scoring weakness stays.

### 4.5 Verification of the fix — prototyped and measured

I prototyped the bare-heading regex against the captured source. Result:

```
BEFORE:  sections: [(None, 42)]          source_summary: ''
AFTER :  sections: [('Professional Summary',2), ('Professional Experience',13),
                    ('Education',2), ('Skills',5), ('Professional Summary',2), ...]
         source_summary: 'Journeyman electrician with 12 years of electrical experience
         spanning commercial construction, industrial maintenance, and facility
         operations. Maintenance electrician focused on electrical distribution and
         building automation systems in occupied commercial properties.'
```

Both documents' summaries are recovered, and the false-positive controls hold:

| line | matched as heading |
|---|---|
| `Professional Summary` | **True** |
| `Education` | **True** |
| `Skills` | **True** |
| `Electrical Systems Technology` | False |
| `Concrete Work` | False |
| prose sentence containing "education" | False |

## 4.6 The second defect — isolated: it is rule ORDER, not a coding error

I earlier recorded this as "un-isolated". It is now isolated by stepping the rules one at a
time on the captured input:

```
after preserve_education_record        has_heading=False  has_12years=False  changes=[]
after preserve_source_summary          has_heading=False  has_12years=False  changes=[]
after preserve_quantified_qualifiers   has_heading=False  has_12years=False  changes=[]
after preserve_skills_block            has_heading=False  has_12years=False  changes=["replaced an unsupported skills block..."]
after ensure_required_headings         has_heading=True   has_12years=False  changes=["added an empty Professional Summary heading"]
```

`preserve_source_facts` runs:

```python
for step in (preserve_education_record,
             preserve_source_summary,
             preserve_quantified_qualifiers,
             preserve_skills_block,
             ensure_required_headings):
```

`preserve_source_summary` runs **before** `ensure_required_headings`. With no summary
section yet in the draft it takes its `if section is None: return markdown, []` branch — the
branch whose comment says insertion "would be composing a section from outside the draft".
`ensure_required_headings` then adds `## Professional Summary` with `lines: [""]`, and
**nothing revisits it**. So the heading exists and the body is permanently empty.

Both behaviours are individually defensible. Their order is not: the engine adds the section
*after* the rule that fills it has already declined.

### Reordering, measured

```
after ensure_required_headings    has12years=False  changes=['added an empty Professional Summary heading']
after preserve_education_record   has12years=False  changes=[]
after preserve_source_summary     has12years=True   changes=['restored the source summary wording']
after preserve_quantified_qualifiers has12years=True changes=[]
after preserve_skills_block       has12years=True   changes=["replaced an unsupported skills block..."]
```

Output:

```markdown
## Professional Summary

Journeyman electrician with 12 years of electrical experience spanning commercial
construction, industrial maintenance, and facility operations. Maintenance electrician
focused on electrical distribution and building automation systems in occupied commercial
properties.
```

## 4.6b Coverage of the two changes, measured against the 9 missing segments

| seg | content | now present |
|---|---|---|
| 0 | `Professional Summary` | **True** |
| 1 | `12 years of electrical experience` | **True** |
| 2 | `commercial construction` | **True** |
| 19 | `Electrical Systems Technology` | False |
| 21 | `Conduit bending` | **True** |
| 22 | `Lockout/tagout` | **True** |
| 23 | `Professional Summary` (doc-1 heading) | False |
| 24 | `Maintenance electrician focused` | **True** |
| 25 | `building automation` | **True** |

**7 of 9.** The two remaining are not detection failures:

* **seg 19** — the draft says `Associate of Applied Science, Spokane Falls Institute of
  Technology` and the source says `Associate of Applied Science, Electrical Systems
  Technology \| 2015`. The writer **dropped the field of study** and merged the institution
  into the award line. `preserve_education_record` does mention both institution and field,
  so this is a rule-coverage gap, not a parsing gap — a third, separate defect.
* **seg 23** — doc-1's `Professional Summary` heading does not survive because the two
  summaries are merged under one heading, which is the intended multi-document behaviour.

## 4.7 The fix, final

Two deterministic-layer changes, both measured:

1. **`_split_sections` recognises bare styled headings** — real DOCX extraction never emits
   `#`, so today every uploaded document is one unheaded section.
2. **`ensure_required_headings` runs before `preserve_source_summary`** — so the section the
   summary rule fills exists before it is asked to fill it.

Neither touches the reviewer, the gate, or the writer. Neither can loosen publication:
both only make the engine **see more source** it can enforce. A draft that omits the summary
is still withheld.

**Third defect, found and NOT fixed:** the education rule does not restore a dropped
**field of study** (seg 19). Separate rule-coverage gap. Recorded, not addressed here.

No code has been changed. Every prototype above ran from this analysis against the preserved
capture, outside the repository.

### Verification plan when the code is changed (offline, no spend)

1. `source_summary(<captured run-12 combined>)` returns both summaries.
2. `preserve_source_facts(<captured run-12 draft>, combined)` produces the summary text and
   `12 years`.
3. Full suite still passes, especially `test_source_preservation.py` §6 (multi-document
   summaries) and `test_route_two_documents.py` (a correct draft still approves).
4. New paired test: a bare-heading source must yield the same preservation result as the same
   source with `##` added — that is the invariant the bug violated.

# 5. What this establishes about the open questions

| Question | Answer |
|---|---|
| Does run 10's appended-outcome defect persist? | **Yes.** Six appends in this draft, same shape (`ensuring…`, `maintaining…`, `improving…`). Run 10 was not an anomaly. |
| Is the withheld draft actually defective? | **Yes, this time demonstrably.** `motor controls` is in the draft and in neither source — an unsupported addition, not a reviewer misjudgement. |
| Was run 11's withhold sound? | **Still unknown** — no draft was retained. Not resolved by run 12. |
| Is the gate's objection correct here? | **Yes for `source_facts_missing`**: the summary and skills were genuinely absent. But the *restoration attempt* was malformed, so the engine never got the chance to fix them. |

# 6. Cost

| | |
|---|---|
| Account credits | 33 (902 → 869) |
| Correction calls | 3 — inventory, review, restore |
| `repair_calls` | 0 (never reached; restoration precedes repair) |
| Tokens | inventory ~2.4k, review ~5.4k, restore ~2.6k |

# 7. Cleanup

Capture preserved and hashed **before** cleanup (§header). Remote directory purged after
verification. See `docs/live-runs/RUN_12_CLEANUP.md`.
