# Run 10 — claim-level findings table

Correlation ID `850ee3ea-ca2e-4210-92b4-0e57ed5149eb`, revision `47c4254`.
Evidence held unchanged in `.live-run-2026-11-02/captures/run10/`:

| Artifact | Bytes | Hash |
|---|---|---|
| `850ee3ea-….json` (capture) | 30062 | sha256 `c141846a0d6ef29242e4cf24095978c52c1a88ae671bc84d639de8db4a98713f` |
| `run10_approved.md` | 2079 | unchanged |
| `run10_summary.json` | 866 | unchanged |

Nothing in this document altered the capture, the source documents, the approved
response, or the original hashes.

## Why this table exists

`SPEC.v2.md` separates **deterministic Tier-A checks** from **semantic Tier-B review**.
Run 10 passed Tier A (publication gate `[]`, review `approved`) and still carries
sentences that no source supports. Mechanical passage is not semantic proof.

**Vocabulary is not the test.** "Words absent from the source" is not a definition of
fabrication: different words routinely express the same fact, and words individually
present in the source can be combined into a claim the source never makes. Each row
below is judged against the complete source text of both documents **and** its
attribution, not against a word list.

Sources:
- `document-0` (1232 chars) — Cascade Power Solutions (electrician), Harborline
  Mechanical Contractors (apprentice), Example Technical College.
- `document-1` (676 chars) — Bluegate Property Services (facilities maintenance),
  Example Community College.

---

## 1. Professional Summary

| Source | Output | Classification | Explanation | Remedy |
|---|---|---|---|---|
| `document-0`: "SYNTHETIC-ONLY Journeyman electrician with 8+ years of commercial and industrial electrical experience across construction, facilities maintenance, and power distribution." | "Journeyman electrician with 8+ years of commercial and industrial electrical experience across construction, facilities maintenance, and power distribution." (line 3) | Faithful paraphrase | Only the provenance marker is removed. Every fact and every list member is present in the same order and attribution. | None |
| `document-1`: "SYNTHETIC-ONLY Facilities maintenance technician focused on preventive maintenance of electrical and mechanical building systems." | "Facilities maintenance technician focused on preventive maintenance of electrical and mechanical building systems." (line 4) | Faithful paraphrase | Marker removed only. | None |
| Both summaries above, no single source containing both | The two clauses appear **consecutively on two lines** inside one `Professional Summary` section | Supported cross-document synthesis | Each clause is independently attributable to its own document and each is preserved verbatim in meaning. Combining two supported summaries is legitimate; it is not itself an attribution error. | None |

## 2. Cascade Power Solutions, LLC — Journeyman Electrician (source: `document-0`)

| Source | Output | Classification | Explanation | Remedy |
|---|---|---|---|---|
| "Installed and maintained commercial electrical systems across 40+ job sites." | "Installed and maintained commercial electrical systems across 40+ job sites, **ensuring compliance with NFPA 70E standards**." (line 6) | Unsupported addition | The activity is faithful. The comma + participial clause asserts a *compliance outcome* of that installation work. `document-0` associates NFPA 70E with **apprenticeship training** (a different bullet) and with the **certification**, never with this installation activity. Activity does not establish its own outcome. | Repair: drop the clause, or withhold. **Not** a wording correction — no rewording makes the source state it. |
| "Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices." | "Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices, **improving workforce safety and code adherence**." (line 7) | Unsupported addition | Training happened; the improvement in safety and "code adherence" is a measured outcome the source never states. It also implies evaluation the source does not describe. | Repair: drop the clause. |
| "Cut rework by 18% by standardizing panel labeling and inspection checklists." | "Cut rework by 18% by standardizing panel labeling and inspection checklists, **increasing installation efficiency and quality control**." (line 8) | Unsupported addition | The 18% rework reduction *is* sourced. The appended "increasing installation efficiency and quality control" restates a **different** claimed outcome (efficiency, quality control) that the 18% figure does not measure. Two causal chains are being attributed to one activity. | Repair: drop the clause. The sourced 18% claim alone is faithful. |

## 3. Harborline Mechanical Contractors — Electrical Apprentice (source: `document-0`)

| Source | Output | Classification | Explanation | Remedy |
|---|---|---|---|---|
| "Assisted with conduit bending, wire pulling, and panel terminations on 25 projects." | "**Supported** conduit bending, wire pulling, and panel terminations across 25 commercial projects **under licensed supervision**." (line 12) | Faithful paraphrase, with one supported inference | "Supported" for "Assisted" is a synonym in this context. "under licensed supervision" is **not** in this bullet, but it *is* in the same document's next bullet ("supervised field work toward licensure"), so the licence-supervision condition is attested for this employment relationship. The word "commercial" is inferred from the summary's "commercial and industrial electrical experience". Both are the kind of inference a reader makes from the record as a whole. | None. (Note: the `< 0.6` word-overlap on the trailing members is a lexical artefact, not an unsupported claim.) |
| "Completed 8,000 hours of supervised field work toward licensure." | "Completed 8,000 hours of supervised field work toward journeyman licensure, **gaining hands-on electrical construction experience**." (line 13) | Supported inference (not an addition) | The clause does **not** introduce a new qualification, employer, date, metric or credential. It characterises the field work the same sentence has just described. `document-0` already establishes hands-on electrical construction work for this employer, and the summary states electrical construction experience. This is a reasonable paraphrase of supervised electrical field work. | **None.** Do not strip this clause. |

## 4. Bluegate Property Services — Facilities Maintenance Technician (source: `document-1`)

| Source | Output | Classification | Explanation | Remedy |
|---|---|---|---|---|
| "Maintained electrical distribution and HVAC controls for 12 commercial buildings." | "Maintained electrical distribution systems and HVAC controls for 12 commercial buildings, **ensuring continuous operations**." (line 15) | Unsupported addition | Maintenance activity is faithful; the sentence now asserts an **operational continuity guarantee** for those buildings. No source describes uptime, continuity, or availability for this site. This is the strongest of the six: the appended clause is a substantive service-level claim. | Repair: drop the clause. |
| "Reduced unplanned downtime by 22% through a preventive maintenance schedule." | "Reduced unplanned downtime by 22% through implementation of preventive maintenance schedules, **enhancing system reliability**." (line 16) | Unsupported addition | "Reduced unplanned downtime by 22%" is sourced. "enhancing system reliability" is a **separate, broader reliability outcome** the 22% downtime figure does not establish; availability and reliability are distinct properties. | Repair: drop the clause. |

## 5. Education, Certifications, Skills

| Source | Output | Classification | Explanation | Remedy |
|---|---|---|---|---|
| `document-0` "**SYNTHETIC-ONLY Example Technical College** -- Example City, WA / *Associate of Applied Science, Electrical Systems Technology \| 2016*" | "**Example Technical College** -- Example City, WA / *Associate of Applied Science, Electrical Systems Technology \| 2016*" | Faithful paraphrase | Marker removed; institution, degree, field and year intact and correctly paired. | None |
| `document-1` "**SYNTHETIC-ONLY Example Community College** -- Sampletown, OR / *Certificate, Building Systems Maintenance \| 2013*" | "**Example Community College** -- Sampletown, OR / *Certificate, Building Systems Maintenance \| 2013*" | Faithful paraphrase | Same. No mixing with the Technical College record. | None |
| `document-0` "SYNTHETIC-ONLY Washington Journey Level Electrician License" / "SYNTHETIC-ONLY NFPA 70E Arc Flash Safety" | "Washington Journey Level Electrician License" / "NFPA 70E Arc Flash Safety" | Faithful paraphrase | Markers removed only. No issuer, year, or credential added. | None |
| `document-0` Skills (3 terms) + `document-1` Skills (3 terms) | One `Technical Skills` block listing all six source terms | Supported cross-document synthesis | Every listed term is present in a source skills block. Merging skills lists is not an attribution claim. | None |
| `document-0` has **no** Projects section; `document-1` has none | Empty `## Projects` heading is emitted (lines 18–19) | Uncertain (structural, not fabrication) | An empty heading asserts nothing. It is carried through from the writer's section skeleton, not from a source claim. Cosmetic only. | Checker correction (optional): suppress empty sections at render. Not a correctness defect. |

---

## Attribution check (independent of wording)

| Test | Result |
|---|---|
| Employer→role pairing | Cascade→Journeyman Electrician, Harborline→Electrical Apprentice, Bluegate→Facilities Maintenance Technician. All correct. |
| Employer→date pairing | `03/2020 – Present`, `06/2016 – 02/2020`, `01/2014 – 05/2016`. All correct, none exchanged. |
| Institution→degree pairing | Technical College→AAS Electrical Systems Technology 2016; Community College→Certificate Building Systems Maintenance 2013. Correct, not mixed. |
| Metric ownership | 40+ sites→Cascade; 18% rework→Cascade; 6 apprentices→Cascade; 25 projects / 8,000 hours→Harborline; 12 buildings / 22% downtime→Bluegate. Every figure stays under its own employer. |
| Duplicate records | Each employer and institution appears exactly once. |
| Invented capitalised entities | None. `_introduces_beyond` returns `[]` for every line, and a whole-document scan finds no capitalised name absent from both sources. |

**Conclusion:** the attribution layer is correct. The defect is confined to **six appended
outcome clauses**, of which **five are unsupported additions** and **one
("gaining hands-on electrical construction experience") is a supported inference that must
be kept**. Treating all six as one category would delete a legitimate paraphrase —
which is exactly why the table judges them individually rather than as a block.
