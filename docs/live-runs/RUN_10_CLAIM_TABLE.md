# Run 10 — claim-level findings table (revised)

Correlation ID `850ee3ea-ca2e-4210-92b4-0e57ed5149eb`, revision `47c4254`.
Evidence held unchanged in `.live-run-2026-11-02/captures/run10/`:

| Artifact | Bytes | SHA-256 |
|---|---|---|
| `850ee3ea-….json` (capture) | 30062 | `c141846a0d6ef29242e4cf24095978c52c1a88ae671bc84d639de8db4a98713f` |
| `run10_approved.md` | 2079 | `23102b5114d3cf545d90c3a87d9dcd822c13e81ea55636f72e38d5a45996dac3` |
| `run10_summary.json` | 866 | `3c930dc1b4c6bcef62f35ceb0ecf2f2d4d4751d3ad6d67c33cfedec8093704bc` |

Nothing here altered the capture, the sources, the approved response, or the hashes.

## Revision note

An earlier version of this table used a flatter classification and left
"enhancing system reliability" unevaluated alongside clear additions. Two changes:

1. The categories are now the five SPEC.v2 asks for, applied per clause.
2. **"enhancing system reliability" is reassessed.** The source bullet already
   reports a measured reliability outcome — "Reduced unplanned downtime by 22%
   through a preventive maintenance schedule." Unplanned downtime *is* a
   reliability measure. Restating that reduction as an improvement in system
   reliability is a qualitative restatement of the same measured fact, not a new
   claim. The absence of a *second*, differently-worded statistic does not make the
   restatement false.

## Classification categories

| Category | Meaning |
|---|---|
| **Faithful paraphrase** | Same facts, different words. No new proposition. |
| **Supported qualitative restatement** | Generalises a fact the source measures, without adding a new measured quantity, entity, or causal link. |
| **New causal or performance outcome** | Asserts an outcome, cause, or performance result the source does not state, or a causal link it does not draw. |
| **Incorrect attribution** | The fact exists but is attached to the wrong employer, date, institution or role. |
| **Uncertain** | The evidence does not settle it, or the judgement depends on information the record lacks. |

A clause is judged against the **complete text of both documents** and its
attribution — never against a word list. Words absent from the source are not
fabrication; words present in the source can be combined into a false claim.

Sources:
- `document-0` (1232 chars) — Cascade Power Solutions (electrician), Harborline
  Mechanical Contractors (apprentice), Example Technical College.
- `document-1` (676 chars) — Bluegate Property Services (facilities maintenance),
  Example Community College.

---

## 1. Professional Summary

| Source | Output | Category | Reasoning | Remedy |
|---|---|---|---|---|
| `doc-0`: "SYNTHETIC-ONLY Journeyman electrician with 8+ years of commercial and industrial electrical experience across construction, facilities maintenance, and power distribution." | "Journeyman electrician with 8+ years of commercial and industrial electrical experience across construction, facilities maintenance, and power distribution." (L3) | Faithful paraphrase | Only the provenance marker is removed. Every fact and list member present, same order. | None |
| `doc-1`: "SYNTHETIC-ONLY Facilities maintenance technician focused on preventive maintenance of electrical and mechanical building systems." | "Facilities maintenance technician focused on preventive maintenance of electrical and mechanical building systems." (L4) | Faithful paraphrase | Marker removed only. | None |
| Both summaries, no single source containing both | The two clauses appear consecutively inside one summary section | Faithful paraphrase (cross-document synthesis) | Each clause is independently attributable and preserved. Combining supported summaries is legitimate, not an attribution error. | None |

## 2. Cascade Power Solutions, LLC — Journeyman Electrician (`document-0`)

| Source | Output | Category | Reasoning | Remedy |
|---|---|---|---|---|
| "Installed and maintained commercial electrical systems across 40+ job sites." | "…across 40+ job sites, **ensuring compliance with NFPA 70E standards**." (L6) | **New causal or performance outcome** | The activity is faithful. The appended clause asserts a *compliance outcome* of that installation work. `doc-0` ties NFPA 70E to apprenticeship **training** and to the **certification**, never to this installation activity, and draws no causal link between installing and achieving compliance. | Repair: drop the clause. No rewording makes the source state it. |
| "Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices." | "…safe work practices, **improving workforce safety and code adherence**." (L7) | **New causal or performance outcome** | Training is sourced; the improvement in safety and code adherence is a measured outcome the source never reports, and the causal link from training to that improvement is not drawn. | Repair: drop the clause. |
| "Cut rework by 18% by standardizing panel labeling and inspection checklists." | "…inspection checklists, **increasing installation efficiency and quality control**." (L8) | **New causal or performance outcome** | The 18% rework reduction is sourced. "Efficiency" and "quality control" are *different* performance measures that the rework figure does not report, and the clause presents them as further effects of the same activity. | Repair: drop the clause. The sourced 18% claim alone is faithful. |

## 3. Harborline Mechanical Contractors — Electrical Apprentice (`document-0`)

| Source | Output | Category | Reasoning | Remedy |
|---|---|---|---|---|
| "Assisted with conduit bending, wire pulling, and panel terminations on 25 projects." | "**Supported** conduit bending, wire pulling, and panel terminations across 25 **commercial** projects **under licensed supervision**." (L12) | Faithful paraphrase | "Supported"/"Assisted" is a synonym here. "Under licensed supervision" is attested by the same document's next bullet ("supervised field work toward licensure"). "Commercial" is inferred from the summary's "commercial and industrial electrical experience". Neither adds a new proposition about this role. | None |
| "Completed 8,000 hours of supervised field work toward licensure." | "…toward journeyman licensure, **gaining hands‑on electrical construction experience**." (L13) | **Supported qualitative restatement** | The clause adds no qualification, employer, date, metric or credential. It characterises the field work the same sentence describes, and the summary already states electrical construction experience. | **None — keep it.** See §7. |

## 4. Bluegate Property Services — Facilities Maintenance Technician (`document-1`)

| Source | Output | Category | Reasoning | Remedy |
|---|---|---|---|---|
| "Maintained electrical distribution and HVAC controls for 12 commercial buildings." | "…for 12 commercial buildings, **ensuring continuous operations**." (L15) | **New causal or performance outcome** | Maintenance is faithful. The clause asserts an operational-continuity guarantee for those buildings. No source describes uptime, continuity or availability for this site. This is the clearest addition of the six. | Repair: drop the clause. |
| "Reduced unplanned downtime by 22% through a preventive maintenance schedule." | "…preventive maintenance schedules, **enhancing system reliability**." (L16) | **Supported qualitative restatement** — **reassessed** | Directly entailed by the measured fact. Unplanned downtime is a reliability measure; a 22% reduction in it *is* improved reliability. The clause adds no new quantity, no new entity, no new causal claim — it names the property the source already measured. It is neither a new performance outcome nor an unbacked assertion. | **None — keep it.** |

## 5. Education, Certifications, Skills

| Source | Output | Category | Reasoning | Remedy |
|---|---|---|---|---|
| `doc-0` Example Technical College — AAS, Electrical Systems Technology, 2016 | Same | Faithful paraphrase | Marker removed; institution, degree, field, year intact and correctly paired. | None |
| `doc-1` Example Community College — Certificate, Building Systems Maintenance, 2013 | Same | Faithful paraphrase | Same. No mixing with the Technical College record. | None |
| `doc-0` Washington Journey Level Electrician License; NFPA 70E Arc Flash Safety | Same | Faithful paraphrase | Markers removed only. No issuer, year or credential added. | None |
| `doc-0` skills (3 terms) + `doc-1` skills (3 terms) | One Technical Skills block with all six source terms | Faithful paraphrase (cross-document synthesis) | Every listed term is in a source skills block. Merging skills lists asserts nothing new. | None |
| Neither document has a Projects section | Empty `## Projects` heading (L18–19) | Uncertain (structural) | An empty heading asserts nothing but is carried from the writer's skeleton, not from a claim. Whether an empty section should render is a presentation decision, not a truth question. | Optional: suppress empty sections at render. |

---

## 6. Attribution check (independent of wording)

| Test | Result |
|---|---|
| Employer→role | Cascade→Journeyman Electrician; Harborline→Electrical Apprentice; Bluegate→Facilities Maintenance Technician. Correct. |
| Employer→dates | `03/2020 – Present`, `06/2016 – 02/2020`, `01/2014 – 05/2016`. Correct, none exchanged. |
| Institution→degree | Technical College→AAS Electrical Systems Technology 2016; Community College→Certificate Building Systems Maintenance 2013. Correct, not mixed. |
| Metric ownership | 40+ sites, 18% rework, 6 apprentices→Cascade; 25 projects, 8,000 hours→Harborline; 12 buildings, 22% downtime→Bluegate. No figure crosses an employer. |
| Duplicates | Each employer and institution appears exactly once. |
| Invented capitalised entities | None. |

**No incorrect attribution was found in run 10.** All five categories appear in the
table (uncertain via the empty Projects heading), but the attribution layer itself is
clean.

## 7. Summary of the six appended clauses

| Clause | Category | Action |
|---|---|---|
| `ensuring compliance with NFPA 70E standards` | New causal/performance outcome | Repair |
| `improving workforce safety and code adherence` | New causal/performance outcome | Repair |
| `increasing installation efficiency and quality control` | New causal/performance outcome | Repair |
| `gaining hands‑on electrical construction experience` | Supported qualitative restatement | **Keep** |
| `ensuring continuous operations` | New causal/performance outcome | Repair |
| `enhancing system reliability` | Supported qualitative restatement | **Keep** |

**Four additions, two legitimate restatements.** The earlier version treated all six
as one category; that was wrong in both directions — it would have deleted "enhancing
system reliability", which is entailed by the sourced 22% downtime reduction.

The pattern separating the two groups: an **addition** asserts an outcome or property
the source never measured (compliance status, safety improvement, efficiency,
operational continuity). A **restatement** names the property the source already
measured (reliability via downtime) or characterises work the source already describes
(field work → hands-on experience).

The work list in `appended_clauses.py` reports all six and assesses **none** of them.
That division is deliberate: the lexical layer localises; this table judges.
