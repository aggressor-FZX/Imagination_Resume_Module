# Acceptance note for Proposal A — the four specified clarifications

Replaces the derived version. These are the user's four, applied as stated.

## 1. "12 years" is a supplied claim, not a recalculated one

`candidate-main.txt` states "Journeyman electrician with 12 years of electrical
experience". The chronology does **not** produce 12:

| Span | Years |
|---|---|
| 03/2012 → 09/2015 (Northline) | 3.5 |
| 10/2015 → 06/2019 (Kettle Falls) | 3.7 |
| 07/2019 → 02/2023 (Pend Oreille) | 3.6 |
| 03/2023 → Present (Riverbend) | ~3.7 |
| **Total elapsed** | **~14.5** |
| **Electrical only** (excluding the welding role) | **~11.0** |

So "12 years" matches neither total exactly. It is the candidate's own supplied claim and
**must be preserved as written**. The run must not:

- recalculate it from dates and rewrite it as 11 or 14;
- treat 12 as an unsupported figure and withhold for it.

Check: "12 years" appears, or a faithful equivalent such as "over a decade". If the system
rewrites it to a different number, that is a tool defect — a **checker-side** expectation,
not a source-fact error.

## 2. Source skills belong in the evidence checklist

Both documents carry Skills blocks. They are retained facts and must be checked, not just
the achievements:

| Source | Skills |
|---|---|
| main | Conduit bending · switchgear troubleshooting · thermographic inspection · lockout/tagout · NFPA 70E |
| supplement | Electrical distribution · BAS controls · thermographic inspection · NFPA 70E · lockout/tagout |

Check: each distinct skill appears in the output's skills area or is faithfully restated.
**Motor controls must not appear** — it is in the job ad and in no skills block. Skills may
be merged, reordered, or grouped; merging is legitimate cross-document synthesis.

## 3. Where the candidate's name and contact information enters the workflow

**This fixture supplies none.** Neither DOCX contains a name, phone, email, or location
header — both start at `## Professional Summary`. This is deliberate, and it matters:

| Question | Answer for this run |
|---|---|
| Does the source carry identity? | **No.** |
| Where would a name come from? | The account/form context, not the documents. |
| Expected résumé header | **Empty or a neutral fallback.** Not "Dana R. Whitfield" — that name exists only in planning documents, never in the frozen inputs. |
| Is a missing name a defect? | **No.** With no identity in the source, an empty header is correct. |
| Is an invented name a defect? | **Yes.** A person's name in neither document nor the form is fabrication. |
| Is a substituted location a defect? | **Yes** — the two real locations in the body must stay as written. |

The README's fictional name and contact address are **planning metadata, not inputs**. They
were never placed in the DOCX files, so they cannot appear in any output, and their absence
is not a finding.

## 4. Faithful paraphrases permitted; unsupported outcomes prohibited

The distinction the run must apply, per clause:

| Permitted | Prohibited |
|---|---|
| Different verbs for the same action ("Supported" for "Assisted") | A new outcome the source never measures |
| Reordered or merged lists ("and" coordination across two documents) | A causal link the source does not draw |
| A restatement naming a property the source already measured ("faster response" for 4.5h→2.75h) | A property the source never mentions ("continuous operations", or "reliability" with no measure behind it) |
| Removing the `SYNTHETIC-ONLY` marker | Adding a credential, issuer, employer, institution, or figure |

A clause is judged against the complete text of both documents. Words absent from the source
are not evidence of fabrication, and words present in the source can be recombined into a
false claim.

---

## Unchanged

Frozen inputs, F01–F22, stop conditions, no-retry, no-deployment, cleanup. The run measures
current behaviour and does not imply the run-10 semantic defect is fixed.
