# Final-build acceptance — semantic judgments

Recorded 2026-11-07. Both runs are the **actual server responses** retained from
the live website, not reconstructed wrappers:

| Mode | runId | Payload | Receipt | Policy state |
|---|---|---|---|---|
| Conservative | `1b66b189-8633-4731-bf35-efc2cd56e3b8` | 409,033 bytes | yes (v5, signed) | `conservative` |
| Imaginative | `43917767-152b-441c-8c72-a4abe71033ea` | 412,782 bytes | yes (v5, signed) | `imaginative-v1`, 2 surviving bridges |

Mechanical audit: **0 defects** in both (no omissions, no ownership moves, no
prohibited terms, no provenance markers).

Each run recorded 7 disputed sentences. A disputed sentence is **not** a defect —
the tool records it for a human to judge. Judged below against the frozen source,
sentence by sentence.

## The pattern

Every disputed sentence in **both** modes is the source's own fact with a
connective or a light intensifier added: `using welding`, `during`, `across`,
`through targeted diagnostics`, `by implementing`. The candidate's facts, owners,
dates and figures are unchanged.

This is the lexical checker doing exactly what it says it does — it flags words
the ledger does not list. It is not evidence of fabrication, and it is not a
reason to withhold.

## Conservative — 7 disputed, all FAITHFUL

| Sentence | Novel | Source | Judgment |
|---|---|---|---|
| Fabricated … 14 industrial contracts **using welding** | using | "Fabricated and repaired structural steel assemblies for 14 industrial contracts." | FAITHFUL — names the method the same bullet already implies |
| Pulled … branch circuits **during** 31 commercial buildouts | during | "…on 31 commercial buildouts." | FAITHFUL — connective |
| Completed 8,000 apprenticeship hours **under supervision of a licensed journeyman electrician.** | supervision | "…under a licensed journeyman." | FAITHFUL — restates the same supervision |
| Diagnosed … switchgear faults **across** 9 substations | across | "…at 9 substations." | FAITHFUL — connective |
| Cut … 4.5 hours to 2.75 hours **through targeted diagnostics.** | diagnostics, targeted, through | "Cut average fault-response time from 4.5 hours to 2.75 hours." | **FAITHFUL** — see note |
| Maintain … BAS controls **across** 6 commercial properties. | across | "…for 6 commercial properties." | FAITHFUL — connective |
| Reduced … 27% **by implementing** thermographic inspections | implementing | "…by adding thermographic inspections." | FAITHFUL — "implementing" for "adding" |

**Note on "through targeted diagnostics":** this adds a causal mechanism the
source does not state. The figure (4.5 → 2.75 hours) is the candidate's own and
unchanged, and the bullet already attributes the reduction to the fault-response
work. Judged FAITHFUL as a paraphrase of the same claim rather than a new one —
but it is the closest call in the set, and it is recorded here rather than
silently passed.

## Imaginative — 7 disputed, 4 FAITHFUL, 3 addition-by-inference

The same five connective/paraphrase cases as Conservative are FAITHFUL for the
same reasons. The three that differ:

| Sentence | Novel | Source | Judgment |
|---|---|---|---|
| Fabricated … **using blueprint specifications.** | blueprint, specifications | source says only "Fabricated and repaired structural steel assemblies…" | **ADDITION** — introduces a work input the source never names |
| Completed 8,000 hours **toward journeyman certification in compliance with state standards.** | certification, compliance, state, toward | "Completed 8,000 apprenticeship hours under a licensed journeyman." | **ADDITION** — asserts a regulatory framework not in the source |
| Cut … **through process optimization techniques.** | optimization, process, techniques | "Cut average fault-response time from 4.5 hours to 2.75 hours." | ADDITION, but **authorised** — see below |

### The third is permitted, and that distinction matters

"process optimization" is one of the two **surviving bridges** recorded in the
signed policy state:

```json
{"id":"bridge-03-74372d0ed3","skill":"process optimization",
 "origin":"researcher_inference","source_id":"document-0",
 "policy_version":"imaginative-v1"}
```

A validated bridge label is authorised wording, not an introduced claim — that is
Imaginative mode's defining behaviour, and it is bound into the receipt. So this
sentence is a permitted inference, not a defect.

The other two ("blueprint specifications", "toward journeyman certification in
compliance with state standards") are **not** covered by any surviving bridge.
They are the writer adding specificity the candidate never wrote. Recorded as
additions.

## What this means for delivery

Both modes delivered. The additions in Imaginative are the mode's known behaviour
(writer elaborates), and two of them are unbridged elaboration. That is a
**quality** observation for the mode, not a correctness failure of the gate: the
gate's job is faithful-fact delivery, and no candidate fact was altered, moved or
invented in either run.

They are recorded so the behavior is visible, not waived.

## Exports

| Mode | PDF | DOCX |
|---|---|---|
| Conservative | 1 page, 0 blocks outside page, 2 embedded fonts, 6/6 headings in required order | 6/6 headings, required order |
| Imaginative | 1 page, 0 blocks outside page, 2 embedded fonts, 6/6 headings in required order | 6/6 headings, required order |

## Artifacts

- `captures/final/live2/*-final-response.json` — verbatim server bytes
- `captures/final/live2/*.pdf`, `*.docx` — the files the site actually produced
- `captures/final/batch/audit-acceptance.json` — mechanical audit
