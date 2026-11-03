# Acceptance note for Proposal A — with four clarifications applied

The request referred to "the four acceptance-note clarifications above," but no list
accompanied it. Rather than invent four, these are **derived from the requirements stated
in the same message** and applied to Proposal A. If the intended four differ, only this
file needs revising before the run.

## Clarification 1 — record, do not predict

The acceptance note states what was observed. It does not forecast the run-10 defect, and
it does not characterise that defect as a regression to be expected.

| Observation | How it is recorded |
|---|---|
| Appended outcome clauses present | "Observed on N bullets; listed verbatim" |
| Appended outcome clauses absent | "Not observed in this run" |

And the interpretation rule, stated in advance so it cannot be applied selectively after
the fact:

- **A single clean result does not clear the historical defect.** Run 10's capture is
  immutable evidence that the defect occurred; one run without it does not retract that.
- **A repeated defect establishes that it persists on this realistic case** — a stronger,
  broader finding than the run-10 case, because this fixture carries more facts, two
  states, and a supplement-only employer.

## Clarification 2 — ordinary interaction first, forced clicks labelled separately

Order is fixed:

1. Attempt the ordinary click. Record the outcome.
2. If it fails, record the failure and the diagnostic (`document.elementFromPoint` at the
   button centre) **before** any workaround.
3. If a forced or JavaScript click is used, its success is reported under a separate
   heading: *pipeline succeeded under injected interaction*, never as ordinary-user
   acceptance.

## Clarification 3 — export metadata observed, not inferred

At export time, record directly from the request and response, not from the saved filename:

| Field | Source |
|---|---|
| Request URL and method | browser network |
| Request payload `headline` | request body |
| Response `filename` | response body |
| Client-chosen download name | the `download` attribute at save time |

The stale-title anomaly is then **observable**: if the request headline and the filename
disagree with the typed job title, the mismatch is a recorded measurement rather than a
reconstruction from saved files. This is the specific correction from the earlier
investigation, where the payload was inferred from the filename.

## Clarification 4 — four reporting dimensions kept separate

| Dimension | What it covers | What it must not absorb |
|---|---|---|
| **UI interaction** | upload, filename visibility, click behaviour, download | content correctness |
| **Factual accuracy** | F01–F22 against the output | whether the download worked |
| **Receipt / export integrity** | receipt accepted, export 200, content bound to the approval | visual appearance |
| **Visual quality** | PDF layout, page breaks, truncation, spacing | factual or gate questions |

A pass in one dimension is never reported as a pass in another. Specifically: an export
that returns 200 proves the receipt bound the content; it says nothing about whether that
content is true. That distinction was the error in the run-10 write-up, where an
instrumentation miss in the download dimension was reported as a product failure, and the
corrected version risked the inverse — treating a 200 as content validation.

## What this note does not change

The frozen inputs, the F01–F22 ledger, the stop conditions, the no-retry rule, the
no-deployment rule, and the cleanup procedure are unchanged from Proposal A.
