# Contract probe findings — the legacy adapter cites the draft as evidence

**No matrix was run.** The contract probe stopped the run five times, and each
attempt to make it pass uncovered a genuine defect in one of the two adapters. That
is the probe doing its job; spending on 28 matrix attempts first would have
produced the same uninterpretable table as v1 at greater cost.

Probe directories: `results/probe-20261109-{b,c,d,e,f}` · 4 calls each, ≈20 total.

---

## Finding 1 — the legacy adapter quotes the DRAFT and calls it source evidence

The single most important result. One `current` answer contained 20 citations.
Classified by where the quoted text actually appears:

| Verdict | Count | Meaning |
|---|---|---|
| `offsets_off` | 12 | Quote is real **source** text; the offset differs |
| `cites_draft` | **5** | Quote is text from the **document under review** |
| `fabricated` | 3 | Quote is in neither document |

Two of the `cites_draft` quotes literally contained the string `DRAFT TO REVIEW:`,
which is harness framing text, not résumé content:

```
quote: '\n\nDRAFT TO REVIEW:\n## Professional Summary\nOperations analyst with …'
```

**This is not fabrication and not an offset error.** The reviewer quoted the
document it was reviewing and attributed it to a source document. The claim may even
be true, but the evidence offered does not establish it.

Why this matters for the comparison: v1 could not see this distinction. It had one
`reference_invalid` bucket, which is why its only completed legacy answer was
recorded as a semantic refusal when all 10 of its quotes were real source text.
`reference:cites_draft` is now its own category and its own outcome
(`reference_failure`), and it is scored as neither a refusal nor an approval.

## Finding 2 — every `passage_not_in_draft` in the revised adapter was a real passage

Across five probes, the revised adapter was rejected for a passage that was not
found — and in every case the passage was real content, written in a surface form a
strict comparison missed:

| As written by the model | Why it failed |
|---|---|
| `Ridgeline Equipment Services -- Fife, WA Inventory Analyst \| 08/2023 - Present` | emphasis dropped, two lines joined |
| `*Inventory Analyst \| 08/2023 - Present* - "Maintained parts …"` | two lines joined with ` - ` |
| `**"08/2023 - Present"** under Ridgeline Equipment Services` | a description, not a quotation |
| `Included Ridgeline Equipment Services as a separate role with dates` | a description, not a quotation |

The fix separates two questions that were previously one:

- **resolution** — "is this real document content?" tolerates presentation, because
  the reviewer's job is to identify a problem, not to transcribe the document;
- **application** — "what exact text do I replace?" stays exact, and refuses when
  the span is unclear.

A passage that merely *describes* the affected record resolves the first question and
fails the second. That is `unapplicable`, not an operational failure: the objection
is about the document and only the pointer is imprecise. Advisory records it and
delivers; enabled withholds; nothing is edited.

**Guards preserved.** An invented passage still fails both questions, and each has a
test.

## Finding 3 — a hazard the probe caught before it reached the matrix

One answer proposed:

```
correction: "(omit)" (Note: If future evidence confirms the timeframe …)
```

The omit marker was not the whole value, so the commentary was applied as
**replacement text**, overwriting a source-backed bullet with a note about the
bullet — the deletion hazard the blank-correction guard exists to prevent, arriving
through a different door.

`is_omit_marker` now recognises a marker with trailing commentary in every observed
quoting style, and `apply_revisions_v2` treats such a correction as an omission
(recorded, never applied).

## Finding 4 — source lists are comma-separated, not bracketed singly

The model wrote `sources: [candidate-main.txt, candidate-supplement.txt]`. The id
extractor required a trailing digit or a colon, so neither id resolved and the whole
bracketed string was kept as one "id" — rejecting a correct review over a comma.
Filenames are now matched as a first-class shape.

---

## What is still unknown

- **The mode-policy question is unanswered.** In every probe the Conservative and
  Imaginative cells reached the same *outcome* on `transferable_skill`. Both
  corrected the capability clause, which is defensible under either policy, so the
  probe cannot distinguish "the policy is inert" from "this case does not test it".
  The probe reports this as a finding about the contract rather than passing it.
- **No comparative result exists.** No matrix was run and none may be inferred from
  these probes: they reuse two cases and are excluded from all totals.

## Status

Production untouched: `decision_contract.py` is imported only by `experiment/` and
`tests/`; `git diff master -- source_correction.py correction_provider.py app.py
publication_gate.py` is empty. Completed the mechanical audit and freeze; the
bounded comparison has **not** run.
