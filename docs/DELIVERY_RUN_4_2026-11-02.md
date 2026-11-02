# Delivery run 4 — result: WITHHELD

Single synthetic single-file analysis submitted through the website. This is the first
run since the retry-resilience work that was executed end to end, and it is the first
one whose evidence is a full live capture rather than a log excerpt.

## Run identity

| Field | Value |
|---|---|
| Correlation ID | `8e6d3d72-1694-4df2-867d-4687ede388b6` |
| Submitted at | 2026-11-02T21:38Z (analysis completed 13:40:37 local) |
| Entry | website, `POST /api/analysis/multi-file?stream=1` (SSE) |
| File | `synthetic-single.docx` (36.3 KB), 1 file |
| Job title | Senior Electrician |
| Job ad | 426 chars |
| Account | userId 22, `jeff.d.calderon@gmail.com` |
| Credits | 1166 → 1133 (33 charged, `Credits used: 33`) |
| Live commit | `f77736164e5c7e3b831407ed0035cc92336e559f` |
| Deploy | `dep-dbkg4qsipgvc73atd9m0` (capture enabled) |
| Capture | `/app/data/live-capture/8e6d3d72-1694-4df2-867d-4687ede388b6.json` |
| Retained as | `.live-run-2026-11-02/retained/delivery-run-4-withheld.json` |
| SHA-256 | `92e03d4bbfde82089bcb428ff563d274a9e215cbabe496428f6565cdff891230` |

## What the customer saw

- Analysis completed; 5 job matches rendered (top match 86%).
- Résumé section: **"The rewritten resume was withheld pending review. Original source
  text is not a released AI rewrite."**
- Status line: "This draft needs review before you can use it"; reasons:
  **`source_facts_missing`**; coverage: none.
- **Download PDF (LaTeX): disabled. Download DOCX: disabled.** Export TXT enabled but
  carries no released rewrite. No publication receipt was issued.

The withhold is therefore a *delivery* failure against the milestone: 33 credits were
spent and the customer received no usable résumé.

## Provenance of the decision

Both provider exchanges returned **HTTP 200**, so this is not a transport failure, not a
429, and not a timeout. It is the correction engine's completeness rule:

```
source_correction.py:844   if "missing" in incomplete:
source_correction.py:845       return blocked("source_facts_missing")
```

The reviewer's first and only call returned:

- `represented`: 10 of 22 required segments
- `missing`: **12** — indices 0, 1, 2, 5, 6, 7, 8, 10, 11, 12, 14, 20

Those 12 segments are the summary block (0–2), the two job title/date lines (5, 10), six
bullets carrying the quantitative facts (6, 7, 8, 11, 12), the college name (14) and the
first skills line (20).

Offline replay (`live_capture.replay`, no network) reproduces the identical result:

```
STATUS  : withheld
REASON  : source_facts_missing
REVIEWS : 1  REPAIRS: 0
EXCHANGES USED: 2 of 2
```

## Demonstrated cause

The reviewer's completeness verdict is defensible on its own terms — the draft really
does not carry those lines verbatim, and it adds content found nowhere in the source
(`Projects`, `Integrated AI-driven diagnostic tools`, `12% reduction in peak demand`,
`GPA: 3.78`, `Languages: Python, SQL, VBA`, `Operating Systems: Windows, Linux`). But the
mechanism has a structural defect that matters more than this particular draft:

**Completeness is decided before repair, and repair can never fix it.**

The code decides `missing` → block immediately, and the comment states the intent:
"no edit may introduce a fact, so a draft that omits one is never repairable." That is
true for *inserting* a fact. It is not true for the common case here: the facts are
present in the draft in *paraphrased* form, and the reviewer marks the source segment
`missing` because no single supported line restates it with its real employer, dates and
numbers. Repair rewrites flagged spans; it cannot restore coverage, so every run whose
draft compresses or paraphrases source material can only ever end withheld.

The decisive detail is that the uncleaned **writer draft is what gets judged**. Two
independent source-fidelity defects reached the reviewer and were never repaired:

1. `*Professional | 03/2020 â Present*` — the job titles were replaced with the word
   "Professional"; the source titles (`Journeyman Electrician`, `Electrical Apprentice`)
   are gone.
2. `â` mojibake throughout, and `fullâtime` / `highâavailability` — a UTF-8 decoding
   fault upstream of the draft.

The reviewer correctly reported the title lines (5, 10) as missing because the titles are
genuinely absent. The engine then withheld — correctly under its own rule — but the run
was already unrecoverable: no repair can re-insert a title, and the publication gate would
have caught the invented Projects/AI/metrics anyway.

## What this establishes and what it does not

Established:

- The delivery path executes end to end and the withhold is honest, not a false approval.
- The invented-metric protection works in production: the draft contains ~6 fabricated
  metrics and none reached the customer.
- The capture mechanism works for a 200-status run and is sufficient for offline replay.

Not established:

- A single-file delivery produces a usable approved résumé. It does not; that is the
  milestone and it is still unmet.
- Whether the mojibake or the title loss is deterministic across drafts, or specific to
  this document's encoding.

## State left behind

- `CORRECTION_LIVE_CAPTURE=0` (verified in-container after redeploy
  `dep-dbkg6q8ae00c73cli7u0`).
- Container capture directory emptied; the only copy is the retained local file above.
- No second paid submission was made.
