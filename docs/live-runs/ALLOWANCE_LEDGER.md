# Allowance ledger — synthetic live submissions, 2026-11-02

Authorisation: up to **3** synthetic website analyses and up to **$1** aggregate
provider spend, additional to prior consumed allowances. Unknown billing tracked
conservatively; stop at either limit.

## Analyses used: 3 of 3 — EXHAUSTED

| # | Correlation ID | Revision | Credits before → after | Outcome | Reason |
|---|---|---|---|---|---|
| 1 | `3ee23fd9-0767-4d8f-977f-50708c5afcbd` | `9057d966` | 1133 → 1100 | withheld | `source_facts_missing` |
| 2 | `709fb7fd-c48b-48a9-90ef-720f974b78c9` | `fb75917` | 1100 → 1067 | withheld | `source_facts_missing` |
| 3 | `0a267e04-affd-4cee-b944-26c574703039` | `b855087` | 1067 → 1034 | withheld | `source_facts_missing` |

Credits consumed: 99 (3 × 33). Balance after: **1034**.

No 4th submission was made. The allowance is spent; the live delivery criteria did
**not** pass.

## Provider spend: $0.006178 recorded, well under the $1 cap

Both calls per run are `qwen/qwen3-coder` (DeepInfra), billed per token and recorded
in each capture's `usage.cost`:

| Run | Reviewer + inventory cost | Tokens |
|---|---|---|
| run 5 | $0.002203 | 5,496 |
| run 6 | $0.001999 | 4,979 |
| run 7 | $0.001977 | 4,966 |
| **total** | **$0.006178** | 15,441 |

### Billing uncertainty

* The figures above are the **provider's own reported `usage.cost`** for the two
  correction calls each run recorded. They exclude the researcher/drafter/star-editor
  calls of the analysis pipeline itself, whose usage is not in the correction capture —
  so **total per-analysis provider cost is higher than the three figures shown**.
* The $1 cap governs aggregate provider spend. The measured correction-call portion is
  0.6% of it. Even if the three full analyses cost an order of magnitude more than
  their correction calls, the total stays far below $1. This is an estimate from
  recorded evidence, not a metered total.
* Credit spend (99) is exact: it is the platform balance delta across the three runs.

## Evidence retained

Local, protected, read-only — container captures were purged after retrieval:

* `.live-run-2026-11-02/captures/run5/3ee23fd9-….json` — sha256 `8972d371…`
* `.live-run-2026-11-02/captures/run6/709fb7fd-….json` — sha256 `269caf5e…`
* `.live-run-2026-11-02/captures/run7/0a267e04-….json` — sha256 (see `sha256sum`)
* `.live-run-2026-11-02/retained/delivery-run-4-withheld.json` — sha256
  `92e03d4bbfde82089bcb428ff563d274a9e215cbabe496428f6565cdff891230`, unchanged.

## Capture state

`CORRECTION_LIVE_CAPTURE=0` verified in the container at revision `b855087`.
`CORRECTION_PROVIDER_DISCLOSURE=1` unchanged (policy not broadened).
Container `/app/data/live-capture/` emptied.

## What progressed and what did not

Each run's draft was materially better than the last, and every improvement came
from a specific fix verified offline first:

* run 5 → run 6: no invented Projects section, no AI tooling, no credential issuer or
  year, no GPA/coursework, no fabricated metric. (Commit `fb75917`.)
* run 6 → run 7: the candidate's own Professional Summary survived into the output
  for the first time, with its facts intact — 8+ years, journeyman, commercial and
  industrial, construction/facilities maintenance/power distribution. Skills stopped
  claiming "crew leadership" and "code compliance". (Commit `b855087`.)

Neither run satisfied the publication gate, so **no live delivery passed**. The
remaining defects are correlated and described in `REMAINING_DEFECTS.md` next to
this file; they are fixable but were not fixable within the 3-analysis allowance.
