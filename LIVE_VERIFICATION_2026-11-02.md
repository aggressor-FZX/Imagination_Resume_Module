# Live verification record — 2026-11-02

Authorized by the owner: enable `CORRECTION_PROVIDER_DISCLOSURE=1`, deploy to
`master`, and run up to two paid synthetic analyses with a **$2 total target
(not an enforced cap)**. Evidence here is synthetic-only; no real candidate data
was uploaded.

## Deployed and verified

| Service | ID | Live commit | Status |
|---|---|---|---|
| imaginator-resume-cowriter | `srv-d3nf73ur433s73bh9j00` | `429a6dd29690` | live |
| orchestrator-backend | `srv-d4fb5o4hg0os738sb3a0` | `dbbdf9077141` | live |
| orchestrator-frontend | `srv-d4fb5f4hg0os738sasgg` | `dbbdf9077141` | live |

`GET /health` on imaginator returns 200 `{"status":"healthy","environment":"production"}`.
Unauthenticated `POST /analyze` returns 403 (auth intact).

**Disclosure flag:** `CORRECTION_PROVIDER_DISCLOSURE=1` added to
imaginator-resume-cowriter while preserving all 17 pre-existing env vars (PUT of
18 total, verified by re-read). Applied by an explicit deploy after the write,
so the running container has it.

**Rollback revisions** (pre-release, all currently deactivated):

| Service | Revision | Deploy ID |
|---|---|---|
| imaginator | `271f25b39816b9db8cd9e0d819f6e00d2a2de2aa` | `dep-dbjuhneq1p3s73b720vg` |
| orchestrator-backend | `3e4eff918d4651681390508fc7d3ea1ed59f4b3f` | `dep-dbjuhqvavr4c73charr0` |
| orchestrator-frontend | `3e4eff918d4651681390508fc7d3ea1ed59f4b3f` | `dep-dbjuhqvavr4c73charfg` |

## Release blocker found and fixed before deploying

The "pre-existing" frontend publication failure was **not** cosmetic. It was a
real blocker for exactly this workflow:

- Failing assertion: `AnalysisPipeline.publication.test.tsx` → export bridge call
  expected `status === 200`, received **409**.
- Cause: `buildExportPayload` wrapped `approved_export_payload` in profile
  defaults including `name: 'Resume'`. The v4 receipt binds the *complete*
  renderer input, and the server signs an empty identity header, so the added
  name changed the fingerprint → `receipt_render_content_mismatch` on **every**
  approved export, live included.
- Fix (`dbbdf90`): when a receipt-bound object exists, send exactly it — no
  invented identity. Verified: frontend publication suite **37/37 pass** (was
  36/37), and a real export through the bridge returned **200** with a 106 KB
  PDF and 37 KB DOCX.

## Live runs

| # | Files | Target role | Credits | Outcome |
|---|---|---|---|---|
| 1 | 1 synthetic DOCX (`synthetic-electrician.docx`, 910 chars) | Journeyman Electrician | 1529 → 1496 (33) | **FAIL** — withheld, reason `timeout` |
| 2 | 2 synthetic DOCX (+`synthetic-admin.docx`, 983 chars) | Administrative Coordinator | 1496 → 1463 (33) | **FAIL** — withheld, reason `timeout` |

Both runs displayed the review notice ("This draft needs review before you can
use it") with PDF/DOCX disabled and TXT export enabled. No receipt was issued,
so no commentary or download could be inspected — the required end-state was
**not** reached.

**What worked live:** upload, document extraction, the full 3-stage pipeline
(both `/analyze` calls returned **200 OK**; run 1 pipeline ~16s), job matching
(10 matches), ATS display, the publication gate correctly withholding, and the
new commentary UI path rendering the review notice.

## Diagnosis of the two failures

The pipeline never failed. `timeout` came from source correction:

1. Run 1 used a 30s **total** correction deadline while each provider call was
   itself allowed 30s → the deadline expired.
2. Run 2 raised the total to 120s and still failed. Investigation showed
   `aiohttp` raises `TimeoutError` when a **single request** exceeds its own
   read timeout; `correct()` reported that as the generic `timeout`, so raising
   the total deadline could not help.

Fixes deployed progressively (each committed and live):

- `2cb6ab2` — whole-operation deadline 30s → 120s (ceiling 150s).
- `9b5671d` — deadline expiry now names the phase
  (`timeout_source_inventory` / `_initial_review` / `_repair` /
  `_post_patch_review`).
- `429a6dd` — per-call provider timeout 30s → **60s**, raised as a distinct
  `ProviderCallTimeout` and reported as
  `provider_call_timeout_<phase>`, no longer conflated with the deadline.

The last fix is deployed but **not yet exercised live** — doing so requires a
third paid analysis, which was not authorized.

## Costs

| Item | Value | Scope |
|---|---|---|
| Account credits | 66 credits (33 × 2) | Account ledger; exact |
| Provider spend | **NOT MEASURED** | Not obtainable after the fact |

Provider cost is not quantifiable. Imaginator's per-run cost line is emitted in
a `run_metrics` payload that is not retained in the accessible log window, and
the correction calls' usage was recorded only inside each withheld response
(which the UI does not surface). The observed evidence is therefore limited to
the credit ledger. **The $2 target was not an enforced cap**; actual spend is
unknown but bounded by two analyses' provider usage.

Correlation IDs from the runs were present only in log lines that have since
aged out of the retention window; no ID was captured into this record, so none is
asserted here. That is a gap in this evidence, not a claimed success.

## Honest status

| Criterion | Result |
|---|---|
| Release deployed, live commits verified | **PASS** |
| Disclosure policy active in production | **PASS** (flag verified set; effect inferred from code path, not from a completed correction) |
| Export blocker fixed (unit + bridge) | **PASS** |
| Single-file live analysis end-to-end | **FAIL** — withheld `timeout` |
| Multi-file live analysis end-to-end | **FAIL** — withheld `timeout` |
| Displayed rewrite / commentary / receipt / PDF / DOCX inspected | **NOT REACHED** |
| Publication checks weakened to force a pass | **NOT DONE** (gate left intact) |
| Provider cost measured | **NOT MEASURED** |

## Smallest next step

One further paid single-file analysis against `429a6dd`. If the per-call timeout
was the cause, it either approves (and the receipt/commentary/PDF/DOCX can be
inspected) or returns a phase-specific
`provider_call_timeout_*` / `timeout_*` reason that identifies the exact failing
call. No code change is pending on that outcome.
