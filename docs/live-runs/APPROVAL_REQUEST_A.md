# APPROVAL REQUEST — one realistic two-file analysis

**Status: awaiting approval. Nothing has been submitted.** No credits spent, no
deployment, no code change.

---

## What is being requested

One analysis of the frozen realistic candidate pack through the normal website flow, to
observe current behaviour on the unchanged live build.

## Live revisions the run would execute against

Read from Render deploy metadata for each service, current as of this request:

| Service | Service ID | Live commit | Deploy completed |
|---|---|---|---|
| **orchestrator-frontend** | `srv-d4fb5f4hg0os738sasgg` | `9e040e3043beb2163a2214bfcef0abdc099a6e99` | 2026-11-03T08:08:12Z |
| **orchestrator-backend** | `srv-d4fb5o4hg0os738sb3a0` | `9e040e3043beb2163a2214bfcef0abdc099a6e99` | 2026-11-03T08:08:13Z |
| **imaginator-resume-cowriter** | `srv-d3nf73ur433s73bh9j00` | `7cc415ba13057de35a10d2870a7ab361a424a82b` | 2026-11-03T16:35:32Z |

Frontend and orchestrator share `9e040e3` because both deploy from `Landing_Login_page`.
The Imaginator is on `7cc415b`, equal to `origin/master` for `Imagination_Resume_Module`.
**These will be re-read immediately before submitting**; if any differs, the run stops and
the request is renewed.

## Inputs — frozen, hash-pinned

| File | Bytes | SHA-256 |
|---|---|---|
| `candidate-main.docx` | 37359 | `1e7a69331913f649b001ae74bf852495404a965e116cb823479a48dccff0eb86` |
| `candidate-supplement.docx` | 37079 | `199b62d6febe6d5d87e7571afa82e72730b8829235fc90cf49df364f9f0cf261` |
| `intended-job-description.txt` | 920 | `06577fab0ca934efd8872e83dc691d76ac918710c6933e8245d0fce0e2163409` |

Job title field: **Maintenance Electrician**. Files staged to
`C:\Users\jeffd\AppData\Local\Temp\cogito-realistic\` (Windows-accessible) and both
filenames verified in the page UI before submitting.

## Account-credit charge

- **Expected displayed charge: 33 credits** (the analysis cost at the time of writing).
- **Hard maximum: 33 credits.**
- If the UI displays a charge above 33, **stop before submitting** and re-request.

Current balance at time of writing: **935**.

## Provider-spend target and enforcement

| | |
|---|---|
| Target | **≤ $0.05** total provider spend for the correction stage |
| Basis | Run 10's two correction calls cost $0.002912976 measured from `usage.cost` in the preserved response bodies. $0.05 is ~17× that, covering retries and a larger prompt. |
| **Not covered by this figure** | The researcher, drafter and star-editor calls are outside the correction capture. **Whole-pipeline provider cost is unmeasured and is not capped by the $0.05** — that is a known limit, stated rather than implied. |
| Enforcement | I cannot throttle a provider call mid-flight. The enforceable limits are: (a) one analysis only — no automatic retry; (b) measurement after the fact from recorded `usage.cost`; (c) if the measured correction cost exceeds $0.05, no further provider calls are made without a new allowance. |

If a provider-dollar *ceiling* rather than a *target* is wanted, say so — the only
mechanism available is "one attempt, then stop", which is what this request already
proposes.

## What will not happen

- **No automatic retry.** If the analysis fails, I stop and diagnose from saved evidence.
- **No code deployment.** No push, no build, no env change. Capture stays off unless
  separately authorised; if capture is authorised, it is enabled before the run and
  disabled after.
- **No second analysis** under any circumstances without a new approval.
- **No gate change.** The publication gate and the lexical work list are untouched.

## Procedure after approval

1. Re-read the three live commits; abort on mismatch.
2. Verify both Windows-accessible files in the upload UI; capture the ready-to-submit state.
3. **Ordinary interactions first.** Record any click failure and its diagnostic before
   using a forced click, and label a later pipeline success separately.
4. Record the correlation ID and both `source_document` IDs.
5. Inspect the output against F01–F22.
6. Record the export request `headline`, the response `filename`, and the client download
   name.
7. Download and inspect **both** actual files, including PDF appearance (layout, page
   breaks, truncation).
8. Report the four dimensions separately: UI interaction · factual accuracy ·
   receipt/export integrity · visual quality.
9. Preserve evidence with hashes, disable capture, purge remote.

## Stop conditions

First of:

1. one analysis submitted;
2. displayed charge > 33 credits;
3. any of the three live commits differs from the values above;
4. an unexpected failure occurs — stop and diagnose; **do not purchase another attempt**;
5. the export is unreachable after one ordinary click plus one diagnosed forced click.

## Explicitly not claimed

This run **does not** test whether the run-10 semantic defect is fixed — the lexical work
list is not in the pipeline, and no writer or reviewer change has been deployed. It
measures current behaviour. Whatever is observed will be recorded as observation, and a
clean result will not be treated as clearing the historical defect.
