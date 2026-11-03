# Run 11 — realistic fixture on the unchanged live build

**Correlation ID:** `4c332713-0d6a-4b41-87fb-53a146e806fe`
**Date:** 2026-11-03, analysis completed 10:26:08 local
**Outcome:** **withheld** — `unresolved_claims`
**Credits:** 935 → 902 (**33**, exactly the approved maximum; UI: "Credits used: 33 / Deducted from your paid balance")
**Provider cost (correction stage):** **$0.005219** across 5 calls

## Live revisions executed

| Service | Commit |
|---|---|
| orchestrator-frontend | `9e040e3043beb2163a2214bfcef0abdc099a6e99` |
| orchestrator-backend | `9e040e3043beb2163a2214bfcef0abdc099a6e99` |
| imaginator-resume-cowriter | `7cc415ba13057de35a10d2870a7ab361a424a82b` |

Re-read immediately before submitting; all three matched the approved values.

## Inputs — frozen and hash-verified after copy

| File | SHA-256 |
|---|---|
| `candidate-main.docx` | `1e7a69331913f649b001ae74bf852495404a965e116cb823479a48dccff0eb86` |
| `candidate-supplement.docx` | `199b62d6febe6d5d87e7571afa82e72730b8829235fc90cf49df364f9f0cf261` |
| `intended-job-description.txt` | `06577fab0ca934efd8872e83dc691d76ac918710c6933e8245d0fce0e2163409` |

---

# Dimension 1 — UI interaction

| Step | Method | Result |
|---|---|---|
| Cookie banner | its own Decline button | dismissed |
| Upload both files | `setInputFiles` (supported mechanism), Windows paths | **2 files uploaded successfully** |
| Filenames in UI | read back from the file input and the page | `candidate-main.docx` (0.04 MB), `candidate-supplement.docx` (0.04 MB) — **both visible, both checked** |
| Continue to description | **ordinary click, no force** | **succeeded first time** |
| Fill title + ad | `fill` | title `Maintenance Electrician`, ad 919 chars |
| Analyze Resume | **ordinary click, no force** | **succeeded**, `POST /api/analysis/multi-file?stream=1` |
| Download PDF (LaTeX) | **ordinary click, no force** | **did not fire** — button is `disabled` |

**Every interaction this run was ordinary.** No forced clicks, no JavaScript `.click()`, no
attribute removal, no state manipulation.

## The run-10 "download stall" is now explained

The download buttons are **`disabled`** when the result is withheld:

```
Download PDF (LaTeX)   disabled: true
Download DOCX          disabled: true
Export TXT             disabled: false
```

`Export TXT` stays enabled because it exports the source text, not an approved rewrite.

This is almost certainly the run-10 mechanism too. In run 10 the listener watched
`/resume/export` and reported no request; the run was approved there, but the page had
been reloaded between attempts, losing the publication state. **A disabled control produces
exactly the symptom that was misdiagnosed as an overlay or hit-target problem.** Recorded
as the probable cause, not asserted as proven for run 10 — I did not re-observe run 10's
DOM.

## Errors encountered

One, and it is not a product defect: `TimeoutError: locator.click: Timeout 8000ms exceeded
… locator resolved to <button disabled …>` — Playwright correctly refused to click a
disabled button.

---

# Dimension 2 — factual accuracy

**Not assessable on the rewrite, because no rewrite was released.** The page states:

> "The rewritten resume was withheld pending review. Original source text is not a
> released AI rewrite."

There is no approved markdown to check against F01–F22. What can be assessed:

| Check | Result |
|---|---|
| Rewrite released | **no** — withheld |
| Source facts retained in output | **not applicable** — nothing released |
| Motor controls in output | **not applicable** — nothing released |

## Two factual findings that ARE observable, outside the rewrite

**1. "12 years" was recalculated to "14.4 years".** This is clarification 1's failure mode,
observed directly. The job-match analysis says:

> "14.4 years of overall experience aligns well with the seniority required for the role."

A second match card says "Their 14 years of experience in electrical systems…". The source
states "12 years". The chronology totals ~14.5 elapsed / ~11.0 electrical, so 14.4 is the
elapsed-span arithmetic. **The supplied claim was replaced by a computed one**, exactly what
clarification 1 prohibits. Note this appears in **job-match analysis**, not in a résumé, so
it may be a different subsystem's behaviour than the writer's — that distinction is not
resolvable from this run alone.

**2. ATS gap analysis correctly identified the absent motor-controls skill:**

> "📚 Skills to Develop: AC/DC motor installation, generator installation"
> "Minor gaps in specific areas such as AC/DC motor installation and repair"

Motor controls were requested by the ad, absent from the record, and surfaced as a **gap**
rather than asserted as a skill. That matches the corrected G1 expectation.

---

# Dimension 3 — receipt / export integrity

| Check | Result |
|---|---|
| Publication gate | **blocked** — reason `unresolved_claims` |
| Review notice text | "This draft needs review before you can use it" |
| Verification coverage disclosed | "none" |
| `/resume/export` request issued | **none** (button disabled — correct behaviour) |
| PDF/DOCX produced | **none** |
| Client-side refusal path | taken; the UI refused locally rather than sending an export the server would reject |

The gate refused to release an unverified draft and the UI disabled export accordingly.
**This is the gate working.** It is the first observed run where the publication gate
blocked on grounds other than `source_facts_missing`, and the first where the withholding
reason is a claims problem rather than a missing-fact problem.

## Correction pipeline detail (from Imaginator logs)

```
source_inventory       ok         1,056 ms
initial_review    callback_error 10,230 ms   ← malformed/timed-out answer
initial_review         ok          7,523 ms   ← retried, succeeded
repair                 ok          1,012 ms   ← FIRST TIME REPAIR EVER RAN
post_patch_review      ok          4,799 ms
correction.end   withheld  reason=unresolved_claims  reviews=3  repairs=1
```

**`repairs: 1` is new.** Every prior run recorded `repair_calls = 0`; the repair path has
been unexercised in this project until now. It ran, and the post-patch review still did not
clear the claims, so the result was withheld. Real-model repair recovery is therefore
**exercised but not successful** on this input — a materially different finding from
"unexercised".

# Dimension 4 — visual quality

**Not assessable.** No PDF or DOCX was produced, so there is nothing to inspect for layout,
page breaks, or truncation. The screenshot of the results page is preserved.

---

# What this run established

1. **The honest two-file flow works with ordinary interaction end to end** — upload,
   normal click to advance, normal click to analyze, 33 credits, results rendered.
2. **The publication gate blocks an unverified rewrite on a realistic, richer fixture.**
   Run 10 approved a similar (but simpler) two-document input; this one was withheld with
   `unresolved_claims`. The realistic pack is harder, and the gate caught it.
3. **Repair recovery ran for the first time** (`repairs: 1`) and still left unresolved
   claims. Repair is exercised, not proven.
4. **A withheld result disables the download buttons.** That is correct behaviour and it is
   the most likely explanation for run 10's misdiagnosed "no export request" — recorded as
   probable, not proven.
5. **"12 years" is not preserved where it is recalculated.** In the job-match narrative it
   became 14.4 years. Whether the writer does the same cannot be determined, because no
   rewrite was released.
6. **Gap analysis correctly reports an absent ad-requested skill** instead of inventing it.

# What did not happen

- No defect prediction was made before the run, and none is made now. The run withdrew; it
  did not approve with appended clauses. **That does not clear run 10's defect** — run 10's
  capture remains evidence that the engine approved six appended outcome clauses on an
  easier input. This run's withholding is consistent with the defect being handled better
  here, and equally consistent with a different failure (a malformed review answer
  requiring a retry) changing the outcome. **One withheld run is not evidence the defect is
  fixed.**
- No PDF/DOCX inspection, because none was produced.
- No retry, no second analysis, no deployment.

# Evidence preserved

| File | Bytes | SHA-256 |
|---|---|---|
| `PREFLIGHT.md` | 1856 | — |
| `run11-ready-to-submit.png` | 206028 | `4a2af383cbfa0b006dc99a08379a0cd734607fdaccf39a5af1ce0086252a949d` |
| `run11-results.png` | 2214870 | `6cbe97aff79f7556d54c13585d0d6f7594ac84b492b84d5217c1c8cff9a6ebd6` |

Correction-stage provider cost recorded from the server's own `cost_usd` fields in
`correction_event` logs: `0.000419 + 0.001230 + 0.001265 + 0.001152 + 0.001153 =
$0.005219`. Whole-pipeline cost remains unmeasured (researcher, drafter and star-editor
calls are not in the correction events).

Capture was **not enabled** for this run, so no container capture exists to purge.

# Unexpected failure?

**No unexpected failure.** The withholding is a designed outcome, the export refusal is
correct, and no retry was made. Nothing requires another attempt.
