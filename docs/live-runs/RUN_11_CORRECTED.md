# Run 11 — corrected report and evidence recovery

**Correlation:** `4c332713-0d6a-4b41-87fb-53a146e806fe` · **inner runId:** `dd19ce05-a7f1-4615-bfdf-7c7306d57b08`
**Outcome: failure of realistic delivery acceptance.** Withheld, `unresolved_claims`.
**Credits:** 935 → 902 (33). **Provider cost:** measured, see §6.

This document replaces the earlier run-11 report. Three claims in that report were wrong
and are withdrawn in §4.

---

# 1. Evidence recovery

## 1.1 Where evidence was found

The draft and every correction exchange were **not** captured. Capture was off. The
`analysis_jobs` row (`id=33`, `results` JSONB, 170,028 bytes) was recovered and contains
the pipeline's own record, including sanitized provider-call metrics.

## 1.2 Recovery status, item by item

| Requested | Status | Evidence |
|---|---|---|
| Original generated draft | **UNAVAILABLE** | `rewrittenResume` = `0` chars, `final_written_section_markdown` = `null`, `final_written_section` = `""`, `approved_export_payload` = `null`. The withheld draft was **deliberately not retained** — the publication control held. |
| Accepted initial-review response | **UNAVAILABLE** | Not persisted. The capture that would hold exchange bodies was disabled. |
| Flagged claim IDs | **UNAVAILABLE** | Same. Only the aggregate reason survived. |
| Repair request and replacement | **UNAVAILABLE** | Same. `calls` records the repair's cost and token counts (7,413 prompt / 67 completion) but no content. |
| Post-repair verdict | **PARTIALLY AVAILABLE** | `correction_status: withheld`, `correction_reason: unresolved_claims`, `reviews: 3`, `repairs: 1`. The *content* of the remaining objection is not. |
| Remaining objections, per claim | **UNAVAILABLE** | Only `review_reasons: ["unresolved_claims"]`. |
| Why the first review gave `callback_error` | **PARTIALLY AVAILABLE** — cause class derivable, message not | See §1.4. |

**No model response was reconstructed.** Where evidence is absent it is marked unavailable
rather than inferred. This is the specific reason the diagnosis below stops where it does.

## 1.3 What the stored record does establish

```json
{ "correction_status": "withheld", "correction_reason": "unresolved_claims",
  "correction_commentary": "The reviewer flagged unresolved claims after two correction
     attempts; independent review against the source is required.",
  "rewritten_resume": "", "final_written_section": "", "approved_export_payload": null,
  "publication_receipt": null, "requires_review": true, "verification_coverage": "none" }
```

Also: `processing_time_seconds: 66.48`, `warnings: []`, `jobs` count 36, `ats_score` 86.

## 1.4 The `callback_error` — cause class identified, message not recoverable

`correction_provider.py` re-raises only five exception types (`ProviderRateLimited`,
`ProviderUnavailable`, `ProviderRejectedRequest`, `ProviderCallTimeout`, `UnjudgedReview`,
`PrivacyReviewRequired`); **every other exception becomes `callback_error`**, and the
adapter logs no exception text by design. So the message is not recoverable.

What *is* determinable from control flow: `_Invoke`'s `callback_error` is a non-typed
exception raised inside the adapter body. Given the call took 10,230 ms (not a timeout) and
the retry then succeeded at 7,523 ms, the candidates are a `ValueError` from one of the
envelope checks (`invalid provider envelope` / `invalid provider choices` / `incomplete
provider response` / `missing provider content` / `invalid provider response`) or a
`_strict_json` failure on non-JSON content. **Which one cannot be determined.** Recorded as
**Protocol/validation error — specific variant unknown because evidence is missing**.

The important structural point: **this was not an engine decision.** It was an adapter
failure on attempt 1, retried successfully as request_index 3. It did not cause the
withholding.

## 1.5 Reported cost — two figures, both real, different scopes

| Metric | Value | Scope |
|---|---|---|
| `estimated_cost_usd` | **$0.0124597** | whole pipeline: 3 pipeline calls + 5 correction calls |
| correction calls only | **$0.0052190** | the 5 `source_correction` entries |

Earlier I reported only the second while calling it the run's cost. Both are stated here.

---

# 2. Investigate the actual failure

## 2.1 What is determinable

The final state is `withheld / unresolved_claims` with `reviews: 3, repairs: 1`. The engine's
control flow pinpoints where this comes from. In `_correct_body`, after a review:

```
if repairs == 2:                              return blocked("unresolved_claims")
if recovery["recovered"] and repairs >= 1:    return blocked("unresolved_claims")
```

`repairs` is 1 and `recovery["recovered"]` is True (the `must_judge` re-ask in §1.4 counted
as a review), so the **second** branch fired. Reading it exactly:

> "when recovery has been used and only one repair pass has happened, the engine stops here
> and names the unresolved claims."

So the withholding is a **deliberate, documented decision**, not a budget failure: after one
repair, the claims the reviewer still flagged were **not cleared**, and the engine declined
to spend a second repair pass because the recovery call had already consumed the allowance.

## 2.2 Objection-to-source table

Each row is what the engine *would* have objected to, with its class. **Because the flagged
claim IDs were not retained, attribution to specific passages is inference from the
mechanism, not observed evidence.** That limitation is why column 4 is mostly "unknown".

| # | Remaining objection | Source passage | Classification | Basis |
|---|---|---|---|---|
| O1 | The reviewer flagged ≥1 span as `unsupported` or `uncertain` on the post-patch review and it survived one repair | Not retained | **Unknown because evidence is missing** (class is `unsupported`/`uncertain` on a draft span) | Only the aggregate reason survived |
| O2 | The first review's answer was rejected at the adapter, not by the engine | N/A — no claim involved | **Protocol/validation error** | `callback_error` at 10,230 ms, retried and succeeded |
| O3 | One repair returned only 67 completion tokens against 7,413 prompt tokens | N/A | **Protocol/validation error (suspected)** — a repair answer that small may be a near no-op or a terse replacement | Token counts only; content not retained |
| O4 | Missing required facts | Not retained | **Not applicable this run** — no `source_facts_missing` reason was emitted | `review_reasons` contains only `unresolved_claims` |
| O5 | Wrong attribution | Not retained | **Not applicable this run** — no `attribution_conflict` reason | Same |

**No row can be assigned to Unsupported addition / Missing required fact / Wrong
attribution / Reviewer error for a *specific* passage**, because the drafts and claim IDs
were withheld from storage. Assigning them would be fabrication. The one class that *is*
established is the protocol/validation error in O2.

## 2.3 Smallest production fix

**None should be applied from this evidence.** No row supports a code change: O2 was already
recovered by the existing retry, O3 is a suspicion without content, O1 has no retained text.

The smallest *diagnostic* change that would make the next run diagnosable — and it is a
change to evidence, not behaviour — is a single structured log line: on the
`unresolved_claims` branch, record the count of surviving flagged claims and their verdict
values (no text). That is two numbers already in scope at that line. It makes the withhold
attributable without exposing content or touching the gate.

**Do not develop the lexical work list further.** It is frozen and would not have helped
here: it reports clauses, and this run never released a draft to report clauses about.

---

# 3. The 14.4-year narrative — separate investigation

## 3.1 The exact text

Survived in `analysis_jobs.results.jobs[0].llm_analysis.reasoning` and rendered on the
results page:

> "Strong match with extensive experience as a Journeyman Electrician and Maintenance
> Electrician. … **14.4 years of overall experience** aligns well with the seniority required
> for the role."

Second card: "Their **14 years** of experience in electrical systems…"

## 3.2 Where the number comes from — reproduced exactly

Not the résumé writer. It is the **job-search ATS scorer**, supplied by
`backend/services/candidateProfile.js`:

```
computeTotalExperience(experience, now):
  parseMonth(start/end) → months since year 0
  role at index 0 with no end date → end = now
  intervals sorted, overlaps merged, total → Math.round((months/12)*10)/10
```

Run on the frozen chronology (`03/2012–09/2015`, `10/2015–06/2019`, `07/2019–02/2023`,
`03/2023–present`) with `now = 2026-11-03`:

```
nowIdx = 24322   months = 173   total_years = 14.4
```

**14.4 reproduced to the digit.** The arithmetic is correct for what it measures: **elapsed
calendar span with overlaps merged**.

## 3.3 Why it disagrees with the supplied claim

| | Value | What it measures |
|---|---|---|
| Source claim | **12 years** | the candidate's own statement |
| `computeTotalExperience` | **14.4 years** | earliest start ÷ merged intervals through today |

`buildCandidateProfile` prefers a **stated** value when one is supplied
(`stated !== null ? {total_years: stated, basis: 'stated'} : computed`), but the multi-file
route passes `req.body?.experience_years`, and the form has no years field — so `stated` was
`null` and the computed span was used.

**Attribution, stated carefully:** this is a **job-search scoring input**, not a résumé
claim. It appeared on the same results page, which is why I previously wrote it up next to
the writer's output. It is a different subsystem with different rules, and the résumé writer
was never observed doing this.

Note also the two lengths disagree inside the same run: 14.4 (merged span) and 14 (rounded,
per a different card). Both derive from the same interval computation.

---

# 4. Withdrawn claims

| Earlier claim | Status | Why |
|---|---|---|
| "The run-10 download stall is explained by the buttons being disabled" | **WITHDRAWN** | I observed `disabled` on **run 11's withheld** page. Run 10 was **approved** — a different outcome and a different DOM state. Worse, I checked: the session *did* inspect run 10's button `disabled` flags at 16:25:44Z, but the tool result was recorded only as `{"success": true}` with no returned value, so run 10's button state was **never actually observed by anyone**. My explanation was inferred from run 11 and presented as an explanation of run 10. Run 10's cause remains **unknown**. |
| "`repairs: 1` is the first time the repair path has ever executed" | **WITHDRAWN — false** | `analysis_jobs id=30` (2026-11-03T05:27) recorded **`repairs: 2`**, and `id=33` is the first with exactly 1. Repair recovery ran earlier; my claim was wrong. |
| Cost "correction stage $0.005219" presented as the run's cost | **CORRECTED** | That is the correction stage only. The pipeline's own `estimated_cost_usd` is $0.0124597. |

Present in the earlier report and *also* withdrawn by implication: any suggestion that the
withholding validated the factual rejection. See §5.

---

# 5. Export protection — what was actually observed

| Observed | Not established |
|---|---|
| Publication gate returned `withheld` | that the rejected content was factually wrong |
| `publication_receipt` was `null` | that the reviewer's judgement was correct |
| `approved_export_payload` was `null` | that a human would reach the same verdict |
| Download buttons were `disabled` | that no export was possible by another route |
| Draft was **not persisted** (`rewritten_resume: ""`) | — |

**This is observed enforcement of a control, not evidence that the content deserved
rejection.** A gate that withholds a *correct* draft would produce an identical record. The
model-assisted review is explicitly "not proof" per its own commentary, and this run gives no
independent grounds to believe the withheld draft was actually defective.

That distinction is the whole point of the earlier acceptance criteria, and I stated it
correctly in the acceptance note and then blurred it in the report.

---

# 6. The missing capture — why, and what changes

## 6.1 Why it was omitted

Capture was **not** a precondition I checked. My approval request named "capture stays off
unless separately authorised", then ran with it off — so the run was executed under a
condition that guaranteed the evidence I would need afterwards could not exist. That was a
design error in the request, not an oversight in execution: I traded evidence for the
smallest authorisation and should have flagged the trade.

## 6.2 Consequence

Every item in §1.2 marked UNAVAILABLE is unavailable *because of that choice*. The run cannot
be diagnosed further without another instrumented run.

## 6.3 Rule for future paid runs

Before submitting, one of these must be true and stated:

1. `CORRECTION_LIVE_CAPTURE=1`, `assert_synthetic` passes, preflight returns `READY`, and
   `CORRECTION_LIVE_CAPTURE_DIR` is on the separate volume — verified in the container, not
   from a health check; **or**
2. an explicitly agreed alternative evidence mechanism (e.g. persisted
   `correction.phase_end` details with claim counts), named in the approval and verified
   before submit.

"If we can reconstruct it later" is not an option, as this run demonstrates.

---

# 7. What is needed to resolve the open question

Existing evidence is **insufficient** to say which clause the reviewer objected to, or
whether the objection was sound. The smallest instrumented experiment that would resolve it:

- **One run, same frozen pack, same revisions**, with capture enabled and verified in the
  container before submit, **or** the claim-count logging from §2.3.
- Then the retained exchange bodies give: draft text, flagged claim IDs, repair request,
  repair replacement, post-repair verdict, and the surviving objection — every item currently
  unavailable.

That is a request for a new allowance, not a retry, and it is not made here. **No retry and
no further paid analysis without explicit authorisation.**

---

# 8. Evidence preserved

| Artifact | Location | Bytes / hash |
|---|---|---|
| `analysis_jobs` row 33 | Postgres `job-search-db` | 170,028 bytes; inner `runId` `dd19ce05-a7f1-4615-bfdf-7c7306d57b08` |
| Preflight | `.live-run-2026-11-02/captures/run11/PREFLIGHT.md` | 1,856 |
| Ready-to-submit screenshot | `…/run11-ready-to-submit.png` | 206,028 / `4a2af383…` |
| Results screenshot | `…/run11-results.png` | 2,214,870 / `6cbe97af…` |
| Correction events | Imaginator logs 18:25:15–18:25:40Z | 13 events |

No capture exists in the container; nothing to purge. Capture flag `0`, remote files `0`.
