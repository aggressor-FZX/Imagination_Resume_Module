# Correction timeout diagnosis — 2026-11-02 (non-billable)

Scope: the two earlier synthetic live analyses that were withheld with a generic
`timeout`. No provider call, analysis or deploy was made for this diagnosis.
All times are UTC. Render logs were read with explicit `startTime`/`endTime`
windows and forward pagination, not "latest N" queries.

## 1. Result

- Both runs returned `timeout` **30.0 s and 30.8 s after the Imaginator pipeline
  finished**. The 30 s figure matches a 30 s limit, not the 120 s total deadline
  that run 2 had.
- **Run 1:** the 30 s whole-operation deadline (commit `b486af0`).
- **Run 2:** one provider request hit its own 30 s timeout (commit `2cb6ab2`).
  The 120 s deadline could not have expired after 30 s.
- **No log identifies the failing phase** (inventory, review, repair or
  post-patch review). The earlier code logged no correction phases. The
  "aiohttp per-call" statement in the live report is now supported by timing for
  run 2, but not by a logged phase or timeout category.
- Implemented here, not deployed: budget propagation from the pipeline's real
  elapsed time, phase-attributed events with no document text, and 25 new offline
  tests. See section 8.
- Increasing timeouts is **not** the fix. Section 5 shows the output allowance is
  the structural limit for large inputs, and per-call duration is the observed
  failure.

## 2. Log evidence

### Deploys (`srv-d3nf73ur433s73bh9j00`, Imaginator)

| Deploy | Commit | Live window (UTC) | Deadline / per-call |
|---|---|---|---|
| `dep-dbk46aajnfac739htd10` | `b486af0` | 08:01:56 → 08:07:54 | 30 s / 30 s |
| `dep-dbk499rm8hqs738djpc0` | `2cb6ab2` | 08:07:54 → 08:14:33 | 120 s / 30 s |
| `dep-dbk4ce3bc2fs73e3jba0` | `9b5671d` | 08:14:33 → 08:17:15 | 120 s / 30 s |
| `dep-dbk4dnavcj2c73fkdts0` | `429a6dd` | 08:17:15 → 08:19:48 | 120 s / 60 s |
| `dep-dbk4ettg1s2s738jgt6g` | `10b20d8` (docs) | 08:19:48 → live | 120 s / 60 s |

### Run 1 (orchestrator request 08:03:56, correlation `14cff81e-0d74-4025-9e5d-76008f3b7dbb`)

| UTC | Event | Source |
|---|---|---|
| 08:03:56.559 | estimate 33 credits | orchestrator |
| 08:03:56.669 | `[MULTI_FILE] Processing 1/1 files` (one file after filtering) | orchestrator |
| 08:04:07.800 | `[MULTI_FILE_IMAGINATION] Calling`, combined evidence 910 chars | orchestrator |
| 08:04:07.928 | Imaginator receives analyze request | Imaginator |
| 08:04:24.242 | `analyze.pipeline.fast_track` (pipeline 16.3 s) | Imaginator |
| 08:04:54.273 | `[MULTI_FILE_IMAGINATION] Success` (30.0 s after fast_track) | orchestrator |
| 08:05:40.820 | `credit_request_outcome` 200 for the same ID | orchestrator |

Run 1 was processed by the multi-file endpoint with one file, not a single-file
endpoint as earlier notes said.

### Run 2 (orchestrator request 08:08:37, correlation `235c04c6-7113-426b-8253-f680be6f547a`)

| UTC | Event | Source |
|---|---|---|
| 08:08:37.733 | `Processing 2/2 files` | orchestrator |
| 08:08:43.301 | `[MULTI_FILE_IMAGINATION] Calling`, combined evidence 1900 chars | orchestrator |
| 08:08:43.363 | Imaginator receives analyze request | Imaginator |
| 08:09:20.045 | `analyze.pipeline.fast_track` (pipeline 36.7 s) | Imaginator |
| 08:09:50.824 | `[MULTI_FILE_IMAGINATION] Success` (30.8 s after fast_track) | orchestrator |
| 08:10:11.073 | `credit_request_outcome` 200 for the same ID | orchestrator |

### Attribution

- **Run 1.** The 30.0 s return matches the 30 s whole-operation deadline. The
  deadline timer was armed before the first provider request, so it fires before
  that request's own 30 s timeout. The old code reported both as `timeout`.
  The phase is not identified.
- **Run 2.** The return was 30.8 s after `fast_track`. Under `2cb6ab2` a 120 s
  deadline cannot have expired at 30 s. The only 30 s limit in that code is the
  per-request timeout in `correction_provider.py`, which raised
  `TimeoutError` and was reported as the generic `timeout`. So one provider
  request exceeded 30 s. Whether it was inventory or review is not logged.
- **Limits of this attribution.** The orchestrator's `Success` line is its
  receipt time, so the Imaginator's own return time is only known to within about
  a second. The `200 OK` access-log lines are written when a request starts, so
  they do not prove the analysis finished. The `x-correlation-id` sent to
  Imaginator is not printed by Imaginator, so the join is by timing and the
  orchestrator's own ID.
- **Other lines seen.** The orchestrator logged `Imaginator: Logged 1 calls,
  total $0.001173` at 08:04:07.979, before the analyze call returned, so it is not
  the analyze cost. An unrelated `POST /analyze 403` from another caller appeared
  at 08:02:11. Neither is used in the diagnosis.

### Unavailable

- Phase identity, provider timing and token counts for the two timed-out calls.
  The provider returns no usage for a request that never completes.
- OpenRouter generation IDs, so provider-side latency cannot be recovered.
- Whether the provider calls were cancelled upstream. Client-side cancellation
  does not stop the remote generation.
- Exact dollar spend. The correction spend is bounded in section 6, but the
  two runs' actual correction spend is unmeasured.
- Any correction-phase log line. The pre-fix code had none.

## 3. Why the old fixed deadline could exceed the upstream timeout

The orchestrator's client timeout for the JSON and single-file calls is 180 s,
with up to three attempts on any error (`mlAuthMiddleware.js`,
`makeAuthenticatedRequest`). The multi-file SSE branch that both runs used sets
`timeout: 300000` and does not retry (`server.cjs`, around line 4470).

The Imaginator's correction deadline was a fixed 120 s, regardless of how long
the pipeline had run. The pipeline itself took 16.3 s and 36.7 s in these runs,
and 63.7 s in an earlier pre-correction run (01:38 UTC, commit `271f25b`, received
01:38:16.9 and staged 01:39:20.6). At 63.7 s:

    63.7 s pipeline + 120 s correction + 20 s reserve = 203.7 s > 180 s

That is a risk on the 180 s paths (retry duplicates the paid pipeline), not an
observed failure. The new code removes it.

## 4. Time budget now

```
upstream timeout (orchestrator)  180 s
- response reserve                20 s
= time available to the handler  160 s
- pipeline elapsed (measured)
= correction deadline            min(120 s, 160 s - pipeline elapsed)
```

`correction_budget_seconds()` returns `None` when less than 30 s would remain.
Then the handler withholds with `pipeline_budget_exhausted` and sends no
provider request.

| Pipeline elapsed | Correction deadline | Result |
|---|---|---|
| 16 s (run 1) | 120 s | runs in full |
| 37 s (run 2) | 120 s | runs in full |
| 63.7 s (earlier run) | 96 s | previously 120 s, which overran |
| 100 s | 60 s | shorter deadline |
| 130 s | 30 s | last permitted start |
| 131 s or more | none | `pipeline_budget_exhausted`, no call |

**Termination.** When a budget is returned,
`pipeline + budget + reserve = 180 s` exactly. The engine returns at its
deadline. An in-flight request is cancelled and not awaited, so the response
does not wait for the remote call. Offline test: with the upstream scaled to
2.0 s and a 0.3 s pipeline, a hung provider call returns in under 2.0 s with
`timeout_source_inventory`, and the event budget satisfies
`pipeline_ms + budget_ms + reserve_ms <= upstream_timeout_ms`.

**Per-call versus deadline.** A single call can still take up to 60 s. Two calls
of 60 s would exceed a 120 s deadline, so once one call is slow the total
deadline is the binding limit. That is deliberate and is reported as
`timeout_<phase>`, not `provider_call_timeout_<phase>`.

## 5. Why the workload needs its output allowance

Every provider call has `max_tokens = 12000`.

- **Inventory** must return one span for each non-blank source span, each with an
  `id`, offsets, a copied `quote` and a `classification`. Output is roughly the
  source text again plus about 110 characters of structure per span.
- **Review** must partition each draft line into claim spans, with a citation
  quoting the source for each supported claim. Output is roughly the draft text
  again plus citation quotes and structure.

Measured on the inventory with the fixture's minimal shape (lower bound; a real
model is at least this long):

| Source chars | Inventory output chars | Estimated tokens (4 to 3 chars/token) | Fits 12,000? |
|---|---|---|---|
| 641 | 1,250 | 310 to 420 | yes |
| 1,900 (run 2 size) | 3,734 | 930 to 1,250 | yes |
| 16,000 | 31,620 | 7,900 to 10,540 | yes |
| 32,000 | 63,570 | 15,890 to 21,190 | **no** |
| 48,000 (accepted maximum) | 95,520 | 23,880 to 31,840 | **no** |

**Conclusion.** The allowance covers the live inputs (about 1 to 2 KB) with room
to spare, so output overflow does not explain the live timeouts. It does mean
the accepted maximum source of 48,000 chars cannot be corrected. The provider
would truncate (`finish_reason: length`), and the run would be withheld as
an incomplete response. The 32,000-character draft limit has the same problem on
review.

The live timeouts are better explained by call duration relative to a 30 s
limit. Provider throughput is not in the logs, so this is unmeasured. The new
events record `completion_tokens` and `elapsed_ms` for each call, which will
measure it.

**Options, in order of preference:**

1. Reject oversized sources before any call (a size gate, withheld with no spend),
   and lower the accepted source size to what one allowance can hold. Needs
   calibrated estimates from the new events.
2. Split inventory by source document or section, so each call's output stays
   well within the allowance. Changes the contract and the coverage proof, so it
   needs review.
3. Derive citation and inventory quotes locally from the offsets, which the
   engine already validates. This roughly halves output. It removes the model's
   copied quote as proof that it read the span, so it changes the validation
   semantics and needs approval.

Raising `max_tokens` increases call duration. It is not recommended as the fix.

## 6. Spend bounds

Per call at the request cap: input of about 24k tokens (96 KB at roughly 4 bytes
per token) at `$0.00015/1K` is about `$0.0036`; output of 12,000 tokens at
`$0.0006/1K` is `$0.0072`. Six calls are bounded at about
**$0.065**. Typical runs with 4 calls and small inputs cost far less. The
Imaginator pipeline is priced separately. Credits were 33 per analysis, and
account credits are not dollars.

## 7. Phase events (implemented)

`correction_events.py` sanitizes every event before it is logged. Unknown keys
are dropped. Strings must be members of fixed vocabularies or short
identifiers. Free text, document content, prompts and model output cannot be
logged even if passed in.

| Event | Fields |
|---|---|
| `correction.budget` | `pipeline_elapsed_ms`, `budget_ms`, `upstream_timeout_ms`, `reserve_ms` |
| `correction.phase_start` | `phase`, `request_index`, `budget_ms` (remaining before the call), `elapsed_ms`, `deadline_ms` |
| `correction.phase_end` | `phase`, `outcome` (`ok`, `deadline`, `provider_call_timeout`, `callback_timeout`, `callback_error`, `privacy`, `cancelled`), `elapsed_ms`, `budget_left_ms` |
| `provider.call` | `phase`, `outcome`, `model`, `elapsed_ms`, `max_tokens`, `call_timeout_ms`, `finish_reason`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `cost_usd` |
| `correction.end` | `status`, `reason`, `reviews`, `repairs`, `elapsed_ms`, `deadline_ms` |

Every event carries `correlation_id`, which is the orchestrator's
`x-correlation-id` when valid, so the two services' logs can be joined on it.

Reason codes are unchanged for existing callers, with one exception: the generic
`timeout` is no longer produced. A deadline expiry reports `timeout_<phase>`, and
a per-request timeout reports `provider_call_timeout_<phase>`. The new code
withholds with `pipeline_budget_exhausted` when too little time remains.

## 8. Verification (offline, synthetic)

- `tests/test_correction_timeout_attribution.py`: deadline and provider-timeout
  reasons for each of the four phases through the real `/analyze` route; budget
  subtraction and clean termination; no provider call when the budget is
  exhausted; events carry timing and tokens; logged events never contain
  sentinel document text; sanitizer; observer failures do not change the outcome;
  the orchestrator correlation ID is used and unsafe IDs are not logged; the
  output-allowance measurement.
- Correction suites: all passing (26 new tests), including the existing route tests after two
  test doubles were updated to accept the new keyword arguments.
- Full Imaginator suite (excluding the docx-blocked module): 584 passed, 3 skipped. Thirteen tests in
  `test_export_receipt_title_binding.py` and `test_renderer_no_defaults.py` fail,
  and `test_export_isolation_and_parity.py` does not collect. All are
  `No module named 'docx'` in this virtualenv. This environment gap predates
  this change; the code paths involved were not modified.

## 9. Still open (not done here)

1. **Orchestrator retry.** On the JSON and single-file paths, `mlAuthMiddleware`
   retries any failed `POST /analyze` up to three times, including timeouts. That
   can rerun the paid pipeline. Recommend no retry on timeout for analyze. This
   is in the Login repo and needs approval.
2. **Privacy of the correction payload.** The sanitized events do not change the
   outbound payload. As recorded in the earlier live report, the disclosure gate
   scans source documents only, not draft lines or claims. This remains a
   disclosed limitation.
3. **Which phase timed out** in either run remains unknown. The first measured
   phase is in the next run's events.
4. **Live verification of the 60 s per-call timeout** still has not happened.
5. **Deployment.** The changes are committed to a local branch only. Pushing to
   `master` auto-deploys to production, so it needs approval.

## 10. Proposed live run (for approval; not executed)

**Approval needed before any step below:** deploy this change to Imaginator
(auto-deploy from `master`), and one paid synthetic analysis.

1. Preconditions: Imaginator live commit equals the approved commit; the
   `CORRECTION_PROVIDER_DISCLOSURE=1` setting is unchanged; the fixture is the
   synthetic single-file DOCX used in run 1 (910 chars); credits are at least 33.
2. Run once through the same UI path as run 1. Record the UTC request time and
   the response's `correction_status`, `correction_reason` and commentary at once.
3. Find the correlation ID from the orchestrator `MULTI_FILE_IMAGINATION Success`
   or `credit_request_outcome` line, then query both services' logs with explicit
   UTC bounds from request time minus 2 min to response time plus 2 min. Page
   forward with `limit` 100. Filter Imaginator logs on `correction_event` and
   join on `correlation_id`.
4. Expected evidence: one `correction.budget` line; a `phase_start` and
   `phase_end` pair for each call; a `provider.call` line with tokens and
   `finish_reason`; and a `correction.end` line naming the reason.
5. Stop rules: if the run is withheld, do not rerun. Diagnose from the events.
   A second paid run needs separate approval.
6. Spend: at most six calls, bounded at about $0.065 (section 6), plus the
   pipeline. Account credits: 33.
7. Rollback: revert to `10b20d8` on `master`, or to `429a6dd` for the last
   verified correction build.

## 11. Git state

- Changes are on local branch `wip/correction-timeout-attribution`, not pushed.
- No deploy, provider call or Render configuration change was made for this
  diagnosis.
