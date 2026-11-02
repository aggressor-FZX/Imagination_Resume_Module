# Retry resilience for transient provider failure

Status: **implemented, tested and deployed.** Not yet exercised by a real transient
failure on the live path — the endpoint recovered before I could force one. That is
stated plainly rather than papered over.

## The gap this closes

A transient failure in a late stage used to discard the value of every earlier
successful stage, and cost a whole website analysis. Measured 2026-11-02: the same valid
13.6 KB review request returned `504` in ~8.5 s on some attempts and `200` in 26–39 s on
others. One bad answer was throwing away a completed pipeline.

## Where the retry lives, and why there

In the adapter (`correction_provider.CorrectionProvider`), which is the only layer that
sees an HTTP status. That placement is the whole design:

- **Completed phases are never re-run.** Only the failed phase repeats. The inventory
  answer, the parsed source, the writer's draft and every earlier review result are all
  still in hand when a later phase retries.
- **Retries do not consume the engine's call budget.** `self.calls` counts engine phases
  (bounded by `MAX_CALLS`); `self.attempts` counts transport tries. If retries counted
  against `MAX_CALLS`, a flaky endpoint would silently shorten the review-and-repair
  cycle those retries exist to protect.

## Classification

| Status | Treated as | Retried |
|---|---|---|
| `200` with a valid envelope | success | — |
| `429` | `provider_rate_limited` | yes |
| `502`, `503`, `504` | `provider_unavailable` | yes |
| `400`, `401`, `403`, `404`, `422` | `provider_rejected_request` | no |
| `200` with unusable envelope | `provider_error` | no |
| read timeout | `provider_call_timeout` | **no** |
| malformed / incomplete review | `malformed_*`, `incomplete_review` | no |
| factual verdict (`uncertain`, `missing`) | `source_completeness_uncertain`, `source_facts_missing` | no |

Two exclusions are deliberate and worth defending:

**Read timeouts are not retried.** A read timeout has already consumed most of a phase's
budget, so repeating it usually *guarantees* failure with nothing left, instead of
stopping cleanly with a reason that says what happened. Availability is retried; slowness
is not.

**A valid but incomplete review is not retried.** It is an unresolved delivery failure
handled by the existing bounded completion re-ask (`incomplete_review`), which names the
omitted lines. Treating it as availability would blur a factual decision into a transport
one.

## The allowance

| Setting | Value | Why |
|---|---|---|
| `RETRY_MAX_ATTEMPTS` | 3 | One initial attempt plus two. A blip, not an outage. |
| `RETRY_BASE_DELAY_SECONDS` | 1.5 | Doubles per attempt when no header is given. |
| `RETRY_MAX_DELAY_SECONDS` | 15 | Caps an advertised delay so it cannot pin the request. |
| `RETRY_MIN_REMAINING_SECONDS` | 10 | No attempt is started without a realistic chance. |

`Retry-After` is honoured in its integer-seconds form. An HTTP-date or a missing header
falls back to backoff; a huge value is capped. The budget check uses the remaining time
the engine offered, minus time already spent in the phase, minus the delay — so a retry is
only started when it can actually finish.

## Telemetry

Each attempt emits one sanitized `provider.call` event carrying `attempt`, `http_status`,
`elapsed_ms`, the serving `provider`, the `phase`, `model` and token counts. No document,
draft, prompt or model text can reach a log line; the sanitizer enforces a fixed
vocabulary and drops anything else.

Billing stays conservative: every failed attempt records `usage_unknown: true`, because a
refused or incomplete request may still be billed for its prompt.

Recording the serving provider was not cosmetic. The 2026-11-02 failures could only be
explained by re-sending captured requests, because the log did not say which endpoint had
answered.

## What was proven

30 tests in `tests/test_provider_retry_resilience.py`, covering all four required
scenarios plus the guardrails:

- `429` then success, and `502`/`503`/`504` then success;
- repeated failure stops cleanly with a specific reason (never a loop);
- `400`/`401`/`403`/`404`/`422` never retried;
- malformed output and factual verdicts never retried;
- `Retry-After` honoured, fallback backoff, and capping;
- no retry without remaining budget;
- retries do not consume the phase budget;
- per-attempt telemetry carries no content.

Mutation-checked rather than merely passing: making `429` non-retryable causes 3 failures,
and retrying `401` causes 2. Both mutants are caught.

**Publication checks are untouched.** `publication_gate.py`, `tier_a_verifier.py`,
`disclosure.py` and `correction_protocol.py` are byte-for-byte unchanged, and a test
proves a retried *success* still faces the metric verifier — an invented figure still
fails.

## Defects found in my own earlier work while doing this

1. `provider_rate_limited` was emitted as an outcome but missing from the sanitizer's
   vocabulary, so the events that explain a retry sequence were being silently dropped.
   The log looked like nothing had happened.
2. A served-but-unusable `200` was reported as outcome `ok` *and* then as a transport
   failure — exactly the ambiguity that made the live failures hard to read.

## Live verification, and its limit

Against the real endpoint, driving the deployed adapter with the exact captured run-3
review request:

| Invocation | Attempts | Status | Latency |
|---|---|---|---|
| 1 | 1 | 200 | 37.6 s |
| 2 | 1 | 200 | **60.9 s** |
| 3 | 1 | 200 | 54.0 s |
| 4 | 1 | 200 | 47.4 s |

All succeeded on the first attempt, so **no real 504 occurred and the retry itself was
not exercised live.** I could not manufacture a transient failure on demand, and I am not
going to claim the retry recovered one.

Invocation 2 is still meaningful: at **60.9 s** it would have been killed by the previous
fixed 60 s ceiling and is why that ceiling was replaced.

## Not yet done

- **One single-file delivery on the live website** is still unproven. The three
  authorized analyses were consumed before this work; delivery now needs a fresh
  allowance, and per the agreed order the single-file delivery must pass *before* any
  multi-file acceptance run.
- The multi-file route does not run source correction at all, so retry resilience does not
  apply to it.
