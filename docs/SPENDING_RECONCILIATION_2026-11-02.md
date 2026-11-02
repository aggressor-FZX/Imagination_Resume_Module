# Spending reconciliation — provider probes vs website credits

Two currencies are involved and they are **not** interchangeable. Conflating them has
already caused one wrong impression ("the probes were free"), so they are reconciled
separately here.

- **Website credits** are the product's own currency, deducted per `/analyze` run
  (33 credits). They are a billing abstraction, not a provider charge.
- **Provider spend** is real USD paid to OpenRouter for tokens. It applies to every
  request, whether it came from a website run or from a diagnostic probe.

Rates used (`pipeline_config.PRICING` for `qwen/qwen3-coder`): input `$0.00015`/1K,
output `$0.0006`/1K. Costs below are computed from token counts actually recorded in
each probe's output or in the capture files — not estimated from a per-run average.

## Website credits

| Run | Credits | Balance after | Outcome |
|---|---|---|---|
| Run 1 (`d6de1e3e…`) | 33 | 1232 | withheld, `provider_error` (429) |
| Run 2 (`4e39c596…`) | 33 | 1199 | withheld, `fallback_output` |
| Run 3 (`b0c46daf…`) | 33 | 1166 | withheld, `provider_call_timeout_initial_review` |
| **Total** | **99** | 1265 → 1166 | — |

**No credits were consumed in session 5.** The resilience work used no website run.

## Provider spend, session 4 (probes)

| Activity | Requests | Basis | Cost |
|---|---|---|---|
| Reproduce run 1's 429 (`probe_429.py`) | 2 | 4,789-byte prompt, no completion | $0.00042 |
| Verify fallback routing (`probe_route.py`) | 1 | 1,386 prompt + 68 completion | $0.00025 |
| Reproduce run 3's review (`probe_review_latency.py`) | 3 | 1,386 prompt each; 2× 504 (no completion), 1× 603 completion | $0.00078 |
| Latency variance probes | 5 | 1,386 prompt each; 4× 504, 1× 603 completion | $0.00143 |
| **Session 4 total** | **11** | — | **≈ $0.0029** |

Rejected requests (429/504) are counted at **prompt cost only, never as free**. A refused
request may still have been billed for its prompt, so treating it as zero would
understate spend. No attempt in this table is recorded as `$0.00`.

## Provider spend, session 5 (probes)

| Activity | Requests | Basis | Cost |
|---|---|---|---|
| Retry proof (`prove_retry.py`) | 5 | 1,386 prompt + ~600 completion each, all `200` | ≈ $0.0034 |
| **Session 5 total** | **5** | — | **≈ $0.0034** |

Four of the five returned `200` on the first attempt at 37.6 s / 60.9 s / 54.0 s / 47.4 s;
the fifth was the initial exploratory invocation. The endpoint had recovered, so no 504
was observed and the retry path was not exercised live.

## Aggregate

| Category | Total |
|---|---|
| Website credits (3 analyses) | **99 credits** (1265 → 1166) |
| Provider spend, sessions 4 + 5 | **≈ $0.0063** |
| Allowance for those sessions | $1.00 |
| **Utilisation** | **≈ 0.6 %** |

## Billing uncertainty, stated

These are **lower bounds**, not settled invoices. Three sources of uncertainty:

1. **Refused requests.** A 429 or 504 consumed no completion tokens, but whether the
   prompt was billed is the provider's decision and is not exposed per request. The table
   assumes it was.
2. **Provider-side rounding.** Costs are recomputed from the published per-1K rates;
   the invoice may differ in the last decimals.
3. **No per-request ledger.** The account endpoint reports `total_usage` for the whole
   account (`$1141.12` at the time of writing), which production traffic shares. Probe
   spend therefore cannot be isolated from it and is derived from token counts instead.
   That derivation is why the figures carry `≈`.

Nothing above is reported as exact, and no figure is reported as `$0.00` for an attempt
that was actually sent.

## What is deliberately not claimed

- The probes did **not** constitute website acceptance and are not presented as such.
- The successful retry probes demonstrate that the *endpoint serves the request*; they do
  **not** demonstrate the retry recovering a real transient failure, because none
  occurred during those five invocations.
- Provider spend is not a proxy for product value: three website runs that withheld still
  cost 99 credits to the user regardless of the ~$0.003 of provider tokens they consumed.
