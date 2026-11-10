# Five-case, three-model screen — pre-execution brief

**No calls have been made.** This is the plan for authorization. Everything here is
resolved against the live OpenRouter catalogue and the frozen local fixtures.

## 1. Model availability — one requested slug does not exist

Verified against `GET /api/v1/models` (432 entries, 2026-11-09), a zero-cost
catalogue read.

| Requested | Status | Used | Prompt $/M | Completion $/M | Context |
|---|---|---|---|---|---|
| `google/gemma-4-31b-it` | **available** | as requested | 0.200 | 0.400 | 262,144 |
| `google/gemini-2.5-flash-lite` | **available** | as requested | 0.100 | 0.400 | 1,048,576 |
| `xiaomi/mimo-v2-flash` | **NOT PRESENT** | `xiaomi/mimo-v2.6-flash` | 0.140 | 0.280 | — |
| `openai/gpt-4o-mini` | **available** | as requested (baseline) | 0.150 | 0.600 | 128,000 |

**The substitution is the one decision that needs your confirmation.** Searching the
catalogue for `mimo` returns five entries — `xiaomi/mimo-v2.6-pro-ultraspeed`,
`xiaomi/mimo-v2.6-flash`, `xiaomi/mimo-v2.6-pro`, `xiaomi/mimo-v2.5-pro`,
`xiaomi/mimo-v2.5` — and no `v2`. `xiaomi/mimo-v2.6-flash` is the nearest MiMo Flash.
I have used it and recorded the substitution in `plan.json` and in every report rather
than making it silently. **If you want a different MiMo variant, say which; otherwise
the screen proceeds with `v2.6-flash`.**

### Endpoint pinning — `cohere/fp8`

`xiaomi/mimo-v2.6-flash` is served by **eight** endpoints. Verified against
`GET /api/v1/models/xiaomi/mimo-v2.6-flash/endpoints`:

| Tag | Quant | Prompt $/M | Completion $/M | Uptime 30m | p50 latency | p50 tok/s |
|---|---|---|---|---|---|---|
| `darkbloom/fp4` | fp4 | 0.100 | 0.280 | — | — | — |
| `gmicloud/bf16` | bf16 | 0.112 | 0.224 | — | — | — |
| **`cohere/fp8`** | **fp8** | **0.140** | **0.280** | **100%** | **517 ms** | **78** |
| `makora/fp8` | fp8 | 0.140 | 0.280 | 100% | 923 ms | 26 |
| `deepinfra/fp8` | fp8 | 0.140 | 0.280 | — | — | — |
| `xiaomi/fp8` | fp8 | 0.140 | 0.280 | — | — | — |
| `tenstorrent/fp8` | fp8 | 0.165 | 0.330 | — | — | — |
| `venice/fp8` | fp8 | 0.175 | 0.350 | — | — | — |

Note the title said `cohere/fp8` while the snippet said `makora/fp8`; **both exist**
as valid tags at identical price. **Confirmed policy: `cohere/fp8` first, `makora/fp8` if cohere cannot serve the
request.** `cohere/fp8` is also the better performer (517 ms p50 vs 923 ms, 78 vs
26 tok/s at the same price).

The fallback advances **only on a routing failure** — OpenRouter reporting that no
endpoint can serve the model (`no endpoints found`, `provider returned error`,
`model not available`). It does **not** advance when the model answers badly. A
refusal, a malformed envelope or a poor objection is still a review, and silently
re-asking a different endpoint would turn one answer into a best-of-two — a
different experiment from the one described. Auth errors, rate limits and 5xx are
likewise surfaced rather than masked.

Every response records which provider actually served it, so a fallback that fired
is visible in the report rather than inferred.

Pinning is now implemented in `make_asker(provider_only=[...])`, which sends:

```json
"provider": { "only": ["cohere/fp8"] }
```

**Why this matters more than convenience:** without a pin, OpenRouter may route each
request independently. An `fp4` routing would be a materially weaker model than the
`fp8` the result is attributed to, and the comparison would be confounded without
showing it. The response's own `provider` field is retained on every call, so a pin
that did not hold is **counted as a violation** in the report rather than assumed
away.

## 2. Exact case list

Five cases, all reused verbatim from the two existing frozen fixtures. No new drafts,
no new sources, no repairs.

| # | Case ID | Group | Probe | Expected | Why this case |
|---|---|---|---|---|---|
| 1 | `faithful_paraphrase` | familiar | clean paraphrase | **approve** | The primary endpoint. Both earlier models refuse it on equivalent-word objections. |
| 2 | `cal_ops_faithful` | unfamiliar | clean paraphrase | **approve** | A clean draft in a domain the models have not seen; qwen approved this one, so it separates "refuses some clean drafts" from "refuses all". |
| 3 | `unsupported_method` | familiar | genuine method defect | **revise** | The most common real defect, and the one both earlier models handled correctly. Tests that a stricter model still finds it. |
| 4 | `wrong_record_attribution` | familiar | wrong ownership | **revise** | Tests ownership specifically. |
| 5 | `omitted_record` | familiar | whole record absent | **revise** | The failure mode a "be stricter" model is most likely to miss. |

Two clean, three defective. Every case's draft and sources are byte-identical to the
calibration run; only the model varies.

## 3. Frozen materiality policy

**One policy, unchanged, used for all four models.** It is the existing frozen
contract (`instructions_sha256 = 4261e9cb5b031230aa2a3adf3f818346eab02318dd229b430eca31f166d17e7b`),
mode `imaginative`. **The materiality proposal is NOT applied** — that instruction
remains untested and is not part of this screen.

The contract already contains the relevant guidance, unchanged:

> "Preserve faithful paraphrases, useful restructuring, and mode-permitted skill
> interpretations. Do not require verbatim wording."
> "Overlapping dates are a fact, not a contradiction… Object only when the DRAFT
> asserts a timeline the source denies."

So this screen asks a narrower question than the proposal did: **with the policy
already frozen and identical, does a different model simply behave better?**

## 4. Request limit and cost-stopping policy

| | |
|---|---|
| Cases | 5 |
| Models | 4 |
| Arms | 1 |
| Logical review calls | **20** |
| Attempts per call | 2 |
| **Hard request ceiling** | **40** |
| **Observed-cost stopping point** | **$0.10** |
| Projected spend (live pricing) | **$0.0144** |
| Re-reviews | **none** |

- The **request ceiling is hard**: `RequestMeter.can_send()` is checked *before every
  HTTP attempt*, including retries, so the run cannot exceed 40 requests no matter how
  many fail.
- The **cost stopping point is not a proof of maximum spend.** It is enforced against
  *observed* cost. Any response that omits usage increments `unmetered_requests` and
  sets `cost_is_known = False`; the report then says the true spend is unknown rather
  than quoting a ceiling it cannot substantiate. Missing usage is recorded as unknown,
  never as zero.
- Per-model projection: gemma $0.0043, gemini-lite $0.0029, mimo $0.0030, 4o-mini
  $0.0043.

## 5. What will be reported — four outcomes, separately

As required, correctness is **not** inferred from a decision word or a keyword.

Endpoint routing is reported for every model: what was pinned, what the response
said actually served each call, and a violation count. A pin is an instruction; the
reported provider is the fact.

| Outcome | How it is decided |
|---|---|
| **valid judgment** | the answer parses into a usable decision with no structural problems. `approve` accompanied by issues is **invalid**, not an approval. |
| **actual detection** | on a defective case, all five criteria met: fact identified, source presence stated, draft presence stated, owner named *where ownership is at issue*, and the correction does the right kind of thing. A `revise` word alone earns nothing. |
| **invented objection** | on a clean case, any objection at all — including one that merely restates a permitted paraphrase. |
| **missed defect** | on a defective case, anything short of full detection. |

`owner-not-explicit`, `wrong ownership` and `source/draft reversal` stay distinct: the
first is a miss only where ownership is at issue; the latter two are wrong claims and
are quoted as such.

The classifier was **validated offline against answers already recorded** before this
plan was written — 14 tests in `tests/test_model_screen.py`, including that
gpt-4o-mini's `approve`-with-issues answer on `omitted_record` is reported as invalid
plus missed, never as a release.

## 6. Per-model output shape

For each of the 4 models: valid / detection / invented / missed counts, plus the first
objection quoted verbatim per case so any count can be checked against the text.

## 7. Explicitly not doing

- No production change of any kind.
- No repair path, no re-review, no release decision.
- No automatic dismissal of objections and no automatic semantic veto — the report
  describes what each model said; it does not overrule it.
- No second arm, no instruction change. The frozen policy is identical for all four
  models, which is what makes the comparison attributable to the model.
- No rerunning of individual failures.

## 8. Interpretations fixed in advance

| Result | Reading |
|---|---|
| some model approves both clean drafts **and** detects all 3 defects | a smarter model is enough; proceed to a repair/release test with that model |
| some model approves clean drafts but misses defects | the trade is strictness, not capability; the model is not a drop-in |
| every model still refuses clean drafts | not a model-capability problem at this size; the defect is in the contract or the task framing |
| usage missing on any call | spend for that call is unknown; the cost figure is reported as a range, not a ceiling |

## 9. Authorization requested

To execute:

```
python -m experiment.model_screen --authorized --out results/model-screen-v1
```

Say go and confirm the MiMo substitution (or name a different variant), and I will
run it once, report the four outcomes per model, and retain every answer including
failures.
