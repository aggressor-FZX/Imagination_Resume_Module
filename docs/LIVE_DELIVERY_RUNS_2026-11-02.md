# Live delivery runs — 2026-11-02 (authorized allowance: 3 analyses)

Status: **three authorized website analyses consumed; live delivery NOT achieved.**
Implementation acceptance and live delivery acceptance are reported separately at the
end of this document, as required.

## Authorization under which these ran

- Capture enabled on `imaginator-resume-cowriter` only, writing to
  `/app/data/live-capture` (the 1 GB persistent disk), synthetic-only, access
  controlled, capture **disabled again** after the runs.
- Up to three website analyses. All three were used. No reserved run remains.
- Additional aggregate provider spend allowance of $1; see the spending record.

## Run 1 — `d6de1e3e-c1b1-46b1-964c-8282586e03da`

| | |
|---|---|
| Path | real website: `cogitometric.ai/app/resume-builder` → orchestrator `/api/analysis/multi-file?stream=1` → Imaginator `/analyze` |
| Document | `synthetic-single.docx`, every line marked `SYNTHETIC-ONLY` |
| Credits | 33 (1265 → 1232) |
| Outcome | withheld, `provider_error` |
| Captured | 1 exchange: a complete `source_inventory` request (4,789 bytes, 22 segments) and **no response** |

Root cause, established by re-sending that exact captured request from inside the
container — not by inference:

```
HTTP 429  provider_error_code=engine_overloaded
          provider_name=DeepInfra   is_byok=false
          limit_source=upstream_provider_shared_pool
```

The model is served by exactly two endpoints (DeepInfra fp4, Venice fp8). The request
had `provider: {allow_fallbacks: False}`, so there was nowhere to route. The same
request with fallbacks enabled returned `200` from Venice, `finish_reason=stop`.

Two defects were fixed (`bb926b4`): fallbacks are now enabled (redirects stay
disabled), saturation is reported as `provider_rate_limited` rather than a generic
error, and the capture wrapper records non-200 bodies. The last one mattered: it had
recorded the live 429 as "request with no response", the one shape that cannot be
diagnosed offline.

## Run 2 — `4e39c596-3e8c-4c68-a2d3-0427b9bb0c4c`

| | |
|---|---|
| Credits | 33 (1232 → 1199) |
| Outcome | withheld, `fallback_output` |
| Captured | **nothing** — the gate blocked before source correction started |

Different, earlier blocker. The StarEditor could not parse its answer, returned an
empty result, and the pipeline marked itself `fallback_output`. Because the failure was
upstream of correction, no capture could exist for it; the container logs identified it.

Found by reading the transport instead of guessing: the LLM adapter accepted
`response_format={"type": "json_object"}` from every stage and **never put it on the
wire** — the payload was a fixed five-key dict. Every stage that asked for strict JSON
silently ran unstructured. The editor, whose whole output is a JSON envelope, took the
visible damage: 3 parse failures in 2 days against 20 successes.

Fixed in `70b9824`: JSON mode is forwarded, and a truncated envelope whose
`final_markdown` string closed cleanly is recovered. A body cut mid-sentence is still
rejected — publishing a half-written resume is worse than withholding it, because the
missing tail reads as the candidate having fewer achievements.

## Run 3 — `b0c46daf-2f9e-49e6-9970-349c460d88f8`

| | |
|---|---|
| Credits | 33 (1199 → 1166) |
| Outcome | withheld, `provider_call_timeout_initial_review` |
| Captured | 2 exchanges: inventory `200` (1,080 bytes), review request 13,578 bytes over 32 lines / 22 segments, **no response** |

Real progress: the editor fix worked and correction ran for the first time. The capture
made this measurable:

- granted budget 120 s; pipeline elapsed 75.6 s; total 183.8 s of a 300 s SSE allowance;
- the review was killed at the adapter's **fixed 60 s per-call ceiling** after ~48 s of
  inventory, with ~12 s of the granted budget still unused;
- replaying that exact review request from the container returned **200 in 38.3 s**.

So the request was never too slow; the ceiling was smaller than the work. Fixed in
`6dd7b1d`: the engine offers each phase its remaining budget out-of-band and the
adapter takes `max(60s, budget)` capped at 120 s.

## The unresolved upstream condition

Measuring that captured request repeatedly shows the endpoints themselves are unstable:

```
attempt A: 504 from Venice in 8.7s, 504 in 8.3s, 504 in 8.4s
attempt B: 504 in 8.4s, then 200 from Venice in 26.6s (finish=stop, 603 completion tokens)
```

A **valid** request returns 504 in ~8.5 s on some attempts and a complete answer on
others. Raising the client timeout cannot fix a server-side 504. The honest statement of
where this stands: the remaining failure is an upstream availability problem, not a
defect that three analyses could have cleared, and it would recur on a customer's run.

## Spending record

Website analyses (33 credits each): **3 used, 99 credits**, 1265 → 1166.

Provider spend for diagnostic probes (billed separately from website credits):

| Activity | Approximate |
|---|---|
| Reproduce run 1's 429 (2 requests, rejected) | ~$0.000 |
| Verify fallback routing (1 request, 200, Venice) | ~$0.001 |
| Reproduce run 3's review request (3 requests) | ~$0.004 |
| Latency probes (5 requests; 4× 504, 1× 200) | ~$0.003 |
| **Total** | **well under $0.02 of the $1 allowance** |

Rejected requests are recorded as near-zero, not as free: a 429/504 consumes no
completion tokens, but the prompt was sent and is counted conservatively.

## What is proven, and what is not

**Proven on the live path:** capture works end-to-end on the deployed service (write,
`0600` in `0700`, on the persistent disk, retrievable, replayable); the volume guard
admits the mounted disk; two distinct live failures were captured or logged precisely
enough to reproduce offline; both replays reproduce their recorded verdicts exactly.

**Not proven:** a published resume, a receipt, or an export. No run reached
`approved`. PDF/DOCX controls stayed **disabled** in the UI on run 1, which is the
correct behaviour for a withheld result, so no export could be inspected. The
publication gate was never bypassed: every run withheld rather than deliver unverified
content.

## Retained evidence

Two synthetic captures are retained locally for replay (never committed; the guidance
forbids resume-shaped text in Git). Both are `0600`, outside the repository:

| File | sha256 |
|---|---|
| `live-run-1-provider-429.json` | `3fe2e99c379b5813c75ed365ed5a62d900ee36636bf0e9f6a7560b0cb7d3053d` |
| `live-run-3-review-timeout.json` | `7ee12d0d0aa0d4a5899a88a3863059e2b0c379441a2829f6c61a28e43db32e26` |

The captures on the container disk were deleted after retrieval, and
`CORRECTION_LIVE_CAPTURE` was set back to `0` with a redeploy.
