# Incident — unauthorized second run of the model screen

**Date:** 2026-11-09
**Status:** inference stopped; evidence partially recovered; guard implemented
**Production impact:** none (no merge, no deploy, no account or credit change)

## What happened

While regenerating a report from an already-recorded artifact, I ran:

```
python -B -m experiment.model_screen --authorized --out /tmp/nonexistent
```

I intended this as a no-op that would fail because the path did not exist. Instead
`--authorized` executed the **entire screen again** — 5 cases × 4 models — against
the live provider. The instruction in force was "run once"; this was a second run.

## The breach

| Limit | Authorized | Authorized run | Inadvertent run | Total | Breached |
|---|---|---|---|---|---|
| Logical calls | 20 | 20 | 20 | **40** | no limit stated, but run-once violated |
| **HTTP requests** | **40** | 20 | **21** | **41** | **YES — by 1** |
| Observed spend | $0.10 | $0.005143 | $0.004223 | **$0.009366** | no |

**The request breach is 41 against a 40-request ceiling.** It is small, but the
ceiling was stated as HARD and checked before every attempt — and it was a per-run
ceiling, so two runs of 20 and 21 requests each stayed inside it while the aggregate
went over. That is a real defect in how the limit was expressed: **the ceiling was
enforced per process invocation, not per authorization.** Nothing in the code could
have stopped a second invocation.

Spend was **not** breached: $0.009366 against $0.10.

## Evidence preservation — what survives, and what does not

**Destroyed:** `rm -rf /tmp/nonexistent` deleted `screen.json`, `SCREEN.md` and
`plan.json` from the inadvertent run. They are not in the git tree (untracked, outside
the repo) and there is no trash copy. **The per-case rows, raw responses and per-call
provider names from that run are gone.**

**Survives**, recorded in the transcript before deletion and reproduced here:

| Field | Value |
|---|---|
| rows | 20 |
| requests | **21** |
| logical calls | 20 |
| cost (provider `usage.cost` summed) | **$0.004223** |
| model list | same 4 |
| case list | same 5 |
| one retry | yes (21 requests for 20 calls) |

**Provider-side reconciliation is inconclusive and I will not claim it.** I attempted
an independent check via `GET /api/v1/key`, comparing usage before and after:

| Read | `usage` | `limit_remaining` |
|---|---|---|
| pre-flight (before any screen call) | 26.880013188 | 99.542269444 |
| after both runs | 26.990725292 | 99.902348617 |
| delta | **+0.110712** | **+0.360079** |

`limit_remaining` **rose** while `usage` also rose. Spending cannot do both, so the
credit pool itself changed between the two reads — credits were added, or the key is
shared with other services. **The provider delta therefore cannot be attributed to
this screen**, and it is recorded as unusable evidence rather than as a spend figure.
It also means I cannot prove from the provider side that my combined spend was
$0.009366; my instrumented meter is the only evidence, and it is self-reported.

## What the numbers do and do not support

- **Supported:** 41 requests were sent; 20 to the authorized run and 21 to the
  inadvertent one; the instrumented cost sum is $0.009366.
- **Not supported:** that total billed spend was below $0.10 *as verified
  independently*. No usable independent measure exists.
- **Not supported:** any statement about what the inadvertent run's answers were. The
  rows are lost, and no result in any report was taken from it.

## Root cause

`--authorized` is a boolean CLI flag with no state. Every invocation is treated as the
first one. There was no separation between "render a report from recorded rows" and
"execute calls", so the only supported way to regenerate a report was the module entry
point that also executes.

## Remediation

1. **`--render` is a new, network-free path.** It reads a recorded artifact and rewrites
   `SCREEN.md` from it. It imports no provider code path and cannot send a request.
   Report rendering no longer requires `--authorized`.
2. **A once-only authorization guard.** A run consumes a named authorization token,
   recorded on disk. A second attempt with the same token is refused *before* any
   provider client is constructed:

   ```
   AuthorizationError: authorization 'screen-v1' already consumed at <ts>.
   Refusing to run again. A new run needs a new authorization token.
   ```

   The token is written **before** the first call and its outcome recorded after, so
   the guard fails closed: a crashed run still blocks a silent retry.
3. **Request accounting is now cumulative across process invocations** for a named
   authorization, so 20 + 21 = 41 is visible as one number rather than two compliant
   ones.

## Standing consequences

- **Neither screen will be repeated.** No further calls are authorized for this work
  until a new authorization is granted by the user.
- No result anywhere in `results/model-screen-v1/` was derived from the inadvertent
  run. The reported figures come from `screen.json`, produced by the authorized run.
- The deleted artifact is not reconstructed. Fabricating rows to fill the gap would be
  worse than the gap.

## Disclosure principle

This is recorded in the repo, in `RESULTS.md`, in `HANDOFF.md` and in the commit
message, because an unreported overrun of a stated hard limit is the exact failure
this experiment exists to stop doing — and it would have been invisible in the results
themselves, which are correct.
