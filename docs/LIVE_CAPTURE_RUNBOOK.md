# Live capture runbook — one website run, fully replayable

Purpose: run **one** synthetic `/analyze` request through the real website and record
enough of it that a failure can be replayed offline instead of re-bought. The earlier
probes drove the correction engine directly with constructed drafts; this records the
live path, so its evidence is about the website.

## 1. Prepare (before any run)

Pick a host directory **outside the repository**, owner-only:

```bash
mkdir -p ~/.local/state/imaginator-live-capture
chmod 700 ~/.local/state/imaginator-live-capture
```

The writer refuses a directory inside the repo that is not git-ignored, so an
accidental `CORRECTION_LIVE_CAPTURE_DIR=.` cannot commit a resume-shaped file.

## 2. Enable capture on the Imaginator service only

Render → `imaginator-resume-cowriter` (`srv-d3nf73ur433s73bh9j00`) → Environment:

| Key | Value | Why |
|---|---|---|
| `CORRECTION_LIVE_CAPTURE` | `1` | Explicit opt-in; anything else leaves capture off. |
| `CORRECTION_LIVE_CAPTURE_DIR` | `/var/data/live-capture` | Must be a persistent disk or a path you can retrieve. |

> The Render container's filesystem is ephemeral. Attach a **disk** mounted at a known
> path, or the capture is lost on the next deploy. Do not use `/tmp` for a run you
> intend to replay.

Enabling these requires a **deploy**, not a restart: environment changes do not apply
to a running container.

## 2b. Preflight before spending a run

A capture that silently fails to write wastes the run it was meant to preserve, and you
would not notice because capture never breaks a request. Run the preflight first:

```bash
PYTHONPATH=. python tools/live_capture_preflight.py
```

| Exit | Meaning |
|---|---|
| `0` | Ready: writes `0600` files in a `0700` directory, and free space is adequate |
| `2` | Not ready — the message names the reason (missing dir, unsafe location, write failure) |
| `3` | Capture disabled |

It writes and removes its own probe file, and prints no document text.

## 3. Run exactly one synthetic analysis

Use the same browser flow the user uses (`/app/resume-builder`) with a
**synthetic-only** resume that contains the literal marker `SYNTHETIC-ONLY`. Every
source document must carry it or the capture refuses to write.

Capture the correlation id from the response headers (`x-correlation-id`) or from the
service logs; it is both the join key across services and the capture filename.

## 4. Read the logs — safely

```bash
timeout 30s render logs -r srv-d3nf73ur433s73bh9j00 --limit 100 --output text \
  | grep -E "correction.(budget|phase_end|end|capture)"
```

`correction.capture` reports **only** counts and a boolean:

```json
{"live_capture":"on","exchanges":4,"source_documents":1,"draft_chars":812,"responses_recorded":4,"written":true}
```

No document, draft, prompt, or model text is ever logged. If you see text in a
`correction_event` or `correction.capture` line, that is a bug — stop and report it.

## 5. Retrieve and replay

Copy the capture out of the disk to a machine with the repository:

```bash
scp … :/var/data/live-capture/<correlation-id>.json ~/captures/
chmod 600 ~/captures/<correlation-id>.json
```

Replay with **no network**:

```python
import live_capture
result, provider, session = live_capture.replay("~/captures/<id>.json")
print(result.status, result.reason, result.review_calls, result.repair_calls)
print([call["phase"] for call in provider.usage["calls"]])
```

The replay feeds the engine the **writer's own recorded draft** and the exact recorded
provider responses, then re-derives the decision with the real validator. It does not
compare against a stored verdict, so a code change that alters the outcome will show.

If the recording ends before the engine is satisfied (for example because recovery
issues a request the live run never made), `ReplaySession` raises `ReplayExhausted`
rather than returning an empty success. A withheld replay result is a real result; an
exhausted replay is an incomplete recording.

## 6. Turn capture off

Set `CORRECTION_LIVE_CAPTURE=0` (or remove it) and redeploy once the case is captured.
Leaving it on accumulates resume-shaped files on a running service.

## What the capture contains

| Field | Content |
|---|---|
| `sources` | Extracted source documents **after** the request's disclosure redaction |
| `draft` | The writer-generated draft that entered correction |
| `exchanges[]` | `{phase, request, response_body, status}` per provider call, verbatim |
| `review` | Normalized review: status, reason, review/repair counts, commentary |
| `gate` | Final gate decision reason codes (identifiers and numbers only) |
| `timing` | Pipeline elapsed, correction budget, upstream timeout, total elapsed |
| `correlation_id` | Join key across orchestrator and Imaginator logs |

## Limits, stated plainly

* Capture covers the **correction** stage. The Drafter's own provider calls are not
  recorded, so a slow or low-quality draft is only visible through its output text.
* It records one request per file; there is no aggregation across runs.
* It is diagnostics. Enabling it must never change a response — a test asserts this by
  comparing responses with capture on and off.
* A capture of a real candidate document is refused by design. If you need to diagnose
  a real document, that requires a separately approved process.
