# Handoff — transient-failure resilience (2026-11-02, session 5)

- **Tests:** 757 passed, 3 skipped.
- **New report:** `docs/RETRY_RESILIENCE_2026-11-02.md`.
- **Live:** capture OFF (`CORRECTION_LIVE_CAPTURE=0`), disclosure ON, retry code deployed.

## Done this session

- [x] Retry classification: `429`/`502`/`503`/`504` retried; `400`/`401`/`403`/`404`/`422`
      never retried; malformed and factual verdicts never retried.
- [x] Retry lives in the adapter so **completed phases are never re-run** and the engine's
      phase budget is untouched by transport retries.
- [x] Bounded allowance: 3 attempts, `Retry-After` honoured and capped, backoff otherwise,
      and no attempt started without 10 s of remaining budget.
- [x] Per-attempt telemetry: `attempt`, `http_status`, `elapsed_ms`, serving `provider` —
      no content. Failed attempts record `usage_unknown: true`.
- [x] Read timeouts deliberately NOT retried (they consume the budget they would need).
- [x] Publication checks unchanged: gate, verifier, disclosure and protocol are
      byte-for-byte identical; a test proves a retried success still faces the verifier.
- [x] 30 tests, mutation-checked (un-retrying 429, or retrying 401, both fail).
- [x] Fixed two defects in earlier work: `provider_rate_limited` was dropped by the
      sanitizer vocabulary; a served-but-unusable 200 was reported both as `ok` and as a
      transport failure.

## Live verification — and its limit

The exact captured run-3 review request, driven through the deployed adapter: 4
invocations, all `200` on the **first** attempt (37.6 s, 60.9 s, 54.0 s, 47.4 s). No real
504 occurred, so **the retry was not exercised live**. Invocation 2 at 60.9 s would have
been killed by the old fixed 60 s ceiling, which is why that ceiling was replaced.

## Remaining — needs an allowance, not more code

1. **One single-file delivery on the live website is still unproven.** Three authorized
   analyses were consumed before this work. Per the agreed order, single-file delivery
   must pass before any multi-file acceptance run.
2. Nothing about approved output is proven: no receipt, no displayed commentary, no
   actual PDF/DOCX bytes.
3. Multi-file does not run source correction, so retry resilience does not apply to it.

## Verification

```bash
cd Imagination_Resume_Module
../.venv/bin/python -m pytest -o addopts="" -q      # 757 passed, 3 skipped
```

---

# Previous session (4) — live delivery attempts, superseded


- **Live:** `master` tracks the deploy; see `git log -1`.
- **Tests:** 727 passed, 3 skipped.
- **Reports:** `docs/LIVE_DELIVERY_RUNS_2026-11-02.md` (the three runs, in detail),
  `docs/CORRECTION_BOUNDARY_ACCEPTANCE_2026-11-02.md` (implementation acceptance),
  `docs/LIVE_CAPTURE_RUNBOOK.md` (how to capture and replay).
- **Capture is OFF** (`CORRECTION_LIVE_CAPTURE=0`) and the container disk was purged.

## Outcome in one paragraph

Implementation acceptance passes and grew substantially. Live delivery **did not** pass:
three authorized website analyses all ended `withheld`, for three *different* reasons,
and each was diagnosed from evidence rather than guessed at. The third run is the
important one — it reached source correction for the first time, and its capture shows a
valid 13.6 KB review request returning `200` in 26–38 s while the same request also
returns `504` from the upstream endpoint in ~8.5 s on other attempts. The remaining
blocker is upstream provider availability, not a defect this session could clear.

## Completed this session

- [x] Found and fixed the reason capture could never write: the deployed image mounts its
      disk at `/app/data`, inside the repo root, and the container has no `.git` and no
      `git` binary, so the git-ignore guard refused the authorized path. A separately
      mounted volume is now accepted as the equivalent guarantee (`f2fb8d4`).
- [x] **Run 1** captured live, then reproduced exactly: `429 engine_overloaded` from
      DeepInfra while `allow_fallbacks: False` pinned the other endpoint out of reach.
      Fixed by enabling fallbacks and reporting `provider_rate_limited` (`bb926b4`). Also
      fixed the capture wrapper dropping non-200 bodies — the reason run 1 looked like
      "no response at all".
- [x] **Run 2** exposed an earlier defect: `response_format={"type":"json_object"}` was
      accepted by the adapter and never put on the wire, so every strict-JSON stage ran
      unstructured. Forwarded it, and added recovery for a truncated envelope whose
      `final_markdown` closed cleanly — mid-sentence cuts are still rejected (`70b9824`).
- [x] **Run 3** reached correction and its capture showed the adapter abandoning a valid
      review at a fixed 60 s ceiling with its own granted budget unspent. Each call's
      timeout is now sized from the remaining budget, floored at 60 s, capped at 120 s
      (`6dd7b1d`).
- [x] Both retained captures replay offline to their exact recorded verdicts.
- [x] Capture disabled after verification; container captures deleted; two synthetic
      files retained locally (`0600`, outside the repo) with checksums in the report.

## Remaining — needs a decision, not more code

1. **Upstream availability is the open blocker.** The correction model has two endpoints
   and both intermittently return `504` (~8.5 s) for a request that otherwise answers
   fully. Options, in preference order:
   a. a second provider model for correction (routing, not only fallback), decided with
      the disclosure policy in mind;
   b. bounded retry of a `5xx` **within the already-granted budget** (not a new timeout
      increase — that was measured and does not address a server-side 504);
   c. accept withholding under provider saturation and surface it as retryable (partly
      done already via `provider_rate_limited`).
   Any option needs a live run to prove, and there is **no reserved run left**.
2. **A fresh live-test allowance is required** for the 66-credit multi-file run that was
   authorized but never reached (runs 1–3 all failed before multi-file delivery was in
   scope), plus verification of whichever option above is chosen.
3. **Never yet exercised on the live path:** an approved result, the receipt, the
   displayed commentary, and the actual PDF/DOCX bytes. Nothing about those is proven.

## Verification

```bash
cd Imagination_Resume_Module
../.venv/bin/python -m pytest -o addopts="" -q          # 727 passed, 3 skipped
PYTHONPATH=. python tools/live_capture_preflight.py    # 0 ready / 2 not ready / 3 disabled

# Replay a retained capture with no network:
CORRECTION_PROVIDER_DISCLOSURE=1 ../.venv/bin/python -c \
  "import live_capture; r,p,s = live_capture.replay('<path>.json'); print(r.status, r.reason)"
```

## Guardrails held

No force-push; fast-forward pushes only. `secure_storage.json.encrypted` untouched. No
document, draft, prompt or model text in ordinary logs or in Git. The publication gate
was never bypassed — every run withheld rather than deliver unverified content, and the
PDF/DOCX controls stayed disabled on the withheld result, which is correct.
