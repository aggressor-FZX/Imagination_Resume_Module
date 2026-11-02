# Release packet — source-aware correction (2026-11-01)

**Status: awaiting explicit approval. Nothing has been pushed, deployed, or paid.**

---

## 1. The production disclosure gate: what it is, what it protects

| Item | Detail |
|---|---|
| Implementation | `Imagination_Resume_Module/correction_provider.py` — `_approved_disclosure_available()` / `approved_disclosure_policy()` |
| Enforcement | Checked after payload serialization and size check, before any socket call. Failure raises `PrivacyReviewRequired`, which `source_correction.correct()` converts to `correction_status="withheld"`, `review_reason="privacy_review_required"`. |
| What it protects | The **raw candidate source documents**, which the correction engine must send to a model to inventory and cite. This is the one thing the existing pipeline does not already send. |
| Already-sent baseline | The normal generation path already sends **sanitized** resume text (email/phone/SSN/address/LinkedIn/contact lines removed by `sanitize_resume_pii`) to the **same** OpenRouter endpoint, **same** API key, and **same** model slug (`qwen/qwen3-coder`) used by the drafter. Correction therefore adds no new vendor, credential, or destination. |
| Unmet requirement as of the previous commit | Correction sent raw, unredacted source text. Because the engine validates citations by exact code-point offsets (`quote == sources[sid][start:end]`), ordinary placeholder redaction (`[email removed]`) shifts every offset and breaks the contract. |

### Resolution implemented (not a bypass)

`disclosure.py` adds a **length-preserving** redactor: each covered PII match is
replaced with an equal-length padded label, so every offset and quote the engine
validates stays exact while the sensitive characters are removed.

The gate is now:

1. **Opt-in only.** `CORRECTION_PROVIDER_DISCLOSURE=1` enables it. Unset/other
   values keep the previous closed behavior.
2. **Verified per request**, not trusted from the caller:
   `verify_disclosure()` re-scans the exact `sources` payload about to be sent and
   refuses if any covered PII remains, if the structure is unexpected, or if the
   policy id does not match.
3. **Fails closed at the route.** If `redact_sources()` cannot prove
   correspondence, `_run_source_correction` returns nothing and the route
   withholds with `privacy_redaction_failed`; no request is made.

Covered categories (identical set to the existing normal pipeline): email, US
phone, SSN, street address, LinkedIn URL, and contact header lines combining a
name with any of those. **Deliberately retained:** candidate name on its own
line, employers, schools, city/state/ZIP, GitHub/portfolio URLs, dates, metrics.
Those are resume content the reviewer needs, and the existing pipeline retains
them too.

**Honest limitation:** this is pattern-based redaction of unambiguous categories,
not a proof that arbitrary text is non-identifying. Unusual formats (non-US
phones, non-English addresses, identifiers embedded in free text) can pass
through. A reviewer model still sees candidate name, employers and schools. If
that residual disclosure is not acceptable, the correct decision is to keep the
gate closed — the code supports that as the default.

**New code fingerprints:** `disclosure.py 065e1a51…079f`, `correction_provider.py 2713cced…2802`, `app.py b10cd6f0…b10f`.

**Offline evidence:** `tests/test_disclosure_gate.py` (12) and
`tests/test_disclosure_gate_route.py` (5) cover: default-closed behavior with zero
requests, PII still present → refuse, redaction length preservation per line,
idempotence, per-category detection, unprovable redaction → withhold, and a real
adapter/route run proving only redacted text reaches transport while an approved
result is still produced. Full offline selection: **357 passed**.

---

## 2. Exact commits

| Repo | Branch | Commit | Subject |
|---|---|---|---|
| Imagination_Resume_Module | `feat/hybrid-resume-ingest-phase1` | `35a5eba8547cf2aa689937b7cb1f4ca202e1abc2` | feat(correction): enforce approved disclosure policy with length-preserving redaction |
| | | `f2cfb6086c2fa81db6e9c296ba07bf2b0c4950e6` | fix(correction): correct uniquely owned attribution and verify exports |
| | | `65198c95cd524455daf1dc086ae4d1d25f5efbbc` | test: cover attribution conflict on analyze route |
| | | `a2a93d1` | wip: contain provider error logs and expand correction controls |
| | | `c879a24` | wip: source-aware correction and export safety |
| Landing_Login_page | `feat/hybrid-resume-ingest-phase1` | `1cab17c62512a3f8d6dc5f3f619bdc41d79778bb` | feat(ui): show server correction commentary on approved results |
| | | `45d5fc9985477862f2735aff287676b0fd3cedf7` | wip: redact analysis proxy and persistence logs |
| | | `675e4a21746c98975cb1f72ea7d8ea0ccbf65a00` | wip: preserve source document evidence |

Both branches are **fast-forwardable onto `origin/master`**
(`master` is an ancestor; 5 and 3 commits ahead respectively). Production services
track `master` with auto-deploy on commit.

---

## 3. Affected services

| Service | ID | Repo | Change | Needs env var |
|---|---|---|---|---|
| imaginator-resume-cowriter | `srv-d3nf73ur433s73bh9j00` | Imagination_Resume_Module | Correction engine, attribution fix, disclosure policy, log containment | **yes** — see §4 |
| orchestrator-backend | `srv-d4fb5o4hg0os738sb3a0` | Landing_Login_page | Source-document forwarding, proxy/parser log containment | no |
| orchestrator-frontend | `srv-d4fb5f4hg0os738sasgg` | Landing_Login_page | Displays server correction commentary on approved results | no (build-time static) |

Not affected: document-reader-service, job-search-api, job-search-celery-worker
(Job Search changes are committed separately on its own branch and are **not** part
of this release).

---

## 4. Decision required before deploy

The gate is off by default. **Deploying these commits alone reproduces today's
behavior** (every analysis withholds at `privacy_review_required`) except that
the attribution fix, log containment and UI commentary ship.

To obtain an approved rewrite, the service needs:

```
CORRECTION_PROVIDER_DISCLOSURE=1
```

on **imaginator-resume-cowriter** only. Env changes take effect on a new deploy,
not a restart.

**This is the explicit confirmation being requested.** Setting it authorizes
sending candidate source text — redacted as described in §1, but including name,
employers, schools and city/state — to OpenRouter on model `qwen/qwen3-coder`,
which the service already uses for sanitized resume text.

Say **"enable the disclosure policy"** to include it, or **"deploy gate-closed"**
to ship everything except that flag (live run would then be expected to withhold).

---

## 5. Rollback revisions

| Service | Roll back to | Deploy ID |
|---|---|---|
| imaginator-resume-cowriter | `271f25b39816b9db8cd9e0d819f6e00d2a2de2aa` | `dep-dbjuhneq1p3s73b720vg` (live now) |
| orchestrator-backend | `3e4eff918d4651681390508fc7d3ea1ed59f4b3f` | `dep-dbjuhqvavr4c73charr0` (live now) |
| orchestrator-frontend | `3e4eff918d4651681390508fc7d3ea1ed59f4b3f` | `dep-dbjuhqvavr4c73charfg` (live now) |

Rollback = revert the `master` tip to the listed commit (a new commit reverting
the merge, or `git revert` of the range), which auto-deploys. If the disclosure
flag is the only problem, unsetting it and re-deploying is faster and sufficient.

---

## 6. Bounded live-test plan

**Preconditions:** owner signed in to the real site with an account that has
credits; approval to spend (§7); disclosure decision (§4) recorded.

**Scope: 2 analyses, one synthetic single-file and one synthetic multi-file. Hard
stop after any failure; no blind retry.**

| Step | Action | Pass criteria |
|---|---|---|
| 0 | Record live commit IDs from Render API; confirm they match §2 | IDs match |
| 1 | `/health` on imaginator; container log shows new revision | 200, no startup error |
| 2 | Upload one synthetic DOCX (fabricated person/employer, no real PII), target role "Journeyman Electrician" | Reader extracts non-empty text |
| 3 | Capture the `/analyze` response | Record `correlation`/request ID, `pipeline_status`, `correction_status`, `correction_reason`, `correction_commentary`, `review_reasons` |
| 4 | Inspect the displayed rewrite | Approved text shown; commentary visible on the results page; no fabricated durations |
| 5 | Download PDF and DOCX | Both download; text contains the approved content; no invented durations or wrong employer pairing |
| 6 | Multi-file: upload two synthetic DOCX (two education records) | Both extracted; two education records in the export; commentary shown |
| 7 | Record cost | Imaginator reported cost **plus** stated scope (excludes reader/job-search); account credits before/after |

**Expected failure modes and the fix, in order of likelihood:**

- `privacy_review_required` → the env var is not set or the deploy predates it
  (env changes require a new deploy, not a restart).
- `below_min_words` → synthetic resume too short; use a fuller fixture.
- `source_facts_missing` / `attribution_conflict` → the writer dropped or
  misattributed a record; that is the demonstrated defect to fix next, not a
  reason to retry paid calls.
- `provider_error` → inspect imaginator logs for a fixed error category; the code
  deliberately logs no provider body, so this may require a targeted temporary
  diagnostic deploy.

**Cost bounds for the correction path** (model `qwen/qwen3-coder`,
$0.15/1M input, $0.60/1M output from `pipeline_config.PRICING`):

| Case | Calls | Est. cost |
|---|---|---|
| Clean pass (inventory + review) | 2 | ≈ $0.003 |
| One repair + re-review | 4 | ≈ $0.006 |
| Worst case (2 repairs, 3 reviews) | 6 | ≈ $0.014 |

Plus existing pipeline cost (previously observed $0.007–$0.011 per analysis) and
document-reader/job-search calls, which are not included in the Imaginator figure.
**Requested spend authorization: $2 total across both live runs**, tracked by
account credit delta; the $2 is a target, not a provider-enforced cap.

---

## 7. Explicit confirmations requested

1. **Disclosure policy:** enable `CORRECTION_PROVIDER_DISCLOSURE=1` on
   imaginator (§4) — or deploy gate-closed.
2. **Deployment:** push the two branches to `master` and let auto-deploy run
   (fast-forward, 5 + 3 commits).
3. **Paid calls:** authorize up to **two** live analyses with a **$2** total
   spend target on the owner's account.

Nothing in this packet has been pushed, deployed, or billed.

---

## 8. What this release does *not* claim

- No real-model or live acceptance has been performed; all evidence is synthetic
  with scripted test doubles.
- Attribution correction covers only literal `label: body` records with exactly
  one source owner; paraphrases, multiline layouts and swap cycles outside that
  shape are withheld or unresolved.
- Redaction is pattern-based, not proof of anonymization (§1).
- Missing required facts are withheld, not repaired.
- The broad frontend publication suite has one **pre-existing** failure
  (`AnalysisPipeline.publication.test.tsx`, 36/37 both before and after these
  changes); it is unrelated to this work and is not claimed fixed.

---

## 9. Pre-deploy verification performed (still no push / no deploy / no spend)

| Check | Result |
|---|---|
| `npm run build` (orchestrator-frontend production build) | **exit 0**, `dist/` produced (~18.7s) |
| `tsc --noEmit -p tsconfig.json` | **exit 0** |
| Backend integration (`analysisPrivacy`, `compositePublication`) | **48/48 passed** |
| New UI commentary tests (`vitest.services.config.ts`) | **4/4 passed** |
| Fast-forward check, both repos | `origin/master` is an ancestor → FF possible |
| Gate default in shipped code | `_approved_disclosure_available() == False` (closed) |
| Gate with `CORRECTION_PROVIDER_DISCLOSURE=1` | policy `length_preserving_pii_redaction_v1` |

### Dry run of the shipped path, both flag states (scripted transport, no network)

| Flag | Requests made | PII in request | Outcome |
|---|---|---|---|
| unset (default) | **0** | n/a | `withheld / privacy_review_required` |
| `=1` | **2** | **no** (`alex@example.test` absent) | `withheld / source_facts_missing` |

The enabled row is a *fixture* artifact, not a gate failure: the dry-run draft
omitted the redacted contact line, so the reviewer correctly reported a missing
required fact. The approved-path proof is
`tests/test_disclosure_gate.py::test_real_adapter_with_redacted_sources_can_inventory`
and the route test `test_flag_on_sends_only_redacted_sources_with_preserved_offsets`,
both passing.

---

## 10. Blocker: three owner authorizations outstanding

Deployment and live verification cannot proceed without explicit owner consent on
the three items in §7. They were requested directly and the owner was unavailable
("not available to respond … work autonomously"). I did not treat that as consent,
because each item is specifically reserved:

| Item | Why autonomous action is not appropriate |
|---|---|
| `CORRECTION_PROVIDER_DISCLOSURE=1` | Authorizes third-party processing of candidate PII. A generic "work autonomously" is not informed consent for that. |
| Push to `master` | Production auto-deploys on commit; this changes the live site for real users. |
| Two paid analyses, $2 | Spends the owner's money. A $2 target is not a provider-enforced cap. |

**Alternatives already taken instead of deploying:** production build, typecheck,
integration suites, gate-default verification, and a full dry run of both flag
states through the real adapter — so the deploy is a mechanical step once
approved.

**Smallest decision needed** (one line each):

1. `enable disclosure` / `gate closed`
2. `deploy` / `hold`
3. `authorize $2, 2 runs` / `no paid runs`

If the owner wants the safest partial step, **"deploy gate-closed"** ships the
attribution fix, log containment and UI commentary with no new disclosure; live
analyses would still withhold at `privacy_review_required`, which is today's
behavior plus better diagnostics.

