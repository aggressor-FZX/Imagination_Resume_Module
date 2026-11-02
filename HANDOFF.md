# Handoff — live capture ready, awaiting allowance (2026-11-02, session 3)

- **Live:** `master` = `11dad31`, Imaginator deploy `dep-dbke2ecs728c73col0l0`.
- **Tests:** 703 passed, 3 skipped.
- **Reports:** `docs/CORRECTION_BOUNDARY_ACCEPTANCE_2026-11-02.md`,
  `docs/LIVE_CAPTURE_RUNBOOK.md`.

## Corrected from the previous session

The two earlier recordings (`captures/live-rewrite-0{1,2}.json`) are **real-provider
probes driven by constructed synthetic drafts**, produced against the correction engine
directly. They are **not** captures of the failed website runs, which were never
recorded. Their unjudged-lines finding is a *candidate* mechanism for the
`malformed_output` reason on website runs 2 and 3, not a demonstrated cause. This is now
stated in the report, the test docstrings and the test names.

## Done this session

- [x] Live-path capture (`live_capture.py`): records the real source, the **writer's own
      draft**, every outgoing request verbatim, every provider response body, the
      normalized review, the gate decision, timings and the correlation id. Verified on
      the deployed service: module present, `enabled: False` by default.
- [x] Access controls, each tested: opt-in only; refuses unmarked sources; refuses an
      unignored repository directory; `0600` in `0700`; summaries keep only
      identifier-shaped strings and numbers; never alters a response; a write failure
      cannot break a request.
- [x] `live_capture.replay()` re-derives the decision offline from the recorded draft.
      Incomplete recordings surface as `ReplayExhausted`, not empty success.
- [x] Bounded recovery for `incomplete_review`: one completion re-ask naming the omitted
      lines; at most once; never after a repair; never past the deadline; never a blind
      retry. Unchanged completeness and factual checks still apply (both proved).
- [x] Preflight tool that verifies capture readiness before a paid run; confirmed on the
      live service via SSH (`python tools/live_capture_preflight.py`).
- [x] Evidence labels corrected in docs, tests and commit messages.

## Remaining — blocked on a specific decision

1. **Additional live-test allowance is required.** The six-run allowance is exhausted
   (1265 credits remain). I requested a narrowly scoped renewal and did **not** treat it
   as granted; no live run was made this session, so live delivery is still unproven.
   Proposed: 2 single-file runs (66 credits) with capture on.
2. **Enable capture before the run** (requires a deploy, not a restart):
   `CORRECTION_LIVE_CAPTURE=1`, `CORRECTION_LIVE_CAPTURE_DIR=/app/data/live-capture`
   (the service has a 1 GB disk at `/app/data`). Run the preflight first.
3. **If the run fails:** replay the exact capture offline before choosing a fix. Do not
   choose a fix from a probe.
4. **If it approves:** verify the displayed commentary, the receipt, and the actual
   bytes of the downloaded PDF/DOCX.
5. Known gaps: invented `label: body` employer not deterministically caught; review
   latency may still exceed budget on large documents; capture covers correction only.

## Verification

```bash
cd Imagination_Resume_Module
../.venv/bin/python -m pytest -o addopts="" -q        # 703 passed, 3 skipped
PYTHONPATH=. python tools/live_capture_preflight.py  # 0 ready / 2 not ready / 3 disabled
```

## Previous session (superseded)
# Handoff — correction timeout attribution (2026-11-02)

- **Active branch:** `wip/correction-timeout-attribution` (local, not pushed); base `10b20d8` on `master`.
- **Last commit:** see `git log -1` on that branch. Do not push without approval: `master` auto-deploys.

## Completed
- [x] Nonbillable diagnosis of the two live runs from Render logs (explicit UTC windows, paginated).
- [x] Phase-attributed deadline vs. per-request timeout reasons, verified through the real `/analyze` route.
- [x] Budget = min(120s, 180s upstream − 20s reserve − pipeline elapsed); withheld with no provider call under 30s.
- [x] Sanitized `correction_event` phase events keyed by the orchestrator `x-correlation-id`; no document text.
- [x] Diagnosis and approval-gated live-run plan: `docs/CORRECTION_TIMEOUT_DIAGNOSIS_2026-11-02.md`.

## Remaining / next steps (need owner approval)
1. Approve deploying the branch to Imaginator (auto-deploy from `master`).
2. Approve one paid synthetic run per the plan in section 10 of the diagnosis doc. Stop after one.
3. Decide the output-allowance fix (size gate, per-source chunking, or local quote derivation). Do not just raise `max_tokens`.
4. Login backend: `mlAuthMiddleware` retries timed-out `POST /analyze` (JSON and single-file paths). Recommend no retry on timeout.
5. Disclosure gate still scans source documents only, not draft lines or claims (disclosed limitation).

## Uncommitted/dirty files
- None intentionally. `secure_storage.json.encrypted` in Job_searcher is untouched and out of scope.

## Verification command
```bash
../.venv/bin/python -m pytest tests/test_correction_timeout_attribution.py tests/test_source_correction.py tests/test_analyze_source_correction_route.py -p no:cacheprovider -q
```
Full suite: 584 passed, 13 failed and 1 collection error, all `No module named 'docx'` in this venv (environment).

---

# Handoff — premerge safety review and test runtime

## 2026-11-01 local acceptance reached (uncommitted, still not deployable)

- Branch `feat/hybrid-resume-ingest-phase1`, base `65198c9`; all changes below
  remain uncommitted. No push, deploy or provider call was made.
- [x] Wrong-employer attribution is now **corrected**, not merely rejected, when
  an explicit `label: body` record has exactly one source owner. Ambiguous
  multi-owner bodies and label rewrite cycles are withheld as
  `attribution_conflict`. A correct-label control still passes.
- [x] Evidence regenerated after the final code change: engine generator exit 0
  (6/6 cases), route generator exit 0 (8 cases, 16 JSON+SSE requests, 411
  assertions, 0 failed), full Python selection exit 0 (**340 passed**).
- [x] Real `/resume/export` verified inside the route evidence: `pdflatex` PDF
  and `python-docx` DOCX both contained the corrected content for all three
  approved cases; the negative control produced no receipt/export.
- [x] Inspectable bundle: `tests/fixtures/correction_acceptance/` — synthetic
  sources, answer key, drafts, scripted decisions, route responses, export text,
  expected-vs-actual checks, code SHA-256 binding. Commands, exit codes and the
  PASS/NOT RUN matrix are in `CORRECTION_EVIDENCE.md`.
- [x] Job Search storage isolation fixed: `JOB_SEARCH_SECURE_STORAGE_PATH` plus a
  conftest redirect to a disposable temp path; `tests/test_analysis_log_privacy.py`
  passed 15 and the tracked encrypted file hash was unchanged. The earlier
  modified file is preserved/unstaged for an owner keep-or-restore decision.
- [ ] Not live, not deployed. The provider disclosure gate stays closed, so the
  current production path would still withhold normal results. No real model
  behavior is proven. Credential-like literals remain in source/docs/Git history.
- Next: 1. Owner decision on the preserved encrypted file. 2. Remove
  credential-like literals safely and assess rotation. 3. Improve bounded
  attribution beyond the literal layout (paraphrase/multiline). 4. Resolve
  disclosure policy and receipt identity compatibility. 5. Explicit authorization
  before deploy/paid tests.
- Verification: exact commands and codes in `CORRECTION_EVIDENCE.md`.

## 2026-11-01 offline outcome and storage incident

- Branch `feat/hybrid-resume-ingest-phase1`, base commit `65198c9`; changes
	remain uncommitted, including an existing credential-bearing source file that
	must not be staged until the literal is removed safely. No push or deploy.
- [x] Original valid-citation wrong-employer fixture now withholds on the
	narrow explicit-label guard; positive control passes. Six engine cases and
	16 actual in-process JSON/SSE route runs pass with scripted doubles. Artifacts
	and exact commands: `tests/fixtures/correction_acceptance/README.md` and
	`CORRECTION_EVIDENCE.md`. Combined Python selection 337 passed.
- [x] Selected Python and Node provider/parser/proxy/error logs are sanitized;
	serial Node suites 51 and 79 passed. See evidence for tested boundaries,
	first failures, and why neither logs nor semantic accuracy are universally safe.
- [ ] A Job Search privacy test unexpectedly changed tracked
	`../Job_searcher/job_search_api/secure_storage.json.encrypted`. Its pre-test
	state is not independently known. Do not read, stage, or restore it without
	owner direction. Additional Job Search log changes need isolated testing that
	cannot initialize or overwrite secure storage.
- Next: 1. Obtain owner direction on restoring or preserving the encrypted file.
	2. Remove credential-like literals safely without exposing values; owner must
	independently assess revocation/rotation. 3. Rerun Job Search tests with a
	proven disposable storage isolation. 4. Review literal attribution limits,
	correction disclosure design, identity/export fidelity, and full legacy paths.
	5. Do not deploy or make paid calls without explicit later authorization.
- Uncommitted: this repository's listed source/tests/evidence; Login has both
	this work and unrelated pre-existing docs/handoff edits; Job Search has the
	encrypted file, source WIP, and pre-existing `.github/`; preserve all.
- Verification: exact 337-test offline command in `CORRECTION_EVIDENCE.md`.

## Synthetic actual-route evidence (2026-11-01; local only)

- Branch `feat/hybrid-resume-ingest-phase1`; last observed commit `65198c9`.
- [x] Added `tests/fixtures/correction_acceptance/route_evidence.py` and
	`route_results.json`; updated the fixture README without editing its existing
	generator or correction/app/test sources.
- [x] Eight synthetic cases in JSON and SSE: 16 requests, 314 checks, zero
	unmet. Includes supported and corrected duration, two education records, wrong
	attribution, bounded repairs, post-correction numeric veto and privacy control.
- Next steps:
	1. Re-run after concurrent changes are integrated; hashes bind only selected
		 working-tree files. Review remaining cases in the fixture README.
	2. Separately assess real model, privacy policy, full pipeline and rendering;
		 this synthetic evidence is not production acceptance.
- Intentionally unstaged: the new generator, route artifact, README and this
	shared handoff. The fixture directory and other source/test files were already
	dirty or are concurrent work; preserve them and check `git status --short`.
	No commit, push or deployment; avoid bundling other agents' WIP.
- Verification from this repository: `env -i PATH=/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 /home/skystarved/Render_Dockers/.venv/bin/python tests/fixtures/correction_acceptance/route_evidence.py`

## OpenRouter model-log privacy (2026-11-01; local only)

- Branch `feat/hybrid-resume-ingest-phase1`; last observed commit `65198c9`.
- [x] Removed requested/fallback model IDs from client logs and timeout errors;
	removed adapter model IDs and untrusted provider usage values from its logs.
- [x] Offline synthetic availability, fallback, stream, timeout and adapter
	tests: 16 passed including existing provider-error and streaming tests.
- [ ] Credential-bearing example block remains unchanged. It was inadvertently
	displayed during a file read; treat the credential as exposed, revoke/rotate
	it and assess Git history. Do not paste it into a patch or diff review.
- Next steps:
	1. Owner removes the example with a secret-aware local process, and rotates
		 the credential; address history using an approved remediation procedure.
	2. Review other consumers of model IDs and usage stats: returned data and
		 `get_usage_stats()` still intentionally contain model/provider values.
- Uncommitted by request: `openrouter_safe_client.py`, `llm_client_adapter.py`,
	`tests/test_openrouter_model_log_privacy.py`, and this handoff. Preserve all
	existing/concurrent WIP; check `git status --short`. No commit or deploy.
- Verification from this repository (network/DNS and dotenv disabled):
	`env -i PATH=/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 STAGE_COOLDOWN_SECONDS=0 /home/skystarved/Render_Dockers/.venv/bin/python -c 'import sys,socket; sys.path.append("/home/skystarved/.local/lib/python3.12/site-packages"); import dotenv; dotenv.load_dotenv=lambda *a,**k: False; from pydantic_settings.sources import DotEnvSettingsSource; DotEnvSettingsSource._read_env_files=lambda self: {}; c=socket.socket.connect; e=socket.socket.connect_ex; socket.socket.connect=lambda self,address: (_ for _ in ()).throw(RuntimeError("network blocked")) if self.family in (socket.AF_INET,socket.AF_INET6) else c(self,address); socket.socket.connect_ex=lambda self,address: (_ for _ in ()).throw(RuntimeError("network blocked")) if self.family in (socket.AF_INET,socket.AF_INET6) else e(self,address); socket.getaddrinfo=lambda *a,**k: (_ for _ in ()).throw(RuntimeError("DNS blocked")); import pytest; sys.exit(pytest.main(["-p", "pytest_asyncio.plugin", "-p", "no:cacheprovider", "-o", "addopts=", "-q", "--tb=short", "tests/test_openrouter_model_log_privacy.py", "tests/test_provider_error_privacy.py", "tests/test_drafter_stream.py"]))'`

## Researcher and editor fallback review fixes (2026-11-01; local only)

- Branch `feat/hybrid-resume-ingest-phase1`; last observed commit `65198c9`
	(concurrent work continues). No commit, push, deployment or network calls.
- [x] Unusable editor replies and editor exceptions retain all supplied draft
	roles and bullets in an internal fallback; publication remains blocked.
- [x] Researcher errors, exceptions, fallback flags, precomputed fallbacks and
	the existing unmarked empty-response default shape propagate fallback status.
- [x] Offline real-stage/orchestration/integration and JSON/SSE route tests;
	144 passed, 1 skipped in the focused selection below. Healthy route control
	reaches a stub correction callback; no real model or correction provider used.
- Next steps:
	1. Review the Researcher's unmarked default signature and replace the
		 compatibility detection with an explicit flag when its contract can change.
	2. Review remaining privacy/semantic concerns and rerun after concurrent work;
		 do not deploy without separate approval.
- Intentionally unstaged for the no-commit request: `orchestrator.py`,
	`stages/star_editor.py`, `tests/test_normal_path_privacy_sentinels.py`,
	`tests/test_research_editor_fallback_regressions.py` and this handoff.
	Existing/concurrent WIP: `app.py`, `imaginator_new_integration.py`,
	`logging_setup.py`, `source_correction.py`, `stages/drafter.py`,
	`stages/researcher.py`, `tests/fixtures/correction_acceptance/`,
	`tests/test_analyze_log_privacy.py`, `tests/test_network_log_privacy.py`,
	`tests/test_source_correction_attribution_guard.py`; check status again.
- Verification from this repository (network/DNS and dotenv disabled):
	`env -i PATH=/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 STAGE_COOLDOWN_SECONDS=0 /home/skystarved/Render_Dockers/.venv/bin/python -c 'import sys,socket; sys.path.append("/home/skystarved/.local/lib/python3.12/site-packages"); import dotenv; dotenv.load_dotenv=lambda *a,**k: False; from pydantic_settings.sources import DotEnvSettingsSource; DotEnvSettingsSource._read_env_files=lambda self: {}; c=socket.socket.connect; e=socket.socket.connect_ex; socket.socket.connect=lambda self,address: (_ for _ in ()).throw(RuntimeError("network blocked")) if self.family in (socket.AF_INET,socket.AF_INET6) else c(self,address); socket.socket.connect_ex=lambda self,address: (_ for _ in ()).throw(RuntimeError("network blocked")) if self.family in (socket.AF_INET,socket.AF_INET6) else e(self,address); socket.getaddrinfo=lambda *a,**k: (_ for _ in ()).throw(RuntimeError("DNS blocked")); import pytest; sys.exit(pytest.main(["-p", "pytest_asyncio.plugin", "-p", "no:cacheprovider", "-o", "addopts=", "-q", "--tb=short", "tests/test_research_editor_fallback_regressions.py", "tests/test_normal_path_privacy_sentinels.py", "tests/test_fallback_and_experience_containment.py", "tests/test_editor_format_preservation.py", "tests/test_editor_render_heading.py", "tests/test_stage_cooldown.py", "tests/test_logic_mocks.py", "tests/test_analyze_source_correction_route.py", "tests/test_analyze_log_privacy.py", "tests/test_provider_error_privacy.py", "tests/test_publication_sse.py"]))'`

## Network logging privacy (2026-11-01; local only)

- Branch `feat/hybrid-resume-ingest-phase1`; last observed commit `65198c9`;
	check `git log -1` for concurrent changes.
- [x] Configure `httpx`, `httpcore`, `urllib3`, and `aiohttp.client` at
	WARNING; unrelated application logger configuration remains unchanged.
- [x] Mock transport tests reproduce the unsafe HTTPX INFO event when enabled,
	and prove sync HTTPX and the actual async O*NET keyword helper sent synthetic
	query values while configured logs omitted them. No external traffic.
- [x] Focused offline verification: 63 passed; scoped `git diff --check` passed.
- Limits: WARNING+ is not sanitized; explicit child logger levels, existing
	handlers, later reconfiguration, other loggers, application logs, third-party
	infrastructure, and actual request payloads are outside this protection.
- Next steps:
	1. Review warnings/errors and other provider/network logs separately.
	2. Rerun focused verification after integration; do not deploy without approval.
- Uncommitted intentionally: `logging_setup.py`,
	`tests/test_network_log_privacy.py`, and this shared handoff. Preserve other
	concurrent WIP shown by `git status --short`; no commit or push requested.
- Verification from this repo (dotenv, network sockets, DNS blocked):
	`env -i PATH=/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 STAGE_COOLDOWN_SECONDS=0 /home/skystarved/Render_Dockers/.venv/bin/python -c 'import sys,socket; sys.path.append("/home/skystarved/.local/lib/python3.12/site-packages"); import dotenv; dotenv.load_dotenv=lambda *a,**k: False; from pydantic_settings.sources import DotEnvSettingsSource; DotEnvSettingsSource._read_env_files=lambda self: {}; c=socket.socket.connect; e=socket.socket.connect_ex; socket.socket.connect=lambda self,address: (_ for _ in ()).throw(RuntimeError("network blocked")) if self.family in (socket.AF_INET,socket.AF_INET6) else c(self,address); socket.socket.connect_ex=lambda self,address: (_ for _ in ()).throw(RuntimeError("network blocked")) if self.family in (socket.AF_INET,socket.AF_INET6) else e(self,address); socket.getaddrinfo=lambda *a,**k: (_ for _ in ()).throw(RuntimeError("DNS blocked")); import pytest; sys.exit(pytest.main(["-p", "pytest_asyncio.plugin", "-p", "no:cacheprovider", "-o", "addopts=", "-q", "--tb=short", "tests/test_network_log_privacy.py", "tests/test_analyze_log_privacy.py", "tests/test_analyze_source_correction_route.py", "tests/test_provider_error_privacy.py", "tests/test_publication_sse.py"]))'`

## Attribution-conflict route tests (2026-11-01; local only)

- Branch `feat/hybrid-resume-ingest-phase1`; test commit `65198c9`.
	Check `git log -1` for the current commit; concurrent work is in progress.
- [x] Added `tests/test_analyze_attribution_conflict.py` for real `/analyze`
	JSON and SSE, exercising actual `correct` and `CorrectionProvider` with
	scripted in-memory transport. Initial and post-repair conflicts, approved
	positive controls, and the default closed privacy boundary are covered.
- [x] Isolated offline run: 44 passed, 4 failed. All four failures are the
	expected route mapping `attribution_conflict` to `correction_failed`; the
	actual guard and privacy/withholding assertions pass before that assertion.
	The synthetic opt-in is not a production disclosure policy or semantic proof.
- Next steps:
	1. Main separately adds `attribution_conflict` to the route's allowed reasons.
	2. Rerun the exact command below, including the SSE reason assertions; do not
		 deploy or perform paid calls without separate authorization.
- Dirty files: preserve concurrent pre-existing `app.py`, `source_correction.py`,
	`imaginator_new_integration.py`, `logging_setup.py`, `orchestrator.py`,
	`stages/drafter.py`, `stages/researcher.py`, `stages/star_editor.py`,
	`tests/fixtures/correction_acceptance/`, `tests/test_analyze_log_privacy.py`,
	`tests/test_network_log_privacy.py`, `tests/test_normal_path_privacy_sentinels.py`,
	`tests/test_research_editor_fallback_regressions.py` and
	`tests/test_source_correction_attribution_guard.py`; check `git status --short`
	for ongoing changes. This shared `HANDOFF.md` remains unstaged to avoid
	committing other agents' work.
- Verification (from this repository, network/DNS and dotenv disabled):
	`env -i PATH=/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 STAGE_COOLDOWN_SECONDS=0 /home/skystarved/Render_Dockers/.venv/bin/python -c 'import sys,socket; sys.path.append("/home/skystarved/.local/lib/python3.12/site-packages"); import dotenv; dotenv.load_dotenv=lambda *a,**k: False; from pydantic_settings.sources import DotEnvSettingsSource; DotEnvSettingsSource._read_env_files=lambda self: {}; c=socket.socket.connect; e=socket.socket.connect_ex; socket.socket.connect=lambda self,address: (_ for _ in ()).throw(RuntimeError("network blocked")) if self.family in (socket.AF_INET,socket.AF_INET6) else c(self,address); socket.socket.connect_ex=lambda self,address: (_ for _ in ()).throw(RuntimeError("network blocked")) if self.family in (socket.AF_INET,socket.AF_INET6) else e(self,address); socket.getaddrinfo=lambda *a,**k: (_ for _ in ()).throw(RuntimeError("DNS blocked")); import pytest; sys.exit(pytest.main(["-p", "pytest_asyncio.plugin", "-p", "no:cacheprovider", "-o", "addopts=", "-q", "--tb=short", "--show-capture=no", "tests/test_analyze_attribution_conflict.py", "tests/test_correction_privacy_hardening.py", "tests/test_source_correction_attribution_guard.py"]))'`

## Normal-path privacy containment (local only)

- Branch `feat/hybrid-resume-ingest-phase1`; last commit `a2a93d1`.
- [x] Scoped normal-pipeline logs and propagated errors to fixed messages; added
	synthetic offline sentinels. The focused selection below passed 84 tests.
- Next steps:
	1. Review the remaining privacy risks in `app.py`, upstream callers, providers,
		 third-party modules and streaming; this is not a complete privacy boundary.
	2. Review fallback classification and real-model behavior before deployment.
- Uncommitted: `orchestrator.py`, `imaginator_new_integration.py`,
	`stages/star_editor.py`, `stages/drafter.py`, `stages/researcher.py`,
	`tests/test_normal_path_privacy_sentinels.py`, and this handoff; intentionally
	not committed. Concurrent unrelated `source_correction.py` and correction
	fixture/test changes are preserved; inspect `git status --short`.
- Verification (from this repository; isolated environment, dotenv and network
	disabled):
	`env -i PATH=/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 STAGE_COOLDOWN_SECONDS=0 /home/skystarved/Render_Dockers/.venv/bin/python -c 'import sys,socket; sys.path.append("/home/skystarved/.local/lib/python3.12/site-packages"); import dotenv; dotenv.load_dotenv=lambda *a,**k: False; from pydantic_settings.sources import DotEnvSettingsSource; DotEnvSettingsSource._read_env_files=lambda self: {}; c=socket.socket.connect; e=socket.socket.connect_ex; socket.socket.connect=lambda self,address: (_ for _ in ()).throw(RuntimeError("network blocked")) if self.family in (socket.AF_INET,socket.AF_INET6) else c(self,address); socket.socket.connect_ex=lambda self,address: (_ for _ in ()).throw(RuntimeError("network blocked")) if self.family in (socket.AF_INET,socket.AF_INET6) else e(self,address); socket.getaddrinfo=lambda *a,**k: (_ for _ in ()).throw(RuntimeError("DNS blocked")); import pytest; sys.exit(pytest.main(["-p", "pytest_asyncio.plugin", "-p", "no:cacheprovider", "-o", "addopts=", "-q", "--tb=short", "tests/test_normal_path_privacy_sentinels.py", "tests/test_provider_error_privacy.py", "tests/test_fallback_and_experience_containment.py", "tests/test_editor_format_preservation.py", "tests/test_editor_render_heading.py", "tests/test_drafter_stream.py", "tests/test_stage_cooldown.py", "tests/test_analyze_source_correction_route.py"]))'`

## Current continuation — local only, do not deploy

- Branch `feat/hybrid-resume-ingest-phase1`; base `c879a24`. Check `git log -1`
	for the current continuation WIP commit. No push or paid calls.
- [x] Partially contained provider, adapter and stage parse-error logging; added
	synthetic privacy and correction controls. Offline isolated selection passed
	147 tests; see `CORRECTION_EVIDENCE.md` for exact command and limitations.
- [ ] Correction privacy boundary remains closed; do not deploy this WIP. The
	wrong-attribution characterization explicitly demonstrates that valid citations
	do not establish semantic correctness.
- Next steps:
	1. Review and contain remaining Node proxy, normal pipeline and legacy raw or
		 exception logging; synthetic tests must cover error and streaming paths.
	2. Develop independently anchored source relationships, missing-fact repair,
		 safe disclosure policy and correction explanations without weakening gates.
	3. Complete renderer identity/fidelity, real-model evaluation and synthetic
		 artifact bundle; request explicit authorization before deploy/live tests.
	4. Owner should assess and rotate/revoke the exposed credential-like literal if
		 valid; its value and validity were not checked. Account for Git history.
- Dirty files: check `git status --short`; Login's pre-existing handoff/docs edits
	and the local non-Git e2e runner are intentionally preserved, not staged.
- Verification command: the exact isolated command in `CORRECTION_EVIDENCE.md`
	under “Additional offline work”.

## Current correction WIP — do not deploy

- Active branch `feat/hybrid-resume-ingest-phase1`; pre-WIP commit
	`271f25b39816b9db8cd9e0d819f6e00d2a2de2aa`; check `git log -1` for the
	local WIP commit. See `../resume_bench/HANDOFF.md` for the current plan.
- [x] Local correction, receipt/export and containment work has combined
	synthetic/offline coverage: 282 passed. See `CORRECTION_EVIDENCE.md` for
	the exact command and what was doubled.
- [ ] Privacy boundary deliberately prevents external correction; do not deploy.
	Missing source facts, semantic ownership, actual render equivalence, contact
	identity, privacy-safe model inputs, full pipeline accounting and exact
	correction commentary remain incomplete. Tests are not provider acceptance.
- Next: complete the numbered steps in the harness handoff; obtain explicit
	deployment and paid-test authorization only after local acceptance.
- Dirty files: source changes and new synthetic tests are WIP; inspect Git
	status after the local WIP commit. Do not commit local environment secrets.

## Exact receipt binding (2026-11-01)

- Branch: `feat/hybrid-resume-ingest-phase1`; last commit: `271f25b`.
- [x] Version 4 binds exact UTF-8 Markdown and the complete parsed renderer
	object, including identity and contact fields. Version 3 requires reanalysis.
- [x] Focused offline verification: 201 passed; `git diff --check` passed.
- Next steps:
	1. Review the compatibility impact: the current analysis issuer leaves profile
		 identity empty and late profile additions now invalidate the receipt.
	2. Review actual PDF/DOCX fidelity separately; the receipt only binds renderer input.
- Uncommitted intentionally: `publication_gate.py`,
	`tests/test_receipt_v4_exact_binding.py`, and this handoff; no commit or deploy.
	Pre-existing dirty: `app.py`, `imaginator_new_integration.py`, `models.py`,
	`orchestrator.py`, `resume_builder/build_resume_from_export.py`,
	`CORRECTION_BASELINE.md`, `correction_provider.py`, `source_correction.py`,
	`tests/test_analyze_source_correction_route.py`, `tests/test_correction_integration_offline.py`,
	`tests/test_correction_privacy_hardening.py`, `tests/test_export_isolation_and_parity.py`,
	`tests/test_export_receipt_title_binding.py`, `tests/test_fallback_and_experience_containment.py`,
	and `tests/test_source_correction.py`; left untouched.
- Verification command, from this repository (network and dotenv disabled):
	`env -i PATH=/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 STAGE_COOLDOWN_SECONDS=0 /home/skystarved/Render_Dockers/.venv/bin/python -c 'import sys,socket; sys.path.append("/home/skystarved/.local/lib/python3.12/site-packages"); import dotenv; dotenv.load_dotenv=lambda *a,**k: False; from pydantic_settings.sources import DotEnvSettingsSource; DotEnvSettingsSource._read_env_files=lambda self: {}; c=socket.socket.connect; e=socket.socket.connect_ex; socket.socket.connect=lambda self,address: (_ for _ in ()).throw(RuntimeError("network blocked")) if self.family in (socket.AF_INET,socket.AF_INET6) else c(self,address); socket.socket.connect_ex=lambda self,address: (_ for _ in ()).throw(RuntimeError("network blocked")) if self.family in (socket.AF_INET,socket.AF_INET6) else e(self,address); socket.getaddrinfo=lambda *a,**k: (_ for _ in ()).throw(RuntimeError("DNS blocked")); import pytest; sys.exit(pytest.main(["-p", "pytest_asyncio.plugin", "-p", "no:cacheprovider", "-o", "addopts=", "-q", "--tb=short", "tests/test_receipt_v4_exact_binding.py", "tests/test_render_approval_contract.py", "tests/test_export_receipt_title_binding.py", "tests/test_publication_gate.py", "tests/test_publication_segments.py", "tests/test_renderer_no_defaults.py", "tests/test_export_isolation_and_parity.py", "tests/test_analyze_source_correction_route.py"]))'`

## Receipt-bound export title fix (2026-11-01)

- Active branch: `feat/hybrid-resume-ingest-phase1`; last commit: `271f25b`
	(`fix(education): preserve parser fields and reject ambiguous claims`).
- [x] Removed the route's unverified outer title override; renderer filename now
	comes from the rendered title. Added offline route-to-renderer regressions.
- [x] Focused verification below: 82 passed; `git diff --check` passed. PDF
	compilation was replaced by a local double; generated LaTeX and DOCX were read.
- Intentionally uncommitted: `app.py`, `resume_builder/build_resume_from_export.py`,
	`tests/test_export_receipt_title_binding.py`, and this handoff (no-commit request).
	Pre-existing untracked `CORRECTION_BASELINE.md` was left untouched.
- Next steps:
	1. Review the isolated diff; do not deploy without separate authorization.
	2. Repeat the verification command below if the branch changes.
- Verification command (from `/tmp`):
	`env -i HOME=/tmp PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 PYTHON_DOTENV_DISABLED=1 STAGE_COOLDOWN_SECONDS=0 /home/skystarved/Render_Dockers/venv/bin/python -m pytest /home/skystarved/Render_Dockers/Imagination_Resume_Module/tests/test_export_receipt_title_binding.py /home/skystarved/Render_Dockers/Imagination_Resume_Module/tests/test_render_approval_contract.py /home/skystarved/Render_Dockers/Imagination_Resume_Module/tests/test_renderer_no_defaults.py /home/skystarved/Render_Dockers/Imagination_Resume_Module/tests/test_sanitize_mojibake.py -o addopts='' -p no:cacheprovider -q --tb=short`

## START HERE — current owner: `../resume_bench/HANDOFF.md`

This file holds Imaginator-side detail. **Current status and remaining work live
in the harness handoff**: `../resume_bench/HANDOFF.md` (§0 documentation map,
§2 remaining tasks). Related documents:

- `../resume_bench/STAGING_RUN.rev9.md` — last staging outcome (stopped before
  spend; all staging resources deleted; acceptance not passed).
- `../resume_bench/BOUNDED_BLOCKERS.rev7.md` — the five bounded defects.
- `../resume_bench/STAGING_APPROVAL.rev8.md` — approved release decisions.
- `../resume_bench/REPRODUCE.md` — offline verification commands.

Branch `feat/hybrid-resume-ingest-phase1`. Reviewed **code** head
`d17a4b67dd3e39dfb81dc53373190751b7ba24a7` (R9 deployed exactly this SHA).
This handoff's own edits are doc-only commits after that SHA; check
`git log --oneline -3` and ignore commits touching only `HANDOFF.md`.
**Do not merge to `master`** — production tracks `master` and would auto-deploy.

Known open item owned by this repository: the completion transport has **no
atomic pre-call attempt/USD reservation** across retries/fallbacks/timeouts
(`openrouter_safe_client.py`, `llm_client_adapter.py`). That is why R9 made zero
paid calls. See §2 of the harness handoff before editing those files.

## Historical: bounded rev7 (supersedes remaining local steps below)

Branch remains `feat/hybrid-resume-ingest-phase1`; base `72a76e4`. Final commit
is recorded after commit in delivery (`git rev-parse HEAD`).

- [x] Final SSE only; raw draft/reasoning callbacks discarded.
- [x] Receipt v3 binds exact renderer career object and unchanged Markdown;
	response includes `approved_export_payload`; no substantive sample defaults.
- [x] Real two-file composite rechecks final numbers/entity headings from input,
	never stale earlier PASS or generated `source_*` echoes.
- [x] Parsing fixtures/independent keys packaged locally; 19 targeted pass.
- [x] Full integration: 190 app passed, 1 new template expectation failed;
	corrected exact test passed; 3 live excluded. Not a repeated all-green full run.
- [x] Final targeted receipt/composite/SSE/privacy matrix: 42 passed.
- Dirty files intentionally left: none expected; verify status at handoff.

Next: read `../resume_bench/BOUNDED_BLOCKERS.rev7.md`; no push/deploy. Owner
decisions: isolated staging, guest export, copy/privacy/counsel approval, rollout
gap and fail-closed rollback owner. v1/v2 approval requires reanalysis.
Verification: bench `verify_offline.py app|harness|render`. Full suites once per
integration; focused corrections only thereafter. Historical evidence follows.

- Branch: `feat/hybrid-resume-ingest-phase1`.
- Starting commit: `d515583af6b7169c8c7064b0691ebc4f64f95375`.
- Final commit: use `git rev-parse HEAD` (recorded after commit in delivery).
- [x] Explicit opt-in for three live tests; no implicit shared secret loading.
- [x] Segmenter adversarial fixtures; narrow preservation exemption.
- [x] Configurable stage cooldown (production default remains 1.5s).
- [x] Offline app gate: 109 passed, 3 live excluded.
- Dirty files for the current bounded SSE task are listed below.

Next steps:
1. Read `../resume_bench/PREMERGE_STATUS.rev6.md`: production merge is BLOCKED.
2. Resolve actual adapter/render/SSE/batch gaps before isolated staging.
3. Obtain owner guest/copy/privacy/rollback decisions; no production push/deploy.

Verification: `cd ../resume_bench && ../venv/bin/python verify_offline.py app`.
Set `STAGE_COOLDOWN_SECONDS=0` only in test processes, never globally in production.

## Bounded blocker 2 — SSE evidence (2026-10-31)

- Active branch: `feat/hybrid-resume-ingest-phase1`; last commit:
	`72a76e49117cf81289a6ee08903e3b8ffbeed223`. No commit/push/deploy performed.
- [x] Executed real FastAPI `/analyze` SSE regressions with stubbed analysis;
	worker-thread callbacks contain an invented `87%`, unapproved rewrite and
	private reasoning, while the final result is approved. Dict and model paths
	are covered; blocked finals emit zero tokens and approved finals remain available.
- [x] Removed raw-token buffering/replay. One `rewrite` event now contains only
	approved `final_written_section_markdown`, followed by the unchanged result.
	Receipt/export/composite regions are untouched.
- Exact command, from `Imagination_Resume_Module`:
	`STAGE_COOLDOWN_SECONDS=0 ../venv/bin/python -m pytest tests/test_publication_sse.py -v --tb=short`.
	Before the app fix: **2 failed, 2 passed in 1.98s**, exit 1; both failures:
	`AssertionError: Final approval leaked an invented draft metric` (`87%`).
	After the fix, identical command: **4 passed in 1.42s**, exit 0.
- Tests use in-process ASGI, deny network connections/DNS, disable dotenv-file
	reads and bypass lifespan/model calls. No model APIs, secret files or broad
	suites were used. `git diff --check` and editor diagnostics passed.
- Intentionally uncommitted: `app.py` (SSE fix),
	`tests/test_publication_sse.py` (regressions), `HANDOFF.md` (this evidence).

Next steps:
1. Main integrates this bounded SSE change alongside its nonoverlapping work.
2. Rerun the exact targeted command after integration. All other rev.6 blockers
	 remain outside this task; this is not production/staging approval.

## Bounded blocker 3 — renderer no-defaults subtask (2026-10-31)

- Active branch: `feat/hybrid-resume-ingest-phase1`; last commit:
	`72a76e49117cf81289a6ee08903e3b8ffbeed223`. No commit/push/deploy performed.
- [x] Removed both substantive sample-resume implementations from
	`resume_builder/build_resume_from_export.py`; the legacy fallback helper
	delegates to structured parsing. Missing identity, contact, headline, summary,
	skills and career data stay empty. Explicit aliases (including scalar
	`user_skills`) remain supported without requiring unrelated fields.
- [x] Removed raw-text-to-experience fallback: canonical structured fields are
	rendered as supplied; markdown parsing is left to Main.
- [x] Empty name/location split to `("", "")`. Explicit role `location` appends
	` -- <location>` to its selected heading and never fills the profile location.
	Titles without location retain their existing selection behavior.
- [x] LaTeX publication rendering now uses escaped supplied publication text,
	not the hardcoded sample. Render API and LaTeX template files are unchanged.
- [x] Added `tests/test_renderer_no_defaults.py` only; no existing tests, app,
	publication gate, canonical hashing or markdown parser changed in this subtask.
- Initial 43-case baseline: **41 failed, 2 passed in 0.45s**. Same suite after
	implementation: **43 passed in 0.27s**. Two additional scalar-alias cases
	bring the final targeted result to **45 passed in 0.23s**.
- Exact verification command, from `Imagination_Resume_Module`:
	`STAGE_COOLDOWN_SECONDS=0 ../venv/bin/python -m pytest tests/test_renderer_no_defaults.py tests/test_sanitize_mojibake.py -o addopts='' -q --tb=short`.
	Final result: **51 passed in 0.20s**. `git diff --check` and editor diagnostics
	passed. Tests cover in-memory LaTeX and temporary DOCX generation, not PDF
	compilation. No model/network calls, secret-file reads or broad suites used
	by these tests.
- Intentionally uncommitted renderer files: `resume_builder/build_resume_from_export.py`
	and `tests/test_renderer_no_defaults.py` (Main integration; no-commit request).
	`HANDOFF.md` contains this required handoff evidence. Pre-existing concurrent
	changes in `app.py`, `tests/test_publication_sse.py` and
	`tests/test_render_approval_contract.py` were left untouched.

Next steps:
1. Main completes canonical career hashing and approved-markdown parsing in its
	 separately owned app/publication-gate work.
2. Rerun the exact renderer verification command and Main's approval-contract
	 tests after integration. This bounded subtask is not overall blocker-3,
	 staging or production approval.

## Bounded blocker 5 — portable parsing fixtures (2026-10-31)

- Active branch: `feat/hybrid-resume-ingest-phase1`; last commit:
	`72a76e49117cf81289a6ee08903e3b8ffbeed223`. No commit/push/deploy.
- [x] Vendored two synthetic profiles and independently authored literal answer
	keys from the RCTB source under `tests/fixtures/parsing_fidelity/`. App tests
	have no sibling-harness import, absolute private paths or external fallback.
	Missing files and malformed keys raise failures, never skips. Existing six
	R7/R8 cases are retained; exact company/date-pair checks were added.
- [x] First ran only the new packaged-availability regression, before vending:
	`../venv/bin/python -m pytest tests/test_phase1_parsing_fidelity.py::test_fidelity_fixture_is_packaged -o addopts='' -q --tb=short`
	from this repo: **4 failed in 0.07s**, exit 1 (all four files absent).
- Full targeted verification command, from this repo:
	`../venv/bin/python -m pytest tests/test_phase1_parsing_fidelity.py -o addopts='' -q --tb=short`.
	First post-vendor run: **1 failed, 18 passed in 1.55s**, exit 1.
	Final run after offline-isolation/diagnostic tightening:
	**1 failed, 18 passed in 1.26s**, exit 1; **zero skips**.
- The remaining failure is the added R8 exact-pair case for the admin profile:
	`Example Office Cooperative` has no structured employer/date block. Parsed
	company/duration pairs are `[('', ''), ('Example Community Services',
	'January 2021 – Present'), ('', ''), ('', '')]`. The original substring-based
	employer check passes because the missing employer survives only as raw text.
	No parser change, weaker assertion, xfail or skip was used to hide this.
- From `/tmp`, using the same absolute Python and test-file paths:
	`/home/skystarved/Render_Dockers/venv/bin/python -m pytest /home/skystarved/Render_Dockers/Imagination_Resume_Module/tests/test_phase1_parsing_fidelity.py -o addopts='' -q --tb=short -k 'fixture or cwd or malformed'`
	yielded **11 passed, 8 deselected in 0.13s**, exit 0. The same command with
	`-k 'test_r7_ or test_r8_dates_are_captured or test_r8_present_is_not_mistaken_for_a_date'`
	yielded **6 passed, 13 deselected in 1.07s**, exit 0 (all original semantics).
- Network sockets/DNS and dotenv-file reads are blocked in this test module;
	inherited model credentials are removed before parser import. No model/API
	calls, private-data reads, secret-file reads or broad suites. Scoped
	`git diff --check` and editor diagnostics passed.
- Intentionally unstaged for Main: `tests/test_phase1_parsing_fidelity.py`,
	`tests/fixtures/parsing_fidelity/` (four data files plus provenance README),
	and this append-only `HANDOFF.md` evidence. Pre-existing concurrent dirty
	`app.py`, `models.py`, `publication_gate.py`, `approved_resume.py`,
	`resume_builder/build_resume_from_export.py`, `tests/test_composite_publication.py`,
	`tests/test_publication_sse.py`, `tests/test_render_approval_contract.py` and
	`tests/test_renderer_no_defaults.py` were not changed by this subtask.

Next steps:
1. Main investigates the structured admin-employer loss in `parse_experiences`
	 (outside this subtask's permitted edit scope), retaining the new exact check.
2. Rerun the full targeted command above. Fixture portability is verified;
	 the complete targeted module is not green and this is not merge approval.

Final verification addendum: after preserving the source profiles' EOF newlines,
static AST comparison confirmed both local profiles and both independent keys
match the synthetic RCTB source (no harness import). The identical full targeted
command above yielded **1 failed, 18 passed in 1.36s**, exit 1, zero skips;
the same admin exact-pair case remains red. Scoped diff/fixture whitespace
checks and editor diagnostics passed.