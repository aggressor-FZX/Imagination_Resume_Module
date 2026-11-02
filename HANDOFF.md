# Handoff — premerge safety review and test runtime

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