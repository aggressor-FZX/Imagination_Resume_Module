# Correction work in progress — local synthetic evidence

## 2026-11-01 final local acceptance

Unsupported durations and wrong single-owner employer labels are now **corrected**
through the real `/analyze` route, and an unresolved attribution remains a
bounded rejection. All local evidence was regenerated after the final code change.

**Command results (all exit 0, no network or provider):**

| Command | Exit | Result |
| --- | --- | --- |
| Engine generator `tests/fixtures/correction_acceptance/generate.py` | 0 | 6 cases, 0 failed assertions |
| Route generator `tests/fixtures/correction_acceptance/route_evidence.py` | 0 | 8 cases × JSON+SSE = 16 requests, 411 assertions, 0 failed |
| Full Python selection (command below) | 0 | **340 passed** |
| Node unit privacy selection | 0 | 51 passed |
| Node integration proxy selection | 0 | 79 passed |
| Job Search privacy suite `tests/test_analysis_log_privacy.py` | 0 | 15 passed; tracked encrypted file hash unchanged |

Real export ran in the route generator: `/resume/export` returned 200 for all
three approved cases, `pdflatex` compiled the PDF and `python-docx` produced the
DOCX (`pdf_bytes=111461`, `docx_bytes=37278` each). Extracted PDF and DOCX text
contained the approved accomplishment, both employers' own records and both
education records, and did not contain the wrong employer/achievement pairing or
the invented duration. The negative control produced no receipt, export payload
or rendered file. Renderer dependencies are recorded in `route_results.json`
(`/usr/bin/pdflatex`, `/usr/bin/pdftotext`, python-docx 1.1.0).

**Acceptance matrix**

| Criterion | Correction engine | Real route | Real export | Live provider |
| --- | --- | --- | --- | --- |
| Unsupported duration corrected + commentary | PASS | PASS | PASS | NOT RUN |
| Supported duration/facts preserved | PASS | PASS | PASS | NOT RUN |
| Wrong employer corrected, facts preserved | PASS | PASS | PASS | NOT RUN |
| Unresolved attribution bounded rejection | PASS | PASS | PASS (no export) | NOT RUN |
| Two education records, no cross-mixing | PASS | PASS | PASS | NOT RUN |
| Repeated invalid repair bounded stop | PASS | PASS | NOT RUN (withheld) | NOT RUN |
| Post-correction revalidation | PASS | PASS | NOT RUN (withheld) | NOT RUN |

**Code fingerprints** (SHA-256 of the exercised working-tree files; base commit
`65198c95cd524455daf1dc086ae4d1d25f5efbbc`, changes uncommitted):
`source_correction.py f57c4dcd…71dd`, `app.py f9ba7f0c…685b`,
`approved_resume.py 83619633…6ebf6a`, `publication_gate.py 74989f00…7f8b`,
`correction_provider.py 1ed9c91c…8850`, `tier_a_verifier.py 151e134b…8a34`,
`resume_builder/build_resume_from_export.py d884af6b…3371`. Artifact hashes:
`results.json 24c2c8fc…3c65`, `route_results.json 8dd0bd90…bba3`. The route
artifact hash changes on regeneration because receipts embed issue/expiry times
and some IDs; the file's own `working_tree_sha256` block is the authoritative
binding of the exercised code, and `results.json` is deterministic for the same
engine.

**The attribution fix:** a literal `label: body` draft line whose body has
exactly one explicit source owner is repaired to that owner *before review*, so
a lying reviewer never sees the wrong label. A body with two owners, or a set of
labels that rewrite into each other (a swap cycle), is withheld as
`attribution_conflict` instead of guessed. Paraphrased bodies, multiline records
and unknown aliases are not resolved. This is a bounded fix for the observed
counterexample, not general semantic ownership verification.

**Privacy:** selected Imaginator provider/adapter/stage/orchestrator/route
sentinel paths pass, and selected Node proxy/parser/persistence/cost paths pass.
This is not universal log safety, and the correction disclosure gate remains
closed for production. Job Search's selected privacy suite now runs against a
disposable secure-storage path; the tracked encrypted file is preserved and
unstaged pending an owner decision on whether to restore it.

**Encrypted storage (RESOLVED for testing):** an earlier offline Job Search run
unintentionally rewrote the tracked
`Job_searcher/job_search_api/secure_storage.json.encrypted`. Root cause: the
module constructs `secure_storage = SecureStorageService()` at import time using
a path derived from the package directory, so any test importing the app wrote
to the repository file. That file remains preserved and unstaged (SHA-256
`282b3853…b6964`, unchanged before/after the isolated run; its pre-test content
is still not independently known).

Fix applied: `SecureStorageService` now honors
`JOB_SEARCH_SECURE_STORAGE_PATH`, `_save_data` creates a missing owner-only
parent directory, and `tests/conftest.py` sets that variable to a per-session
temp directory **before** importing the app, with an autouse assertion that the
singleton points inside it. Verification:
`cd Job_searcher/job_search_api && .venv/bin/python -m pytest tests/test_analysis_log_privacy.py -o addopts='' -p no:cacheprovider -q`
→ **15 passed**, and the tracked encrypted file hash was identical before and
after. Remaining owner decision: whether to keep or restore the repository's
modified encrypted file to its HEAD version; testing no longer depends on it.

Python command, from this repository:

`env -i PATH=/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 STAGE_COOLDOWN_SECONDS=0 OPENROUTER_API_KEY_1=synthetic-offline-key /home/skystarved/Render_Dockers/.venv/bin/python -c 'import sys,socket; sys.path.append("/home/skystarved/.local/lib/python3.12/site-packages"); import dotenv; dotenv.load_dotenv=lambda *a,**k: False; from pydantic_settings.sources import DotEnvSettingsSource; DotEnvSettingsSource._read_env_files=lambda self: {}; c=socket.socket.connect; e=socket.socket.connect_ex; socket.socket.connect=lambda self,address: (_ for _ in ()).throw(RuntimeError("network blocked")) if self.family in (socket.AF_INET,socket.AF_INET6) else c(self,address); socket.socket.connect_ex=lambda self,address: (_ for _ in ()).throw(RuntimeError("network blocked")) if self.family in (socket.AF_INET,socket.AF_INET6) else e(self,address); socket.getaddrinfo=lambda *a,**k: (_ for _ in ()).throw(RuntimeError("DNS blocked")); import pytest; sys.exit(pytest.main(["-p", "pytest_asyncio.plugin", "-p", "no:cacheprovider", "-o", "addopts=", "-q", "--tb=line", "tests/test_source_correction.py", "tests/test_source_correction_attribution_guard.py", "tests/test_analyze_attribution_conflict.py", "tests/test_provider_error_privacy.py", "tests/test_openrouter_model_log_privacy.py", "tests/test_normal_path_privacy_sentinels.py", "tests/test_analyze_log_privacy.py", "tests/test_network_log_privacy.py", "tests/test_research_editor_fallback_regressions.py", "tests/test_correction_integration_offline.py", "tests/test_analyze_source_correction_route.py", "tests/test_fallback_and_experience_containment.py", "tests/test_publication_sse.py", "tests/test_receipt_v4_exact_binding.py", "tests/test_render_approval_contract.py", "tests/test_export_isolation_and_parity.py"]))'`

From `Login_Landing_page`, these commands each exited **0**, 51 unit and 79
integration tests respectively, with loopback/mocked upstreams:

`scripts/safe-test.sh --runInBand --config jest.consolidated.config.cjs --selectProjects=unit --runTestsByPath backend/tests/unit/correctionPrivacy.test.js backend/tests/unit/correctionServicePrivacy.test.js backend/tests/unit/supportingServicesPrivacy.test.js backend/tests/unit/analysisForkJoin.test.js tests/unit/pineconeGoldenBulletsPrivacy.test.js --silent`

`scripts/safe-test.sh --runInBand --config jest.integration.config.js --runTestsByPath tests/integration/analysisPrivacy.proxy.test.js tests/integration/privateCorrelation.proxy.test.js tests/integration/sourceDocuments.proxy.test.js tests/integration/compositePublication.proxy.test.js --silent`

A wider unit selection initially exited **1**: one new async sentinel test was
fixed and rerun above; outdated `mlAuthMiddleware.test.js` expectations included
five active pre-existing failures and suspended-service cases. The intentionally
changed health error test was updated and focused active checks passed. The
older broad suite is not claimed green.

The original attribution counterexample now has a bounded deterministic fix:
a literal record body with one explicit source owner is relabeled before the
reviewer sees it, while a multi-owner body or a label rewrite cycle is withheld.
A correct-label control still passes. The guard remains lexical and does not
resolve paraphrases or multiline records; matching labels are not proof of
identity. The existing correction disclosure gate remains closed; production
execution of this WIP would withhold normal outcomes. Sentinel tests cover
selected paths, not all service or third-party logging. Job Search changes
beyond its initial offline test were only syntax/isolated checked. A test
unexpectedly modified `Job_searcher/job_search_api/secure_storage.json.encrypted`;
it is preserved and unstaged and needs an owner decision. Its prior state is not
proven. Credential-like literals remain in source/documents and Git history;
their validity was not checked or changed.

## Additional offline work (2026-11-01)

- Selected OpenRouter HTTP/error-envelope and adapter failure paths now avoid
  logging or returning provider bodies and arbitrary exceptions. Drafter and
  Researcher parse failures no longer log raw response previews. This is
  **partial containment**, not a privacy audit of the whole pipeline: candidate
  metadata, stage exceptions, legacy paths, Node proxy/request/error logging and
  other logs still need review and fixes. Existing production is unchanged.
- Scripted synthetic regressions retain a supported six-month metric while
  removing a separate unsupported duration, preserve year-only dates while
  removing fabricated months, and exclude a job-ad-only requirement from the
  candidate source. Scripted wrong-employer and swapped-date cases withhold.
  These test orchestration under explicit evaluator decisions, not detection by
  a real provider. A characterization test demonstrates that an evaluator can
  cite a real source span while approving a wrongly attributed employer claim.
- An existing credential-like literal was found in a client example and other
  local documentation. Validity is unknown. Treat it as exposed: the owner
  should assess and rotate/revoke if valid; do not copy it or push it. Removing
  a working-tree copy alone does not remove prior Git history.
- The selected offline run below passed **147 tests**. It disables network and
  dotenv reads. A first privacy-only run exposed an early-exception logger-scope
  error (9 passed, 1 failed); that error was fixed before the final run.

From this repository, exact command:

`env -i PATH=/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 STAGE_COOLDOWN_SECONDS=0 /home/skystarved/Render_Dockers/.venv/bin/python -c 'import sys,socket; sys.path.append("/home/skystarved/.local/lib/python3.12/site-packages"); import dotenv; dotenv.load_dotenv=lambda *a,**k: False; from pydantic_settings.sources import DotEnvSettingsSource; DotEnvSettingsSource._read_env_files=lambda self: {}; c=socket.socket.connect; e=socket.socket.connect_ex; socket.socket.connect=lambda self,address: (_ for _ in ()).throw(RuntimeError("network blocked")) if self.family in (socket.AF_INET,socket.AF_INET6) else c(self,address); socket.socket.connect_ex=lambda self,address: (_ for _ in ()).throw(RuntimeError("network blocked")) if self.family in (socket.AF_INET,socket.AF_INET6) else e(self,address); socket.getaddrinfo=lambda *a,**k: (_ for _ in ()).throw(RuntimeError("DNS blocked")); import pytest; sys.exit(pytest.main(["-p", "pytest_asyncio.plugin", "-p", "no:cacheprovider", "-o", "addopts=", "-q", "--tb=short", "tests/test_provider_error_privacy.py", "tests/test_source_correction.py", "tests/test_correction_integration_offline.py", "tests/test_analyze_source_correction_route.py", "tests/test_fallback_and_experience_containment.py"]))'`

This is not a provider evaluation or live acceptance. The successful route and
provider tests use synthetic sources and explicit test doubles; their disclosure
override exists only in tests. Actual correction currently stops at
`privacy_review_required` before sending a correction request. Do not deploy.

## Checked locally

- A constructed unsupported `6 months` is patched while a supported `30%` is
  retained, then the final document is reviewed again. The test reviewer is a
  double; it does not prove a real model detects the same issue.
- Invalid reference IDs and spans, incomplete review, repeated invalid changes,
  timeout and malformed provider responses fail closed in local tests.
- Local tests exercise source-document forwarding, receipt alteration checks,
  blocked streaming, isolated export directories and representative renderer
  content. Some renderer tests compile a real PDF; others use a compiler double.
- A model-generated inventory and validated quote offsets remain probabilistic
  semantic interpretations. They are not independent proof of ownership or truth.

## Remaining limits (not acceptance claims)

- The guards are scripted controls, not real-model classification accuracy.
- Education/employer ownership is proven only for the exact literal
  `label: body` layout, a single source owner, and no label cycle. Paraphrases,
  multiline records, unknown aliases and swapped-record cycles outside that
  layout are unresolved and can still be misattributed. The numeric verifier
  remains lexical and does not bind a number to an employer.
- Missing required facts are withheld, not repaired.
- Commentary gives a correction count/kind and requires independent review; it
  does not explain each edit's semantics.
- The v4 receipt binds exact Markdown and renderer input. Identity compatibility
  (analysis signs an empty profile header) and full frontend/legacy exports
  remain open. Real PDF/DOCX rendering passed for this synthetic layout only.
- No real provider or live deployment was exercised.

## Historical verification (superseded by the final matrix above)

The following earlier runs are retained for traceability only. They used the
pre-fix attribution behavior and are not the current acceptance result.

- Earlier 282-test and 337-test selections predate the deterministic attribution
  correction; they are not rerun or claimed here. The current selection above is
  the acceptance record.
- Earlier evidence described swapped-date drafts as withheld
  `source_facts_missing`; with the final guard, a label rewrite cycle is withheld
  as `attribution_conflict`. The current artifacts reflect that behavior.
- Earlier route runs described the wrong-employer reviewer as rejected. The final
  route corrects it to the single owner and publishes with commentary; only the
  genuinely unresolved multi-owner and cycle cases are rejected.
- The pre-test state of `secure_storage.json.encrypted` was never established;
  no claim about its historical contents is made.
