# Correction work in progress — local synthetic evidence

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

## Matrix and acceptance gaps

The duration, year-only, job-ad and wrong-employer cases above are scripted
controls, not real-model or complete independent semantic acceptance regressions.
Multi-file education ownership is only partially covered; source spans do not
yet independently prove record-level associations. Missing sections are withheld,
not repaired. Corrected Markdown, PDF and DOCX have not been proven equivalent
end-to-end. Commentary is generic and not an explanation of each correction.
An inspectable complete synthetic input/answer-key/draft/decision/render bundle
is not yet present. No successful real model or live workflow is established.

The v4 receipt binds exact Markdown and renderer input including identity fields;
it rejects v3 receipts and later identity changes. Analysis currently signs an
empty profile header, so frontend identity compatibility needs a deliberate fix.
The parser and renderers can still lose or change content; a receipt is not proof
that the rendered files match the Markdown.

## Combined verification

From this repository, the following local-only targeted selection passed **282
tests**, with two pytest configuration warnings from disabled plugin autoload:

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 STAGE_COOLDOWN_SECONDS=0 ../venv/bin/python -m pytest tests/test_source_correction.py tests/test_correction_integration_offline.py tests/test_analyze_source_correction_route.py tests/test_correction_privacy_hardening.py tests/test_fallback_and_experience_containment.py tests/test_publication_gate.py tests/test_entity_section_boundaries.py tests/test_render_approval_contract.py tests/test_export_receipt_title_binding.py tests/test_receipt_v4_exact_binding.py tests/test_export_isolation_and_parity.py tests/test_publication_sse.py -o addopts='' -q --tb=short`

The Node source-document proxy tests and adjacent composite tests separately
passed 49 cases with loopback doubles and the safe runner. See repository
handoffs for outstanding risks and current revisions. No provider calls or
deployment occurred during this work.