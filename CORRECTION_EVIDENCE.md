# Correction work in progress — local synthetic evidence

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

Supported-duration retention, year-only dates, job-ad-only requirements and
wrong-employer attribution need explicit independent answer-key regressions.
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