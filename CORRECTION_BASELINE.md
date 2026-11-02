# Source-aware correction baseline

Reviewed: 2026-11-02. This is a local baseline, not live acceptance evidence.

## Revisions and deployment

| Component | Local commit | Live commit |
|---|---|---|
| Imaginator | `271f25b39816b9db8cd9e0d819f6e00d2a2de2aa` | same |
| Backend/frontend | `3e4eff918d4651681390508fc7d3ea1ed59f4b3f` | same |
| Local-only harness | `b875e5b04e6b7c627cfe444d87ff044397002ab4` | not deployed |

## Actual path

```text
multi-file upload -> document reader -> per-file LLM parser -> /analyze
  -> researcher -> drafter -> editor -> numeric/heading checks
  -> publication veto OR approved Markdown -> render payload + owner receipt
  -> receipt validation -> PDF/DOCX
```

There is no correction or claim-specific feedback on the normal product fast
path. A separate legacy multi-file synthesis path accepts critic feedback but
is not the site's normal multi-file path and does not prove correction.

## Reproduction (no paid calls)

From the Imaginator checkout:

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 STAGE_COOLDOWN_SECONDS=0 ../venv/bin/python -m pytest tests/test_publication_gate.py tests/test_entity_section_boundaries.py tests/test_render_approval_contract.py -o addopts='' -q --tb=short`

Observed: **40 passed**, exit 0; the two pytest configuration warnings are due
to disabled plugin autoload. These test real deterministic functions, but some
route/renderer tests can use local test doubles and are not provider evidence.

Constructed fixture, not replay: original source contains a supported 30%
diagnostic improvement; generated text adds `6 months`. `verify_metrics` gives
`FAIL` for `6 months`, and the publication gate withholds it. The constructed
sample is short and also receives `below_min_words`. The current fast path
makes **zero corrective calls**.

## Confirmed versus unknown

- Confirmed: the deployed fast path has no bounded claim-specific repair; the
  numeric verifier is lexical and can treat a number elsewhere as support.
- Confirmed: per-file education arrays remain parser candidates, not proof of
  degree/school/date ownership. The receipt binds rendered content, not truth.
- Confirmed: live synthetic runs were blocked; no approved live PDF/DOCX exists.
- Unknown: where the observed live `6 months` first entered (writer/editor/
  parsing) and the exact rejected heading. Full output was truncated or
  withheld, so current evidence is not a complete real-output replay.
- Unknown: total provider spend; individual reported model estimates exclude
  other services. No new provider calls were made in this baseline.

Historical provider-derived bench reports were withdrawn; do not reproduce
their private contents or push `resume_bench` Git history.