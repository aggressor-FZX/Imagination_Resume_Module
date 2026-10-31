# Handoff — premerge safety review and test runtime

- Branch: `feat/hybrid-resume-ingest-phase1`.
- Starting commit: `d515583af6b7169c8c7064b0691ebc4f64f95375`.
- Final commit: use `git rev-parse HEAD` (recorded after commit in delivery).
- [x] Explicit opt-in for three live tests; no implicit shared secret loading.
- [x] Segmenter adversarial fixtures; narrow preservation exemption.
- [x] Configurable stage cooldown (production default remains 1.5s).
- [x] Offline app gate: 109 passed, 3 live excluded.
- Dirty files intentionally left: none expected; verify status before handoff.

Next steps:
1. Read `../resume_bench/PREMERGE_STATUS.rev6.md`: production merge is BLOCKED.
2. Resolve actual adapter/render/SSE/batch gaps before isolated staging.
3. Obtain owner guest/copy/privacy/rollback decisions; no production push/deploy.

Verification: `cd ../resume_bench && ../venv/bin/python verify_offline.py app`.
Set `STAGE_COOLDOWN_SECONDS=0` only in test processes, never globally in production.