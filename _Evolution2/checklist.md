# Evolution2 Checklist

- [ ] Implement startup provider key validation
- [ ] Add `/keys/health` endpoint with provider readiness
- [ ] Remove BYOK headers and update tests
- [ ] Define FastSVM/Hermes interfaces (feature-flagged)
- [ ] Implement confidence scoring mapped to `ProcessedSkills`
- [ ] Validate output via `AnalysisResponse` in assembly path
- [ ] Standardize async mocks in tests; expand integration coverage
- [ ] Add `ruff` and `mypy` to CI and run locally
- [ ] Optimize prompts and add latency/cost tracking
- [ ] Update API docs and system spec to reflect changes

## Acceptance Criteria
- Green test suite including new integration tests
- `/keys/health` reflects accurate status and degrades gracefully
- Confidence buckets populated deterministically in unit tests
- CI runs lint/typecheck without errors
- Documentation updated (API reference, system spec)
