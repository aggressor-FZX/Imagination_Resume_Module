# Evolution2 Checklist

- [x] Implement startup provider key validation
- [x] Add `/keys/health` endpoint with provider readiness
- [x] Remove BYOK headers and update tests
- [x] Define FastSVM/Hermes interfaces (feature-flagged)
- [x] Implement confidence scoring mapped to `ProcessedSkills`
- [x] Validate output via `AnalysisResponse` in assembly path
- [x] Standardize async mocks in tests; expand integration coverage
- [x] Add `ruff` and `mypy` to CI and run locally
- [x] Implement in-memory cache with deterministic cache keys and TTL
- [x] Parallelize independent LLM calls with `asyncio.gather`
- [x] Implement HTTP client connection pooling
- [ ] Optimize LLM prompts and route by complexity via OpenRouter
- [x] Add metrics for stage timings and cache hit flag
- [x] Update API docs and system spec to reflect changes

### Creativity & Quality Enhancements
- [x] Add synthesis step that integrates critique into final written resume section
- [x] Upgrade generation prompts with CAR structure, metrics, and domain vocabulary
- [x] Ensure seniority consistency in tone and scope across generated content
- [x] Prioritize gap analysis and reflect bridging statements in final text
- [x] Add unit tests for synthesis, generation CAR adherence, and gap propagation
- [x] Add integration tests for end-to-end flow including synthesis (mocked LLM)
- [x] Instrument `RUN_METRICS` with synthesis stage timing and flags scaffold
 - [x] Instrument HTTP pool metrics and expose in `run_metrics`

## Acceptance Criteria
- Green test suite including new integration tests
- `/keys/health` reflects accurate status and degrades gracefully
- Confidence buckets populated deterministically in unit tests
- CI runs lint/typecheck without errors
- Documentation updated (API reference, system spec)
