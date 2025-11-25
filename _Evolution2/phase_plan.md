# Evolution2 Plan

## Objectives
- Stabilize authentication and key management across providers.
- Integrate structured skill processing (FastSVM, Hermes) with confidence scoring.
- Maintain resilience with graceful fallbacks and strict schema validation.
- Strengthen tests and CI; introduce lint/type checks.
- Optimize LLM usage and performance; add observability for costs and latency.

## Workstreams
- Authentication & Keys
  - Startup provider key validation; add key health reporting.
  - Remove BYOK header support and references (deprecated).
  - Plan for key rotation and failure fallback.
- Structured Skill Processing
  - Define interface contracts for FastSVM/Hermes integration.
  - Leverage `skill_adjacency.json` and competency maps for inference.
  - Implement confidence scoring mapped to `ProcessedSkills` buckets.
- Resilience & Schema
  - Ensure `HTTPException` propagation for validation; non-LLM fallback maintained.
  - Validate assembled outputs against `models.AnalysisResponse`.
- Testing & CI
  - Standardize async mocks; add integration tests for new interfaces.
  - Add lint (`ruff`) and type checks (`mypy`) to CI.
- Performance & Observability
  - Reduce prompt size; batch calls where possible.
  - Track `RUN_METRICS` with latency and cost; expose simple metrics.

## Milestones
- Week 1: Auth validation; remove BYOK; test harnesses updated; baseline metrics.
- Week 2: FastSVM/Hermes interface shims and feature flags; confidence scoring.
- Week 3: Performance passes; CI lint/typecheck; documentation updates.

## Dependencies
- Provider SDKs and service endpoints (OpenRouter configured).
- Access to FastSVM/Hermes or mock endpoints for integration tests.

## Risks
- External service instability; mitigate with feature flags and fallbacks.
- Token/cost overruns during integration; mitigate with prompt controls and metrics.

## Code Touchpoints
- `app.py:86–173` `/analyze` endpoint assembly and error handling.
- `imaginator_flow.py:889–917, 960–978, 1020–1036` async flow stages.
- `models.py:123–139` response schema validation target.
