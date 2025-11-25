# Evolution2 Plan

## Objectives
- Stabilize authentication and key management across providers.
- Integrate structured skill processing (FastSVM, Hermes) with confidence scoring.
- Maintain resilience with graceful fallbacks and strict schema validation.
- Strengthen tests and CI; introduce lint/type checks.
- Optimize performance end-to-end: multi-level caching, async parallelism, connection pooling, and LLM prompt/model efficiency with observability for costs and latency.

## System Architecture
- Modules do not interact directly; they receive data from the backend and reply with processed data for simplicity and maintainability.
- Processing Flow:
  1. Backend sends info to Doc Loader.
  2. Doc Loader sends processed data back to Backend.
  3. Backend passes data to SVM (FastSVM).
  4. SVM sends processed data back to Backend.
  5. Backend passes data to Hermes.
  6. Hermes sends processed data back to Backend.
  7. Backend passes data to Imaginator.
  8. Imaginator processes and sends to Job Search API (if applicable).

## Service Contracts
### Doc Loader
- Purpose: File ingestion and extraction (resume/doc parsing)
- Auth: None (internal service)
- Endpoints:
  - POST `/loader/upload` → multipart/form-data file upload
  - POST `/loader/process` → raw text payload
- Payloads:
  - Upload: `{ file: <binary>, metadata: {...} }`
  - Process: `{ text: "resume text here" }`
- Response: Extracted structured data (sections, skills, metadata)
- Notes: No LLM calls, cost = $0

### FastSVM
- Purpose: ML-based skill/title extraction using SVM
- Auth: Internal token (service-to-service)
- Endpoint: POST `/fastsvm/extract`
- Payload:
  ```json
  { "resume_text": "string", "features": ["skills", "titles"] }
  ```
- Response: `{ "skills": [...], "titles": [...] }`
- Notes: No LLM calls, cost = $0

### Hermes
- Purpose: Resume/job text extraction and preprocessing
- Auth: Internal token
- Endpoint: POST `/hermes/parse`
- Payload:
  ```json
  { "document_text": "string", "options": { "normalize": true } }
  ```
- Response: Normalized text, structured fields
- Notes: No LLM calls, cost = $0

### Imaginator
- Purpose: Gap analysis, suggestions, and creative resume/job text generation
- Auth: Bearer token (user-level)
- Endpoints: POST `/imaginator/analyze`, `/imaginator/generate`, `/imaginator/criticize`
- Payload:
  ```json
  { "resume_text": "string", "phase": "analysis|generation|criticism", "preferences": { ... } }
  ```
- Response: LLM-generated suggestions, analysis results
- Models Used: Claude-3-haiku (primary), Qwen-30B (generation), DeepSeek v3.1 (criticism); fallback Claude-3-haiku
- Notes: ~3 calls/file, ~3K tokens → $0.015–$0.045 per file

### Job Search API
- Purpose: Job listings, matching, resume upload, applications
- Auth: OAuth2 Bearer Token
- Endpoints:
  - POST `/api/v1/auth/register`, POST `/api/v1/auth/login`, POST `/api/v1/auth/refresh`
  - GET `/api/v1/jobs/search`, GET `/api/v1/jobs/{job_id}`
  - POST `/api/v1/matching/match`, POST `/api/v1/matching/batch`
  - POST `/api/v1/resume/upload`, POST `/api/v1/resume/process`
  - POST `/api/v1/applications/submit`
- Payloads: JSON and multipart/form-data
- Response: Listings, match scores, structured resume data, application confirmation
- Security: Rate limiting, HTTPS, security headers (CSP/XSS)

## Workstreams
- Authentication & Keys
  - Startup provider key validation; add key health reporting.
  - Remove BYOK header support and references (deprecated).
  - Plan for key rotation and failure fallback.
- Structured Skill Processing
  - Define interface contracts for FastSVM/Hermes integration following the system flow.
  - Leverage `skill_adjacency.json` and competency maps for inference.
  - Implement confidence scoring mapped to `ProcessedSkills` buckets.
- Resilience & Schema
  - Ensure `HTTPException` propagation for validation; non-LLM fallback maintained.
  - Validate assembled outputs against `models.AnalysisResponse`.
- Testing & CI
  - Standardize async mocks; add integration tests for new interfaces.
  - Add lint (`ruff`) and type checks (`mypy`) to CI.
- Performance & Observability
  - Caching: implement multi-level caching (in-memory LRU + Redis) with deterministic cache keys and adaptive TTLs.
  - Async: parallelize independent stages with `asyncio.gather`, maintain ordered assembly for dependent steps.
  - Connectivity: add HTTP client connection pooling for external services.
  - LLM: optimize prompts for token efficiency and route by task complexity via OpenRouter model selection.
  - Metrics: track cache hit rates, response times (p50/p95), token counts, and estimated cost; expose simple metrics endpoint.

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

## Performance Design Details
- Cache Key Strategy: generate deterministic keys from canonicalized inputs (`resume_text`, `job_ad`, `extracted_skills`, `domain_insights`, `confidence_threshold`).
- Adaptive TTLs: longer for high-confidence, shorter for low-confidence outputs; invalidate per-user when needed.
- Connection Pooling: shared `aiohttp.ClientSession` with tuned connector limits; initialize in FastAPI lifespan.
- Metrics & Observability: maintain `RUN_METRICS` with token counts, latency per stage, cache hits/misses; consider `/metrics` or extension of `/health`.

## Compliance Summary
- Auth Types:
  - Internal services (Doc Loader, FastSVM, Hermes) → service tokens
  - Imaginator → Bearer token (user-level)
  - Job Search API → OAuth2 (access/refresh tokens)
- Payloads: JSON or multipart/form-data, structured schemas
- Endpoints: RESTful, versioned (`/api/v1/...` where applicable)
- Costs: Only Imaginator incurs LLM costs; others are ML or extraction-based
- Monitoring: Render MCP tools for metrics/logs; Postgres for token/cost tracking; alerts for anomalies
