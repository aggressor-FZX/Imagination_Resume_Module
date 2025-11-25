# Caching, Connection Pooling, and Resilient Output Assembly

## Caching and Connection Pooling
- In-memory LRU cache and optional Redis cache reduce repeated work for identical `resume_text` + `job_ad` pairs.
- Deterministic cache keys are generated from canonicalized inputs; TTLs can be adaptive by confidence.
- In single-instance mode, in-memory LRU is sufficient; in multi-instance or horizontally scaled environments, Redis enables shared cache and invalidation.
- Shared `aiohttp.ClientSession` created during FastAPI lifespan reuses TCP connections and keeps a socket pool, decreasing latency and overhead under load.
- Current per-call client usage lives in `imaginator_flow.py:514–530` via `_post_json`; optimizing to a shared session should be done in the lifespan block `app.py:29–43` and reused across module calls.

## Resilience and Output Assembly
- The backend orchestrates modules in sequence (Loader → FastSVM → Hermes → Imaginator), never letting services talk to each other directly.
- Each hop is isolated with clear auth and payload contracts; failures degrade gracefully with safe defaults returned to the caller.
- Orchestrated results are merged and returned as a combined JSON payload, with optional Job Search API enrichment when enabled.
- The combined assembly includes: `analysis`, `generated_text`, `critique`, `final_written_section` (post-critique synthesis), and optionally `jobs`.
- The FastAPI endpoint assembles final `AnalysisResponse` with metrics and status in `app.py:159–170`, preserving a consistent schema for clients and tests.

## Operational Notes
- Cache results for identical inputs to cut LLM cost and latency; monitor hit/miss and p95 latency.
- Initialize a shared `aiohttp.ClientSession` at startup; close it on shutdown to avoid resource leaks.
 - Track per-hop timings and statuses in `RUN_METRICS`; expose a health/keys endpoint for provider readiness.
 - Perform gap analysis (requirements vs. resume) during analysis and reflect prioritized gaps in `suggested_experiences` and the `final_written_section`.
 - Keep modules stateless and backend-orchestrated for simpler debugging, observability, and evolution.
