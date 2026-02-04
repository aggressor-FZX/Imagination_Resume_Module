# Market Intel (Data USA) Enrichment Plan (Imaginator)

## Goal
Populate `domain_insights.market_intel` with localized labor-market signals (workforce growth, wage growth, shortage/bright-outlook signals) when the user provides a target job location.

This enables the frontend to render the Market Intel panel consistently via existing normalization logic.

## Current State (Feb 2026)
- Frontend already supports `domainInsights.market_intel` via normalization and display components.
- `imaginator_flow.py` detects `location` but does not attach any market intel.
- `imaginator_new_integration.py` contains a working reference implementation that:
  - maps location -> Data USA geo id (MSA)
  - queries Data USA time series
  - attaches `domain_insights["market_intel"]`
- Hermes produces `domain_insights.onet.code` (SOC-like code) but does not do location-based market intel.

## Design Choice: compute market intel in Imaginator
- Hermes remains responsible for domain classification + producing `domain_insights` (including `onet.code` where available).
- Imaginator computes market intel because it already receives the UI-provided `job_location` and controls final response shaping.

## Inputs available at enrichment time
From Imaginator call:
- `location` in kwargs (`location`, `job_location`, `preferred_location`)
- `domain_insights_json` (may include `onet: { code, title, ... }` from Hermes)

We will use:
- `soc_like_code = domain_insights.get("onet", {}).get("code")`
- `geo_id = get_geo_id(location)` (MSA-based mapping; fallback to national if missing)

## Implementation Steps
1. Add feature flag
   - `ENABLE_MARKET_INTEL` (default enabled)
   - If disabled, skip enrichment entirely.

2. Add a small in-memory TTL cache (best-effort)
   - Key: `(soc_like_code, geo_id or "US")`
   - TTL: e.g. 12–24 hours
   - Avoids repeated Data USA calls for identical inputs.

3. Implement enrichment in `imaginator_flow.py`
   - If `location` is present and `soc_like_code` is present:
     - `geo_id = get_geo_id(location)`
     - `workforce = CareerProgressionEnricher()._get_workforce_trends(soc_like_code, geo_id)`
     - `market_intel = CareerProgressionEnricher().calculate_market_intel(workforce, onet_summary={}, job_title=?, location=location)`
       - `job_title` best-effort: from Hermes `onet.title`, first experience title, or empty
     - Attach:
       - `domain_insights["market_intel"] = market_intel`
       - `domain_insights["market_demand"] = market_intel["demand_label"]` when present
       - `domain_insights["salary_range"] = f"${avg_wage:,}+"` when `average_wage` present
   - If any step fails, log and continue without market intel.

4. Avoid O*NET API calls
   - Do NOT perform O*NET search/lookups in Imaginator.
   - This avoids dependency on `ONET_API_AUTH` and reduces latency.
   - Bright Outlook signal will be conservative (false) until Hermes provides explicit bright-outlook tags.

## Output Contract (must match frontend)
`domain_insights.market_intel`:
- `status`: string
- `demand_label`: string
- `is_shortage`: boolean
- `workforce_growth_pct`: number
- `wage_growth_pct`: number
- `has_bright_outlook`: boolean
- `bright_outlook_tags?`: string[]
- `average_wage?`: number
- `workforce_size?`: number
- `data_year?`: string
- `narrative`: string

## Validation
- Local:
  - `python -m py_compile imaginator_flow.py career_progression_enricher.py city_geo_mapper.py`
  - Smoke test by invoking analyze path with a known `location` (e.g. "Seattle, WA") and a `domain_insights_json` containing `onet.code`.
- Render:
  - Deploy Imaginator service and run a single resume analysis with `jobLocation` set.
  - Confirm UI shows Market Intel panel and that `domain_insights.market_intel` appears in payload.

## Follow-ups (optional)
- Improve location parsing and expand `CITY_GEO_MAP` coverage (or replace with a geocoding-backed resolver).
- If Hermes later provides bright-outlook tags, pass them through to enable `has_bright_outlook`.
- Consider persisting cache to Redis if we need cross-instance consistency.
