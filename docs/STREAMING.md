# Imaginator streaming (Researcher / Drafter / StarEditor)

**Status:** Shipped with orch multi-file SSE (`field`: `draft` then `rewrite`).  
**Landing how-it-works:** `Login_Landing_page/docs/superpowers/specs/2026-10-16-streaming-researcher-drafter.md`

## Dual-mode endpoints

| Endpoint | Without `Accept: text/event-stream` | With SSE |
|----------|-------------------------------------|----------|
| `POST /analyze` | JSON result (unchanged) | `token` (`draft` then `rewrite`) → `result` |
| `POST /research` | JSON `{ research_data, status, duration_ms }` | `stage` heartbeats → `research_partial` → `result` (same body) |

## Killswitches (Imaginator)

| Var | Default | Role |
|-----|---------|------|
| `IMAG_STREAM_DRAFTER` | `true` | Pass Drafter tokens via stream context |
| (StarEditor) | always when SSE | `field=rewrite` (existing ABC path) |

Orch also gates draft forwarding with `FORWARD_DRAFT_TOKENS` and research SSE with `IMAG_STREAM_RESEARCHER`.

## Implementation notes

- `stream_context` callbacks are `(field, delta)`; arity-1 callbacks stay rewrite-only for backward compat.
- `LLMClientAdapter.call_llm_async` runs blocking `call_model` in `asyncio.to_thread` so heartbeats and progressive tokens can leave the event loop.
- Token callbacks from worker threads use `loop.call_soon_threadsafe` in the SSE producer.

## Related

- Export PDF/DOCX: [EXPORT.md](./EXPORT.md)
- Fork-join skip Stage 1 when `research_data` is provided on `/analyze`
