# Imaginator Run Behavior

This guide summarizes what API consumers should expect from the Imaginator service in both successful and failed runs, with an emphasis on transparency for billing and retry automation.

## Failure Handling

- **Trigger conditions**: Any hard failure in `run_analysis_async`, `run_generation_async`, or `run_criticism_async` (LLM transport errors, JSON parsing issues, timeouts) bubbles up to the `/analyze` endpoint.
- **API response**: HTTP `500` with payload shape:
  ```json
  {
    "error": "Analysis failed: <root cause>",
    "error_code": "ANALYSIS_FAILED",
    "details": {
      "processing_time_seconds": <float>,
      "run_metrics": null | {
        "calls": [],
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "failures": [
          {
            "provider": "openrouter",
            "model": "...",
            "error": "..."
          }
        ]
      }
    }
  }
  ```
- **Billing/costs**: Because we only update `RUN_METRICS` on successful LLM completions, failed calls leave `estimated_cost_usd` at `0.0` and the `calls` list empty. Clients should treat these runs as **non-billable** and display the upstream error instead of fabricating content.
- **Retry guidance**: You may retry after inspecting `error`. If `failures` includes provider metadata, consider exponential backoff or failover to a different model on the client side.

## Successful Run Output

- **HTTP response**: `200 OK` with a full `AnalysisResponse`. The key fields for auditing usage are inside `run_metrics`:
  ```json
  {
    "run_metrics": {
      "calls": [
        {
          "provider": "openrouter",
          "model": "qwen/qwen3-30b-a3b",
          "prompt_tokens": 1840,
          "completion_tokens": 1473,
          "total_tokens": 3313,
          "estimated_cost_usd": 0.000434
        },
        {
          "provider": "openrouter",
          "model": "deepseek/deepseek-chat-v3.1",
          "prompt_tokens": 1006,
          "completion_tokens": 544,
          "total_tokens": 1550,
          "estimated_cost_usd": 0.000636
        }
      ],
      "total_prompt_tokens": 3362,
      "total_completion_tokens": 2465,
      "total_tokens": 5827,
      "estimated_cost_usd": 0.0012,
      "failures": [],
      "stages": {
        "analysis": {"duration_ms": 1200, "cache_hit": false},
        "generation": {"duration_ms": 980, "cache_hit": false},
        "synthesis": {"duration_ms": 650, "cache_hit": false},
        "criticism": {"duration_ms": 710, "cache_hit": false}
      }
    }
  }
  ```
- **Interpretation**:
  - `calls`: One entry per successful LLM invocation. Costs are estimated using model-specific OpenRouter pricing so the frontend can itemize spend.
  - `total_*_tokens`: Aggregated prompt, completion, and combined token counts for the entire request.
  - `estimated_cost_usd`: Sum of per-call estimates; this is the number to use for billing the user.
- `failures`: Non-empty only if a particular provider errored but was retried successfully in another call.

Additionally, use `stages.*.duration_ms` for per-stage latency tracking; when caching is added, `stages.*.cache_hit` will indicate cache usage for that stage.

## Client Checklist

1. **On success**: Render the normal payload and show token/cost totals from `run_metrics` to the user.
2. **On failure**: Surface the `error` message, mark the job as failed, and charge $0 (the API already reports zero estimated cost).
3. **Monitoring**: Track frequency of `ANALYSIS_FAILED` to decide when to alert operators or trigger automated retries.
