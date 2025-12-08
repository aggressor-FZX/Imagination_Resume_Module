# OpenRouter Model Calling — Guide & Lessons Learned ✅

This document summarizes what we've learned while integrating OpenRouter and other provider models into the Agentic Workshop Plugin. It explains how we choose models, configure credentials, debug common errors, normalize model names/aliases, and test model calls (including code and curl examples).

---

## TL;DR (Two-sentence summary) 💡
- Use provider-specific base URLs and keys (OPENROUTER vs OPENAI vs DEEPSEEK) to avoid 401 Unauthorized and model validation errors. Configure `OPENAI_BASE_URL` if you want OpenAI direct calls.
- Normalize model IDs and avoid attempting incompatible model/credential combinations — we added normalization and credential filtering logic in `shared.py` to reduce 400/401 retries.

---

## What we learned (Key takeaways) 🔍

- Provider base matters: `OPENAI_API_KEY` used against an OpenRouter base or vice versa may cause 401s ("User not found") or 400 invalid model errors. Map keys to the correct provider base or intentionally use OpenRouter as a passthrough.
- Model names differ by provider; always use canonical model IDs for consistency. OpenRouter often uses `provider/model-name` format (e.g., `qwen/qwen3-32b:thinking`), while direct OpenAI models are e.g. `gpt-4o-mini`.
- Normalizing model names (aliases) to canonical IDs resolves many issues; we added a `normalize_model_id(model_id)` function to map "short" names to canonical ones.
- Strict LLM Mode (in `STRICT_LLM_MODE`) reduces unexpected fallback behavior by trying the top candidate model per credential/provider.
- Errors we saw and addressed:
  - `401 - User not found`: credential vs provider mismatch (wrong api key or base)
  - `400 - Not a valid model ID`: attempted model isn't available for that credential or provider
  - `Connection/DNS error`: network issue or unreachable provider host (e.g., `api.deepseek.ai` DNS issue in our environment)
  - `'str' object has no attribute 'model_dump'`: a binding/type mismatch for how model objects are passed to client libraries (fix: pass a proper `model_name` string or update to correct langchain version call signature)

---

## Configuration & Environment Variables ⚙️

Important env variables used:
- `OPENROUTER_API_KEY_1` (primary OpenRouter key; used for OpenRouter model passthrough)
- `OPENROUTER_API_KEY_2` (fallback OpenRouter key)
- `OPENAI_API_KEY` (OpenAI API key; set `OPENAI_BASE_URL` accordingly if using OpenAI directly)
- `OPENAI_BASE_URL` (default: `https://api.openai.com/v1` or `https://openrouter.ai/v1` if using OpenRouter passthrough)
- `DEEPSEEK_API_KEY` or `DeepSeek` (DeepSeek provider creds)
- `STRICT_LLM_MODE` (true/false — if true, the system tries only the single top candidate model per provider before moving to next credential)

Notes:
- If you want to use OpenRouter as a passthrough for OpenAI models, set `OPENAI_BASE_URL` to the OpenRouter base (e.g., `https://openrouter.ai/v1`) and set `OPENAI_API_KEY` to an OpenRouter token that has access to OpenAI-pass-through.
- For consistent provider selection, configure `OPENROUTER_API_KEY_1` and `OPENAI_BASE_URL` correctly in `.env`.

---

## Model selection logic (how `get_llm` decides) 🧠

Summary of `Agentic_work_shop_plugin/shared.py::get_llm`:
- If `FREE_ONLY_MODE` is enabled, we cycle free models via `get_next_free_model()`.
- Otherwise, we pick model sets based on role (`pm` or `worker`) and `worker_type` (free/paid), and then try credentials in the configured priority order.
- For each credential we build a list of credential-specific model candidates:
  - OpenRouter credentials allow openrouter/deepseek/openai_* passthrough names.
  - Non-OpenRouter credentials use `models_to_try` (role-based lists built in code).
- In `STRICT_LLM_MODE` we only try the top candidate model per credential.
- We added these safeguards:
  - `normalize_model_id()` — convert alias/short names to canonical ids found in `openrouter_models.json`.
  - `model_compatible_with_credential()` — avoid trying models that are clearly incompatible with the credential base (e.g., `deepseek/` models with OpenAI base URL, unless OpenRouter passthrough is on).
  - When a provider returns 401 errors, the credential gets marked `disabled` for the run to avoid repeated retries.
  - When a provider returns a 400 code ("model not valid"), we mark that model as unsupported for that credential and skip it on future attempts.

---

## How to call OpenRouter models (examples) 👇

1) Using our `get_llm` helper from `shared.py` (recommended):

```python
from Agentic_work_shop_plugin import shared
llm = shared.get_llm('worker', worker_type='free', temperature=0.0)
# llm is a langchain_openai.ChatOpenAI instance
resp = llm.invoke("Say 'OK' if you can read this")
print(resp.content)
```

2) Using `langchain_openai.ChatOpenAI` directly (when you know provider/base and model):

```python
from langchain_openai import ChatOpenAI
import os

OPENROUTER_API_KEY_1 = os.environ.get('OPENROUTER_API_KEY_1')
llm = ChatOpenAI(base_url='https://openrouter.ai/api/v1', api_key=OPENROUTER_API_KEY_1, model_name='openai/gpt-oss-120b', temperature=0.0)
resp = llm.invoke("What's 1+1? Give short answer")
print(resp.content)
```

3) Using curl to call OpenRouter's Chat Completions (example tested with `openai/gpt-oss-120b`):

```bash
curl -sS -X POST "https://openrouter.ai/api/v1/chat/completions" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY_1" \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-oss-120b","messages":[{"role":"system","content":"You are a test assistant."},{"role":"user","content":"Summarize: 1+1 equals what? Give a short reply."}], "max_tokens": 50, "temperature": 0.0}' | jq .
```

We successfully validated `openai/gpt-oss-120b` through `OPENROUTER_API_KEY_1`.

---

## Common errors & how to fix them 🛠️

- 401 User not found — check that the API key matches the provider base. If you intended to call OpenAI directly, set `OPENAI_BASE_URL=https://api.openai.com/v1`.
- 400 Not a valid model ID — this usually means the model name is unavailable/invalid under that credential. Use `normalize_model_id()` to find canonical ID; verify `openrouter_models.json` for accurate model names.
- Model alias differences (e.g., `qwen3-coder-480b` vs `qwen/qwen3-coder-480b-a35b-instruct`): update `openrouter_models.json` and let `normalize_model_id()` do the mapping — we added logging to show normalization events.
- `'str' object has no attribute 'model_dump'` — indicates an outdated usage or mismatch in the method signature between our code and the `langchain_openai` library. Ensure you're passing `model_name` correctly and that your local `langchain_openai` version matches usage.
- DNS/Connection errors to DeepSeek (or any provider) — verify network connectivity and DNS resolution from the runtime environment. Some hosts may not be resolvable in closed network environments.

---

## Best practices & recommendations ✅

- Always set provider bases explicitly and be consistent in the environment:
  - For direct OpenAI usage: set `OPENAI_BASE_URL=https://api.openai.com/v1`.
  - For OpenRouter passthrough: set `OPENAI_BASE_URL=https://openrouter.ai/v1` and use OpenRouter keys for model passthrough.
- Keep `openrouter_models.json` up-to-date and canonical. Use it to avoid attempting unknown model IDs.
- Prefer `get_llm` for model selection: it understands role-based selection, free-only mode, and has retry/credential reasoning.
- Use `STRICT_LLM_MODE` in production to have predictable model selections per provider, particularly if you have multiple providers configured.
- Add a quality control step (the Orchestrator) that inspects results before merging any commits to a workspace repo. Our system attempts merges when `ci_results` PASS or with `review_node` review fallback.
- Track and disable credentials returning 401s to avoid repeated error spam.

---

## Troubleshooting checklist (if you see errors) 🔎

1. Check `manager.log` for GET_LLM_* logs and the provider/model it attempted.
2. Confirm `OPENAI_BASE_URL` and `OPENROUTER_BASE` values in `.env`.
3. Run `dev_tools/validate_model_selection.py` (or `shared.test_api_key()`) to perform a small, safe model test.
4. If you see `not a valid model ID`, verify the canonical id in `openrouter_models.json`.
5. If you see `401` messages, mark the bad credential as disabled or remove it from env, and double-check which provider/keys are aligned.
6. If you see DNS errors, run `curl -v https://api.deepseek.ai` from the environment and check network/DNS settings.
7. Update `openrouter_models.json` if new models or provider formats are added.

---

## Next steps we recommend 🧭

- Optionally implement a `supported_models` map per-credential from `openrouter_models.json` to prune model candidates per credential automatically.
- Add more mocked unit tests that patch `langchain_openai.ChatOpenAI` to avoid HTTP calls in CI while asserting credential flow/disablement behaviors.
- Add an integration test to confirm a real merge flow where a feature branch is merged when CI passes.

---

## Where to find relevant code
- `Agentic_work_shop_plugin/shared.py` — model selection, normalization, credential collection
- `Agentic_work_shop_plugin/openrouter_models.json` — canonical model ids and pricing metadata
- `dev_tools/validate_model_selection.py` — a helper script to quickly verify and test model selection

---

If you'd like, I can also add the `supported_models` feature or augment `openrouter_models.json` with a few more canonical entries and update examples and README. Would you like me to proceed with that? ✨
