#!/usr/bin/env python3
"""
Imaginator Gateway Module
Handles all LLM interactions with OpenRouter and Google Gemini.
Includes cost tracking, fallback logic, and web search capabilities.
"""
import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

# Optional imports for Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False

from .config import (
    MODEL_STAGE_1, MODEL_STAGE_1_BACKUP, MODEL_STAGE_2, MODEL_STAGE_2_BACKUP,
    MODEL_STAGE_3, MODEL_STAGE_4, MODEL_STAGE_4_BACKUP, MODEL_FALLBACK,
    MODEL_ANALYTICAL, MODEL_BALANCED, OPENROUTER_API_KEYS, GOOGLE_API_KEY,
    REFERER, TITLE, OPENROUTER_PRICE_IN_K, OPENROUTER_PRICE_OUT_K,
    QWEN_PRICE_IN_K, QWEN_PRICE_OUT_K, DEEPSEEK_PRICE_IN_K, DEEPSEEK_PRICE_OUT_K,
    ANTHROPIC_PRICE_IN_K, ANTHROPIC_PRICE_OUT_K, GOOGLE_PRICE_IN_K, GOOGLE_PRICE_OUT_K,
    OPENAI_PRICE_IN_K, OPENAI_PRICE_OUT_K, settings
)
from .config import ANALYSIS_CACHE_TTL_SECONDS

# Global metrics tracking
RUN_METRICS: Dict[str, Any] = {
    "calls": [],
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_tokens": 0,
    "estimated_cost_usd": 0.0,
    "failures": [],
}

# Model registry for unhealthy model tracking
class OpenRouterModelRegistry:
    """Cache OpenRouter's model catalog and track unhealthy models."""
    
    CACHE_TTL_SECONDS = 600
    ERROR_TTL_SECONDS = 120
    UNHEALTHY_TTL_SECONDS = 900
    SAFE_MODELS = [
        "openai/gpt-oss-120b",
        "openai/gpt-4.1-nano",
        "anthropic/claude-3-haiku",
        "deepseek/deepseek-chat-v3.1",
    ]
    
    def __init__(self, api_key: Optional[str], referer: Optional[str] = None) -> None:
        self.api_key = api_key
        self.referer = referer
        self._lock = asyncio.Lock()
        self._catalog: set = set()
        self._expires_at = 0.0
        self._unhealthy: Dict[str, float] = {}
    
    async def _refresh_catalog_locked(self) -> None:
        import aiohttp
        now = time.time()
        if not self.api_key:
            self._expires_at = now + self.ERROR_TTL_SECONDS
            return
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            if self.referer:
                headers["HTTP-Referer"] = self.referer
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://openrouter.ai/api/v1/models",
                    headers=headers,
                    timeout=10,
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    models = {
                        entry.get("id")
                        for entry in data.get("data", [])
                        if entry.get("id")
                    }
                    if models:
                        self._catalog = models
                    self._expires_at = now + self.CACHE_TTL_SECONDS
        except Exception as exc:
            print(f"⚠️  Unable to refresh OpenRouter model catalog: {exc}", flush=True)
            self._expires_at = now + self.ERROR_TTL_SECONDS
    
    async def _purge_unhealthy_locked(self) -> None:
        now = time.time()
        expired = [model for model, until in self._unhealthy.items() if until <= now]
        for model in expired:
            self._unhealthy.pop(model, None)
    
    async def _get_catalog_locked(self) -> set:
        now = time.time()
        if now >= self._expires_at:
            await self._refresh_catalog_locked()
        return set(self._catalog)
    
    async def get_candidates(self, preferred: List[str]) -> List[str]:
        deduped: List[str] = []
        for model in preferred + [
            model for model in self.SAFE_MODELS if model not in preferred
        ]:
            if model and model not in deduped:
                deduped.append(model)
        
        async with self._lock:
            await self._purge_unhealthy_locked()
            catalog = await self._get_catalog_locked()
            unhealthy = {model for model in self._unhealthy}
        
        if catalog:
            candidates = [
                model for model in deduped if model in catalog and model not in unhealthy
            ]
        else:
            candidates = [model for model in deduped if model not in unhealthy]
        return candidates
    
    def mark_unhealthy(self, model: str) -> None:
        if not model:
            return
        self._unhealthy[model] = time.time() + self.UNHEALTHY_TTL_SECONDS


# Initialize registry
openrouter_model_registry = OpenRouterModelRegistry(
    OPENROUTER_API_KEYS[0] if OPENROUTER_API_KEYS else None, 
    REFERER
)


def _estimate_openrouter_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate per-call cost based on model-specific OpenRouter pricing."""
    prompt_rate = OPENROUTER_PRICE_IN_K
    completion_rate = OPENROUTER_PRICE_OUT_K
    
    if "qwen/qwen3-30b-a3b" in model:
        prompt_rate = QWEN_PRICE_IN_K
        completion_rate = QWEN_PRICE_OUT_K
    elif "deepseek/deepseek-chat-v3.1" in model:
        prompt_rate = DEEPSEEK_PRICE_IN_K
        completion_rate = DEEPSEEK_PRICE_OUT_K
    elif "claude-3" in model:
        prompt_rate = ANTHROPIC_PRICE_IN_K
        completion_rate = ANTHROPIC_PRICE_OUT_K
    
    return (prompt_tokens / 1000.0) * prompt_rate + (completion_tokens / 1000.0) * completion_rate


def _get_openrouter_preferences(system_prompt: str, user_prompt: str) -> List[str]:
    """
    Select optimal OpenRouter model based on pipeline stage (COST-OPTIMIZED).
    
    Strategy (4-Stage Pipeline with Global Fallback):
    - RESEARCHER: DeepSeek v3.2:online (cheapest web search) → DeepSeek backup
    - CREATIVE: Skyfall 36B (budget creative) → Cydonia 24B backup
    - STAR_EDITOR: Microsoft Phi-4 for STAR pattern bullet formatting
    - FINAL_EDITOR: Claude 3 Haiku for final editorial polish
    - GLOBAL_FALLBACK: GPT-4o:online (emergency only, logs error)
    - Legacy paths: Analytical (Phi-4), Balanced (Claude)
    """
    sys_lower = system_prompt.lower()
    user_lower = user_prompt.lower()
    preferences: List[str] = []
    
    # Select primary model based on pipeline stage with backup models
    if "researcher" in sys_lower or "web search" in sys_lower:
        preferences.append(MODEL_STAGE_1)           # DeepSeek v3.2:online
        preferences.append(MODEL_STAGE_1_BACKUP)    # DeepSeek chat backup
    elif "creative" in sys_lower or "draft" in sys_lower or "generation" in user_lower:
        preferences.append(MODEL_STAGE_2)             # Skyfall 36B
        preferences.append(MODEL_STAGE_2_BACKUP)      # Cydonia 24B backup
    elif "star" in sys_lower or "bullet" in sys_lower or "format" in sys_lower:
        preferences.append(MODEL_STAGE_3)          # Phi-4 for STAR
    elif "editor" in sys_lower or "polish" in sys_lower or "final" in sys_lower:
        preferences.append(MODEL_STAGE_4)         # Gemini final edit
        preferences.append(MODEL_STAGE_4_BACKUP)  # Mistral Large fallback
    elif "critic" in sys_lower or "review" in user_lower:
        preferences.append(MODEL_ANALYTICAL)           # Phi-4 for analysis
    else:
        preferences.append(MODEL_BALANCED)             # Claude 3 Haiku default
    
    # Add safe models as fallbacks (prevents duplicates)
    for model in OpenRouterModelRegistry.SAFE_MODELS:
        if model not in preferences:
            preferences.append(model)
    
    # ALWAYS add global fallback as last resort (will trigger error logging)
    if MODEL_FALLBACK not in preferences:
        preferences.append(MODEL_FALLBACK)
    
    return preferences


def _should_blacklist_openrouter_model(error: Exception) -> bool:
    message = str(error).lower()
    status_code = getattr(error, "status_code", None)
    if status_code in {400, 404, 410, 422} and "model" in message:
        return True
    keywords = (
        "model_not_found",
        "does not exist",
        "unknown model",
        "invalid model",
        "not found",
    )
    return any(keyword in message for keyword in keywords)


def _extract_gemini_text(response: Any) -> str:
    try:
        text_value = getattr(response, "text", None)
        if isinstance(text_value, str) and text_value.strip():
            return text_value.strip()
    except Exception:
        pass
    
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            continue
        fragments = []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                fragments.append(part_text.strip())
        if fragments:
            return "\n".join(fragments)
    return ""


# Configure Google client library if credentials exist
if GOOGLE_API_KEY and GEMINI_AVAILABLE:
    genai.configure(api_key=GOOGLE_API_KEY)

# Map old or misspelled Gemini model names to the current recommended ones.
GEMINI_MODEL_ALIAS_MAP = {
    "gemini-1.5-flash": "gemini-2.5-flash",
    "gemini-1.5-flash-001": "gemini-2.5-flash",
    "gemini-1.5-pro": "gemini-2.5-pro",
    "gemini-1.5-pro-001": "gemini-2.5-pro",
    "gemino-2.5-flash": "gemini-2.5-flash",  # common typo support
    "gemino-2.5-pro": "gemini-2.5-pro",
}


def _normalize_gemini_model_name(model_name: Optional[str]) -> str:
    """Ensure Gemini model names stay up to date even if configs lag."""
    if not model_name:
        return ""
    cleaned = model_name.strip()
    canonical = GEMINI_MODEL_ALIAS_MAP.get(cleaned.lower())
    if canonical and canonical != cleaned:
        print(f"♻️  Updating Gemini model '{cleaned}' -> '{canonical}'", flush=True)
        return canonical
    return canonical or cleaned


DEFAULT_GEMINI_FLASH_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_PRO_MODEL = "gemini-2.5-pro"

# Google Gemini fallback order (Flash first, then Pro)
GOOGLE_GEMINI_FLASH_MODEL = _normalize_gemini_model_name(
    os.getenv("GOOGLE_GEMINI_FLASH_MODEL", DEFAULT_GEMINI_FLASH_MODEL)
)
GOOGLE_GEMINI_PRO_MODEL = _normalize_gemini_model_name(
    os.getenv("GOOGLE_GEMINI_PRO_MODEL", DEFAULT_GEMINI_PRO_MODEL)
)

GOOGLE_FALLBACK_MODELS: List[str] = []
for model in (GOOGLE_GEMINI_FLASH_MODEL, GOOGLE_GEMINI_PRO_MODEL):
    normalized = _normalize_gemini_model_name(model)
    if normalized and normalized not in GOOGLE_FALLBACK_MODELS:
        GOOGLE_FALLBACK_MODELS.append(normalized)


def _build_google_clients(api_key_override: Optional[str] = None) -> List[Tuple[str, Any]]:
    """Create Gemini clients in the preferred fallback order."""
    if not GEMINI_AVAILABLE:
        return []
    
    api_key = api_key_override or GOOGLE_API_KEY
    if not api_key or not GOOGLE_FALLBACK_MODELS:
        return []
    genai.configure(api_key=api_key)
    clients: List[Tuple[str, Any]] = []
    for model in GOOGLE_FALLBACK_MODELS:
        normalized = _normalize_gemini_model_name(model)
        try:
            pass  # genai.models.get(normalized)
        except Exception as exc:  # pragma: no cover - SDK runtime failures
            print(f"⚠️  Gemini model '{normalized}' unavailable: {exc}", flush=True)
            continue
        try:
            clients.append((normalized, genai.GenerativeModel(normalized)))
        except Exception as exc:  # pragma: no cover - SDK runtime failures
            print(f"⚠️  Failed to initialize Gemini model '{normalized}': {exc}", flush=True)
    return clients


# Initialize OpenRouter client (unified API)
openrouter_client = None
openrouter_async_client = None

if any(OPENROUTER_API_KEYS):
    # Find the first valid key to initialize the clients
    valid_key = next((key for key in OPENROUTER_API_KEYS if key), None)
    if valid_key:
        openrouter_client = openrouter_client or openrouter_client  # Placeholder
        openrouter_async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=valid_key,
            default_headers={
                "HTTP-Referer": REFERER,
                "X-Title": TITLE
            }
        )


async def call_llm_async(
    system_prompt: str,
    user_prompt: str,
    job_ad: Optional[str] = None,
    extracted_skills_json: Optional[Dict] = None,
    domain_insights_json: Optional[Dict] = None,
    confidence_threshold: float = 0.7,
    temperature: float = 0.9,
    max_tokens: int = 1500,
    max_retries: int = 3,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    openrouter_api_keys: Optional[List[str]] = None,
    response_format: Optional[Dict[str, Any]] = None,
    enable_web_search: bool = False,
    web_search_options: Optional[Dict[str, Any]] = None,
    **_ignored: Any,
) -> str:
    """
    Async call to LLM with automatic fallback: OpenRouter primary, Google Gemini fallback.
    
    Args:
        enable_web_search: If True, enables web search plugin for models that support it
                          (currently google/gemini-2.5-pro:online). Researcher stage uses this.
        web_search_options: Optional configuration for web search behavior:
                           - max_results: int (default 5)
                           - search_context_size: "small"|"medium"|"large" (default "medium")
    """
    errors: List[str] = []

    # Determine which keys to use for rotation
    keys_to_try = openrouter_api_keys or OPENROUTER_API_KEYS

    # Use the globally configured async client if available
    or_async_client = openrouter_async_client
    if openrouter_api_key:
        # If a single key is passed directly, create a new client for this call
        or_async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            default_headers={
                "HTTP-Referer": REFERER,
                "X-Title": TITLE
            }
        )

    async def _try_openrouter_model(model: str, client: Any) -> Optional[str]:
        try:
            start_time = time.time()
            
            # Build request payload
            request_payload: Dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
            
            # Add response format if specified
            if response_format:
                request_payload["response_format"] = response_format
            
            # Add web search plugin if enabled (for :online models)
            if enable_web_search and ":online" in model:
                search_options = web_search_options or {"max_results": 5}
                request_payload["plugins"] = [{
                    "id": "web",
                    "engine": "auto",  # Let OpenRouter choose best search engine
                    "web_search_options": search_options
                }]
                print(f"[WEB SEARCH ENABLED] Model: {model}, Options: {search_options}", flush=True)
            
            response = await client.chat.completions.create(**request_payload)
            duration = time.time() - start_time
            
            # CRITICAL WARNING: Log if using expensive GPT-4o fallback
            if "gpt-4o" in model.lower():
                print("\n" + "="*80, flush=True)
                print("🚨 COST ALERT: FALLBACK TO GPT-4O TRIGGERED! 🚨", flush=True)
                print(f"All cheaper models failed. Using expensive fallback: {model}", flush=True)
                print("Cost impact: ~10x more expensive than DeepSeek/Skyfall models", flush=True)
                print("="*80 + "\n", flush=True)
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"🚨🚨🚨 COST ALERT: GPT-4O FALLBACK USED! Model: {model} 🚨🚨🚨")
            
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            cost = _estimate_openrouter_cost(model, prompt_tokens, completion_tokens)
            RUN_METRICS["calls"].append({
                "provider": "OpenRouter",
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "duration_ms": int(duration * 1000),
                "cost_usd": cost,
            })
            RUN_METRICS["total_prompt_tokens"] += prompt_tokens
            RUN_METRICS["total_completion_tokens"] += completion_tokens
            RUN_METRICS["total_tokens"] += prompt_tokens + completion_tokens
            RUN_METRICS["estimated_cost_usd"] += cost
            return response.choices[0].message.content
        except Exception as e:
            if _should_blacklist_openrouter_model(e):
                openrouter_model_registry.mark_unhealthy(model)
            RUN_METRICS["failures"].append({
                "provider": "OpenRouter",
                "model": model,
                "error": str(e),
            })
            return None

    if or_async_client:
        for key in keys_to_try:
            if not key:
                continue
            or_async_client.api_key = key
            candidates = openrouter_model_registry.get_candidates(
                _get_openrouter_preferences(system_prompt, user_prompt)
            )
            # IMPORTANT: Try in deterministic order (primary -> fallback) rather than racing.
            # Racing can cause a fast fallback model to "win" even when the primary is available.
            for model in candidates:
                result = await _try_openrouter_model(model, or_async_client)
                if isinstance(result, str) and result:
                    return result
            print(f"OpenRouter key {key[:5]}... failed, trying next key.")

    # Fallback to DeepSeek
    deepseek_keys_to_try = deepseek_api_key and [deepseek_api_key] or []
    for key in deepseek_keys_to_try:
        if not key:
            continue
        try:
            # This is a placeholder for the actual DeepSeek API call
            print(f"\nAttempting async DeepSeek call...", flush=True)
            # response = await deepseek_client.chat.completions.create(...)
            # return response.choices[0].message.content
        except Exception as e:
            errors.append(f"DeepSeek Error: {e}")

    # Async fallback to Google Gemini
    google_clients = _build_google_clients(google_api_key)
    for attempt, (model, client) in enumerate(google_clients):
        try:
            print(f"\nAttempting async Google Gemini call with {model}...", flush=True)
            start_time = time.time()
            response = await client.generate_content_async(
                f"{system_prompt}\n\n{user_prompt}",
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
            )
            duration = time.time() - start_time
            RUN_METRICS["calls"].append({
                "provider": "Google",
                "model": model,
                "duration_ms": int(duration * 1000),
            })
            text = _extract_gemini_text(response)
            if text and text.strip():
                return text.strip()
            raise ValueError("Empty Gemini output")
        except Exception as e:
            errors.append(f"Google ({model}) Error: {e}")
            RUN_METRICS["failures"].append({
                "provider": "Google",
                "model": model,
                "attempt": attempt + 1,
                "error": str(e),
            })

    raise Exception(f"All LLM providers failed: {' | '.join(errors)}")


def reset_metrics() -> None:
    """Reset the global metrics tracking."""
    global RUN_METRICS
    RUN_METRICS = {
        "calls": [],
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "failures": [],
    }


def get_metrics() -> Dict[str, Any]:
    """Get current metrics snapshot."""
    return RUN_METRICS.copy()