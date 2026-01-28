"""
LLM Client Adapter for New Orchestrator
Wraps OpenRouterSafeClient to provide async call_llm_async method
"""
import json
import logging
from typing import Dict, Any, Optional, List
from openrouter_safe_client import OpenRouterSafeClient

logger = logging.getLogger(__name__)


class LLMClientAdapter:
    """Adapter to make OpenRouterSafeClient compatible with new orchestrator."""
    
    def __init__(self, api_key: str):
        self.client = OpenRouterSafeClient(api_key=api_key)
        # Track token usage for cost estimation
        self.usage_stats = {
            "calls": [],
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Return accumulated usage statistics."""
        return self.usage_stats.copy()
    
    def reset_usage_stats(self):
        """Reset usage tracking."""
        self.usage_stats = {
            "calls": [],
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
    
    async def call_llm_async(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.7,
        response_format: Optional[Dict] = None,
        timeout: Optional[int] = None,
        fallback_models: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Async wrapper for OpenRouterSafeClient.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            model: Model ID
            temperature: Temperature
            response_format: Response format (e.g., {"type": "json_object"})
            timeout: Timeout in seconds
            fallback_models: Optional list of models to try if primary fails
            **kwargs: Additional arguments
            
        Returns:
            LLM response as string
        """
        try:
            # Build messages in OpenRouter format
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})
            
            # ENFORCE HARD TOKEN LIMITS TO PREVENT COST EXPLOSION
            # Increased from 3000 to 20000 to allow richer context and output
            # Researcher: ~1100, Drafter: ~3000, Editor: ~3000
            requested_max = kwargs.get("max_tokens", 3000)
            
            # Define hard cap - increased to support larger context windows
            # Modern models (Gemini, GPT-4, Claude) support 100k+ context
            # We use 20k as a reasonable ceiling for single-stage output
            hard_cap = 20000 
            safe_max_tokens = min(requested_max, hard_cap)
            
            # Call synchronous method (OpenRouterSafeClient is sync)
            result = self.client.call_model(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=safe_max_tokens,
                fallback_models=fallback_models
            )
            
            # LOG THE ACTUAL MODEL USED (Crucial for debugging instruction following)
            model_used = result.get("model_used")
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[LLM_ADAPTER] Model requested: {model} | Model actually used: {model_used}")
            
            # Extract response text and track usage
            if result.get("success"):
                data = result.get("data", {})
                choices = data.get("choices", [])
                
                # Track token usage from response
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                
                # Estimate cost (approximate rates per 1K tokens)
                # Default to conservative estimates for unknown models
                cost = self._estimate_cost(model, prompt_tokens, completion_tokens)
                
                # Update cumulative stats
                self.usage_stats["total_prompt_tokens"] += prompt_tokens
                self.usage_stats["total_completion_tokens"] += completion_tokens
                self.usage_stats["total_tokens"] += total_tokens
                self.usage_stats["estimated_cost_usd"] += cost
                self.usage_stats["calls"].append({
                    "model": model,
                    "model_used": model_used,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost_usd": cost,
                })
                
                logger.info(f"[LLM_ADAPTER] Usage: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total tokens (~${cost:.4f})")
                
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    
                    # SAFETY CHECK: Truncate if response exceeds character-based safety cap
                    # (Roughly 4 chars per token)
                    char_limit = safe_max_tokens * 4
                    if len(content) > char_limit:
                        logger.warning(f"[TOKEN_GUARD] Response truncated: {len(content)} chars exceeded {char_limit} limit")
                        content = content[:char_limit]
                        # Attempt to close JSON if it was a JSON object
                        if response_format and response_format.get("type") == "json_object":
                            if not content.strip().endswith("}"):
                                # Find last valid closing brace if possible
                                last_brace = content.rfind("}")
                                if last_brace != -1:
                                    content = content[:last_brace+1]
                                else:
                                    content += "}"
                    
                    return content
            
            # Fallback on error
            if response_format and response_format.get("type") == "json_object":
                return "{}"
            return ""
            
        except Exception as e:
            logger.error(f"[LLM_ADAPTER] Error: {e}")
            # Return empty JSON on error
            if response_format and response_format.get("type") == "json_object":
                return "{}"
            return ""
    
    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost based on model and token usage."""
        # Pricing per 1K tokens (approximate, varies by model)
        pricing = {
            # OpenAI models
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            # Anthropic
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            # Google
            "gemini-2.0-flash": {"input": 0.00015, "output": 0.0006},
            "gemini-pro": {"input": 0.0005, "output": 0.0015},
            # DeepSeek
            "deepseek": {"input": 0.00014, "output": 0.00028},
            # Default for unknown models
            "default": {"input": 0.001, "output": 0.002},
        }
        
        # Find matching pricing
        model_lower = model.lower()
        for key, rates in pricing.items():
            if key in model_lower:
                return (prompt_tokens / 1000) * rates["input"] + (completion_tokens / 1000) * rates["output"]
        
        # Use default pricing
        rates = pricing["default"]
        return (prompt_tokens / 1000) * rates["input"] + (completion_tokens / 1000) * rates["output"]
