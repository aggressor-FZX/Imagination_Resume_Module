"""
LLM Client Adapter for New Orchestrator
Wraps OpenRouterSafeClient to provide async call_llm_async method
"""
import json
from typing import Dict, Any, Optional, List
from openrouter_safe_client import OpenRouterSafeClient


class LLMClientAdapter:
    """Adapter to make OpenRouterSafeClient compatible with new orchestrator."""
    
    def __init__(self, api_key: str):
        self.client = OpenRouterSafeClient(api_key=api_key)
    
    async def call_llm_async(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.7,
        response_format: Optional[Dict] = None,
        timeout: Optional[int] = None,
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
            # Researcher: ~1024, Drafter: ~2048, Editor: ~2048
            requested_max = kwargs.get("max_tokens", 2000)
            
            # Define hard caps per stage/model type if possible, or a global safety cap
            # We'll use a strict 3000 token cap for any single stage to prevent "runaway" generation
            hard_cap = 3000 
            safe_max_tokens = min(requested_max, hard_cap)
            
            # Call synchronous method (OpenRouterSafeClient is sync)
            result = self.client.call_model(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=safe_max_tokens
            )
            
            # Extract response text
            if result.get("success"):
                data = result.get("data", {})
                choices = data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    
                    # SAFETY CHECK: Truncate if response exceeds character-based safety cap
                    # (Roughly 4 chars per token)
                    char_limit = safe_max_tokens * 4
                    if len(content) > char_limit:
                        import logging
                        logger = logging.getLogger(__name__)
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
            # Return empty JSON on error
            if response_format and response_format.get("type") == "json_object":
                return "{}"
            return ""
