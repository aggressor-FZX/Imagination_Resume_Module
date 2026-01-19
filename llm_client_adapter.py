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
            # Default to 2000, but cap strictly at 4000 to prevent Grok-style verbosity
            requested_max = kwargs.get("max_tokens", 2000)
            hard_cap = 4000  # Absolute maximum tokens allowed per call
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
                    
                    # SAFETY CHECK: Truncate if response exceeds hard cap
                    # This prevents models from "over-explaining" and burning credits
                    if len(content) > (safe_max_tokens * 4):  # Rough char estimate
                        print(f"⚠️  Response truncated: {len(content)} chars exceeded safety cap")
                        # Attempt to find the end of the JSON object if it's JSON
                        if response_format and response_format.get("type") == "json_object":
                            end_brace = content.rfind("}")
                            if end_brace != -1:
                                content = content[:end_brace+1]
                    
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
