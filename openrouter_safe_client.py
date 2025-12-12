#!/usr/bin/env python3
"""
Safe OpenRouter Client with Model Availability Checking and Failsafe Fallback

This module provides a robust way to interact with OpenRouter API:
1. Validates models exist before attempting calls
2. Implements automatic fallback to DeepSeek if preferred models fail
3. Formats requests properly according to OpenRouter API spec
4. Tracks failure metrics for debugging
"""

import json
import time
from typing import List, Dict, Any, Optional
import requests
from config import settings


class OpenRouterSafeClient:
    """
    Safe wrapper around OpenRouter API with model validation and failsafe fallback.
    """

    # Hardcoded fallback chain - these are always available
    FALLBACK_CHAIN = [
        "deepseek/deepseek-chat-v3.1",  # Primary fallback - most reliable
        "anthropic/claude-3-haiku",      # Backup if DeepSeek fails
        "openai/gpt-3.5-turbo",          # Last resort
    ]

    def __init__(self, api_key: str, referer: str = "https://imaginator-resume-cowriter.onrender.com"):
    """
    Initialize the safe OpenRouter client.

    Args:
        api_key: Primary OpenRouter API key (fallback if env vars not set)
        referer: HTTP Referer header for OpenRouter tracking
    """
    import os
    self.api_keys = [
        os.getenv('OPENROUTER_API_KEY_1', api_key),
        os.getenv('OPENROUTER_API_KEY_2')
    ]
    self.api_keys = [k for k in self.api_keys if k]  # Remove empty
    if not self.api_keys:
        raise ValueError("No OpenRouter API keys available")
    self.referer = referer
    self.base_url = "https://openrouter.ai/api/v1"
    self._model_cache: Dict[str, bool] = {}  # Cache of available models
    self._cache_expires_at = 0
    self._cache_ttl_seconds = 600  # 10 minute cache

    def _get_available_models(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetch available models from OpenRouter.

        Args:
            force_refresh: Ignore cache and fetch fresh list

        Returns:
            Dict mapping model IDs to model info
        """
        # Check cache
        if not force_refresh and time.time() < self._cache_expires_at:
            return self._model_cache

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            if self.referer:
                headers["HTTP-Referer"] = self.referer

            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            models = {entry['id']: entry for entry in data.get('data', [])}

            self._model_cache = models
            self._cache_expires_at = time.time() + self._cache_ttl_seconds

            print(f"✅ Loaded {len(models)} available models from OpenRouter")
            return models

        except Exception as e:
            print(f"⚠️  Failed to fetch model list: {e}")
            # Return cached list or empty dict
            return self._model_cache or {}

    def _is_model_available(self, model_id: str, force_refresh: bool = False) -> bool:
        """
        Check if a model is available on OpenRouter.

        Args:
            model_id: Model ID to check (e.g., "anthropic/claude-3-haiku")
            force_refresh: Skip cache and check live

        Returns:
            True if model is available, False otherwise
        """
        models = self._get_available_models(force_refresh=force_refresh)
        available = model_id in models
        status = "✅" if available else "❌"
        print(f"{status} Model {model_id} available: {available}")
        return available

    def _validate_request_format(self, messages: List[Dict[str, str]]) -> bool:
        """
        Validate that messages follow OpenRouter API spec.

        Required format per OpenRouter docs:
        ```json
        {
          "model": "openai/gpt-4o",
          "messages": [
            {
              "role": "system|user|assistant",
              "content": "text"
            }
          ],
          "max_tokens": 1000,
          "temperature": 0.7
        }
        ```
        """
        if not isinstance(messages, list):
            print(f"❌ Messages must be a list, got {type(messages)}")
            return False

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                print(f"❌ Message {i} is not a dict")
                return False
            if 'role' not in msg:
                print(f"❌ Message {i} missing 'role' field")
                return False
            if 'content' not in msg:
                print(f"❌ Message {i} missing 'content' field")
                return False
            valid_roles = {'system', 'user', 'assistant', 'tool'}
            if msg['role'] not in valid_roles:
                print(f"❌ Message {i} has invalid role '{msg['role']}'")
                return False

        print(f"✅ Request format valid ({len(messages)} messages)")
        return True

    def call_model(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        fallback_to_deepseek: bool = True
    ) -> Dict[str, Any]:
        """
        Call OpenRouter with proper request formatting and failsafe fallback.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (uses first available from preferred list if None)
            max_tokens: Max tokens in response
            temperature: Sampling temperature (0-2)
            fallback_to_deepseek: If True, fall back to DeepSeek chain on failure

        Returns:
            Response dict with 'success', 'data', 'error', and 'model_used'
        """
        # Validate request format
        if not self._validate_request_format(messages):
            return {
                "success": False,
                "error": "Invalid request format",
                "model_used": None
            }

        # Determine models to try
        models_to_try = []
        if model and self._is_model_available(model):
            models_to_try.append(model)
        models_to_try.extend(self.FALLBACK_CHAIN)

        # Remove duplicates while preserving order
        seen = set()
        unique_models = []
        for m in models_to_try:
            if m not in seen:
                unique_models.append(m)
                seen.add(m)
        models_to_try = unique_models

        print(f"\n📋 Model fallback chain: {' → '.join(models_to_try)}")

        # Try each model in sequence
        last_error = None
        for attempt, model_id in enumerate(models_to_try, 1):
            print(f"\n🔄 Attempt {attempt}/{len(models_to_try)}: {model_id}")

            # Verify model is available
            if not self._is_model_available(model_id):
                print(f"   ⏭️  Skipping unavailable model")
                continue

            try:
                # Build request per OpenRouter spec
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                if self.referer:
                    headers["HTTP-Referer"] = self.referer

                payload = {
                    "model": model_id,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False  # Explicitly disable streaming
                }

                print(f"   📤 Request payload keys: {list(payload.keys())}")

                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,  # Let requests handle JSON encoding
                    timeout=30
                )

                print(f"   📥 Response status: {response.status_code}")

                if response.status_code == 200:
                    data = response.json()

                    # Validate response structure
                    if 'choices' in data and data['choices']:
                        message = data['choices'][0].get('message', {})
                        content = message.get('content', '')
                        if content:
                            print(f"   ✅ Success with {model_id}")
                            return {
                                "success": True,
                                "data": data,
                                "model_used": model_id,
                                "content": content
                            }

                    # If we get here, response was 200 but malformed
                    last_error = f"Malformed response: {data}"
                    print(f"   ⚠️  {last_error}")

                elif response.status_code == 429:
                    last_error = f"Rate limit exceeded (429)"
                    print(f"   ⏳ {last_error}")

                elif response.status_code == 401:
                    last_error = f"Invalid API key (401)"
                    print(f"   🔐 {last_error}")
                    return {
                        "success": False,
                        "error": last_error,
                        "model_used": None
                    }

                elif response.status_code in (400, 404):
                    # Model doesn't exist or bad request
                    err_data = response.json() if response.text else {}
                    last_error = err_data.get('error', {}).get('message', str(response.status_code))
                    print(f"   ❌ Model error: {last_error}")

                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:100]}"
                    print(f"   ⚠️  {last_error}")

            except requests.exceptions.Timeout:
                last_error = f"{model_id} timed out"
                print(f"   ⏱️  {last_error}")
            except Exception as e:
                last_error = f"Exception: {str(e)}"
                print(f"   💥 {last_error}")

        # All attempts failed
        return {
            "success": False,
            "error": f"All models failed. Last error: {last_error}",
            "model_used": None
        }


# Example usage
if __name__ == "__main__":
    # Test the client
    api_key = os.getenv("OPENROUTER_API_KEY_1", "YOUR_OPENROUTER_API_KEY")
    client = OpenRouterSafeClient(api_key)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]

    result = client.call_model(messages, model="anthropic/claude-3-haiku")
    print(f"\n\n🎯 Final Result: {json.dumps(result, indent=2)}")

    result = client.call_model(messages, model="anthropic/claude-3-haiku")
    print(f"\n\n🎯 Final Result: {json.dumps(result, indent=2)}")

