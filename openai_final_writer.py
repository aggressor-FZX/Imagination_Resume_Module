"""
Helper functions to use OpenAI Chat models (gpt-5.2-chat) for final resume writing.
"""
from typing import Optional, Dict, Any
import time
from openai import AsyncOpenAI, OpenAI


def call_openai_sync(system_prompt: str, user_prompt: str, openai_api_key: Optional[str] = None, model: str = "gpt-5.2-chat", temperature: float = 0.3, max_tokens: int = 1500) -> str:
    """Sync OpenAI call - wrapper for simple integration/testing."""
    client = OpenAI(api_key=openai_api_key) if openai_api_key else OpenAI()
    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    duration = time.time() - start
    try:
        return resp.choices[0].message.content
    except Exception:
        return ""


async def call_openai_async(system_prompt: str, user_prompt: str, openai_api_key: Optional[str] = None, model: str = "gpt-5.2-chat", temperature: float = 0.3, max_tokens: int = 1500) -> str:
    """Async OpenAI call using AsyncOpenAI client. Returns text content."""
    client = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else AsyncOpenAI()
    start = time.time()
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    try:
        return resp.choices[0].message.content
    except Exception:
        return ""
