import asyncio
import json
import sys
import types
from unittest.mock import AsyncMock, patch

import pytest

# Stub google.generativeai to satisfy imaginator_flow import without requiring the package
google_module = types.ModuleType("google")
genai_module = types.ModuleType("google.generativeai")
genai_module.configure = lambda api_key=None: None
genai_module.models = types.SimpleNamespace(get=lambda name: None)
genai_module.GenerativeModel = type("GenerativeModel", (), {})
sys.modules["google"] = google_module
sys.modules["google.generativeai"] = genai_module

# Stub openai OpenAI/AsyncOpenAI to avoid dependency during import
openai_module = types.ModuleType("openai")

class _DummyCompletions:
    def create(self, **kwargs):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="DUMMY"))],
            usage=types.SimpleNamespace(prompt_tokens=0, completion_tokens=0),
        )

class _DummyChat:
    def __init__(self):
        self.completions = _DummyCompletions()

class OpenAI:
    def __init__(self, base_url=None, api_key=None, default_headers=None):
        self.api_key = api_key
        self.chat = _DummyChat()

class AsyncOpenAI(OpenAI):
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        return False

openai_module.OpenAI = OpenAI
openai_module.AsyncOpenAI = AsyncOpenAI
sys.modules["openai"] = openai_module

# Stub aiohttp to avoid import-time dependency
aiohttp_module = types.ModuleType("aiohttp")
sys.modules["aiohttp"] = aiohttp_module

# Stub config.settings to avoid pydantic_settings dependency during import
config_module = types.ModuleType("config")

class _DummySettings:
    IMAGINATOR_AUTH_TOKEN = "test-api-key"
    openrouter_api_key_1 = "test-openrouter-key-1"
    openrouter_api_key_2 = "test-openrouter-key-2"
    environment = "test"
    ENABLE_LOADER = False
    ENABLE_FASTSVM = False
    ENABLE_HERMES = False
    CORS_ORIGINS = "*"

    @property
    def is_production(self):
        return False

    @property
    def is_development(self):
        return True

    @property
    def api_key(self):
        return self.IMAGINATOR_AUTH_TOKEN

settings = _DummySettings()
config_module.settings = settings
sys.modules["config"] = config_module

from imaginator_flow import run_generation, run_generation_async, run_synthesis_async


def test_run_synthesis_async_uses_critique():
    captured = {}

    async def side_effect(system_prompt, user_prompt, **kwargs):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return "Final polished resume section"

    with patch("imaginator_flow.call_llm_async", new=AsyncMock(side_effect=side_effect)):
        generated_text = "Led a team to deliver scalable ML pipelines"
        critique_json = {"feedback": "Strengthen metrics and align to job requirements"}
        job_ad = "Senior ML Engineer with leadership and AWS experience"
        result = asyncio.run(
            run_synthesis_async(
                generated_text=generated_text,
                critique_json=critique_json,
                job_ad=job_ad,
            )
        )

    assert result == "Final polished resume section"
    assert "final resume writing agent" in captured["system"].lower()
    assert "Critique Feedback:" in captured["user"]
    assert json.dumps(critique_json, indent=2) in captured["user"]


def test_run_generation_async_contains_car_instruction():
    captured = {}

    async def side_effect(system_prompt, user_prompt, **kwargs):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return "Generated work experience entry"

    analysis = {"skills": ["Python", "AWS"], "seniority": "senior"}
    job_ad = "Senior Backend Engineer with Python and AWS"

    with patch("imaginator_flow.call_llm_async", new=AsyncMock(side_effect=side_effect)):
        result = asyncio.run(
            run_generation_async(
                analysis_json=analysis,
                job_ad=job_ad,
            )
        )

    assert result == "Generated work experience entry"
    assert "challenge–action–result" in captured["system"].lower()
    assert "Target Job Description:" in captured["user"]
    assert json.dumps(analysis, indent=2) in captured["user"]


def test_run_generation_sync_fallback_structure_on_llm_failure():
    def failing_call(*args, **kwargs):
        raise RuntimeError("LLM failure")

    analysis = {"skills": ["SQL"], "seniority": "mid"}
    job_ad = "Data Engineer role requiring SQL"

    with patch("imaginator_flow.call_llm", new=failing_call):
        result = run_generation(analysis_json=analysis, job_ad=job_ad)

    assert isinstance(result, dict)
    assert "gap_bridging" in result
    assert "metric_improvements" in result
    assert result["gap_bridging"] == []
    assert result["metric_improvements"] == []
