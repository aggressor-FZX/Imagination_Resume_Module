import asyncio
import json
import sys
import types
from unittest.mock import AsyncMock, patch

# Stubs to satisfy imaginator_flow imports
google_module = types.ModuleType("google")
genai_module = types.ModuleType("google.generativeai")
genai_module.configure = lambda api_key=None: None
genai_module.models = types.SimpleNamespace(get=lambda name: None)
genai_module.GenerativeModel = type("GenerativeModel", (), {})
sys.modules["google"] = google_module
sys.modules["google.generativeai"] = genai_module

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

aiohttp_module = types.ModuleType("aiohttp")
sys.modules["aiohttp"] = aiohttp_module

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

from imaginator_flow import run_generation_async, run_synthesis_async


def test_gap_analysis_embedded_in_generation_prompt():
    captured = {}

    async def fake_llm(system_prompt, user_prompt, **kwargs):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return "Generated"

    analysis = {
        "skills": ["Python"],
        "seniority": "mid",
        "gap_analysis": {
            "critical_gaps": ["AWS"],
            "recommendations": ["Gain cloud cert"]
        }
    }

    with patch("imaginator_flow.call_llm_async", new=AsyncMock(side_effect=fake_llm)):
        result = asyncio.run(run_generation_async(analysis_json=analysis, job_ad="Data Engineer"))

    assert result == "Generated"
    assert json.dumps(analysis, indent=2) in captured["user"]
    assert "gap_analysis" in captured["user"]
