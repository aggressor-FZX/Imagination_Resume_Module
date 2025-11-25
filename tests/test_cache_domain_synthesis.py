import asyncio
import json
import sys
import types
from unittest.mock import AsyncMock, patch

# Minimal stubs to satisfy imaginator_flow imports at import-time
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
class ClientSession:  # type: ignore
    pass
aiohttp_module.ClientSession = ClientSession
sys.modules["aiohttp"] = aiohttp_module

# Stub config.settings
config_module = types.ModuleType("config")
class _DummySettings:
    IMAGINATOR_AUTH_TOKEN = "test-api-key"
    openrouter_api_key_1 = "test-openrouter-key-1"
    openrouter_api_key_2 = "test-openrouter-key-2"
    environment = "test"
    @property
    def is_production(self):
        return False
    @property
    def is_development(self):
        return True
settings = _DummySettings()
config_module.settings = settings
sys.modules["config"] = config_module

import imaginator_flow as mod


def _reset_metrics():
    mod.RUN_METRICS.update({
        "calls": [],
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "failures": [],
        "stages": {
            "analysis": {"start": None, "end": None, "duration_ms": None, "cache_hit": False},
            "generation": {"start": None, "end": None, "duration_ms": None, "cache_hit": False},
            "synthesis": {"start": None, "end": None, "duration_ms": None, "cache_hit": False},
            "criticism": {"start": None, "end": None, "duration_ms": None, "cache_hit": False},
        },
    })


def test_analysis_cache_hit_sets_metrics(monkeypatch):
    async def fake_loader(text):
        return {"processed_text": text}

    async def fake_fastsvm(text, extract_pdf=False):
        return {"skills": ["python", "aws"]}

    async def fake_hermes(payload):
        return {}

    _reset_metrics()
    mod.ANALYSIS_CACHE.clear()
    monkeypatch.setattr(mod, "call_loader_process_text_only", fake_loader)
    monkeypatch.setattr(mod, "call_fastsvm_process_resume", fake_fastsvm)
    monkeypatch.setattr(mod, "call_hermes_extract", fake_hermes)

    first = asyncio.run(
        mod.run_analysis_async(
            resume_text="Python developer",
            job_ad="Backend role",
            extracted_skills_json=None,
            domain_insights_json=None,
            confidence_threshold=0.7,
        )
    )

    second = asyncio.run(
        mod.run_analysis_async(
            resume_text="Python developer",
            job_ad="Backend role",
            extracted_skills_json=None,
            domain_insights_json=None,
            confidence_threshold=0.7,
        )
    )

    assert second == first
    s = mod.RUN_METRICS["stages"]["analysis"]
    assert s["cache_hit"] is True
    assert s["start"] is not None and s["end"] is not None
    assert isinstance(s["duration_ms"], int)


def test_domain_insights_defaults(monkeypatch):
    async def fake_loader(text):
        return {"processed_text": text}

    async def fake_fastsvm(text, extract_pdf=False):
        return {"skills": ["python"]}

    async def fake_hermes(payload):
        return {}

    _reset_metrics()
    mod.ANALYSIS_CACHE.clear()
    monkeypatch.setattr(mod, "call_loader_process_text_only", fake_loader)
    monkeypatch.setattr(mod, "call_fastsvm_process_resume", fake_fastsvm)
    monkeypatch.setattr(mod, "call_hermes_extract", fake_hermes)

    result = asyncio.run(
        mod.run_analysis_async(
            resume_text="Resume",
            job_ad="Job",
            extracted_skills_json=None,
            domain_insights_json=None,
            confidence_threshold=0.7,
        )
    )

    di = result.get("domain_insights", {})
    assert di.get("domain") == "unknown"
    assert di.get("market_demand") == "Unknown"
    assert di.get("skill_gap_priority") == "Medium"
    assert isinstance(di.get("emerging_trends"), list)
    assert isinstance(di.get("insights"), list)


def test_synthesis_convenience_parses_json(monkeypatch):
    async def fake_llm(*args, **kwargs):
        return json.dumps({"final_written_section": "Refined section"})

    _reset_metrics()
    with patch("imaginator_flow.call_llm_async", new=AsyncMock(side_effect=fake_llm)):
        result = asyncio.run(
            mod.run_synthesis_async(
                generated_text={"generated_text": "X", "critique": {"c": 1}},
                job_ad="Job",
                openrouter_api_keys=[],
            )
        )

    assert isinstance(result, dict)
    assert result.get("final_written_section") == "Refined section"


def test_synthesis_convenience_wraps_string(monkeypatch):
    async def fake_llm(*args, **kwargs):
        return "Plain section"

    _reset_metrics()
    with patch("imaginator_flow.call_llm_async", new=AsyncMock(side_effect=fake_llm)):
        result = asyncio.run(
            mod.run_synthesis_async(
                generated_text={"generated_text": "Y", "critique": {"c": 2}},
                job_ad="Job",
                openrouter_api_keys=[],
            )
        )

    assert isinstance(result, dict)
    assert result.get("final_written_section") == "Plain section"
