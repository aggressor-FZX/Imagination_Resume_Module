import json
import os
import types
import pytest

import imaginator_flow as mod


@pytest.fixture
def sample_analysis_json():
    return {
        "experiences": [],
        "aggregate_skills": ["python", "react", "aws"],
        "processed_skills": {
            "high_confidence_skills": ["python", "react"],
            "medium_confidence_skills": ["aws"],
            "low_confidence_skills": [],
            "skill_confidences": {"python": 0.9, "react": 0.8, "aws": 0.7},
            "categories": {"general": ["python", "react", "aws"]},
            "filtered_count": 3,
            "total_count": 3,
        },
        "domain_insights": {"domain": "tech", "insights": {}},
        # gap_analysis may be a JSON string containing an object with key gap_analysis
        "gap_analysis": json.dumps(
            {
                "gap_analysis": {
                    "critical_gaps": ["react"],
                    "secondary_gaps": ["aws"],
                    "transferable_skills": ["python"],
                }
            }
        ),
    }


def test_run_generation_with_mock_llm(monkeypatch, sample_analysis_json):
    def mock_call_llm(system_prompt, user_prompt, temperature=0.9, max_tokens=1500, max_retries=3, response_format=None, **kwargs):
        # Return generator JSON
        return json.dumps(
            {
                "gap_bridging": [
                    {"skill_focus": "react", "suggestions": ["A", "B"]}
                ],
                "metric_improvements": [
                    {"skill_focus": "python", "suggestions": ["C", "D"]}
                ],
            }
        )

    monkeypatch.setattr(mod, "call_llm", mock_call_llm)

    out = mod.run_generation(sample_analysis_json, "Senior Engineer")
    assert "gap_bridging" in out and isinstance(out["gap_bridging"], list)
    assert "metric_improvements" in out and isinstance(out["metric_improvements"], list)
    assert out["gap_bridging"][0]["skill_focus"] == "react"


def test_run_generation_fallback_on_decode(monkeypatch, sample_analysis_json):
    # Return non-JSON to trigger fallback
    def mock_call_llm(system_prompt, user_prompt, temperature=0.9, max_tokens=1500, max_retries=3, response_format=None, **kwargs):
        return "NOT JSON"

    monkeypatch.setattr(mod, "call_llm", mock_call_llm)
    out = mod.run_generation(sample_analysis_json, "Senior Engineer")
    assert out == {"gap_bridging": [], "metric_improvements": []}


def test_run_criticism_with_mock_llm(monkeypatch):
    generated = {
        "gap_bridging": [{"skill_focus": "react", "suggestions": ["A", "B"]}],
        "metric_improvements": [{"skill_focus": "aws", "suggestions": ["C", "D"]}],
    }

    def mock_call_llm(system_prompt, user_prompt, temperature=0.3, max_tokens=1500, max_retries=3, response_format=None, **kwargs):
        return json.dumps(
            {
                "suggested_experiences": {
                    "bridging_gaps": [
                        {"skill_focus": "react", "refined_suggestions": ["A1", "B1"]}
                    ],
                    "metric_improvements": [
                        {"skill_focus": "aws", "refined_suggestions": ["C1", "D1"]}
                    ],
                }
            }
        )

    monkeypatch.setattr(mod, "call_llm", mock_call_llm)
    out = mod.run_criticism(generated, "Senior Engineer")
    assert "suggested_experiences" in out
    se = out["suggested_experiences"]
    assert se["bridging_gaps"][0]["refined_suggestions"]


def test_run_criticism_fallback_transform(monkeypatch):
    # Force exception in call_llm to hit fallback transform path
    def mock_call_llm(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "call_llm", mock_call_llm)

    generated = {
        "gap_bridging": [{"skill_focus": "react", "suggestions": ["A", "B"]}],
        "metric_improvements": [{"skill_focus": "aws", "suggestions": ["C", "D"]}],
    }

    with pytest.raises(RuntimeError):
        mod.run_criticism(generated, "Senior Engineer")


def test_validate_output_schema_with_fallbacks(monkeypatch):
    # End-to-end object assembled from fallbacks should validate
    output = {
        "experiences": [],
        "aggregate_skills": ["python"],
        "processed_skills": {
            "high_confidence": ["python"],
            "medium_confidence": [],
            "low_confidence": [],
            "inferred_skills": [],
        },
        "domain_insights": {"domain": "tech", "skill_gap_priority": "medium", "insights": []},
        "gap_analysis": json.dumps(
            {"skill_gaps": [], "experience_gaps": [], "recommendations": []}
        ),
        "suggested_experiences": {
            "bridging_gaps": [
                {"skill_focus": "react", "refined_suggestions": ["A1", "B1"]}
            ],
            "metric_improvements": [
                {"skill_focus": "aws", "refined_suggestions": ["C1", "D1"]}
            ],
        },
        "seniority_analysis": {
            "level": "mid-level",
            "confidence": 0.83,
            "total_years_experience": 5.0,
            "experience_quality_score": 0.7,
            "leadership_score": 0.6,
            "skill_depth_score": 0.8,
            "achievement_complexity_score": 0.5,
            "reasoning": "5.0 years of experience demonstrates significant expertise",
            "recommendations": ["Focus on building technical depth", "Seek mentorship opportunities"]
        },
        "run_metrics": {
            "calls": [],
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "failures": []
        },
        "processing_time_seconds": 1.5
    }

    assert mod.validate_output_schema(output) is True


def test_call_llm_switches_models_on_failure(monkeypatch):
    class DummyCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError('model_not_found: missing')
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='SAFE OUTPUT'))],
                usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

    dummy_registry = types.SimpleNamespace(
        marked=[],
        get_candidates=lambda pref: ['qwen/qwen3-30b-a3b', 'anthropic/claude-3-haiku'],
        mark_unhealthy=lambda model: dummy_registry.marked.append(model),
    )

    dummy_client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=DummyCompletions()))
    monkeypatch.setattr(mod, 'openrouter_client', dummy_client)
    monkeypatch.setattr(mod, 'openrouter_model_registry', dummy_registry)

    result = mod.call_llm('Creative system prompt', 'Please generate something imaginative')
    assert result == 'SAFE OUTPUT'
    assert dummy_registry.marked == ['qwen/qwen3-30b-a3b']


@pytest.mark.asyncio
async def test_call_llm_async_retries_gemini(monkeypatch):
    class DummyGeminiResponse:
        def __init__(self, text=None, parts=None):
            self._text = text
            self.usage = types.SimpleNamespace(input_tokens=0, output_tokens=0)
            self.candidates = []
            if parts:
                part_objs = [types.SimpleNamespace(text=part) for part in parts]
                content = types.SimpleNamespace(parts=part_objs)
                self.candidates = [types.SimpleNamespace(content=content)]

        @property
        def text(self):
            if self._text is None:
                raise ValueError('no direct text')
            return self._text

    class DummyGeminiClient:
        def __init__(self, response):
            self._response = response

        async def generate_content_async(self, *args, **kwargs):
            return self._response

    monkeypatch.setattr(mod, 'openrouter_async_client', None)
    empty_response = DummyGeminiResponse(text='')
    rich_response = DummyGeminiResponse(text=None, parts=['final output'])
    clients = [
        ('gemini-2.5-flash', DummyGeminiClient(empty_response)),
        ('gemini-2.5-pro', DummyGeminiClient(rich_response)),
    ]
    monkeypatch.setattr(mod, '_build_google_clients', lambda api_key=None: list(clients))

    result = await mod.call_llm_async('System prompt', 'User prompt')
    assert result == 'final output'
