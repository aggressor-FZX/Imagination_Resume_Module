import asyncio
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


def test_inferred_skills_from_adjacency(tmp_path, monkeypatch):
    adj = {
        "python": ["django", "flask"],
        "aws": ["lambda", "ecs"]
    }
    p = tmp_path / "skill_adjacency.json"
    p.write_text(json.dumps(adj), encoding="utf-8")

    # Force cwd to tmp_path so loader finds the file
    monkeypatch.chdir(tmp_path)

    result = asyncio.run(
        mod.run_analysis_async(
            resume_text="Python developer with AWS experience",
            job_ad="Backend role requiring Python and AWS",
            extracted_skills_json={"skills": ["python", "aws"]},
            confidence_threshold=0.7
        )
    )

    ps = result.get("processed_skills", {})
    assert "inferred_skills" in ps
    # Should include adjacency-derived items
    assert any(s in ps["inferred_skills"] for s in ["django", "flask", "lambda", "ecs"])


def test_process_structured_skills_confidence_mapping_and_categories():
    skills = {
        "skills": [
            {"skill": "python", "confidence": 0.95, "category": "general"},
            {"skill": "aws", "confidence": 0.65, "category": "general"},
            {"skill": "docker", "confidence": 0.9, "category": "backend"},
            {"skill": "machine learning", "confidence": 0.4, "category": "backend"},
        ]
    }

    processed = mod.process_structured_skills(skills, confidence_threshold=0.7)

    assert processed["high_confidence"] == ["python", "docker"]
    assert processed["medium_confidence"] == ["aws"]
    assert processed["low_confidence"] == ["machine learning"]

    cats = processed["categories"]
    assert "general" in cats and "backend" in cats
    assert cats["general"] == ["python", "aws"]
    assert cats["backend"] == ["docker", "machine learning"]


def test_run_analysis_uses_confidence_mapping(monkeypatch):
    async def fake_loader(text):
        return {"processed_text": text}

    async def fake_fastsvm(resume_text, extract_pdf=False):
        return {"skills": []}

    async def fake_hermes(payload):
        return {"insights": [], "skills": []}

    monkeypatch.setattr(mod, "call_loader_process_text_only", fake_loader)
    monkeypatch.setattr(mod, "call_fastsvm_process_resume", fake_fastsvm)
    monkeypatch.setattr(mod, "call_hermes_extract", fake_hermes)

    extracted = {
        "skills": [
            {"skill": "python", "confidence": 0.9, "category": "general"},
            {"skill": "aws", "confidence": 0.6, "category": "general"},
            {"skill": "docker", "confidence": 0.85, "category": "backend"},
        ]
    }

    result = asyncio.run(
        mod.run_analysis_async(
            resume_text="Python and AWS developer",
            job_ad="Backend role",
            extracted_skills_json=extracted,
            confidence_threshold=0.7,
        )
    )

    ps = result.get("processed_skills", {})
    assert ps["high_confidence"] == ["python", "docker"]
    assert ps["medium_confidence"] == ["aws"]
    assert ps["low_confidence"] == []


def test_run_analysis_maps_fasts_svm_skill_confidences(monkeypatch):
    async def fake_loader(text):
        return {"processed_text": text}

    async def fake_fastsvm(resume_text, extract_pdf=False):
        return {
            "skills": ["python", "aws", "docker", "ml"],
            "skill_confidences": {
                "python": 0.95,
                "docker": 0.9,
                "aws": 0.6,
                "ml": 0.4
            }
        }

    async def fake_hermes(payload):
        return {"insights": [], "skills": []}

    monkeypatch.setattr(mod, "call_loader_process_text_only", fake_loader)
    monkeypatch.setattr(mod, "call_fastsvm_process_resume", fake_fastsvm)
    monkeypatch.setattr(mod, "call_hermes_extract", fake_hermes)

    result = asyncio.run(
        mod.run_analysis_async(
            resume_text="Python and AWS developer",
            job_ad="Backend role",
            extracted_skills_json=None,
            confidence_threshold=0.7,
        )
    )

    ps = result.get("processed_skills", {})
    assert ps["high_confidence"] == ["docker", "python"] or ps["high_confidence"] == ["python", "docker"]
    assert ps["medium_confidence"] == ["aws"]
    assert ps["low_confidence"] == ["ml"]


def test_run_analysis_maps_fasts_svm_dict_items(monkeypatch):
    async def fake_loader(text):
        return {"processed_text": text}

    async def fake_fastsvm(resume_text, extract_pdf=False):
        return {
            "skills": [
                {"skill": "python", "confidence": 0.95, "category": "general"},
                {"skill": "aws", "confidence": 0.6, "category": "general"},
                {"skill": "docker", "confidence": 0.85, "category": "backend"},
                {"skill": "ml", "confidence": 0.4, "category": "backend"},
            ]
        }

    async def fake_hermes(payload):
        return {"insights": [], "skills": []}

    monkeypatch.setattr(mod, "call_loader_process_text_only", fake_loader)
    monkeypatch.setattr(mod, "call_fastsvm_process_resume", fake_fastsvm)
    monkeypatch.setattr(mod, "call_hermes_extract", fake_hermes)

    result = asyncio.run(
        mod.run_analysis_async(
            resume_text="Python and AWS developer",
            job_ad="Backend role",
            extracted_skills_json=None,
            confidence_threshold=0.7,
        )
    )

    ps = result.get("processed_skills", {})
    assert ps["high_confidence"] == ["python", "docker"] or ps["high_confidence"] == ["docker", "python"]
    assert ps["medium_confidence"] == ["aws"]
    assert ps["low_confidence"] == ["ml"]


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


def test_analysis_cache_hit(monkeypatch):
    async def fake_loader(text):
        return {"processed_text": text}

    async def fake_fastsvm(resume_text, extract_pdf=False):
        return {"skills": ["python", "aws"], "skill_confidences": {"python": 0.9, "aws": 0.8}}

    async def fake_hermes(payload):
        return {"insights": [], "skills": ["docker"]}

    monkeypatch.setattr(mod, "call_loader_process_text_only", fake_loader)
    monkeypatch.setattr(mod, "call_fastsvm_process_resume", fake_fastsvm)
    monkeypatch.setattr(mod, "call_hermes_extract", fake_hermes)

    mod.ANALYSIS_CACHE.clear()
    mod.RUN_METRICS["stages"]["analysis"].update({
        "start": None, "end": None, "duration_ms": None, "cache_hit": False
    })

    args = dict(
        resume_text="Python and AWS developer",
        job_ad="Backend role",
        extracted_skills_json=None,
        confidence_threshold=0.7,
    )

    first = asyncio.run(mod.run_analysis_async(**args))
    assert mod.RUN_METRICS["stages"]["analysis"]["cache_hit"] is False

    second = asyncio.run(mod.run_analysis_async(**args))
    assert mod.RUN_METRICS["stages"]["analysis"]["cache_hit"] is True
    assert isinstance(mod.RUN_METRICS["stages"]["analysis"]["duration_ms"], int)
    assert first == second
