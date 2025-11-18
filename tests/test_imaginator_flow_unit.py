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
    async def mock_call_llm_async(system_prompt, user_prompt, temperature=0.9, max_tokens=1500, max_retries=3, response_format=None):
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

    monkeypatch.setattr(mod, "call_llm_async", mock_call_llm_async)

    out = mod.run_generation(sample_analysis_json, "Senior Engineer")
    assert "gap_bridging" in out and isinstance(out["gap_bridging"], list)
    assert "metric_improvements" in out and isinstance(out["metric_improvements"], list)
    assert out["gap_bridging"][0]["skill_focus"] == "react"


def test_run_generation_fallback_on_decode(monkeypatch, sample_analysis_json):
    # Return non-JSON to trigger fallback
    async def mock_call_llm_async(system_prompt, user_prompt, temperature=0.9, max_tokens=1500, max_retries=3, response_format=None):
        return "NOT JSON"

    monkeypatch.setattr(mod, "call_llm_async", mock_call_llm_async)
    with pytest.raises(ValueError):
        mod.run_generation(sample_analysis_json, "Senior Engineer")


def test_run_criticism_with_mock_llm(monkeypatch):
    generated = {
        "gap_bridging": [{"skill_focus": "react", "suggestions": ["A", "B"]}],
        "metric_improvements": [{"skill_focus": "aws", "suggestions": ["C", "D"]}],
    }

    async def mock_call_llm_async(system_prompt, user_prompt, temperature=0.3, max_tokens=1500, max_retries=3, response_format=None):
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

    monkeypatch.setattr(mod, "call_llm_async", mock_call_llm_async)
    out = mod.run_criticism(generated, "Senior Engineer")
    assert "suggested_experiences" in out
    se = out["suggested_experiences"]
    assert se["bridging_gaps"][0]["refined_suggestions"]


def test_run_criticism_fallback_transform(monkeypatch):
    # Force exception in call_llm to hit fallback transform path
    async def mock_call_llm_async(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "call_llm_async", mock_call_llm_async)

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
            "high_confidence_skills": ["python"],
            "medium_confidence_skills": [],
            "low_confidence_skills": [],
            "skill_confidences": {"python": 0.9},
            "categories": {"general": ["python"]},
            "filtered_count": 1,
            "total_count": 1,
        },
        "domain_insights": {"domain": "tech", "insights": {}},
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
    }

    assert mod.validate_output_schema(output) is True
