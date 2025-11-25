import pytest
from unittest.mock import patch, AsyncMock
import os


@pytest.fixture(scope="session", autouse=True)
def mock_config_settings():
    """Session-scoped fixture to mock config settings before any modules are imported."""
    # Set environment variables to override config.py settings
    os.environ["IMAGINATOR_AUTH_TOKEN"] = "test-api-key"
    os.environ["OPENROUTER_API_KEY_1"] = "test-openrouter-key-1"
    os.environ["OPENROUTER_API_KEY_2"] = "test-openrouter-key-2"
    
    # Patch the flow functions to avoid live API calls
    try:
        with patch('config.settings.IMAGINATOR_AUTH_TOKEN', 'test-api-key'):
            with patch('config.settings.openrouter_api_key_1', 'test-openrouter-key-1'):
                with patch('config.settings.openrouter_api_key_2', 'test-openrouter-key-2'):
                    with patch('app.run_analysis_async', new_callable=AsyncMock) as mock_analysis:
                        with patch('app.run_generation_async', new_callable=AsyncMock) as mock_generation:
                            with patch('app.run_criticism_async', new_callable=AsyncMock) as mock_criticism:
                                mock_analysis.return_value = {
                                    "experiences": [
                                        {
                                            "title_line": "Software Engineer at Tech Corp",
                                            "skills": ["Python", "JavaScript", "React"],
                                            "snippet": "Developed web applications using Python and React"
                                        }
                                    ],
                                    "aggregate_skills": ["Python", "JavaScript", "React", "Team Collaboration"],
                                    "processed_skills": {
                                        "high_confidence": ["Python", "JavaScript"],
                                        "medium_confidence": ["React"],
                                        "low_confidence": ["Team Collaboration"],
                                        "inferred_skills": ["Web Development", "Problem Solving"]
                                    },
                                    "domain_insights": {
                                        "domain": "Software Development",
                                        "market_demand": "High",
                                        "skill_gap_priority": "Medium",
                                        "emerging_trends": ["AI/ML", "Cloud Computing"],
                                        "insights": ["Strong demand for full-stack developers"]
                                    },
                                    "gap_analysis": "Candidate has strong technical foundation but lacks cloud platform experience and advanced database design skills.",
                                    "suggested_experiences": {
                                        "technical_gaps": [
                                            {
                                                "suggestion": "Gain AWS or GCP certification",
                                                "relevance": "High",
                                                "implementation": "Complete cloud certification and build projects"
                                            }
                                        ],
                                        "experience_gaps": [
                                            {
                                                "suggestion": "Lead database design initiatives",
                                                "relevance": "Medium",
                                                "implementation": "Take ownership of database schema design"
                                            }
                                        ]
                                    },
                                    "seniority_analysis": {
                                        "level": "Mid-Level",
                                        "confidence": 0.85,
                                        "total_years_experience": 3.5,
                                        "experience_quality_score": 0.8,
                                        "leadership_score": 0.6,
                                        "skill_depth_score": 0.75,
                                        "achievement_complexity_score": 0.7,
                                        "reasoning": "Strong technical skills with some leadership experience",
                                        "recommendations": ["Focus on cloud platforms", "Develop leadership skills"]
                                    },
                                    "run_metrics": {
                                        "calls": [],
                                        "total_prompt_tokens": 1000,
                                        "total_completion_tokens": 500,
                                        "total_tokens": 1500,
                                        "estimated_cost_usd": 0.05,
                                        "failures": []
                                    },
                                    "processing_status": "completed",
                                    "processing_time_seconds": 1.5
                                }
                                mock_generation.return_value = {"status": "success", "suggestions": "Mocked suggestions"}
                                mock_criticism.return_value = {"status": "success", "critique": "Mocked critique"}
                                yield
    except ModuleNotFoundError:
        yield
    
    # Clean up environment variables after tests
    for key in ["IMAGINATOR_AUTH_TOKEN", "OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2"]:
        if key in os.environ:
            del os.environ[key]
