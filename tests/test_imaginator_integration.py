"""
Unit tests for Imaginator new pipeline integration with Career Alchemy
Tests Phase 2.1: Imaginator Pipeline Integration
"""

import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import json

# Set environment variables before importing modules that depend on config
os.environ.setdefault("OPENROUTER_API_KEY_1", "test-openrouter-key-1")
os.environ.setdefault("OPENROUTER_API_KEY_2", "test-openrouter-key-2")
os.environ.setdefault("IMAGINATOR_AUTH_TOKEN", "test-api-key")

from imaginator_new_integration import run_new_pipeline_async, CAREER_ALCHEMY_AVAILABLE


@pytest.mark.unit
class TestCareerAlchemyIntegration:
    """Test Career Alchemy integration in new pipeline"""

    @pytest.fixture
    def sample_resume_text(self):
        """Sample resume text for testing"""
        return """
        John Doe
        Software Engineer
        
        Experience:
        - Senior Software Engineer at Tech Corp (2020-2024)
          Developed web applications using Python, React, and AWS
          Led team of 5 engineers
        
        Skills: Python, JavaScript, React, AWS, Docker, Kubernetes
        
        Education:
        - BS Computer Science, University of Tech (2016-2020)
        """

    @pytest.fixture
    def sample_job_ad(self):
        """Sample job ad for testing"""
        return """
        Senior Software Engineer
        
        We are looking for an experienced software engineer with:
        - 5+ years of experience
        - Strong Python and JavaScript skills
        - AWS cloud experience
        - Leadership experience
        """

    @pytest.fixture
    def mock_orchestrator_result(self):
        """Mock orchestrator pipeline result"""
        return {
            "final_output": {
                "final_written_section_markdown": "## Experience\n\nSenior Software Engineer...",
                "final_written_section": "Senior Software Engineer...",
                "editorial_notes": "Good resume",
                "seniority_level": "senior",
                "domain_terms_used": ["Python", "AWS"],
                "quantification_analysis": {},
                "hallucination_checked": True,
            },
            "stages": {
                "researcher": {
                    "data": {
                        "implied_skills": ["Python", "JavaScript"],
                        "domain_vocab": ["AWS", "Docker"],
                        "implied_metrics": ["performance", "scalability"],
                        "insider_tips": "Focus on cloud certifications",
                        "work_archetypes": ["Full-stack Developer"],
                    }
                }
            },
            "seniority_analysis": {
                "level": "senior",
                "job_zone": "4",
                "experience_required": "5-10 years",
            },
            "metrics": {
                "pipeline_status": "completed",
                "total_duration_seconds": 2.5,
                "stage_durations": {"researcher": 1.0, "drafter": 0.8, "editor": 0.7},
            },
            "errors": [],
        }

    @pytest.fixture
    def mock_career_alchemy_data(self):
        """Mock career alchemy response"""
        return {
            "profile": {
                "canonical_title": "Senior Software Engineer",
                "dynamic_title": "The Master Architect",
                "primary_stats": {
                    "INT": 85,
                    "DEX": 70,
                    "WIS": 80,
                    "CHA": 75,
                    "VIT": 90,
                    "ARC": 65,
                },
                "hero_class": "The Architect",
                "hybrid_archetype": None,
                "seniority": {
                    "tier": "Master",
                    "level": 75,
                    "xp": 8.5,
                },
                "domain_aura": {
                    "name": "Technology",
                    "color": "#00d4ff",
                },
                "rarity": {
                    "score": 0.85,
                    "percentile": "Top 15%",
                },
            },
            "share_content": {
                "twitter": "Check out my Career Alchemy! #CareerAlchemy",
                "linkedin": "Discover your professional signature...",
            },
            "wrapped_sequence": [
                {"slide_number": 1, "title": "THE IDENTIFICATION"},
                {"slide_number": 2, "title": "THE ALCHEMY OF YOU"},
            ],
        }

    @pytest.mark.asyncio
    @patch("imaginator_new_integration.PipelineOrchestrator")
    @patch("imaginator_new_integration.LLMClientAdapter")
    async def test_career_alchemy_field_present_when_available(
        self,
        mock_llm_client_class,
        mock_orchestrator_class,
        sample_resume_text,
        sample_job_ad,
        mock_orchestrator_result,
        mock_career_alchemy_data,
    ):
        """Test that career_alchemy field is present in response when generation succeeds"""
        # Skip if career_alchemy module not available
        if not CAREER_ALCHEMY_AVAILABLE:
            pytest.skip("Career Alchemy module not available")

        # Setup mocks
        mock_llm_client = MagicMock()
        mock_llm_client.call_llm_async = AsyncMock(return_value='{"job_title": "Senior Software Engineer"}')
        mock_llm_client.get_usage_stats = MagicMock(
            return_value={
                "calls": [],
                "total_prompt_tokens": 1000,
                "total_completion_tokens": 500,
                "total_tokens": 1500,
                "estimated_cost_usd": 0.05,
                "failures": [],
            }
        )
        mock_llm_client_class.return_value = mock_llm_client

        mock_orchestrator = MagicMock()
        mock_orchestrator.run_pipeline = AsyncMock(return_value=mock_orchestrator_result)
        mock_orchestrator_class.return_value = mock_orchestrator

        # Mock career_alchemy generation
        with patch("imaginator_new_integration.generate_career_alchemy") as mock_generate:
            mock_generate.return_value = mock_career_alchemy_data

            # Run pipeline
            result = await run_new_pipeline_async(
                resume_text=sample_resume_text,
                job_ad=sample_job_ad,
                openrouter_api_keys=["test-key"],
                location="United States",
            )

            # Assertions
            assert "career_alchemy" in result, "career_alchemy field should be present in response"
            assert result["career_alchemy"] == mock_career_alchemy_data, "career_alchemy data should match"
            assert result["career_alchemy"]["profile"]["canonical_title"] == "Senior Software Engineer"
            assert "primary_stats" in result["career_alchemy"]["profile"]
            assert "wrapped_sequence" in result["career_alchemy"]

            # Verify generate_career_alchemy was called with correct parameters
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            assert "characteristics" in call_args.kwargs
            assert call_args.kwargs["location"] == "United States"

    @pytest.mark.asyncio
    @patch("imaginator_new_integration.PipelineOrchestrator")
    @patch("imaginator_new_integration.LLMClientAdapter")
    async def test_graceful_failure_on_career_alchemy_error(
        self,
        mock_llm_client_class,
        mock_orchestrator_class,
        sample_resume_text,
        sample_job_ad,
        mock_orchestrator_result,
    ):
        """Test that pipeline continues gracefully when career_alchemy generation fails"""
        # Skip if career_alchemy module not available
        if not CAREER_ALCHEMY_AVAILABLE:
            pytest.skip("Career Alchemy module not available")

        # Setup mocks
        mock_llm_client = MagicMock()
        mock_llm_client.call_llm_async = AsyncMock(return_value='{"job_title": "Senior Software Engineer"}')
        mock_llm_client.get_usage_stats = MagicMock(
            return_value={
                "calls": [],
                "total_prompt_tokens": 1000,
                "total_completion_tokens": 500,
                "total_tokens": 1500,
                "estimated_cost_usd": 0.05,
                "failures": [],
            }
        )
        mock_llm_client_class.return_value = mock_llm_client

        mock_orchestrator = MagicMock()
        mock_orchestrator.run_pipeline = AsyncMock(return_value=mock_orchestrator_result)
        mock_orchestrator_class.return_value = mock_orchestrator

        # Mock career_alchemy generation to raise an error
        with patch("imaginator_new_integration.generate_career_alchemy") as mock_generate:
            mock_generate.side_effect = Exception("Career Alchemy generation failed")

            # Run pipeline
            result = await run_new_pipeline_async(
                resume_text=sample_resume_text,
                job_ad=sample_job_ad,
                openrouter_api_keys=["test-key"],
            )

            # Assertions - pipeline should complete successfully even if career_alchemy fails
            assert result is not None, "Pipeline should return a result"
            assert "final_written_section_markdown" in result, "Core fields should be present"
            assert "career_alchemy" not in result, "career_alchemy should not be present on error"

    @pytest.mark.asyncio
    @patch("imaginator_new_integration.PipelineOrchestrator")
    @patch("imaginator_new_integration.LLMClientAdapter")
    async def test_career_alchemy_not_present_when_module_unavailable(
        self,
        mock_llm_client_class,
        mock_orchestrator_class,
        sample_resume_text,
        sample_job_ad,
        mock_orchestrator_result,
    ):
        """Test that career_alchemy is not present when module is unavailable"""
        # Setup mocks
        mock_llm_client = MagicMock()
        mock_llm_client.call_llm_async = AsyncMock(return_value='{"job_title": "Senior Software Engineer"}')
        mock_llm_client.get_usage_stats = MagicMock(
            return_value={
                "calls": [],
                "total_prompt_tokens": 1000,
                "total_completion_tokens": 500,
                "total_tokens": 1500,
                "estimated_cost_usd": 0.05,
                "failures": [],
            }
        )
        mock_llm_client_class.return_value = mock_llm_client

        mock_orchestrator = MagicMock()
        mock_orchestrator.run_pipeline = AsyncMock(return_value=mock_orchestrator_result)
        mock_orchestrator_class.return_value = mock_orchestrator

        # Mock CAREER_ALCHEMY_AVAILABLE to be False
        with patch("imaginator_new_integration.CAREER_ALCHEMY_AVAILABLE", False):
            # Run pipeline
            result = await run_new_pipeline_async(
                resume_text=sample_resume_text,
                job_ad=sample_job_ad,
                openrouter_api_keys=["test-key"],
            )

            # Assertions
            assert result is not None, "Pipeline should return a result"
            assert "career_alchemy" not in result, "career_alchemy should not be present when module unavailable"
            assert "final_written_section_markdown" in result, "Core fields should still be present"

    @pytest.mark.asyncio
    @patch("imaginator_new_integration.PipelineOrchestrator")
    @patch("imaginator_new_integration.LLMClientAdapter")
    async def test_career_alchemy_characteristics_built_correctly(
        self,
        mock_llm_client_class,
        mock_orchestrator_class,
        sample_resume_text,
        sample_job_ad,
        mock_orchestrator_result,
        mock_career_alchemy_data,
    ):
        """Test that characteristics dict is built correctly from pipeline result"""
        # Skip if career_alchemy module not available
        if not CAREER_ALCHEMY_AVAILABLE:
            pytest.skip("Career Alchemy module not available")

        # Setup mocks
        mock_llm_client = MagicMock()
        mock_llm_client.call_llm_async = AsyncMock(return_value='{"job_title": "Senior Software Engineer"}')
        mock_llm_client.get_usage_stats = MagicMock(
            return_value={
                "calls": [],
                "total_prompt_tokens": 1000,
                "total_completion_tokens": 500,
                "total_tokens": 1500,
                "estimated_cost_usd": 0.05,
                "failures": [],
            }
        )
        mock_llm_client_class.return_value = mock_llm_client

        mock_orchestrator = MagicMock()
        mock_orchestrator.run_pipeline = AsyncMock(return_value=mock_orchestrator_result)
        mock_orchestrator_class.return_value = mock_orchestrator

        # Mock career_alchemy generation and capture call arguments
        with patch("imaginator_new_integration.generate_career_alchemy") as mock_generate:
            mock_generate.return_value = mock_career_alchemy_data

            # Run pipeline
            await run_new_pipeline_async(
                resume_text=sample_resume_text,
                job_ad=sample_job_ad,
                openrouter_api_keys=["test-key"],
                location="Seattle, WA",
            )

            # Verify characteristics dict structure
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            characteristics = call_args.kwargs["characteristics"]

            assert "canonical_title" in characteristics
            assert "job_title" in characteristics
            assert "domain" in characteristics
            assert "seniority" in characteristics
            assert "skills" in characteristics
            assert isinstance(characteristics["skills"], list)
            assert "experience_years" in characteristics
            assert "certifications" in characteristics
            assert "education" in characteristics
            assert "achievements" in characteristics

            # Verify location is passed correctly
            assert call_args.kwargs["location"] == "Seattle, WA"
