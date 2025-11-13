"""
Tests for FastAPI endpoints
Based on Context7 research findings for pytest-asyncio and httpx
"""

import pytest
import json
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from app import app
from models import AnalysisRequest, ProcessingStatus
from config import settings


client = TestClient(app, headers={"X-API-Key": settings.API_KEY})


@pytest.mark.unit
class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check(self):
        """Test basic health check"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "environment" in data


@pytest.mark.unit
class TestConfigEndpoint:
    """Test configuration endpoint"""

    def test_get_config(self):
        """Test configuration retrieval"""
        response = client.get("/config")
        assert response.status_code == 200

        data = response.json()
        assert "environment" in data
        assert "confidence_threshold" in data
        # Using OpenRouter as single provider
        assert isinstance(data.get("has_openrouter_key"), bool)


@pytest.mark.unit
class TestAnalysisEndpoint:
    """Test resume analysis endpoint"""

    @patch('app.run_analysis_async')
    @patch('app.run_generation_async')
    @patch('app.run_criticism')
    @patch('app.validate_output_schema')
    def test_successful_analysis(self, mock_validate, mock_criticism, mock_generation, mock_analysis):
        """Test successful resume analysis"""
        # Mock the analysis functions
        mock_analysis.return_value = {
            "experiences": [],
            "aggregate_skills": ["Python", "SQL"],
            "processed_skills": {
                "high_confidence": ["Python"],
                "medium_confidence": ["SQL"],
                "low_confidence": [],
                "inferred_skills": []
            },
            "domain_insights": {
                "domain": "software",
                "market_demand": "high",
                "skill_gap_priority": "medium"
            },
            "gap_analysis": "Test gap analysis",
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
            }
        }

        mock_generation.return_value = {
            "gap_bridging": ["Learn AWS"],
            "metric_improvements": ["Add metrics"]
        }

        mock_criticism.return_value = {
            "suggested_experiences": {
                "bridging_gaps": ["Learn AWS"],
                "metric_improvements": ["Add metrics"]
            }
        }

        mock_validate.return_value = None  # No validation errors

        # Test request
        request_data = {
            "resume_text": "John Doe\nSoftware Engineer\nPython, SQL",
            "job_ad": "Senior Developer\nPython, AWS, SQL required",
            "confidence_threshold": 0.7
        }

        response = client.post("/analyze", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "experiences" in data
        assert "aggregate_skills" in data
        assert "run_metrics" in data
        assert data["processing_status"] == ProcessingStatus.COMPLETED
        assert "processing_time_seconds" in data

    def test_invalid_json_input(self):
        """Test invalid JSON in optional fields"""
        request_data = {
            "resume_text": "Test resume",
            "job_ad": "Test job",
            "extracted_skills_json": "invalid json {"
        }

        response = client.post("/analyze", json=request_data)
        assert response.status_code == 422  # Pydantic validation error
        assert "extracted_skills_json" in str(response.json())

    def test_missing_required_fields(self):
        """Test missing required fields"""
        request_data = {
            "resume_text": "Test resume"
            # Missing job_ad
        }

        response = client.post("/analyze", json=request_data)
        assert response.status_code == 422  # Pydantic validation error

    @patch('app.run_analysis_async')
    @patch('app.run_generation_async')
    @patch('app.run_criticism')
    @patch('app.validate_output_schema')
    def test_analysis_failure(self, mock_validate, mock_criticism, mock_generation, mock_analysis):
        """Test graceful handling of analysis failure"""
        mock_analysis.side_effect = Exception("API Error")
        mock_generation.return_value = {"gap_bridging": [], "metric_improvements": []}
        mock_criticism.return_value = {"suggested_experiences": {"bridging_gaps": [], "metric_improvements": []}}
        mock_validate.return_value = None

        request_data = {
            "resume_text": "This is a longer test resume text that meets the minimum length requirement for validation",
            "job_ad": "This is a longer test job description that also meets the minimum length requirement"
        }

        response = client.post("/analyze", json=request_data)
        # The analysis failure should result in a 500 error after validation passes
        assert response.status_code == 500

        data = response.json()
        assert "error" in data
        assert "ANALYSIS_FAILED" in data["error_code"]


@pytest.mark.unit
class TestJSONIntegration:
    """Test FrontEnd JSON integration (no file upload endpoint)"""

    @patch('app.run_analysis_async')
    @patch('app.run_generation_async')
    @patch('app.run_criticism')
    @patch('app.validate_output_schema')
    def test_json_analysis(self, mock_validate, mock_criticism, mock_generation, mock_analysis):
        """Test resume analysis via JSON payload (FrontEnd integration)"""
        # Mock functions
        mock_analysis.return_value = {
            "experiences": [],
            "aggregate_skills": ["Python"],
            "processed_skills": {"high_confidence": ["Python"], "medium_confidence": [], "low_confidence": [], "inferred_skills": []},
            "domain_insights": {"domain": "tech", "market_demand": "high", "skill_gap_priority": "low"},
            "gap_analysis": "Good fit",
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
            }
        }
        mock_generation.return_value = {"gap_bridging": [], "metric_improvements": []}
        mock_criticism.return_value = {"suggested_experiences": {"bridging_gaps": [], "metric_improvements": []}}
        mock_validate.return_value = None

        # FrontEnd sends structured JSON to /analyze endpoint
        request_body = {
            "resume_text": "John Doe\nSoftware Engineer\nPython experience",
            "job_ad": "Senior Developer position requiring extensive Python experience and cloud platform knowledge",
            "confidence_threshold": 0.8
        }

        response = client.post("/analyze", json=request_body)
        assert response.status_code == 200

        result = response.json()
        assert "experiences" in result
        assert "run_metrics" in result


@pytest.mark.unit
class TestContext7Endpoint:
    """Test Context7 documentation endpoint"""

    @patch('app.settings')
    def test_context7_not_configured(self, mock_settings):
        """Test Context7 endpoint when not configured"""
        mock_settings.context7_api_key = None

        response = client.get("/docs/fastapi")
        assert response.status_code == 503
        assert "Context7 integration not configured" in response.json()["detail"]

    @patch('app.settings')
    def test_context7_success(self, mock_settings):
        """Test successful Context7 documentation retrieval"""
        # Mock settings
        mock_settings.context7_api_key = "test-key"

        # Mock the context7 module
        mock_context7 = MagicMock()
        mock_client = MagicMock()
        
        # Mock async get_docs method
        async def mock_get_docs(*args, **kwargs):
            return {"docs": "FastAPI documentation"}
        
        mock_client.get_docs = mock_get_docs
        mock_context7.Context7Client.return_value = mock_client

        with patch.dict('sys.modules', {'context7': mock_context7}):
            response = client.get("/docs/fastapi?version=0.100.0")
            assert response.status_code == 200

            data = response.json()
            assert data["library"] == "fastapi"
            assert data["version"] == "0.100.0"
            assert "documentation" in data

    @patch('app.settings')
    def test_context7_import_error(self, mock_settings):
        """Test Context7 when library not installed"""
        mock_settings.context7_api_key = "test-key"

        # Mock import error
        with patch.dict('sys.modules', {'context7': None}):
            response = client.get("/docs/fastapi")
            assert response.status_code == 503
            assert "Context7 MCP not installed" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])