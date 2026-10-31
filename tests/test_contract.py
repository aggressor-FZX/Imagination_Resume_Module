import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from typing import Dict, Any, Optional

from models import AnalysisRequest, CreativityMode
from app import app

client = TestClient(app)

@pytest.fixture
def valid_request():
    # Phase 2: `resume_text` is a required field on AnalysisRequest but was
    # absent from this fixture, so the "valid request" case failed schema
    # validation. That was stale test data, not a schema defect.
    return {
        "resume_text": (
            "John Doe\nSenior Python Developer\n\n"
            "PROFESSIONAL EXPERIENCE\n"
            "Acme Corp - San Francisco, CA | 2019 - Present\n"
            "- Built web services in Python and Django\n"
        ),
        "user_profile": {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "+1234567890",
            "location": "San Francisco, CA",
            "experience": [
                {
                    "company": "Tech Corp",
                    "role": "Software Engineer",
                    "duration": "2018-2022",
                    "description": "Developed web applications using Python and JavaScript",
                    "skills": ["Python", "JavaScript", "React", "Django"]
                }
            ],
            "education": [
                {
                    "degree": "Bachelor of Science",
                    "field": "Computer Science",
                    "institution": "State University",
                    "year": 2018
                }
            ],
            "skills": ["Python", "JavaScript", "React", "Django", "AWS", "Docker"],
            "certifications": [
                {
                    "name": "AWS Certified Developer",
                    "issuer": "Amazon Web Services",
                    "year": 2021
                }
            ]
        },
        "job_description": "We are looking for a skilled Python developer with experience in web development and cloud technologies.",
        "creativity_mode": "balanced",
        "target_role": "Python Developer",
        "target_industry": "Technology"
    }

def test_valid_request_schema(valid_request):
    """Test that a valid request passes schema validation."""
    req = AnalysisRequest(**valid_request)
    # AR-1 (Phase 2): this asserted `req.user_profile["name"]`, but
    # AnalysisRequest has no `user_profile` field — the assertion was written
    # against an older schema and had been failing (unseen, because collection
    # was broken) ever since. Assert on fields that actually exist.
    assert req.resume_text.startswith("John Doe")
    assert "Python developer" in req.job_ad
    assert req.creativity_mode == CreativityMode.BALANCED

def test_invalid_request_schema():
    """Test that an invalid request fails schema validation"""
    with pytest.raises(ValidationError):
        AnalysisRequest(
            user_profile={},  # Missing required fields
            job_description="",
            creativity_mode="balanced"
        )

@pytest.mark.fast
def test_fast_contract():
    """Fast contract test that runs quickly"""
    assert True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
