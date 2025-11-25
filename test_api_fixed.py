"""
Fixed test suite for the Imaginator API using proper mocking approach.
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import patch
from typing import Dict, Any, List

# Add the parent directory to the path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Get the absolute path to the test_data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"

def load_test_cases() -> List[Dict[str, Any]]:
    """Load all test cases from the test_data directory."""
    test_cases = []
    
    if not TEST_DATA_DIR.exists():
        pytest.fail(f"Test data directory not found: {TEST_DATA_DIR}")
    
    for json_file in TEST_DATA_DIR.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                test_case = json.load(f)
                test_case["filename"] = json_file.name
                test_cases.append(test_case)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON in {json_file}: {e}")
        except Exception as e:
            pytest.fail(f"Error loading {json_file}: {e}")
    
    if not test_cases:
        pytest.fail(f"No test cases found in {TEST_DATA_DIR}")
    
    return test_cases


@pytest.fixture
def api_test_cases() -> List[Dict[str, Any]]:
    """Fixture to provide test cases to test functions."""
    return load_test_cases()


@pytest.fixture
def client():
    """Create a test client for the FastAPI app with mocked settings."""
    with patch('config.settings.IMAGINATOR_AUTH_TOKEN', 'test-api-key'):
        with patch('config.settings.openrouter_api_key_1', 'test-openrouter-key-1'):
            with patch('config.settings.openrouter_api_key_2', 'test-openrouter-key-2'):
                from fastapi.testclient import TestClient
                from app import app
                return TestClient(app, headers={"X-API-Key": "test-api-key"})


class TestImaginatorAPIFixed:
    """Test class for Imaginator API endpoints with proper mocking."""
    
    def test_api_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "has_openrouter_key" in data
        assert isinstance(data["has_openrouter_key"], bool)
    
    def test_data_driven_positive_cases(self, client, api_test_cases, mocker):
        """Test successful API calls with various positive test cases."""
        positive_cases = [
            case for case in api_test_cases 
            if case.get("expected_output", {}).get("status") == "success"
        ]
        
        assert positive_cases, "No positive test cases found"
        
        for test_case in positive_cases:
            # Mock the run_analysis_async function
            mock_run_analysis = mocker.patch('app.run_analysis_async')
            mock_run_analysis.return_value = test_case["expected_output"]
            
            # Prepare the request payload
            payload = {
                "resume_text": test_case["input"]["resume_text"],
                "job_ad": test_case["input"]["job_ad"]
            }
            
            # No BYOK support; use server keys
            
            # Make the API call
            response = client.post("/analyze", json=payload)
            
            # Assertions
            assert response.status_code == 200, f"Failed for case: {test_case.get('name', 'Unknown')}"
            
            response_data = response.json()
            expected_data = test_case["expected_output"]
            
            # Validate response structure
            assert "status" in response_data
            assert response_data["status"] == "success"
            assert "analysis" in response_data
            assert "confidence_score" in response_data
            assert "processing_time" in response_data
            assert "model_used" in response_data
            
            # Validate analysis content matches expected output
            assert response_data["analysis"] == expected_data["analysis"]
            assert response_data["confidence_score"] == expected_data["confidence_score"]
            assert response_data["model_used"] == expected_data["model_used"]