"""
Comprehensive test suite for the Imaginator API using direct function patching.
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
    """Create a test client for the FastAPI app with mocked authentication."""
    # Import here to avoid circular imports and ensure proper mocking
    from fastapi.testclient import TestClient
    from app import app, get_api_key
    
    # Mock the get_api_key function to always return a valid key
    with patch('app.get_api_key') as mock_get_api_key:
        mock_get_api_key.return_value = "test-api-key"
        
        # Create client with the mocked app
        return TestClient(app, headers={"X-API-Key": "test-api-key"})


class TestImaginatorAPIFinal:
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
            # Mock the run_analysis_async function to return expected output
            mock_run_analysis = mocker.patch('app.run_analysis_async')
            mock_run_analysis.return_value = test_case["expected_output"]
            
            # Prepare the request payload
            payload = {
                "resume_text": test_case["input"]["resume_text"],
                "job_ad": test_case["input"]["job_ad"]
            }
            
            # Add API keys if present
            for key in ["openai_api_key", "anthropic_api_key", "google_api_key", 
                       "deepseek_api_key", "openrouter_api_key_1", "openrouter_api_key_2"]:
                if key in test_case["input"]:
                    payload[key] = test_case["input"][key]
            
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
    
    def test_data_driven_negative_cases(self, client, api_test_cases, mocker):
        """Test API calls with negative/error test cases."""
        negative_cases = [
            case for case in api_test_cases 
            if case.get("expected_output", {}).get("status") == "error"
        ]
        
        for test_case in negative_cases:
            error_response = test_case["expected_output"]
            error_type = error_response["error"]["type"]
            
            # Prepare the request payload
            payload = {
                "resume_text": test_case["input"]["resume_text"],
                "job_ad": test_case["input"]["job_ad"]
            }
            
            if error_type == "ValidationError":
                # Validation errors should surface as 422 from FastAPI/Pydantic
                response = client.post("/analyze", json=payload)
                assert response.status_code == 422, f"Expected 422 for validation case: {test_case.get('name', 'Unknown')}"
            else:
                # For other errors, simulate server-side failure
                mock_run_analysis = mocker.patch('app.run_analysis_async')
                mock_run_analysis.side_effect = Exception(error_response["error"]["message"])
                response = client.post("/analyze", json=payload)
                assert response.status_code == 500, f"Expected 500 for error case: {test_case.get('name', 'Unknown')}"
    
    def test_invalid_api_key(self, client, mocker):
        """Test API calls with invalid API key."""
        # Mock get_api_key to raise HTTPException
        from fastapi import HTTPException
        
        with patch('app.get_api_key') as mock_get_api_key:
            mock_get_api_key.side_effect = HTTPException(status_code=403, detail="Invalid API key")
            
            # Create a new client without the API key header
            from fastapi.testclient import TestClient
            unauthenticated_client = TestClient(client.app)
            
            response = unauthenticated_client.post(
                "/analyze",
                json={
                    "resume_text": "Test resume with sufficient length",
                    "job_ad": "Test job ad with sufficient length"
                }
            )
            
            assert response.status_code == 403
            assert "detail" in response.json()
            assert "Invalid API key" in response.json()["detail"]
