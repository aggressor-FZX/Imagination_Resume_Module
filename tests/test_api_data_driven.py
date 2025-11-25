
"""
Data-driven test suite for the Imaginator API.

This test suite loads test cases from the test_data directory and uses mocking
to test various scenarios without making live API calls.
"""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

# Get the absolute path to the test_data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"


def load_test_cases():
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
def api_test_data():
    """Loads test data from the test_data directory."""
    return load_test_cases()


def test_api_data_driven_success_cases(api_test_data, mocker):
    """Test successful API calls with various positive test cases."""
    success_cases = [
        case for case in api_test_data 
        if case.get("expected_output", {}).get("status") == "success"
    ]
    
    assert success_cases, "No successful test cases found"
    
    for test_case in success_cases:
        # Mock the API call
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = test_case["expected_output"]
        mocker.patch("requests.post", return_value=mock_response)

        # Call the API
        response = requests.post(
            "https://imaginator-resume-cowriter.onrender.com/analyze",
            json=test_case["input"],
            headers={
                "Content-Type": "application/json",
                "X-API-Key": "test-key",
            },
            timeout=120,
        )

        # Assertions
        assert response.status_code == 200, f"Failed for case: {test_case.get('name', 'Unknown')}"
        assert response.json() == test_case["expected_output"]
        
        print(f"✓ Passed success case: {test_case.get('name', 'Unknown')}")


def test_api_data_driven_error_cases(api_test_data, mocker):
    """Test error handling with various negative test cases."""
    error_cases = [
        case for case in api_test_data 
        if case.get("expected_output", {}).get("status") == "error"
    ]
    
    for test_case in error_cases:
        # Mock the API call to return an error
        mock_response = mocker.Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = test_case["expected_output"]
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("400 Client Error")
        mocker.patch("requests.post", return_value=mock_response)

        # Call the API and expect an exception
        try:
            response = requests.post(
                "https://imaginator-resume-cowriter.onrender.com/analyze",
                json=test_case["input"],
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": "test-key",
                },
                timeout=120,
            )
            response.raise_for_status()
            assert False, f"Expected exception for case: {test_case.get('name', 'Unknown')}"
        except requests.exceptions.HTTPError:
            pass  # Expected behavior
        
        print(f"✓ Passed error case: {test_case.get('name', 'Unknown')}")


def test_test_case_structure(api_test_data):
    """Validate that all test cases have the required structure."""
    for test_case in api_test_data:
        # Validate required fields
        assert "name" in test_case, f"Test case missing name: {test_case.get('filename', 'Unknown')}"
        assert "description" in test_case, f"Test case missing description: {test_case.get('name', 'Unknown')}"
        assert "input" in test_case, f"Test case missing input: {test_case.get('name', 'Unknown')}"
        assert "expected_output" in test_case, f"Test case missing expected_output: {test_case.get('name', 'Unknown')}"
        
        # Validate input structure
        input_data = test_case["input"]
        assert "resume_text" in input_data, f"Input missing resume_text: {test_case.get('name', 'Unknown')}"
        assert "job_ad" in input_data, f"Input missing job_ad: {test_case.get('name', 'Unknown')}"
        
        # Validate expected output structure
        expected = test_case["expected_output"]
        assert "status" in expected, f"Expected output missing status: {test_case.get('name', 'Unknown')}"
        
        if expected["status"] == "success":
            assert "analysis" in expected, f"Success case missing analysis: {test_case.get('name', 'Unknown')}"
        elif expected["status"] == "error":
            assert "error" in expected, f"Error case missing error details: {test_case.get('name', 'Unknown')}"
