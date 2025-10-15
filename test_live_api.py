#!/usr/bin/env python3
"""
Test script for the live Imaginator Resume Co-Writer API
"""
import os
import sys
import json
import requests
from typing import Dict, Any

# Configuration
API_URL = "https://imaginator-resume-cowriter.onrender.com"
API_KEY = os.environ.get("API_KEY", "")

# Sample data for testing
SAMPLE_RESUME = """
John Doe
Software Engineer

EXPERIENCE:
Senior Developer at Tech Corp (2020-2025)
- Led team of 5 developers building microservices
- Implemented CI/CD pipelines using Docker and Kubernetes
- Developed REST APIs using Python, Django, and Flask
- Worked with PostgreSQL and Redis for data storage
- Mentored junior developers

SKILLS:
- Languages: Python, JavaScript, SQL
- Frameworks: Django, Flask, FastAPI
- Tools: Docker, Git, Jenkins
- Databases: PostgreSQL, MongoDB, Redis
- Cloud: Basic AWS knowledge
"""

SAMPLE_JOB = """
Senior Full Stack Engineer

We're looking for an experienced engineer to join our team.

REQUIREMENTS:
- 5+ years of software development experience
- Strong Python and JavaScript skills
- Experience with React.js and modern frontend frameworks
- Microservices architecture experience
- AWS or GCP cloud platform expertise
- Strong communication and leadership skills
- Experience with GraphQL is a plus

RESPONSIBILITIES:
- Design and build scalable web applications
- Lead technical discussions and architecture decisions
- Mentor junior team members
- Work closely with product team
"""

def test_health():
    """Test the health endpoint"""
    print("=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200 and response.json().get("status") == "healthy":
            print("✅ Health check PASSED")
            return True
        else:
            print("❌ Health check FAILED")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_auth_required():
    """Test that authentication is required"""
    print("\n" + "=" * 60)
    print("TEST 2: Authentication Required")
    print("=" * 60)
    
    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json={
                "resume_text": "Test",
                "job_description": "Test"
            },
            timeout=10
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 403:
            print("✅ Authentication required (403) - PASSED")
            return True
        else:
            print(f"❌ Expected 403, got {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_full_analysis():
    """Test the full analysis endpoint with authentication"""
    print("\n" + "=" * 60)
    print("TEST 3: Full Analysis with Authentication")
    print("=" * 60)
    
    if not API_KEY or API_KEY == "your-api-key-here":
        print("⚠️  SKIPPED: API_KEY not set")
        print("To run this test:")
        print("  export API_KEY='your-actual-api-key'")
        print("  python test_live_api.py")
        return None
    
    try:
        print("Sending request to /analyze endpoint...")
        print(f"Resume length: {len(SAMPLE_RESUME)} chars")
        print(f"Job description length: {len(SAMPLE_JOB)} chars")
        
        response = requests.post(
            f"{API_URL}/analyze",
            headers={
                "Content-Type": "application/json",
                "X-API-Key": API_KEY
            },
            json={
                "resume_text": SAMPLE_RESUME,
                "job_description": SAMPLE_JOB
            },
            timeout=120  # Analysis can take time
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Analysis SUCCESSFUL!")
            print("\nResponse Summary:")
            print(f"  Matched Skills: {len(result.get('matched_skills', []))} skills")
            print(f"  Missing Skills: {len(result.get('missing_skills', []))} skills")
            print(f"  Skill Gap: {result.get('skill_gap_percentage', 0):.1f}%")
            print(f"  Confidence: {result.get('confidence', 0):.2f}")
            
            if result.get('matched_skills'):
                print("\n  Sample Matched Skills:")
                for skill in result['matched_skills'][:3]:
                    print(f"    - {skill}")
            
            if result.get('missing_skills'):
                print("\n  Sample Missing Skills:")
                for skill in result['missing_skills'][:3]:
                    print(f"    - {skill}")
            
            if result.get('recommendations'):
                print("\n  Recommendations:")
                for rec in result['recommendations'][:2]:
                    print(f"    - {rec[:80]}...")
            
            return True
        else:
            print(f"❌ Analysis FAILED with status {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (analysis may take longer than expected)")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("\n" + "🚀" * 30)
    print("IMAGINATOR API TEST SUITE")
    print("Testing: " + API_URL)
    print("🚀" * 30 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health()))
    results.append(("Auth Required", test_auth_required()))
    
    analysis_result = test_full_analysis()
    if analysis_result is not None:
        results.append(("Full Analysis", analysis_result))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = len([r for _, r in results if r is None])
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED" if result is False else "⚠️  SKIPPED"
        print(f"{test_name:30} {status}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
