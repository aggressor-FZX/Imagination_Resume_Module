#!/usr/bin/env python3
"""
Test script for the Imaginator API
"""
import requests
import json
import time

# API configuration
API_URL = "https://imaginator-resume-cowriter.onrender.com/analyze"
API_KEY = "MNyaVwZvCDKhFkn4OnDchJa6xsEsvgrr0s09KYYquhc="

# Test payload
test_payload = {
    "resume_text": """Senior Software Engineer with 8 years of experience in full-stack development, specializing in Python, React, and cloud technologies. Led teams of 5-10 developers in agile environments. Expert in designing scalable web applications and mentoring junior developers. Proven track record of delivering high-quality software solutions on time and within budget.""",
    "job_ad": """We are looking for a Senior Software Engineer with 5+ years of experience in Python and React. Must have experience with cloud platforms (AWS/GCP), database design, and leading development teams. Strong communication skills and ability to mentor junior developers required.""",
    "confidence_threshold": 0.7
}

def test_api():
    print("🧪 Testing Imaginator API...")
    print(f"📍 API URL: {API_URL}")
    print(f"📝 Resume text: {test_payload['resume_text'][:100]}...")
    print(f"🎯 Job ad: {test_payload['job_ad'][:100]}...")
    print()
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    
    try:
        print("🚀 Sending request...")
        start_time = time.time()
        
        response = requests.post(
            API_URL,
            json=test_payload,
            headers=headers,
            timeout=120  # 2 minute timeout
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"⏱️  Response time: {elapsed_time:.2f} seconds")
        print(f"🔢 Status code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Success! API is working correctly.")
            result = response.json()
            
            # Check required fields
            required_fields = [
                "experiences", "aggregate_skills", "processed_skills", 
                "domain_insights", "gap_analysis", "suggested_experiences",
                "seniority_analysis", "run_metrics", "processing_status", 
                "processing_time_seconds"
            ]
            
            print("\n📋 Response structure:")
            for field in required_fields:
                if field in result:
                    if isinstance(result[field], list):
                        print(f"  ✅ {field}: {len(result[field])} items")
                    elif isinstance(result[field], dict):
                        print(f"  ✅ {field}: {len(result[field])} keys")
                    else:
                        print(f"  ✅ {field}: {type(result[field]).__name__}")
                else:
                    print(f"  ❌ {field}: MISSING")
            
            print(f"\n⏱️  Total processing time: {result.get('processing_time_seconds', 'N/A')} seconds")
            print(f"💰 Estimated cost: ${result.get('run_metrics', {}).get('estimated_cost_usd', 'N/A')}")
            
            # Save response to file for inspection
            with open("test_response.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Response saved to test_response.json")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"📝 Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out (120 seconds)")
    except requests.exceptions.ConnectionError:
        print("🔌 Connection error - API may be down")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")

if __name__ == "__main__":
    test_api()
