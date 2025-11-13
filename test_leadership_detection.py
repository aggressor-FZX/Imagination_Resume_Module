#!/usr/bin/env python3
"""
Test script to understand leadership detection in seniority analysis
"""

from seniority_detector import SeniorityDetector

def test_leadership_detection():
    """Test leadership detection functionality"""
    
    detector = SeniorityDetector()
    
    # Test different experience descriptions
    test_cases = [
        {
            "title": "Senior Data Scientist",
            "duration": "2019-2024", 
            "description": "Led team of 5 data scientists. Architected scalable ML pipelines processing 1M+ requests/day. Mentored junior team members."
        },
        {
            "title": "Data Analyst",
            "duration": "2016-2019",
            "description": "Developed predictive models for customer behavior. Created dashboards for business intelligence."
        },
        {
            "title": "Team Lead",
            "duration": "2020-2023",
            "description": "Managed team of 8 developers. Coordinated project timelines. Supervised code reviews."
        }
    ]
    
    for i, exp in enumerate(test_cases):
        print(f"\n🧪 Test Case {i+1}: {exp['title']}")
        print(f"   Description: {exp['description'][:80]}...")
        
        # Test leadership detection
        leadership_score = detector._detect_leadership_indicators([exp])
        print(f"   Leadership Score: {leadership_score:.2f}")
        
        # Test title seniority
        title_seniority = detector._detect_title_seniority(exp['title'])
        print(f"   Title Seniority Score: {title_seniority:.2f}")
        
        # Test description complexity
        desc_complexity = detector._analyze_description_complexity(exp['description'])
        print(f"   Description Complexity: {desc_complexity:.2f}")

if __name__ == "__main__":
    test_leadership_detection()