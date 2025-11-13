#!/usr/bin/env python3
"""
Test script to verify experience parsing for seniority detection
"""

from imaginator_flow import parse_experiences, extract_duration_from_text, extrapolate_skills_from_text

def test_experience_parsing():
    """Test experience parsing functionality"""
    
    # Sample resume text with various date formats
    sample_resume = """
Jeff Calderon
Data Scientist and IT Specialist

EXPERIENCE
Senior Data Scientist, TechCorp Inc. (2019-2024)
- Led team of 5 data scientists
- Architected scalable ML pipelines processing 1M+ requests/day
- Mentored junior team members
- Improved model accuracy by 25%

Data Analyst, AnalyticsCo (June 2016 - March 2019)
- Developed predictive models for customer behavior
- Created dashboards for business intelligence
- Collaborated with cross-functional teams

Junior Developer, StartupXYZ (2014-2016)
- Built web applications using Python and Django
- Participated in agile development processes

SKILLS
Python, Machine Learning, AWS, Team Leadership, Data Architecture
"""

    print("🧪 Testing experience parsing...")
    
    # Parse experiences
    experiences = parse_experiences(sample_resume)
    
    print(f"✅ Found {len(experiences)} experiences")
    
    for i, exp in enumerate(experiences):
        # Extract skills from the experience text
        skills = extrapolate_skills_from_text(exp.get('description', ''))
        
        print(f"\n📋 Experience {i+1}:")
        print(f"   Title: {exp.get('title', exp.get('title_line', 'N/A'))}")
        print(f"   Duration: {exp.get('duration', 'N/A')}")
        print(f"   Description: {exp.get('description', 'N/A')[:100]}...")
        print(f"   Skills: {list(skills)}")
    
    # Test duration extraction
    print("\n🧪 Testing duration extraction...")
    test_texts = [
        "Senior Data Scientist, TechCorp Inc. (2019-2024)",
        "Data Analyst, AnalyticsCo (June 2016 - March 2019)", 
        "Junior Developer, StartupXYZ (2014-2016)",
        "No dates in this text"
    ]
    
    for text in test_texts:
        duration = extract_duration_from_text(text)
        print(f"   '{text}' -> Duration: '{duration}'")

if __name__ == "__main__":
    test_experience_parsing()