#!/usr/bin/env python3
"""
Test script to verify seniority detection integration with Imaginator flow
"""

import asyncio
from imaginator_flow import run_analysis_async

def test_seniority_integration():
    """Test seniority detection integration"""
    
    # Sample resume text
    sample_resume = """
Jeff Calderon
Data Scientist and IT Specialist

EXPERIENCE
Senior Data Scientist, TechCorp Inc. (2019-2024)
- Led team of 5 data scientists
- Architected scalable ML pipelines processing 1M+ requests/day
- Mentored junior team members
- Improved model accuracy by 25%

Data Analyst, AnalyticsCo (2016-2019)
- Developed predictive models for customer behavior
- Created dashboards for business intelligence
- Collaborated with cross-functional teams

SKILLS
Python, Machine Learning, AWS, Team Leadership, Data Architecture
"""

    sample_job_ad = """
Data Scientist Position
Looking for experienced data scientist with 5+ years experience.
Must have experience with Python, machine learning, and cloud platforms.
Leadership experience preferred.
"""

    async def run_test():
        try:
            print("🧪 Testing seniority detection integration...")
            
            # Run analysis
            result = await run_analysis_async(
                resume_text=sample_resume,
                job_ad=sample_job_ad
            )
            
            # Check if seniority analysis is present
            if 'seniority_analysis' in result:
                print("✅ Seniority detection integration successful!")
                
                seniority = result['seniority_analysis']
                print(f"\n📊 Seniority Analysis Results:")
                print(f"   Level: {seniority['level']}")
                print(f"   Confidence: {seniority['confidence']:.2f}")
                print(f"   Years Experience: {seniority['total_years_experience']}")
                print(f"   Leadership Score: {seniority['leadership_score']:.2f}")
                print(f"   Skill Depth Score: {seniority['skill_depth_score']:.2f}")
                print(f"   Reasoning: {seniority['reasoning']}")
                print(f"   Recommendations: {seniority['recommendations'][:2]}...")
                
                # Verify required fields
                required_fields = ['level', 'confidence', 'total_years_experience', 'reasoning', 'recommendations']
                missing_fields = [field for field in required_fields if field not in seniority]
                
                if missing_fields:
                    print(f"❌ Missing required fields: {missing_fields}")
                else:
                    print("✅ All required fields present")
                    
            else:
                print("❌ Seniority analysis not found in result")
                print(f"Available keys: {list(result.keys())}")
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()

    # Run the test
    asyncio.run(run_test())

if __name__ == "__main__":
    test_seniority_integration()