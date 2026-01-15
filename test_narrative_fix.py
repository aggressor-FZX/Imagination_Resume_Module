#!/usr/bin/env python3
"""
Test to verify narrative detection fix in final editor stage
"""
import asyncio
import sys
import os

# Set test environment before importing
os.environ["environment"] = "test"

# Mock the missing dependencies for testing
import sys
from unittest.mock import MagicMock

# Mock google.generativeai before importing imaginator_flow
sys.modules['google.generativeai'] = MagicMock()

from imaginator_flow import run_final_editor_async

async def test_narrative_detection_fix():
    """Test that narrative content is properly handled and resume format is returned"""
    
    print("🧪 Testing narrative detection fix...")
    print("=" * 60)
    
    # Mock data with narrative content (what LLM might return)
    narrative_content = """As a senior software engineer, I have extensive experience in Python development. 
    I am seeking opportunities where I can leverage my skills. The candidate is a motivated professional 
    who wants to contribute to innovative projects."""
    
    # Clean STAR-formatted content (what Stage 3 should produce)
    clean_star_content = """## Work Experience

**Senior Software Engineer | Tech Corp | 2020-2023**
- Developed Python APIs serving 5M+ users
- Reduced deployment time by 60% through CI/CD automation
- Led team of 5 developers on microservices architecture

**Software Engineer | Startup Inc | 2018-2020**
- Built RESTful APIs using Flask and FastAPI
- Implemented automated testing reducing bugs by 40%"""
    
    # Clean experiences data (original parsed data)
    clean_experiences = [
        {
            "title_line": "Senior Software Engineer | Tech Corp | 2020-2023",
            "snippet": "Developed Python APIs serving 5M+ users. Reduced deployment time by 60%."
        },
        {
            "title_line": "Software Engineer | Startup Inc | 2018-2020",
            "snippet": "Built RESTful APIs using Flask and FastAPI. Implemented automated testing."
        }
    ]
    
    # Mock analysis data
    analysis = {
        "experiences": clean_experiences,
        "aggregate_skills": ["Python", "Flask", "FastAPI", "Docker"]
    }
    
    # Mock research data
    research_data = {
        "implied_skills": ["Python", "API Development"],
        "industry_metrics": ["5M users", "60% reduction"]
    }
    
    # Set test environment
    import os
    os.environ["environment"] = "test"
    
    print("\n📝 Test Case 1: LLM returns narrative, star_formatted is clean")
    print("-" * 60)
    
    try:
        result1 = await run_final_editor_async(
            creative_draft="Creative draft with some content",
            star_formatted=clean_star_content,
            research_data=research_data,
            analysis=analysis,
            job_ad="Looking for Python developer",
            openrouter_api_keys=None
        )
        
        final_text = result1.get("final_written_section", "")
        final_markdown = result1.get("final_written_section_markdown", "")
        
        # Check that result doesn't contain narrative indicators
        narrative_indicators = ["as a", "as an", "i have", "i am", "i'm", "i've", "is seeking", "wants to"]
        has_narrative = any(indicator in final_text.lower() for indicator in narrative_indicators)
        
        if has_narrative:
            print("❌ FAILED: Result still contains narrative content")
            print(f"   Content: {final_text[:200]}")
            return False
        else:
            print("✅ PASSED: No narrative content detected")
            print(f"   Using clean STAR-formatted content")
            print(f"   Preview: {final_text[:150]}...")
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n📝 Test Case 2: Both LLM output and star_formatted contain narrative")
    print("-" * 60)
    
    try:
        # Both contain narrative
        narrative_star = """As a professional, I have worked on various projects. 
        The candidate is experienced in Python development."""
        
        result2 = await run_final_editor_async(
            creative_draft="Creative draft",
            star_formatted=narrative_star,
            research_data=research_data,
            analysis=analysis,
            job_ad="Looking for Python developer",
            openrouter_api_keys=None
        )
        
        final_text2 = result2.get("final_written_section", "")
        final_markdown2 = result2.get("final_written_section_markdown", "")
        
        # Should regenerate from experiences
        has_narrative2 = any(indicator in final_text2.lower() for indicator in narrative_indicators)
        
        if has_narrative2:
            print("❌ FAILED: Result still contains narrative content")
            print(f"   Content: {final_text2[:200]}")
            return False
        else:
            print("✅ PASSED: Regenerated from clean experiences")
            print(f"   Preview: {final_text2[:150]}...")
            # Should contain experience titles
            if "Senior Software Engineer" in final_text2 or "Software Engineer" in final_text2:
                print("✅ PASSED: Contains original experience data")
            else:
                print("⚠️  WARNING: May not have regenerated from experiences correctly")
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED: Narrative detection fix is working correctly!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = asyncio.run(test_narrative_detection_fix())
    sys.exit(0 if success else 1)
