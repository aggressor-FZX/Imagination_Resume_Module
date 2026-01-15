#!/usr/bin/env python3
"""
Full path test for narrative detection - tests the actual code path
Mocks LLM to return narrative and verifies the fix works
"""
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# Don't set test mode - we want to test the actual logic
# os.environ["environment"] = "test"  # Commented out to test real path

# Mock dependencies before importing
sys.modules['google.generativeai'] = MagicMock()

from imaginator_flow import run_final_editor_async

def load_file(filepath):
    """Load text file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"⚠️  Error loading {filepath}: {e}")
        return None

async def test_narrative_detection_full_path():
    """Test the full narrative detection code path"""
    print("🧪 Testing Full Narrative Detection Code Path")
    print("=" * 70)
    
    # Load real resume sample
    base_path = Path("/home/skystarved/Render_Dockers")
    resume_text = load_file(base_path / "HF_Resume_Samples" / "sample_fullstack_dev.txt")
    
    if not resume_text:
        print("❌ Could not load resume sample")
        return False
    
    # Create clean experiences
    experiences = [
        {
            "title_line": "Senior Software Engineer at CloudTech (2022-Present)",
            "snippet": "Architected microservices platform handling 10M+ requests/day using Node.js and AWS Lambda. Reduced API response time by 60% through query optimization."
        },
        {
            "title_line": "Software Engineer at WebSolutions (2020-2022)",
            "snippet": "Developed React-based dashboard used by 50K+ enterprise customers. Built RESTful APIs with Express.js and PostgreSQL, achieving 99.9% uptime."
        }
    ]
    
    # Clean STAR-formatted content
    clean_star_content = """## Work Experience

**Senior Software Engineer at CloudTech (2022-Present)**
- Architected microservices platform handling 10M+ requests/day using Node.js and AWS Lambda
- Reduced API response time by 60% through query optimization and caching strategies
- Implemented CI/CD pipeline with automated testing, reducing deployment time by 75%

**Software Engineer at WebSolutions (2020-2022)**
- Developed React-based dashboard used by 50K+ enterprise customers
- Built RESTful APIs with Express.js and PostgreSQL, achieving 99.9% uptime
- Containerized applications using Docker, deployed to AWS ECS"""
    
    # Mock analysis
    analysis = {
        "experiences": experiences,
        "aggregate_skills": ["JavaScript", "React", "Node.js", "AWS"]
    }
    
    research_data = {
        "implied_skills": ["JavaScript", "Cloud Architecture"],
        "industry_metrics": ["10M+ requests", "60% reduction"]
    }
    
    job_ad = "Looking for a Senior Software Engineer with experience in React, Node.js, and AWS."
    
    # Mock LLM to return narrative content (simulating the bug scenario)
    narrative_llm_response = {
        "final_written_section_markdown": """## Professional Summary

As a senior software engineer, I have extensive experience in building scalable web applications. 
I am seeking opportunities where I can leverage my skills in React and Node.js. The candidate 
is a motivated professional who wants to contribute to innovative projects.

## Work Experience

As a professional, I have worked on various projects involving microservices architecture.""",
        "final_written_section": """PROFESSIONAL SUMMARY

As a senior software engineer, I have extensive experience. I am seeking opportunities. 
The candidate is a motivated professional who wants to contribute.""",
        "editorial_notes": "ATS-optimized"
    }
    
    import json
    narrative_json_response = json.dumps(narrative_llm_response)
    
    print("\n📝 Test Scenario: LLM returns narrative, star_formatted is clean")
    print("-" * 70)
    print("   Simulating: LLM returns narrative content")
    print("   Expected: Should fallback to clean star_formatted")
    
    # Mock the LLM call to return narrative
    with patch('imaginator_flow.call_llm_async', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = narrative_json_response
        
        try:
            result = await run_final_editor_async(
                creative_draft="Some creative draft",
                star_formatted=clean_star_content,
                research_data=research_data,
                analysis=analysis,
                job_ad=job_ad,
                openrouter_api_keys=None
            )
            
            final_section = result.get("final_written_section", "")
            final_markdown = result.get("final_written_section_markdown", "")
            
            # Check for narrative
            narrative_indicators = [
                "as a", "as an", "i have", "i am", "i've", "i'm",
                "is seeking", "wants to", "the candidate"
            ]
            
            has_narrative = any(ind in final_section.lower() for ind in narrative_indicators)
            
            print(f"\n   📊 Results:")
            print(f"   Final section length: {len(final_section)} chars")
            print(f"   Has narrative: {has_narrative}")
            print(f"   Preview: {final_section[:200]}...")
            
            if has_narrative:
                print(f"\n   ❌ FAILED: Result still contains narrative!")
                print(f"   The fix may not be working correctly.")
                return False
            else:
                print(f"\n   ✅ PASSED: No narrative detected!")
                print(f"   The fix correctly fell back to clean star_formatted.")
                
                # Verify it's using star content
                if "10M+ requests" in final_section or "CloudTech" in final_section:
                    print(f"   ✅ Confirmed: Using clean STAR-formatted content")
                else:
                    print(f"   ⚠️  May not be using star_formatted (could be regenerated from experiences)")
                
                return True
                
        except Exception as e:
            print(f"\n   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

async def test_both_narrative_scenario():
    """Test when both LLM and star_formatted have narrative"""
    print("\n📝 Test Scenario: Both LLM and star_formatted have narrative")
    print("-" * 70)
    print("   Simulating: Both sources contain narrative")
    print("   Expected: Should regenerate from original experiences")
    
    experiences = [
        {
            "title_line": "Senior Software Engineer at CloudTech (2022-Present)",
            "snippet": "Architected microservices platform handling 10M+ requests/day. Reduced API response time by 60%."
        }
    ]
    
    narrative_star = """As a professional, I have worked on various projects. 
    The candidate is experienced in Python development."""
    
    analysis = {"experiences": experiences, "aggregate_skills": ["Python"]}
    research_data = {"implied_skills": ["Python"]}
    
    narrative_llm_response = {
        "final_written_section": "As a senior engineer, I have extensive experience.",
        "final_written_section_markdown": "As a professional, I am seeking opportunities."
    }
    
    narrative_json = '{"final_written_section": "As a senior engineer, I have extensive experience.", "final_written_section_markdown": "As a professional, I am seeking opportunities.", "editorial_notes": "ATS-optimized"}'
    
    with patch('imaginator_flow.call_llm_async', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = narrative_json
        
        try:
            result = await run_final_editor_async(
                creative_draft="Creative draft",
                star_formatted=narrative_star,  # Also has narrative
                research_data=research_data,
                analysis=analysis,
                job_ad="Looking for Python developer",
                openrouter_api_keys=None
            )
            
            final_section = result.get("final_written_section", "")
            
            narrative_indicators = ["as a", "as an", "i have", "i am", "the candidate"]
            has_narrative = any(ind in final_section.lower() for ind in narrative_indicators)
            
            print(f"\n   📊 Results:")
            print(f"   Final section length: {len(final_section)} chars")
            print(f"   Has narrative: {has_narrative}")
            print(f"   Preview: {final_section[:200]}...")
            
            if has_narrative:
                print(f"\n   ❌ FAILED: Result still contains narrative!")
                return False
            else:
                print(f"\n   ✅ PASSED: No narrative detected!")
                
                # Should be regenerated from experiences
                if "CloudTech" in final_section or "10M+ requests" in final_section:
                    print(f"   ✅ Confirmed: Regenerated from clean experiences")
                else:
                    print(f"   ⚠️  May not have regenerated correctly")
                
                return True
                
        except Exception as e:
            print(f"\n   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

async def run_all_tests():
    """Run all full path tests"""
    results = []
    
    result1 = await test_narrative_detection_full_path()
    results.append(("Narrative LLM, Clean STAR", result1))
    
    result2 = await test_both_narrative_scenario()
    results.append(("Both Narrative, Regenerate", result2))
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 FINAL TEST SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {test_name}")
    
    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'='*70}")
    
    if passed == total:
        print("\n🎉 ALL FULL PATH TESTS PASSED!")
        print("✅ Narrative detection fix is working correctly in production code path!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
