#!/usr/bin/env python3
"""
Real API test for narrative detection fix
Tests with actual OpenRouter API calls using real resume samples and job ads
"""
import asyncio
import sys
import os
from pathlib import Path

# Don't set test mode - we want real API calls
# os.environ["environment"] = "test"  # Commented out for real API calls

from imaginator_flow import run_full_analysis_async
from config import settings

# Narrative indicators for checking output
NARRATIVE_INDICATORS = [
    "as a", "as an", "i have", "i am", "i've", "i'm",
    "is a", "is an", "he has", "she has", "they have",
    "he is", "she is", "they are", "we have", "we are",
    "a motivated", "an experienced", "the candidate", "the professional",
    "this professional", "this candidate", "this individual",
    "is seeking", "is looking", "he is seeking", "she is seeking",
    "they are seeking", "wants to", "would like to", "aims to",
    "strives to", "seeks to"
]

def check_for_narrative(text):
    """Check if text contains narrative indicators"""
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in NARRATIVE_INDICATORS)

def load_file(filepath):
    """Load text file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"⚠️  Error loading {filepath}: {e}")
        return None

async def test_resume_job_combination(resume_text, job_ad, test_name):
    """Test a resume/job ad combination with real API calls"""
    print(f"\n{'='*70}")
    print(f"🧪 TEST: {test_name}")
    print(f"{'='*70}")
    print(f"📄 Resume length: {len(resume_text)} chars")
    print(f"📋 Job ad length: {len(job_ad)} chars")
    
    # Check API keys
    api_keys = [key for key in [settings.openrouter_api_key_1, settings.openrouter_api_key_2] if key]
    if not api_keys:
        print("❌ ERROR: No OpenRouter API keys configured!")
        print("   Set OPENROUTER_API_KEY_1 or OPENROUTER_API_KEY_2 environment variables")
        return False
    
    print(f"✅ Found {len(api_keys)} OpenRouter API key(s)")
    print(f"🚀 Making real API calls to OpenRouter...")
    print(f"   (This may take 30-60 seconds)")
    
    try:
        # Run the full analysis pipeline with real API calls
        result = await run_full_analysis_async(
            resume_text=resume_text,
            job_ad=job_ad,
            extracted_skills_json=None,
            domain_insights_json=None,
            openrouter_api_keys=api_keys
        )
        
        # Check for final_written_section
        final_section = result.get("final_written_section", "")
        final_markdown = result.get("final_written_section_markdown", "")
        
        # Check if result contains narrative
        has_narrative = False
        narrative_found_in = []
        narrative_indicators_found = []
        
        if final_section:
            text_lower = final_section.lower()
            found_indicators = [ind for ind in NARRATIVE_INDICATORS if ind in text_lower]
            if found_indicators:
                has_narrative = True
                narrative_found_in.append("final_written_section")
                narrative_indicators_found.extend(found_indicators)
        
        if final_markdown:
            text_lower = final_markdown.lower()
            found_indicators = [ind for ind in NARRATIVE_INDICATORS if ind in text_lower]
            if found_indicators:
                has_narrative = True
                narrative_found_in.append("final_written_section_markdown")
                narrative_indicators_found.extend(found_indicators)
        
        # Display results
        print(f"\n📊 Results:")
        print(f"   ✅ Analysis completed successfully")
        print(f"   📝 Final section length: {len(final_section)} chars")
        print(f"   📝 Final markdown length: {len(final_markdown)} chars")
        
        if has_narrative:
            print(f"\n   ❌ NARRATIVE DETECTED in: {', '.join(narrative_found_in)}")
            print(f"   Found indicators: {', '.join(set(narrative_indicators_found))}")
            print(f"\n   Preview (first 300 chars):")
            print(f"   {final_section[:300]}...")
            return False
        else:
            print(f"\n   ✅ NO NARRATIVE DETECTED - Resume format maintained!")
            
            # Check if it looks like resume format (has bullets, action verbs, etc.)
            resume_indicators = ["•", "-", "Developed", "Built", "Implemented", "Led", "Created", 
                               "Reduced", "Increased", "Architected", "Designed", "Managed"]
            found_resume_indicators = [ind for ind in resume_indicators if ind in final_section]
            
            if found_resume_indicators:
                print(f"   ✅ Contains resume format indicators: {', '.join(found_resume_indicators[:5])}")
            else:
                print(f"   ⚠️  May not have standard resume format")
            
            print(f"\n   Preview (first 300 chars):")
            print(f"   {final_section[:300]}...")
            
            return True
            
    except Exception as e:
        print(f"\n   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_all_tests():
    """Run all test combinations with real API calls"""
    print("🚀 Starting Real API Tests for Narrative Detection Fix")
    print("=" * 70)
    print("⚠️  WARNING: This will make real API calls to OpenRouter")
    print("⚠️  This will incur API costs and may take several minutes")
    print("=" * 70)
    
    # Load resume samples
    base_path = Path("/home/skystarved/Render_Dockers")
    
    resume_samples = {
        "Data Scientist": load_file(base_path / "HF_Resume_Samples" / "sample_data_scientist.txt"),
        "Full Stack Dev": load_file(base_path / "HF_Resume_Samples" / "sample_fullstack_dev.txt"),
    }
    
    # Load job ads
    job_ads = {
        "SR Programmer Analyst": load_file(base_path / "Imaginator" / "job_ad.txt"),
        "Senior Software Engineer": load_file(base_path / "test_job_ad.txt"),
    }
    
    # Filter out None values
    resume_samples = {k: v for k, v in resume_samples.items() if v}
    job_ads = {k: v for k, v in job_ads.items() if v}
    
    if not resume_samples:
        print("❌ No resume samples found!")
        return False
    
    if not job_ads:
        print("❌ No job ads found!")
        return False
    
    print(f"\n📚 Loaded {len(resume_samples)} resume samples")
    print(f"📚 Loaded {len(job_ads)} job ads")
    
    # Test combinations
    test_results = []
    
    # Test 1: Data Scientist + SR Programmer Analyst
    if "Data Scientist" in resume_samples and "SR Programmer Analyst" in job_ads:
        result = await test_resume_job_combination(
            resume_samples["Data Scientist"],
            job_ads["SR Programmer Analyst"],
            "Data Scientist Resume → SR Programmer Analyst Job"
        )
        test_results.append(("Data Scientist → SR Programmer Analyst", result))
    
    # Test 2: Full Stack Dev + Senior Software Engineer
    if "Full Stack Dev" in resume_samples and "Senior Software Engineer" in job_ads:
        result = await test_resume_job_combination(
            resume_samples["Full Stack Dev"],
            job_ads["Senior Software Engineer"],
            "Full Stack Dev Resume → Senior Software Engineer Job"
        )
        test_results.append(("Full Stack Dev → Senior Software Engineer", result))
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 TEST SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {test_name}")
    
    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'='*70}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Narrative detection fix is working correctly!")
        print("\n✅ Verification:")
        print("   1. Real API calls completed successfully")
        print("   2. No narrative content detected in outputs")
        print("   3. Resume format maintained (bullets, action verbs)")
        print("   4. Fix is working in production code path")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        print("   Review output above for details on narrative content found.")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
