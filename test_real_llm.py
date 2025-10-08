#!/usr/bin/env python3
"""
Test script to compare baseline vs enhanced prompts with REAL OpenAI LLM
No fallback mode - will report actual API errors
"""
import json
import sys
from imaginator_flow import (
    process_structured_skills,
    suggest_roles,
    generate_gap_analysis_baseline,  # Original simple prompt
    generate_gap_analysis,            # Enhanced multi-perspective prompt
)

def load_test_data():
    """Load sample data for testing"""
    with open('sample_skills.json') as f:
        skills_data = json.load(f)
    
    with open('sample_insights.json') as f:
        domain_insights = json.load(f)
    
    with open('sample_job_ad.txt') as f:
        job_ad = f.read()
    
    with open('sample_resume.txt') as f:
        resume_text = f.read()
    
    return skills_data, domain_insights, job_ad, resume_text

def run_baseline_test():
    """Run test with ORIGINAL simple prompt"""
    print("=" * 80)
    print("BASELINE TEST - Original Simple Prompt")
    print("=" * 80)
    
    skills_data, domain_insights, job_ad, resume_text = load_test_data()
    
    # Process skills
    processed_skills = process_structured_skills(skills_data, confidence_threshold=0.7)
    
    # Get role suggestions (needs set of skill names)
    skill_names = set(processed_skills['high_confidence_skills'])
    roles = suggest_roles(skill_names)
    
    print(f"\n📊 Processed Skills: {len(processed_skills['high_confidence_skills'])} high-confidence")
    print(f"🎯 Suggested Roles: {[r['role'] for r in roles]}")
    print(f"\n🚀 Calling OpenAI API with BASELINE prompt...\n")
    
    try:
        # Call with original simple prompt
        result = generate_gap_analysis_baseline(
            resume_text=resume_text,
            processed_skills=processed_skills,
            roles=roles,
            target_job_ad=job_ad,
            domain_insights=domain_insights
        )
        
        print("✅ API call succeeded!")
        print("\n" + "=" * 80)
        print("BASELINE OUTPUT (Original Prompt)")
        print("=" * 80)
        print(result)
        print("\n" + "=" * 80)
        
        # Save to file
        with open('real_baseline_output.txt', 'w') as f:
            f.write(result)
        
        print("\n✅ Saved to: real_baseline_output.txt")
        return True
        
    except Exception as e:
        print(f"\n❌ API ERROR: {type(e).__name__}: {str(e)}")
        print("\nMake sure your OPENAI_API_KEY is set in .env file")
        return False

def run_enhanced_test():
    """Run test with ENHANCED multi-perspective prompt"""
    print("\n\n" + "=" * 80)
    print("ENHANCED TEST - Multi-Perspective Prompt with Knowledge Bases")
    print("=" * 80)
    
    skills_data, domain_insights, job_ad, resume_text = load_test_data()
    
    # Process skills
    processed_skills = process_structured_skills(skills_data, confidence_threshold=0.7)
    
    # Get role suggestions (needs set of skill names)
    skill_names = set(processed_skills['high_confidence_skills'])
    roles = suggest_roles(skill_names)
    
    print(f"\n📊 Processed Skills: {len(processed_skills['high_confidence_skills'])} high-confidence")
    print(f"🎯 Suggested Roles: {[r['role'] for r in roles]}")
    print(f"\n🚀 Calling OpenAI API with ENHANCED prompt...\n")
    
    try:
        # Call with enhanced multi-perspective prompt
        result = generate_gap_analysis(
            resume_text=resume_text,
            processed_skills=processed_skills,
            roles=roles,
            target_job_ad=job_ad,
            domain_insights=domain_insights
        )
        
        print("✅ API call succeeded!")
        print("\n" + "=" * 80)
        print("ENHANCED OUTPUT (Multi-Perspective Prompt)")
        print("=" * 80)
        print(result)
        print("\n" + "=" * 80)
        
        # Save to file
        with open('real_enhanced_output.txt', 'w') as f:
            f.write(result)
        
        # Try to parse as JSON and save structured version
        try:
            # Extract JSON if it's wrapped in markdown
            json_content = result
            if '```json' in result:
                json_content = result.split('```json')[1].split('```')[0].strip()
            elif '```' in result:
                json_content = result.split('```')[1].split('```')[0].strip()
            
            parsed = json.loads(json_content)
            with open('real_enhanced_output.json', 'w') as f:
                json.dump(parsed, f, indent=2)
            print("✅ Saved structured output to: real_enhanced_output.json")
        except:
            print("⚠️  Output is not valid JSON (saved as text)")
        
        print("✅ Saved to: real_enhanced_output.txt")
        return True
        
    except Exception as e:
        print(f"\n❌ API ERROR: {type(e).__name__}: {str(e)}")
        print("\nMake sure your OPENAI_API_KEY is set in .env file")
        return False

def main():
    print("\n" + "🧪" * 40)
    print("REAL LLM TESTING - Baseline vs Enhanced Prompts")
    print("🧪" * 40)
    print("\nThis will make REAL OpenAI API calls (costs ~$0.01-0.02)")
    print("No demo fallback - errors will be reported\n")
    
    # Run both tests
    baseline_success = run_baseline_test()
    
    if not baseline_success:
        print("\n❌ Baseline test failed. Fix API issues before continuing.")
        sys.exit(1)
    
    enhanced_success = run_enhanced_test()
    
    if not enhanced_success:
        print("\n❌ Enhanced test failed.")
        sys.exit(1)
    
    # Summary
    print("\n\n" + "=" * 80)
    print("✅ BOTH TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Compare real_baseline_output.txt vs real_enhanced_output.txt")
    print("  2. Analyze differences in creativity, specificity, actionability")
    print("  3. Update ENHANCEMENT_COMPARISON.md with real results")
    print("\nFiles created:")
    print("  - real_baseline_output.txt")
    print("  - real_enhanced_output.txt")
    print("  - real_enhanced_output.json (if parseable)")

if __name__ == "__main__":
    main()
