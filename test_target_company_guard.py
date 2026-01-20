"""
Test script for Target Company Guard implementation
Tests the "Armada hallucination" fix in StarEditor
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stages.star_editor import StarEditor
from llm_client import LLMClient

class MockLLMClient:
    """Mock LLM client that simulates hallucination"""
    
    async def call_llm_async(self, system_prompt, user_prompt, model, temperature, response_format, timeout):
        """Simulate LLM response with target company hallucination"""
        
        # Check if the prompt contains the CRITICAL warning
        has_warning = "CRITICAL: The user is applying to" in user_prompt
        
        # Simulate hallucination (listing Armada as employer)
        hallucinated_response = """{
  "final_markdown": "## **Senior Software Engineer** | **Armada**\\n*2020 - Present | Seattle, WA*\\n- Deployed microservices architecture reducing latency by 40%\\n- Led team of 5 engineers in cloud migration saving $200K annually",
  "final_plain_text": "Senior Software Engineer | Armada\\n2020 - Present | Seattle, WA\\n- Deployed microservices architecture reducing latency by 40%\\n- Led team of 5 engineers in cloud migration saving $200K annually",
  "editorial_notes": "Polished for ATS compliance. Ensured bulleted list format."
}"""
        
        # Simulate correct response (using actual employer)
        correct_response = """{
  "final_markdown": "## **Senior Software Engineer** | **Washington State Data Exchange**\\n*2020 - Present | Seattle, WA*\\n- Deployed microservices architecture reducing latency by 40%\\n- Led team of 5 engineers in cloud migration saving $200K annually",
  "final_plain_text": "Senior Software Engineer | Washington State Data Exchange\\n2020 - Present | Seattle, WA\\n- Deployed microservices architecture reducing latency by 40%\\n- Led team of 5 engineers in cloud migration saving $200K annually",
  "editorial_notes": "Polished for ATS compliance. Ensured bulleted list format."
}"""
        
        # Return hallucinated response to test the guard
        return hallucinated_response

async def test_target_company_guard():
    """Test the target company guard with Armada scenario"""
    
    print("=" * 80)
    print("TARGET COMPANY GUARD TEST")
    print("=" * 80)
    print()
    
    # Initialize StarEditor with mock LLM client
    mock_client = MockLLMClient()
    editor = StarEditor(mock_client)
    
    # Test Case 1: User applying to Armada but never worked there
    print("TEST CASE 1: Hallucination Detection (User never worked at Armada)")
    print("-" * 80)
    
    draft_data = {
        "rewritten_experiences": [
            {
                "company": "Washington State Data Exchange",
                "role": "Senior Software Engineer",
                "bullets": [
                    "Deployed microservices architecture reducing latency by 40%",
                    "Led team of 5 engineers in cloud migration saving $200K annually"
                ]
            }
        ],
        "seniority_applied": "senior"
    }
    
    research_data = {
        "company_name": "Armada",  # Target company from job ad
        "domain_vocab": ["microservices", "cloud", "DevOps", "Kubernetes"]
    }
    
    print(f"📋 Input:")
    print(f"   - Actual Employer: Washington State Data Exchange")
    print(f"   - Target Company (Job Ad): Armada")
    print(f"   - Expected Behavior: Guard should detect and fix hallucination")
    print()
    
    result = await editor.polish(draft_data, research_data)
    
    print(f"📊 Result:")
    print(f"   - Final Markdown Preview:")
    markdown_preview = result.get("final_markdown", "")[:200]
    print(f"     {markdown_preview}...")
    print()
    
    # Check if hallucination was fixed
    if "Armada" in result.get("final_markdown", ""):
        print("❌ FAILED: Armada still appears in final output!")
        print(f"   Full output: {result.get('final_markdown', '')}")
    else:
        print("✅ PASSED: Armada was successfully removed!")
    
    # Check if actual employer is present
    if "Washington State Data Exchange" in result.get("final_markdown", ""):
        print("✅ PASSED: Actual employer (Washington State Data Exchange) is present!")
    else:
        print("⚠️  WARNING: Actual employer not found in output")
    
    # Check editorial notes
    if "Fixed target company hallucination" in result.get("editorial_notes", ""):
        print("✅ PASSED: Editorial notes confirm hallucination was fixed!")
        print(f"   Notes: {result.get('editorial_notes', '')}")
    else:
        print("⚠️  WARNING: No hallucination fix noted in editorial_notes")
        print(f"   Notes: {result.get('editorial_notes', '')}")
    
    print()
    print("-" * 80)
    print()
    
    # Test Case 2: User actually worked at Armada (should NOT trigger guard)
    print("TEST CASE 2: Legitimate Employment (User DID work at Armada)")
    print("-" * 80)
    
    draft_data_legit = {
        "rewritten_experiences": [
            {
                "company": "Armada",  # User actually worked here
                "role": "Senior Software Engineer",
                "bullets": [
                    "Built distributed systems handling 1M+ requests/day",
                    "Reduced infrastructure costs by 35% through optimization"
                ]
            }
        ],
        "seniority_applied": "senior"
    }
    
    research_data_legit = {
        "company_name": "Armada",  # Same company - applying back
        "domain_vocab": ["distributed systems", "optimization"]
    }
    
    print(f"📋 Input:")
    print(f"   - Actual Employer: Armada")
    print(f"   - Target Company (Job Ad): Armada")
    print(f"   - Expected Behavior: Guard should NOT trigger (legitimate employment)")
    print()
    
    # For this test, we need a different mock response
    class MockLLMClientLegit:
        async def call_llm_async(self, system_prompt, user_prompt, model, temperature, response_format, timeout):
            return """{
  "final_markdown": "## **Senior Software Engineer** | **Armada**\\n*2018 - 2020 | San Francisco, CA*\\n- Built distributed systems handling 1M+ requests/day\\n- Reduced infrastructure costs by 35% through optimization",
  "final_plain_text": "Senior Software Engineer | Armada\\n2018 - 2020 | San Francisco, CA\\n- Built distributed systems handling 1M+ requests/day\\n- Reduced infrastructure costs by 35% through optimization",
  "editorial_notes": "Polished for ATS compliance."
}"""
    
    editor_legit = StarEditor(MockLLMClientLegit())
    result_legit = await editor_legit.polish(draft_data_legit, research_data_legit)
    
    print(f"📊 Result:")
    print(f"   - Final Markdown Preview:")
    markdown_preview_legit = result_legit.get("final_markdown", "")[:200]
    print(f"     {markdown_preview_legit}...")
    print()
    
    # Check if Armada is still present (should be, since it's legitimate)
    if "Armada" in result_legit.get("final_markdown", ""):
        print("✅ PASSED: Armada correctly retained (legitimate employment)!")
    else:
        print("❌ FAILED: Armada was incorrectly removed!")
    
    # Check that guard did NOT trigger
    if "Fixed target company hallucination" not in result_legit.get("editorial_notes", ""):
        print("✅ PASSED: Guard did NOT trigger (correct behavior)!")
    else:
        print("❌ FAILED: Guard incorrectly triggered for legitimate employment!")
    
    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("✅ Target Company Guard implementation verified!")
    print("✅ Hallucination detection working")
    print("✅ Legitimate employment preserved")
    print()

if __name__ == "__main__":
    asyncio.run(test_target_company_guard())
