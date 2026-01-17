import asyncio
import json
import os
from unittest.mock import MagicMock, patch
from orchestrator import PipelineOrchestrator

# Mock LLM Client
class MockLLMClient:
    async def call_llm_async(self, system_prompt, user_prompt, model, temperature, **kwargs):
        # Return different responses based on the stage (detected via prompt content)
        if "career research agent" in system_prompt.lower():
            return json.dumps({
                "implied_metrics": ["40% reduction in latency", "99.9% uptime"],
                "domain_vocab": ["Kubernetes", "Microservices"],
                "implied_skills": ["System Design", "Docker"],
                "work_archetypes": ["Scaling", "Migration"],
                "insider_tips": "Focus on high-availability."
            })
        elif "expert resume writer" in system_prompt.lower():
            # Check if golden bullets were injected
            if "GOLDEN PATTERN LIBRARY" in system_prompt:
                print("\n✅ SUCCESS: Golden Pattern Library found in Drafter prompt!")
                if "Refactored [Java] monolith" in system_prompt:
                    print("✅ SUCCESS: Structure Mapping instructions found!")
            
            return json.dumps({
                "rewritten_experiences": [
                    {
                        "company": "Tech Corp",
                        "role": "Senior Engineer",
                        "bullets": [
                            "Refactored legacy Python monolith into microservices, reducing deployment time by 40%.",
                            "Optimized database queries resulting in 25% faster page loads."
                        ],
                        "metrics_used": ["40% reduction", "25% faster"]
                    }
                ],
                "seniority_applied": "senior",
                "quantification_score": 0.95
            })
        elif "expert resume editor" in system_prompt.lower():
            return json.dumps({
                "final_markdown": "## Senior Engineer | Tech Corp\n\n* Refactored legacy Python monolith into microservices, reducing deployment time by 40%.",
                "final_plain_text": "Senior Engineer at Tech Corp: Refactored monolith...",
                "editorial_notes": "Polished for impact.",
                "domain_terms_used": ["microservices"],
                "quantification_check": {"score": 0.9}
            })
        return "{}"

async def test_pipeline_with_golden_bullets():
    print("🚀 Starting Pipeline Test with Golden Bullets & Structure Mapping...")
    
    mock_llm = MockLLMClient()
    orchestrator = PipelineOrchestrator(mock_llm)
    
    resume_text = "Senior Engineer at Tech Corp. Fixed messy Python scripts and moved things to servers."
    job_ad = "Looking for a Senior Software Engineer to scale our microservices architecture."
    experiences = [
        {"company": "Tech Corp", "role": "Senior Engineer", "description": "Fixed messy Python scripts."}
    ]
    
    # Sample Golden Bullets (simulating Pinecone results)
    golden_bullets = [
        "Refactored Java monolith to microservices, reducing build time by 40%.",
        "Architected high-availability clusters on AWS, ensuring 99.99% uptime.",
        "Led team of 10 engineers in migrating legacy systems to cloud-native architecture."
    ]
    
    print("\n--- Running Pipeline ---")
    result = await orchestrator.run_pipeline(
        resume_text=resume_text,
        job_ad=job_ad,
        experiences=experiences,
        golden_bullets=golden_bullets
    )
    
    print("\n--- Results Analysis ---")
    final_output = result.get("final_output", {})
    print(f"Status: {result['metrics']['pipeline_status']}")
    print(f"Seniority Applied: {final_output.get('seniority_level')}")
    
    print("\n--- Final Resume Snippet ---")
    print(final_output.get("final_written_section_markdown")[:200])
    
    # Verify the "Structure Mapping" effect in the mock output
    # (In a real test, we'd check if the LLM actually followed the pattern)
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_pipeline_with_golden_bullets())
