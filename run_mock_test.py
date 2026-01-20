import json
from unittest.mock import patch
import asyncio

import imaginator_flow as mod

# Mock LLM responses
MOCK_RESPONSES = {
    "analysis": json.dumps({
        "gap_analysis": {
            "critical_gaps": ["Kubernetes", "React"],
            "nice_to_have_gaps": ["GraphQL"],
            "gap_bridging_strategy": "Focus on container orchestration and frontend frameworks."
        },
        "implied_skills": {}, "environment_capabilities": {}, "transfer_paths": [], "project_briefs": [], "multi_perspective_insights": {}, "action_plan": {}
    }),
    "generation": json.dumps({
        "gap_bridging": [
            {"skill_focus": "Kubernetes", "suggestions": ["Deployed a multi-service application to a K8s cluster, managing resources with Helm charts."]},
            {"skill_focus": "React", "suggestions": ["Developed a responsive single-page application using React and Redux for state management."]}
        ],
        "metric_improvements": [
            {"skill_focus": "Python", "suggestions": ["Optimized a data processing pipeline, reducing execution time by 30%."]}
        ]
    }),
    "criticism": json.dumps({
        "suggested_experiences": {
            "bridging_gaps": [
                {"skill_focus": "Kubernetes", "refined_suggestions": ["Architected and deployed a scalable, multi-service application to a production Kubernetes cluster using Helm, achieving 99.9% uptime."]},
                {"skill_focus": "React", "refined_suggestions": ["Engineered a responsive and accessible single-page e-commerce application using React and Redux, resulting in a 15% increase in user engagement."]}
            ],
            "metric_improvements": [
                {"skill_focus": "Python", "refined_suggestions": ["Re-architected a critical data processing pipeline in Python, reducing average execution time from 2 hours to 30 minutes (a 75% improvement)."]}
            ]
        }
    })
}

# Side effect function to return different mock responses
def llm_side_effect(*args, **kwargs):
    if "SYNTHESIS TASK" in args[1]:  # Analysis prompt
        print("--- Mocking Analysis LLM Call ---")
        return MOCK_RESPONSES["analysis"]
    elif "GENERATE" in args[0].upper():  # Generation prompt
        print("--- Mocking Generation LLM Call ---")
        return MOCK_RESPONSES["generation"]
    elif "CYNICAL" in args[0]:  # Criticism prompt
        print("--- Mocking Criticism LLM Call ---")
        return MOCK_RESPONSES["criticism"]
    return "{}"

async def main():
    with open("sample_resume.txt", "r") as f:
        sample_resume = f.read()
    with open("sample_job_ad.txt", "r") as f:
        sample_job_ad = f.read()

    with patch('imaginator_flow.call_llm_async', side_effect=llm_side_effect) as mock_llm_async, \
         patch('imaginator_flow.call_llm', side_effect=llm_side_effect) as mock_llm:

        print("\n--- Running Analysis Stage ---")
        analysis_result = await mod.run_analysis_async(
            resume_text=sample_resume,
            job_ad=sample_job_ad
        )
        
        print("\n--- Running Generation Stage ---")
        generation_result = await mod.run_generation_async(
            analysis_json=analysis_result,
            job_ad=sample_job_ad
        )

        print("\n--- Running Criticism Stage ---")
        # Extract the text content from the generation result if it's a dict
        generated_text_content = generation_result
        if isinstance(generation_result, dict):
            generated_text_content = generation_result.get("final_written_section", "")

        raw_criticism = await mod.run_criticism_async(
            generated_text=generated_text_content,
            job_ad=sample_job_ad
        )
        criticism_result = mod.ensure_json_dict(raw_criticism, "criticism")

        final_output = {
            **analysis_result,
            **criticism_result
        }
        
        print("\n--- Final Mocked Output ---")
        print(json.dumps(final_output, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
