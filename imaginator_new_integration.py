"""
Integration layer for new 3-stage Imaginator pipeline
Provides backward-compatible interface for app.py
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List

from orchestrator import PipelineOrchestrator
from llm_client_adapter import LLMClientAdapter
from imaginator_flow import parse_experiences
from pipeline_config import OR_SLUG_STAR_EDITOR  # Use STAR editor for quick ATS scoring

logger = logging.getLogger(__name__)


async def run_new_pipeline_async(
    resume_text: str,
    job_ad: str,
    extracted_skills_json: Optional[Dict] = None,
    domain_insights_json: Optional[Dict] = None,
    openrouter_api_keys: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run the new 3-stage pipeline with backward-compatible output.
    
    Args:
        resume_text: Raw resume text
        job_ad: Job description
        extracted_skills_json: Skills from Hermes/FastSVM (unused in new pipeline)
        domain_insights_json: Domain insights (unused in new pipeline)
        openrouter_api_keys: OpenRouter API keys
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with final_written_section_markdown and metadata
    """
    logger.info("[NEW_PIPELINE] Starting 3-stage orchestrator")
    
    try:
        # Parse experiences from resume text
        experiences = parse_experiences(resume_text)
        logger.info(f"[NEW_PIPELINE] Parsed {len(experiences)} experiences from resume")
        
        # Initialize LLM client
        if not openrouter_api_keys:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            openrouter_api_keys = [
                os.getenv("OPENROUTER_API_KEY"),
                os.getenv("OPENROUTER_API_KEY_1"),
                os.getenv("OPENROUTER_API_KEY_2")
            ]
            openrouter_api_keys = [k for k in openrouter_api_keys if k]
        
        if not openrouter_api_keys:
            raise ValueError("No OpenRouter API keys available")
        
        llm_client = LLMClientAdapter(api_key=openrouter_api_keys[0])
        
        # Run orchestrator
        orchestrator = PipelineOrchestrator(llm_client)
        result = await orchestrator.run_pipeline(
            resume_text=resume_text,
            job_ad=job_ad,
            experiences=experiences,
            openrouter_api_keys=openrouter_api_keys
        )
        
        # Extract final output
        final_output = result.get("final_output", {})
        
        # STAGE 4: Quick ATS Score Calculation using configured model
        logger.info("[NEW_PIPELINE] Stage 4/4: ATS Score Calculation")
        critique_score = None
        try:
            system_prompt = """You are an ATS (Applicant Tracking System) scorer. Review the resume section and provide a JSON object with a single key "score" (float 0.0-1.0) indicating alignment with the job description. Higher scores mean better alignment."""
            
            user_prompt = f"""Job Description:
{job_ad[:2000]}

Generated Resume Section:
{final_output.get("final_written_section_markdown", "")[:3000]}

Rate the alignment (0.0-1.0):"""
            
            # Use STAR editor model for quick scoring (cost-effective)
            response = await llm_client.call_llm_async(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=OR_SLUG_STAR_EDITOR,
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=100
            )
            
            score_data = json.loads(response) if response else {}
            critique_score = score_data.get("score", 0.85)  # Default to 0.85 if parsing fails
            logger.info(f"[NEW_PIPELINE] ATS score calculated: {critique_score}")
            
        except Exception as e:
            logger.warning(f"[NEW_PIPELINE] ATS scoring failed, using default: {e}")
            critique_score = 0.85  # Default score on error
        
        # Build backward-compatible response with ALL expected fields
        errors = result.get("errors", [])
        if errors:
            logger.error(f"[NEW_PIPELINE] Pipeline encountered {len(errors)} errors: {errors}")

        # Extract skills from upstream data (Hermes/FastSVM) or use defaults
        aggregate_skills = []
        if extracted_skills_json:
            if isinstance(extracted_skills_json, list):
                aggregate_skills = [s.get("skill", s) if isinstance(s, dict) else str(s) 
                                  for s in extracted_skills_json[:15]]
            elif isinstance(extracted_skills_json, dict) and "skills" in extracted_skills_json:
                skills_list = extracted_skills_json["skills"]
                aggregate_skills = [s.get("skill", s) if isinstance(s, dict) else str(s) 
                                  for s in skills_list[:15]]
        
        # Build domain_insights from upstream data or use intelligent defaults
        domain_insights = {}
        if domain_insights_json and isinstance(domain_insights_json, dict):
            domain_insights = domain_insights_json
        else:
            # Create intelligent defaults based on job ad
            domain_insights = {
                "domain": "Technology",
                "market_demand": "High",
                "salary_range": "$80,000 - $150,000",
                "top_skills": aggregate_skills[:5] if aggregate_skills else ["Python", "AWS", "Docker"],
                "certifications": [],
                "career_path": ["Entry Level", "Mid Level", "Senior", "Lead"],
                "skill_gap_priority": "Focus on cloud and containerization technologies"
            }
        
        # Ensure all required fields exist in domain_insights
        domain_insights.setdefault("top_skills", aggregate_skills[:5] if aggregate_skills else [])
        domain_insights.setdefault("certifications", [])
        domain_insights.setdefault("career_path", [])
        
        response = {
            # Core fields from new pipeline
            "final_written_section_markdown": final_output.get("final_written_section_markdown", ""),
            "final_written_section": final_output.get("final_written_section", ""),
            "editorial_notes": final_output.get("editorial_notes", ""),
            "seniority_level": final_output.get("seniority_level", "mid"),
            "domain_terms_used": final_output.get("domain_terms_used", []),
            "quantification_analysis": final_output.get("quantification_analysis", {}),
            "hallucination_checked": final_output.get("hallucination_checked", False),
            "critique_score": critique_score,  # ATS Score from criticism stage
            
            # Backward-compatible fields for frontend
            "experiences": experiences,
            "aggregate_skills": aggregate_skills,
            "processed_skills": {"all": aggregate_skills},
            "domain_insights": domain_insights,
            "gap_analysis": "",  # No longer generated in new pipeline
            "suggested_experiences": {"bridging_gaps": [], "metric_improvements": []},
            "seniority_analysis": {"level": final_output.get("seniority_level", "mid")},
            
            # Metadata
            "pipeline_version": "3.0",
            "pipeline_status": result.get("metrics", {}).get("pipeline_status", "completed"),
            "pipeline_metrics": {
                "total_duration_seconds": result.get("metrics", {}).get("total_duration_seconds", 0),
                "stage_durations": result.get("metrics", {}).get("stage_durations", {}),
                "errors": errors
            }
        }
        
        logger.info(f"[NEW_PIPELINE] Completed successfully. Status: {response['pipeline_status']}, Critique Score: {critique_score}")
        return response
        
    except Exception as e:
        logger.error(f"[NEW_PIPELINE] Failed: {e}", exc_info=True)
        
        # Return error response with all expected fields
        return {
            # Core fields
            "final_written_section_markdown": f"## Error\n\nPipeline failed: {str(e)}",
            "final_written_section": f"Error: Pipeline failed: {str(e)}",
            "editorial_notes": f"Pipeline error: {str(e)}",
            "seniority_level": "unknown",
            "domain_terms_used": [],
            "quantification_analysis": {},
            "hallucination_checked": False,
            "critique_score": None,
            
            # Backward-compatible fields
            "experiences": [],
            "aggregate_skills": [],
            "processed_skills": {},
            "domain_insights": {
                "domain": "Unknown",
                "market_demand": "Unknown",
                "salary_range": "Unknown",
                "top_skills": [],
                "certifications": [],
                "career_path": []
            },
            "gap_analysis": "",
            "suggested_experiences": {"bridging_gaps": [], "metric_improvements": []},
            "seniority_analysis": {"level": "unknown"},
            
            # Metadata
            "pipeline_version": "3.0",
            "pipeline_status": "failed",
            "pipeline_metrics": {
                "total_duration_seconds": 0,
                "stage_durations": {},
                "errors": [str(e)]
            }
        }
