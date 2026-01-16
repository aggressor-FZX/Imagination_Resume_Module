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
        
        # Build backward-compatible response
        response = {
            "final_written_section_markdown": final_output.get("final_written_section_markdown", ""),
            "final_written_section": final_output.get("final_written_section", ""),
            "editorial_notes": final_output.get("editorial_notes", ""),
            "seniority_level": final_output.get("seniority_level", "mid"),
            "domain_terms_used": final_output.get("domain_terms_used", []),
            "quantification_analysis": final_output.get("quantification_analysis", {}),
            "hallucination_checked": final_output.get("hallucination_checked", False),
            "pipeline_version": "3.0",
            "pipeline_status": result.get("metrics", {}).get("pipeline_status", "completed"),
            "pipeline_metrics": {
                "total_duration_seconds": result.get("metrics", {}).get("total_duration_seconds", 0),
                "stage_durations": result.get("metrics", {}).get("stage_durations", {}),
                "errors": result.get("errors", [])
            }
        }
        
        logger.info(f"[NEW_PIPELINE] Completed successfully. Status: {response['pipeline_status']}")
        return response
        
    except Exception as e:
        logger.error(f"[NEW_PIPELINE] Failed: {e}", exc_info=True)
        
        # Return error response
        return {
            "final_written_section_markdown": f"## Error\n\nPipeline failed: {str(e)}",
            "final_written_section": f"Error: Pipeline failed: {str(e)}",
            "editorial_notes": f"Pipeline error: {str(e)}",
            "seniority_level": "unknown",
            "domain_terms_used": [],
            "quantification_analysis": {},
            "hallucination_checked": False,
            "pipeline_version": "3.0",
            "pipeline_status": "failed",
            "pipeline_metrics": {
                "total_duration_seconds": 0,
                "stage_durations": {},
                "errors": [str(e)]
            }
        }
