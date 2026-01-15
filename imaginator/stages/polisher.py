#!/usr/bin/env python3
"""
Imaginator Stage 4: Polisher
Analytical Finish - Final QC and Editorial Polish
"""
import logging
import time
import json
from typing import Any, Dict, Optional, List

from ..gateway import call_llm_async
from ..config import MODEL_STAGE_4

logger = logging.getLogger(__name__)


async def run_stage4_polisher(
    star_draft: str,
    original_job_ad: str,
    openrouter_api_keys: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Stage 4: Polisher - Analytical Finish and Final QC
    
    Uses Gemini 2.0 Flash to perform final editorial polish and quality control.
    Re-injects the original Job Ad for 100% alignment verification.
    
    Args:
        star_draft: STAR-formatted draft from Stage 3
        original_job_ad: Original job description for final verification
        
    Returns:
        Dict with final_resume_markdown and editorial_notes
    """
    from config import settings
    
    logger.info("[POLISHER] === Stage 4: Final QC and Editorial Polish ===")
    
    # Track stage metrics
    stage_metrics = {"start": time.time()}
    
    system_prompt = """You are the Final QC Editor. 
    Compare the STAR Draft against the ORIGINAL JOB AD.
    1. Check for 100% keyword alignment.
    2. Strip any narrative 'As a...' or first-person 'I...' content.
    3. Ensure formatting is strictly bulleted.
    Return JSON with 'final_resume_markdown' and 'editorial_notes'."""
    
    user_prompt = f"DRAFT: {star_draft}\n\nORIGINAL JOB AD: {original_job_ad}"
    
    try:
        if getattr(settings, "environment", "") == "test":
            mock_result = {
                "final_resume_markdown": star_draft,
                "editorial_notes": "Mock polish applied - QC passed"
            }
            stage_metrics["end"] = time.time()
            stage_metrics["duration_ms"] = int((stage_metrics["end"] - stage_metrics["start"]) * 1000)
            mock_result["_metrics"] = stage_metrics
            return mock_result
        
        logger.info("[POLISHER] Calling Gemini 2.0 Flash for final editorial polish")
        response = await call_llm_async(
            system_prompt,
            user_prompt,
            temperature=0.3,  # Low temperature for precise editing
            max_tokens=3000,
            openrouter_api_keys=openrouter_api_keys,
            **kwargs
        )
        
        # Parse JSON response
        try:
            import json
            result = json.loads(response)
            logger.info(f"[POLISHER] Final polish complete ({len(result.get('final_resume_markdown', ''))} chars)")
            
            stage_metrics["end"] = time.time()
            stage_metrics["duration_ms"] = int((stage_metrics["end"] - stage_metrics["start"]) * 1000)
            result["_metrics"] = stage_metrics
            
            return result
        except json.JSONDecodeError:
            logger.warning("[POLISHER] Failed to parse JSON, returning raw response")
            stage_metrics["end"] = time.time()
            stage_metrics["duration_ms"] = int((stage_metrics["end"] - stage_metrics["start"]) * 1000)
            return {
                "final_resume_markdown": response,
                "editorial_notes": "Warning: JSON parsing failed, raw response returned",
                "_metrics": stage_metrics
            }
    
    except Exception as e:
        logger.error(f"[POLISHER] Error: {e}")
        stage_metrics["end"] = time.time()
        stage_metrics["duration_ms"] = int((stage_metrics["end"] - stage_metrics["start"]) * 1000) if stage_metrics.get("start") else None
        return {
            "final_resume_markdown": star_draft,
            "editorial_notes": f"Polish failed: {str(e)}",
            "_metrics": stage_metrics
        }