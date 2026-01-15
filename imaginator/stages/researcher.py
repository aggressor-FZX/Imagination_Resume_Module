#!/usr/bin/env python3
"""
Imaginator Stage 1: Researcher
Heavy Start - Aggressive Dossier Compiler with Web Search
"""
import json
import logging
import time
from typing import Any, Dict, List, Optional

from ..gateway import call_llm_async
from ..config import MODEL_STAGE_1

logger = logging.getLogger(__name__)


async def run_stage1_researcher(
    resume_text: str,
    job_ad: str,
    extracted_skills: List[str],
    experiences: List[Dict[str, Any]],
    openrouter_api_keys: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Stage 1: Researcher - Heavy Start with Web Search
    
    Uses DeepSeek v3.2:online with web search to discover:
    - Implied skills from job description
    - Reasonable metrics and quantifiable achievements for candidate's field
    - STAR pattern suggestions based on candidate's experience and industry best practices
    
    This stage runs FIRST and provides research findings for downstream stages.
    
    Args:
        resume_text: Raw resume text
        job_ad: Target job description
        extracted_skills: Skills from Hermes/FastSVM
        experiences: Parsed work experiences
        
    Returns:
        Dict with:
            - implied_skills: List[str] - Skills inferred from job ad via web research
            - industry_metrics: List[str] - Typical quantifiable metrics for this role
            - star_suggestions: List[Dict] - STAR pattern ideas for each experience
            - research_notes: str - Additional insights from web research
    """
    from config import settings
    
    logger.info("[RESEARCHER] === Stage 1: Starting web-grounded research ===")
    
    # Track stage metrics
    stage_metrics = {"start": time.time()}
    
    # Build MINIMAL context (job titles + skills ONLY, no full resume data for cost optimization)
    job_titles = []
    for exp in experiences[:5]:  # Limit to 5 most recent
        if isinstance(exp, dict):
            title = exp.get("title_line", "")
            if title:
                job_titles.append(title)
    
    job_titles_text = ", ".join(job_titles) if job_titles else "Not specified"
    skills_text = ", ".join(extracted_skills[:30]) if extracted_skills else "No skills extracted"
    
    system_prompt = """You are a career research assistant with web search. Respond briefly in JSON format.

Task: Use web search to find:
1. IMPLIED SKILLS from job description (not explicitly stated but expected)
2. INDUSTRY METRICS (typical quantifiable achievements for this role)
3. STAR PATTERN IDEAS (Situation-Task-Action-Result examples)

Return JSON:
{
  "implied_skills": ["skill1", "skill2"],
  "industry_metrics": ["metric1", "metric2"],
  "star_suggestions": [{"experience_index": 0, "result_metrics": ["pattern"]}],
  "research_notes": "Brief insights"
}"""
    
    user_prompt = f"""Web search: Software engineering skills and metrics for this role.

Candidate Job Titles: {job_titles_text}
Extracted Skills: {skills_text}

Target Job Description:
{job_ad[:800]}

Find: implied skills, typical metrics, STAR pattern examples."""
    
    try:
        if getattr(settings, "environment", "") == "test":
            # Mock response for testing
            mock_research = {
                "implied_skills": ["CI/CD", "Kubernetes", "Microservices"],
                "industry_metrics": ["Reduced deployment time by 40%", "Increased test coverage to 85%"],
                "star_suggestions": [{
                    "experience_index": 0,
                    "situation": "Modernizing legacy infrastructure",
                    "task_ideas": ["Migrate to cloud", "Implement automation"],
                    "action_patterns": ["Architected cloud solution", "Automated deployment pipeline"],
                    "result_metrics": ["Reduced costs by 30%", "Improved uptime to 99.9%"]
                }],
                "research_notes": "Mock research findings for testing"
            }
            logger.info("[RESEARCHER] Using mock research data (test environment)")
            stage_metrics["end"] = time.time()
            stage_metrics["duration_ms"] = int((stage_metrics["end"] - stage_metrics["start"]) * 1000)
            return mock_research
        
        logger.info("[RESEARCHER] Calling DeepSeek v3.2:online with web search enabled (COST-OPTIMIZED)")
        response = await call_llm_async(
            system_prompt,
            user_prompt,
            enable_web_search=True,
            web_search_options={
                "max_results": 5  # Minimal search for cost
            },
            temperature=0.1,      # Deterministic, short outputs
            max_tokens=1100,      # Extended limit for comprehensive research output
            openrouter_api_keys=openrouter_api_keys,
            **kwargs
        )
        
        # Parse JSON response
        try:
            research_data = json.loads(response)
            logger.info(f"[RESEARCHER] Discovered {len(research_data.get('implied_skills', []))} implied skills")
            logger.info(f"[RESEARCHER] Found {len(research_data.get('industry_metrics', []))} metric patterns")
            logger.info(f"[RESEARCHER] Generated {len(research_data.get('star_suggestions', []))} STAR suggestions")
            
            stage_metrics["end"] = time.time()
            stage_metrics["duration_ms"] = int((stage_metrics["end"] - stage_metrics["start"]) * 1000)
            research_data["_metrics"] = stage_metrics
            
            return research_data
        except json.JSONDecodeError:
            logger.warning("[RESEARCHER] Failed to parse JSON, extracting manually")
            # Fallback: try to extract JSON object from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                research_data = json.loads(json_match.group(0))
                stage_metrics["end"] = time.time()
                stage_metrics["duration_ms"] = int((stage_metrics["end"] - stage_metrics["start"]) * 1000)
                research_data["_metrics"] = stage_metrics
                return research_data
            else:
                logger.error("[RESEARCHER] No JSON found in response, returning minimal data")
                stage_metrics["end"] = time.time()
                stage_metrics["duration_ms"] = int((stage_metrics["end"] - stage_metrics["start"]) * 1000)
                return {
                    "implied_skills": [],
                    "industry_metrics": [],
                    "star_suggestions": [],
                    "research_notes": response[:500],
                    "_metrics": stage_metrics
                }
    
    except Exception as e:
        logger.error(f"[RESEARCHER] Error during research stage: {e}")
        stage_metrics["end"] = time.time()
        stage_metrics["duration_ms"] = int((stage_metrics["end"] - stage_metrics["start"]) * 1000) if stage_metrics.get("start") else None
        return {
            "implied_skills": [],
            "industry_metrics": [],
            "star_suggestions": [],
            "research_notes": f"Research failed: {str(e)}",
            "_metrics": stage_metrics
        }