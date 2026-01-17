"""
New Imaginator Flow - 3-Stage Modular Pipeline
Replaces old monolithic imaginator_flow.py with new orchestrator-based approach
"""
import json
import logging
from typing import Dict, Any, Optional, List
import aiohttp

from orchestrator import PipelineOrchestrator
from llm_client_adapter import LLMClientAdapter
from config import settings

logger = logging.getLogger(__name__)

# Global metrics tracking (for backward compatibility with app.py)
RUN_METRICS = {
    "calls": [],
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_tokens": 0,
    "estimated_cost_usd": 0.0,
    "failures": [],
    "durations_ms": {}
}

# Shared HTTP session (for backward compatibility)
_shared_http_session: Optional[aiohttp.ClientSession] = None

def configure_shared_http_session(session: Optional[aiohttp.ClientSession]):
    """Configure shared HTTP session for microservice calls."""
    global _shared_http_session
    _shared_http_session = session

def redact_for_logging(data: Any) -> Any:
    """Redact sensitive data for logging."""
    if isinstance(data, dict):
        return {k: "***REDACTED***" if "key" in k.lower() or "token" in k.lower() else v for k, v in data.items()}
    return data

def validate_output_schema(output: Dict[str, Any]):
    """Validate output schema (placeholder for backward compatibility)."""
    required_fields = ["final_written_section", "processing_status"]
    for field in required_fields:
        if field not in output:
            logger.warning(f"[SCHEMA_VALIDATION] Missing required field: {field}")

def ensure_json_dict(data: Any) -> Dict:
    """Ensure data is a JSON dict."""
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}

def parse_experiences(text: str) -> List[Dict]:
    """
    Parse experiences from resume text.
    Simple implementation - extracts basic structure.
    """
    experiences = []
    lines = text.split('\n')
    current_exp = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for company/role patterns
        if '|' in line or 'at' in line.lower():
            if current_exp:
                experiences.append(current_exp)
            current_exp = {
                "title_line": line,
                "company": line.split('|')[-1].strip() if '|' in line else "",
                "role": line.split('|')[0].strip() if '|' in line else line,
                "description": "",
                "skills": []
            }
        elif current_exp and line.startswith(('•', '-', '*')):
            current_exp["description"] += line + " "
    
    if current_exp:
        experiences.append(current_exp)
    
    return experiences

async def run_analysis_async(
    resume_text: str,
    job_ad: Optional[str] = None,
    extracted_skills_json: Optional[Dict] = None,
    domain_insights_json: Optional[Dict] = None,
    openrouter_api_keys: Optional[List[str]] = None,
    confidence_threshold: float = 0.7,
    **kwargs
) -> Dict[str, Any]:
    """
    Run the new 3-stage pipeline for resume analysis and enhancement.
    
    This replaces the old monolithic run_analysis_async with the new modular orchestrator.
    
    Args:
        resume_text: Raw resume text
        job_ad: Target job description
        extracted_skills_json: Skills from Hermes/FastSVM
        domain_insights_json: Domain insights from upstream services
        openrouter_api_keys: OpenRouter API keys
        confidence_threshold: Confidence threshold for skills
        **kwargs: Additional arguments
        
    Returns:
        Dict with final_written_section and metadata
    """
    logger.info("[NEW PIPELINE] === Starting 3-Stage Orchestrator ===")
    
    # Parse experiences from resume text
    experiences = parse_experiences(resume_text)
    logger.info(f"[NEW PIPELINE] Parsed {len(experiences)} experiences from resume")
    
    # Initialize LLM client
    api_key = openrouter_api_keys[0] if openrouter_api_keys else settings.openrouter_api_key_1
    llm_client = LLMClientAdapter(api_key=api_key)
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(llm_client)
    
    # Run pipeline
    # Check for golden bullets in domain_insights_json
    golden_bullets = domain_insights_json.get("domain_market_data", {}).get("top_skills", []) if domain_insights_json else []
    # Note: We are using top_skills as a proxy for golden bullets for now, 
    # but in a real scenario, we would call the SmartHarvester here.
    
    result = await orchestrator.run_pipeline(
        resume_text=resume_text,
        job_ad=job_ad or "",
        experiences=experiences,
        openrouter_api_keys=openrouter_api_keys,
        golden_bullets=golden_bullets
    )
    
    # Extract final output
    final_output = result.get("final_output", {})
    
    # Build response in expected format for app.py (AnalysisResponse schema)
    # Add required 'snippet' field to experiences
    for exp in experiences:
        if "snippet" not in exp:
            exp["snippet"] = exp.get("description", "")[:200] if exp.get("description") else ""
    
    response = {
        "final_written_section": final_output.get("final_written_section_markdown", ""),
        "final_written_section_markdown": final_output.get("final_written_section_markdown", ""),
        "editorial_notes": final_output.get("editorial_notes", ""),
        "seniority_level": final_output.get("seniority_level", "mid"),
        "domain_terms_used": final_output.get("domain_terms_used", []),
        "quantification_analysis": final_output.get("quantification_analysis", {}),
        "hallucination_checked": final_output.get("hallucination_checked", False),
        
        # Required fields for AnalysisResponse schema
        "experiences": experiences,
        "aggregate_skills": extracted_skills_json.get("skills", []) if extracted_skills_json else [],
        "processed_skills": {
            "high_confidence": [],
            "medium_confidence": [],
            "low_confidence": [],
            "inferred_skills": []
        },
        "domain_insights": {
            "domain": domain_insights_json.get("domain", "unknown") if domain_insights_json else "unknown",
            "skill_gap_priority": domain_insights_json.get("skill_gap_priority", "medium") if domain_insights_json else "medium",
            **(domain_insights_json or {})
        },
        "gap_analysis": json.dumps({
            "critical_gaps": [],
            "nice_to_have_gaps": [],
            "recommendations": []
        }),
        "suggested_experiences": {
            "bridging_gaps": [],
            "metric_improvements": []
        },
        "seniority_analysis": {
            "detected_level": final_output.get("seniority_level", "mid"),
            "confidence": 0.8,
            "indicators": []
        },
        
        # Pipeline metrics
        "pipeline_version": result.get("pipeline_version", "3.0"),
        "pipeline_metrics": result.get("metrics", {}),
        "pipeline_errors": result.get("errors", []),
        "stages": result.get("stages", {})
    }
    
    # Update global metrics for app.py
    metrics = result.get("metrics", {})
    RUN_METRICS["durations_ms"]["pipeline_total_ms"] = int(metrics.get("total_duration_seconds", 0) * 1000)
    
    logger.info(f"[NEW PIPELINE] === Pipeline completed: {metrics.get('pipeline_status', 'unknown')} ===")
    
    return response

# Placeholder functions for backward compatibility
async def run_generation_async(*args, **kwargs):
    """Deprecated - now handled by orchestrator."""
    logger.warning("[DEPRECATED] run_generation_async called - this is now handled by the orchestrator")
    return {}

async def run_criticism_async(*args, **kwargs):
    """Deprecated - now handled by orchestrator."""
    logger.warning("[DEPRECATED] run_criticism_async called - this is now handled by the orchestrator")
    return {}

async def run_synthesis_async(*args, **kwargs):
    """Deprecated - now handled by orchestrator."""
    logger.warning("[DEPRECATED] run_synthesis_async called - this is now handled by the orchestrator")
    return {}
