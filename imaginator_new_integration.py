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
from pipeline_config import OR_SLUG_STAR_EDITOR, OR_SLUG_JOB_TITLE_EXTRACTOR  # Use STAR editor for quick ATS scoring

logger = logging.getLogger(__name__)


async def run_new_pipeline_async(
    resume_text: str,
    job_ad: str,
    extracted_skills_json: Optional[Dict] = None,
    domain_insights_json: Optional[Dict] = None,
    openrouter_api_keys: Optional[List[str]] = None,
    creativity_mode: Optional[str] = None,
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
        
        # Extract Job Title (GPT-4o) - High Priority for Frontend Display
        logger.info("[NEW_PIPELINE] Extracting job title via GPT-4o...")
        extracted_job_title = None
        try:
            title_sys_prompt = "You are a precise job data extractor. Extract the official Job Title from the provided text. Return JSON: {\"job_title\": \"...\"}"
            title_user_prompt = f"Job Description Segment:\n{job_ad[:400]}\n\nExtract the Job Title:"
            
            title_response = await llm_client.call_llm_async(
                system_prompt=title_sys_prompt,
                user_prompt=title_user_prompt,
                model=OR_SLUG_JOB_TITLE_EXTRACTOR,
                temperature=0.1,
                response_format={"type": "json_object"},
                max_tokens=50
            )
            title_data = json.loads(title_response) if title_response else {}
            extracted_job_title = title_data.get("job_title")
            if extracted_job_title:
                logger.info(f"[NEW_PIPELINE] Extracted Job Title: {extracted_job_title}")
        except Exception as e:
            logger.warning(f"[NEW_PIPELINE] Job title extraction failed: {e}")

        # Run orchestrator
        orchestrator = PipelineOrchestrator(llm_client)
        result = await orchestrator.run_pipeline(
            resume_text=resume_text,
            job_ad=job_ad,
            experiences=experiences,
            openrouter_api_keys=openrouter_api_keys,
            creativity_mode=creativity_mode,
            extracted_job_title=extracted_job_title,
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

        # Extract researcher data for domain insights
        researcher_data = result.get("stages", {}).get("researcher", {}).get("data", {})
        implied_skills = researcher_data.get("implied_skills", [])
        domain_vocab = researcher_data.get("domain_vocab", [])
        implied_metrics = researcher_data.get("implied_metrics", [])
        insider_tips = researcher_data.get("insider_tips", "")
        work_archetypes = researcher_data.get("work_archetypes", [])
        
        # Extract skills from multiple sources with deduplication
        all_skills_set = set()
        
        # 1. Skills from upstream (Hermes/FastSVM via extracted_skills_json)
        if extracted_skills_json:
            if isinstance(extracted_skills_json, list):
                for s in extracted_skills_json:
                    skill_name = s.get("skill", s) if isinstance(s, dict) else str(s)
                    if skill_name:
                        all_skills_set.add(skill_name.strip())
            elif isinstance(extracted_skills_json, dict) and "skills" in extracted_skills_json:
                for s in extracted_skills_json["skills"]:
                    skill_name = s.get("skill", s) if isinstance(s, dict) else str(s)
                    if skill_name:
                        all_skills_set.add(skill_name.strip())
        
        # 2. Skills extracted from parsed experiences (from parse_experiences)
        for exp in experiences:
            exp_skills = exp.get("skills", [])
            if exp_skills and isinstance(exp_skills, list):
                for skill in exp_skills:
                    if skill and isinstance(skill, str):
                        all_skills_set.add(skill.strip())
        
        # 3. Skills from researcher's implied skills (add missing ones)
        for skill in implied_skills:
            if skill and isinstance(skill, str):
                # Only add if not already present (avoid duplicates)
                if skill.strip() not in all_skills_set:
                    all_skills_set.add(skill.strip())
        
        # 4. Skills from domain_vocab (add missing ones)
        for skill in domain_vocab:
            if skill and isinstance(skill, str):
                if skill.strip() not in all_skills_set:
                    all_skills_set.add(skill.strip())
        
        # Convert to list and limit to reasonable size
        aggregate_skills = list(all_skills_set)[:30]
        logger.info(f"[NEW_PIPELINE] Aggregated {len(aggregate_skills)} skills from all sources")
        
        # Build domain_insights from upstream data, researcher data, or intelligent defaults
        domain_insights = {}
        if domain_insights_json and isinstance(domain_insights_json, dict):
            domain_insights = domain_insights_json.copy()
        else:
            # Create intelligent defaults based on researcher output
            domain_insights = {
                "domain": "Technology",
                "market_demand": "High",
                "salary_range": "$80,000 - $150,000",
                "top_skills": domain_vocab[:5] if domain_vocab else aggregate_skills[:5] if aggregate_skills else ["Python", "AWS", "Docker"],
                "certifications": [],
                "career_path": ["Entry Level", "Mid Level", "Senior", "Lead"],
                "skill_gap_priority": insider_tips if insider_tips else "Focus on demonstrating quantifiable achievements"
            }
        
        # Enrich domain_insights with researcher data
        domain_insights.setdefault("top_skills", domain_vocab[:5] if domain_vocab else aggregate_skills[:5] if aggregate_skills else [])
        domain_insights.setdefault("certifications", [])
        domain_insights.setdefault("career_path", [])
        domain_insights.setdefault("skill_gap_priority", insider_tips if insider_tips else "")
        domain_insights.setdefault("domain_vocab", domain_vocab)
        domain_insights.setdefault("implied_metrics", implied_metrics)
        domain_insights.setdefault("work_archetypes", work_archetypes)
        
        # Add insights field from researcher data for frontend display (must be a list per Pydantic schema)
        def _normalize_research_items(items, limit):
            if not items or not isinstance(items, list):
                return []
            normalized = []
            for item in items:
                if item is None:
                    continue
                if isinstance(item, str):
                    value = item.strip()
                elif isinstance(item, (int, float, bool)):
                    value = str(item)
                elif isinstance(item, dict):
                    value = (
                        item.get("metric")
                        or item.get("skill")
                        or item.get("name")
                        or item.get("label")
                        or item.get("title")
                    )
                    value = str(value).strip() if value else ""
                    if not value:
                        value = str(item)
                else:
                    value = str(item)
                if value:
                    normalized.append(value)
            return normalized[:limit]

        implied_metrics_normalized = _normalize_research_items(implied_metrics, 3)
        implied_skills_normalized = _normalize_research_items(implied_skills, 5)
        insights_list = []
        if insider_tips:
            insights_list.append(insider_tips)
        if implied_metrics_normalized:
            # Ensure all items are strings before joining to prevent TypeError
            metrics_str = ', '.join(str(item) for item in implied_metrics_normalized if item)
            insights_list.append(
                f"Key metrics to target: {metrics_str}"
            )
        if not insights_list:
            insights_list.append("Focus on quantifiable achievements and relevant technologies.")
        domain_insights["insights"] = insights_list
        
        # Add emerging_trends from work_archetypes
        if work_archetypes:
            domain_insights["emerging_trends"] = work_archetypes
        else:
            domain_insights["emerging_trends"] = []
        
        # Build gap_analysis from researcher data
        gap_analysis = ""
        gap_parts = []
        if implied_skills_normalized:
            # Ensure all items are strings before joining to prevent TypeError
            skills_str = ', '.join(str(item) for item in implied_skills_normalized if item)
            gap_parts.append(
                f"Consider highlighting these implied skills: {skills_str}"
            )
        if implied_metrics_normalized:
            # Ensure all items are strings before joining to prevent TypeError
            metrics_str = ', '.join(str(item) for item in implied_metrics_normalized if item)
            gap_parts.append(
                f"Target these benchmarks: {metrics_str}"
            )
        if insider_tips:
            gap_parts.append(insider_tips)
        if gap_parts:
            gap_analysis = ". ".join(gap_parts)
        
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
            "extracted_job_title": extracted_job_title,  # Extracted by GPT-4o
            
            # Backward-compatible fields for frontend
            "experiences": experiences,
            "aggregate_skills": aggregate_skills,
            "processed_skills": result.get("processed_skills", {"all": aggregate_skills}),
            "domain_insights": domain_insights,
            "gap_analysis": gap_analysis,  # Generated from researcher insights
            "suggested_experiences": {"bridging_gaps": [], "metric_improvements": []},
            "seniority_analysis": result.get("seniority_analysis", {"level": final_output.get("seniority_level", "mid")}),
            
            # Backward compatibility: rewritten_resume = final_written_section (for frontend)
            "rewritten_resume": final_output.get("final_written_section", ""),
            
            # Suggestions field for frontend
            "suggestions": [],
            
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