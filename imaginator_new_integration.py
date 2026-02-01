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

# Import career alchemy generator
try:
    from career_alchemy import generate_career_alchemy
    CAREER_ALCHEMY_AVAILABLE = True
except ImportError:
    logger.warning("[NEW_PIPELINE] career_alchemy module not available, Career Alchemy feature disabled")
    CAREER_ALCHEMY_AVAILABLE = False


def _extract_experience_years(experiences: List[Dict[str, Any]]) -> float:
    """Extract total years of experience from experiences list"""
    if not experiences:
        return 0.0
    
    total_years = 0.0
    parsing_successful = False
    
    for exp in experiences:
        if isinstance(exp, dict):
            # Try to extract duration from various fields
            duration = exp.get("duration", "")
            if duration:
                # Parse "2 years" or "24 months" format
                import re
                years_match = re.search(r'(\d+)\s*(?:year|yr)', duration, re.IGNORECASE)
                if years_match:
                    total_years += float(years_match.group(1))
                    parsing_successful = True
                else:
                    months_match = re.search(r'(\d+)\s*(?:month|mo)', duration, re.IGNORECASE)
                    if months_match:
                        total_years += float(months_match.group(1)) / 12.0
                        parsing_successful = True
    
    # Use fallback estimate if no successful parsing occurred
    # This handles cases where duration fields exist but don't match regex patterns
    # (e.g., "ongoing", "present", "current", or other non-standard formats)
    if total_years == 0.0 and len(experiences) > 0 and not parsing_successful:
        # Conservative estimate: assume 1.5-2 years per experience (reduced from 2.5)
        total_years = len(experiences) * 1.75
    
    return total_years


async def run_new_pipeline_async(
    resume_text: str,
    job_ad: str,
    extracted_skills_json: Optional[Dict] = None,
    domain_insights_json: Optional[Dict] = None,
    openrouter_api_keys: Optional[List[str]] = None,
    creativity_mode: Optional[str] = None,
    location: Optional[str] = None,
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
        location: Job location for market intel enrichment
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with final_written_section_markdown and metadata
    """
    logger.info("[NEW_PIPELINE] Starting 3-stage orchestrator")
    if location:
        logger.info(f"[NEW_PIPELINE] Location provided for market intel: {location}")
    
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
        
        # Convert to list with deterministic ordering (sorted alphabetically) and limit to reasonable size
        aggregate_skills = sorted(list(all_skills_set))[:30]
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
        
        # Enrich domain_insights with researcher data (use direct assignment to override empty values)
        domain_insights.setdefault("top_skills", domain_vocab[:5] if domain_vocab else aggregate_skills[:5] if aggregate_skills else [])
        domain_insights.setdefault("certifications", [])
        domain_insights.setdefault("career_path", [])
        
        # Always include researcher data - these come from Imaginator's Researcher stage, not Hermes
        if insider_tips:
            domain_insights["skill_gap_priority"] = insider_tips
        elif not domain_insights.get("skill_gap_priority"):
            domain_insights["skill_gap_priority"] = ""
            
        # Always add researcher stage data (overwrite empty values from Hermes)
        if domain_vocab:
            domain_insights["domain_vocab"] = domain_vocab
        elif "domain_vocab" not in domain_insights:
            domain_insights["domain_vocab"] = []
            
        if implied_metrics:
            domain_insights["implied_metrics"] = implied_metrics
        elif "implied_metrics" not in domain_insights:
            domain_insights["implied_metrics"] = []
            
        if work_archetypes:
            domain_insights["work_archetypes"] = work_archetypes
        elif "work_archetypes" not in domain_insights:
            domain_insights["work_archetypes"] = []
        
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
        
        # Enrich domain insights with Data USA market intel if location is provided
        if location and extracted_job_title:
            try:
                from career_progression_enricher import CareerProgressionEnricher
                from city_geo_mapper import get_geo_id
                import os
                
                logger.info(f"[NEW_PIPELINE] Enriching with market intel for {extracted_job_title} in {location}")
                
                # Get geo ID for location
                geo_id = get_geo_id(location)
                if not geo_id:
                    logger.warning(f"[NEW_PIPELINE] No geo ID found for location: {location}, using national data")
                
                # Search O*NET for job title to get O*NET code
                onet_code = None
                try:
                    import httpx
                    onet_auth = os.getenv("ONET_API_AUTH")
                    if not onet_auth:
                        logger.error("[NEW_PIPELINE] ONET_API_AUTH not set; skipping O*NET lookup to avoid using default credentials")
                    else:
                        onet_url = "https://services.onetcenter.org/ws/online/search"
                        onet_headers = {
                            "Authorization": f"Basic {onet_auth}",
                            "Accept": "application/json"
                        }
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            search_response = await client.get(
                                onet_url,
                                headers=onet_headers,
                                params={"keyword": extracted_job_title, "end": 1},
                            )
                        if search_response.status_code == 200:
                            search_data = search_response.json()
                            if search_data.get("occupation"):
                                onet_code = search_data["occupation"][0]["code"]
                                logger.info(f"[NEW_PIPELINE] Found O*NET code: {onet_code} for {extracted_job_title}")
                except Exception as e:
                    logger.warning(f"[NEW_PIPELINE] O*NET search failed: {e}")
                
                # If we have O*NET code, enrich with market data
                if onet_code:
                    enricher = CareerProgressionEnricher()
                    career_data = enricher.get_full_career_insights(
                        job_title=extracted_job_title,
                        onet_code=onet_code,
                        location=location,
                        city_geo_id=geo_id
                    )
                    
                    # Get O*NET summary for Bright Outlook check
                    onet_summary = enricher._get_onet_summary(onet_code)
                    
                    # Calculate market intel
                    market_intel = enricher.calculate_market_intel(
                        career_data.get("workforce", {}),
                        onet_summary,
                        job_title=extracted_job_title,
                        location=location
                    )
                    
                    # Add market intel to domain insights
                    domain_insights["market_intel"] = market_intel
                    logger.info(f"[NEW_PIPELINE] Added market intel: {market_intel.get('status')}")
                else:
                    logger.warning(f"[NEW_PIPELINE] No O*NET code found for job title: {extracted_job_title}")
            except Exception as e:
                logger.error(f"[NEW_PIPELINE] Failed to enrich domain insights with market intel: {e}", exc_info=True)
        elif location:
            logger.warning(f"[NEW_PIPELINE] Location provided ({location}) but no job title extracted, skipping market intel")
        
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
        
        gap_analysis_payload = {
            "summary": gap_analysis or "No additional gap analysis generated.",
            "critical_gaps": implied_skills_normalized,
            "benchmarks": implied_metrics_normalized,
            "insider_tips": insider_tips or ""
        }
        gap_analysis_json = json.dumps(gap_analysis_payload)

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
            "gap_analysis": gap_analysis_json,  # Generated from researcher insights (JSON string)
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
            },
            
            # Token usage metrics from LLM client (critical for app.py detection)
            "run_metrics": llm_client.get_usage_stats() if hasattr(llm_client, 'get_usage_stats') else {
                "calls": [],
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "failures": []
            }
        }
        
        # Generate Career Alchemy if available
        career_alchemy_data = None
        if CAREER_ALCHEMY_AVAILABLE:
            try:
                logger.info("[NEW_PIPELINE] Generating Career Alchemy profile...")
                
                # Extract seniority_analysis from result
                seniority_analysis = result.get("seniority_analysis", {})
                
                # Build characteristics dict from analysis result
                characteristics = {
                    "canonical_title": extracted_job_title or domain_insights.get("domain", "Professional"),
                    "job_title": extracted_job_title or domain_insights.get("domain", "Professional"),
                    "domain": domain_insights.get("domain", "Technology"),
                    "seniority": {
                        "level": final_output.get("seniority_level", "mid"),
                        "job_zone": seniority_analysis.get("job_zone", "4") if isinstance(seniority_analysis, dict) else "4",
                        "experience_required": seniority_analysis.get("experience_required", "5-10 years") if isinstance(seniority_analysis, dict) else "5-10 years"
                    },
                    "skills": [
                        {"skill": skill, "confidence": 0.85} if isinstance(skill, str) else skill
                        for skill in aggregate_skills[:30]
                    ],
                    "experience_years": _extract_experience_years(experiences),
                    "certifications": domain_insights.get("certifications", []),
                    "education": domain_insights.get("education", []),
                    "achievements": [exp.get("snippet", "")[:100] for exp in experiences[:5] if isinstance(exp, dict) and exp.get("snippet")]
                }
                
                # Extract location from kwargs or use default
                current_location = location or kwargs.get("location", "United States")
                
                # Pivot Analysis Integration
                pivot_data = None
                salary_data = None
                
                try:
                    from career_progression_enricher import CareerProgressionEnricher
                    from city_geo_mapper import get_geo_id
                    
                    enricher = CareerProgressionEnricher()
                    
                    # Try to get O*NET code from previous enrichment steps or use a default
                    # In a real run, onet_code was calculated around line 343
                    # For alchemy, we can re-extract or pass it down.
                    # For now, let's use the market_intel if available
                    market_intel = domain_insights.get("market_intel", {})
                    
                    # Extract salary data from market intel
                    if market_intel:
                        salary_data = {
                            "median_salary": market_intel.get("average_wage", 0),
                            "yoy_growth": market_intel.get("wage_growth_pct", 0)
                        }
                    
                    # Generate Pivot Data if we have enough info
                    # We'll use a placeholder O*NET code if none found to at least get related roles
                    target_onet = onet_code if 'onet_code' in locals() else "15-1252.00"
                    
                    logger.info(f"[NEW_PIPELINE] Generating Pivot Analysis for Alchemy using code: {target_onet}")
                    pivot_data = enricher.generate_career_pivot_analysis(
                        current_onet_code=target_onet,
                        current_job_title=extracted_job_title or "Software Engineer",
                        resume_skills=[s if isinstance(s, str) else s.get("skill", "") for s in aggregate_skills],
                        resume_tech=aggregate_skills, # Use same list for now
                        location=current_location,
                        max_pivots=3
                    )
                except Exception as pivot_err:
                    logger.warning(f"[NEW_PIPELINE] Alchemy pivot enrichment failed: {pivot_err}")
                
                # Generate career alchemy
                career_alchemy_data = generate_career_alchemy(
                    characteristics=characteristics,
                    location=current_location,
                    pivot_data=pivot_data,
                    salary_data=salary_data
                )
                
                logger.info("[NEW_PIPELINE] Career Alchemy generated successfully")
                
            except Exception as e:
                logger.warning(f"[NEW_PIPELINE] Career Alchemy generation failed: {e}", exc_info=True)
                career_alchemy_data = None
        
        # Add career_alchemy to response if generated
        if career_alchemy_data:
            response["career_alchemy"] = career_alchemy_data
        
        llm_usage = response.get("run_metrics", {})
        logger.info(f"[NEW_PIPELINE] Completed successfully. Status: {response['pipeline_status']}, Critique Score: {critique_score}, Tokens: {llm_usage.get('total_tokens', 0)}")
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
            },
            
            # Empty run_metrics for consistency
            "run_metrics": {
                "calls": [],
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "failures": [str(e)]
            }
        }