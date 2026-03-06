"""
Integration layer for new 3-stage Imaginator pipeline
Provides backward-compatible interface for app.py
"""

import asyncio
import json
import logging
import os
import re
import httpx
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

from orchestrator import PipelineOrchestrator
from llm_client_adapter import LLMClientAdapter
from imaginator_flow import parse_experiences
from pipeline_config import OR_SLUG_STAR_EDITOR, OR_SLUG_JOB_TITLE_EXTRACTOR  # Use STAR editor for quick ATS scoring

logger = logging.getLogger(__name__)

# Load environment variables once at module load time (not in hot path)
load_dotenv()

# Pre-load all available API keys at startup
OPENROUTER_API_KEYS = [k for k in [
    os.getenv("OPENROUTER_API_KEY"),
    os.getenv("OPENROUTER_API_KEY_1"),
    os.getenv("OPENROUTER_API_KEY_2")
] if k]

# Load API configuration from environment
ONET_API_AUTH = os.getenv("ONET_API_AUTH", "Y29naXRvbWV0cmljOjI4Mzd5cGQ=")  # Fallback for backward compatibility
ONET_API_BASE = os.getenv("ONET_API_BASE", "https://services.onetcenter.org/ws")
CAREERONESTOP_USER_ID = os.getenv("CAREERONESTOP_USER_ID", "")
CAREERONESTOP_TOKEN = os.getenv("CAREERONESTOP_TOKEN", "")
CAREERONESTOP_BASE_URL = os.getenv("CAREERONESTOP_BASE_URL", "https://api.careeronestop.org/v1")
DATAUSA_API_BASE = os.getenv("DATAUSA_API_BASE", "https://api-ithaca.datausa.io/calcs/pums.jsonrecords")
DATAUSA_USER_AGENT = os.getenv("DATAUSA_USER_AGENT", "Cogitometric-Career-Insights/1.0")

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


def _build_onet_search_variants(job_title: str) -> List[str]:
    """
    Build a deterministic keyword fallback chain for O*NET search.

    Example:
    "Senior AI Data Engineer" -> ["Senior AI Data Engineer", "AI Data Engineer", "Data Engineer", "Engineer", "AI Data"]
    """
    if not job_title:
        return []

    clean_title = re.sub(
        r'^(senior|sr\.?|junior|jr\.?|lead|principal|staff|associate)\s+',
        '',
        job_title,
        flags=re.IGNORECASE,
    ).strip()
    if not clean_title:
        return []

    tokens = clean_title.split()
    
    variants = [
        job_title,  # Original title (includes seniority prefix if not already stripped)
        clean_title,  # Title with seniority prefix removed
    ]
    
    # Strip domain prefixes (AI, ML, LLM, etc.)
    for prefix in ["ai", "ml", "llm", "nlp", "gen-ai", "generative", "prompt"]:
        stripped = re.sub(rf'^\s*{prefix}\s+', '', clean_title, flags=re.IGNORECASE).strip()
        if stripped and stripped != clean_title:
            variants.append(stripped)
    
    # Add last token and first two tokens
    if tokens:
        variants.append(tokens[-1])
    if len(tokens) >= 2:
        variants.append(" ".join(tokens[:2]))
    
    seen = set()
    deduped: List[str] = []
    for variant in variants:
        value = (variant or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _search_onet_keyword(keyword: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    Query O*NET keyword search and return candidate occupations.
    """
    if not keyword:
        return []

    headers = {
        "Authorization": f"Basic {ONET_API_AUTH}",
        "Accept": "application/json",
    }

    response = httpx.get(
        f"{ONET_API_BASE}/online/search",
        params={"keyword": keyword, "end": max(1, min(limit, 20))},
        headers=headers,
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    occupations = data.get("occupation") or []
    if not occupations:
        return []

    candidates: List[Dict[str, str]] = []
    for occupation in occupations:
        if not isinstance(occupation, dict):
            continue
        candidates.append(
            {
                "code": occupation.get("code", ""),
                "title": occupation.get("title", ""),
                "query": keyword,
            }
        )
    return candidates


def _validate_onet_code(code: str) -> bool:
    """
    Verify that an O*NET occupation code exists before trusting it.
    """
    clean_code = (code or "").strip()
    if not re.fullmatch(r"\d{2}-\d{4}\.\d{2}", clean_code):
        return False
    headers = {
        "Authorization": f"Basic {ONET_API_AUTH}",
        "Accept": "application/json",
    }
    try:
        response = httpx.get(
            f"{ONET_API_BASE}/online/occupations/{clean_code}/summary",
            headers=headers,
            timeout=10,
        )
        return response.status_code == 200
    except Exception:
        return False


def _tokenize_title(value: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (value or "").lower())


def _score_keyword_candidate(job_title: str, candidate_title: str, query: str) -> float:
    """
    Score an O*NET search candidate by semantic-ish token affinity.
    """
    title_tokens = set(_tokenize_title(job_title))
    candidate_tokens = set(_tokenize_title(candidate_title))
    query_tokens = set(_tokenize_title(query))
    if not title_tokens or not candidate_tokens:
        return 0.0

    overlap_ratio = len(title_tokens & candidate_tokens) / float(len(title_tokens))
    query_overlap = len(query_tokens & candidate_tokens) / float(len(query_tokens)) if query_tokens else 0.0

    score = (0.7 * overlap_ratio) + (0.3 * query_overlap)
    lowered_candidate = (candidate_title or "").lower()
    lowered_title = (job_title or "").lower()

    if lowered_title in lowered_candidate:
        score += 0.2
    if any(token in lowered_candidate for token in ["security", "compliance", "auditor"]) and not any(
        token in lowered_title for token in ["security", "compliance", "auditor"]
    ):
        score -= 0.35
    return max(0.0, min(score, 1.0))


def _scored_keyword_resolve(job_title: str) -> Dict[str, Any]:
    """
    Fallback resolver using O*NET keyword variants + candidate scoring.
    """
    best_match: Dict[str, Any] = {}
    best_score = -1.0

    for keyword in _build_onet_search_variants(job_title):
        try:
            candidates = _search_onet_keyword(keyword, limit=10)
            for candidate in candidates:
                score = _score_keyword_candidate(
                    job_title=job_title,
                    candidate_title=candidate.get("title", ""),
                    query=keyword,
                )
                if score > best_score:
                    best_score = score
                    best_match = {
                        "code": candidate.get("code", ""),
                        "title": candidate.get("title", ""),
                        "query": keyword,
                        "confidence": round(score, 3),
                        "source": "keyword_scored",
                    }
        except Exception as search_err:
            logger.warning(
                "[NEW_PIPELINE] O*NET variant search failed for '%s' via '%s': %s",
                job_title,
                keyword,
                search_err,
            )

    if best_match and best_score >= 0.25:
        return best_match
    return {}


async def _llm_resolve_onet(job_title: str, llm_client: Optional[LLMClientAdapter]) -> Dict[str, Any]:
    """
    Use LLM semantic classification for title -> O*NET mapping.
    """
    if not llm_client or not job_title:
        return {}

    prompt = f"""
You are an O*NET occupation classifier. Given a job title, return the single best matching O*NET occupation code and title.

Rules:
- Return ONLY valid O*NET codes (format: XX-XXXX.XX)
- Pick the closest semantic match, not just keyword overlap
- If the title is a specialization, map to the parent occupation
- Never map to security/compliance roles unless the title explicitly says so

Common mappings for reference:
- AI Engineer / ML Engineer -> 15-2051.00 Data Scientists
- Data Engineer / ETL Engineer -> 15-1243.00 Database Architects
- Full Stack Developer -> 15-1252.00 Software Developers
- DevOps / SRE -> 15-1244.00 Network and Computer Systems Administrators
- Prompt Engineer -> 15-1299.08 Computer Systems Engineers
- AI Product Manager -> 11-3021.00 Computer and Information Systems Managers

Job title: "{job_title}"

Respond in JSON only:
{{"code":"XX-XXXX.XX","title":"O*NET Title","confidence":0.0,"reasoning":"..."}}
""".strip()
    try:
        raw = await llm_client.call_llm_async(
            system_prompt="You map modern job titles to valid O*NET occupations.",
            user_prompt=prompt,
            model=OR_SLUG_JOB_TITLE_EXTRACTOR,
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=220,
            fallback_models=["openai/gpt-4o-mini"],
        )
        response_text = (raw or "").strip()
        if not response_text:
            return {}
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not json_match:
                return {}
            parsed = json.loads(json_match.group(0))
        if not isinstance(parsed, dict):
            return {}
        confidence_raw = parsed.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "code": str(parsed.get("code", "")).strip(),
            "title": str(parsed.get("title", "")).strip(),
            "confidence": max(0.0, min(confidence, 1.0)),
            "reasoning": str(parsed.get("reasoning", "")).strip(),
            "source": "llm",
            "query": job_title,
        }
    except Exception as llm_error:
        logger.warning("[NEW_PIPELINE] LLM O*NET resolution failed for '%s': %s", job_title, llm_error)
        return {}


_ONET_RESOLUTION_CACHE: Dict[str, Dict[str, Any]] = {}


async def _resolve_onet_code(job_title: str, llm_client: Optional[LLMClientAdapter]) -> Dict[str, Any]:
    """
    Resolution chain:
    1) LLM semantic mapping + code validation
    2) Scored keyword variant search
    """
    normalized_title = (job_title or "").strip()
    if not normalized_title:
        return {}
    cache_key = normalized_title.lower()
    if cache_key in _ONET_RESOLUTION_CACHE:
        return _ONET_RESOLUTION_CACHE[cache_key]

    confidence_floor_raw = os.getenv("ONET_LLM_RESOLUTION_CONFIDENCE_MIN", "0.70")
    try:
        confidence_floor = float(confidence_floor_raw)
    except ValueError:
        confidence_floor = 0.70
    use_llm_first = os.getenv("ONET_LLM_RESOLVER_ENABLED", "true").lower() == "true"

    if use_llm_first:
        llm_result = await _llm_resolve_onet(normalized_title, llm_client)
        llm_confidence = float(llm_result.get("confidence", 0.0) or 0.0)
        llm_code = llm_result.get("code", "")
        if llm_code and llm_confidence >= confidence_floor:
            if _validate_onet_code(llm_code):
                logger.info(
                    "[NEW_PIPELINE] O*NET resolved via LLM: '%s' -> %s (%s) confidence=%.2f",
                    normalized_title,
                    llm_code,
                    llm_result.get("title", ""),
                    llm_confidence,
                )
                _ONET_RESOLUTION_CACHE[cache_key] = llm_result
                return llm_result
            logger.warning(
                "[NEW_PIPELINE] LLM suggested invalid/missing O*NET code '%s' for '%s'",
                llm_code,
                normalized_title,
            )
        elif llm_result:
            logger.info(
                "[NEW_PIPELINE] LLM confidence too low for '%s': %.2f < %.2f",
                normalized_title,
                llm_confidence,
                confidence_floor,
            )

    keyword_result = _scored_keyword_resolve(normalized_title)
    if keyword_result:
        logger.info(
            "[NEW_PIPELINE] O*NET resolved via scored keyword: '%s' -> %s (%s) score=%.2f",
            normalized_title,
            keyword_result.get("code", ""),
            keyword_result.get("title", ""),
            float(keyword_result.get("confidence", 0.0) or 0.0),
        )
        _ONET_RESOLUTION_CACHE[cache_key] = keyword_result
        return keyword_result

    _ONET_RESOLUTION_CACHE[cache_key] = {}
    return {}


FALLBACK_SALARY_BY_DOMAIN = {
    "data":        "$110,000 - $160,000",
    "ai_ml":       "$130,000 - $185,000",
    "engineering": "$105,000 - $155,000",
    "devops":      "$100,000 - $145,000",
    "analyst":     "$80,000 - $120,000",
    "default":     "$90,000 - $140,000",
}


def _get_fallback_salary(job_title: str, inferred_domain: str = "") -> str:
    """
    Domain-aware fallback salary range when market data is unavailable.
    Check most specific patterns first (data engineer before generic AI).
    """
    title_lower = (job_title or "").lower()
    
    # Check most specific patterns first
    if any(t in title_lower for t in ["data engineer", "etl", "pipeline"]):
        return FALLBACK_SALARY_BY_DOMAIN["data"]
    if any(t in title_lower for t in ["devops", "sre", "platform"]):
        return FALLBACK_SALARY_BY_DOMAIN["devops"]
    
    # Then check broader patterns
    if any(t in title_lower for t in ["ai", "ml", "machine learning", "llm"]):
        return FALLBACK_SALARY_BY_DOMAIN["ai_ml"]
    if any(t in title_lower for t in ["engineer", "developer"]):
        return FALLBACK_SALARY_BY_DOMAIN["engineering"]
    if any(t in title_lower for t in ["analyst"]):
        return FALLBACK_SALARY_BY_DOMAIN["analyst"]
    return FALLBACK_SALARY_BY_DOMAIN["default"]


def _get_domain_fallback_occupation(job_title: str, inferred_domain: str = "") -> Dict[str, str]:
    """
    Domain-aware fallback O*NET mapping when keyword search fails.
    """
    fallback_by_domain = {
        "engineering": {"code": "15-1252.00", "title": "Software Developers"},
        "data": {"code": "15-1243.00", "title": "Database Architects"},
        "ai_ml": {"code": "15-2051.00", "title": "Data Scientists"},
        "analyst": {"code": "15-2051.00", "title": "Data Scientists"},
        "devops": {"code": "15-1244.00", "title": "Network and Computer Systems Administrators"},
        "prompt_engineering": {"code": "15-1299.08", "title": "Computer Systems Engineers"},
        "product_management": {"code": "11-3021.00", "title": "Computer and Information Systems Managers"},
        "default": {"code": "15-1252.00", "title": "Software Developers"},
    }

    title_lower = (job_title or "").lower()
    domain_lower = (inferred_domain or "").lower()

    if any(token in title_lower for token in ["data engineer", "etl", "pipeline", "database"]):
        return fallback_by_domain["data"]
    if any(token in title_lower for token in ["devops", "sre", "site reliability"]):
        return fallback_by_domain["devops"]
    if any(token in title_lower for token in ["prompt engineer", "prompt", "llm engineer"]):
        return fallback_by_domain["prompt_engineering"]
    if any(token in title_lower for token in ["product manager", "ai product"]):
        return fallback_by_domain["product_management"]
    if any(token in title_lower for token in ["ai", "ml", "machine learning", "model"]):
        return fallback_by_domain["ai_ml"]
    if any(token in title_lower for token in ["engineer", "developer", "architect"]):
        return fallback_by_domain["engineering"]
    if any(token in title_lower for token in ["analyst", "analytics"]):
        return fallback_by_domain["analyst"]

    if "data" in domain_lower:
        return fallback_by_domain["data"]
    if "analyst" in domain_lower:
        return fallback_by_domain["analyst"]
    if "tech" in domain_lower or "engineer" in domain_lower:
        return fallback_by_domain["engineering"]

    return fallback_by_domain["default"]


async def run_new_pipeline_async(
    resume_text: str,
    job_ad: str,
    extracted_skills_json: Optional[Dict] = None,
    domain_insights_json: Optional[Dict] = None,
    openrouter_api_keys: Optional[List[str]] = None,
    creativity_mode: Optional[str] = None,
    location: Optional[str] = None,
    job_title: Optional[str] = None,
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
        job_title: User-provided job title for O*NET market intel (takes priority over LLM extraction)
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with final_written_section_markdown and metadata
    """
    logger.info("[NEW_PIPELINE] Starting 3-stage orchestrator")
    if location:
        logger.info(f"[NEW_PIPELINE] Location provided for market intel: {location}")
    if job_title:
        logger.info(f"[NEW_PIPELINE] User-provided job title for O*NET: {job_title}")
    
    try:
        # Parse experiences from resume text
        experiences = parse_experiences(resume_text)
        logger.info(f"[NEW_PIPELINE] Parsed {len(experiences)} experiences from resume")
        
        # Initialize LLM client
        if not openrouter_api_keys:
            openrouter_api_keys = OPENROUTER_API_KEYS
        
        if not openrouter_api_keys:
            raise ValueError("No OpenRouter API keys available")
        
        # Pass all keys to adapter for rotation/fallback on rate limits
        llm_client = LLMClientAdapter(api_key=openrouter_api_keys[0], fallback_keys=openrouter_api_keys[1:] if len(openrouter_api_keys) > 1 else [])
        
        # Use user-provided job_title if available, otherwise extract via LLM
        extracted_job_title = job_title  # Start with user-provided title
        if extracted_job_title:
            logger.info(f"[NEW_PIPELINE] Using user-provided job title: {extracted_job_title}")
        else:
            # Extract Job Title (Gemini Flash Lite) - High Priority for Frontend Display
            logger.info("[NEW_PIPELINE] Extracting job title via google/gemini-2.5-flash-lite (backup: openai/gpt-4o-mini)...")
            try:
                title_sys_prompt = "You are a precise job data extractor. Extract the official Job Title from the provided text. Return JSON: {\"job_title\": \"...\"}"
                title_user_prompt = f"Job Description Segment:\n{job_ad[:400]}\n\nExtract the Job Title:"
                
                title_response = await llm_client.call_llm_async(
                    system_prompt=title_sys_prompt,
                    user_prompt=title_user_prompt,
                    model=OR_SLUG_JOB_TITLE_EXTRACTOR,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    max_tokens=50,
                    fallback_models=["openai/gpt-4o-mini"]
                )
                title_response_clean = (title_response or "").strip()
                title_data = {}
                if title_response_clean:
                    # Extract JSON from markdown if needed
                    # Try ```json {...}``` pattern
                    json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', title_response_clean, re.DOTALL | re.IGNORECASE)
                    if json_block_match:
                        title_response_clean = json_block_match.group(1)
                    else:
                        # Try ```{...}``` pattern
                        code_block_match = re.search(r'```\s*(\{.*?\})\s*```', title_response_clean, re.DOTALL | re.IGNORECASE)
                        if code_block_match:
                            title_response_clean = code_block_match.group(1)
                    
                    try:
                        title_data = json.loads(title_response_clean)
                    except json.JSONDecodeError:
                        logger.warning(
                            "[NEW_PIPELINE] Job title response was not valid JSON. Falling back to text parsing."
                        )
                        json_match = re.search(r"\{.*\}", title_response_clean, re.DOTALL)
                        if json_match:
                            try:
                                title_data = json.loads(json_match.group(0))
                            except json.JSONDecodeError:
                                title_data = {}
                        if not title_data:
                            title_data = {"job_title": title_response_clean.strip().strip('"')}
                else:
                    logger.warning("[NEW_PIPELINE] Job title extraction returned an empty response")

                extracted_job_title = title_data.get("job_title")
                if extracted_job_title:
                    extracted_job_title = extracted_job_title.strip()
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
            projects=kwargs.get("projects", []),  # Pass projects for students/career changers
            education=kwargs.get("education", []),  # Pass education
            certifications=kwargs.get("certifications", []),  # Pass certifications
            location=kwargs.get("location", ""),  # Pass location
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
            
            # Handle empty, whitespace-only, or empty JSON responses
            response_clean = (response or "").strip()
            
            # Extract JSON from markdown if needed
            if response_clean:
                # Try ```json {...}``` pattern
                json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', response_clean, re.DOTALL | re.IGNORECASE)
                if json_block_match:
                    response_clean = json_block_match.group(1)
                else:
                    # Try ```{...}``` pattern
                    code_block_match = re.search(r'```\s*(\{.*?\})\s*```', response_clean, re.DOTALL | re.IGNORECASE)
                    if code_block_match:
                        response_clean = code_block_match.group(1)
            
            if not response_clean or response_clean == "{}" or response_clean.startswith('{"error"'):
                logger.warning(f"[NEW_PIPELINE] ATS scoring returned empty/error response: {response_clean[:100]}")
                critique_score = 0.75  # Default to reasonable score
            else:
                try:
                    score_data = json.loads(response_clean)
                    critique_score = score_data.get("score", 0.75)
                    # Validate score is in reasonable range
                    if not isinstance(critique_score, (int, float)) or critique_score < 0 or critique_score > 1:
                        logger.warning(f"[NEW_PIPELINE] Invalid ATS score {critique_score}, using default")
                        critique_score = 0.75
                except json.JSONDecodeError as e:
                    logger.warning(f"[NEW_PIPELINE] ATS scoring JSON parse failed: {e}, using default")
                    critique_score = 0.75
            logger.info(f"[NEW_PIPELINE] ATS score calculated: {critique_score}")
            
        except Exception as e:
            logger.warning(f"[NEW_PIPELINE] ATS scoring failed, using default: {e}")
            critique_score = 0.75  # Default score on error (changed from 0.85)
        
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
            
            # FIX: Overwrite "Unknown" or empty values from upstream with intelligent defaults
            if domain_insights.get("market_demand") in [None, "", "Unknown", "unknown"]:
                domain_insights["market_demand"] = "High"
            if domain_insights.get("salary_range") in [None, "", "Unknown", "unknown"]:
                domain_insights["salary_range"] = "$80,000 - $150,000"
            if not domain_insights.get("career_path"):
                domain_insights["career_path"] = ["Entry Level", "Mid Level", "Senior", "Lead"]
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
        if not domain_insights.get("top_skills"):
            domain_insights["top_skills"] = domain_vocab[:5] if domain_vocab else aggregate_skills[:5] if aggregate_skills else []
        
        if not domain_insights.get("certifications"):
            domain_insights["certifications"] = []
            
        if not domain_insights.get("career_path"):
            domain_insights["career_path"] = ["Entry Level", "Mid Level", "Senior", "Lead"]
        
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
        def _is_missing_or_weak(value: Any) -> bool:
            if value is None:
                return True
            text = str(value).strip()
            if not text:
                return True
            weak_values = {"unknown", "market saturation", "$47,947+"}
            return text.lower() in weak_values

        job_title_for_market = extracted_job_title
        onet_code = None
        if not job_title_for_market and domain_insights_json and isinstance(domain_insights_json, dict):
            onet_info = domain_insights_json.get("onet")
            if isinstance(onet_info, dict):
                job_title_for_market = onet_info.get("title") or job_title_for_market
                onet_code = onet_info.get("code") or onet_code

        if not job_title_for_market and experiences:
            first_exp = experiences[0] if isinstance(experiences[0], dict) else {}
            job_title_for_market = (
                first_exp.get("title_line", "").split("|")[0].strip()
                or first_exp.get("title", "").strip()
            ) or job_title_for_market

        if location and (job_title_for_market or onet_code):
            try:
                from career_progression_enricher import CareerProgressionEnricher
                from city_geo_mapper import get_geo_id
                import os
                
                safe_market_fallback = (
                    os.getenv("IMAGINATOR_SAFE_MARKET_FALLBACK_ENABLED", "true").lower() == "true"
                )
                
                logger.info(
                    f"[NEW_PIPELINE] Enriching with market intel for {job_title_for_market or 'O*NET role'} in {location}"
                )
                logger.info(
                    f"[NEW_PIPELINE] Safe market fallback enabled: {safe_market_fallback}"
                )
                
                # Get geo ID for location
                geo_id = get_geo_id(location)
                if not geo_id:
                    logger.warning(f"[NEW_PIPELINE] No geo ID found for location: {location}, using national data")
                
                # Search O*NET for job title to get O*NET code
                fallback_used = False
                fallback_reason = ""
                desired_job_title = job_title_for_market
                
                if not onet_code and job_title_for_market:
                    resolved_match = await _resolve_onet_code(job_title_for_market, llm_client)
                    if resolved_match.get("code"):
                        onet_code = resolved_match["code"]
                        domain_insights["onet"] = {
                            "title": resolved_match.get("title") or job_title_for_market,
                            "code": onet_code,
                            "status": "resolved",
                            "resolution_source": resolved_match.get("source", "unknown"),
                            "resolution_confidence": resolved_match.get("confidence"),
                            "query_used": resolved_match.get("query"),
                        }
                    else:
                        fallback_used = True
                        fallback_reason = (
                            f"O*NET resolution failed for '{job_title_for_market}'"
                        )
                elif not onet_code:
                    fallback_used = True
                    fallback_reason = "No job title provided"

                # If we have O*NET code, enrich with market data
                if onet_code:
                    enricher = CareerProgressionEnricher()
                    career_data = enricher.get_full_career_insights(
                        job_title=job_title_for_market or "",
                        onet_code=onet_code,
                        location=location,
                        city_geo_id=geo_id
                    )
                    
                    # Get O*NET summary for Bright Outlook check
                onet_summary = None
                if onet_code:
                    try:
                        onet_summary = enricher._get_onet_summary(onet_code)
                    except Exception as e:
                        logger.warning(f"[NEW_PIPELINE] Failed to get O*NET summary for {onet_code}: {e}")
                        onet_summary = None
                
                # Calculate market intel
                try:
                    market_intel = enricher.calculate_market_intel(
                        career_data.get("workforce", {}),
                        onet_summary,
                        job_title=job_title_for_market or "",
                        location=location
                    )
                except Exception as e:
                    logger.warning(f"[NEW_PIPELINE] Market intel calculation failed: {e}")
                    # Fallback to basic market data
                    market_intel = {
                        "status": "fallback",
                        "demand_label": "Stable",
                        "average_wage": 95000,
                        "pct25": 75000,
                        "pct75": 120000,
                        "wage_growth_pct": 2.5
                    }
                    # Add market intel to domain insights
                    domain_insights["market_intel"] = market_intel
                    
                    # Map market intel to top-level fields for frontend consumption
                    # Preserve high-quality upstream values; only fill missing or weak placeholders.
                    if market_intel.get("demand_label") and (
                        not safe_market_fallback
                        or _is_missing_or_weak(domain_insights.get("market_demand"))
                    ):
                        domain_insights["market_demand"] = market_intel["demand_label"]
                    
                    if market_intel.get("average_wage") and (
                        not safe_market_fallback
                        or _is_missing_or_weak(domain_insights.get("salary_range"))
                    ):
                        # Prefer COS range (pct25–pct75) when available
                        pct25 = market_intel.get("pct25")
                        pct75 = market_intel.get("pct75")
                        if isinstance(pct25, (int, float)) and isinstance(pct75, (int, float)):
                            domain_insights["salary_range"] = f"${int(pct25):,} - ${int(pct75):,}"
                        else:
                            wage = market_intel["average_wage"]
                            if isinstance(wage, (int, float)):
                                domain_insights["salary_range"] = f"${int(wage):,}+"
                            else:
                                domain_insights["salary_range"] = str(wage)
                            
                    logger.info(f"[NEW_PIPELINE] Added market intel: {market_intel.get('status')}")
                else:
                    logger.warning(f"[NEW_PIPELINE] Fallback to resume-inferred: {fallback_reason}")
                    # Resume fallback
                    inferred_domain = domain_insights.get("domain", "data")
                    fallback_role = _get_domain_fallback_occupation(
                        desired_job_title or extracted_job_title or "",
                        inferred_domain=inferred_domain,
                    )
                    inferred_code = fallback_role["code"]
                    domain_insights["onet"] = {
                        "title": fallback_role["title"],
                        "code": inferred_code,
                        "fallback_used": True,
                        "fallback_reason": fallback_reason,
                        "desired_job_title": desired_job_title or extracted_job_title or "",
                        "status": "resume_inferred"
                    }
                    domain_insights["fallback_used"] = True
                    domain_insights["fallback_reason"] = fallback_reason
                    if safe_market_fallback:
                        if _is_missing_or_weak(domain_insights.get("salary_range")):
                            domain_insights["salary_range"] = _get_fallback_salary(
                                desired_job_title or extracted_job_title or "", inferred_domain
                            )
                        if _is_missing_or_weak(domain_insights.get("market_demand")):
                            domain_insights["market_demand"] = "High"  # Tech roles in WA are not "Market Saturation"
                    else:
                        domain_insights["salary_range"] = _get_fallback_salary(
                            desired_job_title or extracted_job_title or "", inferred_domain
                        )
                        domain_insights["market_demand"] = "High"
                # ...rest unchanged...
            except Exception as e:
                logger.error(f"[NEW_PIPELINE] Failed to enrich domain insights with market intel: {e}", exc_info=True)
        elif location:
            logger.warning(f"[NEW_PIPELINE] Location provided ({location}) but no job title extracted, skipping market intel")
        
        # Build gap_analysis from researcher data - TAILORED TO SPECIFIC JOB
        gap_analysis = ""
        gap_parts = []
        
        # Extract job title for context
        job_title_context = extracted_job_title or "this role"
        
        # Build job-specific narrative
        if implied_skills_normalized:
            # Map implied skills to job requirements
            skills_narrative = f"Based on your experience, you likely possess these transferable skills for {job_title_context}: {', '.join(str(item) for item in implied_skills_normalized[:3] if item)}. "
            skills_narrative += "These align with the role's requirements and should be highlighted in your application."
            gap_parts.append(skills_narrative)
        
        if implied_metrics_normalized:
            # Map metrics to job responsibilities
            metrics_narrative = f"For {job_title_context}, target these quantifiable achievements: {', '.join(str(item) for item in implied_metrics_normalized[:3] if item)}. "
            metrics_narrative += "These benchmarks demonstrate impact aligned with the position's success criteria."
            gap_parts.append(metrics_narrative)
        
        if insider_tips:
            # Add insider tips with job context
            gap_parts.append(f"Key insight for {job_title_context}: {insider_tips}")
        
        if gap_parts:
            gap_analysis = " ".join(gap_parts)
        
        # Build structured gap analysis with job-specific context
        gap_analysis_payload = {
            "summary": gap_analysis or f"Analysis for {job_title_context}: Focus on demonstrating relevant experience and quantifiable achievements.",
            "critical_gaps": implied_skills_normalized[:5],  # Limit to top 5
            "benchmarks": implied_metrics_normalized[:3],  # Limit to top 3
            "insider_tips": insider_tips or f"Research {job_title_context} requirements and tailor your application accordingly.",
            "job_title": job_title_context,
            "alignment_score": critique_score  # Include ATS score for context
        }
        gap_analysis_json = json.dumps(gap_analysis_payload)

        rewritten_text = final_output.get("final_written_section", "") or ""
        rewritten_lower = rewritten_text.lower()
        source_education = kwargs.get("education", []) if isinstance(kwargs.get("education", []), list) else []
        source_projects = kwargs.get("projects", []) if isinstance(kwargs.get("projects", []), list) else []
        source_certifications = kwargs.get("certifications", []) if isinstance(kwargs.get("certifications", []), list) else []

        section_completeness = {
            "has_experience": bool(experiences) and ("experience" in rewritten_lower),
            "has_education": bool(source_education) and ("education" in rewritten_lower or "b.s." in rewritten_lower or "degree" in rewritten_lower),
            "has_projects": bool(source_projects) and ("project" in rewritten_lower),
            "has_certifications": bool(source_certifications) and ("certif" in rewritten_lower or "comptia" in rewritten_lower),
            "missing_sections": [],
        }
        section_completeness["missing_sections"] = [
            section.replace("has_", "")
            for section, present in section_completeness.items()
            if section.startswith("has_") and not present
        ]
        if section_completeness["missing_sections"]:
            logger.warning(
                "[COMPLETENESS] Missing sections in rewrite: %s",
                section_completeness["missing_sections"],
            )

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
            "sectionCompleteness": section_completeness,
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
                    
                    # Validate the onet_code before using it
                    if not _validate_onet_code(target_onet):
                        logger.warning(f"[NEW_PIPELINE] Invalid O*NET code {target_onet}, skipping pivot analysis")
                        pivot_data = None
                    else:
                        logger.info(f"[NEW_PIPELINE] Generating Pivot Analysis for Alchemy using code: {target_onet}")
                        try:
                            pivot_data = enricher.generate_career_pivot_analysis(
                                current_onet_code=target_onet,
                                current_job_title=extracted_job_title or "Software Engineer",
                                resume_skills=[s if isinstance(s, str) else s.get("skill", "") for s in aggregate_skills],
                                resume_tech=aggregate_skills, # Use same list for now
                                location=current_location,
                                max_pivots=3
                            )
                        except Exception as pivot_err:
                            logger.warning(f"[NEW_PIPELINE] Pivot analysis failed: {pivot_err}")
                            pivot_data = None
                except Exception as pivot_err:
                    logger.warning(f"[NEW_PIPELINE] Alchemy pivot enrichment failed: {pivot_err}")
                    pivot_data = None
                
                # Generate career alchemy
                try:
                    career_alchemy_data = generate_career_alchemy(
                        characteristics=characteristics,
                        location=current_location,
                        pivot_data=pivot_data,
                        salary_data=salary_data
                    )
                    logger.info("[NEW_PIPELINE] Career Alchemy generated successfully")
                except Exception as e:
                    logger.warning(f"[NEW_PIPELINE] Career Alchemy generation failed: {e}")
                    career_alchemy_data = None
                
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