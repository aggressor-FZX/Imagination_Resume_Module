#!/usr/bin/env python3
"""
Imaginator Microservices Module
Handles all external service calls (Document Reader, Hermes, FastSVM, Job Search).
Includes HTTP client logic, error handling, and logging.
"""
import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional

import aiohttp

from .config import settings

# Shared HTTP session for connection pooling
_SHARED_HTTP_SESSION: Optional[aiohttp.ClientSession] = None

# Pool metrics for monitoring
_POOL_METRICS: Dict[str, Any] = {
    "http": {
        "requests_total": 0,
        "errors_total": 0,
        "timeouts_total": 0,
        "status_counts": {},
        "latency_ms": {"p50": None, "p95": None, "last": None},
    }
}


def _record_latency_ms(ms: float) -> None:
    """Record latency for monitoring."""
    try:
        hist = _POOL_METRICS["http"].setdefault("_lat_hist", [])
        hist.append(ms)
        _POOL_METRICS["http"]["latency_ms"]["last"] = int(ms)
        if len(hist) >= 5:
            arr = sorted(hist)
            n = len(arr)
            _POOL_METRICS["http"]["latency_ms"]["p50"] = int(arr[n//2])
            _POOL_METRICS["http"]["latency_ms"]["p95"] = int(arr[max(0, int(n*0.95) - 1)])
            if n > 200:
                del hist[:n-100]
    except Exception:
        pass


def configure_shared_http_session(session: Optional[aiohttp.ClientSession]) -> None:
    """Configure shared HTTP session for connection pooling."""
    global _SHARED_HTTP_SESSION
    _SHARED_HTTP_SESSION = session


def _truncate_text(value: str, max_chars: int) -> str:
    """Truncate text for logging."""
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"...(truncated {len(value) - max_chars} chars)"


def _sha256_hex(value: str) -> str:
    """Generate SHA256 hash for redaction."""
    import hashlib
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def redact_for_logging(data: Any) -> Any:
    """Redact sensitive data for logging."""
    sensitive_keys = {
        "resume_text", "text", "raw_text", "processed_text", "job_ad",
        "raw_json_resume", "generated_text", "prompt"
    }
    
    def _walk(value: Any, key: Optional[str] = None) -> Any:
        if isinstance(value, dict):
            return {k: _walk(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [_walk(v, key) for v in value]
        if isinstance(value, str):
            if key in sensitive_keys and not settings.LOG_INCLUDE_RAW_TEXT:
                return f"<redacted len={len(value)} sha256={_sha256_hex(value)[:12]}>"
            return _truncate_text(value, settings.LOG_MAX_TEXT_CHARS)
        return value
    
    return _walk(data)


async def _post_json(
    url: str,
    payload: Dict[str, Any],
    bearer_token: Optional[str] = None,
    timeout: int = 30,
    context: Optional[str] = None,
) -> Dict[str, Any]:
    """Make async POST request with JSON payload."""
    import logging
    logger = logging.getLogger(__name__)
    
    headers = {"Content-Type": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    
    start = time.time()
    request_id = uuid.uuid4().hex[:10]
    label = context or "external_call"
    
    if settings.VERBOSE_MICROSERVICE_LOGS:
        safe_headers = dict(headers)
        if "Authorization" in safe_headers:
            safe_headers["Authorization"] = "<redacted>"
        logger.info(
            json.dumps({
                "event": "microservice.request",
                "id": request_id,
                "label": label,
                "url": url,
                "timeout_s": timeout,
                "headers": safe_headers,
                "json": redact_for_logging(payload),
            }, default=str)
        )
    
    try:
        _POOL_METRICS["http"]["requests_total"] += 1
        
        if _SHARED_HTTP_SESSION:
            async with _SHARED_HTTP_SESSION.post(url, json=payload, headers=headers) as resp:
                text = await resp.text()
                _POOL_METRICS["http"]["status_counts"][str(resp.status)] = (
                    _POOL_METRICS["http"]["status_counts"].get(str(resp.status), 0) + 1
                )
                _record_latency_ms((time.time() - start) * 1000)
                
                if settings.VERBOSE_MICROSERVICE_LOGS:
                    body_preview: Any = None
                    try:
                        body_preview = redact_for_logging(json.loads(text))
                    except Exception:
                        body_preview = _truncate_text(text, settings.LOG_MAX_TEXT_CHARS)
                    logger.info(
                        json.dumps({
                            "event": "microservice.response",
                            "id": request_id,
                            "label": label,
                            "url": url,
                            "status": resp.status,
                            "elapsed_ms": int((time.time() - start) * 1000),
                            "body": body_preview,
                        }, default=str)
                    )
                
                try:
                    return json.loads(text)
                except Exception:
                    return {"status": resp.status, "body": text}
        else:
            t = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=t) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    text = await resp.text()
                    _POOL_METRICS["http"]["status_counts"][str(resp.status)] = (
                        _POOL_METRICS["http"]["status_counts"].get(str(resp.status), 0) + 1
                    )
                    _record_latency_ms((time.time() - start) * 1000)
                    
                    if settings.VERBOSE_MICROSERVICE_LOGS:
                        body_preview: Any = None
                        try:
                            body_preview = redact_for_logging(json.loads(text))
                        except Exception:
                            body_preview = _truncate_text(text, settings.LOG_MAX_TEXT_CHARS)
                        logger.info(
                            json.dumps({
                                "event": "microservice.response",
                                "id": request_id,
                                "label": label,
                                "url": url,
                                "status": resp.status,
                                "elapsed_ms": int((time.time() - start) * 1000),
                                "body": body_preview,
                            }, default=str)
                        )
                    
                    try:
                        return json.loads(text)
                    except Exception:
                        return {"status": resp.status, "body": text}
    except asyncio.TimeoutError:
        _POOL_METRICS["http"]["timeouts_total"] += 1
        _record_latency_ms((time.time() - start) * 1000)
        logger.error(
            json.dumps({
                "event": "microservice.timeout",
                "id": request_id,
                "label": label,
                "url": url,
                "timeout_s": timeout,
                "elapsed_ms": int((time.time() - start) * 1000),
            })
        )
        raise
    except Exception:
        _POOL_METRICS["http"]["errors_total"] += 1
        _record_latency_ms((time.time() - start) * 1000)
        logger.exception(
            json.dumps({
                "event": "microservice.error",
                "id": request_id,
                "label": label,
                "url": url,
                "elapsed_ms": int((time.time() - start) * 1000),
            })
        )
        raise


async def call_loader_process_text_only(text: str) -> Dict[str, Any]:
    """Call Document Reader service to process text only."""
    import logging
    logger = logging.getLogger(__name__)
    
    if not settings.ENABLE_LOADER or not settings.LOADER_BASE_URL:
        logger.info(f"[LOADER] Service DISABLED (ENABLE_LOADER={settings.ENABLE_LOADER}, has_base_url={bool(settings.LOADER_BASE_URL)}) - returning text as-is")
        if settings.VERBOSE_PIPELINE_LOGS:
            logger.info(
                json.dumps({
                    "event": "microservice.skipped",
                    "service": "LOADER",
                    "reason": {
                        "ENABLE_LOADER": settings.ENABLE_LOADER,
                        "has_base_url": bool(settings.LOADER_BASE_URL),
                    },
                }, default=str)
            )
        return {"processed_text": text}
    
    logger.info(f"[LOADER] Calling service at {settings.LOADER_BASE_URL}")
    url = f"{settings.LOADER_BASE_URL}/process-text-only"
    result = await _post_json(url, {"text": text}, bearer_token=settings.API_KEY, context="LOADER.process-text-only")
    logger.info(f"[LOADER] Returned keys: {list(result.keys()) if result else 'none'}")
    return result


async def call_fastsvm_process_resume(resume_text: str, extract_pdf: bool = False) -> Dict[str, Any]:
    """Call FastSVM service to process resume and extract skills/titles."""
    import logging
    logger = logging.getLogger(__name__)
    
    if not settings.ENABLE_FASTSVM or not settings.FASTSVM_BASE_URL:
        logger.warning(f"[FASTSVM] Service DISABLED (ENABLE_FASTSVM={settings.ENABLE_FASTSVM}, has_base_url={bool(settings.FASTSVM_BASE_URL)}) - returning empty skills/titles")
        if settings.VERBOSE_PIPELINE_LOGS:
            logger.info(
                json.dumps({
                    "event": "microservice.skipped",
                    "service": "FASTSVM",
                    "reason": {
                        "ENABLE_FASTSVM": settings.ENABLE_FASTSVM,
                        "has_base_url": bool(settings.FASTSVM_BASE_URL),
                    },
                }, default=str)
            )
        return {"skills": [], "titles": []}
    
    logger.info(f"[FASTSVM] Calling service at {settings.FASTSVM_BASE_URL}")
    url = f"{settings.FASTSVM_BASE_URL}/api/v1/process-resume"
    result = await _post_json(
        url,
        {"resume_text": resume_text, "extract_pdf": extract_pdf},
        bearer_token=settings.FASTSVM_AUTH_TOKEN,
        context="FASTSVM.process-resume",
    )
    skills_count = len(result.get('skills', [])) if result else 0
    logger.info(f"[FASTSVM] Returned {skills_count} skills")
    return result


async def extract_job_skills(job_ad: str) -> List[str]:
    """Extract high-confidence skills from job ad using FastSVM service."""
    import logging
    logger = logging.getLogger(__name__)
    
    if not job_ad or not job_ad.strip():
        logger.warning("[EXTRACT_JOB_SKILLS] Empty job ad provided")
        return []
    
    try:
        # Call FastSVM to extract skills from job ad
        fastsvm_output = await call_fastsvm_process_resume(job_ad, extract_pdf=False)
        logger.info(f"[EXTRACT_JOB_SKILLS] FastSVM returned {len(fastsvm_output.get('skills', []))} skills")
        
        # Process structured output
        structured_input = _structured_from_fastsvm(fastsvm_output)
        if not structured_input:
            logger.warning("[EXTRACT_JOB_SKILLS] No structured input from FastSVM")
            return []
        
        # Process skills with confidence filtering
        processed_skills = process_structured_skills(structured_input, confidence_threshold=0.7)
        high_confidence_skills = processed_skills.get('high_confidence', [])
        
        logger.info(f"[EXTRACT_JOB_SKILLS] Extracted {len(high_confidence_skills)} high-confidence skills")
        return high_confidence_skills
        
    except Exception as e:
        logger.error(f"[EXTRACT_JOB_SKILLS] Failed to extract job skills: {e}")
        return []


async def call_hermes_extract(raw_json_resume: Dict[str, Any]) -> Dict[str, Any]:
    """Call Hermes service to extract insights and skills from resume."""
    import logging
    logger = logging.getLogger(__name__)
    
    if not settings.ENABLE_HERMES or not settings.HERMES_BASE_URL:
        logger.warning(f"[HERMES] Service DISABLED (ENABLE_HERMES={settings.ENABLE_HERMES}, has_base_url={bool(settings.HERMES_BASE_URL)}) - returning empty insights/skills")
        if settings.VERBOSE_PIPELINE_LOGS:
            logger.info(
                json.dumps({
                    "event": "microservice.skipped",
                    "service": "HERMES",
                    "reason": {
                        "ENABLE_HERMES": settings.ENABLE_HERMES,
                        "has_base_url": bool(settings.HERMES_BASE_URL),
                    },
                }, default=str)
            )
        return {"insights": [], "skills": []}
    
    logger.info(f"[HERMES] Calling service at {settings.HERMES_BASE_URL}")
    url = f"{settings.HERMES_BASE_URL}/extract"
    result = await _post_json(url, raw_json_resume, bearer_token=settings.HERMES_AUTH_TOKEN, context="HERMES.extract")
    skills_count = len(result.get('skills', [])) if result else 0
    insights_count = len(result.get('insights', [])) if result else 0
    logger.info(f"[HERMES] Returned {skills_count} skills, {insights_count} insights")
    return result


async def call_job_search_api(query: Dict[str, Any]) -> Dict[str, Any]:
    """Call Job Search API to find matching jobs."""
    import logging
    logger = logging.getLogger(__name__)
    
    if not settings.ENABLE_JOB_SEARCH or not settings.JOB_SEARCH_BASE_URL:
        logger.info(
            f"[JOB_SEARCH] Service DISABLED (ENABLE_JOB_SEARCH={settings.ENABLE_JOB_SEARCH}, has_base_url={bool(settings.JOB_SEARCH_BASE_URL)})"
        )
        return {"status": "disabled"}
    
    url = f"{settings.JOB_SEARCH_BASE_URL}/api/v1/search"
    return await _post_json(url, query, bearer_token=settings.JOB_SEARCH_AUTH_TOKEN, context="JOB_SEARCH.search")


def _structured_from_fastsvm(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert FastSVM output to structured format."""
    if not isinstance(data, dict):
        return None
    
    items: List[Dict[str, Any]] = []
    
    if isinstance(data.get("skills"), list) and data.get("skills"):
        if all(isinstance(x, dict) for x in data["skills"]):
            for x in data["skills"]:
                name = x.get("skill") or x.get("name") or x.get("title")
                conf = x.get("confidence") or x.get("score") or 0
                cat = x.get("category") or "general"
                if name:
                    items.append({"skill": name, "confidence": conf, "category": cat})
        elif all(isinstance(x, str) for x in data["skills"]):
            conf_map = data.get("skill_confidences") or data.get("confidences") or {}
            for name in data["skills"]:
                conf = conf_map.get(name, 0)
                items.append({"skill": name, "confidence": conf, "category": "general"})
    
    if items:
        return {"skills": items}
    return None


def process_structured_skills(skills_data: Dict, confidence_threshold: float = 0.7, domain: Optional[str] = None) -> Dict:
    """
    Process structured skills data from repos with confidence filtering and domain awareness.
    
    Args:
        skills_data: Structured skills data from repos
        confidence_threshold: Minimum confidence score
        domain: Domain context for filtering (optional)
    
    Returns:
        Processed skills with filtering and prioritization
    """
    processed = {
        "high_confidence": [],
        "medium_confidence": [],
        "low_confidence": [],
        "skill_confidences": {},
        "categories": {},
        "inferred_skills": [],
        "filtered_count": 0,
        "total_count": 0
    }
    
    if "skills" not in skills_data:
        return processed
    
    for skill_info in skills_data["skills"]:
        if not isinstance(skill_info, dict):
            continue
        
        skill_name = skill_info.get("skill", skill_info.get("name", ""))
        confidence = skill_info.get("confidence", 0)
        category = skill_info.get("category", "general")
        
        if not skill_name:
            continue
        
        processed["total_count"] += 1
        processed["skill_confidences"][skill_name] = confidence
        
        # Categorize by confidence
        if confidence >= confidence_threshold:
            processed["high_confidence"].append(skill_name)
            processed["filtered_count"] += 1
        elif confidence >= 0.5:
            processed["medium_confidence"].append(skill_name)
        else:
            processed["low_confidence"].append(skill_name)
        
        # Group by category
        if category not in processed["categories"]:
            processed["categories"][category] = []
        processed["categories"][category].append(skill_name)
    
    # Sort skills by confidence (highest first)
    for category in processed["categories"]:
        processed["categories"][category].sort(
            key=lambda s: processed["skill_confidences"].get(s, 0),
            reverse=True
        )
    
    return processed


def get_pool_metrics() -> Dict[str, Any]:
    """Get current HTTP pool metrics."""
    return _POOL_METRICS.copy()