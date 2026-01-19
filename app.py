"""
FastAPI web service for Generative Resume Co-Writer
Based on Context7 research findings for FastAPI best practices
"""

import time
import json
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import aiohttp

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends, Security, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from config import settings

# Configure structured logging
from logging_setup import configure_logging, RequestIDMiddleware
# Configure global logger for this service
logger = configure_logging(service_name="imaginator")
from models import (
    AnalysisRequest, AnalysisResponse, HealthResponse,
    Context7DocsRequest, Context7DocsResponse, ErrorResponse,
    ProcessingStatus
)

# Import the existing flow functions (will be converted to async)
from imaginator_flow import (
    RUN_METRICS,
    validate_output_schema,
    configure_shared_http_session,
    redact_for_logging,
)

# Import new 3-stage pipeline integration
from imaginator_new_integration import run_new_pipeline_async


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events"""
    # Startup
    print(f"🚀 Starting {settings.environment} server on {settings.host}:{settings.port}")

    # Validate required API keys (OpenRouter only)
    if not settings.openrouter_api_key_1 and not settings.openrouter_api_key_2:
        raise RuntimeError("At least one OPENROUTER_API_KEY must be configured")

    connector = aiohttp.TCPConnector(
        limit=settings.max_concurrent_requests,
        force_close=False,
        enable_cleanup_closed=True,
        ttl_dns_cache=300,
    )
    timeout = aiohttp.ClientTimeout(total=settings.request_timeout)
    app.state.http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    configure_shared_http_session(app.state.http_session)

    # Redis cache removed per project decision

    yield

    # Shutdown
    print("🛑 Shutting down server")
    try:
        session = getattr(app.state, "http_session", None)
        if session:
            await session.close()
    finally:
        configure_shared_http_session(None)


# Create FastAPI application
app = FastAPI(
    title="Generative Resume Co-Writer",
    description="AI-powered resume analysis and career development recommendations",
    version="1.0.0",
    lifespan=lifespan
)
# Request ID middleware for tracing
app.add_middleware(RequestIDMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Security: API Key Authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    """Validate API key against configured IMAGINATOR_AUTH_TOKEN"""
    if not api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    # Validate against the configured auth token from environment
    if api_key != settings.IMAGINATOR_AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers and monitoring"""
    from datetime import datetime, timezone
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        environment=settings.environment,
        timestamp=datetime.now(timezone.utc).isoformat(),
        has_openrouter_key=bool(settings.openrouter_api_key_1 or settings.openrouter_api_key_2)
    )


@app.get("/")
async def root():
    return {"status": "ok", "service": "Generative Resume Co-Writer"}


@app.get("/keys/health")
async def keys_health():
    """Report provider key readiness and availability."""
    providers = {
        "openrouter": bool(settings.openrouter_api_key_1 or settings.openrouter_api_key_2),
        "openai": bool(getattr(settings, "openai_api_key", None)),
        "anthropic": bool(getattr(settings, "anthropic_api_key", None)),
        "google": bool(getattr(settings, "google_api_key", None)),
        "deepseek": bool(getattr(settings, "deepseek_api_key", None)),
    }
    ready = providers["openrouter"]  # minimal requirement
    return {
        "ready": ready,
        "providers": providers,
        "environment": settings.environment,
    }


@app.post("/analyze", response_model=AnalysisResponse, dependencies=[Depends(get_api_key)])
async def analyze_resume(
    payload: AnalysisRequest,
    request: Request,
    api_key: str = Depends(get_api_key),
):
    """
    Analyze a resume against a job description and provide career development recommendations.

    This endpoint processes the resume through three stages:
    1. Analysis: Extract skills and identify gaps
    2. Generation: Create improvement suggestions
    3. Criticism: Refine suggestions with adversarial review
    """
    start_time = time.time()

    # Parse incoming body and attach request_id
    body = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    request_id = getattr(request.state, "request_id", None)
    logger.info("analyze.request.incoming", extra={"request_id": request_id, "payload_size": len(json.dumps(body)) if body else 0})

    # Log incoming request and feature flags
    logger.info("[IMAGINATOR ENDPOINT] === Received analyze request ===")
    logger.info(f"[IMAGINATOR ENDPOINT] Resume length: {len(payload.resume_text)}")
    logger.info(f"[IMAGINATOR ENDPOINT] Job ad length: {len(payload.job_ad) if payload.job_ad else 0}")
    logger.info(f"[IMAGINATOR ENDPOINT] Confidence threshold: {payload.confidence_threshold}")
    logger.info(f"[IMAGINATOR ENDPOINT] Feature Flags:")
    logger.info(f"[IMAGINATOR ENDPOINT]   - ENABLE_LOADER: {settings.ENABLE_LOADER}")
    logger.info(f"[IMAGINATOR ENDPOINT]   - ENABLE_FASTSVM: {settings.ENABLE_FASTSVM}")
    logger.info(f"[IMAGINATOR ENDPOINT]   - ENABLE_HERMES: {settings.ENABLE_HERMES}")
    logger.info(f"[IMAGINATOR ENDPOINT]   - VERBOSE_PIPELINE_LOGS: {settings.VERBOSE_PIPELINE_LOGS}")
    logger.info(f"[IMAGINATOR ENDPOINT]   - VERBOSE_MICROSERVICE_LOGS: {settings.VERBOSE_MICROSERVICE_LOGS}")

    if settings.VERBOSE_PIPELINE_LOGS:
        logger.info(
            json.dumps(
                {"event": "proxy.request", "path": "/analyze", "payload": redact_for_logging(payload)},
                default=str,
            )
        )

    # Parse structured data from backend
    skills_data = None
    if payload.extracted_skills_json:
        try:
            skills_data = json.loads(payload.extracted_skills_json) if isinstance(payload.extracted_skills_json, str) else payload.extracted_skills_json
            skill_count = len(skills_data) if isinstance(skills_data, list) else len(skills_data.get('skills', []))
            logger.info(f"[IMAGINATOR ENDPOINT] Received extracted_skills_json with {skill_count} skills")
            logger.info(f"[IMAGINATOR ENDPOINT] Skills structure type: {type(skills_data).__name__}")
            if isinstance(skills_data, list) and len(skills_data) > 0:
                logger.info(f"[IMAGINATOR ENDPOINT] First skill sample: {skills_data[0]}")
                # Check if backend sent structured skills with confidence (COMPLIANCE)
                first_skill = skills_data[0]
                if isinstance(first_skill, dict) and 'confidence' in first_skill and 'source' in first_skill:
                    logger.info(f"[COMPLIANCE] structured-skills-v1: Backend sent STRUCTURED skills with confidence metadata")
                    logger.info(f"[COMPLIANCE] structured-skills-v1: Sample confidence: {first_skill.get('confidence')}, source: {first_skill.get('source')}")
                else:
                    logger.warning(f"[COMPLIANCE] structured-skills-v1: Backend sent RAW skills (missing confidence/source) - will trigger fallback")
        except Exception as e:
            logger.warning(f"[IMAGINATOR ENDPOINT] Could not parse extracted_skills_json: {e}")
            skills_data = None
    else:
        logger.info(f"[IMAGINATOR ENDPOINT] No extracted_skills_json provided (will use fallback)")

    insights_data = None
    if payload.domain_insights_json:
        try:
            insights_data = json.loads(payload.domain_insights_json) if isinstance(payload.domain_insights_json, str) else payload.domain_insights_json
            logger.info(f"[IMAGINATOR ENDPOINT] Received domain_insights_json with keys: {list(insights_data.keys()) if isinstance(insights_data, dict) else 'not a dict'}")
        except Exception as e:
            logger.warning(f"[IMAGINATOR ENDPOINT] Could not parse domain_insights_json: {e}")
            insights_data = None
    else:
        logger.info(f"[IMAGINATOR ENDPOINT] No domain_insights_json provided")

    # Authentication already validated via dependency; no additional checks

    try:
        if not payload.resume_text.strip():
            raise HTTPException(status_code=422, detail="resume_text cannot be empty")
        # Use server-configured OpenRouter keys only (BYOK removed)
        api_keys = [key for key in [settings.openrouter_api_key_1, settings.openrouter_api_key_2] if key]

        # Reset run metrics for this request
        RUN_METRICS.update({
            "calls": [],
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "failures": []
        })

        # Optional fields are validated by Pydantic; do not parse manually
        # extracted_skills = None
        # domain_insights = None

        # Step 1: Run NEW 3-Stage Pipeline (replaces old analysis)
        logger.info("run_new_pipeline.start", extra={"request_id": request_id})
        t0 = time.time()
        analysis_result = await run_new_pipeline_async(
            resume_text=payload.resume_text,
            job_ad=payload.job_ad,
            extracted_skills_json=skills_data,
            domain_insights_json=insights_data,
            openrouter_api_keys=api_keys
        )
        if settings.VERBOSE_PIPELINE_LOGS:
            logger.info(
                json.dumps(
                    {
                        "event": "pipeline.stage_result",
                        "path": "/analyze",
                        "stage": "new_3stage_pipeline",
                        "result": redact_for_logging(analysis_result),
                    },
                    default=str,
                )
            )
        analysis_duration_ms = int((time.time() - t0) * 1000)
        logger.info("run_new_pipeline.end", extra={"request_id": request_id, "duration_ms": analysis_duration_ms})
        RUN_METRICS.setdefault("durations_ms", {})["pipeline_ms"] = analysis_duration_ms
        # If mocked test returns final success payload (v2 shape), return it directly
        if isinstance(analysis_result, dict) and "status" in analysis_result and "analysis" in analysis_result:
            return JSONResponse(status_code=200, content=analysis_result)

        # CHECK FOR NEW PIPELINE RESULT (4-Stage)
        if isinstance(analysis_result, dict) and analysis_result.get("final_written_section"):
            logger.info("analyze.pipeline.fast_track", extra={"request_id": request_id, "msg_text": "4-Stage Pipeline completed in analysis step. Skipping legacy steps."})
            
            # Construct final output from analysis_result
            output = {
                **analysis_result,
                "run_metrics": RUN_METRICS.copy(),
                "processing_status": ProcessingStatus.COMPLETED,
                "processing_time_seconds": time.time() - start_time
            }

            # Fix nested domain_insights structure from upstream services (Hermes/FastSVM)
            if "domain_insights" in output and isinstance(output["domain_insights"], dict):
                di = output["domain_insights"]
                if "domain_insights" in di and isinstance(di["domain_insights"], dict):
                    nested_di = di.pop("domain_insights")
                    di.update(nested_di)
                    logger.info("[SCHEMA_FIX] Merged nested domain_insights")

             ## Validate output schema
            validate_output_schema(output)

            processing_time = time.time() - start_time
            logger.info("analyze.request.completed", extra={"request_id": request_id, "processing_time_seconds": processing_time, "run_metrics": RUN_METRICS.copy()})
            return AnalysisResponse(**output)

        # Step 2: Run Generation
        logger.info("run_generation.start", extra={"request_id": request_id})
        t0 = time.time()
        generation_result = await run_generation_async(
            analysis_json=analysis_result,
            job_ad=payload.job_ad,
            openrouter_api_keys=api_keys
        )
        if settings.VERBOSE_PIPELINE_LOGS:
            logger.info(
                json.dumps(
                    {
                        "event": "pipeline.stage_result",
                        "path": "/analyze",
                        "stage": "generation",
                        "result": redact_for_logging(generation_result),
                    },
                    default=str,
                )
            )
        generation_duration_ms = int((time.time() - t0) * 1000)
        logger.info("run_generation.end", extra={"request_id": request_id, "duration_ms": generation_duration_ms})
        RUN_METRICS.setdefault("durations_ms", {})["generation_ms"] = generation_duration_ms

        # Step 3: Run Criticism
        logger.info("run_criticism.start", extra={"request_id": request_id})
        t0 = time.time()
        criticism_result = await run_criticism_async(
            generated_text=generation_result,
            job_ad=payload.job_ad,
            openrouter_api_keys=api_keys
        )
        criticism_duration_ms = int((time.time() - t0) * 1000)
        logger.info("run_criticism.end", extra={"request_id": request_id, "duration_ms": criticism_duration_ms})
        RUN_METRICS.setdefault("durations_ms", {})["criticism_ms"] = criticism_duration_ms

        # Normalize criticism_result to dict
        if isinstance(criticism_result, str):
            try:
                from imaginator_flow import ensure_json_dict
                criticism_result = ensure_json_dict(criticism_result, "critique")
            except Exception:
                criticism_result = {"suggested_experiences": {"bridging_gaps": [], "metric_improvements": []}}

        # Step 4: Synthesis — integrate critique into final text
        logger.info("run_synthesis.start", extra={"request_id": request_id, "final_writer": getattr(settings, "FINAL_WRITER_PROVIDER", None)})
        t0 = time.time()
        synthesis_result = await run_synthesis_async(
            generated_text=generation_result,
            critique_json=criticism_result,
            job_ad=payload.job_ad,
            openrouter_api_keys=api_keys,
            analysis_result=analysis_result,  # FIX: Changed from analysis_for_provenance to analysis_result
            final_writer=getattr(settings, "FINAL_WRITER_PROVIDER", None)  # Use configured provider (safe getattr)
        )
        synthesis_duration_ms = int((time.time() - t0) * 1000)
        logger.info("run_synthesis.end", extra={"request_id": request_id, "duration_ms": synthesis_duration_ms})
        RUN_METRICS.setdefault("durations_ms", {})["synthesis_ms"] = synthesis_duration_ms

        # Extract structured fields from synthesis result
        if isinstance(synthesis_result, dict):
            final_written_section = synthesis_result.get("final_written_section", synthesis_result)
            final_markdown = synthesis_result.get("final_written_section_markdown")
            final_provenance = synthesis_result.get("final_written_section_provenance", [])
        else:
            final_written_section = synthesis_result
            final_markdown = None
            final_provenance = []

        # Extract critique score
        critique_score = None
        if isinstance(criticism_result, dict):
            critique_score = criticism_result.get("score")

        # Ensure criticism_result has the correct structure
        if "suggested_experiences" not in criticism_result:
            criticism_result = {"suggested_experiences": criticism_result}

        # Assemble final output
        output = {
            **analysis_result,  # experiences, aggregate_skills, processed_skills, domain_insights, gap_analysis, seniority_analysis
            **criticism_result,  # suggested_experiences
            "final_written_section": final_written_section,
            "final_written_section_markdown": final_markdown,
            "final_written_section_provenance": final_provenance,
            "critique_score": critique_score,
            "run_metrics": RUN_METRICS.copy(),
            "processing_status": ProcessingStatus.COMPLETED,
            "processing_time_seconds": time.time() - start_time
        }

        ## Validate output schema
        validate_output_schema(output)

        processing_time = time.time() - start_time
        logger.info("analyze.request.completed", extra={"request_id": request_id, "processing_time_seconds": processing_time, "run_metrics": RUN_METRICS.copy()})
        return AnalysisResponse(**output)

    except HTTPException as e:
        raise e
    except Exception as e:
        processing_time = time.time() - start_time
        logger.exception("analyze.request.failed", extra={"request_id": getattr(request.state, "request_id", None), "processing_time_seconds": processing_time})

        # Return error response with partial results if available
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=f"Analysis failed: {str(e)}",
                error_code="ANALYSIS_FAILED",
                details={
                    "processing_time_seconds": processing_time,
                    "run_metrics": RUN_METRICS.copy() if RUN_METRICS.get("calls") else None
                }
            ).model_dump()
        )


@app.post("/api/analysis/multi-file", response_model=AnalysisResponse, dependencies=[Depends(get_api_key)])
async def analyze_multi_file(
    request: Request,
    files: List[UploadFile] = File(...),
    job_ad: Optional[str] = Form(default=None),
    api_key: str = Depends(get_api_key),
):
    start_time = time.time()

    # Multipart/form-data requests do not support request.json()
    request_id = getattr(request.state, "request_id", None)
    logger.info("analyze_multi_file.request.incoming", extra={"request_id": request_id, "file_count": len(files)})
    try:
        contents: List[str] = []
        for f in files:
            data = await f.read()
            if isinstance(data, bytes):
                contents.append(data.decode("utf-8", errors="ignore"))
            else:
                contents.append(str(data))
        resume_text = "\n\n".join([c for c in contents if c])
        if not resume_text.strip():
            raise HTTPException(status_code=422, detail="files cannot be empty")
        api_keys = [key for key in [settings.openrouter_api_key_1, settings.openrouter_api_key_2] if key]
        RUN_METRICS.update({
            "calls": [],
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "failures": []
        })
        logger.info("run_analysis.start", extra={"request_id": request_id})
        t0 = time.time()
        analysis_result = await run_analysis_async(
            resume_text=resume_text,
            job_ad=job_ad,
            extracted_skills_json=None,
            domain_insights_json=None,
            confidence_threshold=settings.confidence_threshold,
            openrouter_api_keys=api_keys
        )
        analysis_duration_ms = int((time.time() - t0) * 1000)
        logger.info("run_analysis.end", extra={"request_id": request_id, "duration_ms": analysis_duration_ms})
        RUN_METRICS.setdefault("durations_ms", {})["analysis_ms"] = analysis_duration_ms
        logger.info("run_generation.start", extra={"request_id": request_id})
        t0 = time.time()
        generation_result = await run_generation_async(
            analysis_json=analysis_result,
            job_ad=job_ad,
            openrouter_api_keys=api_keys
        )
        generation_duration_ms = int((time.time() - t0) * 1000)
        logger.info("run_generation.end", extra={"request_id": request_id, "duration_ms": generation_duration_ms})
        RUN_METRICS.setdefault("durations_ms", {})["generation_ms"] = generation_duration_ms
        logger.info("run_criticism.start", extra={"request_id": request_id})
        t0 = time.time()
        criticism_result = await run_criticism_async(
            generated_text=generation_result,
            job_ad=job_ad,
            openrouter_api_keys=api_keys
        )
        criticism_duration_ms = int((time.time() - t0) * 1000)
        logger.info("run_criticism.end", extra={"request_id": request_id, "duration_ms": criticism_duration_ms})
        RUN_METRICS.setdefault("durations_ms", {})["criticism_ms"] = criticism_duration_ms
        if isinstance(criticism_result, str):
            try:
                from imaginator_flow import ensure_json_dict
                criticism_result = ensure_json_dict(criticism_result, "critique")
            except Exception:
                criticism_result = {"suggested_experiences": {"bridging_gaps": [], "metric_improvements": []}}
        logger.info("run_synthesis.start", extra={"request_id": request_id, "final_writer": getattr(settings, "FINAL_WRITER_PROVIDER", None)})
        t0 = time.time()
        synthesis_result = await run_synthesis_async(
            generated_text=generation_result,
            critique_json=criticism_result,
            job_ad=job_ad,
            openrouter_api_keys=api_keys,
            analysis_result=analysis_result,  # FIX: Changed from analysis_for_provenance to analysis_result
            final_writer=getattr(settings, "FINAL_WRITER_PROVIDER", None)  # Use configured provider (safe getattr)
        )
        synthesis_duration_ms = int((time.time() - t0) * 1000)
        logger.info("run_synthesis.end", extra={"request_id": request_id, "duration_ms": synthesis_duration_ms})
        RUN_METRICS.setdefault("durations_ms", {})["synthesis_ms"] = synthesis_duration_ms

        # Extract structured fields from synthesis result
        if isinstance(synthesis_result, dict):
            final_written_section = synthesis_result.get("final_written_section", synthesis_result)
            final_markdown = synthesis_result.get("final_written_section_markdown")
            final_provenance = synthesis_result.get("final_written_section_provenance", [])
        else:
            final_written_section = synthesis_result
            final_markdown = None
            final_provenance = []

        # Extract critique score
        critique_score = None
        if isinstance(criticism_result, dict):
            critique_score = criticism_result.get("score")

        if "suggested_experiences" not in criticism_result:
            criticism_result = {"suggested_experiences": criticism_result}
        output = {
            **analysis_result,
            **criticism_result,
            "final_written_section": final_written_section,
            "final_written_section_markdown": final_markdown,
            "final_written_section_provenance": final_provenance,
            "critique_score": critique_score,
            "run_metrics": RUN_METRICS.copy(),
            "processing_status": ProcessingStatus.COMPLETED,
            "processing_time_seconds": time.time() - start_time
        }
        validate_output_schema(output)
        processing_time = time.time() - start_time
        logger.info("analyze_multi_file.request.completed", extra={"request_id": request_id, "processing_time_seconds": processing_time, "run_metrics": RUN_METRICS.copy()})
        return AnalysisResponse(**output)
    except HTTPException as e:
        raise e
    except Exception as e:
        processing_time = time.time() - start_time
        logger.exception("analyze_multi_file.request.failed", extra={"request_id": getattr(request.state, "request_id", None), "processing_time_seconds": processing_time})
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=f"Analysis failed: {str(e)}",
                error_code="ANALYSIS_FAILED",
                details={
                    "processing_time_seconds": processing_time,
                    "run_metrics": RUN_METRICS.copy() if RUN_METRICS.get("calls") else None
                }
            ).model_dump()
        )

# Note: File uploads are no longer required. FrontEnd sends structured JSON only.
# The `/analyze` endpoint accepts the JSON payload and is the single supported entrypoint.


@app.get("/docs/{library}", response_model=Context7DocsResponse)
async def get_library_docs(library: str, version: str = "latest"):
    """
    Get up-to-date documentation for a library using Context7 MCP.

    Requires CONTEXT7_API_KEY to be configured.
    """
    if not settings.context7_api_key:
        raise HTTPException(
            status_code=503,
            detail="Context7 integration not configured. Set CONTEXT7_API_KEY environment variable."
        )

    try:
        # Import Context7 client (optional dependency)
        from context7 import Context7Client

        client = Context7Client(api_key=settings.context7_api_key)
        docs = await client.get_docs(library, version)

        return Context7DocsResponse(
            library=library,
            version=version,
            documentation=docs,
            last_updated=None  # Context7 doesn't provide timestamps
        )

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Context7 MCP not installed. Install with: pip install context7-mcp"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Context7 documentation retrieval failed: {str(e)}"
        )


@app.get("/keys/health")
async def keys_health():
    """Check health of configured API keys"""
    async def check_key(key: str) -> Dict[str, Any]:
        if not key:
            return {"status": "missing"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {key}"}
                ) as response:
                    return {
                        "status": "healthy" if response.status == 200 else "unhealthy",
                        "details": await response.text() if response.status != 200 else None
                    }
        except Exception as e:
            return {"status": "error", "details": str(e)}

    key1_status = await check_key(settings.openrouter_api_key_1)
    key2_status = await check_key(settings.openrouter_api_key_2)

    return {
        "openrouter_key_1": key1_status,
        "openrouter_key_2": key2_status,
        "overall": "healthy" if any(s["status"] == "healthy" for s in [key1_status, key2_status]) else "unhealthy"
    }

@app.get("/config")
async def get_config():
    """Get current configuration (without sensitive API keys)"""
    return {
        "environment": settings.environment,
        "confidence_threshold": settings.confidence_threshold,
        "max_concurrent_requests": settings.max_concurrent_requests,
        "request_timeout": settings.request_timeout,
        "debug": settings.debug,
        "has_openrouter_key": bool(settings.openrouter_api_key_1 or settings.openrouter_api_key_2),
        "has_context7_key": bool(settings.context7_api_key)
    }

@app.get("/debug/flags")
async def debug_flags():
    """Return non-sensitive feature flag values and presence of related service URLs."""
    return {
        "ENABLE_LOADER": bool(getattr(settings, "ENABLE_LOADER", False)),
        "ENABLE_FASTSVM": bool(getattr(settings, "ENABLE_FASTSVM", False)),
        "ENABLE_HERMES": bool(getattr(settings, "ENABLE_HERMES", False)),
        "FINAL_WRITER_PROVIDER": getattr(settings, "FINAL_WRITER_PROVIDER", None),
        "LOADER_BASE_URL_SET": bool(getattr(settings, "LOADER_BASE_URL", None)),
        "FASTSVM_BASE_URL_SET": bool(getattr(settings, "FASTSVM_BASE_URL", None)),
        "HERMES_BASE_URL_SET": bool(getattr(settings, "HERMES_BASE_URL", None)),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        workers=settings.workers
    )
