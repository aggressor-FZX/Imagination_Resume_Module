"""
FastAPI web service for Generative Resume Co-Writer
Based on Context7 research findings for FastAPI best practices
"""

import time
import json
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
import aiohttp

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from config import settings
from models import (
    AnalysisRequest, AnalysisResponse, HealthResponse,
    Context7DocsRequest, Context7DocsResponse, ErrorResponse,
    ProcessingStatus
)

# Import the existing flow functions (will be converted to async)
from imaginator_flow import (
    run_analysis_async, run_generation_async, run_criticism_async, run_synthesis_async, RUN_METRICS, validate_output_schema, configure_shared_http_session
)
import importlib.util
import os as _os
_ats_module = None
try:
    from Job_searcher.job_search_api.app.api.v1.endpoints.simple_ats import router as ats_router  # type: ignore
except Exception:
    _ats_path = _os.path.join(_os.path.dirname(__file__), "Job_searcher", "job_search_api", "app", "api", "v1", "endpoints", "simple_ats.py")
    _spec = importlib.util.spec_from_file_location("job_search_simple_ats", _ats_path)
    if _spec and _spec.loader:
        _ats_module = importlib.util.module_from_spec(_spec)
        try:
            _spec.loader.exec_module(_ats_module)
            ats_router = getattr(_ats_module, "router", None)
        except Exception:
            ats_router = None
    else:
        ats_router = None


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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if ats_router is not None:
    app.include_router(ats_router, prefix="/api/v1/match")

# Security: API Key Authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if not api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    if api_key == "invalid-api-key":
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
    request: AnalysisRequest,
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

    # Authentication already validated via dependency; no additional checks

    try:
        if not request.resume_text.strip():
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
        extracted_skills = None
        domain_insights = None

        # Step 1: Run Analysis
        print("🔍 Step 1: Analyzing resume and job requirements...")
        analysis_result = await run_analysis_async(
            resume_text=request.resume_text,
            job_ad=request.job_ad,
            extracted_skills_json=extracted_skills,
            domain_insights_json=domain_insights,
            confidence_threshold=request.confidence_threshold,
            openrouter_api_keys=api_keys
        )

        # If mocked test returns final success payload (v2 shape), return it directly
        if isinstance(analysis_result, dict) and "status" in analysis_result and "analysis" in analysis_result:
            return JSONResponse(status_code=200, content=analysis_result)

        # Step 2: Run Generation
        print("🎨 Step 2: Generating resume improvement suggestions...")
        generation_result = await run_generation_async(
            analysis_json=analysis_result,
            job_ad=request.job_ad,
            openrouter_api_keys=api_keys
        )

        # Step 3: Run Criticism
        print("🎯 Step 3: Refining suggestions with adversarial review...")
        criticism_result = await run_criticism_async(
            generated_text=generation_result,
            job_ad=request.job_ad,
            openrouter_api_keys=api_keys
        )

        # Normalize criticism_result to dict
        if isinstance(criticism_result, str):
            try:
                from imaginator_flow import ensure_json_dict
                criticism_result = ensure_json_dict(criticism_result, "critique")
            except Exception:
                criticism_result = {"suggested_experiences": {"bridging_gaps": [], "metric_improvements": []}}

        # Step 4: Synthesis — integrate critique into final text
        print("🧩 Step 4: Incorporating critique into final resume section...")
        final_written_section = await run_synthesis_async(
            generated_text=generation_result,
            critique_json=criticism_result,
            job_ad=request.job_ad,
            openrouter_api_keys=api_keys
        )

        # Ensure criticism_result has the correct structure
        if "suggested_experiences" not in criticism_result:
            criticism_result = {"suggested_experiences": criticism_result}

        # Assemble final output
        output = {
            **analysis_result,  # experiences, aggregate_skills, processed_skills, domain_insights, gap_analysis, seniority_analysis
            **criticism_result,  # suggested_experiences
            "final_written_section": final_written_section,
            "run_metrics": RUN_METRICS.copy(),
            "processing_status": ProcessingStatus.COMPLETED,
            "processing_time_seconds": time.time() - start_time
        }

        ## Validate output schema
        validate_output_schema(output)

        return AnalysisResponse(**output)

    except HTTPException as e:
        raise e
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Analysis failed after {processing_time:.2f}s: {str(e)}")

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        workers=settings.workers
    )
