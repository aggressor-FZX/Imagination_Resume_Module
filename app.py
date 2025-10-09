"""
FastAPI web service for Generative Resume Co-Writer
Based on Context7 research findings for FastAPI best practices
"""

import time
import json
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from models import (
    AnalysisRequest, AnalysisResponse, HealthResponse,
    Context7DocsRequest, Context7DocsResponse, ErrorResponse,
    ProcessingStatus
)

# Import the existing flow functions (will be converted to async)
from imaginator_flow import (
    run_analysis_async, run_generation_async, run_criticism,
    validate_output_schema, RUN_METRICS
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events"""
    # Startup
    print(f"🚀 Starting {settings.environment} server on {settings.host}:{settings.port}")

    # Validate required API keys
    if not settings.openai_api_key and not settings.anthropic_api_key:
        raise RuntimeError("At least one API key (OpenAI or Anthropic) must be configured")

    yield

    # Shutdown
    print("🛑 Shutting down server")


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
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers and monitoring"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        environment=settings.environment
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_resume(request: AnalysisRequest):
    """
    Analyze a resume against a job description and provide career development recommendations.

    This endpoint processes the resume through three stages:
    1. Analysis: Extract skills and identify gaps
    2. Generation: Create improvement suggestions
    3. Criticism: Refine suggestions with adversarial review
    """
    start_time = time.time()

    try:
        # Reset run metrics for this request
        RUN_METRICS.update({
            "calls": [],
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "failures": []
        })

        # Parse optional JSON inputs
        extracted_skills = None
        domain_insights = None

        if request.extracted_skills_json:
            try:
                extracted_skills = json.loads(request.extracted_skills_json)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid extracted_skills_json format"
                )

        if request.domain_insights_json:
            try:
                domain_insights = json.loads(request.domain_insights_json)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid domain_insights_json format"
                )

        # Step 1: Run Analysis
        print("🔍 Step 1: Analyzing resume and job requirements...")
        analysis_result = await run_analysis_async(
            resume_text=request.resume_text,
            job_ad=request.job_ad,
            extracted_skills_json=extracted_skills,
            domain_insights_json=domain_insights,
            confidence_threshold=request.confidence_threshold
        )

        # Step 2: Run Generation (with graceful degradation)
        print("🎨 Step 2: Generating resume improvement suggestions...")
        try:
            generation_result = await run_generation_async(
                analysis_json=analysis_result,
                job_ad=request.job_ad
            )
        except Exception as e:
            print(f"⚠️  Generation step failed: {str(e)}")
            print("🔄 Continuing with analysis results only...")
            generation_result = {"gap_bridging": [], "metric_improvements": []}

        # Step 3: Run Criticism (with graceful degradation)
        print("🎯 Step 3: Refining suggestions with adversarial review...")
        try:
            criticism_result = run_criticism(
                generated_suggestions=generation_result,
                job_ad=request.job_ad
            )
        except Exception as e:
            print(f"⚠️  Criticism step failed: {str(e)}")
            print("🔄 Continuing with generation results only...")
            criticism_result = {
                "suggested_experiences": {
                    "bridging_gaps": generation_result.get("gap_bridging", []),
                    "metric_improvements": generation_result.get("metric_improvements", [])
                }
            }

        # Assemble final output
        output = {
            **analysis_result,  # experiences, aggregate_skills, processed_skills, domain_insights, gap_analysis
            **criticism_result,  # suggested_experiences
            "run_metrics": RUN_METRICS.copy(),
            "processing_status": ProcessingStatus.COMPLETED,
            "processing_time_seconds": time.time() - start_time
        }

        # Validate output schema
        validate_output_schema(output)

        return output

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


@app.post("/analyze-file", response_model=AnalysisResponse)
async def analyze_resume_file(
    job_ad: str = Form(..., description="Target job description text"),
    resume_file: UploadFile = File(..., description="Resume file (text format)"),
    confidence_threshold: float = Form(0.7, description="Confidence threshold for skills"),
    extracted_skills_json: str = Form(None, description="JSON string of extracted skills"),
    domain_insights_json: str = Form(None, description="JSON string of domain insights")
):
    """
    Analyze a resume file upload against a job description.

    Accepts multipart form data with file upload for the resume.
    """
    # Read resume file content
    try:
        resume_content = await resume_file.read()
        resume_text = resume_content.decode('utf-8')
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Resume file must be valid UTF-8 text"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read resume file: {str(e)}"
        )
    finally:
        await resume_file.close()

    # Create analysis request and delegate to main endpoint logic
    request = AnalysisRequest(
        resume_text=resume_text,
        job_ad=job_ad,
        confidence_threshold=confidence_threshold,
        extracted_skills_json=extracted_skills_json,
        domain_insights_json=domain_insights_json
    )

    # Use the same logic as the JSON endpoint
    return await analyze_resume(request)


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


@app.get("/config")
async def get_config():
    """Get current configuration (without sensitive API keys)"""
    return {
        "environment": settings.environment,
        "confidence_threshold": settings.confidence_threshold,
        "max_concurrent_requests": settings.max_concurrent_requests,
        "request_timeout": settings.request_timeout,
        "debug": settings.debug,
        "has_openai_key": bool(settings.openai_api_key),
        "has_anthropic_key": bool(settings.anthropic_api_key),
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