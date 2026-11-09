"""
FastAPI web service for Generative Resume Co-Writer
Based on Context7 research findings for FastAPI best practices
"""

import os
import time
import json
import hashlib
import hmac
import re
import math
import uuid
from typing import Callable, Dict, Any, Optional, List
from contextlib import asynccontextmanager
import aiohttp

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Security,
    UploadFile,
    File,
    Form,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
import asyncio
from stream_context import set_token_callback, reset_token_callback

from config import settings
from logging_setup import configure_logging, RequestIDMiddleware
from models import (
    AnalysisRequest,
    AnalysisResponse,
    HealthResponse,
    Context7DocsResponse,
    ErrorResponse,
    ProcessingStatus,
)
from pydantic import BaseModel, Field
from pipeline_config import TIMEOUTS, OPENROUTER_API_KEYS
from llm_client_adapter import LLMClientAdapter
from stages.researcher import Researcher
import imaginator_flow as imaginator_flow_module
from imaginator_flow import (
    RUN_METRICS,
    validate_output_schema,
    configure_shared_http_session,
    redact_for_logging,
    run_analysis_async,
    run_criticism_async,
    run_synthesis_async,
)
from imaginator_new_integration import run_new_pipeline_async
import publication_gate
from approved_resume import unsupported_entities
import tier_a_verifier
import live_capture
from correction_provider import CorrectionProvider
from correction_events import log_event
from source_correction import SourceDocument, correct

# Configure global logger for this service
logger = configure_logging(service_name="imaginator")
run_generation_async = imaginator_flow_module.run_generation_async


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events"""
    # Startup
    print(
        f"🚀 Starting {settings.environment} server on {settings.host}:{settings.port}"
    )

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
    lifespan=lifespan,
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


# ── Trusted owner identity (owner item 3) ────────────────────────────────
# Approval receipts are bound to an owner so one candidate's approval cannot
# authorise another candidate's export. That owner MUST come from the trusted
# request context, not the request body: a caller who supplies
# `{"owner": "someone-else"}` is asserting, not authenticating.
#
# This service is invoked service-to-service by the orchestrator, which
# authenticates the end user and forwards their identity on a dedicated header.
# The header is only meaningful because the request already passed `get_api_key`;
# an unauthenticated caller cannot set it at all.
OWNER_IDENTITY_HEADER = "x-cogito-owner-id"
# A guest owner, distinct from an account owner. A signed-out visitor has no
# account, so the account header is legitimately empty; putting a device value
# there would mint an approval that LOOKS account-owned and could then be spent
# under a stale sign-in. Its own header keeps the type explicit.
#
# The value is NOT a raw visitor id. The orchestrator derives it as `guest:<hash>`
# from a signed cookie it minted after checking a real free-trial record (see
# Login_Landing_page/backend/services/guestSession.js). This service therefore
# never has to trust a client-chosen string: it only has to check the namespace,
# and the binding is provided by the orchestrator's signature.
GUEST_IDENTITY_HEADER = "x-cogito-guest-id"
GUEST_OWNER_PREFIX = "guest:"
_GUEST_OWNER_PATTERN = re.compile(r"guest:[A-Za-z0-9_-]{16,128}")


def trusted_owner_identity(request) -> Optional[str]:
    """Owner identity from the authenticated request context, or None.

    Deliberately reads ONLY the header. Falling back to body fields
    (`owner`, `user_id`, `email`) would let the caller choose who the approval
    belongs to, which defeats the binding.
    """
    try:
        raw = request.headers.get(OWNER_IDENTITY_HEADER)
    except Exception:  # noqa: BLE001
        return None
    value = (raw or "").strip()
    return value or None


def trusted_guest_identity(request) -> Optional[str]:
    """A server-derived guest owner from the request context, or None.

    Reads ONLY the dedicated header, for the same reason the account owner does:
    a body field would let the caller nominate whose approval this is.

    The value's provenance lives upstream. The orchestrator mints it as
    ``guest:<hash>`` from a cookie IT signed after confirming a real
    ``credits.free_trials`` record, so a client cannot produce this value by
    typing a visitor id. This function enforces the namespace and shape; it is not
    a substitute for that signature, and a direct caller speaking to this service
    without the orchestrator gets no guest owner at all.

    The namespace is load-bearing: an account id is a verified login, while a guest
    owner is a device capability. Keeping them disjoint means export can always
    tell which claim it holds, and a guest approval can never be spent under a
    stale sign-in.
    """
    try:
        raw = request.headers.get(GUEST_IDENTITY_HEADER)
    except Exception:  # noqa: BLE001
        return None
    value = (raw or "").strip()
    if not value or not _GUEST_OWNER_PATTERN.fullmatch(value):
        return None
    return value


def resolve_publication_owner(request):
    """The identity an approval may be bound to, and whether it is an account.

    A signed-in account always wins: it is a verified login, and an approval minted
    under it can be re-verified later. A free-trial guest is a real person who paid
    with credits and is entitled to the resume they paid for, so a server-verified
    guest capability is a legitimate owner -- refusing it discarded completed work
    for being logged out.

    Returns ``(owner, is_account)``. ``owner`` is None only when neither identity is
    available, which means no trustworthy binding exists and the approval must not
    be issued.
    """
    account = trusted_owner_identity(request)
    if account:
        return account, True
    guest = trusted_guest_identity(request)
    if guest:
        return guest, False
    return None, False


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers and monitoring"""
    from datetime import datetime, timezone

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        environment=settings.environment,
        timestamp=datetime.now(timezone.utc).isoformat(),
        has_openrouter_key=bool(
            settings.openrouter_api_key_1 or settings.openrouter_api_key_2
        ),
    )


@app.get("/")
async def root():
    return {"status": "ok", "service": "Generative Resume Co-Writer"}


def _snapshot_original_documents(resume_text, documents, supplied):
    """Snapshot Node's ordered candidate texts, rejecting incomplete evidence."""
    if not isinstance(resume_text, str) or not resume_text.strip():
        return None
    if not supplied:
        return (("resume", resume_text),)
    if type(documents) is not list or not documents:
        return None
    result, ids = [], set()
    for doc in documents:
        if (type(doc) is not dict or set(doc) != {"source_id", "text", "kind"}
                or type(doc["source_id"]) is not str
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", doc["source_id"])
                or doc["source_id"] in ids or type(doc["text"]) is not str
                or not doc["text"].strip() or doc["kind"] != "candidate_document"):
            return None
        ids.add(doc["source_id"])
        result.append((doc["source_id"], doc["text"]))
    if "\n\n---\n\n".join(text for _, text in result) != resume_text:
        return None
    return tuple(result)


def _correction_usage(pipeline_metrics, provider=None):
    """Return known usage only; never forward provider errors or model content."""
    base = pipeline_metrics if isinstance(pipeline_metrics, dict) else {}
    result = {"calls": [], "failures": [], "total_prompt_tokens": 0,
              "total_completion_tokens": 0, "total_tokens": 0,
              "estimated_cost_usd": 0.0}
    for key in ("total_prompt_tokens", "total_completion_tokens", "total_tokens"):
        value = base.get(key)
        if type(value) is int and value >= 0:
            result[key] = value
    cost = base.get("estimated_cost_usd")
    if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0:
        result["estimated_cost_usd"] = float(cost)
    for call in base.get("calls", []) if isinstance(base.get("calls"), list) else []:
        if isinstance(call, dict):
            clean = {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = call.get(key)
                if type(value) is int and value >= 0:
                    clean[key] = value
            value = call.get("cost_usd")
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                clean["cost_usd"] = float(value)
            clean["stage"] = "pipeline"
            result["calls"].append(clean)
    if base.get("failures"):
        result["failures"].append({"stage": "pipeline", "usage_unknown": True})
    if provider is not None:
        usage = provider.usage
        result["calls"].extend(dict(call) for call in usage["calls"])
        result["failures"].extend(dict(failure) for failure in usage["failures"])
        for key in ("total_prompt_tokens", "total_completion_tokens", "total_tokens",
                    "estimated_cost_usd"):
            result[key] += usage[key]
        # Cancellation may still be propagating when the overall deadline fires.
        if provider.calls > len(usage["calls"]) + sum(
                1 for failure in usage["failures"] if failure.get("usage_unknown")):
            result["failures"].append({"stage": "source_correction", "usage_unknown": True})
    return result


def _mode_from_payload(payload):
    """The effective inference mode stated by a request, for withheld results.

    An ABSENT field is not an unknown policy. ``normalize_mode(None)`` resolves to
    the documented default (imaginative / imaginative-v1), and the pipeline runs
    that default — so a withheld result must report it. Returning None here meant
    that every caller who relied on the documented default (including the
    website, which does not send the field) was told nothing about which policy
    produced their refusal, while a caller who sent the field explicitly was told.
    That difference reflected how the request was spelled, not what happened.

    A value that cannot be interpreted still falls back to None rather than
    guessing: validation refuses unknown modes elsewhere, so reaching that branch
    means something is wrong with this process, and asserting a policy that may
    not have run would be worse than saying nothing.
    """
    try:
        from skill_inference import mode_contract, normalize_mode

        requested = getattr(payload, "skill_inference_mode", None)
        contract = mode_contract(normalize_mode(requested))
        return (
            contract["skill_inference_mode"],
            contract["skill_inference_policy_version"],
        )
    except Exception:  # noqa: BLE001
        return None, None


# Reasons that describe a failure of THIS SERVICE, not of the candidate's resume.
# They are transient, retryable, and must never be reported as a judgement about
# the document. Measured 2026-11-08: 6 of 9 observed withholds were in this class
# (3 provider timeouts, 2 malformed reviewer answers, 1 no-op repair), yet all
# nine told the user their resume "could not be verified".
_INFRASTRUCTURE_REASONS = frozenset({
    "provider_rate_limited", "provider_unavailable", "provider_rejected_request",
    "provider_error", "privacy_redaction_failed", "privacy_review_required",
    "timeout", "pipeline_budget_exhausted",
    "timeout_source_inventory", "timeout_initial_review", "timeout_repair",
    "timeout_post_patch_review", "timeout_restore", "timeout_post_restore_review",
    "provider_call_timeout_source_inventory", "provider_call_timeout_initial_review",
    "provider_call_timeout_repair", "provider_call_timeout_post_patch_review",
    "provider_call_timeout_restore", "provider_call_timeout_post_restore_review",
    # Our reviewer emitted an answer our own validator rejected, or declined to
    # change a span it had itself flagged. Neither is evidence about the document.
    "malformed_output", "malformed_source_inventory", "no_op_repair",
    "incomplete_review", "correction_failed", "analysis_failed",
    "receipt_signing_unavailable",
    "render_contract_ambiguous", "section_contract_unavailable",
    "source_documents_mismatch",
})

# A third class, deliberately NOT folded into infrastructure: the user CAN fix it,
# but not by retrying the same request. "Please try again" would be as wrong here
# as "your resume could not be verified" -- a receipt that could not be bound to
# an owner fails identically every time the visitor stays logged out.
_ACTIONABLE_REASONS = frozenset({"receipt_unavailable"})

# The opposite class: the reviewer could not substantiate something IN the
# document. Only these may be phrased as a statement about the resume.
_CONTENT_REASONS = frozenset({
    "source_facts_missing", "source_completeness_uncertain",
    "source_section_missing", "attribution_conflict", "unresolved_claims",
    "no_supported_claims", "no_required_source_claims", "invalid_input",
})

_RETRY_MESSAGE = ("Our review service could not finish checking your resume in "
                  "time. Your resume was not rejected — please try again.")
_CONTENT_MESSAGE = ("We could not confirm every detail of your resume against "
                    "the document you uploaded. Please review the flagged "
                    "details and try again.")
_ACTIONABLE_MESSAGE = ("Your resume passed review, but it could not be released "
                      "to a signed-in account. Please sign in to continue.")


def _review_message_for(reason):
    """The user-facing sentence for a withhold, chosen by WHOSE failure it was.

    One hardcoded string used to cover every reason, so a 74-second provider
    timeout and a genuinely unplaceable source fact produced identical text:
    "The generated resume could not be verified and needs review." That sentence
    reads as a judgement on the candidate's document, which is false for every
    infrastructure failure — and infrastructure failures were the majority.

    An unrecognised reason is treated as infrastructure (retryable, non-
    accusatory). That is the safe default: telling someone to re-check a resume
    that was actually fine costs them work, while inviting a retry costs one
    request.
    """
    if reason in _CONTENT_REASONS:
        return _CONTENT_MESSAGE
    if reason in _ACTIONABLE_REASONS:
        return _ACTIONABLE_MESSAGE
    return _RETRY_MESSAGE


def _withhold_correction(*, reason, pipeline_status, metrics, start_time, decision=None,
                         aggregate_skills=None, skill_inference_mode=None,
                         skill_inference_policy_version=None):
    """Only public, fixed metadata survives a withheld correction.

    The effective inference mode is *public* metadata: the caller selected it and
    is entitled to know which policy produced the refusal. It is a plain string
    with no candidate content, so it does not weaken the withholding rule — and
    omitting it left the client unable to explain the outcome it had asked for.

    ``review_message`` is derived from the reason so the user is told whose
    failure this was; see ``_review_message_for``. ``retryable`` is the machine-
    readable form of the same distinction, so a client can offer a retry button
    instead of asking the user to edit a document that was never the problem.
    """
    reasons = decision.reason_codes if decision and decision.blocked() else [reason]
    primary = reasons[0]
    # True only when repeating the SAME request could succeed. An actionable
    # reason is neither retryable nor a content judgement: it needs a sign-in.
    retryable = (primary not in _CONTENT_REASONS
                 and primary not in _ACTIONABLE_REASONS)
    return AnalysisResponse(
        processing_status="needs_review", requires_review=True,
        review_reason=primary, review_reasons=reasons,
        review_message=_review_message_for(primary),
        retryable=retryable,
        verification_coverage="none", publication_receipt=None,
        approved_export_payload=None, pipeline_status=pipeline_status,
        correction_status="withheld", correction_reason=reason,
        correction_commentary={
            "attribution_conflict": "A literal source record appears under a different label; independent review of the original record is required.",
            "unresolved_claims": "The reviewer flagged unresolved claims after two correction attempts; independent review against the source is required.",
            "source_facts_missing": "The reviewer reported a source fact missing; independent review of the original source is required.",
            "source_section_missing": "A section present in your uploaded document arrived empty; independent review is required.",
            "privacy_redaction_failed": "Personal identifiers could not be removed without changing the source text; independent review is required.",
            "provider_rate_limited": "The correction service was temporarily busy; no unverified rewrite was issued. Please retry.",
            "provider_unavailable": "The correction service could not be reached after retrying; no unverified rewrite was issued. Please retry.",
            "provider_rejected_request": "The correction service refused the request and retrying would repeat the refusal; this needs operator attention.",
            "receipt_unavailable": "Your resume was approved, but the approval could not be bound to a signed-in owner, so it was not released. Sign in and export it.",
        }.get(reason, "Content withheld; independent review is required."),
        aggregate_skills=aggregate_skills,
        skill_inference_mode=skill_inference_mode,
        skill_inference_policy_version=skill_inference_policy_version,
        run_metrics=metrics, processing_time_seconds=time.time() - start_time,
    )


# The orchestrator calls this endpoint on one of two paths, measured in
# Login_Landing_page/backend:
#
# * JSON  -> mlAuthMiddleware.makeAuthenticatedRequest, which HARDCODES
#            timeout: 180000 (services/mlAuthMiddleware.js) and retries up to 3
#            attempts.
# * SSE   -> server.cjs calls axios directly with timeout: 300000, no retry, and
#            validateStatus: () => true. mlAuthMiddleware is bypassed entirely.
#
# Both clocks start before this handler runs, so correction may only spend what is
# left. Using the JSON ceiling for an SSE caller understated the available budget by
# 120s in every live run: run 6 spent 53,516ms of an actually-available 173,516ms,
# then hit its own (correct) 120s cap. The caller's path is therefore detected from
# the request rather than assumed, and the smaller ceiling remains the default
# because it is the safe one to be wrong about.
UPSTREAM_CLIENT_TIMEOUT_SECONDS = 180.0
UPSTREAM_SSE_TIMEOUT_SECONDS = 300.0
UPSTREAM_RESPONSE_RESERVE_SECONDS = 20.0
# Raised from 120s after measuring why runs timed out. Every observed run was
# granted the full budget, so timeouts were never upstream starvation — the
# 120s ceiling itself was the constraint. Measured on run 04:08:55 on 2026-11-08:
# source_inventory 45.7s, then initial_review needed 74.3s and had exactly that,
# ending at the deadline. Run 04:12:21 spent 33.6s + 59.2s = 92.8s and approved.
# The two phases need ~93-120s combined, so jitter decided the outcome.
#
# The orchestrator allows 300s (SSE) and 180s (JSON) for the whole request, and
# ``correction_budget_seconds`` still clamps to ``upstream - reserve - elapsed``,
# so raising this cannot overrun the caller: a JSON caller automatically keeps
# the smaller remaining value. The engine's own ceiling (MAX_TIMEOUT_SECONDS in
# source_correction) must stay >= this.
MAX_CORRECTION_DEADLINE_SECONDS = 220.0
# Below this, one provider call (60s per-call timeout) cannot finish meaningfully;
# do not start it.
MIN_CORRECTION_BUDGET_SECONDS = 30.0


def _upstream_timeout_seconds(request=None):
    """The client timeout actually governing this request, measured per route.

    An SSE request is served by the orchestrator's direct-axios branch (300s); every
    other request goes through mlAuthMiddleware (180s). Detection is by the Accept
    header the orchestrator sets on the direct branch, and an ambiguous request
    keeps the smaller ceiling.
    """
    if request is None:
        return UPSTREAM_CLIENT_TIMEOUT_SECONDS
    headers = getattr(request, "headers", None) or {}
    try:
        accept = headers.get("accept", "") or ""
    except Exception:
        return UPSTREAM_CLIENT_TIMEOUT_SECONDS
    if "text/event-stream" in accept.lower():
        return max(UPSTREAM_CLIENT_TIMEOUT_SECONDS, UPSTREAM_SSE_TIMEOUT_SECONDS)
    return UPSTREAM_CLIENT_TIMEOUT_SECONDS


def correction_budget_seconds(pipeline_elapsed_seconds, upstream_timeout_seconds=None):
    """Overall correction deadline left after the pipeline, or None if too little remains.

    Guarantee when a budget is returned:
    pipeline_elapsed + budget + UPSTREAM_RESPONSE_RESERVE_SECONDS <= upstream timeout,
    so the response still reaches the caller before their client gives up.
    """
    if type(pipeline_elapsed_seconds) not in (int, float) or not math.isfinite(pipeline_elapsed_seconds):
        return None
    upstream = upstream_timeout_seconds
    if type(upstream) not in (int, float) or not math.isfinite(upstream):
        upstream = UPSTREAM_CLIENT_TIMEOUT_SECONDS
    remaining = upstream - UPSTREAM_RESPONSE_RESERVE_SECONDS - pipeline_elapsed_seconds
    budget = min(MAX_CORRECTION_DEADLINE_SECONDS, remaining)
    return budget if budget >= MIN_CORRECTION_BUDGET_SECONDS else None


def _join_correlation_id(request, request_id):
    """Prefer the orchestrator's x-correlation-id so both services' logs can be joined."""
    value = request.headers.get("x-correlation-id", "") if request is not None else ""
    if re.fullmatch(r"[A-Za-z0-9-]{1,64}", value or ""):
        return value
    return request_id or str(uuid.uuid4())


def _correction_observer(correlation_id):
    """Sanitized phase events for one request; see correction_events."""
    def observe(event):
        log_event(logger, {**event, "correlation_id": correlation_id})
    return observe


async def _run_source_correction(markdown, source_documents, api_key, request, *,
                                 timeout_seconds=120.0, observer=None, capture=None,
                                 permitted_bridges=None):
    """Use the application session when available, otherwise close a local session."""
    from disclosure import redact_sources

    # Length-preserving redaction keeps every code-point offset valid, so the
    # engine's exact-citation contract still holds after the sensitive
    # characters are removed. When correspondence cannot be proven, withhold.
    redacted, attestation = redact_sources(source_documents)
    if attestation is None:
        return None, None
    sources = tuple(SourceDocument(source_id, text) for source_id, text in redacted)
    session = getattr(request.app.state, "http_session", None)
    if capture is not None:
        # Record what the live request actually used: the redacted sources that are
        # about to be reviewed, and the writer's own draft. Both are read from the
        # live path, never reconstructed from a fixture.
        try:
            capture.record_sources(redacted)
            capture.record_draft(markdown)
        except Exception:
            pass

    async def run(session):
        wrapped = live_capture.CapturingSession(session, capture) if capture else session
        provider = CorrectionProvider(api_key, wrapped, observer=observer)
        if capture is not None:
            provider = _PhaseRecordingProvider(provider, capture)
        result = await correct(markdown, sources, provider,
                               timeout_seconds=timeout_seconds, observer=observer,
                               permitted_bridges=permitted_bridges)
        # Allow a cancelled provider coroutine to record its usage state.
        await asyncio.sleep(0)
        return result, provider

    if session is not None and not getattr(session, "closed", False):
        return await run(session)
    async with aiohttp.ClientSession() as session:
        return await run(session)


class _PhaseRecordingProvider:
    """Record each engine phase's own payload and normalized answer.

    The wire exchange alone cannot explain a restoration refusal: the transmitted
    request deliberately drops the internal ``source_claim_id`` and the engine's
    line mapping, so "which fact was paired with which line" exists nowhere in the
    captured HTTP body. This wrapper stores the engine's own phase input, the
    adapter's normalized return value and a bounded failure label.

    It is a transparent proxy: every attribute is delegated, ``set_budget`` is
    forwarded because ``_invoke`` depends on it, and any recording error is
    swallowed so diagnostics can never change the correction decision or the
    response a user receives.
    """

    def __init__(self, inner, capture):
        self._inner = inner
        self._capture = capture

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def set_budget(self, remaining_seconds):
        setter = getattr(self._inner, "set_budget", None)
        if callable(setter):
            return setter(remaining_seconds)
        return None

    async def __call__(self, phase, payload):
        try:
            normalized = await self._inner(phase, payload)
        except BaseException as exc:  # noqa: BLE001 - record, then re-raise unchanged
            try:
                self._capture.record_phase(phase, _copy_phase_payload(payload),
                                           error=type(exc).__name__)
            except Exception:
                pass
            raise
        try:
            self._capture.record_phase(phase, _copy_phase_payload(payload),
                                       result=_copy_phase_payload(normalized))
        except Exception:
            pass
        return normalized


def _copy_phase_payload(value):
    """A JSON-safe deep copy, or a shape label when it cannot be represented.

    The phase payloads are engine-owned plain structures, but a caller may attach
    arbitrary objects to a direct-callback payload. Copying through JSON keeps the
    record serialisable without leaking a live object into the capture file.
    """
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return {"<unserializable>": type(value).__name__}



async def _analyze_resume_impl(
    payload: AnalysisRequest,
    request: Request,
    api_key: str,
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
    request_id = getattr(request.state, "log_request_id", None)
    source_documents_snapshot = _snapshot_original_documents(
        payload.resume_text, payload.source_documents,
        "source_documents" in getattr(payload, "model_fields_set", set()),
    )
    if source_documents_snapshot is None:
        return AnalysisResponse(
            processing_status="needs_review", requires_review=True,
            review_reason="source_documents_mismatch", review_reasons=["source_documents_mismatch"],
            review_message="Source documents could not be verified.",
            publication_receipt=None, approved_export_payload=None,
            verification_coverage="none", processing_time_seconds=time.time() - start_time,
        )
    # ── Immutable source evidence, captured at INGRESS (owner item 6) ────
    # Taken from the submitted body and the uploaded text, before any stage
    # runs. Every gate call below uses this snapshot instead of re-deriving
    # evidence from `analysis_result`, which contains generated projects and
    # could therefore approve itself.
    source_snapshot = _snapshot_source_evidence(body, body.get("resume_text"))
    logger.info(
        "analyze.request.incoming",
        extra={
            "request_id": request_id,
            "payload_size": len(json.dumps(body)) if body else 0,
        },
    )

    # Log incoming request and feature flags
    logger.info("[IMAGINATOR ENDPOINT] === Received analyze request ===")
    logger.info(f"[IMAGINATOR ENDPOINT] Resume length: {len(payload.resume_text)}")
    logger.info(
        f"[IMAGINATOR ENDPOINT] Job ad length: {len(payload.job_ad) if payload.job_ad else 0}"
    )
    logger.info(
        f"[IMAGINATOR ENDPOINT] Confidence threshold: {payload.confidence_threshold}"
    )
    logger.info("[IMAGINATOR ENDPOINT] Feature Flags:")
    logger.info(f"[IMAGINATOR ENDPOINT]   - ENABLE_LOADER: {settings.ENABLE_LOADER}")
    logger.info(f"[IMAGINATOR ENDPOINT]   - ENABLE_FASTSVM: {settings.ENABLE_FASTSVM}")
    logger.info(f"[IMAGINATOR ENDPOINT]   - ENABLE_HERMES: {settings.ENABLE_HERMES}")
    logger.info(
        f"[IMAGINATOR ENDPOINT]   - VERBOSE_PIPELINE_LOGS: {settings.VERBOSE_PIPELINE_LOGS}"
    )
    logger.info(
        "[IMAGINATOR ENDPOINT]   - VERBOSE_MICROSERVICE_LOGS: %s",
        settings.VERBOSE_MICROSERVICE_LOGS,
    )

    # Parse structured data from backend
    skills_data = None
    if payload.extracted_skills_json:
        try:
            skills_data = (
                json.loads(payload.extracted_skills_json)
                if isinstance(payload.extracted_skills_json, str)
                else payload.extracted_skills_json
            )
            skill_count = (
                len(skills_data)
                if isinstance(skills_data, list)
                else len(skills_data.get("skills", []))
            )
            logger.info(
                f"[IMAGINATOR ENDPOINT] Received extracted_skills_json with {skill_count} skills"
            )
            logger.info(
                f"[IMAGINATOR ENDPOINT] Skills structure type: {type(skills_data).__name__}"
            )
            if isinstance(skills_data, list) and len(skills_data) > 0:
                # Check if backend sent structured skills with confidence (COMPLIANCE)
                first_skill = skills_data[0]
                if (
                    isinstance(first_skill, dict)
                    and "confidence" in first_skill
                    and "source" in first_skill
                ):
                    logger.info(
                        "[COMPLIANCE] structured-skills-v1: Backend sent STRUCTURED skills with confidence metadata"
                    )
                else:
                    logger.warning(
                        "[COMPLIANCE] structured-skills-v1: Backend sent RAW skills (missing confidence/source) - will trigger fallback"
                    )
        except Exception:
            logger.warning("[IMAGINATOR ENDPOINT] Could not parse extracted_skills_json")
            skills_data = None
    else:
        logger.info(
            "[IMAGINATOR ENDPOINT] No extracted_skills_json provided (will use fallback)"
        )

    insights_data = None
    if payload.domain_insights_json:
        try:
            insights_data = (
                json.loads(payload.domain_insights_json)
                if isinstance(payload.domain_insights_json, str)
                else payload.domain_insights_json
            )
            logger.info(
                "[IMAGINATOR ENDPOINT] Received domain_insights_json (object: %s)",
                isinstance(insights_data, dict),
            )
        except Exception:
            logger.warning("[IMAGINATOR ENDPOINT] Could not parse domain_insights_json")
            insights_data = None
    else:
        logger.info("[IMAGINATOR ENDPOINT] No domain_insights_json provided")

    # Authentication already validated via dependency; no additional checks

    try:
        if not payload.resume_text.strip():
            raise HTTPException(status_code=422, detail="resume_text cannot be empty")
        # Use server-configured OpenRouter keys only (BYOK removed)
        api_keys = [
            key
            for key in [settings.openrouter_api_key_1, settings.openrouter_api_key_2]
            if key
        ]

        # Reset run metrics for this request
        RUN_METRICS.update(
            {
                "calls": [],
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "failures": [],
            }
        )

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
            openrouter_api_keys=api_keys,
            creativity_mode=payload.creativity_mode,
            # Permission policy, validated at the model boundary. Omitted =>
            # imaginative; an unknown value already failed request validation.
            skill_inference_mode=(
                payload.skill_inference_mode.value
                if payload.skill_inference_mode is not None
                else None
            ),
            location=payload.location,  # Pass location for market intel enrichment
            job_title=payload.job_title,  # Pass user-provided job title for O*NET
            projects=payload.projects,  # Pass projects for students/career changers
            education=payload.education,  # Pass education entries when provided
            certifications=payload.certifications,  # Pass certifications when provided
            golden_bullets=payload.golden_bullets,
            onet_code=payload.onet_code,
            experience=payload.experience,
            skills=payload.skills,
            classified_skills=payload.classified_skills,
            job_titles=payload.job_titles,
            research_data=payload.research_data,
            # The immutable candidate documents. Stage 1 needs them to anchor a
            # bridge to a real activity, and this call previously dropped them —
            # so the Researcher received None, had nothing it was allowed to
            # quote, and correctly proposed zero bridges. The mode looked
            # implemented while its only source of evidence was missing.
            #
            # Taken from the request snapshot, never from pipeline output.
            source_documents=[
                {"source_id": sid, "text": text, "kind": "candidate_document"}
                for sid, text in (source_documents_snapshot or ())
            ] or None,
        )
        analysis_duration_ms = int((time.time() - t0) * 1000)
        logger.info(
            "run_new_pipeline.end",
            extra={"request_id": request_id, "duration_ms": analysis_duration_ms},
        )
        RUN_METRICS.setdefault("durations_ms", {})["pipeline_ms"] = analysis_duration_ms
        # Only the normal completed pipeline may enter source-aware correction.
        # A malformed or failed pipeline must not trigger additional paid calls.
        if not isinstance(analysis_result, dict):
            _w_mode, _w_policy = _mode_from_payload(payload)
            return _withhold_correction(
                reason="pipeline_invalid", pipeline_status="failed",
                metrics=_correction_usage(RUN_METRICS), start_time=start_time,
                skill_inference_mode=_w_mode,
                skill_inference_policy_version=_w_policy)
        if isinstance(analysis_result, dict):
            logger.info(
                "analyze.pipeline.fast_track",
                extra={
                    "request_id": request_id,
                    "msg_text": "4-Stage Pipeline completed in analysis step. Skipping legacy steps.",
                },
            )

            pipeline_metrics = analysis_result.get("run_metrics", {})
            if not isinstance(pipeline_metrics, dict):
                pipeline_metrics = RUN_METRICS
            metrics = _correction_usage(pipeline_metrics)
            status = analysis_result.get("pipeline_status")
            status = status if status in {"completed", "failed", "timeout",
                                          "completed_with_fallback"} else "unknown"
            if status != "completed" and not metrics["failures"]:
                metrics["failures"].append({"stage": "pipeline", "usage_unknown": True})
            draft = analysis_result.get("final_written_section_markdown")
            fallback = (status != "completed" or any(
                analysis_result.get(key) is True for key in
                ("fallback", "fallback_used", "emergency_fallback")))
            if analysis_result.get("finish_reason") not in (None, "stop"):
                fallback = True
            if isinstance(draft, str) and draft.strip() == payload.resume_text.strip():
                fallback = True
            # Run structure checks before any correction requests; do not use the
            # old numeric verdict, which pertains to a different document.
            preflight = publication_gate.decide(
                resume_markdown=draft if isinstance(draft, str) else None,
                metrics_support={"verdict": "PASS (tier A only)", "checked": True},
                fallback_used=fallback, finish_reason=analysis_result.get("finish_reason"),
                approved_entities=[])
            if preflight.blocked():
                # The gate sees the original aliases but its output is never
                # forwarded wholesale: unknown nested fields may hold drafts.
                publication_gate.apply_to_payload(analysis_result, preflight)
                skills = analysis_result.get("aggregate_skills")
                safe_skills = ([skill for skill in skills
                                if isinstance(skill, str) and skill.strip()
                                and len(skill) <= 200
                                and skill.lower() in payload.resume_text.lower()]
                               if isinstance(skills, list) else None)
                _w_mode, _w_policy = _mode_from_payload(payload)
                return _withhold_correction(reason=preflight.reason_codes[0],
                    pipeline_status=status, metrics=metrics, start_time=start_time,
                    decision=preflight, aggregate_skills=safe_skills,
                    skill_inference_mode=_w_mode,
                    skill_inference_policy_version=_w_policy)
            try:
                from approved_resume import export_payload
                export_payload(draft)
            except (ValueError, TypeError):
                return _withhold_correction(reason="render_contract_ambiguous",
                    pipeline_status=status, metrics=metrics, start_time=start_time)
            if not api_keys:
                return _withhold_correction(reason="provider_unavailable",
                    pipeline_status=status, metrics=metrics, start_time=start_time)

            # The orchestrator's clock started before the pipeline ran, so the
            # correction deadline is what remains of THIS route's upstream budget.
            pipeline_elapsed = time.time() - start_time
            upstream_timeout = _upstream_timeout_seconds(request)
            correction_budget = correction_budget_seconds(pipeline_elapsed, upstream_timeout)
            correlation_id = _join_correlation_id(request, request_id)
            log_event(logger, {"event": "correction.budget", "correlation_id": correlation_id,
                               "pipeline_elapsed_ms": int(pipeline_elapsed * 1000),
                               "budget_ms": int((correction_budget or 0) * 1000),
                               "upstream_timeout_ms": int(upstream_timeout * 1000),
                               "reserve_ms": int(UPSTREAM_RESPONSE_RESERVE_SECONDS * 1000),
                               "max_deadline_ms": int(MAX_CORRECTION_DEADLINE_SECONDS * 1000)})
            # Opt-in, synthetic-only, access-controlled capture of this live exchange.
            # ``live_capture.start`` returns None unless an operator enabled it, so the
            # default request path pays nothing for this.
            live_capture_record = live_capture.start(correlation_id)

            def finish_capture(*, correction_result=None, decision=None, reason=None):
                """Persist the capture. Never raises and never alters the response."""
                if live_capture_record is None:
                    return None
                try:
                    if correction_result is not None:
                        live_capture_record.record_review({
                            "status": correction_result.status,
                            "reason": correction_result.reason,
                            "review_calls": correction_result.review_calls,
                            "repair_calls": correction_result.repair_calls,
                            # Restoration is the phase this capture exists to
                            # explain, so its count is recorded explicitly. It was
                            # previously omitted, which made "restored nothing" and
                            # "never asked to restore" indistinguishable.
                            "restore_calls": getattr(correction_result, "restore_calls", None),
                            "commentary": correction_result.commentary,
                            # Numeric coverage, not just its type: the earlier
                            # `{TypeName: True}` form discarded every count needed to
                            # tell a near-complete draft from an empty one.
                            "coverage": (None if correction_result.coverage is None else {
                                "level": correction_result.coverage.level,
                                "line_count": correction_result.coverage.line_count,
                                "claim_count": correction_result.coverage.claim_count,
                                "supported_count": correction_result.coverage.supported_count,
                                "structural_count": correction_result.coverage.structural_count,
                            }),
                        })
                    if decision is not None:
                        live_capture_record.record_gate(getattr(decision, "reason_codes", None))
                    live_capture_record.record_timing(
                        reason=reason,
                        pipeline_elapsed_ms=int(pipeline_elapsed * 1000),
                        budget_ms=int((correction_budget or 0) * 1000),
                        upstream_timeout_ms=int(upstream_timeout * 1000),
                        total_elapsed_ms=int((time.time() - start_time) * 1000))
                    path = live_capture_record.write()
                    logger.info("correction.capture", extra={
                        "request_id": correlation_id,
                        **live_capture.summarise(live_capture_record),
                        "written": bool(path)})
                    return path
                except Exception:
                    # Diagnostics must never break a user's request.
                    logger.warning("correction.capture.failed",
                                   extra={"request_id": correlation_id})
                    return None

            if correction_budget is None:
                # Not enough upstream time remains for one provider call; send nothing.
                finish_capture(reason="pipeline_budget_exhausted")
                return _withhold_correction(reason="pipeline_budget_exhausted",
                    pipeline_status=status, metrics=metrics, start_time=start_time)

            # Validated, mode-permitted bridges (Phase B), read from the pipeline
            # result rather than from the request body: a client cannot nominate
            # its own exception. In Conservative mode this list is empty.
            _permitted_bridges = [
                b for b in ((analysis_result or {}).get("skill_bridges") or [])
                if isinstance(b, dict) and b.get("skill")
            ]
            try:
                correction, provider = await _run_source_correction(
                    draft, source_documents_snapshot, api_keys[0], request,
                    timeout_seconds=correction_budget,
                    observer=_correction_observer(correlation_id),
                    capture=live_capture_record,
                    permitted_bridges=_permitted_bridges)
            except Exception:
                metrics["failures"].append({"stage": "source_correction", "usage_unknown": True})
                finish_capture(reason="provider_error")
                return _withhold_correction(reason="provider_error", pipeline_status=status,
                    metrics=metrics, start_time=start_time)
            if correction is None or provider is None:
                # Corresponding redaction could not be proven; nothing was sent.
                finish_capture(reason="privacy_redaction_failed")
                return _withhold_correction(reason="privacy_redaction_failed",
                    pipeline_status=status, metrics=metrics, start_time=start_time)
            metrics = _correction_usage(pipeline_metrics, provider)
            if correction.status != "approved" or not isinstance(correction.markdown, str):
                allowed_reasons = {"invalid_input", "timeout", "malformed_source_inventory",
                    "source_completeness_uncertain", "no_required_source_claims",
                    "provider_error", "no_supported_claims", "source_facts_missing",
                    "unresolved_claims", "malformed_output", "privacy_review_required",
                    "privacy_redaction_failed", "attribution_conflict",
                    # The reviewer answered incompletely (lines left unjudged) or
                    # returned an edit that changed nothing; both are distinct from
                    # malformed_output and reported so the cause is identifiable.
                    "incomplete_review", "no_op_repair",
                    # Every eligible upstream endpoint was saturated. Transient and
                    # unrelated to the request, so it must not be reported as a
                    # generic correction failure.
                    "provider_rate_limited",
                    # A retryable 5xx outlived the retry allowance, and a request
                    # the provider refused outright (auth/schema/model).
                    "provider_unavailable", "provider_rejected_request",
                    "timeout_source_inventory", "timeout_initial_review",
                    "timeout_repair", "timeout_post_patch_review",
                    "provider_call_timeout_source_inventory",
                    "provider_call_timeout_initial_review",
                    "provider_call_timeout_repair",
                    "provider_call_timeout_post_patch_review"}
                reason = correction.reason if correction.reason in allowed_reasons else "correction_failed"
                finish_capture(correction_result=correction, reason=reason)
                _w_mode, _w_policy = _mode_from_payload(payload)
                return _withhold_correction(reason=reason, pipeline_status=status,
                    metrics=metrics, start_time=start_time,
                    skill_inference_mode=_w_mode,
                    skill_inference_policy_version=_w_policy)

            final_markdown = correction.markdown
            support = tier_a_verifier.verify_metrics(final_markdown, payload.resume_text)
            missing_entities = unsupported_entities(final_markdown, payload.resume_text)
            decision = publication_gate.decide(
                resume_markdown=final_markdown, metrics_support=support,
                fallback_used=False, finish_reason=analysis_result.get("finish_reason"),
                required_facts_missing=missing_entities,
                approved_entities=_approved_entities_from_payload(None, None, source_snapshot))
            if decision.blocked():
                finish_capture(correction_result=correction, decision=decision,
                               reason=decision.reason_codes[0])
                _w_mode, _w_policy = _mode_from_payload(payload)
                return _withhold_correction(reason=decision.reason_codes[0],
                    pipeline_status=status, metrics=metrics, start_time=start_time,
                    decision=decision,
                    skill_inference_mode=_w_mode,
                    skill_inference_policy_version=_w_policy)

            # Do not forward any pipeline draft aliases, notes, or provenance.
            # Receipt creation parses this exact markdown into the sole export object.
            #
            # Policy state is carried explicitly because the render object does not
            # represent it: without these two fields the approval would say nothing
            # about which inference mode produced the document or which bridges
            # were permitted, and receipt v5 could not bind them.
            output = publication_gate.apply_to_payload({
                "final_written_section_markdown": final_markdown,
                "final_written_section": final_markdown,
                "rewritten_resume": final_markdown,
                "metrics_support": support,
                "pipeline_status": status,
                "correction_status": "approved",
                "correction_commentary": correction.commentary,
                "run_metrics": metrics,
                "processing_time_seconds": time.time() - start_time,
                "skill_inference_mode": (analysis_result or {}).get(
                    "skill_inference_mode"),
                "skill_inference_policy_version": (analysis_result or {}).get(
                    "skill_inference_policy_version"),
                "skill_bridges": _surviving_bridges(
                    (analysis_result or {}).get("skill_bridges"), final_markdown),
            }, decision)
            # A free-trial visitor is a legitimate owner: they spent credits for
            # this resume. Binding only to a signed-in account discarded an
            # already-corrected document purely because the user was logged out
            # (measured: 1 of 20 runs, 2026-11-08).
            owner, owner_is_account = resolve_publication_owner(request)
            output = _issue_publication_receipt(output, owner=owner)
            if not owner or not output.get("publication_receipt") or not output.get("approved_export_payload"):
                finish_capture(correction_result=correction, decision=decision,
                               reason="receipt_unavailable")
                return _withhold_correction(reason="receipt_unavailable",
                    pipeline_status=status, metrics=metrics, start_time=start_time)
            # Tells the client which kind of approval it holds, so it can ask for a
            # sign-in before export (account-bound) or not (device-bound).
            output["publication_owner_kind"] = (
                "account" if owner_is_account else "guest")
            # The compatibility experience alias must derive from the signed
            # render payload, never from the earlier Drafter output.
            output["experiences"] = [
                {"title_line": item.get("title", ""),
                 "snippet": "\n".join(item.get("bullets", []))}
                for item in output["approved_export_payload"]["experience"]
            ]
            finish_capture(correction_result=correction, decision=decision, reason="approved")
            return AnalysisResponse(**output)

        # Step 2: Run Generation
        logger.info("run_generation.start", extra={"request_id": request_id})
        t0 = time.time()
        generation_result = await imaginator_flow_module.run_generation_async(
            analysis_json=analysis_result,
            job_ad=payload.job_ad,
            openrouter_api_keys=api_keys,
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
        logger.info(
            "run_generation.end",
            extra={"request_id": request_id, "duration_ms": generation_duration_ms},
        )
        RUN_METRICS.setdefault("durations_ms", {})[
            "generation_ms"
        ] = generation_duration_ms

        # Step 3: Run Criticism
        logger.info("run_criticism.start", extra={"request_id": request_id})
        t0 = time.time()
        criticism_result = await run_criticism_async(
            generated_text=generation_result,
            job_ad=payload.job_ad,
            openrouter_api_keys=api_keys,
        )
        criticism_duration_ms = int((time.time() - t0) * 1000)
        logger.info(
            "run_criticism.end",
            extra={"request_id": request_id, "duration_ms": criticism_duration_ms},
        )
        RUN_METRICS.setdefault("durations_ms", {})[
            "criticism_ms"
        ] = criticism_duration_ms

        # Normalize criticism_result to dict
        if isinstance(criticism_result, str):
            try:
                from imaginator_flow import ensure_json_dict

                criticism_result = ensure_json_dict(criticism_result, "critique")
            except Exception:
                criticism_result = {
                    "suggested_experiences": {
                        "bridging_gaps": [],
                        "metric_improvements": [],
                    }
                }

        # Step 4: Synthesis — integrate critique into final text
        logger.info(
            "run_synthesis.start",
            extra={
                "request_id": request_id,
                "final_writer": getattr(settings, "FINAL_WRITER_PROVIDER", None),
            },
        )
        t0 = time.time()
        synthesis_result = await run_synthesis_async(
            generated_text=generation_result,
            critique_json=criticism_result,
            job_ad=payload.job_ad,
            openrouter_api_keys=api_keys,
            analysis_result=analysis_result,  # FIX: Changed from analysis_for_provenance to analysis_result
            final_writer=getattr(
                settings, "FINAL_WRITER_PROVIDER", None
            ),  # Use configured provider (safe getattr)
        )
        synthesis_duration_ms = int((time.time() - t0) * 1000)
        logger.info(
            "run_synthesis.end",
            extra={"request_id": request_id, "duration_ms": synthesis_duration_ms},
        )
        RUN_METRICS.setdefault("durations_ms", {})[
            "synthesis_ms"
        ] = synthesis_duration_ms

        # Extract structured fields from synthesis result
        if isinstance(synthesis_result, dict):
            final_written_section = synthesis_result.get(
                "final_written_section", synthesis_result
            )
            final_markdown = synthesis_result.get("final_written_section_markdown")
            final_provenance = synthesis_result.get(
                "final_written_section_provenance", []
            )
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

        # ── Publication decision (single authoritative gate) ─────────────
        # Same module as the fast-track path. Migration note: this used to be a
        # second, hand-rolled copy of the containment rules; two copies meant
        # /api/analysis/multi-file could be forgotten — and it was.
        _support = (analysis_result or {}).get("metrics_support") or {}
        if not _support:
            _support = ((analysis_result or {}).get("final_output") or {}).get(
                "metrics_support") or {}

        _decision = publication_gate.decide(
            resume_markdown=final_markdown,
            metrics_support=_support,
            fallback_used=(analysis_result or {}).get("pipeline_status") in {
                "failed", "completed_with_fallback", "timeout"},
            required_facts_missing=unsupported_entities(
                final_markdown, payload.resume_text),
            approved_entities=_approved_entities_from_payload(
                analysis_result, body, source_snapshot),
        )

        output = publication_gate.apply_to_payload(
            {
                **analysis_result,
                **criticism_result,
                "final_written_section": final_written_section,
                "final_written_section_markdown": final_markdown,
                "final_written_section_provenance": final_provenance,
                "critique_score": critique_score,
                "run_metrics": RUN_METRICS.copy(),
                "processing_time_seconds": time.time() - start_time,
            },
            _decision,
        )
        # Same resolution as the fast-track path: a free-trial visitor's approval
        # is bound to their stable device identity rather than discarded.
        _owner, _owner_is_account = resolve_publication_owner(request)
        output = _issue_publication_receipt(output, owner=_owner)
        if _owner:
            output["publication_owner_kind"] = (
                "account" if _owner_is_account else "guest")

        ## Validate output schema
        validate_output_schema(output)

        processing_time = time.time() - start_time
        logger.info(
            "analyze.request.completed",
            extra={
                "request_id": request_id,
                "processing_time_seconds": processing_time,
                "run_metrics": RUN_METRICS.copy(),
            },
        )
        return AnalysisResponse(**output)

    except HTTPException as e:
        raise e
    except Exception:
        # Exceptions can contain private draft, source, or provider data.
        logger.error("analyze.request.failed", extra={"request_id": request_id})
        known_metrics = locals().get("metrics") or _correction_usage(RUN_METRICS)
        known_metrics["failures"].append({"stage": "pipeline", "usage_unknown": True})
        return _withhold_correction(reason="analysis_failed", pipeline_status="failed",
                                   metrics=known_metrics, start_time=start_time)


class ResearchRequest(BaseModel):
    """Job-ad-only Researcher fork (no structured profile)."""

    job_ad: str = Field(..., min_length=1, description="Target job description text")


EMPTY_RESEARCH_FALLBACK = {
    "implied_metrics": [],
    "domain_vocab": [],
    "implied_skills": [],
    "work_archetypes": [],
    "fallback": True,
}


def _research_partial_patch(research_data: Dict[str, Any]) -> Dict[str, Any]:
    """Small preview patch of research_data for SSE research_partial events."""
    return {
        "domain_vocab_preview": list((research_data.get("domain_vocab") or [])[:8]),
        "implied_metrics_count": len(research_data.get("implied_metrics") or []),
        "implied_skills_count": len(research_data.get("implied_skills") or []),
    }


async def _research_job_ad_impl(
    payload: ResearchRequest,
    request: Request,
    on_progress: Optional[Callable[[str, Dict[str, Any]], None]] = None,
):
    """
    Shared /research implementation (JSON + SSE). Strict timeout → empty
    research_data (never hangs Drafter). ``on_progress(event, data)`` is only
    used by the SSE path; the JSON path passes None.
    """
    request_id = getattr(request.state, "request_id", None)
    timeout_s = float(TIMEOUTS.get("researcher", 45))
    api_keys = [
        key
        for key in [settings.openrouter_api_key_1, settings.openrouter_api_key_2]
        if key
    ] or [k for k in OPENROUTER_API_KEYS if k]

    if not api_keys:
        logger.warning("research.no_openrouter_keys", extra={"request_id": request_id})
        return JSONResponse(
            {
                "research_data": {**EMPTY_RESEARCH_FALLBACK, "error": "no_openrouter_keys"},
                "status": "fallback",
            }
        )

    llm_client = LLMClientAdapter(
        api_key=api_keys[0],
        fallback_keys=api_keys[1:] if len(api_keys) > 1 else [],
    )
    researcher = Researcher(llm_client)
    t0 = time.time()
    if on_progress:
        on_progress(
            "stage",
            {"stage": "researcher", "state": "started", "message": "Researcher started"},
        )
    try:
        research_data = await asyncio.wait_for(
            researcher.analyze(payload.job_ad, experiences=None),
            timeout=timeout_s,
        )
        status = "completed" if "error" not in research_data else "failed"
        if on_progress:
            on_progress("research_partial", _research_partial_patch(research_data))
    except asyncio.TimeoutError:
        logger.warning(
            "research.timeout",
            extra={"request_id": request_id, "timeout_s": timeout_s},
        )
        research_data = {
            **EMPTY_RESEARCH_FALLBACK,
            "error": f"timeout_after_{timeout_s}s",
        }
        status = "timeout"
        if on_progress:
            on_progress(
                "stage",
                {"stage": "researcher", "state": "fallback", "reason": "timeout"},
            )
    except Exception as exc:
        logger.exception("research.failed", extra={"request_id": request_id})
        research_data = {**EMPTY_RESEARCH_FALLBACK, "error": str(exc)[:300]}
        status = "error"
        if on_progress:
            on_progress(
                "stage",
                {"stage": "researcher", "state": "fallback", "reason": "error"},
            )

    duration_ms = int((time.time() - t0) * 1000)
    logger.info(
        "research.done",
        extra={
            "request_id": request_id,
            "status": status,
            "duration_ms": duration_ms,
            "metrics": len(research_data.get("implied_metrics") or []),
        },
    )
    return {
        "research_data": research_data,
        "status": status,
        "duration_ms": duration_ms,
        "forked": True,
        "run_metrics": _research_run_metrics(llm_client),
    }


def _research_run_metrics(llm_client: Any) -> Optional[Dict[str, Any]]:
    """Usage for the orchestrator's cost log; never allowed to fail /research."""
    try:
        usage = llm_client.get_usage_stats()
        usage["calls"] = [{**c, "stage": "researcher"} for c in usage.get("calls", [])]
        return usage
    except Exception:
        return None


def _surviving_bridges(bridges, final_markdown):
    """Only the bridges whose label actually survived into the approved text.

    A proposal that never reached the document — or was removed during
    correction — must not appear in the signed policy state. Binding an unused
    bridge would overstate what was authorised, and would make the state differ
    between two documents that are in fact identical.
    """
    if not isinstance(bridges, list):
        return []
    text = (final_markdown or "").casefold()
    import re as _re

    surviving = []
    for bridge in bridges:
        if not isinstance(bridge, dict):
            continue
        skill = bridge.get("skill")
        if not isinstance(skill, str) or not skill.strip():
            continue
        label = " ".join(skill.casefold().split())
        if not label:
            continue
        # Word-bounded: a label must appear as itself, not inside another word.
        if _re.search(r"(?<![a-z0-9])" + _re.escape(label) + r"(?![a-z0-9])", text):
            surviving.append(bridge)
    return surviving


def _policy_state_from_output(output):
    """Policy state to bind into the receipt, or None for a source-only result.

    Reads ONLY server-resolved fields: the effective mode, the policy version,
    and the validated bridge records. A client cannot contribute any of these,
    and an empty bridge list on an imaginative result is still meaningful (it
    records that no inference was used), so the mode alone is enough to bind.
    """
    if not isinstance(output, dict):
        return None
    from skill_inference import SkillInferenceMode, normalize_mode

    mode_value = output.get("skill_inference_mode")
    if not mode_value:
        return None
    try:
        mode = normalize_mode(mode_value)
    except Exception:  # noqa: BLE001 - never fail approval on a bad mode string
        return None
    bridges = output.get("skill_bridges") or []
    surviving = []
    for bridge in bridges:
        if not isinstance(bridge, dict):
            continue
        if not bridge.get("id") or not bridge.get("skill"):
            continue
        surviving.append({
            "id": bridge.get("id"),
            "skill": bridge.get("skill"),
            "origin": bridge.get("origin"),
            "source_id": bridge.get("source_id"),
            "policy_version": bridge.get("policy_version"),
        })
    if mode is SkillInferenceMode.CONSERVATIVE and surviving:
        # Impossible by construction; refuse rather than sign an incoherent
        # approval.
        return None
    return {
        "skill_inference_mode": mode.value,
        "skill_inference_policy_version": output.get(
            "skill_inference_policy_version"
        ),
        "surviving_bridges": surviving,
    }


def _issue_publication_receipt(output, owner=None, identity=None):
    """Attach a signed approval receipt to a publishable result.

    Export verifies this against the content it is about to render, so a caller
    cannot forge `metrics_support` or edit the resume between approval and
    export. Non-publishable results get no receipt.

    The receipt binds both approved Markdown and the exact career object the
    renderer consumes. The same object is returned as approved_export_payload,
    eliminating cross-parser claim-set alternatives at export time.

    ``identity`` is the profile header (name/contact) that the caller supplied
    with the analysis, if any. It is signed as part of the render content so a
    later export cannot inject identity that was never approved. When no
    identity was supplied, the signed header stays empty and export must send
    the same empty header — the client may not invent a default name.
    """
    try:
        if not isinstance(output, dict):
            return output
        support = output.get("metrics_support") or {}
        verdict = str(support.get("verdict") or "")
        if (output.get("requires_review") or output.get("processing_status") == "needs_review"
                or verdict != publication_gate.Verdict.PASS_TIER_A.value):
            output["publication_receipt"] = None
            output["approved_export_payload"] = None
            return output
        md = output.get("final_written_section_markdown")
        from approved_resume import export_payload
        try:
            approved_payload = export_payload(md)
        except ValueError:
            output["publication_receipt"] = None
            output["approved_export_payload"] = None
            output["requires_review"] = True
            output["processing_status"] = "needs_review"
            output["review_reasons"] = ["render_contract_ambiguous"]
            for field in publication_gate.DELIVERABLE_FIELDS:
                output[field] = None
            return output
        if isinstance(identity, dict):
            for field in ("name", "location", "phone", "email", "linkedin", "github"):
                value = identity.get(field)
                if isinstance(value, str) and value.strip():
                    approved_payload[field] = value.strip()
        # Skill-inference policy state (mode + surviving bridges). Passing it
        # upgrades the receipt to v5, which binds a fingerprint of that state;
        # a v4 receipt therefore cannot later authorise an inferred skill.
        # Source-only approvals pass nothing and stay v4.
        policy_state = _policy_state_from_output(output)
        receipt = publication_gate.issue_receipt(
            markdown=md, verdict=verdict, owner=owner,
            approved_payload=approved_payload,
            policy_state=policy_state)
        if policy_state:
            output["skill_inference_policy_state"] = policy_state
        output["publication_receipt"] = receipt
        output["approved_export_payload"] = approved_payload if receipt else None
        if receipt is None:
            # Signing unavailable: do not claim approval we cannot vouch for.
            output["verification_coverage"] = "none"
            output["requires_review"] = True
            output["processing_status"] = "needs_review"
            output["review_reasons"] = list(
                output.get("review_reasons") or []) + [
                    "receipt_signing_unavailable"]
        return output
    except Exception:  # noqa: BLE001
        if isinstance(output, dict):
            output["publication_receipt"] = None
        return output


def _multifile_llm_client(api_keys):
    """Client for the multi-file route's generation / criticism / synthesis stages.

    Returns an `LLMClientAdapter`, or None when no key is configured (the stages
    then use their own ambient path and the gate blocks as unverified). Tests
    replace `LLMClientAdapter` in this module's namespace with a stage-aware
    stub, which is why this indirection exists rather than constructing the
    adapter at each call site: one seam, three stages.
    """
    if not api_keys:
        return None
    return LLMClientAdapter(
        api_key=api_keys[0],
        fallback_keys=api_keys[1:] if len(api_keys) > 1 else [],
    )


def _markdown_from_export_payload(payload):
    """Reconstruct a comparable markdown proxy from a structured export payload.

    The gate needs *text* to run its structural checks. Export payloads are
    structured (experience/projects/education arrays). Building an equivalent
    markdown view lets the same checks apply without a second code path.

    Returns None when the payload carries no resume content at all, which the
    gate treats as EMPTY_OUTPUT and blocks.
    """
    if not isinstance(payload, dict):
        return None

    parts = []

    exp = payload.get("experience") or payload.get("experiences") or []
    if isinstance(exp, list) and exp:
        parts.append("## Professional Experience")
        for e in exp:
            if not isinstance(e, dict):
                continue
            title = e.get("title") or e.get("role") or ""
            company = e.get("company") or ""
            dates = " ".join(
                x for x in (e.get("start_date"), e.get("end_date")) if x)
            line = " ".join(x for x in (title, company, dates) if x)
            if line:
                parts.append(line)
            for b in (e.get("bullets") or []):
                if isinstance(b, str) and b.strip():
                    parts.append(f"- {b}")

    projects = payload.get("projects") or []
    if isinstance(projects, list) and projects:
        parts.append("## Projects")
        for pr in projects:
            if not isinstance(pr, dict):
                continue
            nm = pr.get("title") or pr.get("name") or ""
            if nm:
                parts.append(f"**{nm}**")
            for b in (pr.get("bullets") or []):
                if isinstance(b, str) and b.strip():
                    parts.append(f"- {b}")

    edu = payload.get("education") or []
    if isinstance(edu, list) and edu:
        parts.append("## Education")
        for item in edu:
            if isinstance(item, str) and item.strip():
                parts.append(f"- {item}")
            elif isinstance(item, dict):
                line = " ".join(
                    str(v) for v in (item.get("degree"), item.get("institution"))
                    if v)
                if line:
                    parts.append(f"- {line}")

    certs = payload.get("certifications") or []
    if isinstance(certs, list) and certs:
        parts.append("## Certifications")
        for c in certs:
            if isinstance(c, str) and c.strip():
                parts.append(f"- {c}")

    summary = payload.get("summary")
    if isinstance(summary, str) and summary.strip():
        parts.append("## Summary")
        parts.append(summary)

    return "\n".join(parts) if parts else None


def _approved_entities_from_payload(analysis_result, payload=None,
                                    source_snapshot=None):
    """Compose the approved-entity allow-list for the publication gate.

    CRITICAL: evidence must come from SOURCE / CONFIRMED CANDIDATE DATA only —
    never from generated output.

    The previous version read `result.get("projects")` / `experiences` /
    `education` from `analysis_result`. That object contains the GENERATED
    resume. An invented project therefore appeared in the evidence list and
    approved itself: the editor writes "**Automated Reporting Dashboard**", the
    allow-list is built from that same text, and the gate finds it "allowed".
    Measured: this is exactly how entity checking could be defeated.

    Sources, in order:
      1. `source_snapshot` — an immutable tuple captured at INGRESS from the
         request body and the uploaded text, before any generation ran. Every
         route passes this, so evidence cannot be influenced by the pipeline.
      2. `payload` — what the CALLER submitted (confirmed candidate data).
      3. `analysis_result["source_*"]` — legacy echo fields, used only when no
         snapshot exists (direct callers). Real routes never take this path,
         because a pipeline that ever wrote `source_projects` from generated
         content would otherwise self-approve (owner item 6).
      4. Nothing else. Generated projects cannot self-approve.

    Returns a list (possibly empty) or None when no source data exists at all.
    """
    entities = []

    def _add(value, keys=("name", "company", "institution", "title",
                          "canonical_title", "school", "issuer", "degree")):
        if not value:
            return
        if isinstance(value, str):
            v = value.strip()
            if v:
                entities.append(v)
        elif isinstance(value, dict):
            for k in keys:
                _add(value.get(k), keys)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _add(item, keys)

    saw_source = False

    # ── 0. immutable ingress snapshot (authoritative; owner item 6) ─────
    # A tuple, so a later stage cannot mutate the evidence in place. Presence
    # of the snapshot ends the search: nothing downstream may add to it.
    if source_snapshot is not None:
        for entry in tuple(source_snapshot):
            _add(entry)
        return entities

    # ── 1. caller-supplied candidate data (authoritative) ───────────────
    #
    # `payload` is what the CALLER submitted — confirmed candidate data.
    # `analysis_result` is the pipeline output and may contain generated
    # projects, so it is only consulted for explicit `source_*` echo fields.
    #
    # The caller may arrive as a pydantic model; normalise it first.
    if payload is not None and not isinstance(payload, dict):
        payload = getattr(payload, "__dict__", None) or getattr(
            payload, "model_dump", lambda: None)()
    if isinstance(payload, dict):
        for key in ("projects", "education", "certifications", "experience",
                    "experiences", "skills"):
            if key in payload:
                saw_source = True
                _add(payload.get(key))

    # ── 2. explicit SOURCE echo fields on the result (legacy) ───────────
    # Names must indicate provenance. Anything else risks generated content.
    if isinstance(analysis_result, dict):
        for key in ("source_projects", "source_education",
                    "source_certifications", "source_experiences",
                    "source_experience", "confirmed_projects",
                    "parsed_projects", "original_projects",
                    "original_experiences"):
            if key in analysis_result:
                saw_source = True
                _add(analysis_result.get(key))

    # ── 3. de-duplicate, preserve order ─────────────────────────────────
    seen = set()
    out = []
    for e in entities:
        key = e.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(e)

    # Empty list is meaningful: source data was examined and yielded nothing
    # (a candidate with no projects). None means we never saw source data.
    return out if saw_source else None


def _snapshot_source_evidence(payload=None, resume_text=None) -> tuple:
    """Capture approved-entity evidence at ingress, before generation runs.

    Owner item 6: approved evidence must be immutable and independent of
    generated output. Both properties come from WHERE this runs — at request
    entry, over the submitted structured fields and the uploaded resume text.
    The returned tuple is then passed explicitly to every gate call, so no
    later stage can substitute its own (possibly generated) view of the
    candidate's employers, projects, education or certifications.

    Structured sections are re-derived from the resume text when the caller did
    not supply them, using the same conservative section parser the pipeline
    uses, so a text-only upload still has real evidence to check against.
    """
    collected = []
    saw_source = False

    if isinstance(payload, dict):
        for key in ("projects", "education", "certifications", "experience",
                    "experiences", "skills"):
            if key in payload:
                saw_source = True
                collected.append(payload.get(key))
    elif payload is not None:
        as_dict = (getattr(payload, "__dict__", None)
                   or (payload.model_dump() if hasattr(payload, "model_dump")
                       else None))
        if isinstance(as_dict, dict):
            for key in ("projects", "education", "certifications", "experience",
                        "experiences", "skills"):
                if key in as_dict:
                    saw_source = True
                    collected.append(as_dict.get(key))

    if isinstance(resume_text, str) and resume_text.strip():
        saw_source = True
        try:
            from imaginator_new_integration import (
                _extract_structured_sections_from_resume,
            )
            inferred = _extract_structured_sections_from_resume(resume_text)
        except Exception:  # noqa: BLE001 - parsing is best effort
            inferred = {}
        for key in ("projects", "education", "certifications"):
            if inferred.get(key):
                collected.append(inferred[key])
        # Employers named in the uploaded text are evidence too.
        try:
            from imaginator_flow import parse_experiences
            collected.append(parse_experiences(resume_text) or [])
        except Exception:  # noqa: BLE001
            pass

    if not saw_source:
        return tuple()

    keys = ("name", "company", "institution", "title", "canonical_title",
            "school", "issuer", "degree", "role")
    flat = []

    def _flatten(value):
        if not value:
            return
        if isinstance(value, str):
            if value.strip():
                flat.append(value.strip())
        elif isinstance(value, dict):
            for k in keys:
                _flatten(value.get(k))
        elif isinstance(value, (list, tuple)):
            for item in value:
                _flatten(item)

    for item in collected:
        _flatten(item)

    seen = set()
    unique = []
    for value in flat:
        lowered = value.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique.append(value)
    return tuple(unique)


@app.post("/research", dependencies=[Depends(get_api_key)])
async def research_job_ad(
    payload: ResearchRequest,
    request: Request,
    api_key: str = Depends(get_api_key),
):
    """
    Dual-mode job-ad-only Researcher for orch fork-join.
    - Default: JSON (unchanged)
    - Accept: text/event-stream → ``stage`` (+ heartbeats, ``research_partial``)
      then ``result`` with the same body as the JSON response.
    Strict timeout → empty research_data (never hangs Drafter).
    """
    accept = (request.headers.get("accept") or "").lower()
    if "text/event-stream" not in accept:
        return await _research_job_ad_impl(payload, request)

    queue: asyncio.Queue = asyncio.Queue()
    heartbeat_s = max(
        0.05, float(os.getenv("IMAG_RESEARCH_HEARTBEAT_S", "5") or 5)
    )

    def on_progress(event: str, data: Dict[str, Any]) -> None:
        # Runs on the event loop (impl is async) → ordered before the result put.
        try:
            queue.put_nowait((event, data))
        except Exception:
            pass

    async def produce() -> None:
        try:
            result = await _research_job_ad_impl(payload, request, on_progress)
            if isinstance(result, JSONResponse):
                try:
                    result = json.loads(result.body.decode("utf-8")) if result.body else {}
                except Exception:
                    result = {"status": "error"}
            await queue.put(("result", result))
        except Exception as exc:
            await queue.put(("error", {"message": str(exc)[:300]}))

    async def event_stream():
        yield (
            "event: stage\n"
            f"data: {json.dumps({'stage': 'researcher', 'state': 'queued'})}\n\n"
        )
        task = asyncio.create_task(produce())
        t0 = time.time()
        try:
            while True:
                try:
                    kind, data = await asyncio.wait_for(queue.get(), timeout=heartbeat_s)
                except asyncio.TimeoutError:
                    hb = {
                        "stage": "researcher",
                        "state": "running",
                        "elapsed_ms": int((time.time() - t0) * 1000),
                    }
                    yield f"event: stage\ndata: {json.dumps(hb)}\n\n"
                    continue
                yield f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"
                if kind in ("result", "error"):
                    break
        finally:
            await task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/analyze", dependencies=[Depends(get_api_key)])
async def analyze_resume(
    payload: AnalysisRequest,
    request: Request,
    api_key: str = Depends(get_api_key),
):
    """
    Dual-mode /analyze (C3):
    - Default: JSON AnalysisResponse (unchanged)
    - Accept: text/event-stream → approved final content then final result event
    """
    accept = (request.headers.get("accept") or "").lower()
    wants_sse = "text/event-stream" in accept

    if not wants_sse:
        return await _analyze_resume_impl(payload, request, api_key)

    queue: asyncio.Queue = asyncio.Queue()

    # ── SSE containment (owner item 5) ──────────────────────────────────
    # Approval covers the final Markdown, never earlier draft/rewrite/reasoning
    # tokens. Discard raw callbacks instead of buffering them for later replay.
    def on_token(field: str, delta: str) -> None:
        pass

    async def produce() -> None:
        token = set_token_callback(on_token)
        try:
            result = await _analyze_resume_impl(payload, request, api_key)
            await queue.put(("result", result))
        except Exception:
            await queue.put(("error", "analysis_failed"))
        finally:
            reset_token_callback(token)

    async def event_stream():
        task = asyncio.create_task(produce())
        try:
            while True:
                kind, payload_data = await queue.get()
                if kind == "result":
                    if isinstance(payload_data, AnalysisResponse):
                        data = (
                            payload_data.model_dump()
                            if hasattr(payload_data, "model_dump")
                            else payload_data.dict()
                        )
                    elif isinstance(payload_data, JSONResponse):
                        # Error/partial JSONResponse — surface as error event
                        try:
                            body = payload_data.body
                            data = json.loads(body.decode("utf-8")) if body else {}
                        except Exception:
                            data = {"error": "analysis_failed"}
                        yield f"event: error\ndata: {json.dumps(data, default=str)}\n\n"
                        break
                    elif isinstance(payload_data, dict):
                        data = payload_data
                    else:
                        data = {"result": str(payload_data)}

                    # Emit only the content actually covered by final approval.
                    published = (
                        data.get("processing_status") == "completed"
                        and not data.get("requires_review")
                    )
                    if published:
                        final_markdown = data.get("final_written_section_markdown")
                        if isinstance(final_markdown, str) and final_markdown.strip():
                            yield (
                                "event: token\n"
                                f"data: {json.dumps({'field': 'rewrite', 'delta': final_markdown})}\n\n"
                            )
                    else:
                        # Emit no tokens at all for a blocked result. The client
                        # never sees a partial resume, so it cannot render one.
                        yield (
                            "event: withheld\n"
                            f"data: {json.dumps({'reasons': data.get('review_reasons') or [], 'message': data.get('review_message')}, default=str)}\n\n"
                        )
                    yield f"event: result\ndata: {json.dumps(data, default=str)}\n\n"
                    break
                elif kind == "error":
                    yield (
                        "event: error\n"
                        f"data: {json.dumps({'message': payload_data})}\n\n"
                    )
                    break
        finally:
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.post(
    "/api/analysis/multi-file",
    response_model=AnalysisResponse,
    dependencies=[Depends(get_api_key)],
)
async def analyze_multi_file(
    request: Request,
    files: List[UploadFile] = File(...),
    job_ad: Optional[str] = Form(default=None),
    api_key: str = Depends(get_api_key),
):
    start_time = time.time()

    # Multipart/form-data requests do not support request.json()
    request_id = getattr(request.state, "request_id", None)
    logger.info(
        "analyze_multi_file.request.incoming",
        extra={"request_id": request_id, "file_count": len(files)},
    )
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

        # ── Immutable source evidence, captured at INGRESS (owner item 6) ─
        # This route has no structured body (multipart only), so evidence comes
        # from the uploaded text alone — parsed before the pipeline runs, and
        # never re-derived from generated output afterwards.
        source_snapshot = _snapshot_source_evidence(None, resume_text)
        api_keys = [
            key
            for key in [settings.openrouter_api_key_1, settings.openrouter_api_key_2]
            if key
        ]
        RUN_METRICS.update(
            {
                "calls": [],
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "failures": [],
            }
        )
        logger.info("run_analysis.start", extra={"request_id": request_id})
        t0 = time.time()
        analysis_result = await run_analysis_async(
            resume_text=resume_text,
            job_ad=job_ad,
            extracted_skills_json=None,
            domain_insights_json=None,
            confidence_threshold=settings.confidence_threshold,
            openrouter_api_keys=api_keys,
            location=None,  # No location provided in multi-file upload
            # The legacy multipart route does not expose an inference-mode control,
            # so it runs the documented default. Passing it explicitly keeps the
            # cache identity honest: an omission here would hash as the default
            # anyway, but a future caller adding the control must not be able to
            # collide with an existing entry.
            skill_inference_mode="imaginative",
        )
        analysis_duration_ms = int((time.time() - t0) * 1000)
        logger.info(
            "run_analysis.end",
            extra={"request_id": request_id, "duration_ms": analysis_duration_ms},
        )
        RUN_METRICS.setdefault("durations_ms", {})["analysis_ms"] = analysis_duration_ms
        logger.info("run_generation.start", extra={"request_id": request_id})
        t0 = time.time()
        generation_result = await run_generation_async(
            analysis_json=analysis_result,
            job_ad=job_ad,
            openrouter_api_keys=api_keys,
            llm_client=_multifile_llm_client(api_keys),
        )
        generation_duration_ms = int((time.time() - t0) * 1000)
        logger.info(
            "run_generation.end",
            extra={"request_id": request_id, "duration_ms": generation_duration_ms},
        )
        RUN_METRICS.setdefault("durations_ms", {})[
            "generation_ms"
        ] = generation_duration_ms
        logger.info("run_criticism.start", extra={"request_id": request_id})
        t0 = time.time()
        criticism_result = await run_criticism_async(
            generated_text=generation_result,
            job_ad=job_ad,
            openrouter_api_keys=api_keys,
            llm_client=_multifile_llm_client(api_keys),
        )
        criticism_duration_ms = int((time.time() - t0) * 1000)
        logger.info(
            "run_criticism.end",
            extra={"request_id": request_id, "duration_ms": criticism_duration_ms},
        )
        RUN_METRICS.setdefault("durations_ms", {})[
            "criticism_ms"
        ] = criticism_duration_ms
        if isinstance(criticism_result, str):
            try:
                from imaginator_flow import ensure_json_dict

                criticism_result = ensure_json_dict(criticism_result, "critique")
            except Exception:
                criticism_result = {
                    "suggested_experiences": {
                        "bridging_gaps": [],
                        "metric_improvements": [],
                    }
                }
        logger.info(
            "run_synthesis.start",
            extra={
                "request_id": request_id,
                "final_writer": getattr(settings, "FINAL_WRITER_PROVIDER", None),
            },
        )
        t0 = time.time()
        synthesis_result = await run_synthesis_async(
            generated_text=generation_result,
            critique_json=criticism_result,
            job_ad=job_ad,
            openrouter_api_keys=api_keys,
            analysis_result=analysis_result,  # FIX: Changed from analysis_for_provenance to analysis_result
            final_writer=getattr(
                settings, "FINAL_WRITER_PROVIDER", None
            ),  # Use configured provider (safe getattr)
            llm_client=_multifile_llm_client(api_keys),
        )
        synthesis_duration_ms = int((time.time() - t0) * 1000)
        logger.info(
            "run_synthesis.end",
            extra={"request_id": request_id, "duration_ms": synthesis_duration_ms},
        )
        RUN_METRICS.setdefault("durations_ms", {})[
            "synthesis_ms"
        ] = synthesis_duration_ms

        # Extract structured fields from synthesis result
        if isinstance(synthesis_result, dict):
            final_written_section = synthesis_result.get(
                "final_written_section", synthesis_result
            )
            final_markdown = synthesis_result.get("final_written_section_markdown")
            final_provenance = synthesis_result.get(
                "final_written_section_provenance", []
            )
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

        # ── Tier-A verification for THIS route (owner item 1) ────────────
        # This route composes its final document from `run_synthesis_async`,
        # which never runs the editor — so `analysis_result` carries no
        # `metrics_support`. Previously that meant the gate could only ever
        # block here (`verification_missing`), and a legitimate multi-file run
        # was unpublishable. The acceptance condition is NOT relaxed: the same
        # Tier-A checker runs, against the ORIGINAL uploaded text as evidence.
        # If that check cannot run, the gate still blocks.
        # Earlier analysis PASS is not approval of a later synthesis. Always
        # verify THIS final document against immutable combined uploaded text.
        _mf_support = tier_a_verifier.verify_metrics(final_markdown, resume_text)

        _mf_decision = publication_gate.decide(
            resume_markdown=final_markdown,
            metrics_support=_mf_support,
            fallback_used=(analysis_result or {}).get("pipeline_status") in {
                "failed", "completed_with_fallback", "timeout"},
            required_facts_missing=unsupported_entities(
                final_markdown, resume_text),
            approved_entities=_approved_entities_from_payload(
                analysis_result, None, source_snapshot),
        )
        output = publication_gate.apply_to_payload(
            {
                **analysis_result,
                **criticism_result,
                # The verdict the gate decided on MUST be the verdict the client
                # receives. `apply_to_payload` copies `metrics_support` from this
                # dict, so omitting it here produced a response that said
                # `completed` while carrying no verdict at all — the client could
                # not tell a verified resume from an unverified one, and no
                # receipt could be issued because issuance reads this field.
                "metrics_support": _mf_support,
                "final_written_section": final_written_section,
                "final_written_section_markdown": final_markdown,
                "final_written_section_provenance": final_provenance,
                "critique_score": critique_score,
                "run_metrics": RUN_METRICS.copy(),
                "processing_time_seconds": time.time() - start_time,
            },
            _mf_decision,
        )
        output = _issue_publication_receipt(
            output, owner=trusted_owner_identity(request))
        validate_output_schema(output)
        processing_time = time.time() - start_time
        logger.info(
            "analyze_multi_file.request.completed",
            extra={
                "request_id": request_id,
                "processing_time_seconds": processing_time,
                "run_metrics": RUN_METRICS.copy(),
            },
        )
        return AnalysisResponse(**output)
    except HTTPException as e:
        raise e
    except Exception as e:
        processing_time = time.time() - start_time
        logger.exception(
            "analyze_multi_file.request.failed",
            extra={
                "request_id": getattr(request.state, "request_id", None),
                "processing_time_seconds": processing_time,
            },
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=f"Analysis failed: {str(e)}",
                error_code="ANALYSIS_FAILED",
                details={
                    "processing_time_seconds": processing_time,
                    "run_metrics": (
                        RUN_METRICS.copy() if RUN_METRICS.get("calls") else None
                    ),
                },
            ).model_dump(),
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
            detail="Context7 integration not configured. Set CONTEXT7_API_KEY environment variable.",
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
            last_updated=None,  # Context7 doesn't provide timestamps
        )

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Context7 MCP not installed. Install with: pip install context7-mcp",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Context7 documentation retrieval failed: {str(e)}"
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
                    headers={"Authorization": f"Bearer {key}"},
                ) as response:
                    return {
                        "status": "healthy" if response.status == 200 else "unhealthy",
                        "details": (
                            await response.text() if response.status != 200 else None
                        ),
                    }
        except Exception as e:
            return {"status": "error", "details": str(e)}

    key1_status = await check_key(settings.openrouter_api_key_1)
    key2_status = await check_key(settings.openrouter_api_key_2)

    return {
        "openrouter_key_1": key1_status,
        "openrouter_key_2": key2_status,
        "overall": (
            "healthy"
            if any(s["status"] == "healthy" for s in [key1_status, key2_status])
            else "unhealthy"
        ),
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
        "has_openrouter_key": bool(
            settings.openrouter_api_key_1 or settings.openrouter_api_key_2
        ),
        "has_context7_key": bool(settings.context7_api_key),
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


@app.post("/resume/export", dependencies=[Depends(get_api_key)])
async def export_resume(payload: Dict[str, Any], request: Request, api_key: str = Depends(get_api_key)):
    """Generate a formatted PDF + DOCX resume from structured Imaginator output.
    
    Accepts the same output payload from /analyze and renders it through
    LaTeX (moderncv banking style) + matching python-docx.
    
    Request body:
        {
            "name": "Jeff Calderon",          # required
            "location": "Marysville, WA",      # optional
            "phone": "916.712.8396",           # optional
            "email": "jeff.d.calderon@...",    # optional
            "linkedin": "linkedin.com/in/...", # optional
            "github": "github.com/...",        # optional
            "headline": "Data Scientist | ML", # optional
            "summary": "Professional summary", # optional
            "skills": [{ "category": "...", "items": [...] }],  # optional
            "experience": [{ "title": "...", "bullets": [...], "start_date": "...", "end_date": "..." }],  # optional
            "projects": [{ "title": "...", "bullets": [...] }],  # optional
            "education": ["B.S. Computer Science, ..."],  # optional
            "certifications": ["AWS Solutions Architect"],  # optional
            "job_title": "Data Scientist"     # optional, for filename
        }
    
    Returns:
        { "pdf_base64": "...", "docx_base64": "...", "filename": "Data_Scientist" }
    """
    request_id = getattr(request.state, "request_id", None)
    logger.info("resume.export.request", extra={"request_id": request_id})
    
    try:
        # ── Export containment: SERVER-AUTHORITATIVE (owner item 2) ─────
        # Previously this read `metrics_support` out of the request body, so a
        # caller could POST a forged
        #   {"verdict": "PASS (tier A only)", "checked": true}
        # alongside invented content and be approved. Client-supplied approval
        # is not approval.
        #
        # Now: the caller must present the signed receipt issued by /analyze.
        # The receipt binds the exact content fingerprint, the canonical render
        # structure, and the owner, so neither a forged verdict, edited markdown,
        # nor swapped structured fields passes (owner items 2 and 3).
        #
        # OWNER: read from the authenticated request context only. This used to
        # read `payload["owner"]` / `user_id` / `email`, which let the caller
        # nominate whose approval they were presenting. A normal application
        # export with no trusted owner identity is rejected outright — an
        # unowned approval would cross tenants.
        _export_markdown = (payload.get("final_written_section_markdown")
                            or _markdown_from_export_payload(payload))
        _receipt = payload.get("publication_receipt")
        # Accept either identity the orchestrator may present. It retries each
        # owner it can prove, so a receipt minted for a guest session exports
        # normally after the user signs in — without a second, charged analysis.
        # Which one was presented is decided upstream; here both are equally
        # valid claims, and the signature decides.
        _owner = trusted_owner_identity(request) or trusted_guest_identity(request)
        if not _owner:
            logger.warning(
                "resume.export.blocked",
                extra={"request_id": request_id, "reason": "owner_unbound"},
            )
            return JSONResponse(
                status_code=409,
                content={
                    "error": "publication_blocked",
                    "reasons": ["receipt_owner_unbound"],
                    "verification_coverage": "none",
                    "entity_evidence": "unknown",
                    "message": (
                        "This export was not made on behalf of an identified "
                        "account, so no approval can be attributed to it. Sign "
                        "in and retry from your analysis results."
                    ),
                },
            )
        _receipt_ok, _receipt_reason = publication_gate.verify_receipt(
            _receipt, markdown=_export_markdown, owner=_owner,
            render_payload=payload,
            # Declared policy state, if the caller sent one. This never GRANTS
            # anything: it can only make verification stricter, because a v5
            # receipt must match it and a v4 receipt carries no binding at all.
            policy_state=payload.get("skill_inference_policy_state"))

        if not _receipt_ok:
            logger.warning(
                "resume.export.blocked",
                extra={"request_id": request_id, "reason": _receipt_reason},
            )
            return JSONResponse(
                status_code=409,
                content={
                    "error": "publication_blocked",
                    "reasons": [_receipt_reason],
                    "verification_coverage": "none",
                    "entity_evidence": "unknown",
                    "message": (
                        "This document has not been approved for export. Run the "
                        "resume analysis first; only verified results can be "
                        "exported. Your original information is preserved."
                    ),
                },
            )

        # Receipt verified: the content is what the server approved.
        _export_support = payload.get("metrics_support")
        _export_decision = publication_gate.decide(
            resume_markdown=_export_markdown,
            metrics_support=_export_support
            or {"verdict": publication_gate.Verdict.PASS_TIER_A.value,
                "checked": True},
            approved_entities=_approved_entities_from_payload(payload, payload),
        )
        if _export_decision.blocked():
            logger.warning(
                "resume.export.blocked",
                extra={"request_id": request_id,
                       "reasons": _export_decision.reason_codes},
            )
            return JSONResponse(
                status_code=409,
                content={
                    "error": "publication_blocked",
                    "reasons": _export_decision.reason_codes,
                    "verification_coverage": _export_decision.coverage,
                    "entity_evidence": _export_decision.entity_evidence,
                    "message": _export_decision.review_message
                    or ("We couldn't verify some generated claims, so this "
                        "document was not produced. Your original information "
                        "is preserved."),
                },
            )

        from resume_builder.build_resume_from_export import build_resume_from_payload_base64
        
        # The receipt verifier and renderer both parse this same payload.  Do
        # not supply a separate headline override: with nested `data`, outer
        # title fields are not part of the verified render content.
        result = build_resume_from_payload_base64(payload)
        
        logger.info(
            "resume.export.complete",
            extra={
                "request_id": request_id, 
                "export_name": result["filename"],
                "pdf_size_chars": len(result["pdf_base64"]),
                "docx_size_chars": len(result["docx_base64"]),
            }
        )
        return JSONResponse(status_code=200, content=result)
        
    except Exception as e:
        logger.exception(
            "resume.export.failed",
            extra={"request_id": request_id, "err_msg": str(e)},
        )
        return JSONResponse(
            status_code=500,
            content={"error": f"Resume export failed: {str(e)}"},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        workers=settings.workers,
    )
