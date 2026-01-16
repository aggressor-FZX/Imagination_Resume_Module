"""
Models for request/response schemas
Based on SYSTEM_IO_SPECIFICATION.md and Context7 research on Pydantic
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, AliasChoices, ConfigDict
import json


class ProcessingStatus(str, Enum):
    """Status of processing operations"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SkillData(BaseModel):
    """Structured skill data with confidence scores"""
    title: str = Field(..., description="Job title or role")
    canonical_title: str = Field(..., description="Normalized job title")
    skills: List[str] = Field(..., description="List of skills")
    confidence_scores: Optional[Dict[str, float]] = Field(
        None, description="Confidence scores for each skill"
    )


class DomainInsights(BaseModel):
    """Domain-specific insights and market intelligence"""
    domain: str = Field(..., description="Primary domain/industry")
    market_demand: Optional[str] = Field(None, description="Market demand level")
    skill_gap_priority: str = Field(..., description="Priority level for skill gaps")
    emerging_trends: Optional[List[str]] = Field(
        None, description="Emerging industry trends"
    )
    insights: Optional[List[str]] = Field(
        None, description="Additional domain insights"
    )


class ExperienceEntry(BaseModel):
    """Parsed work experience entry"""
    title_line: str = Field(..., description="Job title and company")
    skills: List[str] = Field(..., description="Skills associated with this experience")
    snippet: str = Field(..., description="Relevant experience description")


class ProcessedSkills(BaseModel):
    """Processed skills with categorization and confidence"""
    high_confidence: List[str] = Field(..., description="High confidence skills")
    medium_confidence: List[str] = Field(..., description="Medium confidence skills")
    low_confidence: List[str] = Field(..., description="Low confidence skills")
    inferred_skills: List[str] = Field(
        ..., description="Skills inferred from skill adjacency"
    )


class SeniorityAnalysis(BaseModel):
    """Seniority level analysis results"""
    level: str = Field(..., description="Detected seniority level")
    confidence: float = Field(..., description="Confidence score (0-1)")
    total_years_experience: float = Field(..., description="Total years of experience")
    experience_quality_score: float = Field(..., description="Experience quality score")
    leadership_score: float = Field(..., description="Leadership experience score")
    skill_depth_score: float = Field(..., description="Technical skill depth score")
    achievement_complexity_score: float = Field(..., description="Achievement complexity score")
    reasoning: str = Field(..., description="Human-readable reasoning")
    recommendations: List[str] = Field(..., description="Career development recommendations")


class RunMetrics(BaseModel):
    """Usage metrics and cost tracking"""
    calls: List[Dict[str, Any]] = Field(..., description="Individual API calls")
    total_prompt_tokens: int = Field(..., description="Total prompt tokens used")
    total_completion_tokens: int = Field(..., description="Total completion tokens used")
    total_tokens: int = Field(..., description="Total tokens used")
    estimated_cost_usd: float = Field(..., description="Estimated cost in USD")
    failures: List[Dict[str, Any]] = Field(
        default_factory=list, description="Failed API calls"
    )


class AnalysisRequest(BaseModel):
    """Request model for resume analysis"""
    model_config = ConfigDict(populate_by_name=True)

    resume_text: str = Field(..., min_length=0, description="Raw resume text content")
    job_ad: str = Field(
        ...,
        min_length=1,
        description="Target job description text",
        validation_alias=AliasChoices("job_ad", "job_description", "jobDescription"),
    )
    confidence_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum confidence score for skills"
    )
    extracted_skills_json: Optional[str] = Field(
        None, description="JSON string of structured skills data"
    )
    domain_insights_json: Optional[str] = Field(
        None, description="JSON string of domain insights"
    )

    @field_validator('extracted_skills_json')
    @classmethod
    def validate_skills_json(cls, v):
        """Validate that skills JSON can be parsed"""
        if v is not None:
            try:
                json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("extracted_skills_json must be valid JSON")
        return v

    @field_validator('domain_insights_json')
    @classmethod
    def validate_insights_json(cls, v):
        """Validate that insights JSON can be parsed"""
        if v is not None:
            try:
                json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("domain_insights_json must be valid JSON")
        return v


class ProvenanceEntry(BaseModel):
    """Provenance tracking for a claim in the final resume"""
    claim: str = Field(..., description="The specific claim or sentence")
    experience_index: Optional[int] = Field(None, description="Index in experiences array")
    skill_references: List[str] = Field(default_factory=list, description="Skills cited in this claim")
    is_synthetic: bool = Field(False, description="Whether this claim is inferred vs directly from resume")


class AnalysisResponse(BaseModel):
    """Response model for resume analysis results"""
    experiences: Optional[List[ExperienceEntry]] = Field(None, description="Parsed work experiences")
    aggregate_skills: Optional[List[str]] = Field(None, description="All unique skills found")
    processed_skills: Optional[ProcessedSkills] = Field(None, description="Categorized skills")
    domain_insights: Optional[DomainInsights] = Field(None, description="Domain intelligence")
    gap_analysis: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Gap analysis (string or structured object)")
    suggested_experiences: Optional[Dict[str, Any]] = Field(
        None, description="Refined improvement suggestions"
    )
    seniority_analysis: Optional[SeniorityAnalysis] = Field(None, description="Seniority level analysis")
    final_written_section: Optional[str] = Field(None, description="Generated resume section text")
    final_written_section_markdown: Optional[str] = Field(None, description="Markdown-formatted resume section")
    final_written_section_provenance: List[ProvenanceEntry] = Field(
        default_factory=list, description="Claim-to-source mapping for trust and verification"
    )
    critique_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quality score from critique phase")
    run_metrics: Optional[RunMetrics] = Field(None, description="Usage metrics and costs")
    processing_status: ProcessingStatus = Field(
        ProcessingStatus.COMPLETED, description="Processing status"
    )
    processing_time_seconds: float = Field(..., description="Total processing time")

    @field_validator("final_written_section_provenance", mode="before")
    @classmethod
    def coerce_provenance_entries(cls, v):
        """Allow provenance to arrive as strings/dicts and coerce into ProvenanceEntry-compatible dicts."""
        if v is None:
            return []
        if not isinstance(v, list):
            return []

        normalized: List[Any] = []
        for item in v:
            if isinstance(item, ProvenanceEntry):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, str) and item.strip():
                normalized.append(
                    {
                        "claim": item.strip(),
                        "experience_index": None,
                        "skill_references": [],
                        "is_synthetic": False,
                    }
                )
        return normalized


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Deployment environment")
    timestamp: str = Field(..., description="Timestamp of health check")
    has_openrouter_key: bool = Field(..., description="Whether OpenRouter API key is configured")


class Context7DocsRequest(BaseModel):
    """Request model for Context7 documentation"""
    library: str = Field(..., min_length=1, description="Library name")
    version: str = Field("latest", description="Version to get docs for")


class Context7DocsResponse(BaseModel):
    """Response model for Context7 documentation"""
    library: str = Field(..., description="Library name")
    version: str = Field(..., description="Version requested")
    documentation: Dict[str, Any] = Field(..., description="Documentation content")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
