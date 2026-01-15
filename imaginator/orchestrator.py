#!/usr/bin/env python3
"""
Imaginator Orchestrator
THE FUNNEL: The 4-Stage Pipeline Logic

This module coordinates the complete resume enhancement pipeline:
1. Researcher (Heavy Start) - Compiles master dossier with web research
2. Drafter (Lean Middle) - Creates creative narrative
3. STAR Editor (Lean Middle) - Formats with STAR methodology
4. Polisher (Analytical Finish) - Final QC and job ad alignment

The funnel architecture demonstrates data discarding between stages to optimize
context usage and cost efficiency.
"""

import logging
from typing import Dict, Any, List

from .stages.researcher import run_stage1_researcher
from .stages.drafter import run_stage2_drafter
from .stages.star_editor import run_stage3_star_editor
from .stages.polisher import run_stage4_polisher

logger = logging.getLogger(__name__)


async def run_full_funnel_pipeline(
    resume_text: str,
    job_ad: str,
    hermes_data: Dict[str, Any],
    svm_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute the complete 4-stage funnel pipeline.
    
    Architecture Pattern:
    - Stage 1: Heavy Start (ingests everything)
    - Stage 2 & 3: Lean Middle (only high-signal data)
    - Stage 4: Analytical Finish (re-injects job ad for verification)
    
    Args:
        resume_text: Raw resume text from Document Reader
        job_ad: Target job description
        hermes_data: Structured insights from Hermes service
        svm_data: Validated skills/titles from FastSVM service
    
    Returns:
        Final polished resume with metrics and metadata
    """
    logger.info("🚀 Starting 4-Stage Funnel Pipeline")
    
    # --- STAGE 1: HEAVY START ---
    # Ingests everything: raw text + structured data + web research
    logger.info("📊 STAGE 1: Researcher (Heavy Start) - Compiling Master Dossier")
    
    master_dossier = await run_stage1_researcher(
        resume_text=resume_text,
        job_ad=job_ad,
        hermes_data=hermes_data,
        svm_data=svm_data
    )
    
    # NOTICE: We now have a high-signal dossier. Raw resume_text is discarded.
    logger.info("✅ Stage 1 Complete: Master dossier compiled")
    
    # --- STAGE 2: LEAN MIDDLE (Creative) ---
    # Only passes the high-signal dossier, NOT the raw resume text
    logger.info("📝 STAGE 2: Drafter (Lean Middle) - Creative Narrative")
    
    creative_draft = await run_stage2_drafter(
        job_ad=job_ad,
        research_data=master_dossier
    )
    
    # NOTICE: We keep the creative draft but discard the research data
    logger.info("✅ Stage 2 Complete: Creative draft generated")
    
    # --- STAGE 3: LEAN MIDDLE (STAR Formatting) ---
    # Formats the creative draft into STAR methodology
    logger.info("⭐ STAGE 3: STAR Editor (Lean Middle) - STAR Formatting")
    
    star_draft = await run_stage3_star_editor(
        creative_draft=creative_draft,
        research_data=master_dossier
    )
    
    # NOTICE: We keep the STAR-formatted draft
    logger.info("✅ Stage 3 Complete: STAR formatting applied")
    
    # --- STAGE 4: ANALYTICAL FINISH ---
    # Re-injects original job ad for final QC and drift prevention
    logger.info("🔍 STAGE 4: Polisher (Analytical Finish) - Final QC")
    
    final_output = await run_stage4_polisher(
        star_draft=star_draft,
        original_job_ad=job_ad
    )
    
    logger.info("✅ Stage 4 Complete: Final polish and verification")
    logger.info("🎉 Pipeline Complete!")
    
    return final_output


async def run_pipeline_stages(
    resume_text: str,
    job_ad: str,
    hermes_data: Dict[str, Any],
    svm_data: Dict[str, Any],
    stages: List[int]
) -> Dict[str, Any]:
    """
    Run specific stages of the pipeline for testing or partial processing.
    
    Args:
        resume_text: Raw resume text
        job_ad: Target job description
        hermes_data: Hermes insights
        svm_data: FastSVM data
        stages: List of stage numbers to run (e.g., [1, 2, 4])
    
    Returns:
        Intermediate or final results based on stages run
    """
    logger.info(f"🔧 Running pipeline stages: {stages}")
    
    result = {}
    
    if 1 in stages:
        logger.info("📊 Running Stage 1: Researcher")
        result = await run_stage1_researcher(resume_text, job_ad, hermes_data, svm_data)
    
    if 2 in stages:
        logger.info("📝 Running Stage 2: Drafter")
        if 1 not in stages:
            raise ValueError("Stage 2 requires Stage 1 output")
        result = await run_stage2_drafter(job_ad, result)
    
    if 3 in stages:
        logger.info("⭐ Running Stage 3: STAR Editor")
        if 2 not in stages:
            raise ValueError("Stage 3 requires Stage 2 output")
        result = await run_stage3_star_editor(result, result)
    
    if 4 in stages:
        logger.info("🔍 Running Stage 4: Polisher")
        if 3 not in stages:
            raise ValueError("Stage 4 requires Stage 3 output")
        result = await run_stage4_polisher(result, job_ad)
    
    return result


# Backward compatibility with old monolithic function names
async def process_resume_enhancement(
    resume_text: str,
    job_ad: str,
    hermes_data: Dict[str, Any],
    svm_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Backward compatibility wrapper for the old monolithic function.
    Routes to the new funnel pipeline.
    """
    logger.warning("Using deprecated process_resume_enhancement - use run_full_funnel_pipeline instead")
    return await run_full_funnel_pipeline(resume_text, job_ad, hermes_data, svm_data)