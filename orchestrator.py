"""
Orchestrator for Imaginator 3-Stage Pipeline

Coordinates the Researcher → Drafter → StarEditor pipeline with error handling,
cost tracking, and quality assurance.
"""

import asyncio
import json
import logging
import time
import re
from typing import Dict, Any, Optional, List
from datetime import datetime

from stages.researcher import Researcher
from stages.drafter import Drafter
from stages.star_editor import StarEditor
from pipeline_config import estimate_cost, TIMEOUTS

logger = logging.getLogger(__name__)

# ============================================================================
# ORCHESTRATOR CLASS
# ============================================================================

class PipelineOrchestrator:
    """Orchestrates the 3-stage resume enhancement pipeline."""
    
    def __init__(self, llm_client):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            llm_client: LLM client for making API calls
        """
        self.llm_client = llm_client
        self.researcher = Researcher(llm_client)
        self.drafter = Drafter(llm_client)
        self.star_editor = StarEditor(llm_client)
        
        # Pipeline metrics
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_cost_usd": 0.0,
            "stage_durations": {},
            "stage_costs": {}
        }
    
    async def run_pipeline(self, resume_text: str, job_ad: str, 
                          experiences: List[Dict], 
                          openrouter_api_keys: Optional[List[str]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Run the complete 3-stage pipeline.
        
        Args:
            resume_text: Raw resume text
            job_ad: Job description text
            experiences: Parsed experiences from resume
            openrouter_api_keys: Optional API keys for LLM calls
            **kwargs: Additional parameters
            
        Returns:
            Complete pipeline result with final resume and metadata
        """
        start_time = time.time()
        pipeline_start = datetime.utcnow().isoformat()
        
        logger.info(f"[ORCHESTRATOR] Starting pipeline for resume ({len(resume_text)} chars), "
                   f"job ad ({len(job_ad)} chars), {len(experiences)} experiences")
        
        result = {
            "pipeline_version": "3.0",
            "pipeline_start": pipeline_start,
            "stages": {},
            "metrics": {},
            "errors": [],
            "final_output": None
        }
        
        try:
            # STAGE 1: RESEARCHER - Extract metrics and domain vocabulary
            stage1_start = time.time()
            logger.info("[ORCHESTRATOR] Starting Stage 1: Researcher")
            
            research_data = await self.researcher.analyze(job_ad)
            stage1_duration = time.time() - stage1_start
            
            result["stages"]["researcher"] = {
                "status": "completed",
                "duration_seconds": stage1_duration,
                "data": research_data,
                "summary": self.researcher.get_metrics_summary(research_data)
            }
            
            logger.info(f"[ORCHESTRATOR] Stage 1 completed in {stage1_duration:.2f}s")
            logger.info(f"[ORCHESTRATOR] Researcher extracted {len(research_data.get('implied_metrics', []))} metrics")
            
            # STAGE 2: DRAFTER - Create STAR-formatted bullets
            stage2_start = time.time()
            logger.info("[ORCHESTRATOR] Starting Stage 2: Drafter")
            
            draft_data = await self.drafter.draft(experiences, job_ad, research_data)
            stage2_duration = time.time() - stage2_start
            
            result["stages"]["drafter"] = {
                "status": "completed",
                "duration_seconds": stage2_duration,
                "data": draft_data,
                "summary": self.drafter.get_draft_summary(draft_data)
            }
            
            logger.info(f"[ORCHESTRATOR] Stage 2 completed in {stage2_duration:.2f}s")
            logger.info(f"[ORCHESTRATOR] Drafter created {draft_data.get('total_bullets', 0)} bullets "
                      f"with {draft_data.get('quantification_score', 0):.1%} quantification")
            
            # STAGE 3: STAR EDITOR - Polish into final resume
            stage3_start = time.time()
            logger.info("[ORCHESTRATOR] Starting Stage 3: StarEditor")
            
            editor_data = await self.star_editor.polish(draft_data, research_data)
            stage3_duration = time.time() - stage3_start
            
            result["stages"]["star_editor"] = {
                "status": "completed",
                "duration_seconds": stage3_duration,
                "data": editor_data,
                "summary": self.star_editor.get_editor_summary(editor_data)
            }
            
            logger.info(f"[ORCHESTRATOR] Stage 3 completed in {stage3_duration:.2f}s")
            logger.info(f"[ORCHESTRATOR] StarEditor produced {len(editor_data.get('final_markdown', '').split())} word resume")
            
            # Combine results
            total_duration = time.time() - start_time
            
            # Check for fallback outputs
            pipeline_status = "completed"
            errors = result.get("errors", [])
            
            if draft_data.get("fallback") or editor_data.get("fallback"):
                pipeline_status = "completed_with_fallback"
                errors.append("Pipeline used fallback generation for one or more stages")
            
            result.update({
                "final_output": {
                    "final_written_section_markdown": editor_data.get("final_markdown", ""),
                    "final_written_section": editor_data.get("final_plain_text", ""),
                    "editorial_notes": editor_data.get("editorial_notes", ""),
                    "seniority_level": draft_data.get("seniority_applied", "mid"),
                    "domain_terms_used": editor_data.get("domain_terms_used", []),
                    "quantification_analysis": editor_data.get("quantification_check", {}),
                    "hallucination_checked": not editor_data.get("fallback", False)
                },
                "metrics": {
                    "total_duration_seconds": total_duration,
                    "stage_durations": {
                        "researcher": stage1_duration,
                        "drafter": stage2_duration,
                        "star_editor": stage3_duration
                    },
                    "stage_summaries": {
                        "researcher": f"Extracted {len(research_data.get('implied_metrics', []))} metrics",
                        "drafter": f"Created {draft_data.get('total_bullets', 0)} bullets",
                        "star_editor": f"Produced {len(editor_data.get('final_markdown', '').split())} word resume"
                    },
                    "pipeline_status": pipeline_status,
                    "pipeline_end": datetime.utcnow().isoformat()
                },
                "errors": errors
            })
            
            logger.info(f"[ORCHESTRATOR] Pipeline completed in {total_duration:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Pipeline failed: {e}", exc_info=True)
            
            # Create emergency fallback output
            emergency_output = self._create_emergency_output(experiences, job_ad)
            
            result.update({
                "final_output": emergency_output,
                "metrics": {
                    "total_duration_seconds": time.time() - start_time,
                    "pipeline_status": "failed",
                    "pipeline_end": datetime.utcnow().isoformat(),
                    "error": str(e)
                },
                "errors": result.get("errors", []) + [f"Pipeline failed: {str(e)}"]
            })
            
            return result
    
    def _create_emergency_output(self, experiences: List[Dict], job_ad: str) -> Dict[str, Any]:
        """
        Create emergency output when pipeline fails completely.
        
        Args:
            experiences: Original experiences
            job_ad: Job description
            
        Returns:
            Emergency output structure
        """
        # Simple markdown generation from experiences
        markdown_lines = ["## Professional Experience"]
        
        for exp in experiences[:3]:  # Limit to 3 experiences
            company = exp.get("company") or exp.get("title_line", "").split("|")[-1].strip()
            role = exp.get("role") or exp.get("title_line", "").split("|")[0].strip()
            
            if company and role:
                markdown_lines.append(f"\n**{role}** at *{company}*")
                markdown_lines.append("- Applied relevant skills to achieve business objectives")
                markdown_lines.append("- Collaborated with team members to deliver results")
        
        markdown = "\n".join(markdown_lines)
        
        return {
            "final_written_section_markdown": markdown,
            "final_written_section": re.sub(r'\*\*|\*|##', '', markdown),
            "editorial_notes": "Emergency fallback generated due to pipeline failure.",
            "seniority_level": "mid",
            "domain_terms_used": [],
            "quantification_analysis": {
                "total_bullets": 0,
                "quantified_bullets": 0,
                "quantification_score": 0,
                "needs_improvement": True
            },
            "hallucination_checked": True,
            "emergency_fallback": True
        }
    
    async def run_with_timeout(self, resume_text: str, job_ad: str, 
                              experiences: List[Dict], 
                              timeout: Optional[int] = None,
                              **kwargs) -> Dict[str, Any]:
        """
        Run pipeline with timeout protection.
        
        Args:
            resume_text: Raw resume text
            job_ad: Job description text
            experiences: Parsed experiences from resume
            timeout: Maximum time in seconds (defaults to TIMEOUTS["total"])
            **kwargs: Additional parameters
            
        Returns:
            Pipeline result or timeout error
        """
        timeout = timeout or TIMEOUTS["total"]
        
        try:
            return await asyncio.wait_for(
                self.run_pipeline(resume_text, job_ad, experiences, **kwargs),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"[ORCHESTRATOR] Pipeline timeout after {timeout}s")
            
            # Create timeout output
            emergency_output = self._create_emergency_output(experiences, job_ad)
            
            return {
                "final_output": emergency_output,
                "metrics": {
                    "total_duration_seconds": timeout,
                    "pipeline_status": "timeout",
                    "pipeline_end": datetime.utcnow().isoformat(),
                    "error": f"Pipeline timeout after {timeout}s"
                },
                "errors": [f"Pipeline timeout after {timeout}s"],
                "stages": {
                    "researcher": {"status": "timeout"},
                    "drafter": {"status": "timeout"},
                    "star_editor": {"status": "timeout"}
                }
            }
    
    def validate_inputs(self, resume_text: str, job_ad: str, experiences: List[Dict]) -> List[str]:
        """
        Validate pipeline inputs.
        
        Args:
            resume_text: Raw resume text
            job_ad: Job description text
            experiences: Parsed experiences
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check resume text
        if not resume_text or not resume_text.strip():
            errors.append("Resume text is empty")
        elif len(resume_text.strip()) < 50:
            errors.append("Resume text is too short (minimum 50 characters)")
        
        # Check job ad
        if not job_ad or not job_ad.strip():
            errors.append("Job description is empty")
        elif len(job_ad.strip()) < 20:
            errors.append("Job description is too short (minimum 20 characters)")
        
        # Check experiences
        if not experiences:
            errors.append("No experiences extracted from resume")
        else:
            valid_experiences = 0
            for exp in experiences:
                if exp.get("company") or exp.get("role"):
                    valid_experiences += 1
            
            if valid_experiences == 0:
                errors.append("No valid experiences with company/role information")
        
        return errors
    
    def get_pipeline_summary(self, result: Dict[str, Any]) -> str:
        """
        Create human-readable pipeline summary.
        
        Args:
            result: Pipeline result from run_pipeline()
            
        Returns:
            Formatted summary string
        """
        metrics = result.get("metrics", {})
        stages = result.get("stages", {})
        final_output = result.get("final_output", {})
        
        summary = "=" * 80 + "\n"
        summary += "IMAGINATOR 3-STAGE PIPELINE SUMMARY\n"
        summary += "=" * 80 + "\n\n"
        
        # Pipeline status
        status = metrics.get("pipeline_status", "unknown")
        duration = metrics.get("total_duration_seconds", 0)
        summary += f"STATUS: {status.upper()}\n"
        summary += f"DURATION: {duration:.2f}s\n\n"
        
        # Stage summaries
        summary += "STAGE PERFORMANCE:\n"
        summary += "-" * 40 + "\n"
        
        for stage_name, stage_data in stages.items():
            stage_status = stage_data.get("status", "unknown")
            stage_duration = stage_data.get("duration_seconds", 0)
            stage_summary = stage_data.get("summary", "No summary")
            
            summary += f"{stage_name.upper()}: {stage_status} ({stage_duration:.2f}s)\n"
            summary += f"  {stage_summary[:100]}...\n\n"
        
        # Final output summary
        if final_output:
            markdown = final_output.get("final_written_section_markdown", "")
            word_count = len(markdown.split())
            line_count = len(markdown.split('\n'))
            seniority = final_output.get("seniority_level", "unknown")
            
            summary += "FINAL OUTPUT:\n"
            summary += "-" * 40 + "\n"
            summary += f"Seniority: {seniority.upper()}\n"
            summary += f"Word Count: {word_count}\n"
            summary += f"Lines: {line_count}\n"
            
            # Check for emergency fallback
            if final_output.get("emergency_fallback"):
                summary += "⚠️  EMERGENCY FALLBACK USED\n"
            
            # Show first few lines
            preview_lines = markdown.split('\n')[:5]
            summary += "\nPreview:\n"
            for line in preview_lines:
                summary += f"  {line}\n"
            if line_count > 5:
                summary += f"  ... and {line_count - 5} more lines\n"
        
        # Errors
        errors = result.get("errors", [])
        if errors:
            summary += "\nERRORS:\n"
            summary += "-" * 40 + "\n"
            for error in errors[:3]:  # Show first 3 errors
                summary += f"• {error}\n"
            if len(errors) > 3:
                summary += f"  ... and {len(errors) - 3} more errors\n"
        
        summary += "=" * 80 + "\n"
        
        return summary