#!/usr/bin/env python3
"""
Imaginator Stage 3: STAR Editor
Lean Middle - STAR Pattern Formatting
"""
import logging
import time
from typing import Any, Dict, List, Optional

from ..gateway import call_llm_async
from ..config import MODEL_STAGE_3

logger = logging.getLogger(__name__)


async def run_stage3_star_editor(
    creative_draft: str,
    research_data: Dict[str, Any],
    experiences: List[Dict[str, Any]],
    openrouter_api_keys: Optional[List[str]] = None,
    **kwargs,
) -> str:
    """
    Stage 3: STAR Editor - STAR Pattern Formatting
    
    Uses Microsoft Phi-4 to format into STAR pattern bullets.
    Takes creative draft and structures it using Situation-Task-Action-Result format.
    
    Args:
        creative_draft: Output from Stage 2 (creative drafter)
        research_data: Research findings with STAR suggestions
        experiences: Original parsed experiences
        
    Returns:
        STAR-formatted resume section as markdown
    """
    from config import settings
    
    logger.info("[STAR EDITOR] === Stage 3: Formatting into STAR pattern ===")
    
    # Track stage metrics
    stage_metrics = {"start": time.time()}
    
    star_suggestions = research_data.get("star_suggestions", [])
    star_context = "\n".join([
        f"- Experience {s.get('experience_index', 0)+1}: {', '.join(s.get('result_metrics', [])[:3])}"
        for s in star_suggestions[:3]
    ]) if star_suggestions else "No specific STAR suggestions"
    
    system_prompt = """You are an analytical editor specializing in STAR format (Situation-Task-Action-Result) resume bullets.

Your task: Take the creative draft and restructure each achievement bullet to follow STAR pattern.

STAR Format Guidelines:
- **Situation**: Brief context (1 phrase)
- **Task**: What needed to be done
- **Action**: Specific actions taken (strong action verbs)
- **Result**: Quantifiable outcome with metrics

Example:
"Reduced deployment time by 60% (Result) by implementing automated CI/CD pipeline (Action) to address frequent production delays (Situation/Task)"

Key keyword: "star" and "format" - This triggers Microsoft Phi-4 routing for analytical precision.

Return formatted experience section maintaining all original content but restructured into STAR bullets."""
    
    user_prompt = f"""CREATIVE DRAFT TO FORMAT:
{creative_draft}

STAR SUGGESTIONS FROM RESEARCH:
{star_context}

TASK:
Restructure the draft above into STAR-formatted bullets. Each bullet should clearly show Situation-Task-Action-Result.
Focus on making metrics and outcomes prominent. Maintain authenticity - don't invent new achievements."""
    
    try:
        if getattr(settings, "environment", "") == "test":
            mock_star = f"# Professional Experience (STAR Formatted)\n\n{creative_draft[:200]}\n\n(Mock STAR formatting applied)"
            stage_metrics["end"] = time.time()
            stage_metrics["duration_ms"] = int((stage_metrics["end"] - stage_metrics["start"]) * 1000)
            return mock_star
        
        logger.info("[STAR EDITOR] Calling Microsoft Phi-4 for STAR formatting")
        star_formatted = await call_llm_async(
            system_prompt,
            user_prompt,
            temperature=0.5,  # Lower temperature for structured formatting
            max_tokens=2500,
            openrouter_api_keys=openrouter_api_keys,
            **kwargs
        )
        
        logger.info(f"[STAR EDITOR] Formatted into STAR pattern ({len(star_formatted)} chars)")
        return star_formatted
    
    except Exception as e:
        logger.error(f"[STAR EDITOR] Error: {e}")
        return creative_draft  # Fallback to creative draft
    finally:
        stage_metrics["end"] = time.time()
        stage_metrics["duration_ms"] = int((stage_metrics["end"] - stage_metrics["start"]) * 1000) if stage_metrics.get("start") and stage_metrics.get("end") else None