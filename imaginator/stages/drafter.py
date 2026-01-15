#!/usr/bin/env python3
"""
Imaginator Stage 2: Creative Drafter
Lean Middle - Creative Resume Content Generation
"""
import logging
import time
from typing import Any, Dict, List, Optional

from ..gateway import call_llm_async
from ..config import MODEL_STAGE_2

logger = logging.getLogger(__name__)


async def run_stage2_drafter(
    analysis: Dict[str, Any],
    job_ad: str,
    research_data: Dict[str, Any],
    openrouter_api_keys: Optional[List[str]] = None,
    **kwargs,
) -> str:
    """
    Stage 2: Creative Drafter - Creative Resume Content Generation
    
    Uses Skyfall 36B to draft enhanced resume content.
    Takes research findings and existing analysis to create compelling narratives.
    
    Args:
        analysis: Base analysis with skills and experiences
        job_ad: Target job description
        research_data: Output from Stage 1 (researcher)
        
    Returns:
        Enhanced resume draft as markdown string
    """
    from config import settings
    
    logger.info("[CREATIVE DRAFTER] === Stage 2: Drafting enhanced resume ===")
    
    # Track stage metrics
    stage_metrics = {"start": time.time()}
    
    # Extract key data (PRIORITY: Hermes + FastSVM + Researcher data FIRST)
    experiences = analysis.get("experiences", [])
    aggregate_skills = list(analysis.get("aggregate_skills", []))  # From Hermes/FastSVM
    implied_skills = research_data.get("implied_skills", [])       # From researcher
    industry_metrics = research_data.get("industry_metrics", [])   # From researcher
    research_notes = research_data.get("research_notes", "")       # From researcher
    
    # Build experiences summary - FIT AS MUCH AS POSSIBLE within context limits
    # Priority: Recent experiences get more detail
    exp_details = []
    estimated_tokens = 0
    max_context_tokens = 16000  # Conservative estimate for Skyfall 36B context window
    
    # Reserve tokens for: system prompt (~300), skills (~200), job ad (~400), research (~500) = ~1400 tokens
    # This leaves ~14,600 tokens for experiences
    # WARNING: Stage 3 (Phi-4) has a 16k context window, and its prompt includes the OUTPUT of this stage.
    # To be safe, we must ensure Stage 2 input + output fits easily within limits.
    # But Stage 3 INPUT is Stage 2 OUTPUT.
    # Stage 2 INPUT can be large (Skyfall/Gemini have large context).
    # But Stage 2 OUTPUT (the draft) needs to be reasonable length.
    # The output length is limited by `max_tokens=2500` in the call_llm_async below.
    # So Stage 3 INPUT will be roughly 2500 tokens (draft) + Star Prompt (~1000).
    # That should fit easily in 16k.
    # So why did we get 18k requested? 
    # Maybe `experiences_text` was huge and it somehow got echoed? 
    # Or maybe the error came from THIS stage (Creative Drafter)?
    # "Context Window exceeded" in Imaginator usually comes from the model we are calling.
    # If the user saw "Requested 18k, Limit 16k", and it was Phi-4, then Phi-4 received 18k tokens.
    # Phi-4 is used in Stage 3.
    # Stage 3 prompt: creative_draft + star_suggestions.
    # If creative_draft is 15k tokens, that explains it.
    # Does Gemini/Skyfall produce 15k output? No, max_tokens=2500.
    # Wait, did the user confuse the stage?
    # If the error was at Stage 2, maybe Skyfall/Gemini has a limit?
    # Skyfall 36B might have a smaller context than we think?
    # Let's reduce available_tokens_for_exp just in case.
    available_tokens_for_exp = 8000  # Reduced from 14000 to be safe
    
    for i, exp in enumerate(experiences, 1):
        if isinstance(exp, dict):
            title = exp.get("title_line", "Position")
            skills_list = exp.get("skills", [])
            snippet = exp.get("snippet", "")
            
            # More recent experiences get full detail, older ones get truncated
            if i <= 3:  # First 3 experiences: full detail
                exp_text = f"**Experience {i}: {title}**\nSkills: {', '.join(skills_list[:15])}\n{snippet[:800]}"
            elif i <= 5:  # Next 2: moderate detail
                exp_text = f"**Experience {i}: {title}**\nSkills: {', '.join(skills_list[:10])}\n{snippet[:400]}"
            else:  # Remaining: minimal detail
                exp_text = f"**Experience {i}: {title}**\nSkills: {', '.join(skills_list[:5])}\n{snippet[:200]}"
            
            # Rough token estimate: 1 token ≈ 4 characters
            exp_tokens = len(exp_text) // 4
            
            if estimated_tokens + exp_tokens > available_tokens_for_exp:
                logger.info(f"[CREATIVE DRAFTER] Reached context limit at experience {i}/{len(experiences)}")
                break
            
            exp_details.append(exp_text)
            estimated_tokens += exp_tokens
    
    experiences_text = "\n\n".join(exp_details) or "No detailed experiences"
    logger.info(f"[CREATIVE DRAFTER] Included {len(exp_details)}/{len(experiences)} experiences (~{estimated_tokens} tokens)")
    
    system_prompt = """You are an expert resume writer specializing in creating compelling, achievement-focused narratives.

Your task: Draft an enhanced "Professional Experience" section that positions the candidate optimally for the target role.

Guidelines:
1. **Use Candidate's ACTUAL Experiences** - Never invent positions, companies, or accomplishments
2. **Incorporate Research Insights** - Weave in implied skills and industry metrics discovered via web research
3. **Write Compelling Narratives** - Transform basic job duties into achievement stories
4. **Match Target Role** - Align language and focus with the job description
5. **Be Specific** - Use concrete examples from candidate's background
6. **Maintain Authenticity** - Enhance what exists, don't fabricate
7. **Handle Chronological Positions Carefully**:
   - When a candidate has overlapping roles (same time period, different titles), choose the ONE most relevant to target job
   - Understand career progression: people often change titles slightly while in same position or hold multiple roles
   - Prioritize roles that best match target job requirements
   - In reverse chronological order (most recent first)

Key keyword: "creative" - This triggers Gemini 2.5 Pro routing.

Return the enhanced experience section in clean markdown format with:
- Company name and dates (from original)
- Enhanced position titles (if appropriate)
- 4-6 achievement bullets per position
- Focus on impact and results"""
    
    user_prompt = f"""PRIORITY DATA (Hermes + FastSVM Analysis):
Extracted Skills: {', '.join(aggregate_skills[:50])}

WEB RESEARCH INSIGHTS (DeepSeek Search):
- Implied Skills: {', '.join(implied_skills[:25])}
- Industry Metrics: {', '.join(industry_metrics[:15])}
- Research Notes: {research_notes[:600]}

TARGET JOB DESCRIPTION:
{job_ad[:1000]}

CANDIDATE'S ACTUAL EXPERIENCES (fit within context):
{experiences_text}

TASK:
Draft an enhanced "Professional Experience" section using:
1. Hermes/FastSVM extracted skills (PRIMARY SOURCE)
2. Web research insights (implied skills & metrics)
3. Candidate's actual experiences (fitted within context)

Use ACTUAL experiences only. Incorporate research insights naturally. Make it compelling and aligned with target role."""
    
    try:
        if getattr(settings, "environment", "") == "test":
            mock_draft = f"# Professional Experience\n\n## {experiences[0].get('title_line', 'Position')} (Enhanced)\nMock creative draft using research insights and candidate background."
            stage_metrics["end"] = time.time()
            stage_metrics["duration_ms"] = int((stage_metrics["end"] - stage_metrics["start"]) * 1000)
            return mock_draft
        
        logger.info("[CREATIVE DRAFTER] Calling Skyfall 36B for creative drafting (COST-OPTIMIZED)")
        draft = await call_llm_async(
            system_prompt,
            user_prompt,
            temperature=0.8,  # Higher temperature for creativity
            max_tokens=2500,
            openrouter_api_keys=openrouter_api_keys,
            **kwargs
        )
        
        logger.info(f"[CREATIVE DRAFTER] Generated draft ({len(draft)} chars)")
        return draft
    
    except Exception as e:
        logger.error(f"[CREATIVE DRAFTER] Error: {e}")
        return f"# Professional Experience\n\n_Draft generation failed: {str(e)}_"
    finally:
        stage_metrics["end"] = time.time()
        stage_metrics["duration_ms"] = int((stage_metrics["end"] - stage_metrics["start"]) * 1000) if stage_metrics.get("start") and stage_metrics.get("end") else None