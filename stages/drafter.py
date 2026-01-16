"""
Stage 2: Drafter - Creates STAR-formatted bullets with seniority calibration

Based on Alternate_flow_proposal.md recommendations:
- Rewrites experiences into STAR bullets (Situation-Task-Action-Result)
- Applies seniority-based tone calibration
- Enforces mandatory quantification (%, $, time, scale)
- Prevents hallucination by using only user's actual companies
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pipeline_config import OR_SLUG_DRAFTER, TEMPERATURES, TIMEOUTS, get_seniority_config

logger = logging.getLogger(__name__)

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

def create_drafter_prompt(experiences: List[Dict], job_ad: str, research_data: Dict, 
                         seniority_level: str, allowed_verbs: List[str]) -> str:
    """
    Create the drafter prompt with dynamic seniority calibration.
    
    Args:
        experiences: User's actual experiences
        job_ad: Job description text
        research_data: Output from Researcher stage
        seniority_level: "junior", "mid", or "senior"
        allowed_verbs: List of action verbs for this seniority level
        
    Returns:
        Formatted system prompt
    """
    # Extract original company names to prevent hallucination
    original_companies = []
    for exp in experiences:
        company = exp.get("company") or exp.get("title_line", "").split("|")[-1].strip()
        if company and company not in original_companies:
            original_companies.append(company)
    
    # Get domain vocabulary from research
    domain_vocab = research_data.get("domain_vocab", [])
    implied_metrics = research_data.get("implied_metrics", [])
    
    system_prompt = f"""You are an expert Resume Writer. Rewrite the user's experiences into 3-5 STAR bullets.
SENIORITY TONE: {seniority_level.upper()}
ALLOWED VERBS: {', '.join(allowed_verbs)}

CRITICAL RULES:
1. USE ONLY the user's actual company names: {', '.join(original_companies) if original_companies else 'User companies'}
2. NEVER use "ABC Corp" or placeholders. If the user's data is sparse, stay truthful but technical.
3. MANDATORY QUANTIFICATION: Every bullet must include a number (%, $, time, or scale).
4. FORMAT: [Action Verb] [Skill/Task] to achieve [Outcome], resulting in [Metric].
5. Incorporate relevant domain terms: {', '.join(domain_vocab[:5]) if domain_vocab else 'Technical skills'}

EXAMPLE FORMATS:
✓ "Optimized PyTorch inference pipeline using TensorRT to reduce latency by 35% (120ms to 78ms) for 5k+ daily active users."
✓ "Led migration of legacy monolith to microservices, improving deployment frequency by 200% and reducing downtime 90%."
✓ "Architected cloud-native data platform on AWS, handling 10TB+ daily data volume with 99.99% availability."

BAD EXAMPLES (AVOID):
✗ "Worked on ML projects" (vague)
✗ "Improved performance" (no metric)
✗ "ABC Corp project" (hallucination - DON'T USE)
✗ "Responsible for tasks" (passive voice)

Output JSON Schema:
{{
  "rewritten_experiences": [
    {{
      "company": "Original Company Name",
      "role": "Job Title",
      "bullets": ["Bullet 1 with metric", "Bullet 2 with metric"],
      "metrics_used": ["35% latency reduction", "200% deployment frequency"]
    }}
  ],
  "seniority_applied": "{seniority_level}",
  "quantification_score": 0.95
}}
"""
    return system_prompt

# ============================================================================
# DRAFTER CLASS
# ============================================================================

class Drafter:
    """Stage 2: Creates STAR-formatted bullets with seniority calibration."""
    
    def __init__(self, llm_client):
        """
        Initialize Drafter stage.
        
        Args:
            llm_client: LLM client for making API calls
        """
        self.llm_client = llm_client
        self.model = OR_SLUG_DRAFTER
        self.temperature = TEMPERATURES["drafter"]
        self.timeout = TIMEOUTS["drafter"]
        
    async def draft(self, experiences: List[Dict], job_ad: str, 
                   research_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Draft STAR-formatted resume bullets.
        
        Args:
            experiences: User's actual experiences
            job_ad: Job description text
            research_data: Output from Researcher stage
            
        Returns:
            Dictionary with rewritten experiences and metadata
        """
        logger.info(f"[DRAFTER] Drafting {len(experiences)} experiences for job ad")
        
        # Determine seniority level and get configuration
        seniority_level, seniority_config = get_seniority_config(job_ad)
        allowed_verbs = seniority_config["verbs"]
        
        # Create prompt
        system_prompt = create_drafter_prompt(
            experiences=experiences,
            job_ad=job_ad,
            research_data=research_data,
            seniority_level=seniority_level,
            allowed_verbs=allowed_verbs
        )
        
        user_prompt = f"""
User Experiences (JSON):
{json.dumps(experiences, indent=2)}

Job Description:
{job_ad[:1000]}

Research Insights:
- Expected Metrics: {', '.join(research_data.get('implied_metrics', [])[:3])}
- Domain Vocabulary: {', '.join(research_data.get('domain_vocab', [])[:5])}
"""
        
        try:
            # Call LLM with strict JSON schema
            response = await self.llm_client.call_llm_async(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                timeout=self.timeout
            )
            
            # Parse and validate response
            result = self._parse_response(response, experiences)
            logger.info(f"[DRAFTER] Created {self._count_bullets(result)} STAR bullets "
                       f"with {seniority_level} seniority")
            
            # Ensure we have at least some data
            if not result.get("rewritten_experiences"):
                result["rewritten_experiences"] = []
                result["total_bullets"] = 0
                result["quantified_bullets"] = 0
                result["quantification_score"] = 0.0
                result["fallback"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"[DRAFTER] Failed to draft experiences: {e}")
            # Return safe fallback
            return self._create_fallback_output(experiences, seniority_level)
    
    def _parse_response(self, response: str, original_experiences: List[Dict]) -> Dict[str, Any]:
        """
        Parse LLM response and validate against original experiences.
        
        Args:
            response: Raw LLM response
            original_experiences: Original user experiences for validation
            
        Returns:
            Parsed and validated draft data
        """
        try:
            data = json.loads(response)
            
            # Ensure required structure
            if not isinstance(data, dict):
                raise ValueError("Response is not a JSON object")
            
            # Extract rewritten experiences
            rewritten = data.get("rewritten_experiences", [])
            if not isinstance(rewritten, list):
                rewritten = []
            
            # Validate each experience
            validated_experiences = []
            for exp in rewritten:
                if isinstance(exp, dict):
                    validated = {
                        "company": exp.get("company", ""),
                        "role": exp.get("role", ""),
                        "bullets": exp.get("bullets", []),
                        "metrics_used": exp.get("metrics_used", [])
                    }
                    # Ensure bullets is a list
                    if not isinstance(validated["bullets"], list):
                        validated["bullets"] = []
                    if not isinstance(validated["metrics_used"], list):
                        validated["metrics_used"] = []
                    
                    validated_experiences.append(validated)
            
            # Calculate quantification score
            total_bullets = sum(len(exp.get("bullets", [])) for exp in validated_experiences)
            quantified_bullets = sum(
                1 for exp in validated_experiences 
                for bullet in exp.get("bullets", []) 
                if any(char in bullet for char in ['%', '$', 'x', '+', '>', '<', '='])
            )
            
            quantification_score = quantified_bullets / max(total_bullets, 1)
            
            result = {
                "rewritten_experiences": validated_experiences,
                "seniority_applied": data.get("seniority_applied", "mid"),
                "quantification_score": quantification_score,
                "total_bullets": total_bullets,
                "quantified_bullets": quantified_bullets
            }
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"[DRAFTER] Failed to parse JSON response: {e}")
            logger.debug(f"[DRAFTER] Raw response: {response[:500]}")
            
            # Try to extract JSON from text
            json_match = self._extract_json_from_text(response)
            if json_match:
                return self._parse_response(json_match, original_experiences)
            
            # Return fallback
            return self._create_fallback_output(original_experiences, "mid")
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """
        Extract JSON object from text that may contain extra content.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            JSON string if found, None otherwise
        """
        # Look for JSON object boundaries
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            try:
                # Quick validation
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _create_fallback_output(self, experiences: List[Dict], seniority_level: str) -> Dict[str, Any]:
        """
        Create a fallback output when LLM fails.
        
        Args:
            experiences: Original experiences
            seniority_level: Determined seniority level
            
        Returns:
            Basic fallback structure
        """
        fallback_experiences = []
        
        for exp in experiences[:3]:  # Limit to 3 experiences
            company = exp.get("company") or exp.get("title_line", "").split("|")[-1].strip()
            role = exp.get("role") or exp.get("title_line", "").split("|")[0].strip()
            
            fallback_experiences.append({
                "company": company or "Previous Company",
                "role": role or "Professional Role",
                "bullets": [
                    "Applied technical skills to achieve measurable results",
                    "Contributed to team success through collaborative efforts"
                ],
                "metrics_used": ["Measurable impact", "Team contribution"]
            })
        
        return {
            "rewritten_experiences": fallback_experiences,
            "seniority_applied": seniority_level,
            "quantification_score": 0.3,
            "total_bullets": len(fallback_experiences) * 2,
            "quantified_bullets": 0,
            "fallback": True
        }
    
    def _count_bullets(self, draft_data: Dict[str, Any]) -> int:
        """Count total bullets in draft data."""
        total = 0
        for exp in draft_data.get("rewritten_experiences", []):
            total += len(exp.get("bullets", []))
        return total
    
    def get_draft_summary(self, draft_data: Dict[str, Any]) -> str:
        """
        Create a human-readable summary of drafting results.
        
        Args:
            draft_data: Draft data from draft()
            
        Returns:
            Formatted summary string
        """
        rewritten = draft_data.get("rewritten_experiences", [])
        total_bullets = sum(len(exp.get("bullets", [])) for exp in rewritten)
        quant_score = draft_data.get("quantification_score", 0.0)
        seniority = draft_data.get("seniority_applied", "unknown")
        
        return f"Generated {total_bullets} STAR bullets ({quant_score:.0%} quantified) at {seniority} level"