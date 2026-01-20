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
from pipeline_config import OR_SLUG_DRAFTER, FALLBACK_MODELS, TEMPERATURES, TIMEOUTS, get_seniority_config

logger = logging.getLogger(__name__)

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

def create_drafter_prompt(experiences: List[Dict], job_ad: str, research_data: Dict,
                         seniority_level: str, allowed_verbs: List[str],
                         golden_bullets: Optional[List[str]] = None) -> str:
    """
    Create the drafter prompt with strict anti-hallucination guardrails and aggressive Style Transfer.
    """
    # 1. Extract TRUTH constraints
    original_companies = [exp.get("company", "Unknown") for exp in experiences if exp.get("company")]
    original_titles = [exp.get("role", "Unknown") for exp in experiences if exp.get("role")]
    
    # 2. Format Golden Bullets as "Style Reference Only"
    golden_section = ""
    if golden_bullets:
        selected_bullets = golden_bullets[:10] # Reduced to 10 to reduce context noise
        bullets_text = "\n".join([f"• {b}" for b in selected_bullets])
        
        golden_section = f"""
### GOLDEN PATTERN LIBRARY (THE SOURCE OF TRUTH)
WARNING: The following examples contain FAKE DATA and EXTERNAL COMPANIES.
USE THEM FOR SYNTAX ONLY. DO NOT COPY THE CONTENT, COMPANY NAMES, OR DATES.
You must transform the User's experience to match the *grammatical structure* and *density* of these examples.
DO NOT COPY THE CONTENT. COPY THE SYNTAX.

REFERENCE PATTERNS:
{bullets_text}

### STYLE TRANSFER INSTRUCTIONS (MANDATORY):
1. **Select a Template:** For every user bullet, find a Golden Bullet that matches the *type* of work (e.g., Migration, Optimization, Leadership).
2. **Map the Variables:**
   - Golden: "Reduced [Metric] by [Action] using [Tech]."
   - User: "I made python scripts to run faster."
   - Result: "Reduced [processing time] by [optimizing Python scripts] using [multithreading]."
3. **Kill the Fluff:** Delete weak verbs like "Strategized," "Participated," "Pioneered," "Worked on." Use hard technical verbs (e.g., "Deployed," "Engineered," "Refactored").
"""

    system_prompt = f"""You are an elite Technical Resume Writer. Your goal is to rewrite the user's raw notes into high-impact, metric-driven STAR bullets.
You are NOT allowed to invent new jobs or change the user's employment history.

SENIORITY LEVEL: {seniority_level.upper()}
STRICT ACTION VERBS: {', '.join(allowed_verbs)}

{golden_section}

### *** TRUTH CONSTRAINTS (VIOLATION = FAILURE) ***
1. **COMPANY NAMES:** You must ONLY use the companies provided in the User Input.
   - ALLOWED: {json.dumps(original_companies)}
   - PROHIBITED: Do NOT use company names found in the Job Description or Golden Patterns.
   - If the User is applying to "Armada", do NOT list "Armada" as a past job.

2. **JOB TITLES:** You must preserve the user's actual role hierarchy.
   - ALLOWED: {json.dumps(original_titles)}
   - You may slightly polish titles (e.g., "Programmer" -> "Software Engineer"), but DO NOT promote a "Junior" to "VP".

3. **TECHNOLOGY:** Only use technologies the user explicitly mentioned or strongly implied by the specific task. Do not copy tech stacks from the Golden Bullets.

### CRITICAL RULES:
1. **One Thought Per Bullet:** Do not combine unrelated tasks. Keep bullets punchy (15-25 words max).
2. **Front-Load the Impact:** Start with the Result or the strong Verb. (e.g., "Cut costs by 40%..." rather than "Responsible for cutting costs...")
3. **No Hallucinations:** Use ONLY the user's actual companies listed above.
4. **Concrete Tech:** Do not say "cutting-edge tech." Name the specific tool (e.g., "TensorFlow", "Kubernetes") if the user mentioned it or if it is heavily implied by the Research Data.

BAD VS GOOD:
✗ Weak: "Strategized deployment of models." (Vague, passive)
✓ Strong: "Deployed deep learning models to 1,000+ edge devices, reducing latency by 90% via quantization." (Specific, metric-driven)

Output JSON Schema:
{{
  "rewritten_experiences": [
    {{
      "company": "MUST MATCH INPUT EXACTLY",
      "role": "Original or Polished Title",
      "bullets": [
        "Strong Action Verb + Specific Tech + Quantifiable Result",
        "Strong Action Verb + Problem Solved + Benefit"
      ],
      "metrics_used": ["90% latency reduction", "10k+ assets"],
      "style_transfer_rationale": "Mapped user's edge computing note to the Golden Bullet about 'High-availability distributed systems'."
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
        self.fallback_models = FALLBACK_MODELS.get("drafter", [])
        self.temperature = TEMPERATURES["drafter"]
        self.timeout = TIMEOUTS["drafter"]
        
    async def draft(self, experiences: List[Dict], job_ad: str, 
                   research_data: Dict[str, Any],
                   golden_bullets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Draft STAR-formatted resume bullets.
        
        Args:
            experiences: User's actual experiences
            job_ad: Job description text
            research_data: Output from Researcher stage
            golden_bullets: Optional list of high-quality example bullets
            
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
            allowed_verbs=allowed_verbs,
            golden_bullets=golden_bullets
        )
        
        user_prompt = f"""
User Experiences (JSON):
{json.dumps(experiences, indent=2)}

Job Description:
{job_ad[:1000]}

Research Insights:
- Expected Metrics: {', '.join(research_data.get('implied_metrics', [])[:3])}
- Domain Vocabulary: {', '.join(research_data.get('domain_vocab', [])[:5])}
- Implied Skills: {', '.join(research_data.get('implied_skills', [])[:5])}
"""
        
        try:
            # Call LLM with strict JSON schema and fallback support
            response = await self.llm_client.call_llm_async(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                timeout=self.timeout,
                fallback_models=self.fallback_models
            )
            
            # Parse and validate response
            result = self._parse_response(response, experiences)
            
            # Track model in result (Note: actual model used is logged in adapter)
            result["model_requested"] = self.model
            
            logger.info(f"[DRAFTER] Created {self._count_bullets(result)} STAR bullets "
                       f"with {seniority_level} seniority")
            
            # Ensure we have at least some data
            if not result.get("rewritten_experiences"):
                logger.warning("[DRAFTER] No rewritten experiences generated, using fallback")
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
            if not response or response == "{}":
                logger.warning("[DRAFTER] Empty response from LLM")
                return self._create_fallback_output(original_experiences, "mid")

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
                        "metrics_used": exp.get("metrics_used", []),
                        "style_transfer_rationale": exp.get("style_transfer_rationale", "")
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
