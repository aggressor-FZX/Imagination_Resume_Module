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
import re
from typing import Dict, Any, List, Optional
from pipeline_config import OR_SLUG_DRAFTER, FALLBACK_MODELS, TEMPERATURES, TIMEOUTS, get_seniority_config

logger = logging.getLogger(__name__)


def _unwrap_markdown_json_fence(text: str) -> str:
    """Strip ``` / ```json fences when providers wrap JSON despite instructions."""
    if not text:
        return text
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    body: List[str] = []
    for i, line in enumerate(lines):
        if i == 0 and line.strip().startswith("```"):
            continue
        body.append(line)
    s = "\n".join(body).strip()
    if "```" in s:
        s = s[: s.rfind("```")].strip()
    return s


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

def create_drafter_prompt(experiences: List[Dict], job_ad: str, research_data: Dict,
                         seniority_level: str, allowed_verbs: List[str],
                         golden_bullets: Optional[List[str]] = None,
                         tone_instruction: str = "Maintain a standard, professional corporate tone.",
                         extracted_job_title: Optional[str] = None) -> str:
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
    
    title_context = ""
    if extracted_job_title:
        title_context = f"\nTARGET JOB TITLE: {extracted_job_title}\nOptimize the resume to target this specific role."

    system_prompt = f"""You are an elite Technical Resume Writer. Your goal is to rewrite the user's raw notes into high-impact, metric-driven STAR bullets.
You are NOT allowed to invent new jobs or change the user's employment history.

SENIORITY LEVEL: {seniority_level.upper()}
STRICT ACTION VERBS: {', '.join(allowed_verbs)}
TONE INSTRUCTION: {tone_instruction}{title_context}

{golden_section}

### *** TRUTH CONSTRAINTS (VIOLATION = FAILURE) ***
1. **COMPANY NAMES:** You must ONLY use the companies provided in the User Input.
   - ALLOWED: {json.dumps(original_companies) if original_companies else '["Current Employer"]'}
   - PROHIBITED: Do NOT invent company names like "Tech Innovations Inc", "Data Solutions LLC", "ABC Corp", "Acme Corp", or any company NOT in the input.
   - If no specific company is provided, use "Current Employer" as the company name.

2. **JOB TITLES:** You must preserve the user's actual role hierarchy.
   - ALLOWED: {json.dumps(original_titles) if original_titles else '["Professional"]'}
   - You may slightly polish titles (e.g., "Programmer" -> "Software Engineer"), but DO NOT promote a "Junior" to "VP".

3. **TECHNOLOGY:** Only use technologies the user explicitly mentioned or strongly implied by the specific task. Do not copy tech stacks from the Golden Bullets.

4. **NO PLAGIARISM FROM JOB DESCRIPTION:**
   - The Job Description is provided ONLY for context on what is relevant.
   - DO NOT copy sentences, requirements, or specific projects from the Job Description into the resume.
   - DO NOT claim to have done the specific tasks listed as "Responsibilities" in the Job Ad unless the user's notes explicitly support it.

5. **NO FABRICATED METRICS:** Only include metrics (%, $, time) that appear in the user's input or are reasonable extrapolations. Do NOT invent "1 million users", "500K customers", "99.9% uptime" etc. unless explicitly provided.

### CRITICAL RULES:
1. **One Thought Per Bullet:** Do not combine unrelated tasks. Keep bullets punchy (15-25 words max).
2. **Front-Load the Impact:** Start with the Result or the strong Verb. (e.g., "Cut costs by 40%..." rather than "Responsible for cutting costs...")
3. **No Hallucinations:** Use ONLY the user's actual companies listed above.
4. **Concrete Tech:** Do not say "cutting-edge tech." Name the specific tool (e.g., "TensorFlow", "Kubernetes") if the user mentioned it or if it is heavily implied by the Research Data.

BAD VS GOOD:
✗ Weak: "Strategized deployment of models." (Vague, passive)
✓ Strong: "Deployed deep learning models to 1,000+ edge devices, reducing latency by 90% via quantization." (Specific, metric-driven)
✗ Hallucinated (BAD): Copying "Must have 5 years of Python" from Job Ad as "I have 5 years of Python". (DO NOT DO THIS)

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
        
    async def draft(
        self,
        experiences: List[Dict],
        job_ad: str,
        research_data: Dict[str, Any],
        golden_bullets: Optional[List[str]] = None,
        tone_instruction: Optional[str] = None,
        temperature_override: Optional[float] = None,
        extracted_job_title: Optional[str] = None,
    ) -> Dict[str, Any]:
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
        
        # Check for sparse input to encourage safe expansion
        total_input_words = sum(len(str(exp).split()) for exp in experiences)
        is_sparse_input = total_input_words < 50
        
        sparse_instruction = ""
        if is_sparse_input:
             logger.info("[DRAFTER] Sparse input detected, adding expansion instruction")
             sparse_instruction = "\nNOTE: The user's notes are very brief. You must EXPAND them into full bullets by inferring standard industry tasks associated with these roles/skills. Focus on HOW the work was likely done, but DO NOT invent specific metrics or projects found in the Job Description."

        # Create prompt
        system_prompt = create_drafter_prompt(
            experiences=experiences,
            job_ad=job_ad,
            research_data=research_data,
            seniority_level=seniority_level,
            allowed_verbs=allowed_verbs,
            golden_bullets=golden_bullets,
            tone_instruction=tone_instruction or "Maintain a standard, professional corporate tone.",
            extracted_job_title=extracted_job_title,
        ) + sparse_instruction
        
        user_prompt = f"""
User Experiences (JSON) - THIS IS THE SOURCE OF TRUTH:
{json.dumps(experiences, indent=2)}

Target Job Description - FOR CONTEXT ONLY (DO NOT COPY):
{job_ad[:500]}

Research Insights:
- Expected Metrics: {', '.join(research_data.get('implied_metrics', [])[:3])}
- Domain Vocabulary: {', '.join(research_data.get('domain_vocab', [])[:5])}
- Implied Skills: {', '.join(research_data.get('implied_skills', [])[:5])}
"""
        
        try:
            # Call LLM with strict JSON schema and fallback support
            temperature = (
                temperature_override if temperature_override is not None else self.temperature
            )
            response = await self.llm_client.call_llm_async(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.model,
                temperature=temperature,
                response_format={"type": "json_object"},
                timeout=self.timeout,
                fallback_models=self.fallback_models,
                max_tokens=6000,
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
            response = _unwrap_markdown_json_fence(response)
            # Handle empty, whitespace-only, or empty JSON responses
            if not response or not response.strip() or response.strip() == "{}":
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
            for i, exp in enumerate(rewritten):
                if isinstance(exp, dict):
                    # Preserve duration and location from original experience if available
                    original_exp = original_experiences[i] if i < len(original_experiences) else {}
                    validated = {
                        "company": exp.get("company", ""),
                        "role": exp.get("role", ""),
                        "duration": exp.get("duration") or original_exp.get("duration") or original_exp.get("dates", ""),
                        "location": exp.get("location") or original_exp.get("location", ""),
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
                if any(char in bullet for char in ['%', '$', ' x ', '+', '>', '<', '=', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            )
            # Refined check for quantification: must have a number or specific symbols
            quantified_bullets_refined = 0
            placeholder_regex = re.compile(r"\[.*?\]|\{.*?\}", re.IGNORECASE)
            has_placeholders = False

            for exp in validated_experiences:
                payload_bullets = exp.get("bullets", [])
                for bullet in payload_bullets:
                    # Check for placeholders
                    if placeholder_regex.search(bullet):
                        has_placeholders = True
                    
                    # More robust quantification check
                    # Look for numbers (excluding years like 2024) or symbols
                    if re.search(r"\d+(?!\d{3})|[%$x+><=]", bullet):
                        quantified_bullets_refined += 1

            quantification_score = (
                quantified_bullets_refined / total_bullets if total_bullets > 0 else 0.0
            )

            return {
                "rewritten_experiences": validated_experiences,
                "seniority_applied": data.get("seniority_applied", ""),
                "total_bullets": total_bullets,
                "quantified_bullets": quantified_bullets_refined,
                "quantification_score": round(quantification_score, 3),
                "has_placeholders": has_placeholders,
                "fallback": False
            }
            
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
            logger.error(f"[DRAFTER] Raw response length: {len(response) if response else 0}")
            logger.error(f"[DRAFTER] Raw response preview: {(response or '')[:500]}")
            
            # Try to extract JSON from text
            json_match = self._extract_json_from_text(response)
            if json_match:
                logger.info("[DRAFTER] Successfully extracted JSON from text")
                return self._parse_response(json_match, original_experiences)
            
            # Return fallback with error flag
            logger.warning("[DRAFTER] Using fallback output due to JSON parse failure")
            fallback = self._create_fallback_output(original_experiences, "mid")
            fallback["fallback_used"] = True
            fallback["parse_error"] = str(e)
            return fallback
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """
        Extract JSON object from text that may contain extra content.
        Handles markdown code blocks like ```json {...}``` or ```{...}```
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            JSON string if found, None otherwise
        """
        if not text:
            return None
            
        # First, try to find markdown code blocks
        # Pattern: ```json ... ``` or ``` ... ```
        import re
        
        # Try ```json {...}``` pattern (greedy match for full JSON)
        json_block_match = re.search(r'```json\s*(\{.*\})\s*```', text, re.DOTALL | re.IGNORECASE)
        if json_block_match:
            candidate = json_block_match.group(1)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
        
        # Try ```{...}``` pattern (no explicit json, greedy match)
        code_block_match = re.search(r'```\s*(\{.*\})\s*```', text, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            candidate = code_block_match.group(1)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON after "Here's the JSON output:" or similar
        json_after_colon = re.search(r'(?:Here\'s the JSON output:|JSON output:|```json)\s*(\{.*\})', text, re.DOTALL | re.IGNORECASE)
        if json_after_colon:
            candidate = json_after_colon.group(1)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
        
        # Fallback: Look for JSON object boundaries
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            try:
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
            duration = exp.get("duration") or exp.get("dates", "")
            location = exp.get("location", "")
            
            fallback_experiences.append({
                "company": company or "Previous Company",
                "role": role or "Professional Role",
                "duration": duration,
                "location": location,
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
