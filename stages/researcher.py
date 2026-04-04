"""
Stage 1: Researcher - Extracts metrics and domain vocabulary from job ads

Based on Alternate_flow_proposal.md recommendations:
- Extracts implied metrics/benchmarks from job description
- Provides domain-specific vocabulary
- Uses strict JSON schema to prevent analysis leakage
"""

import json
import logging
from typing import Dict, Any, Optional, List
from pipeline_config import OR_SLUG_RESEARCHER, TEMPERATURES, TIMEOUTS

logger = logging.getLogger(__name__)


def _unwrap_markdown_json_fence(text: str) -> str:
    """Strip ``` / ```json fences when providers wrap JSON despite json_object mode."""
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

RESEARCHER_SYSTEM_PROMPT = """You are an expert Career Transition Strategist. Your goal is to "bridge the gap" between a candidate's past experience and a target role, even if they seem unrelated.

### INPUT DATA:
1. **User Profile:** The candidate's resume/experience.
2. **Target Job:** The job description.

### CORE TASK:
Analyze the text to identify "Work Archetypes" and "Transferable Bridges." 
You must translate the user's *past* domain language into the *target* domain language.

### 1. SYNTACTIC ARCHETYPE IDENTIFICATION
Identify the primary "Work Archetypes" required by this job. Choose 1-2:
["Migration", "Greenfield Build", "Scaling/Optimization", "Turnaround/Rescue", "Maintenance/Stability"].
*Why? This tells the Drafter what tone to use.*

### 2. THE "LATERAL BRIDGE" (Crucial)
Identify skills the user *has* but didn't explicitly name using the target's vocabulary.
*   *Logic:* If Job requires X, and User did Y, and Y is mathematically/logically similar to X, that is a Match.
*   *Example:* User did "Sonar Analysis" (Physics). Job needs "Time-Series Anomaly Detection" (Data). 
*   *Output:* "Time-Series Analysis (Implied by Sonar work - handling noisy, continuous signal data)."

### 3. PROJECTS SECTION HANDLING (Students/Career Changers)
*   If the user is a student or career changer, identify any "Projects" section in their profile.
*   Treat "Projects" with EQUAL WEIGHT to "Professional Experience".
*   Extract skills, technologies, and achievements from projects just as you would from work experience.
*   Flag projects that demonstrate relevant competencies for the target role.

### 4. IMPLIED METRICS & DOMAIN VOCAB
*   Extract quantifiable metrics from the Job Ad (e.g., "handle 1M+ requests").
*   Extract specific tech stack keywords (e.g., "Kubernetes", "Snowflake").

### OUTPUT JSON SCHEMA:
{
  "work_archetypes": ["Optimization", "Migration"],
  "implied_skills": [
    "Skill Name (Bridge Explanation)", 
    "Example: Predictive Modeling (Implied by User's Physics Simulation experience - Monte Carlo methods map directly to stochastic forecasting)"
  ],
  "implied_metrics": [
    "Must be able to handle [Metric] (Context: [Job Requirement])"
  ],
  "domain_vocab": ["List", "Of", "Hard", "Keywords"],
  "has_projects_section": true/false,
  "is_student_or_career_changer": true/false,
  "insider_tips": "2 sentences on how to frame the past experience to look like the future requirement."
}

CRITICAL RULES:
1. Do NOT browse the live web. Use the provided text only.
2. Be aggressive in finding connections. If a physicist applies for a Data job, find the math connection.
3. Return VALID JSON only.
"""

# ============================================================================
# RESEARCHER CLASS
# ============================================================================

class Researcher:
    """Stage 1: Extracts metrics and domain vocabulary from job ads."""
    
    def __init__(self, llm_client):
        """
        Initialize Researcher stage.
        
        Args:
            llm_client: LLM client for making API calls
        """
        self.llm_client = llm_client
        self.model = OR_SLUG_RESEARCHER
        self.temperature = TEMPERATURES["researcher"]
        self.timeout = TIMEOUTS["researcher"]
        
    async def analyze(
        self,
        job_ad: str,
        experiences: Optional[List[Dict]] = None,
        temperature_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Analyze job ad and user profile to extract metrics, vocabulary, and implied skills.
        
        Args:
            job_ad: Job description text
            experiences: Optional list of user experiences
            
        Returns:
            Dictionary with implied_metrics, domain_vocab, implied_skills, and insider_tips
        """
        logger.info(f"[RESEARCHER] Analyzing job ad ({len(job_ad)} chars)")
        
        user_prompt = f"Job Description:\n\n{job_ad[:2000]}"
        
        if experiences:
            exp_text = json.dumps(experiences, indent=2)
            user_prompt += f"\n\nUser Profile/Experiences:\n{exp_text[:2000]}"
            user_prompt += "\n\nBased on the user's profile and the job description, what IMPLIED skills do they likely possess?"
        
        try:
            # Call LLM with strict JSON schema and optimized parameters
            temperature = (
                temperature_override if temperature_override is not None else self.temperature
            )
            response = await self.llm_client.call_llm_async(
                system_prompt=RESEARCHER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model=self.model,
                temperature=temperature,
                response_format={"type": "json_object"},
                timeout=self.timeout,
                max_tokens=1024  # Token limit for Perplexity Sonar Pro output
            )
            
            # Parse and validate response
            result = self._parse_response(response)
            result["model_used"] = self.model # Track model in result
            logger.info(f"[RESEARCHER] Extracted {len(result.get('implied_metrics', []))} metrics, "
                       f"{len(result.get('domain_vocab', []))} domain terms")
            
            # Ensure we have at least some data
            if not result.get("implied_metrics"):
                result["implied_metrics"] = ["Demonstrate measurable impact", "Show technical proficiency"]
            if not result.get("domain_vocab"):
                result["domain_vocab"] = ["Relevant technical skills"]
            
            return result
            
        except Exception as e:
            logger.error(f"[RESEARCHER] Failed to analyze job ad: {e}")
            # Return safe defaults
            return {
                "implied_metrics": ["Improve system performance", "Scale to handle increased load"],
                "domain_vocab": ["Technical skills relevant to the role"],
                "insider_tips": "Focus on quantifiable achievements and relevant technologies.",
                "error": str(e)
            }
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response and ensure it matches expected schema.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed and validated research data
        """
        try:
            # Handle empty, whitespace-only, or empty JSON responses
            response = _unwrap_markdown_json_fence(response)
            if not response or not response.strip() or response.strip() == "{}":
                logger.warning("[RESEARCHER] Empty response from LLM, using defaults")
                return {
                    "implied_metrics": ["Improve system performance", "Scale to handle increased load"],
                    "domain_vocab": ["Technical skills relevant to the role"],
                    "implied_skills": ["Problem Solving", "Communication"],
                    "work_archetypes": [],
                    "insider_tips": "Focus on quantifiable achievements and relevant technologies."
                }

            data = json.loads(response)
            
            # Ensure required fields exist
            if not isinstance(data, dict):
                raise ValueError("Response is not a JSON object")
            
            # Set defaults for missing fields
            result = {
                "implied_metrics": data.get("implied_metrics", []),
                "domain_vocab": data.get("domain_vocab", []),
                "implied_skills": data.get("implied_skills", []),
                "work_archetypes": data.get("work_archetypes", []),
                "has_projects_section": data.get("has_projects_section", False),
                "is_student_or_career_changer": data.get("is_student_or_career_changer", False),
                "insider_tips": data.get("insider_tips", "Focus on relevant achievements.")
            }
            
            # Validate types
            if not isinstance(result["implied_metrics"], list):
                result["implied_metrics"] = []
            if not isinstance(result["domain_vocab"], list):
                result["domain_vocab"] = []
            if not isinstance(result["implied_skills"], list):
                result["implied_skills"] = []
            if not isinstance(result["work_archetypes"], list):
                result["work_archetypes"] = []
            if not isinstance(result["insider_tips"], str):
                result["insider_tips"] = "Focus on relevant achievements."
            
            # Ensure we have at least some data
            if not result["implied_metrics"]:
                result["implied_metrics"] = ["Demonstrate measurable impact", "Show technical proficiency"]
            if not result["domain_vocab"]:
                result["domain_vocab"] = ["Relevant technical skills"]
            if not result["implied_skills"]:
                result["implied_skills"] = ["Problem Solving", "Communication"]
                
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"[RESEARCHER] Failed to parse JSON response: {e}")
            logger.error(f"[RESEARCHER] Raw response length: {len(response) if response else 0}")
            logger.error(f"[RESEARCHER] Raw response preview: {(response or '')[:500]}")
            
            # Try to extract JSON from text
            json_match = self._extract_json_from_text(response)
            if json_match:
                logger.info("[RESEARCHER] Successfully extracted JSON from text")
                return self._parse_response(json_match)
            
            # Return defaults with error flag
            logger.warning("[RESEARCHER] Using fallback defaults due to JSON parse failure")
            return {
                "implied_metrics": ["Improve system performance", "Scale to handle increased load"],
                "domain_vocab": ["Technical skills relevant to the role"],
                "implied_skills": ["Problem Solving", "Communication"],
                "work_archetypes": [],
                "insider_tips": "Focus on quantifiable achievements and relevant technologies.",
                "parse_error": str(e),
                "fallback_used": True
            }
    
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
    
    def get_metrics_summary(self, research_data: Dict[str, Any]) -> str:
        """
        Create a human-readable summary of research findings.
        
        Args:
            research_data: Research data from analyze()
            
        Returns:
            Formatted summary string
        """
        metrics = research_data.get("implied_metrics", [])
        vocab = research_data.get("domain_vocab", [])
        tips = research_data.get("insider_tips", "")
        
        summary = f"Research Analysis:\n"
        summary += f"- Expected Metrics: {', '.join(metrics[:3])}\n"
        summary += f"- Key Technologies: {', '.join(vocab[:5])}\n"
        summary += f"- Insider Tip: {tips}\n"
        
        return summary
