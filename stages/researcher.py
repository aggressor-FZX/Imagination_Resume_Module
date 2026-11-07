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

### 2. SKILL BRIDGES (the only inference you may propose)
Identify general, transferable-skill labels that the candidate's own stated
activity plausibly supports but did not name.

HARD REQUIREMENTS for every bridge:
1. `evidence_quote` MUST be copied VERBATIM from the candidate material supplied
   in the user message. Do not paraphrase it, shorten it mid-word, add ellipses,
   or invent a sentence. If you cannot quote it exactly, do not propose it.
2. `skill` MUST be a short, general skill label (2-4 words). It must NOT contain a
   specific implementation, algorithm, named software product, formal methodology
   certification, metric, proficiency level, seniority or leadership claim.
3. `rationale` MUST be one short sentence explaining what activity the quote
   describes and why the general label follows from it.

ALLOWED (general label from a real activity):
* quoted: "Compared weekly sales figures to identify trends"
  skill: "trend analysis"
* quoted: "analyzed measurement series over time"
  skill: "time-series analysis"

NOT ESTABLISHED (do not propose these):
* "ARIMA forecasting" from unspecified analysis  -> a specific algorithm
* "Python" from a physics or analyst job title   -> names a tool, not an activity
* "10+ years of SQL" from a job requirement      -> a requirement is not evidence
* "expert in...", "led a team of..."             -> proficiency/seniority claim

Return at most 6 bridges. Zero bridges is a correct and common answer.

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
  "skill_bridges": [
    {
      "skill": "trend analysis",
      "evidence_quote": "Compared weekly sales figures to identify trends",
      "rationale": "The described activity identifies changes over time, which is trend analysis.",
      "source_id": "document-0"
    }
  ],
  "implied_skills": [],
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
2. "implied_skills" is DEPRECATED: always return an empty array. All inference
   must go into "skill_bridges" with a verbatim quote.
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
        source_documents: Optional[List[Dict[str, Any]]] = None,
        skill_inference_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze job ad and user profile, proposing candidate-anchored skill bridges.
        
        Args:
            job_ad: Job description text
            experiences: Optional list of user experiences
            source_documents: Immutable candidate documents ``[{source_id, text}]``.
                These are the ONLY quotable evidence for a bridge. When omitted,
                the serialized experiences are used instead and no bridge can be
                validated against an unknown document.
            skill_inference_mode: Permission policy; ``conservative`` returns no
                proposals at all.
            
        Returns:
            Dictionary with skill_bridges proposals, implied_metrics, domain_vocab
            and insider_tips
        """
        from skill_inference import SkillInferenceMode, normalize_mode

        logger.info("[RESEARCHER] Analyzing job ad")

        if normalize_mode(skill_inference_mode) is SkillInferenceMode.CONSERVATIVE:
            # Defence in depth: the orchestrator does not call this stage in
            # Conservative mode, and this guard keeps a direct caller from
            # obtaining proposals the policy does not permit.
            return {
                "skill_bridges": [],
                "implied_skills": [],
                "implied_metrics": [],
                "domain_vocab": [],
                "work_archetypes": [],
                "insider_tips": "",
                "research_status": "skipped_by_policy",
            }

        user_prompt = f"Job Description (requirements only, NOT evidence):\n\n{job_ad[:1500]}"

        # Quotable candidate evidence. Each document is labelled with the id a
        # bridge must cite, and is bounded so one huge upload cannot crowd out
        # the job description or exceed the provider budget.
        quotable: List[Dict[str, str]] = []
        for doc in source_documents or []:
            if not isinstance(doc, dict):
                continue
            sid = doc.get("source_id")
            text = doc.get("text")
            if isinstance(sid, str) and sid.strip() and isinstance(text, str) and text.strip():
                quotable.append({"source_id": sid, "text": text[:6000]})

        if quotable:
            user_prompt += (
                "\n\nCANDIDATE EVIDENCE — the ONLY source you may quote.\n"
                "Every evidence_quote must appear character-for-character in one of\n"
                "these documents, and source_id must name that document.\n"
            )
            for doc in quotable:
                user_prompt += f"\n--- source_id: {doc['source_id']} ---\n{doc['text']}\n"
        elif experiences:
            exp_text = json.dumps(experiences, indent=2)
            user_prompt += f"\n\nUser Profile/Experiences:\n{exp_text[:4000]}"
        
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
                max_tokens=2048  # Increased to preserve richer job-ad + experience context
            )
            
            # Parse and validate response
            result = self._parse_response(response)
            result["model_used"] = self.model # Track model in result
            logger.info("[RESEARCHER] Extraction complete")

            # NOTE: no default qualifications are injected on a thin response.
            # The previous code filled `implied_metrics`/`domain_vocab` with
            # generic strings, which then behaved like findings. An empty list is
            # the honest answer; Phase B tests assert the old defaults are gone.
            return result
            
        except Exception:
            logger.error("[RESEARCHER] Failed to analyze job ad")
            # A failure returns NO proposals or invented qualifications.
            return {
                "skill_bridges": [],
                "implied_skills": [],
                "implied_metrics": [],
                "domain_vocab": [],
                "work_archetypes": [],
                "insider_tips": "",
                "error": "research_failed",
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
                # Explicit marker: the stage now reports its own empty outcome.
                # Previously this returned a distinctive default-shape bundle and
                # the orchestrator had to guess by matching those strings, which
                # stopped working once the invented defaults were removed.
                return {
                    "skill_bridges": [],
                    "implied_metrics": [],
                    "domain_vocab": [],
                    "implied_skills": [],
                    "work_archetypes": [],
                    "insider_tips": "",
                    "parse_error": "empty_response",
                }

            data = json.loads(response)
            
            # Ensure required fields exist
            if not isinstance(data, dict):
                raise ValueError("Response is not a JSON object")

            if data.get("error") or data.get("fallback") or data.get("fallback_used"):
                return {
                    "skill_bridges": [], "implied_skills": [], "implied_metrics": [],
                    "domain_vocab": [], "work_archetypes": [], "insider_tips": "",
                    "error": "research_provider_failed", "fallback_used": True,
                }

            if not any(key in data for key in ("skill_bridges", "implied_metrics", "domain_vocab", "work_archetypes")):
                return {
                    "skill_bridges": [], "implied_skills": [], "implied_metrics": [],
                    "domain_vocab": [], "work_archetypes": [], "insider_tips": "",
                    "parse_error": "missing_research_fields", "fallback_used": True,
                }

            if any(key in data and not isinstance(data[key], list) for key in
                   ("skill_bridges", "implied_metrics", "domain_vocab", "implied_skills", "work_archetypes")):
                return {
                    "skill_bridges": [], "implied_skills": [], "implied_metrics": [],
                    "domain_vocab": [], "work_archetypes": [], "insider_tips": "",
                    "parse_error": "invalid_research_fields", "fallback_used": True,
                }
            
            # Set defaults for missing fields. NOTE: no invented qualifications —
            # an absent list stays absent rather than becoming a claimed finding.
            result = {
                "skill_bridges": data.get("skill_bridges", []),
                "implied_metrics": data.get("implied_metrics", []),
                "domain_vocab": data.get("domain_vocab", []),
                # Deprecated in the prompt; kept only so old callers do not break.
                "implied_skills": data.get("implied_skills", []),
                "work_archetypes": data.get("work_archetypes", []),
                "has_projects_section": data.get("has_projects_section", False),
                "is_student_or_career_changer": data.get("is_student_or_career_changer", False),
                "insider_tips": data.get("insider_tips", "")
            }
            
            # Validate types
            for _list_key in ("skill_bridges", "implied_metrics", "domain_vocab",
                              "implied_skills", "work_archetypes"):
                if not isinstance(result[_list_key], list):
                    result[_list_key] = []
            if not isinstance(result["insider_tips"], str):
                result["insider_tips"] = ""
                
            return result
            
        except json.JSONDecodeError:
            logger.error("[RESEARCHER] Failed to parse JSON response")
            
            # Try to extract JSON from text
            json_match = self._extract_json_from_text(response)
            if json_match:
                logger.info("[RESEARCHER] Successfully extracted JSON from text")
                return self._parse_response(json_match)
            
            # Return NO proposals or invented qualifications, with an error flag.
            logger.warning("[RESEARCHER] Using empty result due to JSON parse failure")
            return {
                "skill_bridges": [],
                "implied_metrics": [],
                "domain_vocab": [],
                "implied_skills": [],
                "work_archetypes": [],
                "insider_tips": "",
                "parse_error": "invalid_json",
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
