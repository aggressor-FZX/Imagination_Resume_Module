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

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

RESEARCHER_SYSTEM_PROMPT = """You are a career research agent. Analyze the Job Description and User Profile to find technical benchmarks and implied skills.
DO NOT summarize the job. Extract ONLY quantifiable metrics, domain vocabulary, and implied skills.

Output JSON Schema:
{
  "implied_metrics": ["40% reduction in latency", "99.9% uptime", "10k+ concurrent users"],
  "domain_vocab": ["Kubernetes", "PyTorch", "CI/CD", "Microservices"],
  "implied_skills": ["Docker (implied by Kubernetes)", "System Design (implied by Senior role)", "Mentorship"],
  "insider_tips": "Focus on scale and high-availability architecture."
}

CRITICAL RULES:
1. Extract ONLY from the job description text and user profile
2. Provide 3-5 specific, quantifiable metrics
3. List 5-10 domain-specific keywords
4. Identify 3-5 IMPLIED skills that the user likely has based on their experience but didn't explicitly list
5. Keep insider_tips concise (1-2 sentences)
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
        
    async def analyze(self, job_ad: str, experiences: Optional[List[Dict]] = None) -> Dict[str, Any]:
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
            response = await self.llm_client.call_llm_async(
                system_prompt=RESEARCHER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                timeout=self.timeout,
                max_tokens=1024  # Ensure sufficient tokens for complex JSON
            )
            
            # Parse and validate response
            result = self._parse_response(response)
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
            data = json.loads(response)
            
            # Ensure required fields exist
            if not isinstance(data, dict):
                raise ValueError("Response is not a JSON object")
            
            # Set defaults for missing fields
            result = {
                "implied_metrics": data.get("implied_metrics", []),
                "domain_vocab": data.get("domain_vocab", []),
                "implied_skills": data.get("implied_skills", []),
                "insider_tips": data.get("insider_tips", "Focus on relevant achievements.")
            }
            
            # Validate types
            if not isinstance(result["implied_metrics"], list):
                result["implied_metrics"] = []
            if not isinstance(result["domain_vocab"], list):
                result["domain_vocab"] = []
            if not isinstance(result["implied_skills"], list):
                result["implied_skills"] = []
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
            logger.debug(f"[RESEARCHER] Raw response: {response[:500]}")
            
            # Try to extract JSON from text
            json_match = self._extract_json_from_text(response)
            if json_match:
                return self._parse_response(json_match)
            
            # Return defaults
            return {
                "implied_metrics": ["Improve system performance", "Scale to handle increased load"],
                "domain_vocab": ["Technical skills relevant to the role"],
                "insider_tips": "Focus on quantifiable achievements and relevant technologies.",
                "parse_error": str(e)
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