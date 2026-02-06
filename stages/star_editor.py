"""
Stage 3: StarEditor - Polishes and formats final resume with hallucination guard

Based on Alternate_flow_proposal.md recommendations:
- Converts STAR data into clean, ATS-friendly Markdown
- Applies hallucination guard to detect and remove placeholder content
- Ensures no analysis metadata leaks into final output
- Provides editorial notes for quality assurance
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional
from pipeline_config import OR_SLUG_STAR_EDITOR, TEMPERATURES, TIMEOUTS, contains_hallucination

logger = logging.getLogger(__name__)

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

STAR_EDITOR_SYSTEM_PROMPT = """You are a professional resume editor. Convert the STAR bullet data into a high-end, ATS-friendly Markdown resume.

### THE FORMULA:
Your output must be a clean, reverse-chronological list of experiences.
Each experience MUST follow this visual structure:
## **Job Title** | **Company Name**
*Dates | Location*
- Achievement Bullet 1
- Achievement Bullet 2

STRICT RULES:
1. **NO BLOCKS OF TEXT:** Do not write paragraphs. Every achievement must be a distinct bullet point starting with a hyphen (-).
2. **DELETE ALL LABELS:** Remove all "Situation:", "Task:", "Action:", "Result:", "STAR:", "CAR:" labels.
3. **MANDATORY QUANTIFICATION:** Ensure every bullet point includes a metric (%, $, time, count).
4. **STYLE TRANSFER:** Maintain the punchy, technical syntax from the input.
5. **NO LEAKAGE:** Do not include any meta-commentary, "editorial notes" within the markdown, or analysis results. Output ONLY the resume content.
6. **ANTI-HALLUCINATION:** NEVER list the Target Company (the company from the Job Ad) as an employer. You must ONLY use the company names provided in the user's input data.

Output JSON Schema:
{
  "final_markdown": "## **Software Engineer** | **Google**\\n*2020 - Present*\\n- Deployed CI/CD...",
  "final_plain_text": "Software Engineer | Google...",
  "editorial_notes": "Polished for ATS compliance. Ensured bulleted list format."
}
"""

# ============================================================================
# STAR EDITOR CLASS
# ============================================================================

class StarEditor:
    """Stage 3: Polishes and formats final resume with hallucination guard."""
    
    def __init__(self, llm_client):
        """
        Initialize StarEditor stage.
        
        Args:
            llm_client: LLM client for making API calls
        """
        self.llm_client = llm_client
        self.model = OR_SLUG_STAR_EDITOR
        self.temperature = TEMPERATURES["star_editor"]
        self.timeout = TIMEOUTS["star_editor"]
        
    async def polish(self, draft_data: Dict[str, Any], 
                    research_data: Dict[str, Any],
                    temperature_override: Optional[float] = None) -> Dict[str, Any]:
        """
        Polish STAR draft into final resume format with TARGET COMPANY GUARD.
        
        Args:
            draft_data: Output from Drafter stage
            research_data: Output from Researcher stage
            temperature_override: Optional temperature setting to override default
            
        Returns:
            Dictionary with final_markdown, final_plain_text, and editorial_notes
        """
        logger.info("[STAR_EDITOR] Polishing draft into final resume")
        
        # Prepare input data
        experiences = draft_data.get("rewritten_experiences", [])
        seniority = draft_data.get("seniority_applied", "mid")
        domain_vocab = research_data.get("domain_vocab", [])
        
        # EXTRACT TARGET COMPANY from Research Data (e.g., "Armada")
        target_company = research_data.get("company_name", "")
        
        # Create prompt with EXPLICIT warning about the target company
        user_prompt = self._create_user_prompt(experiences, seniority, domain_vocab, target_company)
        
        try:
            # Call LLM with strict JSON schema
            temperature = temperature_override if temperature_override is not None else self.temperature
            response = await self.llm_client.call_llm_async(
                system_prompt=STAR_EDITOR_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model=self.model,
                temperature=temperature,
                response_format={"type": "json_object"},
                timeout=self.timeout
            )
            
            # Parse and validate response
            result = self._parse_response(response)
            result["model_used"] = self.model # Track model in result
            
            # Apply hallucination guard
            # --- CRITICAL FIX: Pass Target Company to Guard ---
            result = self._apply_hallucination_guard(
                result, 
                experiences, 
                target_company=target_company
            )
            # --------------------------------------------------
            
            # Add metadata
            result.update({
                "seniority_level": seniority,
                "domain_terms_used": self._extract_domain_terms(result.get("final_markdown", ""), domain_vocab),
                "quantification_check": self._check_quantification(result.get("final_markdown", ""))
            })
            
            # Ensure we have at least some data
            if not result.get("final_markdown"):
                result["final_markdown"] = "## Professional Experience\n\nNo resume content generated."
                result["final_plain_text"] = "No resume content generated."
                result["editorial_notes"] = "Fallback resume generated due to processing error."
                result["fallback"] = True
            
            logger.info(f"[STAR_EDITOR] Created polished resume with {len(result.get('final_markdown', '').split())} words")
            
            return result
            
        except Exception as e:
            logger.error(f"[STAR_EDITOR] Failed to polish resume: {e}")
            # Return safe fallback
            return self._create_fallback_output(experiences, seniority)
    
    def _create_user_prompt(self, experiences: List[Dict], seniority: str, 
                           domain_vocab: List[str], target_company: str = "") -> str:
        """
        Create user prompt for the editor.
        
        Args:
            experiences: Rewritten experiences from Drafter
            seniority: Seniority level applied
            domain_vocab: Domain vocabulary from Researcher
            target_company: Target company name to guard against hallucination
            
        Returns:
            Formatted user prompt
        """
        prompt_parts = []
        
        # Add seniority context
        prompt_parts.append(f"SENIORITY LEVEL: {seniority.upper()}")
        
        # Add domain vocabulary guidance
        if domain_vocab:
            prompt_parts.append(f"DOMAIN VOCABULARY TO INCORPORATE: {', '.join(domain_vocab[:10])}")
        
        # Add SPECIFIC negative constraint for target company
        if target_company:
            prompt_parts.append(f"\nCRITICAL: The user is applying to '{target_company}'. DO NOT list '{target_company}' as their employer in the headers unless it explicitly appears in the experience list below.")
        
        # Add experiences
        prompt_parts.append("\nEXPERIENCES TO FORMAT:")
        for i, exp in enumerate(experiences[:5], 1):  # Limit to 5 experiences
            company = exp.get("company", "Company")
            role = exp.get("role", "Role")
            bullets = exp.get("bullets", [])
            
            prompt_parts.append(f"\n{i}. {role} at {company}:")
            for bullet in bullets[:3]:  # Limit to 3 bullets per experience
                prompt_parts.append(f"   - {bullet}")
        
        # Add formatting instructions
        prompt_parts.append("\nFORMATTING REQUIREMENTS:")
        prompt_parts.append("- Use ## for section headers (e.g., '## Professional Experience')")
        prompt_parts.append("- Use **bold** for job titles and company names")
        prompt_parts.append("- Use *italic* for dates and locations")
        prompt_parts.append("- Use - for bullet points")
        prompt_parts.append("- Ensure every bullet has at least one number/metric")
        prompt_parts.append("- Remove any 'Situation:', 'Task:', 'Action:', 'Result:' labels")
        prompt_parts.append("- Keep language professional and achievement-oriented")
        
        return "\n".join(prompt_parts)
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response and validate structure.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed and validated editor data
        """
        try:
            # Handle empty, whitespace-only, or empty JSON responses
            if not response or not response.strip() or response.strip() == "{}":
                logger.warning("[STAR_EDITOR] Empty response from LLM")
                return self._create_fallback_output([])

            data = json.loads(response)
            
            # Ensure required fields exist
            if not isinstance(data, dict):
                raise ValueError("Response is not a JSON object")
            
            result = {
                "final_markdown": data.get("final_markdown", ""),
                "final_plain_text": data.get("final_plain_text", ""),
                "editorial_notes": data.get("editorial_notes", "Polished for ATS compliance.")
            }
            
            # Ensure we have markdown content
            if not result["final_markdown"]:
                # Try to extract from plain text if markdown is empty
                if result["final_plain_text"]:
                    result["final_markdown"] = self._convert_to_markdown(result["final_plain_text"])
                else:
                    raise ValueError("No resume content generated")
            
            # Ensure markdown doesn't contain analysis metadata
            result["final_markdown"] = self._clean_analysis_metadata(result["final_markdown"])
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"[STAR_EDITOR] Failed to parse JSON response: {e}")
            logger.debug(f"[STAR_EDITOR] Raw response: {response[:500]}")
            
            # Try to extract JSON from text
            json_match = self._extract_json_from_text(response)
            if json_match:
                return self._parse_response(json_match)
            
            # Check if response looks like markdown directly
            if "## " in response or "**" in response or "- " in response:
                return {
                    "final_markdown": response,
                    "final_plain_text": re.sub(r'\*\*|\*|##|- ', '', response),
                    "editorial_notes": "Converted from direct markdown output",
                    "parse_error": str(e)
                }
            
            # Return empty result
            return {
                "final_markdown": "",
                "final_plain_text": "",
                "editorial_notes": "Failed to parse editor response",
                "error": str(e)
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
    
    def _convert_to_markdown(self, plain_text: str) -> str:
        """
        Convert plain text to basic markdown format.
        
        Args:
            plain_text: Plain text resume
            
        Returns:
            Basic markdown formatted resume
        """
        lines = plain_text.strip().split('\n')
        markdown_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for headers
            if line.upper() in ["PROFESSIONAL EXPERIENCE", "WORK EXPERIENCE", "EDUCATION", "SKILLS"]:
                markdown_lines.append(f"\n## {line}")
            # Check for job titles (often followed by "at" or contain "|")
            elif " at " in line or "|" in line:
                parts = line.split(" at ")[0] if " at " in line else line.split("|")[0]
                markdown_lines.append(f"\n**{parts.strip()}**")
            # Check for bullet points
            elif line.startswith("-") or line.startswith("•") or line.startswith("*"):
                markdown_lines.append(f"- {line[1:].strip()}")
            else:
                markdown_lines.append(line)
        
        return "\n".join(markdown_lines)
    
    def _clean_analysis_metadata(self, markdown: str) -> str:
        """
        Remove any analysis metadata that might have leaked through.
        
        Args:
            markdown: Resume markdown text
            
        Returns:
            Cleaned markdown without analysis metadata
        """
        # Patterns that indicate analysis metadata, not resume content
        analysis_patterns = [
            r'gap_analysis.*',
            r'seniority_analysis.*',
            r'suggested_experiences.*',
            r'critical_gaps.*',
            r'bridging_strategies.*',
            r'overall_summary.*',
            r'This analysis shows.*',
            r'The candidate needs.*',
            r'Recommendations:.*'
        ]
        
        lines = markdown.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines that look like analysis metadata
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in analysis_patterns):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _apply_hallucination_guard(self, result: Dict[str, Any], 
                                  original_experiences: List[Dict],
                                  target_company: str = "") -> Dict[str, Any]:
        """
        Apply hallucination guard to detect and fix placeholder content AND target company bleed.
        
        Args:
            result: Editor result with final_markdown
            original_experiences: Original experiences for validation
            target_company: Target company name to guard against hallucination
            
        Returns:
            Result with hallucination fixes applied
        """
        markdown = result.get("final_markdown", "")
        
        # 1. Build list of legitimate companies from input
        original_companies = []
        for exp in original_experiences:
            # Handle various input formats
            comp = exp.get("company")
            if not comp and "title_line" in exp:
                comp = exp.get("title_line", "").split("|")[-1].strip()
            
            if comp and comp not in original_companies:
                original_companies.append(comp)
        
        # 2. TARGET COMPANY GUARD (The "Armada" Fix)
        if target_company:
            # Check if Target Company appears in a Header format: "| **Armada**" or "| Armada"
            # But ONLY if the user didn't actually work there (not in original_companies)
            is_legit_employee = any(target_company.lower() in c.lower() for c in original_companies)
            
            if not is_legit_employee:
                # Regex to find the target company in a header position
                # Looks for: | **Armada** or | Armada at end of line
                pattern = re.compile(rf'\|\s*\**{re.escape(target_company)}\**', re.IGNORECASE)
                
                if pattern.search(markdown):
                    logger.warning(f"[STAR_EDITOR] Target company '{target_company}' hallucinated as employer. Fixing...")
                    
                    # Replace strictly prohibited company with actual employer or safe fallback
                    replacement = f"| **{original_companies[0] if original_companies else 'Previous Employer'}**"
                    markdown = pattern.sub(replacement, markdown)
                    
                    result["editorial_notes"] = f"{result.get('editorial_notes', '')} Fixed target company hallucination ({target_company})."
        
        # 3. Standard Placeholder Guard (Existing logic)
        if contains_hallucination(markdown):
            logger.warning("[STAR_EDITOR] Hallucination detected in final output")
            
            # Replace placeholder companies with first original company or generic
            replacement = original_companies[0] if original_companies else "Previous Company"
            
            for phrase in contains_hallucination.__globals__["FORBIDDEN_PHRASES"]:
                # Case-insensitive replacement
                markdown = re.sub(rf'\b{re.escape(phrase)}\b', replacement, markdown, flags=re.IGNORECASE)
            
            result["editorial_notes"] = f"{result.get('editorial_notes', '')} Hallucination guard applied."
        
        # Update result
        result["final_markdown"] = markdown
        result["final_plain_text"] = re.sub(r'\*\*|\*|##|- ', '', markdown)
        
        return result
    
    def _extract_domain_terms(self, markdown: str, domain_vocab: List[str]) -> List[str]:
        """
        Extract which domain terms were actually used in the resume.
        
        Args:
            markdown: Resume markdown text
            domain_vocab: Available domain vocabulary
            
        Returns:
            List of domain terms found in the resume
        """
        used_terms = []
        markdown_lower = markdown.lower()
        
        for term in domain_vocab:
            term_lower = term.lower()
            if term_lower in markdown_lower:
                used_terms.append(term)
        
        return used_terms
    
    def _check_quantification(self, markdown: str) -> Dict[str, Any]:
        """
        Check quantification metrics in the resume.
        
        Args:
            markdown: Resume markdown text
            
        Returns:
            Quantification analysis
        """
        lines = markdown.split('\n')
        bullet_lines = [line for line in lines if line.strip().startswith('-')]
        
        total_bullets = len(bullet_lines)
        quantified_bullets = 0
        
        quantification_patterns = [
            r'\d+%',           # Percentage
            r'\$\d+',          # Dollar amount
            r'\d+x\b',         # Multiplier (2x, 3x)
            r'\d+\.?\d*\s*(?:ms|s|min|hour|day|month|year)s?',  # Time
            r'\d+[kKmM]?\s*(?:users|requests|queries|transactions)',  # Scale
            r'reduced by \d+', # Reduction
            r'increased by \d+', # Increase
            r'\d+\.?\d*\s*(?:TB|GB|MB|KB)', # Data size
        ]
        
        for bullet in bullet_lines:
            bullet_text = bullet.lower()
            if any(re.search(pattern, bullet_text) for pattern in quantification_patterns):
                quantified_bullets += 1
        
        score = quantified_bullets / max(total_bullets, 1)
        
        return {
            "total_bullets": total_bullets,
            "quantified_bullets": quantified_bullets,
            "quantification_score": score,
            "needs_improvement": score < 0.8 and total_bullets > 0
        }
    
    def _create_fallback_output(self, experiences: List[Dict], seniority: str) -> Dict[str, Any]:
        """
        Create a fallback output when LLM fails.
        
        Args:
            experiences: Rewritten experiences
            seniority: Seniority level
            
        Returns:
            Basic fallback structure
        """
        markdown_lines = ["## Professional Experience"]
        
        for exp in experiences[:3]:  # Limit to 3 experiences
            company = exp.get("company", "Company")
            role = exp.get("role", "Role")
            bullets = exp.get("bullets", [])
            
            markdown_lines.append(f"\n**{role}** at *{company}*")
            
            for bullet in bullets[:2]:  # Limit to 2 bullets per experience
                markdown_lines.append(f"- {bullet}")
        
        markdown = "\n".join(markdown_lines)
        
        return {
            "final_markdown": markdown,
            "final_plain_text": re.sub(r'\*\*|\*|##', '', markdown),
            "editorial_notes": "Fallback resume generated due to processing error.",
            "seniority_level": seniority,
            "domain_terms_used": [],
            "quantification_check": {"total_bullets": 0, "quantified_bullets": 0, "quantification_score": 0, "needs_improvement": True},
            "fallback": True
        }
    
    def get_editor_summary(self, editor_data: Dict[str, Any]) -> str:
        """
        Create a human-readable summary of editing results.
        
        Args:
            editor_data: Editor data from polish()
            
        Returns:
            Formatted summary string
        """
        markdown = editor_data.get("final_markdown", "")
        word_count = len(markdown.split())
        notes = editor_data.get("editorial_notes", "")[:100]
        
        summary = f"Produced {word_count}-word ATS-ready resume"
        if notes:
            summary += f". {notes}"
        
        return summary