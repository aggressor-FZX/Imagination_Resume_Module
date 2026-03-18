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

STAR_EDITOR_SYSTEM_PROMPT = """You are a professional resume editor. Convert the provided resume data into a high-end, ATS-friendly Markdown resume with ALL required sections.

### THE FORMULA:
Your output MUST include ALL sections provided in the input:
1. **EDUCATION** (if provided)
2. **CERTIFICATIONS** (if provided)  
3. **PROJECTS** (if provided, especially for students/career changers)
4. **PROFESSIONAL EXPERIENCE** (always required)

### SECTION FORMATS:

**EDUCATION:**
## Education
- **Degree Name** - Institution Name (Year)
- **Degree Name** - Institution Name (Year) - GPA: X.XX

**CERTIFICATIONS:**
## Certifications
- Certification Name - Issuer (Year)
- Certification Name - Issuer (Year)

**PROJECTS:**
## Projects
- **Project Name** - Tech Stack (Duration)
  Description of project impact and what was built
- **Project Name** - Tech Stack (Duration)
  Description of project impact and what was built

**PROFESSIONAL EXPERIENCE:**
## Professional Experience
## **Job Title** | **Company Name**
*Actual Dates | Actual Location*
- Achievement Bullet 1 with metric
- Achievement Bullet 2 with metric

### STRICT RULES:
1. **INCLUDE ALL PROVIDED SECTIONS:** If education, certifications, or projects are provided in the input, they MUST appear in the output. Do not omit any section.
   
2. **NO PLACEHOLDERS - CRITICAL:** You MUST use the actual dates and location provided. NEVER output these forbidden placeholders:
   - WRONG: *Dates | Location*, *Dates Provided*, *N/A*, *Unknown*
   - CORRECT: *2020 - Present | San Francisco, CA*, *04/2025 – Present | Washington State*
   
3. **NO BLOCKS OF TEXT:** Every achievement must be a distinct bullet point starting with a hyphen (-).
4. **DELETE ALL LABELS:** Remove "Situation:", "Task:", "Action:", "Result:", "STAR:", "CAR:" labels.
5. **MANDATORY QUANTIFICATION:** Every bullet point must include a metric (%, $, time, count).
6. **STYLE TRANSFER:** Maintain punchy, technical syntax from input.
7. **NO LEAKAGE:** No meta-commentary or editorial notes. Output ONLY resume content.
8. **ANTI-HALLUCINATION:** NEVER list the Target Company as an employer. Use ONLY company names from input.
9. **SECTION ORDER:** Education → Certifications → Projects → Professional Experience (if all provided)

Output JSON Schema:
{
  "final_markdown": "## Education\\n- **BS Computer Science** - MIT (2020)\\n\\n## Certifications\\n- AWS Certified Solutions Architect - Amazon (2023)\\n\\n## Projects\\n- **AI Chatbot** - Python, OpenAI (6 months)\\n\\n## Professional Experience\\n## **Software Engineer** | **Google**\\n*2020 - Present | Mountain View, CA*\\n- Deployed CI/CD...",
  "final_plain_text": "Education: BS Computer Science - MIT (2020) | Certifications: AWS Certified Solutions Architect | Projects: AI Chatbot | Experience: Software Engineer at Google...",
  "editorial_notes": "Polished for ATS compliance with all required sections."
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

    def _is_placeholder_duration(self, value: str) -> bool:
        """Return True when duration is a known placeholder string."""
        if not value:
            return True
        normalized = value.strip().lower()
        return normalized in {
            "duration not specified",
            "durtion not specified",
            "not specified",
            "n/a",
            "na",
            "unknown",
            "tbd",
        }
        
    async def polish(self, draft_data: Dict[str, Any], 
                    research_data: Dict[str, Any],
                    education: List[Dict] = None,
                    projects: List[Dict] = None,
                    certifications: List[Dict] = None,
                    location: str = "",
                    temperature_override: Optional[float] = None) -> Dict[str, Any]:
        """
        Polishes STAR draft into final resume format with TARGET COMPANY GUARD.
        Now includes education, projects, and certifications sections.
        
        Args:
            draft_data: Output from Drafter stage
            research_data: Output from Researcher stage
            education: Education entries from resume
            projects: Project entries from resume
            certifications: Certification entries from resume
            location: Job location for market intel
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
        
        # Create prompt with ALL sections including education, projects, certifications
        user_prompt = self._create_user_prompt(
            experiences, seniority, domain_vocab, target_company,
            education=education, projects=projects, certifications=certifications,
            location=location
        )
        
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
                           domain_vocab: List[str], target_company: str = "",
                           education: List[Dict] = None, projects: List[Dict] = None,
                           certifications: List[Dict] = None, location: str = "") -> str:
        """
        Create user prompt for the editor with ALL resume sections.
        
        Args:
            experiences: Rewritten experiences from Drafter
            seniority: Seniority level applied
            domain_vocab: Domain vocabulary from Researcher
            target_company: Target company name to guard against hallucination
            education: Education entries
            projects: Project entries
            certifications: Certification entries
            location: Job location
            
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
        
        # Add location if provided
        if location:
            prompt_parts.append(f"\nLOCATION CONTEXT: {location}")
        
        # Add education section
        if education:
            prompt_parts.append("\nEDUCATION TO INCLUDE:")
            for edu in education[:5]:  # Limit to 5 entries
                degree = edu.get("degree", "")
                institution = edu.get("institution", "")
                dates = edu.get("dates", "")
                gpa = edu.get("gpa", "")
                details = f"{degree}"
                if institution:
                    details += f" - {institution}"
                if dates:
                    details += f" ({dates})"
                if gpa:
                    details += f" - GPA: {gpa}"
                prompt_parts.append(f"- {details}")
        
        # Add certifications section
        if certifications:
            prompt_parts.append("\nCERTIFICATIONS TO INCLUDE:")
            for cert in certifications[:5]:  # Limit to 5 entries
                name = cert.get("name", "")
                issuer = cert.get("issuer", "")
                year = cert.get("year", "")
                details = name
                if issuer:
                    details += f" - {issuer}"
                if year:
                    details += f" ({year})"
                prompt_parts.append(f"- {details}")
        
        # Add projects section (for students/career changers)
        if projects:
            prompt_parts.append("\nPROJECTS TO INCLUDE:")
            for proj in projects[:5]:  # Limit to 5 entries
                name = proj.get("name", "")
                description = proj.get("description", "")
                technologies = proj.get("technologies", [])
                duration = proj.get("duration", "")
                details = name
                if technologies:
                    details += f" - {', '.join(technologies[:3])}"
                if duration and not self._is_placeholder_duration(duration):
                    details += f" ({duration})"
                prompt_parts.append(f"- {details}")
                if description:
                    prompt_parts.append(f"  {description[:100]}")
        
        # Add experiences with dates and location
        prompt_parts.append("\nEXPERIENCES TO FORMAT:")
        for i, exp in enumerate(experiences[:10], 1):  # Limit to 10 experiences
            company = exp.get("company", "Company")
            role = exp.get("role") or exp.get("title", "Role")
            
            # Extract duration from multiple possible sources
            duration = exp.get("duration") or exp.get("dates", "")
            if not duration:
                # Try to extract from title_line (Drafter format)
                title_line = exp.get("title_line", "")
                if "(" in title_line and ")" in title_line:
                    match = re.search(r'\(([^)]+)\)', title_line)
                    if match:
                        duration = match.group(1)
            
            # Extract just years from duration (e.g., "2020 - Present" or "Jan 2020 - Dec 2023")
            if duration and not self._is_placeholder_duration(duration):
                # Find all 4-digit years
                years = re.findall(r'\b(19|20)\d{2}\b', duration)
                if len(years) >= 2:
                    duration = f"{years[0]} - {years[-1]}"  # "2020 - 2024"
                elif len(years) == 1:
                    duration = years[0]  # "2020" or "Present"
            else:
                duration = ""
            
            # Extract location from multiple possible sources
            location_exp = exp.get("location", "")
            if not location_exp:
                # Try to extract from title_line or snippet
                title_line = exp.get("title_line", "")
                snippet = exp.get("snippet", "")
                # Look for "City, ST" pattern
                loc_match = re.search(r'([A-Za-z\s]+,\s*[A-Z]{2})(?:\s|$|\)|-)', title_line + " " + snippet)
                if loc_match:
                    location_exp = loc_match.group(1).strip()
            
            bullets = exp.get("bullets", [])
            if not bullets and exp.get("snippet"):
                # Convert snippet to bullets if no bullets provided
                bullets = [exp.get("snippet")]
            
            # Rank bullets by relevance to target job (domain_vocab matches)
            if domain_vocab and bullets:
                def score_bullet(b):
                    b_lower = b.lower()
                    return sum(1 for v in domain_vocab if v.lower() in b_lower)
                # Stable sort: highest score first, preserves original order on ties
                bullets = sorted(bullets, key=score_bullet, reverse=True)
            
            # Build the header line with available data
            header = f"{i}. {role} at {company}"
            if duration or location:
                header += f" | {duration}"
                if location:
                    header += f" | {location}"
            
            prompt_parts.append(f"\n{header}")
            for bullet in bullets[:6]:  # Limit to 6 bullets per experience
                prompt_parts.append(f"   - {bullet}")
        
        # Add formatting instructions
        prompt_parts.append("\nFORMATTING REQUIREMENTS:")
        prompt_parts.append("- Act as a ruthless editor: Keep only the most relevant, high-impact bullets for the target role.")
        prompt_parts.append("- Use years only for dates (e.g., '2020 - 2024' not 'Jan 2020 - Dec 2024').")
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
            
            # CRITICAL: Remove any remaining placeholders
            result["final_markdown"] = self._remove_placeholders(result["final_markdown"])
            # Normalize section spacing so section starts are easy to scan
            result["final_markdown"] = self._normalize_section_boundaries(result["final_markdown"])
            
            return result
            
        except json.JSONDecodeError as e:
            # Try to extract JSON from text
            json_match = self._extract_json_from_text(response)
            if json_match:
                logger.warning(
                    "[STAR_EDITOR] Direct JSON parse failed but markdown JSON extraction succeeded: %s",
                    e,
                )
                return self._parse_response(json_match)

            logger.error(f"[STAR_EDITOR] Failed to parse JSON response: {e}")
            logger.error(f"[STAR_EDITOR] Raw response length: {len(response) if response else 0}")
            logger.error(f"[STAR_EDITOR] Raw response preview: {(response or '')[:500]}")
            
            # Check if response looks like markdown directly
            if response and ("## " in response or "**" in response or "- " in response):
                logger.warning("[STAR_EDITOR] Using direct markdown output as fallback")
                return {
                    "final_markdown": response,
                    "final_plain_text": re.sub(r'\*\*|\*|##|- ', '', response),
                    "editorial_notes": "Converted from direct markdown output",
                    "parse_error": str(e),
                    "fallback_used": True
                }
            
            # Return empty result with error flag
            logger.error("[STAR_EDITOR] No usable content from editor, returning empty result")
            return {
                "final_markdown": "",
                "final_plain_text": "",
                "editorial_notes": "Failed to parse editor response",
                "error": str(e),
                "fallback_used": True
            }
    
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
    
    def _remove_placeholders(self, markdown: str) -> str:
        """
        Remove any placeholder text that the LLM might have output.
        
        This is a safety net to catch any placeholders that slip through
        despite the prompt instructions.
        
        Args:
            markdown: Resume markdown text
            
        Returns:
            Markdown with placeholders removed or replaced
        """
        # List of forbidden placeholder patterns
        placeholder_patterns = [
            (r'\*Dates \| Location\*', ''),  # Classic placeholder
            (r'\*Dates Provided \| Location Provided\*', ''),  # Variation
            (r'\*Date \| Location\*', ''),  # Singular variation
            (r'\*Present - Present\*', ''),  # Invalid date range
            (r'\*N/A \| N/A\*', ''),  # N/A placeholder
            (r'\*Unknown \| Unknown\*', ''),  # Unknown placeholder
            (r'\*Dates \| [A-Za-z\s]+\*', ''),  # Partial placeholder
            (r'\*[A-Za-z\s]+ \| Location\*', ''),  # Partial placeholder
            (r'\s*\((?:dur(?:a)?tion\s+not\s+specified|not\s+specified|n/?a|unknown|tbd)\)', ''),
            (r'\|\s*(?:dur(?:a)?tion\s+not\s+specified|not\s+specified|n/?a|unknown|tbd)\b', ''),
        ]
        
        result = markdown
        for pattern, replacement in placeholder_patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Clean up separators left behind after placeholder removal.
        result = re.sub(r'\|\s*\|', '|', result)
        result = re.sub(r'\*\s*\|\s*', '*', result)
        result = re.sub(r'\s{2,}', ' ', result)
        
        # Log if we removed placeholders
        if result != markdown:
            logger.warning("[STAR_EDITOR] Removed placeholder text from output")
        
        return result

    def _normalize_section_boundaries(self, markdown: str) -> str:
        """Ensure resume section headings are clearly separated by blank lines."""
        if not markdown:
            return markdown

        result = markdown

        # If model returned plain labels, promote them to markdown headers.
        if "##" not in result:
            for section in ["Education", "Certifications", "Projects", "Professional Experience"]:
                pattern = rf'(?<!#)\b{re.escape(section)}\b\s*:?'
                result = re.sub(pattern, f"\n\n## {section}\n", result, flags=re.IGNORECASE)

        # Ensure known section headers start on a fresh block.
        result = re.sub(
            r'\s*(##\s*(Education|Certifications|Projects|Professional Experience)\b)',
            r'\n\n\1',
            result,
            flags=re.IGNORECASE,
        )
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result.strip()
    
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