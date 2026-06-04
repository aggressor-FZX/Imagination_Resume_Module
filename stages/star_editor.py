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

STAR_EDITOR_SYSTEM_PROMPT = """You are a professional resume editor. You MUST output ALL 5 sections below. Do NOT skip any section.

OUTPUT ALL 5 SECTIONS IN THIS EXACT ORDER:

## Professional Experience
For each job:
**Company Name** -- Location
*Job Title | dates*
- bullet point 1
- bullet point 2

## Projects  
For each project:
**Project Name** -- Tech Stack (duration)
- bullet point

## Education
**Degree, Institution** -- Date
- GPA: X.XX
- Relevant Courses: ...

## Certifications
- Certification Name -- Issuer (Year)

## Technical Skills
**Category:** skill1, skill2, skill3

RULES:
- Every section above MUST appear in the output.
- Company names go on bold lines, NOT merged with job titles.
- Each job has its own sub-section with bullets.
- Skills are grouped by category (2-4 categories).
- Never invent companies, dates, or metrics not in the input.
- Output ONLY resume content, no meta-commentary.

OUTPUT JSON:
{
  "final_markdown": "## Professional Experience\n**NOAA** -- Seattle, WA\n*Operations Manager | 2019 -- 2023*\n- bullet\n\n## Projects\n**Name** -- Tech (duration)\n- bullet\n\n## Education\n**BS, WSU** -- 2026\n\n## Certifications\n- Cert -- Issuer (year)\n\n## Technical Skills\n**Languages:** Python, SQL",
  "final_plain_text": "plain text summary",
  "editorial_notes": "all sections present"
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
                    original_resume_text: str = "",
                    education: List[Dict] = None,
                    projects: List[Dict] = None,
                    certifications: List[Dict] = None,
                    skills: List[str] = None,
                    location: str = "",
                    temperature_override: Optional[float] = None) -> Dict[str, Any]:
        """
        Polishes STAR draft into final resume format with TARGET COMPANY GUARD.
        Now includes education, projects, certifications, and skills sections.
        
        Args:
            draft_data: Output from Drafter stage
            research_data: Output from Researcher stage
            original_resume_text: Original user resume text for metric validation
            education: Education entries from resume
            projects: Project entries from resume
            certifications: Certification entries from resume
            skills: Skill names to include in Skills section
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
        
        # Create prompt with ALL sections including education, projects, certifications, skills
        user_prompt = self._create_user_prompt(
            experiences, seniority, domain_vocab, target_company,
            education=education, projects=projects, certifications=certifications,
            skills=skills, location=location
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
                timeout=self.timeout,
                max_tokens=6000,
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
            
            # Validate metrics - remove any that weren't in the original input
            result = self._validate_metrics_against_input(result, original_resume_text=original_resume_text)
            
            # POST-PROCESSING: Ensure all required sections are present
            result = self._ensure_all_sections(
                result, education=education, certifications=certifications,
                skills=skills, projects=projects,
                original_resume_text=original_resume_text
            )
            
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
                           certifications: List[Dict] = None, skills: List[str] = None,
                           location: str = "") -> str:
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
            skills: Skill names for Skills section
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
            prompt_parts.append("\nEDUCATION:")
            for edu in education[:5]:
                degree = edu.get("degree", "")
                institution = edu.get("institution", "")
                dates = edu.get("dates", "")
                gpa = edu.get("gpa", "")
                line = f"  {degree} -- {institution} ({dates})" if dates else f"  {degree} -- {institution}"
                if gpa:
                    line += f" GPA: {gpa}"
                prompt_parts.append(line)
        
        # Add certifications section
        if certifications:
            prompt_parts.append("\nCERTIFICATIONS:")
            for cert in certifications[:5]:
                name = cert.get("name", "")
                issuer = cert.get("issuer", "")
                year = cert.get("year", "")
                line = f"  {name} -- {issuer} ({year})" if year else f"  {name} -- {issuer}"
                prompt_parts.append(line)
        
        # Add projects section
        if projects:
            prompt_parts.append("\nPROJECTS:")
            for proj in projects[:5]:
                name = proj.get("name", "")
                description = proj.get("description", "")
                technologies = proj.get("technologies", [])
                duration = proj.get("duration", "")
                tech = ", ".join(technologies[:5]) if technologies else ""
                dur = f" ({duration})" if duration and not self._is_placeholder_duration(duration) else ""
                prompt_parts.append(f"  {name} -- {tech}{dur}")
                if description:
                    prompt_parts.append(f"    {description[:120]}")
        
        # Add skills
        if skills:
            prompt_parts.append(f"\nSKILLS: {', '.join(skills[:25])}")
        else:
            prompt_parts.append("\nSKILLS: (extract from experience bullets)")
        
        # Add experiences with dates and location
        prompt_parts.append("\nEXPERIENCES:")
        for i, exp in enumerate(experiences[:10], 1):
            company = exp.get("company", "Company")
            role = exp.get("role") or exp.get("title", "Role")
            
            duration = exp.get("duration") or exp.get("dates", "")
            if not duration:
                title_line = exp.get("title_line", "")
                match = re.search(r'\(([^)]+)\)', title_line)
                if match:
                    duration = match.group(1)
            
            if duration and not self._is_placeholder_duration(duration):
                years = re.findall(r'\b(19|20)\d{2}\b', duration)
                if len(years) >= 2:
                    duration = f"{years[0]} - {years[-1]}"
                elif len(years) == 1:
                    duration = years[0]
            else:
                duration = ""
            
            location_exp = exp.get("location", "")
            if not location_exp:
                title_line = exp.get("title_line", "")
                snippet = exp.get("snippet", "")
                loc_match = re.search(r'([A-Za-z\s]+,\s*[A-Z]{2})(?:\s|$|\)|-)', title_line + " " + snippet)
                if loc_match:
                    location_exp = loc_match.group(1).strip()
            
            bullets = exp.get("bullets", [])
            if not bullets and exp.get("snippet"):
                bullets = [exp.get("snippet")]
            
            if domain_vocab and bullets:
                def score_bullet(b):
                    return sum(1 for v in domain_vocab if v.lower() in b.lower())
                bullets = sorted(bullets, key=score_bullet, reverse=True)
            
            loc = f" -- {location_exp}" if location_exp else ""
            dur = f" | {duration}" if duration else ""
            prompt_parts.append(f"  {company}{loc} | {role}{dur}")
            for bullet in bullets[:6]:
                prompt_parts.append(f"    - {bullet}")
        
        # Minimal formatting instructions
        prompt_parts.append("\nFORMAT: Company on bold line, title+dates on italic line, bullets start with -.")
        
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
        
        v2: Much broader section header detection — the LLM frequently returns
        bare section names without ``##`` prefixes. We detect ALL standard
        resume section names and convert them to proper markdown headers.
        
        Args:
            plain_text: Plain text resume
            
        Returns:
            Basic markdown formatted resume
        """
        # All recognized section header names (case-insensitive)
        _KNOWN_SECTIONS: dict[str, str] = {
            "professional experience":  "## Professional Experience",
            "work experience":          "## Professional Experience",
            "experience":               "## Professional Experience",
            "relevant experience":      "## Professional Experience",
            "employment history":       "## Professional Experience",
            "career history":           "## Professional Experience",
            "education":                "## Education",
            "academic background":      "## Education",
            "academic history":         "## Education",
            "technical skills":         "## Technical Skills",
            "skills":                   "## Technical Skills",
            "skills & technologies":    "## Technical Skills",
            "skills and technologies":  "## Technical Skills",
            "core competencies":        "## Technical Skills",
            "tools & infrastructure":   "## Technical Skills",
            "certifications":           "## Certifications",
            "certifications & publications": "## Certifications",
            "certifications and licenses":    "## Certifications",
            "projects":                 "## Projects",
            "personal projects":        "## Projects",
            "key projects":             "## Projects",
            "capstone projects":        "## Projects",
        }

        lines = plain_text.strip().split('\n')
        markdown_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers (exact or with trailing colon)
            lower_no_colon = line.lower().rstrip(':').strip()
            if lower_no_colon in _KNOWN_SECTIONS:
                markdown_lines.append(f"\n{_KNOWN_SECTIONS[lower_no_colon]}")
                continue
            
            # Check for job titles (contain "|" or " at " — pipe is standard resume format)
            if " at " in line or " | " in line:
                parts = line.split(" at ")[0] if " at " in line else line.split("|")[0]
                markdown_lines.append(f"\n**{parts.strip()}**")
                continue
            
            # Check for bullet points (already prefixed)
            if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                markdown_lines.append(f"- {line.lstrip('-•* ')}")
                continue
            
            # Lines inside experience/skills sections that contain commas
            # and are >20 chars likely represent a skill category line
            if len(line) > 20 and ',' in line:
                # Check if this looks like a skill category line
                # e.g. "Python, SQL, Docker, AWS — Data Engineering"
                markdown_lines.append(f"- {line}")
                continue
            
            # Keep all other lines as-is (company headers, date lines, etc.)
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
            (r'\b20XX\b', ''),  # Placeholder year tokens
            (r'\bXX/XX\b', ''),  # Placeholder date fragments
            (r'\bLocation\b', ''),  # Placeholder location tokens
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

    def _ensure_all_sections(self, result: Dict[str, Any], 
                            education: List[Dict] = None,
                            certifications: List[Dict] = None,
                            skills: List[str] = None,
                            projects: List[Dict] = None,
                            original_resume_text: str = "") -> Dict[str, Any]:
        """Post-process the LLM output to ensure all required sections exist and have content.
        
        If the LLM dropped a section or left it empty, this appends a version from input data.
        Falls back to raw text extraction if structured data is empty.
        """
        md = result.get("final_markdown", "")
        if not md:
            return result

        # Check which sections exist AND have actual content (not just empty headers)
        sections_with_content = set()
        for section_name in ["Professional Experience", "Experience", "Projects", 
                            "Education", "Certifications", "Technical Skills", "Skills"]:
            match = re.search(rf'##\s*{re.escape(section_name)}\s*\n(.*?)(?=##\s|\Z)', md, re.IGNORECASE | re.DOTALL)
            if match:
                body = match.group(1).strip()
                if body and len(body) > 5:
                    sections_with_content.add(section_name.lower())
                elif match:
                    # Header exists but empty — remove the empty header
                    md = md[:match.start()] + md[match.end():]

        added = []

        # If structured data is empty, extract from raw resume text
        if not education and original_resume_text:
            education = self._extract_education_from_text(original_resume_text)
        if not certifications and original_resume_text:
            certifications = self._extract_certs_from_text(original_resume_text)
        if not skills and original_resume_text:
            skills = self._extract_skills_from_text(original_resume_text)

        # Ensure Education section with content
        if "education" not in sections_with_content and education:
            edu_block = "\n\n## Education\n"
            for edu in education[:5]:
                degree = edu.get("degree", "")
                inst = edu.get("institution", "")
                dates = edu.get("dates", "")
                gpa = edu.get("gpa", "")
                # Clean pipe-delimited data
                if "|" in degree:
                    parts = [p.strip() for p in degree.split("|")]
                    if len(parts) >= 2:
                        degree = parts[0]
                        inst = inst or parts[1]
                        if len(parts) >= 3:
                            dates = dates or parts[2]
                if not dates and inst and "|" in inst:
                    parts = [p.strip() for p in inst.split("|")]
                    if len(parts) >= 2:
                        inst = parts[0]
                        dates = dates or parts[1]
                line = f"**{degree}, {inst}**"
                if dates:
                    line += f" -- {dates}"
                edu_block += line + "\n"
                if gpa:
                    edu_block += f"- GPA: {gpa}\n"
            md += edu_block
            added.append("Education")

        # Ensure Certifications section with content
        if "certifications" not in sections_with_content and certifications:
            cert_block = "\n\n## Certifications\n"
            for cert in certifications[:5]:
                name = cert.get("name", "")
                issuer = cert.get("issuer", "")
                year = cert.get("year", "")
                line = f"- {name} -- {issuer}" if issuer else f"- {name}"
                if year:
                    line += f" ({year})"
                cert_block += line + "\n"
            md += cert_block
            added.append("Certifications")

        # Ensure Technical Skills section with content
        if "technical skills" not in sections_with_content and "skills" not in sections_with_content and skills:
            skill_block = "\n\n## Technical Skills\n"
            ml_skills = [s for s in skills if any(w in s.lower() for w in ['pytorch', 'tensorflow', 'keras', 'ml', 'deep learning', 'nlp', 'llm', 'langchain', 'scikit'])]
            data_skills = [s for s in skills if any(w in s.lower() for w in ['sql', 'pandas', 'numpy', 'spark', 'etl', 'tableau', 'power bi', 'data'])]
            other_skills = [s for s in skills if s not in ml_skills and s not in data_skills]
            if ml_skills:
                skill_block += f"**Machine Learning & AI:** {', '.join(ml_skills[:10])}\n"
            if data_skills:
                skill_block += f"**Data & Analytics:** {', '.join(data_skills[:10])}\n"
            if other_skills:
                skill_block += f"**Tools & Platforms:** {', '.join(other_skills[:10])}\n"
            md += skill_block
            added.append("Technical Skills")

        if added:
            logger.warning(f"[STAR_EDITOR] LLM dropped/empty sections, appended from input: {added}")
            result["final_markdown"] = md

        return result

    def _extract_education_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract education entries directly from raw resume text as fallback."""
        edu_entries = []
        lines = text.splitlines()
        in_edu = False
        for line in lines:
            stripped = line.strip().lower()
            if stripped in ("education:", "education", "academic background:", "academic background"):
                in_edu = True
                continue
            if in_edu:
                if stripped.startswith(("experience:", "experience", "certification", "project", "skills:", "technical")):
                    break
                if not line.strip() or line.strip().startswith("-"):
                    continue
                # Try to parse "Degree | Institution | Year" or "Degree - Institution (Year)"
                raw = line.strip()
                parts = [p.strip() for p in re.split(r'\s*\|\s*|\s*-\s+', raw)]
                degree = parts[0] if parts else ""
                inst = parts[1] if len(parts) > 1 else ""
                dates = ""
                year_match = re.search(r'(19|20)\d{2}', raw)
                if year_match:
                    dates = year_match.group(0)
                gpa_match = re.search(r'GPA:\s*([\d.]+)', raw, re.IGNORECASE)
                gpa = gpa_match.group(1) if gpa_match else ""
                if degree:
                    edu_entries.append({"degree": degree, "institution": inst, "dates": dates, "gpa": gpa})
        return edu_entries

    def _extract_certs_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract certification entries directly from raw resume text as fallback."""
        cert_entries = []
        lines = text.splitlines()
        in_certs = False
        for line in lines:
            stripped = line.strip().lower()
            if stripped in ("certifications:", "certifications", "certification:", "certification", "licenses:", "licenses & certifications:"):
                in_certs = True
                continue
            if in_certs:
                if stripped.startswith(("experience:", "project", "skills:", "technical", "education")):
                    break
                raw = line.strip()
                if not raw or raw.startswith("-"):
                    continue
                year_match = re.search(r'(19|20)\d{2}', raw)
                year = year_match.group(0) if year_match else ""
                cert_entries.append({"name": raw, "issuer": "", "year": year})
        return cert_entries

    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skill names directly from raw resume text as fallback."""
        lines = text.splitlines()
        for line in lines:
            stripped = line.strip().lower()
            if stripped.startswith(("skills:", "skills", "technical skills:", "skills & technologies")):
                # Extract skills from the line or following lines
                skills_text = line.split(":", 1)[-1].strip() if ":" in line else ""
                if not skills_text:
                    continue
                # Split by common delimiters
                skills = [s.strip() for s in re.split(r'[,;]', skills_text) if s.strip()]
                return skills[:25]
        return []

        if added:
            logger.warning(f"[STAR_EDITOR] LLM dropped/empty sections, appended from input: {added}")
            result["final_markdown"] = md

        return result

    def _normalize_section_boundaries(self, markdown: str) -> str:
        """Ensure resume section headings are clearly separated by blank lines.
        
        v3: Adds a newline-preprocessing step before all section detection.
        Many LLM failures produce the ENTIRE resume as a single line with no
        newlines at all, making line-by-line detection impossible.  We inject
        ``\n`` before each known section name so the line-based steps work
        reliably even on the worst-case single-line input.
        """
        if not markdown:
            return markdown

        result = markdown
        
        # ------------------------------------------------------------------
        # STEP 0-NEW: Inject newlines before known section names that appear
        # mid-text (single-line resume problem).  This ensures line-by-line
        # detection in later steps always finds every section.
        #
        # We process LONGER (multi-word) phrases FIRST, then only match
        # single-word names at positions NOT previously claimed by a phrase.
        # ------------------------------------------------------------------
        _SECTION_NAMES_LONG = [
            "Technical Skills",
            "Professional Experience",
            "Work Experience",
            "Skills & Technologies",
            "Skills and Technologies",
        ]
        _SECTION_NAMES_SHORT = [
            "Skills",
            "Experience",
            "Projects",
            "Education",
            "Certifications",
        ]

        # --- Phase 1: Insert newlines before long (multi-word) phrases ---
        for name in _SECTION_NAMES_LONG:
            pattern = rf'(?<!\n)\b({re.escape(name)})\s+([A-Z0-9*])'
            def _add_newline_long(m: "re.Match[str]") -> str:
                return "\n" + m.group(0)
            result = re.sub(pattern, _add_newline_long, result)

        # --- Phase 2: Insert newlines before single-word names,
        #     SKIPPING those already part of a multi-word phrase ---
        for name in _SECTION_NAMES_SHORT:
            pattern = rf'(?<!\n)\b({re.escape(name)})\s+([A-Z0-9*])'

            def _add_newline_short(m: "re.Match[str]", _name: str = name) -> str:
                """Only insert newline if this word is NOT part of a longer phrase."""
                substr = result[max(0, m.start() - 40):m.start() + len(_name) + 5]
                is_part_of_longer = any(
                    phrase.lower() in substr.lower()
                    for phrase in _SECTION_NAMES_LONG
                    if _name.lower() in phrase.lower()
                )
                if is_part_of_longer:
                    return m.group(0)  # no newline — already part of a longer phrase
                return "\n" + m.group(0)

            result = re.sub(pattern, _add_newline_short, result)
        
        # ------------------------------------------------------------------
        # STEP 1: Identify ALL required sections and add ## headers if missing
        # ------------------------------------------------------------------
        # Ordered from most specific to least specific to avoid partial matches
        _SECTION_ALIASES: list[tuple[str, list[str]]] = [
            ("Technical Skills", [
                "technical skills", "technicalskills", "tech skills",
                "skills & technologies", "skills and technologies",
                "core competencies", "tools & infrastructure",
                "skills",  # last resort — only catch standalone "## Skills" 
            ]),
            ("Professional Experience", [
                "professional experience", "work experience", "experience",
                "employment history", "career history", "relevant experience",
            ]),
            ("Projects", [
                "projects", "personal projects", "key projects",
                "portfolio", "selected projects", "capstone projects",
            ]),
            ("Certifications", [
                "certifications", "certifications & publications",
                "certificates", "licenses", "certifications and licenses",
            ]),
            ("Education", [
                "education", "academic background", "academic history",
            ]),
        ]

        # Build a set of section names that already have ## headers
        existing_headers = re.findall(r'^##\s*(.+)', result, re.MULTILINE)
        existing_lower = {h.strip().lower() for h in existing_headers}

        # ------------------------------------------------------------------
        # STEP 0: If no section headers at all but bullets exist, inject a
        #         "## Professional Experience" header before the first bullet.
        #         This handles the common LLM failure mode where the model
        #         returns a wall of text with only inline bullets.
        # ------------------------------------------------------------------
        if not existing_headers:
            first_bullet = re.search(r'(?:^|\n)\s*[-•*▪●►◦]\s+\S', result, re.MULTILINE)
            if first_bullet:
                insert_at = first_bullet.start()
                # Find the preceding paragraph break — insert before it
                if insert_at > 0 and result[insert_at - 1] != '\n':
                    insert_at = result.rfind('\n', 0, insert_at) + 1
                result = result[:insert_at] + "\n\n## Professional Experience\n\n" + result[insert_at:]

        for canonical, aliases in _SECTION_ALIASES:
            # Skip if this canonical section already has a ## header
            already_has = any(a in existing_lower for a in [canonical.lower()] + [a.lower() for a in aliases])
            if already_has:
                continue

            found = False
            for alias in aliases:
                # --- Pattern A: section name on its own line (standalone header) ---
                pattern_ownline = rf'(?<!\#)(?:^|\n)\s*{re.escape(alias)}\s*:?\s*(?:\n|$)'
                if re.search(pattern_ownline, result, re.IGNORECASE):
                    result = re.sub(
                        pattern_ownline,
                        f"\n\n## {canonical}\n",
                        result, count=1, flags=re.IGNORECASE,
                    )
                    found = True
                    break

                # --- Pattern B: section name at start of a line, followed by content
                #     on the same line.  We split it into:
                #         [## Canonical]\n  remaining content
                #     Uses a line-by-line scan for robustness against \n capture drift.
                alias_re = re.compile(rf'^\s*{re.escape(alias)}\s+', re.IGNORECASE)
                new_lines: list[str] = []
                for line in result.split("\n"):
                    m = alias_re.match(line)
                    # Only convert if the section name is at the START of the line
                    # and it's the FIRST unconverted occurrence
                    if m and not found:
                        rest = line[m.end():]
                        new_lines.append(f"\n## {canonical}")
                        new_lines.append(rest)
                        found = True
                    else:
                        new_lines.append(line)
                if found:
                    result = "\n".join(new_lines)
                    break

                # --- Pattern C: colon variant at line start  "Education: B.S. ..." ---
                colon_re = re.compile(rf'^\s*{re.escape(alias)}\s*:\s', re.IGNORECASE)
                new_lines = []
                for line in result.split("\n"):
                    m = colon_re.match(line)
                    if m and not found:
                        rest = line[m.end():]
                        new_lines.append(f"\n## {canonical}")
                        new_lines.append(rest)
                        found = True
                    else:
                        new_lines.append(line)
                if found:
                    result = "\n".join(new_lines)
                    break

            if found:
                continue

        # ------------------------------------------------------------------
        # STEP 2: Add ``- `` bullets under experience sections if missing
        # Many LLM responses concatenate bullet items into paragraphs.
        # We detect this pattern: experience header OR company line followed
        # by a long line (>60 chars) without ``- `` or ``• `` prefix.
        # ------------------------------------------------------------------
        lines = result.split('\n')
        out_lines: list[str] = []
        in_experience_section = False

        for line in lines:
            stripped = line.strip()
            lower = stripped.lower()

            # Track whether we are in an experience/education section
            if re.match(r'^##\s', stripped):
                in_experience_section = any(
                    kw in lower
                    for kw in ['experience', 'project', 'education', 'certification']
                )
                out_lines.append(line)
                continue

            # Inside experience section: detect unmarked lines that look like bullets
            if in_experience_section and stripped:
                is_already_bullet = bool(re.match(r'^[-•*▪●►◦]\s', stripped))
                is_bold_header = bool(re.match(r'^\*\*.*\*\*$', stripped))
                is_italic_header = bool(re.match(r'^\*.*\*$', stripped))
                is_section_header = bool(re.match(r'^##\s', stripped))

                if (
                    not is_already_bullet
                    and not is_bold_header
                    and not is_italic_header
                    and not is_section_header
                    and len(stripped) > 15
                ):
                    # This line is inside a section but has no bullet — add one
                    out_lines.append(f"- {stripped}")
                    continue

            out_lines.append(line)

        result = '\n'.join(out_lines)

        # ------------------------------------------------------------------
        # STEP 3: Final normalization — fix spacing, remove double ##/bolts
        # ------------------------------------------------------------------
        # If model returned plain labels only (no ## at all), fallback header pass
        if "##" not in result:
            for section in ["Education", "Certifications", "Projects", "Professional Experience"]:
                pattern = rf'(?<!#)\b{re.escape(section)}\b\s*:?'
                result = re.sub(pattern, f"\n\n## {section}\n", result, flags=re.IGNORECASE)

        # Ensure known section headers start on a fresh block.
        result = re.sub(
            r'\s*(##\s*(Education|Certifications|Projects|Professional Experience|Technical Skills)\b)',
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
    
    def _validate_metrics_against_input(self, result: Dict[str, Any], original_resume_text: str = "") -> Dict[str, Any]:
        """
        Remove fabricated metrics that weren't in the original input.
        
        Looks for percentage/dollar/time metrics in output and validates they exist in input.
        """
        markdown = result.get("final_markdown", "")
        original_text_lower = original_resume_text.lower() if original_resume_text else ""
        
        # Find all metrics in the output
        metric_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(%|percent|\$\d+|k|thousand|million|billion|hours|days|weeks|months|years)', re.IGNORECASE)
        
        def is_metric_in_original(match):
            value = match.group(1)
            unit = match.group(2).lower()
            # Check if this exact value appears in original text with unit
            # Original might have "30%." or "30 percent" - handle both
            search_patterns = [
                f"{value}%",           # "30%"
                f"{value} percent",    # "30 percent"
            ]
            for pattern in search_patterns:
                if pattern in original_text_lower:
                    return True
            return False
        
        # Remove metrics not in original
        new_lines = []
        for line in markdown.split('\n'):
            if line.strip().startswith('-'):
                # Check each metric in this line
                new_line = line
                for match in metric_pattern.finditer(line):
                    if not is_metric_in_original(match):
                        metric_str = match.group(0)
                        # Clean removal: remove metric but also " by" if it leads to nothing after
                        new_line = new_line.replace(metric_str, "").replace(" by", "").replace("  ", " ").strip()
                        # Remove trailing prepositions that are now orphaned
                        new_line = re.sub(r'\b(by|,)\s*$', '', new_line).strip()
                        logger.warning(f"[STAR_EDITOR] Removed fabricated metric: {metric_str}")
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        
        result["final_markdown"] = "\n".join(new_lines)
        result["final_plain_text"] = re.sub(r'\*\*|\*|##|- ', '', result["final_markdown"])
        result["hallucination_checked"] = True
        
        return result
    
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
            
            for bullet in bullets[:6]:  # Limit to 6 bullets per experience
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