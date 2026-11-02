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
import os
import re
from typing import Callable, Dict, Any, List, Optional
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

# ============================================================================
# METRIC EXTRACTION
# ============================================================================
# R-2 fix (2026-10-30). The previous implementation used re.findall() with
# patterns that contained leading optional capture groups, then took m[0].
# For "$500K" the first group matched the empty string, so the metric was
# dropped; "13 vessels" and "10+ TB" had no matching pattern at all; and
# "30 hours" produced a bare "30" with the unit discarded.
#
# This version returns a NORMALIZED QUANTITY (value + unit) per SPEC §5.3,
# which is what the truth checks compare against.
import re as _metrics_re

_METRIC_PATTERNS = [
    # percentage: 40%, 12.5 %
    ("percent", _metrics_re.compile(r'(?<![\d.])(\d+(?:\.\d+)?)\s*%')),
    # currency: $500K, $1.2M, $3,000, $250,000
    ("currency", _metrics_re.compile(
        r'\$\s?(\d[\d,]*(?:\.\d+)?)\s*([KkMmBb](?![A-Za-z]))?')),
    # data size: 10+ TB, 500 GB, 2MB
    ("data_size", _metrics_re.compile(
        r'(?<![\d.])(\d+(?:\.\d+)?)\s*\+?\s*(TB|GB|MB|KB|PB)\b', _metrics_re.I)),
    # count with unit: 13 vessels, 200 users, 6 engineers, 3 models
    ("count", _metrics_re.compile(
        r'(?<![\d.])(\d+(?:\.\d+)?)\s*\+?\s*(vessels?|users?|customers?|clients?|'
        r'devices?|assets?|records?|rows?|documents?|models?|servers?|'
        r'engineers?|apprentices?|pipelines?|sites?|locations?|employees?|'
        r'teams?|projects?|systems?|applications?)\b', _metrics_re.I)),
    # time with unit: 30 hours, 12 weeks, 5 days
    ("time", _metrics_re.compile(
        r'(?<![\d.])(\d+(?:\.\d+)?)\s*\+?\s*(hours?|hrs?|days?|weeks?|months?|'
        r'years?|minutes?|mins?)\b', _metrics_re.I)),
    # multiplier: 3x, 10x
    ("multiplier", _metrics_re.compile(r'(?<![\d.])(\d+(?:\.\d+)?)\s*[xX](?![A-Za-z])')),
    # verb-led change without explicit percent: "reduced ... by 40"
    ("change", _metrics_re.compile(
        r'(?:reduced|increased|improved|decreased|cut|saved|grew|boosted|'
        r'lowered|accelerated|shortened|expanded|scaled)\b[^.\n]{0,40}?\bby\s+'
        r'(\d+(?:\.\d+)?)', _metrics_re.I)),
]


def _extract_metrics(text: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Extract normalized metric quantities from free text.

    Returns a list of dicts::

        {"value": "30", "unit": "hours", "kind": "time",
         "display": "30 hours", "source_text": "...context..."}

    ``value`` and ``unit`` are what SPEC.v2 §5.3 compares; ``display`` keeps the
    surface form for rendering. Digits are never returned without their unit,
    which is the defect this function replaces (evidence C5).
    """
    if not text:
        return []

    found: List[Dict[str, Any]] = []
    seen = set()

    for kind, pattern in _METRIC_PATTERNS:
        for match in pattern.finditer(text):
            groups = [g for g in match.groups() if g]
            if not groups:
                continue
            value = str(groups[0]).replace(",", "")
            unit = str(groups[1]) if len(groups) > 1 else _DEFAULT_UNITS[kind]

            # Skip bare years such as 2024 — they are dates, not achievements.
            if kind == "time" and _METRIC_RE_YEAR.match(unit) and len(value) == 4:
                continue

            key = (value, unit.lower())
            if key in seen:
                continue
            seen.add(key)

            display = match.group(0).strip()
            context_start = max(0, match.start() - 40)
            context = text[context_start:match.end() + 40].replace("\n", " ").strip()

            found.append({
                "value": value,
                "unit": unit.lower(),
                "kind": kind,
                "display": display,
                "source_text": context,
            })
            if len(found) >= limit:
                return found

    return found


_DEFAULT_UNITS = {
    "percent": "%",
    "currency": "usd",
    "data_size": "bytes",
    "count": "count",
    "time": "time",
    "multiplier": "x",
    "change": "count",
}

_METRIC_RE_YEAR = _metrics_re.compile(r'^(hours?|days?|weeks?|months?|years?)$', _metrics_re.I)


# ============================================================================
# PROMPT CONSTRUCTION
# ============================================================================

def create_drafter_prompt(experiences: List[Dict], job_ad: str, research_data: Dict,
                         seniority_level: str, allowed_verbs: List[str],
                         golden_bullets: Optional[List[str]] = None,
                         tone_instruction: str = "Maintain a standard, professional corporate tone.",
                         extracted_job_title: Optional[str] = None) -> str:
    """
    Create the drafter prompt with strict anti-hallucination guardrails and aggressive Style Transfer.
    """
    # 1. Extract TRUTH constraints from structured experience data.
    # The LLM Parser writes the job title into the 'title' key; we also check
    # 'role' for backward compat and 'title_line' as a last resort.
    def _clean_company_name(name: str) -> str:
        """Strip trailing date patterns that sometimes appear in company names
        (e.g. "Company Name 04/2025 - Present" → "Company Name")."""
        import re
        return re.sub(r'\s+\d{1,2}/\d{4}\s*[-–]\s*(Present|\d{1,2}/\d{4})\s*$', '', name).strip()

    def _extract_role(exp: Dict) -> str:
        """Extract role from experience entry, checking all known key variants.

        An entry with no recognizable title yields an EMPTY string, never the word
        "Professional". Substituting a placeholder turns a missing fact into a
        fabricated one, and downstream stages then treat it as the candidate's real
        role — which is exactly how delivery run 4 lost both of its job titles.
        """
        return exp.get("title") or exp.get("role") or (exp.get("title_line") or "").split("|")[0].strip()

    original_companies = [
        _clean_company_name(exp.get("company", "")) or "Unknown"
        for exp in experiences if exp.get("company") or _extract_role(exp)
    ]
    original_titles = [_extract_role(exp) for exp in experiences if _extract_role(exp)]
    
    # 2. Format Golden Bullets as "Style Reference Only"
    # Raised 3→6 (2026-06-04): 6 references provide good style signal
    # without OOM. 10 caused instance crash on 1GB Render.
    golden_section = ""
    if golden_bullets:
        selected_bullets = golden_bullets[:6]
        bullets_text = "\n".join([f"• {b}" for b in selected_bullets])
        
        golden_section = f"""
### STYLE PATTERN REFERENCE (SYNTAX ONLY - ALL DATA IS FAKE)
WARNING: The following examples contain COMPLETELY FAKE DATA (company names, metrics, projects).
USE THEM ONLY FOR GRAMMATICAL STRUCTURE. 
YOU MUST NEVER COPY: company names, metrics, technologies, project names, or any factual content.
ALL companies in these examples ARE FICTIONAL. The user did NOT work at any of these companies.

SYNTAX REFERENCE (copy this structure only):
• [Action Verb] + [Technical Task] + [Specific Tool/Method] + [Measurable Outcome if available]

EXAMPLE SYNTAX TRANSFORMATION:
- User input: "I improved database queries"
- Good output: "Improved database query performance using indexing and query optimization"
- GOOD output: "Optimized SQL queries, reducing load time by 40%" (if user said "40%")
- BAD output: "Optimized SQL queries, reducing load time by 40%" (if user did NOT say 40% — do NOT fabricate metrics)

REFERENCE PATTERNS (FICTIONAL - DO NOT COPY DATA):
{bullets_text}

### STYLE TRANSFER RULES (MANDATORY):
1. **Strong Verbs:** Use "Deployed," "Engineered," "Refactored," "Optimized," "Led," "Built" — NOT "Strategized," "Pioneered," "Worked on"
2. **NORMALIZE TERMINOLOGY ONLY:** You MAY restate an activity the candidate's own input
   already describes using a clearer term for the SAME activity (e.g. "created embeddings in
   Pinecone" -> "vector search with Pinecone"). You may NOT add a tool, technology, platform
   or activity that the candidate's input does not contain. "database work" does NOT license
   "Python and SQL" unless the candidate's input or verified skills say so — that is an
   invention, not context.
3. **Only Use Real Metrics:** Only include metrics (%, $, time) that appear EXPLICITLY in the user's input. Do NOT invent "increased revenue by X%" unless the user said it.
4. **One Thought Per Bullet:** Do not combine unrelated tasks. Keep bullets punchy (15-25 words max).
5. **Front-Load the Impact:** Start with the Result or the strong Verb. (e.g., "Cut costs by 40%..." rather than "Responsible for cutting costs...")
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

### STYLE TRANSFER RULES (MANDATORY):
1. **Strong Verbs:** Use "Deployed," "Engineered," "Refactored," "Optimized," "Led," "Built" — NOT "Strategized," "Pioneered," "Worked on"
2. **NORMALIZE TERMINOLOGY ONLY:** You MAY restate an activity the candidate's own input
   already describes using a clearer term for the SAME activity (e.g. "created embeddings in
   Pinecone" -> "vector search with Pinecone"). You may NOT add a tool, technology, platform
   or activity that the candidate's input does not contain. "database work" does NOT license
   "Python and SQL" unless the candidate's input or verified skills say so — that is an
   invention, not context.
3. **Only Use Real Metrics:** Only include metrics (%, $, time) that appear EXPLICITLY in the user's input. Do NOT invent "increased revenue by X%" unless the user said it.
4. **One Thought Per Bullet:** Do not combine unrelated tasks. Keep bullets punchy (15-25 words max).
5. **Front-Load the Impact:** Start with the Result or the strong Verb. (e.g., "Cut costs by 40%..." rather than "Responsible for cutting costs...")

### *** TRUTH CONSTRAINTS (VIOLATION = FAILURE) ***
1. **COMPANY NAMES:** You must ONLY use the companies provided in the User Input.
   - ALLOWED COMPANIES (MUST USE EXACTLY): {json.dumps(original_companies)}
   - PROHIBITED: Do NOT invent company names like "Tech Innovations Inc", "Data Solutions LLC", "ABC Corp", "Acme Corp", or any company NOT in the input.
   - CRITICAL: If the only allowed company is "[UNKNOWN COMPANY - USE 'Current Employer' ONLY]", you MUST use "Current Employer" as the company name. Do NOT invent any other company.

2. **JOB TITLES:** You must preserve the user's actual role hierarchy.
   - ALLOWED: {json.dumps(original_titles) if original_titles else '[] (no title stated — leave the title out)'}
   - You may slightly polish titles (e.g., "Programmer" -> "Software Engineer"), but DO NOT promote a "Junior" to "VP".

3. **TECHNOLOGY — EVIDENCE ONLY:**
   - Use technologies named in the user's input. Terminology normalization is
     allowed: "database work" → "SQL" is acceptable ONLY if SQL appears in the
     candidate's input or verified skills.
   - Do NOT introduce a new ACTIVITY the source does not describe. Listing PyTorch
     does not establish "analyzed deep learning model internals"; that is a new
     claim, not a restatement.
   - Do NOT add technologies unrelated to the user's work.

4. **JD KEYWORD WEAVING (ATS OPTIMIZATION) — NO NEW CLAIMS:**
   - The Job Description contains vocabulary the target employer values. You may
     reuse that vocabulary ONLY to relabel an activity the candidate's input
     already describes.
   - Safe: candidate wrote "created embeddings in Pinecone" → "vector search with
     Pinecone". Same activity, clearer term.
   - UNSAFE: candidate lists "PyTorch" → writing "Analyzed deep learning model
     internals using PyTorch". That invents an activity the source never mentions.
   - DO NOT fabricate experiences to match JD requirements. DO NOT copy full
     sentences or responsibilities verbatim from the JD.

5. **METRICS — EVIDENCE ONLY (see "VERIFIED METRICS" in the user message):**
   - Use a number ONLY when it appears in the verified metrics list, and keep the
     value AND unit together. Write "30 hours", never a bare "30".
   - The Job Description and research may guide relevance and vocabulary, but they
     may NEVER supply accomplishments, durations, budgets, team sizes, or outcomes.
   - A requirement is not evidence: "5+ years of SQL" in the JD does not become
     "5+ years of SQL experience" in the resume unless the candidate's own profile
     states it.
   - If a useful metric is missing, leave the bullet unquantified. NEVER estimate,
     infer, or round.
   - A bullet without a metric is NOT a failure.

6. **STRONG BULLETS REQUIRED:**
   - Each bullet should be 15-25 words, starting with a strong action verb
   - Include specific technologies, tools, and methods from the user's input
   - Lead with the impact or result when metrics are available
   - BAD: "Applied technical skills to achieve measurable results" (generic, no specifics)
   - GOOD: "<Strong verb> <specific task> for <N> <unit from VERIFIED METRICS>; <second outcome>"

7. **DATE FORMAT & PRESERVATION (IMPORTANT):**
   - Copy the EXACT start and end dates from the User Input "Duration" field into the output `duration` field. Do NOT invent, approximate, or omit dates.
   - If the original says "04/2025 – Present" you MUST output "04/2025 – Present" (keep "Present" for current roles).
   - If the original says "12/2019 – 05/2023" you MUST output "12/2019 – 05/2023" (do NOT change past end-dates to "Present").
   - If the user typed a typo like "presetn", correct it to "Present" — but ONLY for the CURRENT (most recent, open-ended) role.
   - Format: use an en-dash (–) with a single space on each side. NEVER double-hyphens (--) or em-dashes (—).

### CRITICAL RULES:
1. **One Thought Per Bullet:** Do not combine unrelated tasks. Keep bullets punchy (15-25 words max).
2. **Front-Load the Impact:** Start with the Result or the strong Verb. (e.g., "Cut costs by 40%..." rather than "Responsible for cutting costs...")
3. **No Hallucinations:** Use ONLY the user's actual companies listed above.
4. **Concrete Tech:** Do not say "cutting-edge tech." Name the specific tool (e.g., "TensorFlow", "Kubernetes") ONLY if the user's input or the verified skills list names it. Research Data may suggest vocabulary for an activity the candidate already describes; it is never a source of tools the candidate did not use.

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
        "Strong Action Verb + Specific Tech + Quantifiable Result",
        "Strong Action Verb + Problem Solved + Benefit",
        "Strong Action Verb + Specific Tech + Business Outcome"
      ],
      "metrics_used": ["<value> <unit from VERIFIED METRICS>"],
      "style_transfer_rationale": "Mapped user's edge computing note to the Golden Bullet about 'High-availability distributed systems'."
    }}
  ],
  "seniority_applied": "{seniority_level}",
  "quantification_score": 0.95
}}
"""
    return system_prompt

def _build_drafter_prompts(
    experiences: List[Dict],
    job_ad: str,
    research_data: Dict[str, Any],
    golden_bullets: Optional[List[str]] = None,
    resume_text: str = "",
    extracted_job_title: Optional[str] = None,
) -> tuple:
    """Build (system_prompt, user_prompt) without calling a model.

    Extracted so R-5 (prompt policy regression, SPEC.v2 §9) can assert on the
    exact prompt text offline, at zero cost. Behaviour is identical to what
    Drafter.draft() sends.
    """
    seniority_level, seniority_config = get_seniority_config(job_ad)
    system_prompt = create_drafter_prompt(
        experiences=experiences,
        job_ad=job_ad,
        research_data=research_data,
        seniority_level=seniority_level,
        allowed_verbs=seniority_config["verbs"],
        golden_bullets=golden_bullets,
        tone_instruction="Maintain a standard, professional corporate tone.",
        extracted_job_title=extracted_job_title,
    )

    _search_text = (resume_text or "") + "\n" + json.dumps(experiences)
    _found_metrics = _extract_metrics(_search_text, limit=12)

    if _found_metrics:
        _metrics_block = "\n".join(
            f"- {m['value']} {m['unit']} ({m['kind']})"
            + (f" — from: …{m['source_text']}…" if m.get("source_text") else "")
            for m in _found_metrics
        )
    else:
        _metrics_block = "- (none found — leave bullets unquantified; never estimate)"

    user_prompt = f"""
User Experiences (JSON) - THIS IS THE SOURCE OF TRUTH.
Each experience has a 'bullets' (or 'description') field containing the ORIGINAL bullet content.
Mine this field for: metrics, technologies, action verbs, and domain-specific language.
DO NOT fabricate anything not in these descriptions.

{json.dumps(experiences, indent=2)}

Target Job Description - FOR CONTEXT ONLY (DO NOT COPY):
{job_ad[:3000]}

=== EVIDENCE BOUNDARY ===
Web research about this occupation is NOT included in this prompt, deliberately.
Research describing what people in this role typically do is not evidence about
this candidate. You may align vocabulary with the job description, but you may
never introduce a tool, platform, method, credential, project or accomplishment
that the User Experiences above do not contain. If the job description asks for
something the candidate's material does not mention, leave it out.

=== VERIFIED METRICS (extracted from the candidate's own resume) ===
{_metrics_block}

METRIC RULE (evidence only):
- You may use a number ONLY when it appears in the verified list above and you
  keep its value AND unit together (write "30 hours", never a bare "30").
- The Job Description may guide vocabulary and relevance. It may NEVER supply
  accomplishments, durations, budgets, team sizes, or outcomes. A requirement in
  the JD is not evidence of the candidate's history.
- If a useful metric is missing, leave the bullet unquantified. NEVER estimate,
  infer, round, or extrapolate a number.
- A bullet without a metric is completely acceptable and is NOT a failure.
"""
    return system_prompt, user_prompt


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
        resume_text: Optional[str] = None,
        on_token: Optional[Callable[[str], None]] = None,
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
        logger.info("[DRAFTER] Drafting experiences for job ad")
        
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
        
        # Pre-extract metrics from BOTH the raw resume text and structured experiences.
        #
        # R-2 fix (2026-10-30): use the normalized-quantity extractor so every
        # metric arrives with its unit ("30 hours", not "30") and $500K / 13
        # vessels / 10+ TB are no longer silently dropped (evidence C5).
        _search_text = (resume_text or '') + '\n' + json.dumps(experiences)
        _found_metrics = _extract_metrics(_search_text, limit=12)

        # R-5 fix (2026-10-30): the previous prompt contained a "3-TIER METRICS
        # POLICY" that explicitly authorised turning a JD requirement into a
        # candidate claim (TIER 2) and inventing team sizes / vessel counts
        # (TIER 3), plus a mandatory "METRIC INJECTION" block with a real
        # candidate's numbers hardcoded. That is the licence to fabricate.
        # Replaced below with an evidence-only rule.
        if _found_metrics:
            _metrics_block = "\n".join(
                f"- {m['value']} {m['unit']} ({m['kind']})"
                + (f" — from: …{m['source_text']}…" if m.get("source_text") else "")
                for m in _found_metrics
            )
        else:
            _metrics_block = "- (none found — leave bullets unquantified; never estimate)"

        user_prompt = f"""
User Experiences (JSON) - THIS IS THE SOURCE OF TRUTH.
Each experience has a 'bullets' (or 'description') field containing the ORIGINAL bullet content.
Mine this field for: metrics, technologies, action verbs, and domain-specific language.
DO NOT fabricate anything not in these descriptions.

{json.dumps(experiences, indent=2)}

Target Job Description - FOR CONTEXT ONLY (DO NOT COPY):
{job_ad[:3000]}

=== EVIDENCE BOUNDARY ===
Web research about this occupation is NOT included in this prompt, deliberately.
Research describing what people in this role typically do is not evidence about
this candidate. You may align vocabulary with the job description, but you may
never introduce a tool, platform, method, credential, project or accomplishment
that the User Experiences above do not contain. If the job description asks for
something the candidate's material does not mention, leave it out.

=== VERIFIED METRICS (extracted from the candidate's own resume) ===
{_metrics_block}

METRIC RULE (evidence only):
- You may use a number ONLY when it appears in the verified list above and you
  keep its value AND unit together (write "30 hours", never a bare "30").
- The Job Description may guide vocabulary and relevance. It may NEVER supply
  accomplishments, durations, budgets, team sizes, or outcomes. A requirement in
  the JD is not evidence of the candidate's history.
- If a useful metric is missing, leave the bullet unquantified. NEVER estimate,
  infer, round, or extrapolate a number.
- A bullet without a metric is completely acceptable and is NOT a failure.
"""
        
        try:
            # Call LLM with strict JSON schema and fallback support
            temperature = (
                temperature_override if temperature_override is not None else self.temperature
            )
            llm_kwargs: Dict[str, Any] = dict(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.model,
                temperature=temperature,
                response_format={"type": "json_object"},
                timeout=self.timeout,
                fallback_models=self.fallback_models,
                max_tokens=6000,  # 4000→6000: 6 experiences × 4 bullets + metrics + rationale needs ~5K tokens
            )
            # Optional token streaming (killswitch IMAG_STREAM_DRAFTER=false disables).
            # The client accumulates deltas and returns the full string, so the
            # JSON parse below is identical to the non-stream path.
            if on_token is not None and os.getenv("IMAG_STREAM_DRAFTER", "true").lower() != "false":
                llm_kwargs["stream_tokens"] = True
                llm_kwargs["on_token"] = on_token
            response = await self.llm_client.call_llm_async(**llm_kwargs)
            
            # Parse and validate response
            result = self._parse_response(response, experiences)
            
            # Track model in result (Note: actual model used is logged in adapter)
            result["model_requested"] = self.model
            
            logger.info("[DRAFTER] Created STAR bullets")
            
            # Ensure we have at least some data
            if not result.get("rewritten_experiences"):
                logger.warning("[DRAFTER] No rewritten experiences generated, using fallback")
                result["rewritten_experiences"] = []
                result["total_bullets"] = 0
                result["quantified_bullets"] = 0
                result["quantification_score"] = 0.0
                result["fallback"] = True
            
            return result
            
        except Exception:
            logger.error("[DRAFTER] Failed to draft experiences")
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
            #
            # R-6 fix (2026-10-30): join on a stable experience_id instead of
            # list position. The previous code used original_experiences[i],
            # so a model that reordered or dropped a role attached one job's
            # dates/location to another job — the most damaging resume error
            # and invisible to any keyword scorer.
            _by_id = {}
            _by_company = {}
            for _idx, _orig in enumerate(original_experiences):
                if not isinstance(_orig, dict):
                    continue
                _eid = _orig.get("experience_id") or _orig.get("id")
                if _eid:
                    _by_id[str(_eid)] = _orig
                _co = (_orig.get("company") or "").strip().lower()
                if _co:
                    _by_company.setdefault(_co, _orig)

            validated_experiences = []
            for i, exp in enumerate(rewritten):
                if isinstance(exp, dict):
                    # Resolve the ORIGINAL entry: id first, then company name,
                    # then (last resort) positional — never blindly by index.
                    _eid = exp.get("experience_id") or exp.get("id")
                    original_exp = None
                    if _eid and str(_eid) in _by_id:
                        original_exp = _by_id[str(_eid)]
                    if original_exp is None:
                        _co = (exp.get("company") or "").strip().lower()
                        original_exp = _by_company.get(_co)
                    if original_exp is None:
                        original_exp = (
                            original_experiences[i]
                            if i < len(original_experiences)
                            else {}
                        )
                    # Identity fields are owned by the ORIGINAL record. The model
                    # may rewrite bullets; it may not move dates between jobs.
                    _orig_company = original_exp.get("company") or ""
                    _orig_role = original_exp.get("role") or original_exp.get("title") or ""
                    _dur = (
                        original_exp.get("duration")
                        or original_exp.get("dates", "")
                        or exp.get("duration", "")
                    )
                    if not _dur and original_exp.get("start_date"):
                        _start = original_exp.get("start_date", "")
                        _raw_end = original_exp.get("end_date", "")
                        # Only default to "Present" for the most-recent job (index 0)
                        # when no end_date is provided; past jobs get an empty end date
                        # rather than a misleading "Present".
                        if _raw_end:
                            _end = _raw_end
                        elif i == 0:
                            _end = "Present"
                        else:
                            _end = ""
                        _dur = f"{_start} – {_end}" if _end else _start
                    if not _dur:
                        _dur = exp.get("duration", "")
                    validated = {
                        "experience_id": _eid or original_exp.get("experience_id", ""),
                        "company": _orig_company or exp.get("company", ""),
                        "role": _orig_role or exp.get("role", ""),
                        "duration": _dur,
                        "start_date": original_exp.get("start_date", ""),
                        "end_date": original_exp.get("end_date", ""),
                        "location": original_exp.get("location")
                        or exp.get("location", ""),
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
            
        except json.JSONDecodeError:
            logger.error("[DRAFTER] Failed to parse JSON response")
            
            # Try to extract JSON from text
            json_match = self._extract_json_from_text(response)
            if json_match:
                logger.info("[DRAFTER] Successfully extracted JSON from text")
                return self._parse_response(json_match, original_experiences)
            
            # Return fallback with error flag
            logger.warning("[DRAFTER] Using fallback output due to JSON parse failure")
            fallback = self._create_fallback_output(original_experiences, "mid")
            fallback["fallback_used"] = True
            fallback["parse_error"] = "invalid_json"
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
        
        import re as _re
        def _clean_company(name):
            return _re.sub(r'\s+\d{1,2}/\d{4}\s*[-–]\s*(Present|\d{1,2}/\d{4})\s*$', '', (name or '')).strip()

        for exp in experiences[:3]:  # Limit to 3 experiences
            raw_company = exp.get("company") or ""
            company = _clean_company(raw_company) or exp.get("title_line", "").split("|")[-1].strip()
            role = exp.get("title") or exp.get("role") or exp.get("title_line", "").split("|")[0].strip()
            duration = exp.get("duration") or exp.get("dates", "")
            location = exp.get("location", "")
            
            fallback_experiences.append({
                "company": company or "Previous Company",
                # Never invent a role title. An empty title is honest; "Professional
                # Role" is a fabricated fact that every later stage treats as real.
                "role": role or "",
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
