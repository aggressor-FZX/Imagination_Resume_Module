#!/usr/bin/env python3
"""
Imaginator Local Agentic Flow

Parses a resume, extrapolates skills, suggests roles, and performs a gap analysis via OpenAI API.
"""
import re
import json
import re
import json
import argparse
import os
from typing import Any, Dict, List, Optional, Set, Union
import asyncio
import google.generativeai as genai
# from deepseek import DeepSeekAPI  # Commented out - module not available

from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
import jsonschema

# Seniority detection integration
from seniority_detector import SeniorityDetector

# Load environment variables (expects OPENAI_API_KEY, ANTHROPIC_API_KEY, and GOOGLE_API_KEY)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Pricing (USD) per 1K tokens; can be overridden via env vars
OPENAI_PRICE_IN_K = float(os.getenv("OPENAI_PRICE_INPUT_PER_1K", "0.0005"))
OPENAI_PRICE_OUT_K = float(os.getenv("OPENAI_PRICE_OUTPUT_PER_1K", "0.0015"))
ANTHROPIC_PRICE_IN_K = float(os.getenv("ANTHROPIC_PRICE_INPUT_PER_1K", "0.003"))
ANTHROPIC_PRICE_OUT_K = float(os.getenv("ANTHROPIC_PRICE_OUTPUT_PER_1K", "0.015"))
GOOGLE_PRICE_IN_K = float(os.getenv("GOOGLE_PRICE_INPUT_PER_1K", "0.00025"))
GOOGLE_PRICE_OUT_K = float(os.getenv("GOOGLE_PRICE_OUTPUT_PER_1K", "0.0005"))
DEEPSEEK_PRICE_IN_K = float(os.getenv("DEEPSEEK_PRICE_INPUT_PER_1K", "0.0002"))
DEEPSEEK_PRICE_OUT_K = float(os.getenv("DEEPSEEK_PRICE_OUTPUT_PER_1K", "0.0008"))
QWEN_PRICE_IN_K = float(os.getenv("QWEN_PRICE_INPUT_PER_1K", "0.00006"))
QWEN_PRICE_OUT_K = float(os.getenv("QWEN_PRICE_OUTPUT_PER_1K", "0.00022"))
OPENROUTER_PRICE_IN_K = float(os.getenv("OPENROUTER_PRICE_INPUT_PER_1K", "0.0005"))
OPENROUTER_PRICE_OUT_K = float(os.getenv("OPENROUTER_PRICE_OUTPUT_PER_1K", "0.0015"))


def _estimate_openrouter_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate per-call cost based on model-specific OpenRouter pricing."""
    prompt_rate = OPENROUTER_PRICE_IN_K
    completion_rate = OPENROUTER_PRICE_OUT_K

    if "qwen/qwen3-30b-a3b" in model:
        prompt_rate = QWEN_PRICE_IN_K
        completion_rate = QWEN_PRICE_OUT_K
    elif "deepseek/deepseek-chat-v3.1" in model:
        prompt_rate = DEEPSEEK_PRICE_IN_K
        completion_rate = DEEPSEEK_PRICE_OUT_K
    elif "claude-3" in model:
        prompt_rate = ANTHROPIC_PRICE_IN_K
        completion_rate = ANTHROPIC_PRICE_OUT_K

    return (prompt_tokens / 1000.0) * prompt_rate + (completion_tokens / 1000.0) * completion_rate

# Run metrics accumulator
RUN_METRICS: Dict[str, Any] = {
    "calls": [],            # list of per-call usage/cost entries
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_tokens": 0,
    "estimated_cost_usd": 0.0,
    "failures": []          # list of failure dicts with provider/attempt/error
}

# Initialize clients (1.0+ APIs)
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
deepseek_client = None  # DeepSeekAPI module not available

# Initialize OpenRouter client (unified API)
openrouter_client = None
openrouter_async_client = None
if os.getenv("OPENROUTER_API_KEY"):
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "https://imaginator-resume-cowriter.onrender.com",
            "X-Title": "Imaginator Resume Co-Writer"
        }
    )
    # For async operations, we use the same client (OpenAI client is async-compatible)
    openrouter_async_client = openrouter_client

# Initialize async clients
openai_async_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None  # OpenAI client is async-compatible
anthropic_async_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None  # Anthropic client is async-compatible
google_async_client = genai.GenerativeModel('gemini-pro') if GOOGLE_API_KEY else None  # Google Gemini client
deepseek_async_client = None  # DeepSeekAPI module not available

# Keyword-based skill mapping
_SKILL_KEYWORDS = {
    "python": ["python", "pandas", "numpy", "django", "flask"],
    "data-analysis": ["analysis", "analytics", "sql", "tableau", "excel", "powerbi"],
    "machine-learning": ["model", "training", "ml", "scikit", "tensorflow", "pytorch"],
    "project-management": ["project", "pm", "managed", "scrum", "kanban", "stakeholder"],
    "cloud": ["aws", "azure", "gcp", "cloud", "lambda", "ecs", "s3"],
    "api": ["api", "rest", "graphql", "endpoint", "integration"],
    "devops": ["ci/cd", "docker", "kubernetes", "terraform", "jenkins"],
    "testing": ["test", "pytest", "integration test", "qa"],
    "communication": ["present", "communicat", "collaborat", "stakeholder", "writer"],
    "leadership": ["lead", "mentor", "manager", "managed team"]
}

# Role mapping for suggestions
_ROLE_MAP = {
    "data-scientist": {"python", "machine-learning", "data-analysis"},
    "data-engineer": {"python", "cloud", "api", "devops"},
    "ml-engineer": {"python", "machine-learning", "devops", "cloud"},
    "product-manager": {"project-management", "communication", "leadership"},
    "software-engineer": {"python", "api", "devops", "testing"},
}


def parse_experiences(text: str) -> List[Dict]:
    blocks = re.split(r'\n{2,}|experience|work history', text, flags=re.IGNORECASE)
    experiences = []
    for b in blocks:
        b = b.strip()
        if not b or len(b) < 40:
            continue
        lines = [l.strip() for l in b.splitlines() if l.strip()]
        title_line = lines[0] if lines else ""
        body = " ".join(lines[1:]) if len(lines) > 1 else " ".join(lines)
        
        # Extract duration information for seniority detection
        duration = extract_duration_from_text(b)
        
        experiences.append({
            "raw": b, 
            "title_line": title_line, 
            "body": body,
            "duration": duration,
            "description": f"{title_line} {body}"
        })
    return experiences


def extract_duration_from_text(text: str) -> str:
    """Extract duration information from experience text."""
    # Look for common date patterns
    date_patterns = [
        r'\d{1,2}/\d{4}\s*-\s*(?:\d{1,2}/\d{4}|present|current)',
        r'\w+\s+\d{4}\s*-\s*(?:\w+\s+\d{4}|present|current)',
        r'\d{4}\s*-\s*(?:\d{4}|present|current)',
        r'(?:\d+\s+years?|\d+\s+months?|\d+\s+yrs?)'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group()
    
    return ""


def extrapolate_skills_from_text(text: str) -> Set[str]:
    t = text.lower()
    found = set()
    for skill, keywords in _SKILL_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                found.add(skill)
                break
    if re.search(r'\b\d+%|\d+\s+users|\d+\s+clients|metrics\b', t):
        found.add("data-analysis")
    return found


def process_structured_skills(skills_data: Dict, confidence_threshold: float = 0.7, domain: str = None) -> Dict:
    """
    Process structured skills data from repos with confidence filtering and domain awareness
    
    Args:
        skills_data: Structured skills data from repos
        confidence_threshold: Minimum confidence score
        domain: Domain context for filtering (optional)
    
    Returns:
        Processed skills with filtering and prioritization
    """
    processed = {
        "high_confidence_skills": [],
        "medium_confidence_skills": [],
        "low_confidence_skills": [],
        "skill_confidences": {},
        "categories": {},
        "filtered_count": 0,
        "total_count": 0
    }
    
    if "skills" not in skills_data:
        return processed
    
    for skill_info in skills_data["skills"]:
        if not isinstance(skill_info, dict):
            continue
            
        skill_name = skill_info.get("skill", skill_info.get("name", ""))
        confidence = skill_info.get("confidence", 0)
        category = skill_info.get("category", "general")
        
        if not skill_name:
            continue
            
        processed["total_count"] += 1
        processed["skill_confidences"][skill_name] = confidence
        
        # Categorize by confidence
        if confidence >= confidence_threshold:
            processed["high_confidence_skills"].append(skill_name)
            processed["filtered_count"] += 1
        elif confidence >= 0.5:
            processed["medium_confidence_skills"].append(skill_name)
        else:
            processed["low_confidence_skills"].append(skill_name)
        
        # Group by category
        if category not in processed["categories"]:
            processed["categories"][category] = []
        processed["categories"][category].append(skill_name)
    
    # Sort skills by confidence (highest first)
    for category in processed["categories"]:
        processed["categories"][category].sort(
            key=lambda s: processed["skill_confidences"].get(s, 0), 
            reverse=True
        )
    
    return processed


def _load_json_payload(source: Union[str, Dict[str, Any], None], label: str) -> Dict[str, Any]:
    """Normalize JSON payloads, accepting dicts, JSON strings, or file paths."""
    if not source:
        return {}

    if isinstance(source, dict):
        return source

    if isinstance(source, str):
        candidate = source.strip()
        if not candidate:
            return {}

        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8") as handle:
                data = json.load(handle)
        else:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{label} must be a dict, valid JSON string, or file path"
                ) from exc

        if isinstance(data, dict):
            return data
        raise ValueError(f"{label} JSON content must deserialize to an object/dict")

    raise TypeError(f"{label} must be provided as a dict, JSON string, or file path")


def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing Markdown code fences to aid JSON parsing."""
    if not text:
        return ""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:]
        newline = stripped.find("\n")
        if newline != -1:
            stripped = stripped[newline + 1 :]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def _extract_json_object(text: str) -> Optional[str]:
    """Attempt to locate the first balanced JSON object within the text."""
    if not text:
        return None
    stack = 0
    start_idx: Optional[int] = None
    for idx, ch in enumerate(text):
        if ch == "{":
            if stack == 0:
                start_idx = idx
            stack += 1
        elif ch == "}":
            if stack:
                stack -= 1
                if stack == 0 and start_idx is not None:
                    candidate = text[start_idx : idx + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        continue
    return None


def ensure_json_dict(raw_text: str, label: str) -> Dict[str, Any]:
    """Best-effort conversion of model output into a JSON object."""
    cleaned = _strip_code_fences(raw_text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    candidate = _extract_json_object(cleaned)
    if candidate:
        return json.loads(candidate)

    raise ValueError(f"{label} response was not valid JSON. Raw output: {raw_text[:1200]}")


def call_llm(
    system_prompt: str, 
    user_prompt: str, 
    temperature: float = 0.9, 
    max_tokens: int = 1500, 
    max_retries: int = 3,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Call LLM with automatic fallback: OpenRouter -> OpenAI -> Google -> Anthropic
    Includes retry logic with exponential backoff and timeout handling.
    Returns the response text or raises an exception if all fail
    """
    import time
    
    # Simplified: Use OpenRouter as the sole LLM provider. If not configured, raise.
    if openrouter_client is None:
        raise RuntimeError("OpenRouter client is not configured. Set OPENROUTER_API_KEY in environment.")

    # Choose model based on prompt intent
    if "creative" in system_prompt.lower() or "generation" in user_prompt.lower():
        model = "qwen/qwen3-30b-a3b"
    elif "critic" in system_prompt.lower() or "review" in user_prompt.lower():
        model = "deepseek/deepseek-chat-v3.1"
    else:
        model = "anthropic/claude-3-haiku"

    try:
        create_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": 60.0,
        }
        if response_format:
            create_kwargs["response_format"] = response_format

        response = openrouter_client.chat.completions.create(**create_kwargs)
        text = response.choices[0].message.content.strip()

        # Attempt to track usage if available
        try:
            u = getattr(response, "usage", None)
            prompt_t = int(getattr(u, "prompt_tokens", 0) or 0)
            completion_t = int(getattr(u, "completion_tokens", 0) or 0)
        except Exception:
            prompt_t = completion_t = 0

        cost = _estimate_openrouter_cost(model, prompt_t, completion_t)
        RUN_METRICS["calls"].append({
            "provider": "openrouter",
            "model": model,
            "prompt_tokens": prompt_t,
            "completion_tokens": completion_t,
            "total_tokens": prompt_t + completion_t,
            "estimated_cost_usd": round(cost, 6)
        })
        RUN_METRICS["total_prompt_tokens"] += prompt_t
        RUN_METRICS["total_completion_tokens"] += completion_t
        RUN_METRICS["total_tokens"] += (prompt_t + completion_t)
        RUN_METRICS["estimated_cost_usd"] = round(RUN_METRICS["estimated_cost_usd"] + cost, 6)

        return text
    except Exception as e:
        raise RuntimeError(f"OpenRouter call failed: {type(e).__name__}: {e}")


async def call_llm_async(
    system_prompt: str, 
    user_prompt: str, 
    temperature: float = 0.9, 
    max_tokens: int = 1500, 
    max_retries: int = 3,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Async version of call_llm: Call LLM with automatic fallback using async clients.
    Includes retry logic with exponential backoff and timeout handling.
    Returns the response text or raises an exception if all fail
    """
    errors = []

    for attempt in range(max_retries):
        # Use temporary clients if API keys are provided
        temp_openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else openai_async_client
        temp_anthropic_client = Anthropic(api_key=anthropic_api_key) if anthropic_api_key else anthropic_async_client
        
        if google_api_key:
            genai.configure(api_key=google_api_key)
            temp_google_client = genai.GenerativeModel('gemini-pro')
        else:
            temp_google_client = google_async_client

        temp_deepseek_client = deepseek_async_client  # DeepSeekAPI module not available

        # Try OpenRouter first (primary provider)
        if openrouter_async_client:
            # Determine best model for the task
            if "creative" in system_prompt.lower() or "generation" in user_prompt.lower():
                model = "qwen/qwen3-30b-a3b"  # Creative writing specialist
                print("🎨 Using OpenRouter (Qwen3-30B-A3B for creative generation)", flush=True)
            elif "critic" in system_prompt.lower() or "review" in user_prompt.lower():
                model = "deepseek/deepseek-chat-v3.1"  # Critical analysis specialist  
                print("🎯 Using OpenRouter (DeepSeek Chat v3.1 for critical analysis)", flush=True)
            else:
                model = "anthropic/claude-3-haiku"  # General purpose analysis
                print("🔍 Using OpenRouter (Claude-3-Haiku for analysis)", flush=True)
            
            try:
                import inspect
                create_kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": 60.0,
                }
                if response_format:
                    create_kwargs["response_format"] = response_format

                maybe = openrouter_async_client.chat.completions.create(**create_kwargs)
                # Some OpenRouter client versions return an awaitable coroutine, others
                # return a ChatCompletion object directly. Handle both safely.
                try:
                    if inspect.isawaitable(maybe):
                        response = await maybe
                    else:
                        response = maybe
                except TypeError as te:
                    # Protect against `TypeError: object ChatCompletion can't be used in 'await' expression`
                    print(f"⚠️  OpenRouter returned non-awaitable object: {type(maybe).__name__} - {te}", flush=True)
                    response = maybe

                # Log the response type for easier debugging on Render if it changes
                print(f"🔎 OpenRouter returned type: {type(response).__name__}", flush=True)
                text = response.choices[0].message.content.strip()
                # Usage and cost tracking
                try:
                    u = getattr(response, "usage", None)
                    prompt_t = int(getattr(u, "prompt_tokens", 0) or 0)
                    completion_t = int(getattr(u, "completion_tokens", 0) or 0)
                    total_t = int(getattr(u, "total_tokens", prompt_t + completion_t))
                except Exception:
                    prompt_t = completion_t = 0
                    total_t = 0
                
                cost = _estimate_openrouter_cost(model, prompt_t, completion_t)
                RUN_METRICS["calls"].append({
                    "provider": "openrouter",
                    "model": model,
                    "prompt_tokens": prompt_t,
                    "completion_tokens": completion_t,
                    "total_tokens": total_t,
                    "estimated_cost_usd": round(cost, 6)
                })
                RUN_METRICS["total_prompt_tokens"] += prompt_t
                RUN_METRICS["total_completion_tokens"] += completion_t
                RUN_METRICS["total_tokens"] += total_t
                RUN_METRICS["estimated_cost_usd"] = round(RUN_METRICS["estimated_cost_usd"] + cost, 6)
                return text
            except Exception as e:
                error_msg = f"OpenRouter Error: {type(e).__name__}: {str(e)}"
                errors.append(error_msg)
                RUN_METRICS["failures"].append({"attempt": attempt + 1, "provider": "openrouter", "error": str(e)})
                print(f"⚠️  OpenRouter failed: {type(e).__name__}. Trying OpenAI...", flush=True)

        # Try OpenAI next
        if temp_openai_client:
            try:
                response = await temp_openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=60.0  # 60 second timeout
                )
                print("✅ Using OpenAI GPT-3.5-turbo", flush=True)
                text = response.choices[0].message.content.strip()
                # Usage and cost
                try:
                    u = getattr(response, "usage", None)
                    prompt_t = int(getattr(u, "prompt_tokens", 0) or 0)
                    completion_t = int(getattr(u, "completion_tokens", 0) or 0)
                    total_t = int(getattr(u, "total_tokens", prompt_t + completion_t))
                except Exception:
                    prompt_t = completion_t = 0
                    total_t = 0
                cost = (prompt_t / 1000.0) * OPENAI_PRICE_IN_K + (completion_t / 1000.0) * OPENAI_PRICE_OUT_K
                RUN_METRICS["calls"].append({
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "prompt_tokens": prompt_t,
                    "completion_tokens": completion_t,
                    "total_tokens": total_t,
                    "estimated_cost_usd": round(cost, 6)
                })
                RUN_METRICS["total_prompt_tokens"] += prompt_t
                RUN_METRICS["total_completion_tokens"] += completion_t
                RUN_METRICS["total_tokens"] += total_t
                RUN_METRICS["estimated_cost_usd"] = round(RUN_METRICS["estimated_cost_usd"] + cost, 6)
                return text
            except Exception as e:
                error_msg = f"OpenAI Error: {type(e).__name__}: {str(e)}"
                errors.append(error_msg)
                RUN_METRICS["failures"].append({"attempt": attempt + 1, "provider": "openai", "error": str(e)})
                print(f"⚠️  OpenAI failed: {type(e).__name__}. Trying Anthropic...", flush=True)

        # Try Google next
        if temp_google_client:
            try:
                response = await temp_google_client.generate_content_async(
                    f"{system_prompt}\n\n{user_prompt}",
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    )
                )
                print("✅ Using Google Gemini", flush=True)
                # Extract text and usage
                text = response.text
                try:
                    u = getattr(response, "usage", None)
                    input_t = int(getattr(u, "input_tokens", 0) or 0)
                    output_t = int(getattr(u, "output_tokens", 0) or 0)
                    total_t = input_t + output_t
                except Exception:
                    input_t = output_t = total_t = 0
                cost = (input_t / 1000.0) * GOOGLE_PRICE_IN_K + (output_t / 1000.0) * GOOGLE_PRICE_OUT_K
                RUN_METRICS["calls"].append({
                    "provider": "google",
                    "model": "gemini-pro",
                    "prompt_tokens": input_t,
                    "completion_tokens": output_t,
                    "total_tokens": total_t,
                    "estimated_cost_usd": round(cost, 6)
                })
                RUN_METRICS["total_prompt_tokens"] += input_t
                RUN_METRICS["total_completion_tokens"] += output_t
                RUN_METRICS["total_tokens"] += total_t
                RUN_METRICS["estimated_cost_usd"] = round(RUN_METRICS["estimated_cost_usd"] + cost, 6)
                return text
            except Exception as e:
                error_msg = f"Google Gemini Error: {type(e).__name__}: {str(e)}"
                errors.append(error_msg)
                RUN_METRICS["failures"].append({"attempt": attempt + 1, "provider": "google", "error": str(e)})
                print(f"⚠️  Google Gemini failed: {type(e).__name__}. Trying Anthropic...", flush=True)

        # Fallback to Anthropic
        if temp_anthropic_client:
            try:
                response = await temp_anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",  # Latest Sonnet model
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                print("✅ Using Anthropic Claude 3.5 Sonnet", flush=True)
                # Extract text and usage
                text = response.content[0].text
                try:
                    u = getattr(response, "usage", None)
                    input_t = int(getattr(u, "input_tokens", 0) or 0)
                    output_t = int(getattr(u, "output_tokens", 0) or 0)
                    total_t = input_t + output_t
                except Exception:
                    input_t = output_t = total_t = 0
                cost = (input_t / 1000.0) * ANTHROPIC_PRICE_IN_K + (output_t / 1000.0) * ANTHROPIC_PRICE_OUT_K
                RUN_METRICS["calls"].append({
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet-20241022",
                    "prompt_tokens": input_t,
                    "completion_tokens": output_t,
                    "total_tokens": total_t,
                    "estimated_cost_usd": round(cost, 6)
                })
                RUN_METRICS["total_prompt_tokens"] += input_t
                RUN_METRICS["total_completion_tokens"] += output_t
                RUN_METRICS["total_tokens"] += total_t
                RUN_METRICS["estimated_cost_usd"] = round(RUN_METRICS["estimated_cost_usd"] + cost, 6)
                return text
            except Exception as e:
                error_msg = f"Anthropic Error: {type(e).__name__}: {str(e)}"
                errors.append(error_msg)
                RUN_METRICS["failures"].append({"attempt": attempt + 1, "provider": "anthropic", "error": str(e)})
                print(f"❌ Anthropic also failed: {type(e).__name__}", flush=True)

        # Fall-back to DeepSeek
        if deepseek_async_client:
            try:
                response = await deepseek_async_client.chat.completions.create(
                    model="deepseek/deepseek-chat-v3.1",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                print("✅ Using DeepSeek", flush=True)
                text = response.choices[0].message.content.strip()
                # Usage and cost
                try:
                    u = getattr(response, "usage", None)
                    prompt_t = int(getattr(u, "prompt_tokens", 0) or 0)
                    completion_t = int(getattr(u, "completion_tokens", 0) or 0)
                    total_t = int(getattr(u, "total_tokens", prompt_t + completion_t))
                except Exception:
                    prompt_t = completion_t = 0
                    total_t = 0
                cost = (prompt_t / 1000.0) * DEEPSEEK_PRICE_IN_K + (completion_t / 1000.0) * DEEPSEEK_PRICE_OUT_K
                RUN_METRICS["calls"].append({
                    "provider": "deepseek",
                    "model": "gemini-pro",
                    "prompt_tokens": prompt_t,
                    "completion_tokens": completion_t,
                    "total_tokens": total_t,
                    "estimated_cost_usd": round(cost, 6)
                })
                RUN_METRICS["total_prompt_tokens"] += prompt_t
                RUN_METRICS["total_completion_tokens"] += completion_t
                RUN_METRICS["total_tokens"] += total_t
                RUN_METRICS["estimated_cost_usd"] = round(RUN_METRICS["estimated_cost_usd"] + cost, 6)
                return text
            except Exception as e:
                error_msg = f"DeepSeek Error: {type(e).__name__}: {str(e)}"
                errors.append(error_msg)
                RUN_METRICS["failures"].append({"attempt": attempt + 1, "provider": "deepseek", "error": str(e)})
                print(f"❌ DeepSeek also failed: {error_msg}", flush=True)
        
        # If we get here, all providers failed on this attempt
        if attempt < max_retries - 1:
            wait_time = min(2 ** attempt, 30)
            print(f"🔄 All providers failed, retrying in {wait_time}s...", flush=True)
            await asyncio.sleep(wait_time)

    # All retries exhausted
    error_msg = f"All LLM providers failed after {max_retries} attempts:\n" + "\n".join(errors)
    raise RuntimeError(error_msg)
def suggest_roles(skills: Set[str]) -> List[Dict]:
    suggestions = []
    for role, reqs in _ROLE_MAP.items():
        match = skills & reqs
        if match:
            score = round(len(match) / len(reqs), 2)
            suggestions.append({"role": role, "score": score, "matched_skills": sorted(match)})
    return sorted(suggestions, key=lambda x: x['score'], reverse=True)


def load_knowledge_bases():
    """Load skill adjacency and verb competency knowledge bases"""
    skill_adjacency = {}
    verb_competency = {}
    
    try:
        with open("skill_adjacency.json", encoding="utf-8") as f:
            skill_adjacency = json.load(f).get("mappings", {})
    except FileNotFoundError:
        pass  # Optional knowledge base
    
    try:
        with open("verb_competency.json", encoding="utf-8") as f:
            verb_competency = json.load(f).get("mappings", {})
    except FileNotFoundError:
        pass  # Optional knowledge base
    
    return skill_adjacency, verb_competency


def infer_implied_skills(high_conf_skills: List[str], skill_adjacency: Dict) -> Dict[str, float]:
    """Infer skills that are likely present based on adjacent skill mappings"""
    implied = {}
    
    for skill in high_conf_skills:
        skill_lower = skill.lower().replace(" ", "_")
        if skill_lower in skill_adjacency:
            for adjacent_skill, confidence in skill_adjacency[skill_lower].items():
                if adjacent_skill not in implied or implied[adjacent_skill] < confidence:
                    implied[adjacent_skill] = confidence
    
    # Filter out low-confidence inferences (< 0.75)
    return {skill: conf for skill, conf in implied.items() if conf >= 0.75}


def extract_competencies(resume_text: str, verb_competency: Dict) -> Dict[str, List[Dict]]:
    """Extract competency domains from action verbs in resume"""
    import re
    competencies = {}
    
    for verb, domains in verb_competency.items():
        # Look for past-tense verbs in resume
        pattern = rf'\b{verb}\b'
        matches = re.findall(pattern, resume_text, re.IGNORECASE)
        
        if matches:
            for domain, confidence in domains.items():
                if domain not in competencies:
                    competencies[domain] = []
                competencies[domain].append({
                    "verb": verb,
                    "confidence": confidence,
                    "occurrences": len(matches)
                })
    
    # Sort by total confidence * occurrences
    for domain in competencies:
        competencies[domain] = sorted(
            competencies[domain],
            key=lambda x: x['confidence'] * x['occurrences'],
            reverse=True
        )
    
    return competencies


def generate_gap_analysis(resume_text: str, processed_skills: Dict, roles: List[Dict], target_job_ad: str, domain_insights: Dict = None) -> str:
    """
    Generate creative gap analysis using multi-perspective synthesis and knowledge bases
    
    Args:
        resume_text: Raw resume text
        processed_skills: Processed skills with confidence filtering from process_structured_skills
        roles: Role suggestions
        target_job_ad: Target job description
        domain_insights: Domain-specific insights from Hermes (optional)
    """
    
    # Load knowledge bases
    skill_adjacency, verb_competency = load_knowledge_bases()
    
    # Extract skill information
    high_conf_skills = processed_skills.get("high_confidence_skills", [])
    skill_confidences = processed_skills.get("skill_confidences", {})
    skill_categories = processed_skills.get("categories", {})
    
    # Infer implied skills using knowledge base
    implied_skills = infer_implied_skills(high_conf_skills, skill_adjacency)
    
    # Extract competencies from resume verbs
    competencies = extract_competencies(resume_text, verb_competency)
    
    # Build enhanced multi-perspective prompt
    prompt_parts = [
        "You are an AI career strategist with three expert perspectives:",
        "",
        "🎯 PERSPECTIVE 1: HIRING MANAGER",
        "Focus: What would make this candidate stand out to me? What gaps are dealbreakers vs. nice-to-haves?",
        "",
        "🏗️ PERSPECTIVE 2: DOMAIN ARCHITECT  ",
        "Focus: Technical depth, system design capabilities, architectural thinking, scalability mindset.",
        "",
        "🚀 PERSPECTIVE 3: CAREER COACH",
        "Focus: Growth trajectory, learning agility, transferable skills, creative development paths.",
        "",
        "=== CANDIDATE PROFILE ===",
        f"Resume Extract:\n{resume_text[:500]}...\n",
        "",
        f"🔥 HIGH-CONFIDENCE SKILLS (≥0.7): {', '.join(sorted(high_conf_skills))}",
        "",
        f"📊 SKILL CONFIDENCE MATRIX:\n{json.dumps(skill_confidences, indent=2)}",
        ""
    ]
    
    # Add implied skills section
    if implied_skills:
        prompt_parts.extend([
            f"💡 IMPLIED SKILLS (inferred from adjacent skill mappings):",
            json.dumps(dict(sorted(implied_skills.items(), key=lambda x: x[1], reverse=True)[:10]), indent=2),
            ""
        ])
    
    # Add competency domains
    if competencies:
        top_competencies = sorted(
            [(domain, sum(v['confidence'] * v['occurrences'] for v in verbs)) 
             for domain, verbs in competencies.items()],
            key=lambda x: x[1],
            reverse=True
        )[:8]
        prompt_parts.extend([
            f"🎓 EXTRACTED COMPETENCY DOMAINS (from action verbs):",
            ", ".join([f"{domain} ({score:.2f})" for domain, score in top_competencies]),
            ""
        ])
    
    # Add skill categories if available
    if skill_categories:
        prompt_parts.append("📚 SKILL CATEGORIES:")
        for category, skills in skill_categories.items():
            if skills:
                prompt_parts.append(f"  • {category.title()}: {', '.join(skills[:5])}")
        prompt_parts.append("")
    
    prompt_parts.extend([
        f"🎯 TARGET ROLE MATCHES: {', '.join(r['role'] for r in roles)}",
        "",
        f"📋 TARGET JOB DESCRIPTION:\n{target_job_ad}\n"
    ])
    
    # Add domain insights if available
    if domain_insights:
        if "domain" in domain_insights:
            prompt_parts.append(f"🏢 INDUSTRY DOMAIN: {domain_insights['domain'].upper()}")
        
        if "insights" in domain_insights:
            insights = domain_insights["insights"]
            if "strengths" in insights:
                prompt_parts.append(f"✅ AI-DETECTED STRENGTHS: {', '.join(insights['strengths'])}")
            if "gaps" in insights:
                prompt_parts.append(f"⚠️ AI-DETECTED GAPS: {', '.join(insights['gaps'])}")
            if "market_alignment" in insights:
                prompt_parts.append(f"📈 MARKET ALIGNMENT: {insights['market_alignment']}/1.0")
        prompt_parts.append("")
    
    prompt_parts.append("""
=== SYNTHESIS TASK ===

Synthesize the THREE perspectives above into a comprehensive career development analysis with the following structure:

**OUTPUT MUST BE VALID JSON WITH THIS EXACT STRUCTURE:**

```json
{
  "gap_analysis": {
    "critical_gaps": ["skill1", "skill2"],
    "nice_to_have_gaps": ["skill3", "skill4"],
    "gap_bridging_strategy": "2-3 sentences on how to prioritize"
  },
  
  "implied_skills": {
    "skill_name": {
      "confidence": 0.85,
      "evidence": "Why we think candidate has this",
      "development_path": "How to formalize/strengthen it"
    }
  },
  
  "environment_capabilities": {
    "tech_stack": ["inferred_tech1", "inferred_tech2"],
    "tools": ["tool1", "tool2"],
    "platforms": ["platform1", "platform2"],
    "reasoning": "Why we infer these based on skills/experience"
  },
  
  "transfer_paths": [
    {
      "from_role": "current likely role",
      "to_role": "target role",
      "timeline": "6-12 months",
      "key_bridges": ["skill to develop", "experience to gain"],
      "probability": 0.75
    }
  ],
  
  "project_briefs": [
    {
      "title": "Concrete Project Idea",
      "description": "1-2 sentences what to build",
      "skills_practiced": ["skill1", "skill2"],
      "estimated_duration": "2-4 weeks",
      "impact_on_gaps": "How this project bridges specific gaps",
      "difficulty": "beginner|intermediate|advanced"
    }
  ],
  
  "multi_perspective_insights": {
    "hiring_manager_view": "What would excite/concern a hiring manager",
    "architect_view": "Technical depth assessment and growth areas",
    "coach_view": "Growth potential and creative development strategies"
  },
  
  "action_plan": {
    "quick_wins": ["actionable item 1", "actionable item 2"],
    "3_month_goals": ["goal1", "goal2"],
    "6_month_goals": ["goal1", "goal2"],
    "long_term_vision": "Where this candidate should aim in 1-2 years"
  }
}
```

Be creative, specific, and actionable. Use the implied skills, competency domains, and confidence scores to make nuanced recommendations.""")
    
    prompt = "\n".join(prompt_parts)
    
    # Call LLM with automatic fallback (OpenAI → Anthropic)
    system_prompt = "You are a creative career strategist synthesizing hiring manager, architect, and coach perspectives. Respond ONLY with valid JSON matching the requested structure."
    return call_llm(system_prompt, prompt, temperature=0.9, max_tokens=1500)


async def generate_gap_analysis_async(resume_text: str, processed_skills: Dict, roles: List[Dict], target_job_ad: str, domain_insights: Dict = None) -> str:
    """
    Async version of generate_gap_analysis: Generate creative gap analysis using multi-perspective synthesis and knowledge bases

    Args:
        resume_text: Raw resume text
        processed_skills: Processed skills with confidence filtering from process_structured_skills
        roles: Role suggestions
        target_job_ad: Target job description
        domain_insights: Domain-specific insights from Hermes (optional)
    """

    # Load knowledge bases
    skill_adjacency, verb_competency = load_knowledge_bases()

    # Extract skill information
    high_conf_skills = processed_skills.get("high_confidence_skills", [])
    skill_confidences = processed_skills.get("skill_confidences", {})
    skill_categories = processed_skills.get("categories", {})

    # Infer implied skills using knowledge base
    implied_skills = infer_implied_skills(high_conf_skills, skill_adjacency)

    # Extract competencies from resume verbs
    competencies = extract_competencies(resume_text, verb_competency)

    # Build enhanced multi-perspective prompt
    prompt_parts = [
        "You are an AI career strategist with three expert perspectives:",
        "",
        "🎯 PERSPECTIVE 1: HIRING MANAGER",
        "Focus: What would make this candidate stand out to me? What gaps are dealbreakers vs. nice-to-haves?",
        "",
        "🏗️ PERSPECTIVE 2: DOMAIN ARCHITECT  ",
        "Focus: Technical depth, system design capabilities, architectural thinking, scalability mindset.",
        "",
        "🚀 PERSPECTIVE 3: CAREER COACH",
        "Focus: Growth trajectory, learning agility, transferable skills, creative development paths.",
        "",
        "=== CANDIDATE PROFILE ===",
        f"Resume Extract:\n{resume_text[:500]}...\n",
        "",
        f"🔥 HIGH-CONFIDENCE SKILLS (≥0.7): {', '.join(sorted(high_conf_skills))}",
        "",
        f"📊 SKILL CONFIDENCE MATRIX:\n{json.dumps(skill_confidences, indent=2)}",
        ""
    ]

    # Add implied skills section
    if implied_skills:
        prompt_parts.extend([
            f"💡 IMPLIED SKILLS (inferred from adjacent skill mappings):",
            json.dumps(dict(sorted(implied_skills.items(), key=lambda x: x[1], reverse=True)[:10]), indent=2),
            ""
        ])

    # Add competency domains
    if competencies:
        top_competencies = sorted(
            [(domain, sum(v['confidence'] * v['occurrences'] for v in verbs))
             for domain, verbs in competencies.items()],
            key=lambda x: x[1],
            reverse=True
        )[:8]
        prompt_parts.extend([
            f"🎓 EXTRACTED COMPETENCY DOMAINS (from action verbs):",
            ", ".join([f"{domain} ({score:.2f})" for domain, score in top_competencies]),
            ""
        ])

    # Add skill categories if available
    if skill_categories:
        prompt_parts.append("📚 SKILL CATEGORIES:")
        for category, skills in skill_categories.items():
            if skills:
                prompt_parts.append(f"  • {category.title()}: {', '.join(skills[:5])}")
        prompt_parts.append("")

    prompt_parts.extend([
        f"🎯 TARGET ROLE MATCHES: {', '.join(r['role'] for r in roles)}",
        "",
        f"📋 TARGET JOB DESCRIPTION:\n{target_job_ad}\n"
    ])

    # Add domain insights if available
    if domain_insights:
        if "domain" in domain_insights:
            prompt_parts.append(f"🏢 INDUSTRY DOMAIN: {domain_insights['domain'].upper()}")

        if "insights" in domain_insights:
            insights = domain_insights["insights"]
            if "strengths" in insights:
                prompt_parts.append(f"✅ AI-DETECTED STRENGTHS: {', '.join(insights['strengths'])}")
            if "gaps" in insights:
                prompt_parts.append(f"⚠️ AI-DETECTED GAPS: {', '.join(insights['gaps'])}")
            if "market_alignment" in insights:
                prompt_parts.append(f"📈 MARKET ALIGNMENT: {insights['market_alignment']}/1.0")
        prompt_parts.append("")

    prompt_parts.append("""
=== SYNTHESIS TASK ===

Synthesize the THREE perspectives above into a comprehensive career development analysis with the following structure:

**OUTPUT MUST BE VALID JSON WITH THIS EXACT STRUCTURE:**

```json
{
  "gap_analysis": {
    "critical_gaps": ["skill1", "skill2"],
    "nice_to_have_gaps": ["skill3", "skill4"],
    "gap_bridging_strategy": "2-3 sentences on how to prioritize"
  },
  
  "implied_skills": {
    "skill_name": {
      "confidence": 0.85,
      "evidence": "Why we think candidate has this",
      "development_path": "How to formalize/strengthen it"
    }
  },
  
  "environment_capabilities": {
    "tech_stack": ["inferred_tech1", "inferred_tech2"],
    "tools": ["tool1", "tool2"],
    "platforms": ["platform1", "platform2"],
    "reasoning": "Why we infer these based on skills/experience"
  },
  
  "transfer_paths": [
    {
      "from_role": "current likely role",
      "to_role": "target role",
      "timeline": "6-12 months",
      "key_bridges": ["skill to develop", "experience to gain"],
      "probability": 0.75
    }
  ],
  
  "project_briefs": [
    {
      "title": "Concrete Project Idea",
      "description": "1-2 sentences what to build",
      "skills_practiced": ["skill1", "skill2"],
      "estimated_duration": "2-4 weeks",
      "impact_on_gaps": "How this project bridges specific gaps",
      "difficulty": "beginner|intermediate|advanced"
    }
  ],
  
  "multi_perspective_insights": {
    "hiring_manager_view": "What would excite/concern a hiring manager",
    "architect_view": "Technical depth assessment and growth areas",
    "coach_view": "Growth potential and creative development strategies"
  },
  
  "action_plan": {
    "quick_wins": ["actionable item 1", "actionable item 2"],
    "3_month_goals": ["goal1", "goal2"],
    "6_month_goals": ["goal1", "goal2"],
    "long_term_vision": "Where this candidate should aim in 1-2 years"
  }
}
```

Be creative, specific, and actionable. Use the implied skills, competency domains, and confidence scores to make nuanced recommendations.""")
    
    prompt = "\n".join(prompt_parts)
    
    # Call LLM with automatic fallback (OpenAI → Anthropic)
    system_prompt = "You are a creative career strategist synthesizing hiring manager, architect, and coach perspectives. Respond ONLY with valid JSON matching the requested structure."
    return await call_llm_async(system_prompt, prompt, temperature=0.9, max_tokens=1500)


def generate_gap_analysis_baseline(resume_text: str, processed_skills: Dict, roles: List[Dict], target_job_ad: str, domain_insights: Dict = None) -> str:
    """Original baseline prompt for comparison testing - simpler single-perspective approach"""
    
    high_conf_skills = processed_skills.get("high_confidence_skills", [])
    skill_confidences = processed_skills.get("skill_confidences", {})
    
    prompt = f"""You are a career coach analyzing a candidate's profile.

RESUME EXTRACT:
{resume_text[:500]}...

HIGH-CONFIDENCE SKILLS: {', '.join(sorted(high_conf_skills))}

SKILL CONFIDENCE SCORES:
{json.dumps(skill_confidences, indent=2)}

TARGET JOB:
{target_job_ad}

Provide a creative gap analysis with:
- Key strengths
- Development recommendations  
- Domain-specific insights
- Action plan

Use emojis and bullet points for readability."""
    
    # Call LLM with automatic fallback (OpenAI → Anthropic)
    system_prompt = "You are a helpful career coach. Provide creative, actionable advice."
    return call_llm(system_prompt, prompt, temperature=0.8, max_tokens=800)


# Removed all fallback code below - errors will now propagate properly
def _old_fallback_removed():
    """This function marks where the old 400+ line fallback JSON was removed"""
    return json.dumps({
        "_note": "Demo fallback removed. OpenAI API errors will now propagate.",
            "gap_analysis": {
                "critical_gaps": ["react_production_experience", "kubernetes_at_scale"],
                "nice_to_have_gaps": ["graphql", "serverless_architecture"],
                "gap_bridging_strategy": "Focus on React + AWS combination through portfolio projects. Critical to demonstrate production K8s experience through contributions or side projects."
            },
            "implied_skills": {
                "scripting": {
                    "confidence": 0.95,
                    "evidence": "Python expertise (0.95) strongly implies shell scripting, automation scripts, and general scripting capabilities",
                    "development_path": "Formalize with GitHub repos showcasing automation tooling and CI/CD scripts"
                },
                "backend_development": {
                    "confidence": 0.88,
                    "evidence": "Python (0.95) + API experience suggests strong backend development foundation",
                    "development_path": "Build RESTful API with FastAPI/Django, deploy on AWS with Docker"
                },
                "cloud_native": {
                    "confidence": 0.85,
                    "evidence": "AWS (0.92) + Docker (0.87) + Kubernetes (0.78) indicates cloud-native architecture understanding",
                    "development_path": "Design and document a cloud-native microservices architecture"
                },
                "devops": {
                    "confidence": 0.82,
                    "evidence": "Docker + Kubernetes + AWS combo strongly suggests DevOps practices",
                    "development_path": "Create end-to-end CI/CD pipeline with GitHub Actions + AWS + K8s"
                }
            },
            "environment_capabilities": {
                "tech_stack": ["python_backend", "node.js_services", "react_frontend", "postgresql", "redis_caching"],
                "tools": ["git", "github_actions", "terraform", "prometheus", "grafana", "elk_stack"],
                "platforms": ["aws_ec2", "aws_lambda", "aws_rds", "aws_s3", "docker_hub", "kubernetes_clusters"],
                "reasoning": "AWS + Docker + Python/JS stack suggests modern cloud-native development environment. K8s proficiency implies familiarity with monitoring (Prometheus/Grafana) and logging (ELK). React indicates frontend tooling (Webpack, Babel). Infrastructure-as-code tools like Terraform are standard with this profile."
            },
            "transfer_paths": [
                {
                    "from_role": "Backend Python Developer",
                    "to_role": "Full-Stack Engineer (Target Role)",
                    "timeline": "4-6 months",
                    "key_bridges": ["React production projects", "TypeScript proficiency", "full-stack authentication"],
                    "probability": 0.85
                },
                {
                    "from_role": "DevOps Engineer",
                    "to_role": "Platform/SRE Engineer",
                    "timeline": "6-9 months",
                    "key_bridges": ["Advanced K8s (Helm, Operators)", "Observability stack", "Incident response"],
                    "probability": 0.75
                },
                {
                    "from_role": "Cloud Engineer",
                    "to_role": "Solutions Architect",
                    "timeline": "9-12 months",
                    "key_bridges": ["Multi-cloud (GCP/Azure)", "Enterprise architecture patterns", "Cost optimization"],
                    "probability": 0.65
                }
            ],
            "project_briefs": [
                {
                    "title": "Real-Time Collaborative Task Manager",
                    "description": "Build a production-ready task management SaaS with React frontend, Python/FastAPI backend, WebSockets for real-time updates, PostgreSQL + Redis, deployed on AWS with K8s.",
                    "skills_practiced": ["react", "python", "websockets", "kubernetes", "aws", "postgresql", "redis"],
                    "estimated_duration": "6-8 weeks",
                    "impact_on_gaps": "Directly addresses React production experience gap and demonstrates full-stack capabilities. Shows K8s deployment skills.",
                    "difficulty": "intermediate"
                },
                {
                    "title": "ML Model Deployment Pipeline",
                    "description": "Create MLOps pipeline: Train scikit-learn model, containerize with Docker, deploy to AWS Lambda + API Gateway, add monitoring with CloudWatch.",
                    "skills_practiced": ["machine_learning", "docker", "aws_lambda", "ci_cd", "monitoring"],
                    "estimated_duration": "3-4 weeks",
                    "impact_on_gaps": "Bridges ML gap while leveraging existing AWS/Docker strengths. Demonstrates end-to-end ML deployment.",
                    "difficulty": "intermediate"
                },
                {
                    "title": "Infrastructure-as-Code Portfolio",
                    "description": "Use Terraform to provision multi-tier AWS architecture (VPC, EC2, RDS, S3, ALB). Document architecture decisions and cost optimization strategies.",
                    "skills_practiced": ["terraform", "aws", "infrastructure_as_code", "networking", "security"],
                    "estimated_duration": "2-3 weeks",
                    "impact_on_gaps": "Formalizes cloud infrastructure knowledge, demonstrates architectural thinking.",
                    "difficulty": "beginner"
                },
                {
                    "title": "Kubernetes Operator for Custom Resource",
                    "description": "Build a simple K8s operator in Python to manage custom resources. Shows advanced K8s understanding beyond basic deployments.",
                    "skills_practiced": ["kubernetes", "python", "api_development", "operator_pattern"],
                    "estimated_duration": "4-5 weeks",
                    "impact_on_gaps": "Elevates K8s skills to advanced level, differentiates from typical DevOps candidates.",
                    "difficulty": "advanced"
                }
            ],
            "multi_perspective_insights": {
                "hiring_manager_view": "Strong Python+AWS foundation makes this candidate immediately productive. Main concern: React experience seems limited. Would I hire for full-stack role? Maybe with conditional offer pending React proficiency demonstration. For backend/platform role? Definitely yes. Recommendation: Candidate should lead with backend/platform expertise and position React as 'actively developing' skill.",
                "architect_view": "Solid technical depth in core areas (Python 0.95, AWS 0.92). Docker+K8s combo indicates understanding of containerization journey. Gap: No evidence of designing systems at scale - needs to articulate architectural decisions, trade-offs, CAP theorem applications. Should study system design patterns and contribute to architectural discussions/RFC documents. ML confidence is low (0.65) - either invest seriously or remove from resume to avoid false signals.",
                "coach_view": "Excellent growth trajectory potential. High-confidence skills in demanded technologies position candidate well for 2025 market. Creative development strategy: Don't try to be 'full-stack' - instead, own the 'Backend + Infrastructure' niche with React as complementary skill. This differentiation is more valuable. Focus on depth over breadth. Consider T-shaped skill development: Deep in Python/AWS/K8s, broad awareness in frontend/ML. Networking strategy: Contribute to OSS in K8s ecosystem, write technical blogs about cloud-native patterns."
            },
            "action_plan": {
                "quick_wins": [
                    "Deploy existing Python project to AWS with Docker + K8s (document the process)",
                    "Build one complete React CRUD app and deploy to Netlify/Vercel",
                    "Write 2-3 technical blog posts about AWS + K8s learnings",
                    "Complete AWS Certified Solutions Architect Associate (validates existing knowledge)"
                ],
                "3_month_goals": [
                    "Complete 'Real-Time Collaborative Task Manager' portfolio project",
                    "Contribute meaningful PR to established OSS project (K8s ecosystem)",
                    "Achieve conversational TypeScript proficiency",
                    "Build personal brand: Tech blog + GitHub presence"
                ],
                "6_month_goals": [
                    "Transition to Senior Full-Stack or Platform Engineer role",
                    "Complete all 4 portfolio projects",
                    "AWS Certified DevOps Professional certification",
                    "Mentor 1-2 junior developers (builds leadership credibility)"
                ],
                "gap_bridging": "removed"
            }
        }, indent=2)
    # This fallback code is no longer used - kept as reference only


async def run_analysis_async(
    resume_text: str,
    job_ad: str,
    extracted_skills_json: Union[str, Dict[str, Any], None] = None,
    domain_insights_json: Union[str, Dict[str, Any], None] = None,
    confidence_threshold: float = 0.7,
) -> Dict:
    """
    Async version of run_analysis: Run the analysis phase of the generative resume co-writer.

    Args:
        resume_text: Raw resume text
        job_ad: Target job description text
        extracted_skills_json: Dict, JSON string, or file path with extracted skills (optional)
        domain_insights_json: Dict, JSON string, or file path with domain insights (optional)
        confidence_threshold: Minimum confidence threshold for skills

    Returns:
        Dict containing: experiences, aggregate_skills, processed_skills, domain_insights, gap_analysis
    """
    # Load structured skills data
    skills_data = _load_json_payload(extracted_skills_json, "extracted_skills_json")
    domain_insights = _load_json_payload(domain_insights_json, "domain_insights_json")

    # Process skills with confidence filtering
    processed_skills = {}
    domain = domain_insights.get("domain") if domain_insights else None

    if skills_data:
        # Use structured skill processing
        processed_skills = process_structured_skills(skills_data, confidence_threshold, domain)
        all_skills = set(processed_skills["high_confidence_skills"])

        # Build experience results from structured data
        raw_exp_results = skills_data.get("experiences", [])
        exp_results = []
        for exp in raw_exp_results:
            # Ensure experiences have required fields for seniority detection
            exp_data = {
                "title_line": exp.get("title", exp.get("title_line", "")),
                "duration": exp.get("duration", ""),
                "description": exp.get("description", exp.get("snippet", "")),
                "skills": exp.get("skills", []),
                "snippet": exp.get("snippet", "")
            }
            # Ensure title_line is present
            if "title_line" not in exp_data:
                exp_data["title_line"] = exp_data.get("title", "")
            exp_results.append(exp_data)

    else:
        # Fallback to keyword-based extraction
        experiences = parse_experiences(resume_text)
        all_skills = set()
        exp_results = []
        for exp in experiences:
            skills = extrapolate_skills_from_text(exp['raw'])
            # Create structured experience data for seniority detection
            exp_data = {
                "title_line": exp['title_line'],
                "duration": exp['duration'],
                "description": exp['description'],
                "skills": sorted(skills),
                "snippet": exp['raw'][:200]
            }
            exp_results.append(exp_data)
            all_skills.update(skills)

    # Generate role suggestions
    roles = suggest_roles(all_skills)

    # Perform seniority detection
    seniority_detector = SeniorityDetector()
    
    # Prepare experiences for seniority detection
    seniority_experiences = []
    for exp in exp_results:
        seniority_exp = {
            "title": exp.get("title_line", exp.get("title", "")),
            "duration": exp.get("duration", ""),
            "description": exp.get("description", exp.get("snippet", ""))
        }
        seniority_experiences.append(seniority_exp)
    
    seniority_result = seniority_detector.detect_seniority(
        experiences=seniority_experiences,
        skills=all_skills
    )

    # Generate enhanced gap analysis
    try:
        gap = await generate_gap_analysis_async(resume_text, processed_skills, roles, job_ad, domain_insights)
    except Exception as e:
        print(f"⚠️  Gap analysis failed: {str(e)}", flush=True)
        print("🔄 Continuing with basic gap analysis...", flush=True)
        gap = json.dumps({
            "skill_gaps": [],
            "experience_gaps": [],
            "recommendations": ["Unable to perform detailed gap analysis due to API issues"]
        })

    return {
        "experiences": exp_results,
        "aggregate_skills": sorted(all_skills),
        "processed_skills": processed_skills,
        "domain_insights": domain_insights,
        "gap_analysis": gap,
        "role_suggestions": roles,  # Keep for compatibility
        "seniority_analysis": seniority_result
    }


async def run_generation_async(analysis_json: Dict, job_ad: str) -> Dict:
    """
    Async version of run_generation: Run the generation phase of the generative resume co-writer.
    Acts as a creative career coach to generate resume bullet points bridging skill gaps.

    Args:
        analysis_json: Output from run_analysis_async containing gap_analysis and other data
        job_ad: Target job description text

    Returns:
        Dict containing generated_suggestions with gap_bridging and metric_improvements
    """
    gap_analysis = analysis_json.get("gap_analysis", "")
    aggregate_skills = analysis_json.get("aggregate_skills", [])
    processed_skills = analysis_json.get("processed_skills", {})

    # Extract critical gaps - try to parse as JSON first, fallback to text extraction
    critical_gaps = []
    if isinstance(gap_analysis, str):
        try:
            gap_data = json.loads(gap_analysis)
            if isinstance(gap_data, dict) and "gap_analysis" in gap_data:
                critical_gaps = gap_data["gap_analysis"].get("critical_gaps", [])
            elif isinstance(gap_data, dict) and "critical_gaps" in gap_data:
                critical_gaps = gap_data.get("critical_gaps", [])
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: extract common tech gaps from text
            gap_text = gap_analysis.lower()
            potential_gaps = ["react", "kubernetes", "docker", "aws", "typescript", "node.js", "graphql"]
            critical_gaps = [gap for gap in potential_gaps if gap in gap_text][:3]
    elif isinstance(gap_analysis, dict) and "gap_analysis" in gap_analysis:
        critical_gaps = gap_analysis["gap_analysis"].get("critical_gaps", [])

    # If no gaps found, use some defaults based on common tech roles
    if not critical_gaps:
        critical_gaps = ["react", "kubernetes", "typescript"]

    system_prompt = """You are a creative career coach specializing in resume optimization and skill gap bridging. Your expertise is in crafting compelling, specific resume bullet points that demonstrate transferable skills and quantifiable achievements.

Your task is to generate resume bullet points that bridge skill gaps by:
1. Connecting existing skills to missing requirements plausibly
2. Using strong action verbs (Engineered, Architected, Optimized, etc.)
3. Including specific metrics and measurable outcomes
4. Making skill connections that hiring managers will believe

Focus on creating 2-3 bullet points for each major skill gap, plus suggestions for enhancing existing strengths with better metrics."""

    user_prompt = f"""Based on this candidate analysis, generate resume bullet point suggestions:

CANDIDATE SKILLS: {', '.join(aggregate_skills)}
CRITICAL SKILL GAPS: {', '.join(critical_gaps)}
TARGET JOB: {job_ad[:500]}...

For each critical skill gap, generate 2-3 specific resume bullet points that:
- Connect existing skills to the missing skill plausibly
- Use strong action verbs and specific metrics
- Demonstrate the skill through concrete examples

Also suggest improvements to existing skills by adding quantifiable metrics.

Return JSON format:
{{
  "gap_bridging": [
    {{
      "skill_focus": "react",
      "suggestions": [
        "Engineered interactive React components serving 10,000+ users, demonstrating frontend expertise transferable to modern frameworks",
        "Architected component library reducing development time by 40%, showcasing framework-agnostic UI development skills"
      ]
    }}
  ],
  "metric_improvements": [
    {{
      "skill_focus": "python",
      "suggestions": [
        "Developed Python automation scripts processing 1M+ records daily, reducing manual work by 80%",
        "Built machine learning models with 95% accuracy on 500K+ data points using scikit-learn and pandas"
      ]
    }}
  ]
}}"""

    try:
        response = await call_llm_async(
            system_prompt,
            user_prompt,
            temperature=0.9,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
    except Exception as e:
        raise RuntimeError(f"Generation LLM call failed: {e}") from e

    try:
        return ensure_json_dict(response, "Generation")
    except ValueError as e:
        print(f"⚠️  Generation response was not valid JSON: {e}", flush=True)
        print("🔄 Using fallback empty suggestions...", flush=True)
        return {
            "gap_bridging": [],
            "metric_improvements": []
        }


async def run_criticism_async(generated_suggestions: Dict, job_ad: str) -> Dict:
    """Asynchronously refine generated suggestions via an adversarial review."""
    system_prompt = """You are a cynical, experienced hiring manager with 15+ years in tech recruiting. You've seen thousands of resumes and can spot bullshit from a mile away.

Your job is to review resume bullet points and:
1. Remove passive language, clichés, and vague statements
2. Replace weak verbs with strong action verbs (Engineered, Architected, Optimized, etc.)
3. Ensure metrics are specific, believable, and impressive
4. Make suggestions sound like real accomplishments, not generic fluff
5. Add credibility through concrete details and outcomes

Be constructively critical - improve the content while maintaining honesty."""

    user_prompt = f"""Review these generated resume suggestions and make them better. Be brutally honest about what's weak or unbelievable.

TARGET JOB REQUIREMENTS: {job_ad[:300]}...

GENERATED SUGGESTIONS:
{json.dumps(generated_suggestions, indent=2)}

For each suggestion, provide a refined version that:
- Uses stronger, more specific action verbs
- Includes believable, specific metrics
- Removes generic phrases like "improved efficiency"
- Adds concrete details that hiring managers trust
- Sounds like real experience, not wishful thinking

Return JSON format:
{{
  "suggested_experiences": {{
    "bridging_gaps": [
      {{
        "skill_focus": "react",
        "refined_suggestions": [
          "Architected React component library serving 50,000+ monthly active users, reducing development time by 60% through reusable design patterns",
          "Engineered interactive dashboard processing 2M+ data points in real-time, improving user engagement metrics by 40%"
        ]
      }}
    ],
    "metric_improvements": [
      {{
        "skill_focus": "python",
        "refined_suggestions": [
          "Developed Python ETL pipeline processing 10TB+ of financial data daily with 99.9% uptime, supporting 500+ concurrent analysts",
          "Built machine learning recommendation engine with 94% accuracy on 50M+ user interactions, increasing conversion rates by 35%"
        ]
      }}
    ]
  }}
}}"""

    try:
        response = await call_llm_async(
            system_prompt,
            user_prompt,
            temperature=0.3,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
    except Exception as e:
        raise RuntimeError(f"Criticism LLM call failed: {e}") from e
    
    return ensure_json_dict(response, "Criticism")


# JSON Schema for the output
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "experiences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title_line": {"type": "string"},
                    "skills": {"type": "array", "items": {"type": "string"}},
                    "snippet": {"type": "string"}
                },
                "required": ["title_line", "skills", "snippet"]
            }
        },
        "aggregate_skills": {
            "type": "array",
            "items": {"type": "string"}
        },
        "processed_skills": {"type": "object"},
        "domain_insights": {"type": "object"},
        "gap_analysis": {"type": "string"},
        "suggested_experiences": {
            "type": "object",
            "properties": {
                "bridging_gaps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "skill_focus": {"type": "string"},
                            "refined_suggestions": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["skill_focus", "refined_suggestions"]
                    }
                },
                "metric_improvements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "skill_focus": {"type": "string"},
                            "refined_suggestions": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["skill_focus", "refined_suggestions"]
                    }
                }
            },
            "required": ["bridging_gaps", "metric_improvements"]
        },
        "seniority_analysis": {
            "type": "object",
            "properties": {
                "level": {"type": "string"},
                "confidence": {"type": "number"},
                "total_years_experience": {"type": "number"},
                "experience_quality_score": {"type": "number"},
                "leadership_score": {"type": "number"},
                "skill_depth_score": {"type": "number"},
                "achievement_complexity_score": {"type": "number"},
                "reasoning": {"type": "string"},
                "recommendations": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["level", "confidence", "total_years_experience", "reasoning", "recommendations"]
        }
    },
    "required": ["experiences", "aggregate_skills", "processed_skills", "domain_insights", "gap_analysis", "suggested_experiences", "seniority_analysis"]
}

def validate_output_schema(output: Dict[str, Any]) -> bool:
    """
    Validate the output JSON against the schema.
    
    Args:
        output: The output dictionary to validate
        
    Returns:
        True if valid, raises ValidationError if invalid
    """
    try:
        jsonschema.validate(instance=output, schema=OUTPUT_SCHEMA)
        return True
    except jsonschema.ValidationError as e:
        raise ValueError(f"Output schema validation failed: {e.message}") from e


def main():
    import argparse
    p = argparse.ArgumentParser(description="Generative Resume Co-Writer with Adversarial Refinement")
    p.add_argument("--resume", help="Path to resume file")
    p.add_argument("--parsed_resume_text", help="Parsed resume text from Resume_Document_Loader")
    p.add_argument("--extracted_skills_json", help="JSON file with extracted skills from FastSVM_Skill_Title_Extraction or Hermes")
    p.add_argument("--domain_insights_json", help="JSON file with domain insights from Hermes")
    p.add_argument("--target_job_ad", required=True, help="Target job ad text to focus creativity")
    p.add_argument("--confidence_threshold", type=float, default=0.7, help="Minimum confidence threshold for skills (default: 0.7)")
    args = p.parse_args()
    
    # Load resume text
    if args.parsed_resume_text:
        resume_text = args.parsed_resume_text
    elif args.resume:
        resume_text = open(args.resume, encoding="utf-8").read()
    else:
        raise ValueError("Either --resume or --parsed_resume_text must be provided")
    
    # Step 1: Run Analysis
    print("🔍 Step 1: Analyzing resume and job requirements...", flush=True)
    analysis_result = run_analysis(
        resume_text=resume_text,
        job_ad=args.target_job_ad,
        extracted_skills_json=args.extracted_skills_json,
        domain_insights_json=args.domain_insights_json,
        confidence_threshold=args.confidence_threshold
    )
    
    # Step 2: Run Generation (with graceful degradation)
    print("🎨 Step 2: Generating resume improvement suggestions...", flush=True)
    try:
        generation_result = run_generation(
            analysis_json=analysis_result,
            job_ad=args.target_job_ad
        )
    except Exception as e:
        print(f"⚠️  Generation step failed: {str(e)}", flush=True)
        print("🔄 Continuing with analysis results only...", flush=True)
        generation_result = {"gap_bridging": [], "metric_improvements": []}
    
    # Step 3: Run Criticism (with graceful degradation)
    print("🎯 Step 3: Refining suggestions with adversarial review...", flush=True)
    try:
        criticism_result = run_criticism(
            generated_suggestions=generation_result,
            job_ad=args.target_job_ad
        )
    except Exception as e:
        print(f"⚠️  Criticism step failed: {str(e)}", flush=True)
        print("🔄 Continuing with generation results only...", flush=True)
        criticism_result = {
            "suggested_experiences": {
                "bridging_gaps": generation_result.get("gap_bridging", []),
                "metric_improvements": generation_result.get("metric_improvements", [])
            }
        }
    
    # Assemble final output
    output = {
        **analysis_result,  # Includes experiences, aggregate_skills, processed_skills, domain_insights, gap_analysis
        **criticism_result  # Includes suggested_experiences
    }

    # Append run metrics for caller to parse (tokens, cost, failures)
    output["run_metrics"] = RUN_METRICS
    
    # Validate output schema
    validate_output_schema(output)
    
    print(json.dumps(output, indent=2))


# Sync wrapper functions for backward compatibility (CLI usage)
def run_analysis(resume_text: str, job_ad: str, extracted_skills_json: str = None, domain_insights_json: str = None, confidence_threshold: float = 0.7) -> Dict:
    """Sync wrapper for run_analysis_async for CLI compatibility"""
    import asyncio
    return asyncio.run(run_analysis_async(resume_text, job_ad, extracted_skills_json, domain_insights_json, confidence_threshold))
    

def run_generation(analysis_json: Dict, job_ad: str) -> Dict:
    """Sync wrapper for run_generation_async for CLI compatibility"""
    import asyncio
    return asyncio.run(run_generation_async(analysis_json, job_ad))


def run_criticism(generated_suggestions: Dict, job_ad: str) -> Dict:
    """Sync wrapper for run_criticism_async for CLI compatibility"""
    import asyncio
    return asyncio.run(run_criticism_async(generated_suggestions, job_ad))


if __name__ == "__main__":
    main()