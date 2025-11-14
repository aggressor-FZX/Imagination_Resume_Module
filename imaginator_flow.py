#!/usr/bin/env python3
"""
Imaginator Local Agentic Flow

Parses a resume, extrapolates skills, suggests roles, and performs a gap analysis via OpenAI API.
"""
import re
import json
import argparse
import os
from typing import List, Dict, Set, Any, Optional

import aiohttp
import asyncio
import google.generativeai as genai
# from deepseek import DeepSeekAPI  # Commented out - module not available

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
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
# For async operations, initialize a separate async client
openrouter_async_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": "https://imaginator-resume-cowriter.onrender.com",
        "X-Title": "Imaginator Resume Co-Writer"
    }
) if os.getenv("OPENROUTER_API_KEY") else None# Initialize async clients
openai_async_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
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


def call_llm(
    system_prompt: str, 
    user_prompt: str, 
    temperature: float = 0.9, 
    max_tokens: int = 1500, 
    max_retries: int = 3,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None
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
        response = openrouter_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=60.0
        )
        text = response.choices[0].message.content.strip()

        # Attempt to track usage if available
        try:
            u = getattr(response, "usage", None)
            prompt_t = int(getattr(u, "prompt_tokens", 0) or 0)
            completion_t = int(getattr(u, "completion_tokens", 0) or 0)
        except Exception:
            prompt_t = completion_t = 0

        # Estimate cost using OpenRouter pricing fields
        if model == "meta-llama/llama-3.1-70b-instruct":
            input_price = 0.0000004
            output_price = 0.0000004
        elif model == "mistralai/mistral-large":
            input_price = 0.000002
            output_price = 0.000006
        else:
            input_price = 0.00000025
            output_price = 0.00000125

        cost = (prompt_t / 1000.0) * input_price + (completion_t / 1000.0) * output_price
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
    deepseek_api_key: Optional[str] = None
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
                maybe = openrouter_async_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=60.0
                )
                response = await maybe if inspect.isawaitable(maybe) else maybe
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
                
                # Calculate cost based on actual model used
                if "qwen" in model:
                    input_price = QWEN_PRICE_IN_K / 1000.0  # Convert from per 1K to per token
                    output_price = QWEN_PRICE_OUT_K / 1000.0
                elif "deepseek" in model:
                    input_price = DEEPSEEK_PRICE_IN_K / 1000.0  # Convert from per 1K to per token
                    output_price = DEEPSEEK_PRICE_OUT_K / 1000.0
                elif "claude" in model:
                    input_price = ANTHROPIC_PRICE_IN_K / 1000.0  # Convert from per 1K to per token
                    output_price = ANTHROPIC_PRICE_OUT_K / 1000.0
                else:  # fallback to OpenRouter default
                    input_price = OPENROUTER_PRICE_IN_K / 1000.0
                    output_price = OPENROUTER_PRICE_OUT_K / 1000.0
                
                cost = (prompt_t / 1000.0) * input_price + (completion_t / 1000.0) * output_price
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
                    model="deepseek-chat",
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
    return await call_llm_async(system_prompt, prompt, temperature=0.9, max_tokens=1500)


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
    return await call_llm_async(system_prompt, prompt, temperature=0.9, max_tokens=1500)


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
    return await call_llm_async(system_prompt, prompt, temperature=0.9, max_tokens=1500)


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