"""
Pipeline Configuration for Imaginator 3-Stage Pipeline

Contains model registry, pricing, and pipeline configuration.
Based on the Alternate_flow_proposal.md recommendations.
"""

import os
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# MODEL REGISTRY - 3-Stage Pipeline (Cost-Optimized)
# ============================================================================

# Correct OpenRouter Slugs for the 3-Stage Pipeline
OR_SLUG_RESEARCHER = "perplexity/sonar-pro"       # Grounded search capability (Optimized)
#OR_SLUG_DRAFTER = "google/gemini-3-flash-preview"   # High-quality STAR reasoning
OR_SLUG_STAR_EDITOR = "google/gemini-2.0-flash-001" # High speed, clean Markdown
OR_SLUG_DRAFTER = "anthropic/claude-3-haiku"   # Switched to Haiku to prevent hallucinations

# Fallback models (in case primary models are unavailable)
FALLBACK_MODELS = {
    "researcher": ["perplexity/sonar", "google/gemini-2.0-flash-001"],
    "drafter": ["google/gemini-3-flash-preview", "anthropic/claude-3.5-haiku"],
    "star_editor": ["google/gemini-2.0-flash-001", "anthropic/claude-3-haiku"]
}

# ============================================================================
# PRICING CONFIGURATION (USD per 1K tokens)
# ============================================================================

PRICING = {
    "perplexity/sonar-pro": {"input": 0.003, "output": 0.015}, # Verified via OpenRouter API
    "perplexity/sonar": {"input": 0.001, "output": 0.001},
    "google/gemini-2.0-flash-001": {"input": 0.0001, "output": 0.0004}, # Corrected: was 2.5x too high
    "google/gemini-2.0-pro": {"input": 0.00125, "output": 0.0025}, # Needs verification
    "google/gemini-3-flash-preview": {"input": 0.0005, "output": 0.003}, # New high-quality drafter
    "xiaomi/mimo-v2-flash": {"input": 0.00015, "output": 0.0006}, # DEPRECATED: Hallucinates too much
    "anthropic/claude-3-haiku": {"input": 0.00025, "output": 0.00125}, # Fallback: Reliable & cheap
    "anthropic/claude-3.5-haiku": {"input": 0.0008, "output": 0.004}, # New high-quality fallback
    "anthropic/claude-3.5-sonnet": {"input": 0.006, "output": 0.030}, # DEPRECATED: 2x higher than estimated
    "deepseek/deepseek-chat-v3.1": {"input": 0.00015, "output": 0.00075} # Added: cheap alternative
}

# ============================================================================
# HALLUCINATION GUARD CONFIGURATION
# ============================================================================

FORBIDDEN_PHRASES = {
    "abc corp", "abc tech", "acme inc", "xyz corp", "example company",
    "sample corp", "test inc", "generic company", "john doe", "jane doe",
    "retail sales associate at", "tech startup x", "company x", "company y",
    "startup abc", "acme corp", "example inc", "test company"
}

# ============================================================================
# SENIORITY CALIBRATION CONFIGURATION
# ============================================================================

SENIORITY_CONFIG = {
    "junior": {
        "keywords": ["entry", "junior", "0-2 years", "1-3 years", "associate"],
        "verbs": ["Assisted in", "Contributed to", "Supported", "Collaborated on", "Built", "Helped"],
        "tone": "Collaborative, learning-focused"
    },
    "mid": {
        "keywords": ["mid", "3-5 years", "5+ years", "experienced", "senior"],
        "verbs": ["Led", "Designed", "Implemented", "Optimized", "Deployed", "Engineered"],
        "tone": "Results-driven, independent"
    },
    "senior": {
        "keywords": ["senior", "lead", "principal", "staff", "10+ years", "architect"],
        "verbs": ["Architected", "Strategized", "Pioneered", "Directed", "Orchestrated", "Established"],
        "tone": "Strategic, leadership-focused"
    }
}

# ============================================================================
# API KEYS AND ENVIRONMENT
# ============================================================================

OPENROUTER_API_KEYS = [
    os.getenv("OPENROUTER_API_KEY_1"),
    os.getenv("OPENROUTER_API_KEY_2"),
]

OPENROUTER_APP_REFERER = "https://imaginator-resume-cowriter.onrender.com"
OPENROUTER_APP_TITLE = "Imaginator Resume Co-Writer"

# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================

# Timeout settings (seconds)
TIMEOUTS = {
    "researcher": 30,
    "drafter": 45,
    "star_editor": 30,
    "total": 120
}

# Temperature settings
TEMPERATURES = {
    "researcher": 0.1,
    "drafter": 0.4,
    "star_editor": 0.1
}

CREATIVITY_TEMPERATURES = {
    "conservative": {"drafter": 0.2},
    "balanced": {"drafter": 0.4},
    "bold": {"drafter": 0.65}
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_temperatures(creativity_mode: Optional[str] = None) -> Dict[str, float]:
    mode = (creativity_mode or "balanced").lower()
    preset = CREATIVITY_TEMPERATURES.get(mode, CREATIVITY_TEMPERATURES["balanced"])
    drafter_temp = preset.get("drafter", TEMPERATURES["drafter"])
    drafter_temp = min(drafter_temp, 0.7)
    return {
        "researcher": TEMPERATURES["researcher"],
        "drafter": drafter_temp,
        "star_editor": TEMPERATURES["star_editor"],
    }

def get_tone_instruction(creativity_mode: Optional[str] = None) -> str:
    mode = (creativity_mode or "balanced").lower()
    if mode == "bold":
        return "Use dynamic, evocative language. Be bold. Use varied sentence structures to capture attention."
    if mode == "conservative":
        return "Use strict, formal, and concise language. Prioritize factual accuracy over flair."
    return "Maintain a standard, professional corporate tone."

def get_seniority_config(job_ad: str) -> Tuple[str, Dict]:
    """
    Determine seniority level from job ad and return configuration.
    
    Args:
        job_ad: Job description text
        
    Returns:
        Tuple of (level, config_dict)
    """
    ad_lower = job_ad.lower()
    
    for level, config in SENIORITY_CONFIG.items():
        for keyword in config["keywords"]:
            if keyword in ad_lower:
                return level, config
    
    # Default to mid-level if no keywords found
    return "mid", SENIORITY_CONFIG["mid"]

def contains_hallucination(text: str) -> bool:
    """
    Detect if text contains placeholder/invented content.
    
    Args:
        text: Text to check
        
    Returns:
        True if hallucination detected
    """
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in FORBIDDEN_PHRASES)

def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Estimate cost for a given model and token usage.
    
    Args:
        model: Model identifier
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        
    Returns:
        Estimated cost in USD
    """
    if model not in PRICING:
        # Default pricing if model not in registry
        return (prompt_tokens / 1000.0) * 0.0005 + (completion_tokens / 1000.0) * 0.0015
    
    pricing = PRICING[model]
    return (prompt_tokens / 1000.0) * pricing["input"] + (completion_tokens / 1000.0) * pricing["output"]
