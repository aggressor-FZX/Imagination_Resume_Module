"""
Pipeline Configuration for Imaginator 3-Stage Pipeline

Contains model registry, pricing, and pipeline configuration.
Based on the Alternate_flow_proposal.md recommendations.
"""

import os
from typing import Dict, List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# MODEL REGISTRY - 3-Stage Pipeline (Cost-Optimized)
# ============================================================================

# Correct OpenRouter Slugs for the 3-Stage Pipeline
OR_SLUG_RESEARCHER = "google/gemini-2.0-flash-001"  # Grounded search capability
OR_SLUG_DRAFTER = "anthropic/claude-3.5-sonnet"     # Best for STAR reasoning
OR_SLUG_STAR_EDITOR = "google/gemini-2.0-flash-001" # High speed, clean Markdown

# Fallback models (in case primary models are unavailable)
FALLBACK_MODELS = {
    "researcher": ["google/gemini-2.0-flash-001", "anthropic/claude-3-haiku"],
    "drafter": ["anthropic/claude-3.5-sonnet", "google/gemini-2.0-pro"],
    "star_editor": ["google/gemini-2.0-flash-001", "anthropic/claude-3-haiku"]
}

# ============================================================================
# PRICING CONFIGURATION (USD per 1K tokens)
# ============================================================================

PRICING = {
    "google/gemini-2.0-flash-001": {"input": 0.00025, "output": 0.0005},
    "google/gemini-2.0-pro": {"input": 0.00125, "output": 0.0025},
    "anthropic/claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
    "anthropic/claude-3-haiku": {"input": 0.00025, "output": 0.00125}
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
    "researcher": 0.1,    # Low for consistent metric extraction
    "drafter": 0.3,       # Moderate for creative but consistent drafting
    "star_editor": 0.1    # Low for consistent formatting
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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