#!/usr/bin/env python3
"""
Imaginator Configuration Module
Centralizes all model assignments, API keys, and pricing information.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Model Floor Assignments - 4-Stage Pipeline (COST-OPTIMIZED)
# ============================================================================
# Pipeline Flow:
# 1. RESEARCHER (DeepSeek v3.2:online) - Ultra-cheap web search for implied skills
# 2. CREATIVE DRAFTER (Skyfall 36B) - Cost-effective creative resume drafting
# 3. STAR EDITOR (Microsoft Phi-4) - Analytical STAR pattern formatting
# 4. FINAL EDITOR (Gemini 2.0 Flash) - Editorial polish and integration
# 5. GLOBAL FALLBACK (GPT-4o:online) - Last resort with error logging

# Stage 1: Researcher (Heavy Start - Web Search)
MODEL_STAGE_1 = "deepseek/deepseek-v3.2:online"  # Cheapest with web search
MODEL_STAGE_1_BACKUP = "deepseek/deepseek-chat-v3-0324"  # Backup without web search

# Stage 2: Creative Drafter (Lean Middle - Narrative Generation)
MODEL_STAGE_2 = "thedrummer/skyfall-36b-v2"  # Budget creative writing
MODEL_STAGE_2_BACKUP = "thedrummer/cydonia-24b-v4.1"  # Creative backup

# Stage 3: STAR Editor (Lean Middle - Structured Formatting)
MODEL_STAGE_3 = "microsoft/phi-4"  # Analytical precision for STAR bullets

# Stage 4: Polisher (Analytical Finish - Final QC)
MODEL_STAGE_4 = "google/gemini-2.0-flash-exp"  # User preferred final editor
MODEL_STAGE_4_BACKUP = "mistralai/mistral-large-3.1"  # Editorial backup

# Global Fallback (Emergency Only)
MODEL_FALLBACK = "gpt-4o:online"  # Expensive emergency fallback

# Legacy compatibility
MODEL_ANALYTICAL = "microsoft/phi-4"
MODEL_BALANCED = "anthropic/claude-3-haiku"

# ============================================================================
# API Keys
# ============================================================================
OPENROUTER_API_KEYS = [
    os.getenv("OPENROUTER_API_KEY_1"),
    os.getenv("OPENROUTER_API_KEY_2"),
]
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ============================================================================
# App Metadata
# ============================================================================
REFERER = "https://imaginator-resume-cowriter.onrender.com"
TITLE = "Imaginator Resume Co-Writer"

# ============================================================================
# Pricing (USD per 1K tokens)
# ============================================================================
# OpenRouter Pricing
OPENROUTER_PRICE_IN_K = float(os.getenv("OPENROUTER_PRICE_INPUT_PER_1K", "0.0005"))
OPENROUTER_PRICE_OUT_K = float(os.getenv("OPENROUTER_PRICE_OUTPUT_PER_1K", "0.0015"))

# Model-specific pricing
QWEN_PRICE_IN_K = float(os.getenv("QWEN_PRICE_INPUT_PER_1K", "0.00006"))
QWEN_PRICE_OUT_K = float(os.getenv("QWEN_PRICE_OUTPUT_PER_1K", "0.00022"))

DEEPSEEK_PRICE_IN_K = float(os.getenv("DEEPSEEK_PRICE_INPUT_PER_1K", "0.0002"))
DEEPSEEK_PRICE_OUT_K = float(os.getenv("DEEPSEEK_PRICE_OUTPUT_PER_1K", "0.0008"))

ANTHROPIC_PRICE_IN_K = float(os.getenv("ANTHROPIC_PRICE_INPUT_PER_1K", "0.003"))
ANTHROPIC_PRICE_OUT_K = float(os.getenv("ANTHROPIC_PRICE_OUTPUT_PER_1K", "0.015"))

GOOGLE_PRICE_IN_K = float(os.getenv("GOOGLE_PRICE_INPUT_PER_1K", "0.00025"))
GOOGLE_PRICE_OUT_K = float(os.getenv("GOOGLE_PRICE_OUTPUT_PER_1K", "0.0005"))

OPENAI_PRICE_IN_K = float(os.getenv("OPENAI_PRICE_INPUT_PER_1K", "0.0005"))
OPENAI_PRICE_OUT_K = float(os.getenv("OPENAI_PRICE_OUTPUT_PER_1K", "0.0015"))

# Expected cost per resume: ~$0.015 (down from $0.039, 62% cost reduction!)

# ============================================================================
# Cache Settings
# ============================================================================
ANALYSIS_CACHE_TTL_SECONDS = int(os.getenv("ANALYSIS_CACHE_TTL_SECONDS", "600"))

# ============================================================================
# Environment Settings
# ============================================================================
ENABLE_SEMANTIC_GAPS = os.getenv("ENABLE_SEMANTIC_GAPS", "true").lower() == "true"

# Settings object for compatibility with existing code
class Settings:
    """Settings object for backward compatibility"""
    def __init__(self):
        self.ENABLE_LOADER = os.getenv("ENABLE_LOADER", "true").lower() == "true"
        self.ENABLE_FASTSVM = os.getenv("ENABLE_FASTSVM", "true").lower() == "true"
        self.ENABLE_HERMES = os.getenv("ENABLE_HERMES", "true").lower() == "true"
        self.ENABLE_JOB_SEARCH = os.getenv("ENABLE_JOB_SEARCH", "true").lower() == "true"
        
        self.LOADER_BASE_URL = os.getenv("LOADER_BASE_URL", "https://document-reader-service.onrender.com")
        self.FASTSVM_BASE_URL = os.getenv("FASTSVM_BASE_URL", "https://fast-svm-ml-tools-for-skill-and-job.onrender.com")
        self.HERMES_BASE_URL = os.getenv("HERMES_BASE_URL", "https://hermes-resume-extractor.onrender.com")
        self.JOB_SEARCH_BASE_URL = os.getenv("JOB_SEARCH_BASE_URL", "https://job-search-api-h9rl.onrender.com")
        
        self.API_KEY = os.getenv("API_KEY", "05c2765ea794c6e15374f2a63ac35da8e0e665444f6232225a3d4abfe5238c45")
        self.FASTSVM_AUTH_TOKEN = os.getenv("FASTSVM_AUTH_TOKEN", "05ed52ad03b2c2288d631fa5e5d96d4647a840bd298dee8a280ad38c04eb6f1e")
        self.HERMES_AUTH_TOKEN = os.getenv("HERMES_AUTH_TOKEN", "33dfd17ce830dcbf942f7f00d375d23eb8d386b59a487266080de6372978a16a")
        self.JOB_SEARCH_AUTH_TOKEN = os.getenv("JOB_SEARCH_AUTH_TOKEN", "05c2765ea794c6e15374f2a63ac35da8e0e665444f6232225a3d4abfe5238c45")
        
        self.VERBOSE_MICROSERVICE_LOGS = os.getenv("VERBOSE_MICROSERVICE_LOGS", "true").lower() == "true"
        self.VERBOSE_PIPELINE_LOGS = os.getenv("VERBOSE_PIPELINE_LOGS", "true").lower() == "true"
        self.LOG_INCLUDE_RAW_TEXT = os.getenv("LOG_INCLUDE_RAW_TEXT", "false").lower() == "true"
        self.LOG_MAX_TEXT_CHARS = int(os.getenv("LOG_MAX_TEXT_CHARS", "200"))
        
        self.environment = os.getenv("ENVIRONMENT", "production")

settings = Settings()