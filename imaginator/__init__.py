#!/usr/bin/env python3
"""
Imaginator Package - Resume Analysis and Enhancement System

A modular 4-stage pipeline for resume processing:
1. Researcher (Heavy Start) - Web-grounded research
2. Drafter (Lean Middle) - Creative narrative writing
3. STAR Editor (Lean Middle) - STAR pattern formatting
4. Polisher (Analytical Finish) - Final QC and alignment
"""

__version__ = "2.0.0"
__author__ = "Cogito Metric"

from .config import settings
from .orchestrator import run_full_funnel_pipeline

__all__ = [
    "settings",
    "run_full_funnel_pipeline",
]