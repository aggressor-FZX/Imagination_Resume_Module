"""
Imaginator 3-Stage Pipeline Modules

This package contains the three core stages of the new Imaginator pipeline:
1. Researcher - Extracts metrics and domain vocabulary from job ads
2. Drafter - Creates STAR-formatted bullets with seniority calibration
3. StarEditor - Polishes and formats final resume with hallucination guard
"""

from .researcher import Researcher
from .drafter import Drafter
from .star_editor import StarEditor

__all__ = ["Researcher", "Drafter", "StarEditor"]