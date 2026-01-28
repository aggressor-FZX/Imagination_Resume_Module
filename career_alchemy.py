#!/usr/bin/env python3
"""
Career Alchemy - "Spotify Wrapped" Style Career Identity Generator

Transforms 89 user characteristics into:
- 6 Primary Stats (Hexagon Radar Chart)
- 5 Hero Classes
- Hybrid Archetypes
- Dynamic Shareable Titles
- Social Media Content Templates

"You entered the market as a Senior Software Engineer, 
but your data says you're something more."
"""

import json
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

class PrimaryStat(Enum):
    """The 6 Primary Stats for the Hexagon Graph"""
    INT = "Logic"        # Raw technical horsepower
    DEX = "Precision"    # Versatility and execution speed
    WIS = "Strategy"     # High-level decision making
    CHA = "Influence"    # Leadership and persuasion
    VIT = "Durability"   # Career stamina and reliability
    ARC = "Synthesis"    # Learning ability and domain bridging


class HeroClass(Enum):
    """The 5 Career Hero Classes"""
    ARCHITECT = "The Architect"      # WIS/INT - Big picture + code
    SPECIALIST = "The Specialist"    # INT/DEX - Deep technical expert
    VANGUARD = "The Vanguard"        # CHA/WIS - Leadership + strategy
    ARTISAN = "The Artisan"          # DEX/ARC - Design + execution
    RESEARCHER = "The Researcher"    # ARC/INT - R&D and math problems


# =============================================================================
# CIP CODE EDUCATION MAPPING (Degree → Stat Boosts)
# =============================================================================
# The gold standard for mapping degrees to stats is the CIP (Classification of 
# Instructional Programs) code. O*NET provides direct mapping from CIP to SOC.

# CIP Family Code → (Primary Stat, Secondary Stat, Primary Boost, Secondary Boost)
CIP_STAT_MAPPING = {
    # Computer Science / Engineering (CIP 11, 14, 15)
    "11": {"primary": "INT", "secondary": "DEX", "primary_boost": 15, "secondary_boost": 8, 
           "name": "Computer Science", "justification": "High algorithmic and structural requirements"},
    "14": {"primary": "INT", "secondary": "DEX", "primary_boost": 15, "secondary_boost": 10,
           "name": "Engineering", "justification": "Technical precision and system design"},
    "15": {"primary": "INT", "secondary": "DEX", "primary_boost": 12, "secondary_boost": 8,
           "name": "Engineering Technology", "justification": "Applied technical skills"},
    
    # Business / MBA (CIP 52)
    "52": {"primary": "WIS", "secondary": "CHA", "primary_boost": 12, "secondary_boost": 10,
           "name": "Business/MBA", "justification": "Strategic decision making and networking"},
    
    # Fine Arts / Design (CIP 50)
    "50": {"primary": "ARC", "secondary": "CHA", "primary_boost": 12, "secondary_boost": 8,
           "name": "Fine Arts/Design", "justification": "Aesthetic synthesis and persuasion"},
    
    # Physical Sciences (CIP 40, 41)
    "40": {"primary": "DEX", "secondary": "INT", "primary_boost": 12, "secondary_boost": 10,
           "name": "Physical Sciences", "justification": "Laboratory precision and data calculation"},
    "41": {"primary": "DEX", "secondary": "INT", "primary_boost": 10, "secondary_boost": 8,
           "name": "Science Technologies", "justification": "Technical laboratory work"},
    
    # Mathematics / Statistics (CIP 27)
    "27": {"primary": "INT", "secondary": "ARC", "primary_boost": 18, "secondary_boost": 10,
           "name": "Mathematics/Statistics", "justification": "Pure logical reasoning and pattern recognition"},
    
    # Liberal Arts / Psychology (CIP 42, 45)
    "42": {"primary": "CHA", "secondary": "ARC", "primary_boost": 10, "secondary_boost": 8,
           "name": "Psychology", "justification": "Human behavior understanding"},
    "45": {"primary": "CHA", "secondary": "ARC", "primary_boost": 8, "secondary_boost": 6,
           "name": "Social Sciences", "justification": "Multidisciplinary synthesis"},
    
    # Health / Nursing (CIP 51)
    "51": {"primary": "VIT", "secondary": "DEX", "primary_boost": 12, "secondary_boost": 10,
           "name": "Health/Nursing", "justification": "Physical stamina and precise execution"},
    
    # Philosophy / History (CIP 38, 54)
    "38": {"primary": "ARC", "secondary": "WIS", "primary_boost": 12, "secondary_boost": 10,
           "name": "Philosophy", "justification": "Pure pattern recognition and wisdom"},
    "54": {"primary": "ARC", "secondary": "WIS", "primary_boost": 10, "secondary_boost": 8,
           "name": "History", "justification": "Long-term wisdom and synthesis"},
    
    # Education (CIP 13)
    "13": {"primary": "CHA", "secondary": "ARC", "primary_boost": 10, "secondary_boost": 8,
           "name": "Education", "justification": "Communication and knowledge transfer"},
    
    # Communications / Journalism (CIP 09)
    "09": {"primary": "CHA", "secondary": "ARC", "primary_boost": 12, "secondary_boost": 6,
           "name": "Communications", "justification": "Persuasion and storytelling"},
    
    # Law (CIP 22)
    "22": {"primary": "WIS", "secondary": "CHA", "primary_boost": 14, "secondary_boost": 10,
           "name": "Law", "justification": "Strategic argumentation and influence"},
    
    # Economics (CIP 45.06 mapped to 45)
    "4506": {"primary": "INT", "secondary": "WIS", "primary_boost": 12, "secondary_boost": 10,
             "name": "Economics", "justification": "Quantitative analysis and strategic thinking"},
    
    # Biology / Life Sciences (CIP 26)
    "26": {"primary": "ARC", "secondary": "DEX", "primary_boost": 10, "secondary_boost": 8,
           "name": "Biology/Life Sciences", "justification": "Research and laboratory precision"},
    
    # Agriculture (CIP 01)
    "01": {"primary": "VIT", "secondary": "DEX", "primary_boost": 8, "secondary_boost": 6,
           "name": "Agriculture", "justification": "Durability and practical skills"},
}

# Degree level multipliers (applied to CIP boosts)
DEGREE_LEVEL_MULTIPLIERS = {
    "high_school": 0.3,
    "some_college": 0.5,
    "associate": 0.6,
    "bachelor": 1.0,
    "master": 1.4,
    "mba": 1.5,  # Special case for MBA
    "phd": 1.8,
    "doctorate": 1.8,
    "professional": 1.6,  # JD, MD, etc.
    "default": 0.8
}

# Keyword to CIP mapping for fuzzy matching
# NOTE: Order matters! More specific terms should come FIRST in this list
# Use a list of tuples for guaranteed ordering
DEGREE_KEYWORD_TO_CIP_ORDERED = [
    # Most specific multi-word terms FIRST
    ("computer science", "11"),
    ("data science", "11"),
    ("information technology", "11"),
    ("applied mathematics", "27"),
    ("pure mathematics", "27"),
    ("life science", "26"),
    ("public health", "51"),
    
    # Mathematics/Statistics (specific, must come before generic matches)
    ("mathematics", "27"),
    ("mathematical", "27"),
    ("statistics", "27"),
    ("actuarial", "27"),
    ("math", "27"),
    
    # Computer Science family
    ("computing", "11"),
    ("informatics", "11"),
    ("software", "11"),
    ("cs", "11"),
    
    # Engineering family
    ("mechanical engineering", "14"),
    ("electrical engineering", "14"),
    ("civil engineering", "14"),
    ("chemical engineering", "14"),
    ("aerospace", "14"),
    ("biomedical", "14"),
    ("industrial engineering", "14"),
    ("engineering", "14"),
    
    # Business family
    ("mba", "52"),
    ("business", "52"),
    ("management", "52"),
    ("finance", "52"),
    ("accounting", "52"),
    ("marketing", "52"),
    ("economics", "52"),
    ("administration", "52"),
    
    # Arts family
    ("graphic design", "50"),
    ("visual arts", "50"),
    ("design", "50"),
    ("art", "50"),
    ("music", "50"),
    ("film", "50"),
    ("photography", "50"),
    ("animation", "50"),
    ("ux", "50"),
    ("ui", "50"),
    
    # Sciences
    ("physics", "40"),
    ("chemistry", "40"),
    ("biology", "26"),
    
    # Psychology / Social
    ("psychology", "42"),
    ("sociology", "45"),
    ("anthropology", "45"),
    ("political", "45"),
    
    # Health
    ("nursing", "51"),
    ("health", "51"),
    ("medical", "51"),
    ("pharmacy", "51"),
    ("healthcare", "51"),
    
    # Humanities
    ("philosophy", "38"),
    ("history", "54"),
    ("english", "23"),
    ("literature", "23"),
    ("communications", "09"),
    ("journalism", "09"),
    ("media", "09"),
    
    # Law
    ("law", "22"),
    ("legal", "22"),
    ("jd", "22"),
    
    # Education
    ("education", "13"),
    ("teaching", "13"),
    ("pedagogy", "13"),
]

# Create dict for backward compatibility
DEGREE_KEYWORD_TO_CIP = {k: v for k, v in DEGREE_KEYWORD_TO_CIP_ORDERED}

# =============================================================================
# SKILL-BASED STAT MAPPING
# =============================================================================

# Stat mapping: which characteristics contribute to each stat
STAT_SKILL_MAPPING = {
    PrimaryStat.INT: {
        "skills": ["programming", "python", "java", "javascript", "c++", "sql", 
                   "algorithms", "mathematics", "data structures", "machine learning",
                   "computer science", "software development", "coding", "analytical"],
        "knowledge": ["computers and electronics", "mathematics", "engineering"],
        "weight": 1.0
    },
    PrimaryStat.DEX: {
        "skills": ["docker", "kubernetes", "aws", "azure", "gcp", "devops", 
                   "ci/cd", "testing", "qa", "automation", "frameworks", "react",
                   "node.js", "spring", "django", "git", "agile"],
        "knowledge": ["technology design", "operations analysis"],
        "weight": 1.0
    },
    PrimaryStat.WIS: {
        "skills": ["systems analysis", "architecture", "design patterns", 
                   "strategic planning", "decision making", "problem solving",
                   "critical thinking", "project management", "requirements"],
        "knowledge": ["administration and management", "systems evaluation"],
        "weight": 1.2  # Slightly higher weight for seniority impact
    },
    PrimaryStat.CHA: {
        "skills": ["communication", "leadership", "teamwork", "collaboration",
                   "mentoring", "presentation", "negotiation", "stakeholder",
                   "active listening", "persuasion", "public speaking"],
        "knowledge": ["personnel and human resources", "customer service"],
        "weight": 1.0
    },
    PrimaryStat.VIT: {
        "skills": ["reliability", "consistency", "dedication", "tenure",
                   "project delivery", "deadline management", "quality"],
        "knowledge": ["production and processing"],
        "weight": 0.8  # Experience duration is primary driver
    },
    PrimaryStat.ARC: {
        "skills": ["research", "learning", "innovation", "cross-functional",
                   "adaptability", "continuous learning", "certifications",
                   "emerging technologies", "r&d"],
        "knowledge": ["education and training"],
        "weight": 1.1
    }
}

# Hero Class determination based on peak stats
HERO_CLASS_MAPPING = {
    (PrimaryStat.WIS, PrimaryStat.INT): HeroClass.ARCHITECT,
    (PrimaryStat.INT, PrimaryStat.WIS): HeroClass.ARCHITECT,
    (PrimaryStat.INT, PrimaryStat.DEX): HeroClass.SPECIALIST,
    (PrimaryStat.DEX, PrimaryStat.INT): HeroClass.SPECIALIST,
    (PrimaryStat.CHA, PrimaryStat.WIS): HeroClass.VANGUARD,
    (PrimaryStat.WIS, PrimaryStat.CHA): HeroClass.VANGUARD,
    (PrimaryStat.DEX, PrimaryStat.ARC): HeroClass.ARTISAN,
    (PrimaryStat.ARC, PrimaryStat.DEX): HeroClass.ARTISAN,
    (PrimaryStat.ARC, PrimaryStat.INT): HeroClass.RESEARCHER,
    (PrimaryStat.INT, PrimaryStat.ARC): HeroClass.RESEARCHER,
}

# Hybrid Archetypes when two stats are within 10% of each other
HYBRID_ARCHETYPES = {
    ("INT", "DEX"): {"name": "The Code Virtuoso", "vibe": "You write elegant, performant code with surgical precision."},
    ("INT", "WIS"): {"name": "The Systems Architect", "vibe": "You see both the code and the cathedral it builds."},
    ("INT", "CHA"): {"name": "The Tech Evangelist", "vibe": "You translate complexity into clarity for any audience."},
    ("INT", "VIT"): {"name": "The Reliable Engine", "vibe": "You ship production-grade code, every single time."},
    ("INT", "ARC"): {"name": "The Polymath Coder", "vibe": "You master new paradigms while others are still reading docs."},
    ("DEX", "WIS"): {"name": "The Pragmatic Architect", "vibe": "You balance speed and scalability with rare precision."},
    ("DEX", "CHA"): {"name": "The DevRel Champion", "vibe": "You build tools that people actually want to use."},
    ("DEX", "VIT"): {"name": "The Infrastructure Titan", "vibe": "Your systems run for years without incident."},
    ("DEX", "ARC"): {"name": "The Toolsmith", "vibe": "You forge new frameworks from raw innovation."},
    ("WIS", "CHA"): {"name": "The Strategic Commander", "vibe": "You move the people who move the code."},
    ("WIS", "VIT"): {"name": "The Seasoned Sage", "vibe": "Your experience is measured in battle scars and wins."},
    ("WIS", "ARC"): {"name": "The Systems Alchemist", "vibe": "You turn messy workflows into gold-standard pipelines."},
    ("CHA", "VIT"): {"name": "The Trusted Leader", "vibe": "Teams follow you because you've earned every ounce of respect."},
    ("CHA", "ARC"): {"name": "The Innovation Catalyst", "vibe": "You inspire teams to build what hasn't been imagined yet."},
    ("VIT", "ARC"): {"name": "The Adaptive Veteran", "vibe": "You've reinvented yourself across tech generations."},
    # Special creative hybrids
    ("aesthetics", "math"): {"name": "The Geometric Artist", "vibe": "You use geometry and logic to create visual perfection."},
    ("engineering", "empathy"): {"name": "The Humanist Engineer", "vibe": "You build tools specifically for human accessibility."},
    ("data", "storytelling"): {"name": "The Data Oracle", "vibe": "You turn raw numbers into compelling narratives."},
}

# Seniority Prefix tiers
SENIORITY_TIERS = [
    (1, 3, "Initiate"),
    (4, 7, "Adept"),
    (8, 12, "Master"),
    (13, 99, "Grandmaster")
]

# Leadership Modifiers
LEADERSHIP_MODIFIERS = {
    "executive": "Exarch",
    "high_leadership": "Commander",
    "mentorship": "Vanguard",
    "none": ""
}

# Domain Auras
DOMAIN_AURAS = {
    "fintech": {"name": "Gold Aura", "element": "Profit", "color": "#FFD700"},
    "finance": {"name": "Gold Aura", "element": "Profit", "color": "#FFD700"},
    "healthtech": {"name": "Vitality Aura", "element": "Life", "color": "#00FF7F"},
    "healthcare": {"name": "Vitality Aura", "element": "Life", "color": "#00FF7F"},
    "creative": {"name": "Prism Aura", "element": "Impact", "color": "#FF69B4"},
    "design": {"name": "Prism Aura", "element": "Impact", "color": "#FF69B4"},
    "edtech": {"name": "Wisdom Aura", "element": "Knowledge", "color": "#9370DB"},
    "education": {"name": "Wisdom Aura", "element": "Knowledge", "color": "#9370DB"},
    "security": {"name": "Shadow Aura", "element": "Protection", "color": "#4B0082"},
    "cybersecurity": {"name": "Shadow Aura", "element": "Protection", "color": "#4B0082"},
    "ai": {"name": "Neural Aura", "element": "Intelligence", "color": "#00CED1"},
    "ml": {"name": "Neural Aura", "element": "Intelligence", "color": "#00CED1"},
    "devops": {"name": "Forge Aura", "element": "Infrastructure", "color": "#FF4500"},
    "infrastructure": {"name": "Forge Aura", "element": "Infrastructure", "color": "#FF4500"},
    "data": {"name": "Crystal Aura", "element": "Insight", "color": "#00BFFF"},
    "analytics": {"name": "Crystal Aura", "element": "Insight", "color": "#00BFFF"},
    "default": {"name": "Tech Aura", "element": "Innovation", "color": "#7B68EE"}
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PrimaryStats:
    """The 6 Primary Stats (0-100 scale)"""
    logic: float = 0.0       # INT
    precision: float = 0.0   # DEX
    strategy: float = 0.0    # WIS
    influence: float = 0.0   # CHA
    durability: float = 0.0  # VIT
    synthesis: float = 0.0   # ARC
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "INT": round(self.logic, 1),
            "DEX": round(self.precision, 1),
            "WIS": round(self.strategy, 1),
            "CHA": round(self.influence, 1),
            "VIT": round(self.durability, 1),
            "ARC": round(self.synthesis, 1)
        }
    
    def get_peaks(self, threshold: float = 0.8) -> List[str]:
        """Get stats that are 'peaking' (above threshold)"""
        peaks = []
        for stat, value in self.to_dict().items():
            if value / 100 >= threshold:
                peaks.append(stat)
        return peaks
    
    def get_top_two(self) -> Tuple[str, str]:
        """Get the two highest stats"""
        stats = self.to_dict()
        sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
        return (sorted_stats[0][0], sorted_stats[1][0])


@dataclass
class CareerAlchemyProfile:
    """Complete Career Alchemy Profile"""
    # Identity
    canonical_title: str
    dynamic_title: str
    hero_class: str
    hybrid_archetype: Optional[str]
    hybrid_vibe: Optional[str]
    
    # Stats
    primary_stats: Dict[str, float]
    peak_stats: List[str]
    
    # Seniority
    level: int
    seniority_tier: str
    leadership_modifier: str
    xp_points: int
    
    # Aura & Domain
    domain: str
    aura: Dict[str, str]
    
    # Inventory
    certifications: List[str]
    cert_enchantments: List[Dict[str, Any]]
    
    # Education
    education_entries: List[Dict[str, Any]]
    education_stat_boosts: Dict[str, float]
    highest_degree: str
    degree_field: str
    
    # Rarity
    rarity_score: float
    rarity_percentile: str
    
    # Location context
    location: str
    
    # Timestamps
    generated_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# CAREER ALCHEMY ENGINE
# =============================================================================

class CareerAlchemyEngine:
    """
    Transforms 89 user characteristics into a shareable Career Alchemy profile
    """
    
    def __init__(self):
        self.stat_mapping = STAT_SKILL_MAPPING
        self.hero_class_mapping = HERO_CLASS_MAPPING
        self.hybrid_archetypes = HYBRID_ARCHETYPES
    
    def generate_profile(
        self,
        characteristics: Dict[str, Any],
        location: str = "United States"
    ) -> CareerAlchemyProfile:
        """
        Generate a complete Career Alchemy profile from user characteristics
        
        Args:
            characteristics: The 89 user characteristics from Imaginator
            location: User's location for rarity context
            
        Returns:
            Complete CareerAlchemyProfile ready for social sharing
        """
        # Step 1: Calculate Primary Stats
        primary_stats = self._calculate_primary_stats(characteristics)
        
        # Step 2: Determine Hero Class
        hero_class = self._determine_hero_class(primary_stats)
        
        # Step 3: Calculate Hybrid Archetype
        hybrid_archetype, hybrid_vibe = self._calculate_hybrid_archetype(primary_stats)
        
        # Step 4: Calculate Seniority
        xp_points, seniority_tier, level = self._calculate_seniority(characteristics)
        
        # Step 5: Determine Leadership Modifier
        leadership_modifier = self._determine_leadership_modifier(characteristics)
        
        # Step 6: Generate Dynamic Title
        base_class = hybrid_archetype if hybrid_archetype else hero_class
        dynamic_title = self._generate_dynamic_title(
            seniority_tier=seniority_tier,
            leadership_modifier=leadership_modifier,
            base_class=base_class
        )
        
        # Step 7: Determine Domain Aura
        domain = characteristics.get("domain", "technology").lower()
        aura = self._get_domain_aura(domain)
        
        # Step 8: Process Certifications as Enchantments
        certs = characteristics.get("certifications", [])
        cert_enchantments = self._process_cert_enchantments(certs)
        
        # Step 9: Process Education (CIP-based)
        education = characteristics.get("education", [])
        edu_boosts = self._calculate_education_stat_boosts(education)
        edu_entries, highest_degree, degree_field = self._process_education_entries(education)
        
        # Step 10: Calculate Rarity Score
        rarity_score, rarity_percentile = self._calculate_rarity(primary_stats)
        
        # Build profile
        stats_dict = primary_stats.to_dict()
        
        return CareerAlchemyProfile(
            canonical_title=characteristics.get("canonical_title", 
                           characteristics.get("job_title", "Professional")),
            dynamic_title=dynamic_title,
            hero_class=hero_class,
            hybrid_archetype=hybrid_archetype,
            hybrid_vibe=hybrid_vibe,
            primary_stats=stats_dict,
            peak_stats=primary_stats.get_peaks(0.75),
            level=level,
            seniority_tier=seniority_tier,
            leadership_modifier=leadership_modifier,
            xp_points=xp_points,
            domain=domain,
            aura=aura,
            certifications=[c if isinstance(c, str) else c.get("name", str(c)) for c in certs[:5]],
            cert_enchantments=cert_enchantments,
            education_entries=edu_entries,
            education_stat_boosts=edu_boosts,
            highest_degree=highest_degree,
            degree_field=degree_field,
            rarity_score=rarity_score,
            rarity_percentile=rarity_percentile,
            location=location,
            generated_at=datetime.now().isoformat()
        )
    
    def _calculate_primary_stats(self, characteristics: Dict[str, Any]) -> PrimaryStats:
        """
        Calculate the 6 Primary Stats from characteristics
        
        Aggregates skills, experience, and achievements into normalized 0-100 scores
        """
        # Extract relevant data
        skills = self._extract_skills_list(characteristics)
        experience_years = self._extract_experience_years(characteristics)
        seniority = characteristics.get("seniority", {})
        achievements = characteristics.get("achievements", [])
        certifications = characteristics.get("certifications", [])
        education = characteristics.get("education", [])
        
        # Normalize skills to lowercase for matching
        skills_lower = [s.lower() if isinstance(s, str) else str(s).lower() for s in skills]
        
        stats = PrimaryStats()
        
        # Calculate each stat
        for stat_enum, mapping in self.stat_mapping.items():
            score = 0.0
            matches = 0
            
            # Match skills
            for skill_keyword in mapping["skills"]:
                for user_skill in skills_lower:
                    if skill_keyword in user_skill or user_skill in skill_keyword:
                        matches += 1
                        score += 10
                        break
            
            # Match knowledge areas
            for knowledge in mapping.get("knowledge", []):
                for user_skill in skills_lower:
                    if knowledge in user_skill:
                        score += 5
                        break
            
            # Apply weight
            score *= mapping["weight"]
            
            # Cap at 100 and normalize
            score = min(score, 100)
            
            # Set the appropriate stat
            if stat_enum == PrimaryStat.INT:
                stats.logic = score
            elif stat_enum == PrimaryStat.DEX:
                stats.precision = score
            elif stat_enum == PrimaryStat.WIS:
                stats.strategy = score
            elif stat_enum == PrimaryStat.CHA:
                stats.influence = score
            elif stat_enum == PrimaryStat.VIT:
                stats.durability = score
            elif stat_enum == PrimaryStat.ARC:
                stats.synthesis = score
        
        # Apply experience bonus to VIT and WIS
        exp_bonus = min(experience_years * 3, 30)  # Max 30 points from experience
        stats.durability = min(stats.durability + exp_bonus, 100)
        stats.strategy = min(stats.strategy + exp_bonus * 0.5, 100)
        
        # Apply certification bonus to ARC
        cert_bonus = min(len(certifications) * 8, 24)  # Max 24 points from certs
        stats.synthesis = min(stats.synthesis + cert_bonus, 100)
        
        # Apply CIP-based education stat boosts
        edu_boosts = self._calculate_education_stat_boosts(education)
        stats.logic = min(stats.logic + edu_boosts.get("INT", 0), 100)
        stats.precision = min(stats.precision + edu_boosts.get("DEX", 0), 100)
        stats.strategy = min(stats.strategy + edu_boosts.get("WIS", 0), 100)
        stats.influence = min(stats.influence + edu_boosts.get("CHA", 0), 100)
        stats.durability = min(stats.durability + edu_boosts.get("VIT", 0), 100)
        stats.synthesis = min(stats.synthesis + edu_boosts.get("ARC", 0), 100)
        
        # Apply seniority bonus
        seniority_level = seniority.get("level", "").lower()
        if "senior" in seniority_level or "lead" in seniority_level:
            stats.strategy = min(stats.strategy + 15, 100)
        if "principal" in seniority_level or "staff" in seniority_level:
            stats.strategy = min(stats.strategy + 25, 100)
            stats.influence = min(stats.influence + 10, 100)
        
        # Ensure minimum values for visual appeal
        stats.logic = max(stats.logic, 25)
        stats.precision = max(stats.precision, 20)
        stats.strategy = max(stats.strategy, 15)
        stats.influence = max(stats.influence, 15)
        stats.durability = max(stats.durability, 20)
        stats.synthesis = max(stats.synthesis, 15)
        
        return stats
    
    def _extract_skills_list(self, characteristics: Dict[str, Any]) -> List[str]:
        """Extract all skills from various characteristic fields"""
        skills = []
        
        # Direct skills
        if "skills" in characteristics:
            skill_data = characteristics["skills"]
            if isinstance(skill_data, list):
                for s in skill_data:
                    if isinstance(s, dict):
                        skills.append(s.get("skill", s.get("name", "")))
                    else:
                        skills.append(str(s))
            elif isinstance(skill_data, dict):
                skills.extend(skill_data.get("all", []))
        
        # Aggregate skills
        if "aggregate_skills" in characteristics:
            skills.extend(characteristics["aggregate_skills"])
        
        # Core skills from career progression
        if "core_skills" in characteristics:
            for s in characteristics["core_skills"]:
                if isinstance(s, dict):
                    skills.append(s.get("name", ""))
                else:
                    skills.append(str(s))
        
        # Technologies
        if "technologies" in characteristics:
            skills.extend(characteristics["technologies"])
        
        return [s for s in skills if s]
    
    def _extract_experience_years(self, characteristics: Dict[str, Any]) -> float:
        """Extract total years of experience"""
        # Try direct field
        if "experience_years" in characteristics:
            return float(characteristics["experience_years"])
        
        # Try experience duration
        if "experience_duration" in characteristics:
            return float(characteristics["experience_duration"])
        
        # Calculate from experiences
        experiences = characteristics.get("experiences", [])
        if experiences:
            return min(len(experiences) * 2.5, 20)  # Rough estimate
        
        # Try seniority level inference
        seniority = characteristics.get("seniority", {})
        level = seniority.get("level", "").lower()
        
        if "principal" in level or "staff" in level:
            return 12
        elif "senior" in level or "lead" in level:
            return 7
        elif "mid" in level:
            return 4
        else:
            return 2
    
    def _calculate_education_stat_boosts(
        self, 
        education: List[Any]
    ) -> Dict[str, float]:
        """
        Calculate stat boosts from education using CIP code mapping
        
        Uses the Classification of Instructional Programs (CIP) to determine
        which stats get boosted based on degree field and level.
        
        Args:
            education: List of education entries from characteristics
            
        Returns:
            Dict mapping stat abbreviations to boost values
        """
        stat_boosts = {"INT": 0, "DEX": 0, "WIS": 0, "CHA": 0, "VIT": 0, "ARC": 0}
        
        if not education:
            return stat_boosts
        
        # Process each degree
        for edu in education:
            if isinstance(edu, dict):
                degree_type = edu.get("degree", "").lower()
                field = edu.get("field", edu.get("field_of_study", edu.get("major", ""))).lower()
                institution = edu.get("institution", "").lower()
            elif isinstance(edu, str):
                degree_type = edu.lower()
                field = edu.lower()
                institution = ""
            else:
                continue
            
            # Determine degree level multiplier
            level_multiplier = DEGREE_LEVEL_MULTIPLIERS["default"]
            
            if any(kw in degree_type for kw in ["phd", "ph.d", "doctorate", "doctoral"]):
                level_multiplier = DEGREE_LEVEL_MULTIPLIERS["phd"]
            elif "mba" in degree_type or "mba" in field:
                level_multiplier = DEGREE_LEVEL_MULTIPLIERS["mba"]
            elif any(kw in degree_type for kw in ["master", "m.s.", "m.a.", "ms ", "ma "]):
                level_multiplier = DEGREE_LEVEL_MULTIPLIERS["master"]
            elif any(kw in degree_type for kw in ["jd", "j.d.", "md", "m.d."]):
                level_multiplier = DEGREE_LEVEL_MULTIPLIERS["professional"]
            elif any(kw in degree_type for kw in ["bachelor", "b.s.", "b.a.", "bs ", "ba "]):
                level_multiplier = DEGREE_LEVEL_MULTIPLIERS["bachelor"]
            elif any(kw in degree_type for kw in ["associate", "a.s.", "a.a."]):
                level_multiplier = DEGREE_LEVEL_MULTIPLIERS["associate"]
            elif "some college" in degree_type or "attended" in degree_type:
                level_multiplier = DEGREE_LEVEL_MULTIPLIERS["some_college"]
            elif "high school" in degree_type or "ged" in degree_type:
                level_multiplier = DEGREE_LEVEL_MULTIPLIERS["high_school"]
            
            # Find CIP mapping from degree field
            cip_code = None
            
            # Priority 1: Match field of study specifically (most accurate)
            # Use ordered list for guaranteed priority matching
            for keyword, cip in DEGREE_KEYWORD_TO_CIP_ORDERED:
                if keyword in field:
                    cip_code = cip
                    break
            
            # Priority 2: If no field match, try combined text
            if not cip_code:
                combined_text = f"{degree_type} {field}".lower()
                for keyword, cip in DEGREE_KEYWORD_TO_CIP_ORDERED:
                    if keyword in combined_text:
                        cip_code = cip
                        break
            
            # Apply CIP-based stat boosts
            if cip_code and cip_code in CIP_STAT_MAPPING:
                cip_data = CIP_STAT_MAPPING[cip_code]
                
                # Primary stat boost
                primary_stat = cip_data["primary"]
                primary_boost = cip_data["primary_boost"] * level_multiplier
                stat_boosts[primary_stat] = max(stat_boosts[primary_stat], primary_boost)
                
                # Secondary stat boost
                secondary_stat = cip_data["secondary"]
                secondary_boost = cip_data["secondary_boost"] * level_multiplier
                stat_boosts[secondary_stat] = max(stat_boosts[secondary_stat], secondary_boost)
            else:
                # Default: Any degree gives a small boost to ARC (learning ability)
                base_boost = 8 * level_multiplier
                stat_boosts["ARC"] = max(stat_boosts["ARC"], base_boost)
                stat_boosts["INT"] = max(stat_boosts["INT"], base_boost * 0.5)
        
        # Round all boosts
        return {k: round(v, 1) for k, v in stat_boosts.items()}
    
    def _process_education_entries(
        self, 
        education: List[Any]
    ) -> Tuple[List[Dict[str, Any]], str, str]:
        """
        Process education entries for display and extract highest degree
        
        Returns:
            Tuple of (education_entries, highest_degree, degree_field)
        """
        if not education:
            return [], "Unknown", "General"
        
        entries = []
        highest_degree = "Unknown"
        highest_degree_rank = 0
        degree_field = "General"
        
        degree_ranks = {
            "phd": 6, "doctorate": 6, "ph.d": 6,
            "professional": 5, "jd": 5, "md": 5,
            "master": 4, "mba": 4, "m.s.": 4, "m.a.": 4,
            "bachelor": 3, "b.s.": 3, "b.a.": 3,
            "associate": 2, "a.s.": 2, "a.a.": 2,
            "certificate": 1, "diploma": 1,
            "high school": 0, "ged": 0
        }
        
        for edu in education:
            entry = {}
            
            if isinstance(edu, dict):
                entry["degree"] = edu.get("degree", "Degree")
                entry["field"] = edu.get("field", edu.get("field_of_study", 
                                edu.get("major", "General Studies")))
                entry["institution"] = edu.get("institution", "")
                entry["year"] = edu.get("graduation_date", edu.get("year", ""))
            elif isinstance(edu, str):
                entry["degree"] = edu
                entry["field"] = "General"
                entry["institution"] = ""
                entry["year"] = ""
            else:
                continue
            
            # Find CIP mapping for display (prioritize field over degree name)
            field_lower = entry['field'].lower()
            cip_found = None
            
            # First try to match field specifically (using ordered list for priority)
            for keyword, cip in DEGREE_KEYWORD_TO_CIP_ORDERED:
                if keyword in field_lower and cip in CIP_STAT_MAPPING:
                    cip_found = cip
                    break
            
            # If no field match, try combined
            if not cip_found:
                combined = f"{entry['degree']} {entry['field']}".lower()
                for keyword, cip in DEGREE_KEYWORD_TO_CIP_ORDERED:
                    if keyword in combined and cip in CIP_STAT_MAPPING:
                        cip_found = cip
                        break
            
            if cip_found and cip_found in CIP_STAT_MAPPING:
                entry["cip_family"] = CIP_STAT_MAPPING[cip_found]["name"]
                entry["stat_bonus"] = f"+{CIP_STAT_MAPPING[cip_found]['primary_boost']} {CIP_STAT_MAPPING[cip_found]['primary']}"
            else:
                entry["cip_family"] = "General Studies"
                entry["stat_bonus"] = "+8 ARC"
            
            entries.append(entry)
            
            # Track highest degree
            degree_lower = entry["degree"].lower()
            for degree_kw, rank in degree_ranks.items():
                if degree_kw in degree_lower and rank > highest_degree_rank:
                    highest_degree_rank = rank
                    highest_degree = entry["degree"]
                    degree_field = entry["field"]
                    break
        
        return entries, highest_degree, degree_field
    
    def _determine_hero_class(self, stats: PrimaryStats) -> str:
        """Determine the Hero Class based on top 2 stats"""
        top_two = stats.get_top_two()
        
        # Map stat abbreviations to enums
        stat_enum_map = {
            "INT": PrimaryStat.INT,
            "DEX": PrimaryStat.DEX,
            "WIS": PrimaryStat.WIS,
            "CHA": PrimaryStat.CHA,
            "VIT": PrimaryStat.VIT,
            "ARC": PrimaryStat.ARC
        }
        
        stat1 = stat_enum_map.get(top_two[0])
        stat2 = stat_enum_map.get(top_two[1])
        
        # Check mapping
        hero_class = self.hero_class_mapping.get((stat1, stat2))
        if hero_class:
            return hero_class.value
        
        # Default based on highest stat
        highest = top_two[0]
        defaults = {
            "INT": HeroClass.SPECIALIST.value,
            "DEX": HeroClass.ARTISAN.value,
            "WIS": HeroClass.ARCHITECT.value,
            "CHA": HeroClass.VANGUARD.value,
            "VIT": HeroClass.ARCHITECT.value,
            "ARC": HeroClass.RESEARCHER.value
        }
        return defaults.get(highest, HeroClass.SPECIALIST.value)
    
    def _calculate_hybrid_archetype(
        self, 
        stats: PrimaryStats
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Calculate Hybrid Archetype if two stats are within 10% of each other
        """
        stats_dict = stats.to_dict()
        sorted_stats = sorted(stats_dict.items(), key=lambda x: x[1], reverse=True)
        
        top_stat, top_value = sorted_stats[0]
        second_stat, second_value = sorted_stats[1]
        
        # Check if within 10% of each other
        if top_value > 0:
            difference = (top_value - second_value) / top_value
            if difference <= 0.15:  # 15% threshold for more variety
                # Create key for lookup (alphabetically sorted)
                key = tuple(sorted([top_stat, second_stat]))
                
                archetype = self.hybrid_archetypes.get(key)
                if archetype:
                    return archetype["name"], archetype["vibe"]
        
        return None, None
    
    def _calculate_seniority(
        self, 
        characteristics: Dict[str, Any]
    ) -> Tuple[int, str, int]:
        """
        Calculate XP Points, Seniority Tier, and Level
        
        XP = Experience Years + Job Zone modifier
        
        Returns:
            (xp_points, seniority_tier, level)
        """
        experience_years = self._extract_experience_years(characteristics)
        
        # Get Job Zone from seniority or default
        seniority = characteristics.get("seniority", {})
        job_zone = int(seniority.get("job_zone", 3))
        
        # Calculate XP Points
        xp_points = int(experience_years + job_zone)
        
        # Determine Seniority Tier
        seniority_tier = "Initiate"
        for min_xp, max_xp, tier in SENIORITY_TIERS:
            if min_xp <= xp_points <= max_xp:
                seniority_tier = tier
                break
        
        # Calculate Level (1-100)
        # Formula: (years * 5) + (job_zone * 4) + certification_bonus
        certs = characteristics.get("certifications", [])
        cert_bonus = min(len(certs) * 3, 15)
        
        level = int(min(
            (experience_years * 5) + (job_zone * 4) + cert_bonus,
            100
        ))
        level = max(level, 1)  # Minimum level 1
        
        return xp_points, seniority_tier, level
    
    def _determine_leadership_modifier(
        self, 
        characteristics: Dict[str, Any]
    ) -> str:
        """
        Determine leadership modifier based on titles and skills
        """
        # Check for executive titles
        title = characteristics.get("canonical_title", "").lower()
        job_title = characteristics.get("job_title", "").lower()
        
        executive_keywords = ["cto", "cio", "ceo", "vp", "vice president", 
                            "director", "head of", "chief"]
        
        for keyword in executive_keywords:
            if keyword in title or keyword in job_title:
                return LEADERSHIP_MODIFIERS["executive"]
        
        # Check leadership skills
        skills = self._extract_skills_list(characteristics)
        skills_lower = [s.lower() for s in skills]
        
        leadership_keywords = ["leadership", "management", "mentoring", 
                              "team lead", "resource management", "budgeting",
                              "strategic planning", "executive"]
        
        leadership_count = sum(
            1 for skill in skills_lower 
            if any(kw in skill for kw in leadership_keywords)
        )
        
        if leadership_count >= 3:
            return LEADERSHIP_MODIFIERS["high_leadership"]
        
        # Check for mentorship
        achievements = characteristics.get("achievements", [])
        if any("mentor" in str(a).lower() for a in achievements):
            return LEADERSHIP_MODIFIERS["mentorship"]
        
        return LEADERSHIP_MODIFIERS["none"]
    
    def _generate_dynamic_title(
        self,
        seniority_tier: str,
        leadership_modifier: str,
        base_class: str
    ) -> str:
        """
        Generate the shareable dynamic title
        
        Format: [Leadership Modifier] [Seniority Tier] [Base Class]
        Example: "Exarch Grandmaster Systems Alchemist"
        """
        parts = []
        
        if leadership_modifier:
            parts.append(leadership_modifier)
        
        parts.append(seniority_tier)
        
        # Clean base class (remove "The " prefix if present)
        clean_class = base_class.replace("The ", "")
        parts.append(clean_class)
        
        return " ".join(parts)
    
    def _get_domain_aura(self, domain: str) -> Dict[str, str]:
        """Get the aura based on domain"""
        domain_lower = domain.lower()
        
        for key, aura in DOMAIN_AURAS.items():
            if key in domain_lower:
                return aura
        
        return DOMAIN_AURAS["default"]
    
    def _process_cert_enchantments(
        self, 
        certifications: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Process certifications as stat enchantments (gems)
        """
        enchantments = []
        
        cert_stat_mapping = {
            "aws": {"stat": "WIS", "bonus": 5, "name": "Cloud Mastery"},
            "azure": {"stat": "WIS", "bonus": 5, "name": "Cloud Mastery"},
            "gcp": {"stat": "WIS", "bonus": 5, "name": "Cloud Mastery"},
            "kubernetes": {"stat": "DEX", "bonus": 5, "name": "Container Forge"},
            "docker": {"stat": "DEX", "bonus": 3, "name": "Container Craft"},
            "python": {"stat": "INT", "bonus": 3, "name": "Serpent Code"},
            "java": {"stat": "INT", "bonus": 3, "name": "Legacy Mastery"},
            "security": {"stat": "WIS", "bonus": 4, "name": "Shadow Shield"},
            "ml": {"stat": "INT", "bonus": 5, "name": "Neural Link"},
            "machine learning": {"stat": "INT", "bonus": 5, "name": "Neural Link"},
            "data": {"stat": "INT", "bonus": 4, "name": "Data Sight"},
            "scrum": {"stat": "CHA", "bonus": 3, "name": "Team Sync"},
            "pmp": {"stat": "WIS", "bonus": 4, "name": "Project Oracle"},
            "cisco": {"stat": "DEX", "bonus": 4, "name": "Network Weaver"},
        }
        
        for cert in certifications[:5]:  # Max 5 enchantments
            cert_name = cert if isinstance(cert, str) else cert.get("name", str(cert))
            cert_lower = cert_name.lower()
            
            for keyword, enchantment in cert_stat_mapping.items():
                if keyword in cert_lower:
                    enchantments.append({
                        "certification": cert_name,
                        "enchantment_name": enchantment["name"],
                        "stat_bonus": f"+{enchantment['bonus']} {enchantment['stat']}",
                        "stat": enchantment["stat"],
                        "bonus": enchantment["bonus"]
                    })
                    break
            else:
                # Default enchantment
                enchantments.append({
                    "certification": cert_name,
                    "enchantment_name": "Credential Glow",
                    "stat_bonus": "+2 ARC",
                    "stat": "ARC",
                    "bonus": 2
                })
        
        return enchantments
    
    def _calculate_rarity(
        self, 
        stats: PrimaryStats
    ) -> Tuple[float, str]:
        """
        Calculate rarity score based on stat distribution
        
        Higher rarity = more balanced high stats OR exceptional peak
        """
        stats_dict = stats.to_dict()
        values = list(stats_dict.values())
        
        # Average stat value
        avg = sum(values) / len(values)
        
        # Standard deviation (lower = more balanced)
        variance = sum((x - avg) ** 2 for x in values) / len(values)
        std_dev = math.sqrt(variance)
        
        # Peak value bonus
        peak_bonus = max(values) / 100 * 20
        
        # Balance bonus (lower std_dev = higher bonus)
        balance_bonus = (40 - std_dev) / 40 * 30 if std_dev < 40 else 0
        
        # Base score from average
        base_score = avg * 0.5
        
        rarity_score = min(base_score + peak_bonus + balance_bonus, 100)
        
        # Determine percentile
        if rarity_score >= 95:
            percentile = "Top 1%"
        elif rarity_score >= 90:
            percentile = "Top 2%"
        elif rarity_score >= 80:
            percentile = "Top 5%"
        elif rarity_score >= 70:
            percentile = "Top 10%"
        elif rarity_score >= 60:
            percentile = "Top 20%"
        else:
            percentile = "Top 50%"
        
        return round(rarity_score, 1), percentile


# =============================================================================
# SOCIAL SHARING CONTENT GENERATORS
# =============================================================================

class SocialShareGenerator:
    """
    Generate shareable content for social media platforms
    """
    
    @staticmethod
    def generate_twitter_share(profile: CareerAlchemyProfile) -> Dict[str, str]:
        """Generate Twitter/X share content"""
        stats = profile.primary_stats
        top_stat = max(stats.items(), key=lambda x: x[1])
        
        caption = f"""The industry says I'm a "{profile.canonical_title}."

Cogito Metric just told me I'm actually a {profile.dynamic_title}.

🧪 Rarity Score: {profile.rarity_percentile} in {profile.location}.
📊 Peak Stat: {top_stat[0]} at {top_stat[1]:.0f}
⚔️ Class: {profile.hero_class}

My professional DNA is officially a weird hexagon.
See yours: [Link] #CareerAlchemy #TalentLattice"""
        
        return {
            "platform": "twitter",
            "caption": caption,
            "image_type": "hexagon_radar",
            "hashtags": ["#CareerAlchemy", "#TalentLattice", "#ProfessionalDNA"],
            "character_count": len(caption)
        }
    
    @staticmethod
    def generate_linkedin_share(profile: CareerAlchemyProfile) -> Dict[str, str]:
        """Generate LinkedIn share content"""
        stats = profile.primary_stats
        top_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:3]
        
        caption = f"""🎯 PROFESSIONAL ARCHETYPE UNLOCKED

I just discovered my "Career Alchemy" profile, and the results are fascinating.

📊 My Primary Stats:
"""
        for stat, value in top_stats:
            stat_name = {
                "INT": "Logic", "DEX": "Precision", "WIS": "Strategy",
                "CHA": "Influence", "VIT": "Durability", "ARC": "Synthesis"
            }.get(stat, stat)
            caption += f"• {stat_name}: {value:.0f}/100\n"
        
        caption += f"""
🏆 Official Archetype: {profile.dynamic_title}
🎭 Hero Class: {profile.hero_class}
"""
        
        if profile.hybrid_archetype:
            caption += f"✨ Hybrid Specialty: {profile.hybrid_archetype}\n"
            caption += f'"{profile.hybrid_vibe}"\n'
        
        caption += f"""
🌟 Rarity: {profile.rarity_percentile} ({profile.rarity_score:.0f}% score)

The tool analyzes your skills, experience, and career trajectory to generate a unique "professional signature."

What's your archetype? Try it free: [Link]

#CareerDevelopment #ProfessionalGrowth #CareerAlchemy"""
        
        return {
            "platform": "linkedin",
            "caption": caption,
            "image_type": "character_card",
            "character_count": len(caption)
        }
    
    @staticmethod
    def generate_instagram_share(profile: CareerAlchemyProfile) -> Dict[str, str]:
        """Generate Instagram share content (Stories-optimized)"""
        caption = f"""✨ ARCHETYPE REVEALED ✨

I'm not just a "{profile.canonical_title}"

I'm a {profile.dynamic_title}

Class: {profile.hero_class}
Rarity: {profile.rarity_percentile}
"""
        
        if profile.hybrid_archetype:
            caption += f'\n"{profile.hybrid_vibe}"'
        
        caption += "\n\n🔗 Link in bio to find yours"
        caption += "\n\n#CareerAlchemy #TalentLattice #CareerGoals #TechCareers #ProfessionalGrowth"
        
        return {
            "platform": "instagram",
            "caption": caption,
            "image_type": "story_card",
            "recommended_format": "vertical_9x16",
            "character_count": len(caption)
        }
    
    @staticmethod
    def generate_linkedin_bio_snippet(profile: CareerAlchemyProfile) -> str:
        """Generate a snippet for LinkedIn bio"""
        stats = profile.primary_stats
        top_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:2]
        
        stat_summary = ", ".join([f"{s[0]}: {s[1]:.0f}" for s in top_stats])
        
        return f"Archetype: {profile.dynamic_title} | {stat_summary} | Verified via Cogito Metric"
    
    @staticmethod
    def generate_share_data(profile: CareerAlchemyProfile) -> Dict[str, Any]:
        """Generate complete share data for all platforms"""
        return {
            "twitter": SocialShareGenerator.generate_twitter_share(profile),
            "linkedin": SocialShareGenerator.generate_linkedin_share(profile),
            "instagram": SocialShareGenerator.generate_instagram_share(profile),
            "linkedin_bio": SocialShareGenerator.generate_linkedin_bio_snippet(profile),
            "meta_tags": {
                "og_title": f"{profile.dynamic_title} | Career Alchemy",
                "og_description": f"I'm a {profile.dynamic_title} - {profile.rarity_percentile} rarity. Discover your professional archetype.",
                "og_image_type": "hexagon_radar_dark"
            }
        }


# =============================================================================
# WRAPPED SEQUENCE GENERATOR (Spotify Wrapped Style)
# =============================================================================

class WrappedSequenceGenerator:
    """
    Generate the 5-slide "Spotify Wrapped" style story sequence
    """
    
    @staticmethod
    def generate_sequence(
        profile: CareerAlchemyProfile,
        pivot_data: Optional[Dict[str, Any]] = None,
        salary_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate the complete 5-slide wrapped sequence
        
        Args:
            profile: The Career Alchemy profile
            pivot_data: Optional career pivot analysis data
            salary_data: Optional salary/market data
            
        Returns:
            List of 5 slide definitions
        """
        slides = []
        
        # Slide 1: The Identification
        slides.append({
            "slide_number": 1,
            "title": "THE IDENTIFICATION",
            "graphic_type": "title_card",
            "primary_element": profile.canonical_title,
            "copy": f'You entered the market as a "{profile.canonical_title}," but your data says you\'re something more.',
            "animation": "fade_in_scale",
            "color_scheme": "dark_mode",
            "accent_color": profile.aura.get("color", "#7B68EE")
        })
        
        # Slide 2: The Alchemy of You
        stats = profile.primary_stats
        top_stat = max(stats.items(), key=lambda x: x[1])
        second_stat = sorted(stats.items(), key=lambda x: x[1], reverse=True)[1]
        
        stat_names = {
            "INT": "Logic", "DEX": "Precision", "WIS": "Strategy",
            "CHA": "Influence", "VIT": "Durability", "ARC": "Synthesis"
        }
        
        slides.append({
            "slide_number": 2,
            "title": "THE ALCHEMY OF YOU",
            "graphic_type": "hexagon_radar",
            "stats": stats,
            "copy": f"Your {stat_names.get(top_stat[0], top_stat[0])} ({top_stat[0]}) is in the {profile.rarity_percentile}, but it's your {stat_names.get(second_stat[0], second_stat[0])} ({second_stat[0]}) that makes you rare.",
            "animation": "pulse_glow",
            "color_scheme": "neon_dark",
            "highlight_stats": [top_stat[0], second_stat[0]]
        })
        
        # Slide 3: The Class Reveal
        rarity_fraction = {
            "Top 1%": "1 in 10,000",
            "Top 2%": "1 in 5,000",
            "Top 5%": "1 in 2,000",
            "Top 10%": "1 in 1,000",
            "Top 20%": "1 in 500",
            "Top 50%": "1 in 200"
        }.get(profile.rarity_percentile, "unique")
        
        hybrid_text = ""
        if profile.hybrid_archetype:
            hybrid_text = f' Only {rarity_fraction} users blend these skills like this.'
        
        slides.append({
            "slide_number": 3,
            "title": "THE CLASS REVEAL",
            "graphic_type": "character_avatar",
            "archetype_name": profile.dynamic_title,
            "hero_class": profile.hero_class,
            "hybrid_archetype": profile.hybrid_archetype,
            "copy": f'You are "{profile.dynamic_title}".{hybrid_text}',
            "subtext": profile.hybrid_vibe if profile.hybrid_vibe else f"Class: {profile.hero_class}",
            "animation": "dramatic_reveal",
            "color_scheme": "character_spotlight",
            "aura": profile.aura
        })
        
        # Slide 4: The Bounty (Market Value)
        if salary_data:
            median_salary = salary_data.get("median_salary", 0)
            yoy_growth = salary_data.get("yoy_growth", 0)
            salary_text = f"${median_salary:,}"
            growth_text = f"+{yoy_growth}% increase since last year" if yoy_growth > 0 else "stable market"
        else:
            salary_text = "High Demand"
            growth_text = "Growing market opportunity"
        
        slides.append({
            "slide_number": 4,
            "title": "THE BOUNTY",
            "graphic_type": "bounty_poster",
            "value_display": salary_text,
            "copy": f"Your skill-stack is currently valued at {salary_text}. That's a {growth_text}.",
            "animation": "gold_shimmer",
            "color_scheme": "bounty_gold",
            "location": profile.location
        })
        
        # Slide 5: The Next Quest (Career Pivot)
        if pivot_data and pivot_data.get("pivot_opportunities"):
            best_pivot = pivot_data["pivot_opportunities"][0]
            pivot_title = best_pivot.get("title", "Your Next Role")
            qualification = best_pivot.get("qualification_match", 0)
            skill_gaps = best_pivot.get("skill_gap", {}).get("missing", [])
            gap_skill = skill_gaps[0]["name"] if skill_gaps else "key skills"
            
            copy = f'Your next evolution? {pivot_title}. You\'re {qualification:.0f}% of the way there. Just add "{gap_skill}" to your inventory to unlock new opportunities.'
        else:
            copy = f"Your next quest awaits. Level up your {stat_names.get(second_stat[0], 'skills')} to unlock new career paths."
            pivot_title = "Your Next Evolution"
        
        slides.append({
            "slide_number": 5,
            "title": "THE NEXT QUEST",
            "graphic_type": "portal_gateway",
            "destination": pivot_title,
            "copy": copy,
            "animation": "portal_open",
            "color_scheme": "mystical_purple",
            "call_to_action": "Share Your Archetype"
        })
        
        return slides


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_career_alchemy(
    characteristics: Dict[str, Any],
    location: str = "United States",
    pivot_data: Optional[Dict[str, Any]] = None,
    salary_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main entry point: Generate complete Career Alchemy package
    
    Args:
        characteristics: The 89 user characteristics
        location: User's location
        pivot_data: Optional career pivot analysis
        salary_data: Optional salary/market data
        
    Returns:
        Complete Career Alchemy package with profile, shares, and wrapped sequence
    """
    engine = CareerAlchemyEngine()
    profile = engine.generate_profile(characteristics, location)
    
    # Generate shareable content
    share_data = SocialShareGenerator.generate_share_data(profile)
    
    # Generate wrapped sequence
    wrapped = WrappedSequenceGenerator.generate_sequence(
        profile, pivot_data, salary_data
    )
    
    return {
        "profile": profile.to_dict(),
        "share_content": share_data,
        "wrapped_sequence": wrapped,
        "generated_at": datetime.now().isoformat()
    }


# =============================================================================
# TEST
# =============================================================================

def main():
    """Test the Career Alchemy system"""
    print("=" * 80)
    print("CAREER ALCHEMY - Spotify Wrapped Style Career Identity Generator")
    print("=" * 80)
    
    # Sample characteristics (simulating what Imaginator would have)
    sample_characteristics = {
        "canonical_title": "Senior Software Engineer",
        "job_title": "Senior Software Engineer",
        "domain": "Technology",
        "seniority": {
            "level": "Senior",
            "job_zone": "4",
            "experience_required": "5-10 years"
        },
        "skills": [
            {"skill": "Python", "confidence": 0.95},
            {"skill": "JavaScript", "confidence": 0.88},
            {"skill": "SQL", "confidence": 0.92},
            {"skill": "AWS", "confidence": 0.85},
            {"skill": "Docker", "confidence": 0.82},
            {"skill": "Kubernetes", "confidence": 0.78},
            {"skill": "React", "confidence": 0.80},
            {"skill": "Systems Analysis", "confidence": 0.75},
            {"skill": "Architecture", "confidence": 0.70},
            {"skill": "Problem Solving", "confidence": 0.95},
            {"skill": "Critical Thinking", "confidence": 0.90},
            {"skill": "Communication", "confidence": 0.85},
            {"skill": "Leadership", "confidence": 0.72},
            {"skill": "Mentoring", "confidence": 0.68},
            {"skill": "Agile", "confidence": 0.85}
        ],
        "experience_years": 8,
        "certifications": [
            {"name": "AWS Solutions Architect"},
            {"name": "Kubernetes Administrator"}
        ],
        "education": [
            {"degree": "Bachelor of Science", "field": "Computer Science"}
        ],
        "achievements": [
            "Led migration to microservices architecture",
            "Mentored 5 junior developers",
            "Reduced system latency by 40%"
        ]
    }
    
    # Sample pivot data
    sample_pivot_data = {
        "pivot_opportunities": [
            {
                "title": "Software Architect",
                "qualification_match": 78,
                "skill_gap": {
                    "missing": [
                        {"name": "Enterprise Architecture", "importance": 85}
                    ]
                }
            }
        ]
    }
    
    # Generate Career Alchemy
    print("\n🧪 Generating Career Alchemy profile...")
    result = generate_career_alchemy(
        characteristics=sample_characteristics,
        location="Austin, TX",
        pivot_data=sample_pivot_data
    )
    
    profile = result["profile"]
    
    # Display results
    print("\n" + "=" * 80)
    print("🎭 CAREER ALCHEMY PROFILE")
    print("=" * 80)
    
    print(f"\n📛 Canonical Title: {profile['canonical_title']}")
    print(f"✨ Dynamic Title: {profile['dynamic_title']}")
    print(f"🏆 Hero Class: {profile['hero_class']}")
    
    if profile['hybrid_archetype']:
        print(f"🔮 Hybrid Archetype: {profile['hybrid_archetype']}")
        print(f"   \"{profile['hybrid_vibe']}\"")
    
    print(f"\n📊 Primary Stats (Hexagon):")
    for stat, value in profile['primary_stats'].items():
        bar = "█" * int(value / 5) + "░" * (20 - int(value / 5))
        print(f"   {stat}: {bar} {value:.0f}")
    
    print(f"\n⚡ Peak Stats: {', '.join(profile['peak_stats'])}")
    print(f"🎚️  Level: {profile['level']}")
    print(f"📈 Seniority: {profile['seniority_tier']}")
    if profile['leadership_modifier']:
        print(f"👑 Leadership: {profile['leadership_modifier']}")
    
    print(f"\n🌟 Rarity: {profile['rarity_percentile']} ({profile['rarity_score']})")
    print(f"🌈 Aura: {profile['aura']['name']} ({profile['aura']['element']})")
    
    if profile['cert_enchantments']:
        print(f"\n💎 Enchantments (from Certifications):")
        for ench in profile['cert_enchantments']:
            print(f"   • {ench['enchantment_name']} ({ench['stat_bonus']})")
    
    # Display share content
    print("\n" + "=" * 80)
    print("📱 SHAREABLE CONTENT")
    print("=" * 80)
    
    twitter = result["share_content"]["twitter"]
    print(f"\n🐦 Twitter/X ({twitter['character_count']} chars):")
    print("-" * 40)
    print(twitter["caption"][:300] + "...")
    
    print(f"\n📋 LinkedIn Bio Snippet:")
    print("-" * 40)
    print(result["share_content"]["linkedin_bio"])
    
    # Display wrapped sequence
    print("\n" + "=" * 80)
    print("🎬 WRAPPED SEQUENCE (5 Slides)")
    print("=" * 80)
    
    for slide in result["wrapped_sequence"]:
        print(f"\n📍 Slide {slide['slide_number']}: {slide['title']}")
        print(f"   Type: {slide['graphic_type']}")
        print(f"   Copy: \"{slide['copy'][:100]}...\"")
    
    # Save full output
    output_file = "/home/skystarved/Render_Dockers/career_alchemy_output.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n💾 Full output saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("✅ Career Alchemy Generation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
