#!/usr/bin/env python3
"""
Imaginator Local Agentic Flow

Parses a resume, extrapolates skills, suggests roles, and performs a gap analysis via OpenAI API.
"""
import re
import json
import argparse
import os
from typing import List, Dict, Set

from dotenv import load_dotenv
import openai

# Load environment variables (expects OPENAI_API_KEY)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY. Copy .env.sample to .env and set your key.")
openai.api_key = OPENAI_API_KEY

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
        experiences.append({"raw": b, "title_line": title_line, "body": body})
    return experiences


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


def suggest_roles(skills: Set[str]) -> List[Dict]:
    suggestions = []
    for role, reqs in _ROLE_MAP.items():
        match = skills & reqs
        if match:
            score = round(len(match) / len(reqs), 2)
            suggestions.append({"role": role, "score": score, "matched_skills": sorted(match)})
    return sorted(suggestions, key=lambda x: x['score'], reverse=True)


def generate_gap_analysis(resume_text: str, processed_skills: Dict, roles: List[Dict], target_job_ad: str, domain_insights: Dict = None) -> str:
    """
    Generate creative gap analysis using structured skill data and domain insights
    
    Args:
        resume_text: Raw resume text
        processed_skills: Processed skills with confidence filtering from process_structured_skills
        roles: Role suggestions
        target_job_ad: Target job description
        domain_insights: Domain-specific insights from Hermes (optional)
    """
    
    # Extract skill information
    high_conf_skills = processed_skills.get("high_confidence_skills", [])
    skill_confidences = processed_skills.get("skill_confidences", {})
    skill_categories = processed_skills.get("categories", {})
    
    # Build enhanced prompt with confidence-weighted insights
    prompt_parts = [
        "You are a creative career coach specializing in personalized development strategies.",
        f"Resume Text:\n{resume_text}\n",
        f"High-Confidence Skills (confidence ≥ 0.7): {', '.join(sorted(high_conf_skills))}\n",
        f"Skill Confidence Scores: {json.dumps(skill_confidences, indent=2)}\n"
    ]
    
    # Add skill categories if available
    if skill_categories:
        prompt_parts.append("Skill Categories:")
        for category, skills in skill_categories.items():
            if skills:
                prompt_parts.append(f"  {category.title()}: {', '.join(skills[:5])}")
        prompt_parts.append("")
    
    prompt_parts.extend([
        f"Role Suggestions: {', '.join(r['role'] for r in roles)}\n",
        f"Target Job Ad:\n{target_job_ad}\n"
    ])
    
    # Add domain insights if available
    if domain_insights:
        if "domain" in domain_insights:
            prompt_parts.append(f"Detected Domain: {domain_insights['domain']}\n")
        
        if "insights" in domain_insights:
            insights = domain_insights["insights"]
            if "strengths" in insights:
                prompt_parts.append(f"Resume Strengths: {', '.join(insights['strengths'])}\n")
            if "gaps" in insights:
                prompt_parts.append(f"Identified Gaps: {', '.join(insights['gaps'])}\n")
            if "recommendations" in insights:
                prompt_parts.append(f"AI Recommendations: {', '.join(insights['recommendations'])}\n")
            if "market_alignment" in insights:
                prompt_parts.append(f"Market Alignment Score: {insights['market_alignment']}\n")
    
    prompt_parts.append("""
Analyze the candidate's profile relative to the target job and provide creative, actionable recommendations:

1. **Gap Analysis**: Identify skill gaps considering confidence scores and domain context
2. **Creative Development Path**: Suggest innovative ways to bridge gaps (projects, courses, networking)
3. **Strength Leveraging**: Show how existing high-confidence skills can be applied creatively
4. **Domain-Specific Insights**: Provide industry-tailored recommendations
5. **Confidence-Based Prioritization**: Focus on high-confidence skills for quick wins, low-confidence areas for development

Provide specific, measurable action items with timelines and expected impact.""")
    
    prompt = "\n".join(prompt_parts)
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a creative career strategist who provides personalized, actionable development plans based on structured skill analysis and domain insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,  # Increased for more creativity
            max_tokens=800,   # Increased for more detailed analysis
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Fallback for testing/demo purposes
        return f"""[DEMO MODE - OpenAI Error: {str(e)}]

Based on the structured analysis, here's a creative gap analysis:

🎯 **Key Strengths Identified:**
• High-confidence Python skills (0.95) - Strong foundation for backend development
• AWS expertise (0.92) - Excellent for cloud-native applications  
• JavaScript proficiency (0.89) - Good for full-stack capabilities

🚀 **Creative Development Recommendations:**
1. **Bridge ML Gap**: Leverage existing Python skills to build practical ML projects using scikit-learn
2. **Frontend Enhancement**: Strengthen React skills through targeted projects and tutorials
3. **DevOps Integration**: Combine AWS and Docker knowledge for complete CI/CD pipelines

💡 **Domain-Specific Insights:**
• Tech industry favors full-stack developers with cloud expertise
• ML integration is becoming essential for modern applications
• Focus on production-ready deployments and scalability

📈 **Action Plan:**
• Short-term: Build 2-3 React projects to complement backend skills
• Medium-term: Complete AWS DevOps Professional certification
• Long-term: Develop expertise in MLOps for ML model deployment"""


def main():
    import argparse
    p = argparse.ArgumentParser(description="Enhanced Imaginator with structured skill analysis and domain insights")
    p.add_argument("--resume", help="Path to resume file")
    p.add_argument("--parsed_resume_text", help="Parsed resume text from Resume_Document_Loader")
    p.add_argument("--extracted_skills_json", help="JSON file with extracted skills from FastSVM_Skill_Title_Extraction or Hermes")
    p.add_argument("--domain_insights_json", help="JSON file with domain insights from Hermes")
    p.add_argument("--target_job_ad", required=True, help="Target job ad text to focus creativity")
    p.add_argument("--confidence_threshold", type=float, default=0.7, help="Minimum confidence threshold for skills (default: 0.7)")
    args = p.parse_args()
    
    # Load resume text
    if args.parsed_resume_text:
        text = args.parsed_resume_text
    elif args.resume:
        text = open(args.resume, encoding="utf-8").read()
    else:
        raise ValueError("Either --resume or --parsed_resume_text must be provided")
    
    # Load structured skills data
    skills_data = {}
    domain_insights = {}
    
    if args.extracted_skills_json:
        with open(args.extracted_skills_json, encoding="utf-8") as f:
            skills_data = json.load(f)
    
    if args.domain_insights_json:
        with open(args.domain_insights_json, encoding="utf-8") as f:
            domain_insights = json.load(f)
    
    # Process skills with confidence filtering
    processed_skills = {}
    domain = domain_insights.get("domain") if domain_insights else None
    
    if skills_data:
        # Use structured skill processing
        processed_skills = process_structured_skills(skills_data, args.confidence_threshold, domain)
        all_skills = set(processed_skills["high_confidence_skills"])
        
        # Build experience results from structured data
        exp_results = skills_data.get("experiences", [])
        
    else:
        # Fallback to keyword-based extraction
        experiences = parse_experiences(text)
        all_skills = set()
        exp_results = []
        for exp in experiences:
            skills = extrapolate_skills_from_text(exp['raw'])
            exp_results.append({
                "title_line": exp['title_line'],
                "skills": sorted(skills),
                "snippet": exp['raw'][:200]
            })
            all_skills.update(skills)
    
    # Generate role suggestions
    roles = suggest_roles(all_skills)
    
    # Generate enhanced gap analysis
    gap = generate_gap_analysis(text, processed_skills, roles, args.target_job_ad, domain_insights)
    
    # Prepare output
    output = {
        "experiences": exp_results,
        "aggregate_skills": sorted(all_skills),
        "processed_skills": processed_skills,
        "skills_data": skills_data,
        "domain_insights": domain_insights,
        "role_suggestions": roles,
        "target_job_ad": args.target_job_ad,
        "confidence_threshold": args.confidence_threshold,
        "gap_analysis": gap
    }
    
    print(json.dumps(output, indent=2))
