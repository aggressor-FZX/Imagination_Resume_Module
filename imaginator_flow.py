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
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a creative career strategist synthesizing hiring manager, architect, and coach perspectives. Respond ONLY with valid JSON matching the requested structure."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,  # Increased for maximum creativity
            max_tokens=1500,  # Increased for comprehensive JSON output
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Enhanced fallback with structured JSON output
        return json.dumps({
            "_demo_mode_note": f"OpenAI API Error: {str(e)}",
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
                "long_term_vision": "Within 18-24 months: Principal Engineer or Engineering Manager focusing on cloud-native architecture. Recognized expert in Python+K8s+AWS ecosystem through conference talks, blog, and OSS contributions. Leading design of large-scale distributed systems. Optionally pivot to Solutions Architect or DevRel if people-facing work appeals."
            }
        }, indent=2)


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
