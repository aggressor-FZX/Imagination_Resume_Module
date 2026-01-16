#!/usr/bin/env python3
import asyncio
import json
import sys
from pathlib import Path
sys.path.insert(0, '.')  # Allow local imports

# Mock data - Sample Data Scientist resume
RESUME_TEXT = '''Michael Chen
New York, NY | michael.chen@email.com | (555) 234-5678 | linkedin.com/in/michaelchen

PROFESSIONAL SUMMARY
Data Scientist with 4 years of experience in statistical analysis, predictive modeling, and 
data visualization. Expertise in Python, SQL, and machine learning algorithms. Strong 
business acumen with proven ability to translate data insights into actionable strategies.

WORK EXPERIENCE
Data Scientist at FinTech Analytics (2021-Present)
• Developed churn prediction model with 87% accuracy, saving $2M annually in customer retention
• Built real-time analytics dashboard using Tableau and Python, used by C-suite executives
• Implemented A/B testing framework that increased conversion rates by 12%
• Automated ETL pipelines reducing manual reporting time by 20 hours/week

Data Analyst at Retail Insights (2019-2021)
• Analyzed customer behavior data for 1M+ users, identified key purchasing patterns
• Created forecasting models for inventory management, reducing waste by 18%
• Collaborated with marketing team to optimize campaign targeting

EDUCATION
Master of Science in Data Science, Columbia University, 2019
Bachelor of Science in Statistics, NYU, 2017

SKILLS
Python, R, SQL, Pandas, NumPy, Scikit-learn, TensorFlow, Tableau, PowerBI, PostgreSQL, 
MySQL, Git, Jupyter, Statistical Analysis, Hypothesis Testing, Regression, Classification, 
Time Series Analysis, A/B Testing, Data Visualization

CERTIFICATIONS
Google Data Analytics Professional Certificate
Tableau Desktop Specialist
'''

JOB_AD = '''Senior Full-Stack Developer - Tech Startup

We're looking for a Senior Full-Stack Developer to join our growing engineering team. You'll be responsible for designing, developing, and maintaining scalable web applications that serve millions of users.

Key Requirements:
- 5+ years of experience in full-stack development
- Strong proficiency in Python and JavaScript
- Experience with React, Node.js, and modern frontend frameworks
- Cloud platforms (AWS, GCP, or Azure) - deployment and scaling
- Container orchestration (Docker, Kubernetes)
- Database design and optimization (SQL and NoSQL)
- CI/CD pipeline implementation
- Experience with microservices architecture
- Knowledge of machine learning integration is a plus

What You'll Do:
- Design and implement scalable backend services
- Build responsive, user-friendly frontend interfaces
- Deploy and maintain applications on cloud infrastructure
- Collaborate with data scientists to integrate ML models
- Mentor junior developers and contribute to technical decisions
- Participate in code reviews and architectural discussions

Benefits:
- Competitive salary and equity
- Flexible work arrangements
- Health, dental, and vision insurance
- Professional development budget
- Collaborative and innovative work environment'''

HERMES_MOCK = {
    "structured_skills": ["Python", "Machine Learning"],
    "job_titles": ["Data Scientist"],
    "experiences": [{"company": "Tech Corp", "role": "Data Scientist", "description": "Built ML models"}]
}

SVM_MOCK = {
    "skills": [{"skill": "Python", "confidence": 0.95}],
    "seniority_levels": ["mid"]
}

# Use REAL LLM client adapter
from llm_client_adapter import LLMClientAdapter
import os

# Load API keys from environment
from dotenv import load_dotenv
load_dotenv('../.env')  # Load from parent directory

OPENROUTER_API_KEYS = [
    os.getenv("OPENROUTER_API_KEY"),
    os.getenv("OPENROUTER_API_KEY_1"),
    os.getenv("OPENROUTER_API_KEY_2")
]
OPENROUTER_API_KEYS = [k for k in OPENROUTER_API_KEYS if k]

if not OPENROUTER_API_KEYS:
    print("WARNING: No OpenRouter API keys found. Set OPENROUTER_API_KEY in .env")
    print("Falling back to mock client...")
    
    class MockLLMClient:
        async def call_llm_async(self, **kwargs):
            if 'researcher' in str(kwargs):
                return json.dumps({
                    "implied_metrics": ["40% accuracy improvement", "10k predictions/day"],
                    "domain_vocab": ["PyTorch", "Scikit-learn"],
                    "insider_tips": "Emphasize production ML."
                })
            elif 'drafter' in str(kwargs):
                return json.dumps({
                    "rewritten_experiences": [{"company": "Tech Corp", "role": "Data Scientist", "bullets": ["Led ML project achieving 40% accuracy boost."], "metrics_used": ["40%"]}],
                    "seniority_applied": "mid",
                    "quantification_score": 0.95
                })
            elif 'star_editor' in str(kwargs):
                return json.dumps({
                    "final_markdown": "## Experience\\n**Data Scientist** at *Tech Corp*\\n- Led ML project achieving 40% accuracy boost.",
                    "final_plain_text": "Experience Data Scientist at Tech Corp Led ML project achieving 40% accuracy boost.",
                    "editorial_notes": "Polished."
                })
            return '{}'
    
    RealLLMClient = MockLLMClient
else:
    print(f"✅ Found {len(OPENROUTER_API_KEYS)} OpenRouter API key(s)")
    RealLLMClient = LLMClientAdapter

async def test_new_orchestrator():
    from orchestrator import PipelineOrchestrator
    try:
        from imaginator_flow import parse_experiences
    except ImportError:
        def parse_experiences(text):
            return [{"company": "FinTech Analytics", "role": "Data Scientist", "description": text[:200]}]
    
    experiences = parse_experiences(RESUME_TEXT)
    
    # Use REAL LLM client
    if OPENROUTER_API_KEYS:
        client = RealLLMClient(api_key=OPENROUTER_API_KEYS[0])
    else:
        client = RealLLMClient()
    
    orch = PipelineOrchestrator(client)
    
    result = await orch.run_pipeline(RESUME_TEXT, JOB_AD, experiences, openrouter_api_keys=OPENROUTER_API_KEYS)
    print("=== NEW 3-STAGE OUTPUT ===")
    print(json.dumps(result['final_output'], indent=2))
    print(f"\nSummary: {orch.get_pipeline_summary(result)}")
    return result

async def test_old_orchestrator():
    sys.path.insert(0, '.')
    try:
        from orchestrator_old import run_full_funnel_pipeline
        print("OLD orchestrator found")
        result = await run_full_funnel_pipeline(RESUME_TEXT, JOB_AD, HERMES_MOCK, SVM_MOCK)
        print("=== OLD 4-STAGE OUTPUT ===")
        print(json.dumps(result, indent=2))
        return result
    except ImportError as e:
        print(f"OLD orchestrator import failed: {e}")
        return {"old": "SKIPPED - orchestrator_old.py not found"}

async def main():
    new_result = await test_new_orchestrator()
    old_result = await test_old_orchestrator()
    
    with open('comparison_results.json', 'w') as f:
        json.dump({'new': new_result, 'old': old_result}, f, indent=2)
    
    print("Results saved to comparison_results.json")

if __name__ == "__main__":
    asyncio.run(main())
