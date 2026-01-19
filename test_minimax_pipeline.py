#!/usr/bin/env python3
"""
MiniMax M2.1 Pipeline Test - Real Resume/Job Ad with Hermes/FastSVM inputs
"""

import json
import time
import requests
from datetime import datetime

# Pipeline inputs (simulated from other services)
RESUME_DATA = """Michael Chen
New York, NY | michael.chen@email.com | (555) 234-5678

PROFESSIONAL SUMMARY
Data Scientist with 4 years of experience in statistical analysis, predictive modeling, and 
data visualization. Expertise in Python, SQL, and machine learning algorithms.

WORK EXPERIENCE
Data Scientist at FinTech Analytics (2021-Present)
• Developed churn prediction model with 87% accuracy, saving $2M annually
• Built real-time analytics dashboard using Tableau and Python
• Implemented A/B testing framework that increased conversion rates by 12%
• Automated ETL pipelines reducing manual reporting time by 20 hours/week

Data Analyst at Retail Insights (2019-2021)
• Analyzed customer behavior data for 1M+ users, identified key purchasing patterns
• Created forecasting models for inventory management, reducing waste by 18%

EDUCATION
Master of Science in Data Science, Columbia University, 2019

SKILLS
Python, SQL, Machine Learning, Tableau, TensorFlow, Statistical Analysis
"""

JOB_AD = """Armada AI - Senior ML Engineer
We're looking for a Senior ML Engineer to lead our AI infrastructure team.

Requirements:
- 5+ years of experience with machine learning and deep learning
- Strong background in Python, PyTorch, and TensorFlow
- Experience with distributed systems and cloud infrastructure (AWS/GCP)
- Leadership experience mentoring junior engineers
- Experience with production ML systems and MLOps

Responsibilities:
- Design and implement scalable ML pipelines
- Lead technical architecture decisions
- Mentor junior engineers
"""

# Simulated Hermes output
HERMES_OUTPUT = {
    "extracted_skills": [
        {"skill": "Python", "confidence": 0.95, "source": "resume"},
        {"skill": "Machine Learning", "confidence": 0.92, "source": "resume"},
        {"skill": "Statistical Analysis", "confidence": 0.88, "source": "resume"},
        {"skill": "Data Visualization", "confidence": 0.85, "source": "resume"},
        {"skill": "SQL", "confidence": 0.90, "source": "resume"},
        {"skill": "TensorFlow", "confidence": 0.82, "source": "resume"},
    ],
    "domain_insights": {
        "primary_domain": "Data Science / ML",
        "trending_skills": ["LLMs", "MLOps", "Distributed Systems"]
    }
}

# Simulated FastSVM output
FASTSVM_OUTPUT = {
    "extracted_job_titles": [
        {"title": "Data Scientist", "confidence": 0.92},
        {"title": "ML Engineer", "confidence": 0.88},
    ],
    "extracted_skills": [
        {"skill": "Python", "confidence": 0.95},
        {"skill": "Machine Learning", "confidence": 0.92},
        {"skill": "SQL", "confidence": 0.90},
    ]
}

def create_drafter_prompt(resume, job_ad, hermes, fastsvm):
    """Create drafter prompt with pipeline inputs."""
    
    system_prompt = """You are an expert Resume Writer specializing in STAR format bullets.
Rewrite the user's experiences into 3-5 professional STAR bullets.

CRITICAL RULES:
1. Use ONLY the user's actual company names and roles
2. DO NOT hallucinate technologies the user didn't mention
3. Every bullet MUST include at least one quantifiable metric (%, $, time, scale)
4. Use strong action verbs
5. Focus on impact and business value
6. Return VALID JSON only

Output JSON Schema:
{
  "rewritten_experiences": [
    {
      "company": "Company Name",
      "role": "Job Title",
      "bullets": ["Bullet 1 with metric", "Bullet 2 with metric"]
    }
  ],
  "seniority_applied": "senior",
  "quantification_score": 0.95
}
"""
    
    hermes_skills = [s["skill"] for s in hermes["extracted_skills"][:5]]
    fastsvm_titles = [t["title"] for t in fastsvm["extracted_job_titles"][:3]]
    
    user_prompt = f"""
RESUME:
{resume}

TARGET JOB:
{job_ad}

HERMES SKILLS: {', '.join(hermes_skills)}
FASTSVM TITLES: {', '.join(fastsvm_titles)}
MARKET DOMAIN: {hermes['domain_insights']['primary_domain']}

TASK: Rewrite experiences for Senior ML Engineer role.
Focus on: Leadership, ML infrastructure, production systems, quantifiable impact.
"""
    
    return system_prompt, user_prompt

def test_minimax():
    """Test MiniMax M2.1 with pipeline inputs."""
    
    import os
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    system_prompt, user_prompt = create_drafter_prompt(
        RESUME_DATA, JOB_AD, HERMES_OUTPUT, FASTSVM_OUTPUT
    )
    
    print("=" * 80)
    print("MiniMax M2.1 Pipeline Test")
    print("=" * 80)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"System Prompt: {len(system_prompt)} chars")
    print(f"User Prompt: {len(user_prompt)} chars")
    print()
    
    start_time = time.time()
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "model": "minimax/minimax-m2.1",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 2000
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://imaginator-resume-cowriter.onrender.com",
                "X-Title": "Imaginator Pipeline Test"
            },
            timeout=120
        )
        
        response_time = time.time() - start_time
        
        print(f"Response Time: {response_time:.2f}s")
        print(f"Status: {response.status_code}")
        print()
        
        if response.status_code == 200:
            result = response.json()
            usage = result.get("usage", {})
            content = result["choices"][0]["message"]["content"]
            
            print("=" * 80)
            print("RAW MODEL OUTPUT:")
            print("=" * 80)
            print(content[:2000])
            print("..." if len(content) > 2000 else "")
            print()
            
            # Save full output
            output_file = f"minimax_output_{int(time.time())}.json"
            with open(output_file, "w") as f:
                json.dump({
                    "model": "minimax/minimax-m2.1",
                    "timestamp": datetime.now().isoformat(),
                    "response_time": response_time,
                    "tokens_in": usage.get("prompt_tokens", 0),
                    "tokens_out": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "cost": usage.get("cost", 0),
                    "raw_response": content,
                    "pipeline_inputs": {
                        "hermes_skills": HERMES_OUTPUT["extracted_skills"],
                        "fastsvm_titles": FASTSVM_OUTPUT["extracted_job_titles"],
                        "domain": HERMES_OUTPUT["domain_insights"]["primary_domain"]
                    }
                }, f, indent=2)
            
            print(f"✅ Full output saved to: {output_file}")
            print()
            print("=" * 80)
            print("TOKEN USAGE:")
            print("=" * 80)
            print(f"  Input Tokens: {usage.get('prompt_tokens', 0)}")
            print(f"  Output Tokens: {usage.get('completion_tokens', 0)}")
            print(f"  Total Tokens: {usage.get('total_tokens', 0)}")
            print(f"  Cost: ${usage.get('cost', 0):.6f}")
            
            # Try to parse JSON
            try:
                json_content = content
                if json_content.startswith("```json"):
                    json_content = json_content[7:]
                if json_content.startswith("```"):
                    json_content = json_content[3:]
                if json_content.endswith("```"):
                    json_content = json_content[:-3]
                
                start_idx = json_content.find('{')
                end_idx = json_content.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_content = json_content[start_idx:end_idx+1]
                
                parsed = json.loads(json_content.strip())
                
                print()
                print("=" * 80)
                print("PARSED RESUME OUTPUT:")
                print("=" * 80)
                for exp in parsed.get("rewritten_experiences", []):
                    print(f"\n### {exp.get('role')} at {exp.get('company')}")
                    for bullet in exp.get("bullets", []):
                        print(f"  • {bullet}")
                
                print()
                print(f"Seniority Applied: {parsed.get('seniority_applied', 'N/A')}")
                print(f"Quantification Score: {parsed.get('quantification_score', 0):.2%}")
                
            except json.JSONDecodeError as e:
                print(f"\n⚠️  JSON Parse Error: {e}")
                print("Raw output saved to file for inspection.")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_minimax()