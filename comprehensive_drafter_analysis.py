#!/usr/bin/env python3
"""
Comprehensive Drafter Model Analysis
Tests multiple models with identical inputs and generates a detailed comparison document
"""

import json
import time
import asyncio
import os
from typing import Dict, List, Any
from datetime import datetime
import requests

# Test data - Real resumes and job ads
RESUME_DATA_SCIENTIST = """Michael Chen
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
"""

JOB_AD_ARMADA = """Armada AI - Senior ML Engineer
San Francisco, CA | Remote OK

We're looking for a Senior ML Engineer to lead our AI infrastructure team.

Requirements:
- 5+ years of experience with machine learning and deep learning
- Strong background in Python, PyTorch, and TensorFlow
- Experience with distributed systems and cloud infrastructure (AWS/GCP)
- Leadership experience mentoring junior engineers
- Experience with production ML systems and MLOps
- Strong understanding of model deployment and monitoring

Responsibilities:
- Design and implement scalable ML pipelines
- Lead technical architecture decisions
- Mentor junior engineers
- Collaborate with product and data teams
- Optimize model performance and inference latency

Nice to have:
- Experience with LLMs and transformers
- Kubernetes and Docker expertise
- Experience with feature stores and data pipelines
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
        {"skill": "Leadership", "confidence": 0.75, "source": "inferred"},
    ],
    "domain_insights": {
        "primary_domain": "Data Science / ML",
        "market_insights": "High demand for ML engineers with 5+ years experience",
        "salary_range": "$180K - $250K",
        "trending_skills": ["LLMs", "MLOps", "Distributed Systems"]
    }
}

# Simulated FastSVM output
FASTSVM_OUTPUT = {
    "extracted_job_titles": [
        {"title": "Data Scientist", "confidence": 0.92},
        {"title": "ML Engineer", "confidence": 0.88},
        {"title": "Analytics Engineer", "confidence": 0.75}
    ],
    "extracted_skills": [
        {"skill": "Python", "confidence": 0.95},
        {"skill": "Machine Learning", "confidence": 0.92},
        {"skill": "Statistical Analysis", "confidence": 0.88},
        {"skill": "Data Visualization", "confidence": 0.85},
        {"skill": "SQL", "confidence": 0.90},
        {"skill": "TensorFlow", "confidence": 0.82},
        {"skill": "PyTorch", "confidence": 0.70},
        {"skill": "AWS", "confidence": 0.65},
    ]
}

# Model configurations with CORRECTED pricing
MODELS = {
    "DeepSeek v3.2": {
        "slug": "deepseek/deepseek-v3.2",
        "input_price": 0.00009,  # $0.09 per 1M
        "output_price": 0.00029,  # $0.29 per 1M
        "notes": "Ultra-cheap, precise reasoning"
    },
    "Claude 3 Haiku": {
        "slug": "anthropic/claude-3-haiku",
        "input_price": 0.00025,  # $0.25 per 1K
        "output_price": 0.00125,  # $1.25 per 1K
        "notes": "Fast, reliable, good JSON"
    },
    "Xiaomi MiMo v2 Flash": {
        "slug": "xiaomi/mimo-v2-flash",
        "input_price": 0.00009,  # $0.09 per 1M
        "output_price": 0.00029,  # $0.29 per 1M
        "notes": "Cheapest, fast, good quality"
    },
    "Mistral Large 3": {
        "slug": "mistralai/mistral-large-2512",
        "input_price": 0.0005,  # $0.50 per 1K
        "output_price": 0.0015,  # $1.50 per 1K
        "notes": "High quality, expensive"
    },
    "Grok 4.1 Fast": {
        "slug": "x-ai/grok-4.1-fast",
        "input_price": 0.0002,  # $0.20 per 1K (≤128K)
        "output_price": 0.0005,  # $0.50 per 1K (≤128K)
        "notes": "Fast, reasonable cost"
    }
}

class ComprehensiveDrafterAnalysis:
    """Comprehensive analysis of drafter models with detailed output comparison."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://imaginator-resume-cowriter.onrender.com",
            "X-Title": "Imaginator Comprehensive Drafter Analysis"
        }
        self.results = []
    
    def create_drafter_prompt(self) -> tuple:
        """Create the exact prompt used for all models."""
        
        system_prompt = """You are an expert Resume Writer specializing in STAR format bullets.
Your task: Rewrite the user's experiences into 3-5 professional STAR bullets.

CRITICAL RULES:
1. Use ONLY the user's actual company names and roles
2. DO NOT hallucinate technologies the user didn't mention
3. Every bullet MUST include at least one quantifiable metric (%, $, time, scale)
4. Use strong action verbs appropriate for senior-level roles
5. Focus on impact and business value
6. Return VALID JSON only

Output JSON Schema:
{
  "rewritten_experiences": [
    {
      "company": "Company Name",
      "role": "Job Title",
      "bullets": ["Bullet 1 with metric", "Bullet 2 with metric"],
      "metrics_used": ["metric1", "metric2"]
    }
  ],
  "seniority_applied": "senior",
  "quantification_score": 0.95
}
"""
        
        # Extract key data from Hermes and FastSVM
        hermes_skills = [s["skill"] for s in HERMES_OUTPUT["extracted_skills"][:5]]
        fastsvm_titles = [t["title"] for t in FASTSVM_OUTPUT["extracted_job_titles"][:3]]
        fastsvm_skills = [s["skill"] for s in FASTSVM_OUTPUT["extracted_skills"][:5]]
        
        user_prompt = f"""
RESUME DATA:
{RESUME_DATA_SCIENTIST}

TARGET JOB DESCRIPTION:
{JOB_AD_ARMADA}

HERMES EXTRACTED SKILLS (from resume analysis):
{', '.join(hermes_skills)}

FASTSVM DETECTED JOB TITLES:
{', '.join(fastsvm_titles)}

FASTSVM EXTRACTED SKILLS:
{', '.join(fastsvm_skills)}

MARKET INSIGHTS FROM HERMES:
- Primary Domain: {HERMES_OUTPUT['domain_insights']['primary_domain']}
- Trending Skills: {', '.join(HERMES_OUTPUT['domain_insights']['trending_skills'])}

TASK: Rewrite the user's experiences to match the target job (Senior ML Engineer).
Focus on:
1. Leadership and mentoring experience
2. ML/AI infrastructure and scalability
3. Production systems and deployment
4. Quantifiable business impact
"""
        
        return system_prompt, user_prompt
    
    async def call_model(self, model_slug: str, system_prompt: str, 
                        user_prompt: str, model_name: str) -> Dict[str, Any]:
        """Call OpenRouter with the exact same prompts for all models."""
        start_time = time.time()
        
        try:
            payload = {
                "model": model_slug,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 2000
            }
            
            response = requests.post(
                self.base_url,
                json=payload,
                headers=self.headers,
                timeout=60
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                usage = result.get("usage", {})
                content = result["choices"][0]["message"]["content"]
                
                # Try to parse JSON response
                try:
                    # Remove markdown code blocks if present
                    json_content = content
                    if json_content.startswith("```json"):
                        json_content = json_content[7:]
                    if json_content.startswith("```"):
                        json_content = json_content[3:]
                    if json_content.endswith("```"):
                        json_content = json_content[:-3]
                    
                    # Find the JSON object
                    start_idx = json_content.find('{')
                    end_idx = json_content.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_content = json_content[start_idx:end_idx+1]
                    
                    parsed = json.loads(json_content.strip())
                    
                    return {
                        "success": True,
                        "model": model_name,
                        "model_slug": model_slug,
                        "tokens_in": usage.get("prompt_tokens", 0),
                        "tokens_out": usage.get("completion_tokens", 0),
                        "response_time": response_time,
                        "parsed": parsed,
                        "raw_response": content,
                        "cost": self.calculate_cost(
                            usage.get("prompt_tokens", 0),
                            usage.get("completion_tokens", 0),
                            model_name
                        )
                    }
                except json.JSONDecodeError as e:
                    return {
                        "success": False,
                        "model": model_name,
                        "error": f"JSON parse error: {str(e)}",
                        "raw_response": content,
                        "tokens_in": usage.get("prompt_tokens", 0),
                        "tokens_out": usage.get("completion_tokens", 0),
                        "response_time": response_time
                    }
            else:
                return {
                    "success": False,
                    "model": model_name,
                    "error": f"API Error {response.status_code}: {response.text[:200]}"
                }
        
        except Exception as e:
            return {
                "success": False,
                "model": model_name,
                "error": str(e)
            }
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model_name: str) -> float:
        """Calculate cost for the model."""
        model = MODELS[model_name]
        input_cost = (input_tokens / 1_000_000) * model["input_price"] * 1000
        output_cost = (output_tokens / 1_000_000) * model["output_price"] * 1000
        return input_cost + output_cost
    
    async def run_test(self):
        """Run the comprehensive test."""
        print("=" * 80)
        print("COMPREHENSIVE DRAFTER MODEL ANALYSIS")
        print("=" * 80)
        print(f"Date: {datetime.now().isoformat()}\n")
        
        # Create prompts
        system_prompt, user_prompt = self.create_drafter_prompt()
        
        print(f"System Prompt Length: {len(system_prompt)} chars")
        print(f"User Prompt Length: {len(user_prompt)} chars\n")
        
        # Test each model
        for model_name, model_info in MODELS.items():
            print(f"\n{'='*60}")
            print(f"Testing: {model_name}")
            print(f"{'='*60}")
            
            result = await self.call_model(
                model_info["slug"],
                system_prompt,
                user_prompt,
                model_name
            )
            
            if result["success"]:
                print(f"✅ Success!")
                print(f"   Tokens: {result['tokens_in']} in / {result['tokens_out']} out")
                print(f"   Response Time: {result['response_time']:.2f}s")
                print(f"   Cost: ${result['cost']:.6f}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
            
            self.results.append(result)
            await asyncio.sleep(1)
        
        # Generate comprehensive document
        self.generate_comprehensive_document(system_prompt, user_prompt)
    
    def generate_comprehensive_document(self, system_prompt: str, user_prompt: str):
        """Generate a comprehensive analysis document."""
        doc = []
        
        doc.append("=" * 100)
        doc.append("COMPREHENSIVE DRAFTER MODEL ANALYSIS & OUTPUT COMPARISON")
        doc.append("=" * 100)
        doc.append(f"Date: {datetime.now().isoformat()}")
        doc.append(f"Test Type: Realistic Drafter Test with Real Pipeline Inputs")
        doc.append("")
        
        # Section 1: Input Data
        doc.append("\n" + "=" * 100)
        doc.append("SECTION 1: INPUT DATA (IDENTICAL FOR ALL MODELS)")
        doc.append("=" * 100)
        
        doc.append("\n### ORIGINAL RESUME")
        doc.append("```")
        doc.append(RESUME_DATA_SCIENTIST)
        doc.append("```")
        
        doc.append("\n### TARGET JOB DESCRIPTION")
        doc.append("```")
        doc.append(JOB_AD_ARMADA)
        doc.append("```")
        
        doc.append("\n### HERMES EXTRACTED SKILLS")
        doc.append("```json")
        doc.append(json.dumps(HERMES_OUTPUT, indent=2))
        doc.append("```")
        
        doc.append("\n### FASTSVM EXTRACTED DATA")
        doc.append("```json")
        doc.append(json.dumps(FASTSVM_OUTPUT, indent=2))
        doc.append("```")
        
        # Section 2: Prompts
        doc.append("\n" + "=" * 100)
        doc.append("SECTION 2: EXACT PROMPTS SENT TO ALL MODELS")
        doc.append("=" * 100)
        
        doc.append("\n### SYSTEM PROMPT")
        doc.append(f"Length: {len(system_prompt)} characters")
        doc.append("```")
        doc.append(system_prompt)
        doc.append("```")
        
        doc.append("\n### USER PROMPT")
        doc.append(f"Length: {len(user_prompt)} characters")
        doc.append("```")
        doc.append(user_prompt)
        doc.append("```")
        
        # Section 3: Model Outputs
        doc.append("\n" + "=" * 100)
        doc.append("SECTION 3: MODEL OUTPUTS & ANALYSIS")
        doc.append("=" * 100)
        
        for result in self.results:
            doc.append(f"\n{'='*100}")
            doc.append(f"MODEL: {result['model']}")
            doc.append(f"{'='*100}")
            
            if result["success"]:
                doc.append(f"\n**Status:** ✅ SUCCESS")
                doc.append(f"**Model Slug:** {result['model_slug']}")
                doc.append(f"**Response Time:** {result['response_time']:.2f}s")
                doc.append(f"**Tokens Used:** {result['tokens_in']} input / {result['tokens_out']} output (Total: {result['tokens_in'] + result['tokens_out']})")
                doc.append(f"**Cost:** ${result['cost']:.8f}")
                
                doc.append(f"\n#### RAW RESPONSE")
                doc.append("```")
                doc.append(result['raw_response'][:500] + "..." if len(result['raw_response']) > 500 else result['raw_response'])
                doc.append("```")
                
                doc.append(f"\n#### PARSED JSON OUTPUT")
                doc.append("```json")
                doc.append(json.dumps(result['parsed'], indent=2))
                doc.append("```")
                
                doc.append(f"\n#### FORMATTED RESUME OUTPUT")
                doc.append("```markdown")
                doc.append("## Professional Experience\n")
                for exp in result['parsed'].get("rewritten_experiences", []):
                    doc.append(f"### {exp.get('role', 'Role')} at {exp.get('company', 'Company')}\n")
                    for bullet in exp.get("bullets", []):
                        doc.append(f"- {bullet}")
                    doc.append("")
                doc.append(f"\n**Seniority Level:** {result['parsed'].get('seniority_applied', 'N/A')}")
                doc.append(f"**Quantification Score:** {result['parsed'].get('quantification_score', 0):.0%}")
                doc.append("```")
            else:
                doc.append(f"\n**Status:** ❌ FAILED")
                doc.append(f"**Error:** {result.get('error', 'Unknown error')}")
                doc.append(f"**Raw Response:** {result.get('raw_response', 'N/A')[:200]}")
        
        # Section 4: Comparison
        doc.append("\n" + "=" * 100)
        doc.append("SECTION 4: MODEL COMPARISON")
        doc.append("=" * 100)
        
        successful = [r for r in self.results if r.get("success")]
        
        doc.append("\n| Model | Cost | Speed | Input Tokens | Output Tokens | Total Tokens |")
        doc.append("|-------|------|-------|--------------|---------------|--------------|")
        for result in sorted(successful, key=lambda x: x["cost"]):
            doc.append(f"| {result['model']:20} | ${result['cost']:.8f} | {result['response_time']:6.2f}s | {result['tokens_in']:12} | {result['tokens_out']:13} | {result['tokens_in'] + result['tokens_out']:12} |")
        
        # Save document
        filename = f"comprehensive_drafter_analysis_{int(time.time())}.md"
        with open(filename, "w") as f:
            f.write("\n".join(doc))
        
        print(f"\n✅ Comprehensive analysis saved to: {filename}")

async def main():
    """Main execution."""
    api_key = os.getenv("OPENROUTER_API_KEY_1") or os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("❌ ERROR: No OpenRouter API key found!")
        return
    
    test = ComprehensiveDrafterAnalysis(api_key)
    await test.run_test()

if __name__ == "__main__":
    asyncio.run(main())
