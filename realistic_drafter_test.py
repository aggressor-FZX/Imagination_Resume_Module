#!/usr/bin/env python3
"""
Realistic Drafter Test - Simulates Real Pipeline Conditions
Tests the Drafter stage with actual resume/job ad data and real Hermes/FastSVM outputs.
Saves the actual resume outputs for inspection.
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

# Simulated Hermes output (what the Hermes service would return)
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

# Simulated FastSVM output (what the FastSVM service would return)
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

# Model configurations
MODELS = {
    "MiniMax M2.1": {
        "slug": "minimax/minimax-m2.1",
        "input_price": 0.00010,
        "output_price": 0.00040,
        "notes": "MiniMax M2.1 - Fast reasoning model"
    },
    "DeepSeek v3.2": {
        "slug": "deepseek/deepseek-v3.2",
        "input_price": 0.00015,
        "output_price": 0.00075,
        "notes": "Ultra-cheap, precise reasoning"
    },
    "Claude 3 Haiku": {
        "slug": "anthropic/claude-3-haiku",
        "input_price": 0.00025,
        "output_price": 0.00125,
        "notes": "Fast, reliable, good JSON"
    },
    "Xiaomi MiMo v2 Flash": {
        "slug": "xiaomi/mimo-v2-flash",
        "input_price": 0.10,
        "output_price": 0.30,
        "notes": "Fast, but expensive"
    }
}

class RealisticDrafterTest:
    """Tests the Drafter stage with real pipeline inputs."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://imaginator-resume-cowriter.onrender.com",
            "X-Title": "Imaginator Realistic Drafter Test"
        }
        self.results = []
    
    def create_drafter_prompt(self, resume_text: str, job_ad: str, 
                             hermes_data: Dict, fastsvm_data: Dict) -> tuple:
        """Create realistic drafter prompts with Hermes/FastSVM data."""
        
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
        hermes_skills = [s["skill"] for s in hermes_data["extracted_skills"][:5]]
        fastsvm_titles = [t["title"] for t in fastsvm_data["extracted_job_titles"][:3]]
        fastsvm_skills = [s["skill"] for s in fastsvm_data["extracted_skills"][:5]]
        
        user_prompt = f"""
RESUME DATA:
{resume_text}

TARGET JOB DESCRIPTION:
{job_ad}

HERMES EXTRACTED SKILLS (from resume analysis):
{', '.join(hermes_skills)}

FASTSVM DETECTED JOB TITLES:
{', '.join(fastsvm_titles)}

FASTSVM EXTRACTED SKILLS:
{', '.join(fastsvm_skills)}

MARKET INSIGHTS FROM HERMES:
- Primary Domain: {hermes_data['domain_insights']['primary_domain']}
- Trending Skills: {', '.join(hermes_data['domain_insights']['trending_skills'])}

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
        """Call OpenRouter with realistic drafter prompts."""
        start_time = time.time()
        
        try:
            payload = {
                "model": model_slug,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,  # Low for consistency
                "max_tokens": 2000   # Hard limit
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
                
                # Save raw response
                filename = f"drafter_output_{model_name.replace(' ', '_')}_{int(time.time())}.json"
                with open(filename, "w") as f:
                    json.dump({
                        "model": model_name,
                        "model_slug": model_slug,
                        "timestamp": datetime.now().isoformat(),
                        "tokens_in": usage.get("prompt_tokens", 0),
                        "tokens_out": usage.get("completion_tokens", 0),
                        "response_time": response_time,
                        "raw_response": content
                    }, f, indent=2)
                
                # Try to parse JSON response (handle markdown code blocks and extra text)
                try:
                    # Remove markdown code blocks if present
                    json_content = content
                    if json_content.startswith("```json"):
                        json_content = json_content[7:]  # Remove ```json
                    if json_content.startswith("```"):
                        json_content = json_content[3:]  # Remove ```
                    if json_content.endswith("```"):
                        json_content = json_content[:-3]  # Remove trailing ```
                    
                    # Find the JSON object (handle text before/after)
                    start_idx = json_content.find('{')
                    end_idx = json_content.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_content = json_content[start_idx:end_idx+1]
                    
                    parsed = json.loads(json_content.strip())
                    resume_filename = f"resume_{model_name.replace(' ', '_')}_{int(time.time())}.md"
                    with open(resume_filename, "w") as f:
                        f.write(f"# Resume - Generated by {model_name}\n")
                        f.write(f"Generated: {datetime.now().isoformat()}\n")
                        f.write(f"Tokens Used: {usage.get('prompt_tokens', 0)} in / {usage.get('completion_tokens', 0)} out\n")
                        f.write(f"Response Time: {response_time:.2f}s\n\n")
                        f.write("## Professional Experience\n\n")
                        
                        for exp in parsed.get("rewritten_experiences", []):
                            f.write(f"### {exp.get('role', 'Role')} at {exp.get('company', 'Company')}\n\n")
                            for bullet in exp.get("bullets", []):
                                f.write(f"- {bullet}\n")
                            f.write("\n")
                        
                        f.write(f"\n**Seniority Level Applied:** {parsed.get('seniority_applied', 'N/A')}\n")
                        f.write(f"**Quantification Score:** {parsed.get('quantification_score', 0):.2%}\n")
                    
                    return {
                        "success": True,
                        "model": model_name,
                        "tokens_in": usage.get("prompt_tokens", 0),
                        "tokens_out": usage.get("completion_tokens", 0),
                        "response_time": response_time,
                        "parsed": parsed,
                        "raw_file": filename,
                        "resume_file": resume_filename,
                        "cost": self.calculate_cost(
                            usage.get("prompt_tokens", 0),
                            usage.get("completion_tokens", 0),
                            model_name
                        )
                    }
                except json.JSONDecodeError:
                    return {
                        "success": False,
                        "model": model_name,
                        "error": "Failed to parse JSON response",
                        "raw_file": filename,
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
        input_cost = (input_tokens / 1000) * model["input_price"]
        output_cost = (output_tokens / 1000) * model["output_price"]
        return input_cost + output_cost
    
    async def run_test(self):
        """Run the realistic drafter test."""
        print("=" * 80)
        print("REALISTIC DRAFTER TEST - Real Pipeline Conditions")
        print("=" * 80)
        print(f"Date: {datetime.now().isoformat()}\n")
        
        print("📋 Test Setup:")
        print(f"  Resume: Data Scientist (4 years experience)")
        print(f"  Target Job: Senior ML Engineer at Armada AI")
        print(f"  Hermes Skills Extracted: {len(HERMES_OUTPUT['extracted_skills'])}")
        print(f"  FastSVM Job Titles: {len(FASTSVM_OUTPUT['extracted_job_titles'])}")
        print(f"  Models to Test: {len(MODELS)}\n")
        
        # Create prompts
        system_prompt, user_prompt = self.create_drafter_prompt(
            RESUME_DATA_SCIENTIST,
            JOB_AD_ARMADA,
            HERMES_OUTPUT,
            FASTSVM_OUTPUT
        )
        
        print(f"System Prompt Length: {len(system_prompt)} chars")
        print(f"User Prompt Length: {len(user_prompt)} chars\n")
        
        # Test each model
        for model_name, model_info in MODELS.items():
            print(f"\n{'='*60}")
            print(f"Testing: {model_name}")
            print(f"{'='*60}")
            print(f"Model: {model_info['slug']}")
            print(f"Notes: {model_info['notes']}")
            
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
                print(f"   Resume saved to: {result['resume_file']}")
                print(f"   Raw JSON saved to: {result['raw_file']}")
                
                # Show bullet preview
                parsed = result.get("parsed", {})
                for exp in parsed.get("rewritten_experiences", [])[:1]:
                    print(f"\n   Preview - {exp.get('role')} at {exp.get('company')}:")
                    for bullet in exp.get("bullets", [])[:2]:
                        print(f"     • {bullet[:80]}...")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
            
            self.results.append(result)
            await asyncio.sleep(1)  # Rate limiting
        
        # Generate summary report
        self.generate_summary()
    
    def generate_summary(self):
        """Generate summary report."""
        print("\n" + "=" * 80)
        print("SUMMARY REPORT")
        print("=" * 80)
        
        successful = [r for r in self.results if r.get("success")]
        
        print(f"\n✅ Successful Tests: {len(successful)}/{len(self.results)}")
        
        if successful:
            print("\n📊 Token Usage Comparison:")
            for result in successful:
                total_tokens = result["tokens_in"] + result["tokens_out"]
                print(f"  {result['model']:20} | In: {result['tokens_in']:4} | Out: {result['tokens_out']:4} | Total: {total_tokens:5} | Cost: ${result['cost']:.6f}")
            
            print("\n⚡ Speed Comparison:")
            for result in sorted(successful, key=lambda x: x["response_time"]):
                print(f"  {result['model']:20} | {result['response_time']:6.2f}s")
            
            print("\n💰 Cost Comparison:")
            for result in sorted(successful, key=lambda x: x["cost"]):
                print(f"  {result['model']:20} | ${result['cost']:.6f}")
        
        print("\n📁 Output Files:")
        print("  All resume outputs saved in current directory:")
        print("  - resume_*.md (Formatted resumes)")
        print("  - drafter_output_*.json (Raw API responses)")
        
        # Save summary
        summary_file = f"drafter_test_summary_{int(time.time())}.json"
        with open(summary_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "test_type": "Realistic Drafter Test",
                "resume": "Data Scientist",
                "target_job": "Senior ML Engineer",
                "models_tested": list(MODELS.keys()),
                "results": self.results
            }, f, indent=2)
        
        print(f"\n✅ Summary saved to: {summary_file}")

async def main():
    """Main execution."""
    api_key = os.getenv("OPENROUTER_API_KEY_1") or os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("❌ ERROR: No OpenRouter API key found!")
        return
    
    test = RealisticDrafterTest(api_key)
    await test.run_test()

if __name__ == "__main__":
    asyncio.run(main())
