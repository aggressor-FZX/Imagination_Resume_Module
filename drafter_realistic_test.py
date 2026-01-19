#!/usr/bin/env python3
"""
Realistic Drafter Test - Tests with actual resume/job ad data
Simulates real pipeline inputs from Hermes (skills) and FastSVM (job title extraction)
Tests only the DRAFTER stage with consistent inputs across all models
Saves actual resume outputs for inspection
"""

import json
import time
import asyncio
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import requests
import os

# ============================================================================
# REALISTIC PIPELINE DATA (Simulating Hermes + FastSVM outputs)
# ============================================================================

# Sample 1: Data Scientist → AI Engineer role
SAMPLE_1_RESUME = """Michael Chen
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
Time Series Analysis, A/B Testing, Data Visualization"""

SAMPLE_1_JOB_AD = """Armada is an edge computing startup that provides computing infrastructure to remote areas. 
We're looking for AI Engineers with hands-on expertise in real-time computer vision, statistical machine learning, 
natural language processing, transformers, and large-scale distributed AI systems.

Key Responsibilities:
- Building AI/ML/DL models by applying state-of-the-art algorithms, especially transformers
- Preparing data to train and evaluate AI/ML/DL models
- Deploying models in production by containerizing them
- Testing and evaluating models, benchmarking quality

Required Qualifications:
- 3+ years of work-related experience in software development with Python, Java, and/or C/C++
- Hands-on expertise with traditional statistical machine learning and deep-learning
- Experience with DNN architectures (Transformers, CNN, RNN, BERT, GAN, etc.)
- Experience with PyTorch, Tensorflow, or similar frameworks
- Familiarity with containers and Kubernetes"""

# Simulated Hermes output (extracted skills with confidence)
SAMPLE_1_HERMES_SKILLS = [
    {"skill": "Python", "confidence": 0.95, "source": "resume"},
    {"skill": "Machine Learning", "confidence": 0.92, "source": "resume"},
    {"skill": "Statistical Analysis", "confidence": 0.88, "source": "resume"},
    {"skill": "TensorFlow", "confidence": 0.85, "source": "resume"},
    {"skill": "Data Visualization", "confidence": 0.82, "source": "resume"},
    {"skill": "SQL", "confidence": 0.90, "source": "resume"},
    {"skill": "ETL", "confidence": 0.78, "source": "resume"},
]

# Simulated FastSVM output (job title extraction)
SAMPLE_1_FASTSVM_ANALYSIS = {
    "current_title": "Data Scientist",
    "target_title": "AI Engineer",
    "seniority_level": "mid",
    "domain": "Machine Learning / AI",
    "gap_analysis": {
        "critical_gaps": ["Transformers", "Real-time Computer Vision", "Containerization/Kubernetes"],
        "nice_to_have": ["Reinforcement Learning", "Edge Computing"],
        "transferable_skills": ["Python", "ML Algorithms", "Statistical Analysis"]
    }
}

# Simulated Researcher output (domain insights)
SAMPLE_1_RESEARCHER_DATA = {
    "implied_metrics": ["87% accuracy", "$2M savings", "10M+ requests", "99.9% uptime"],
    "implied_skills": ["Transformers", "Computer Vision", "Distributed Systems", "Model Deployment"],
    "domain_vocab": ["edge computing", "real-time inference", "containerization", "autonomous learning"],
    "action_plan": "Emphasize ML expertise, highlight scalability experience, add transformer/CV context"
}

# ============================================================================
# EXPERIENCE DATA (What Drafter receives from pipeline)
# ============================================================================

SAMPLE_1_EXPERIENCES = [
    {
        "company": "FinTech Analytics",
        "role": "Data Scientist",
        "period": "2021-Present",
        "bullets": [
            "Developed churn prediction model with 87% accuracy, saving $2M annually in customer retention",
            "Built real-time analytics dashboard using Tableau and Python, used by C-suite executives",
            "Implemented A/B testing framework that increased conversion rates by 12%",
            "Automated ETL pipelines reducing manual reporting time by 20 hours/week"
        ]
    },
    {
        "company": "Retail Insights",
        "role": "Data Analyst",
        "period": "2019-2021",
        "bullets": [
            "Analyzed customer behavior data for 1M+ users, identified key purchasing patterns",
            "Created forecasting models for inventory management, reducing waste by 18%",
            "Collaborated with marketing team to optimize campaign targeting"
        ]
    }
]

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODELS = {
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
        "notes": "MoE, good for complex tasks"
    }
}

# ============================================================================
# DRAFTER PROMPT (Real pipeline prompt)
# ============================================================================

DRAFTER_SYSTEM_PROMPT = """You are an expert Resume Writer specializing in STAR format bullets.
Your task: Rewrite the user's experiences into 3-4 STAR bullets that bridge gaps to the target role.

CRITICAL RULES:
1. USE ONLY the user's actual company names - NO hallucination
2. Structure each bullet: [Action Verb] [What] [Tech/Tool] [Result with Metric]
3. Emphasize metrics: %, $, time, scale, users
4. Seniority tone: Mid-level (confident, independent, results-driven)
5. Bridge to target role: Highlight ML/AI relevance, scalability, deployment experience

EXAMPLE FORMATS:
✓ "Engineered churn prediction model using TensorFlow, achieving 87% accuracy and saving $2M annually"
✓ "Architected real-time analytics pipeline processing 10M+ daily events with 99.9% uptime"
✓ "Optimized ML model inference latency by 60% through containerization and caching strategies"

Output ONLY valid JSON with this exact structure:
{
  "rewritten_experiences": [
    {
      "company": "Company Name",
      "role": "Job Title",
      "bullets": ["Bullet 1", "Bullet 2", "Bullet 3"],
      "metrics_used": ["metric1", "metric2"]
    }
  ],
  "seniority_applied": "mid",
  "quantification_score": 0.95
}"""

@dataclass
class DrafterTestResult:
    model_name: str
    model_slug: str
    input_tokens: int
    output_tokens: int
    cost: float
    response_time: float
    success: bool
    quality_score: float
    output_file: str = ""
    error: str = ""

class RealisticDrafterTest:
    """Tests Drafter stage with realistic pipeline inputs."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://imaginator-resume-cowriter.onrender.com",
            "X-Title": "Imaginator Drafter Test"
        }
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"drafter_outputs_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_drafter_prompt(self, experiences: List[Dict], job_ad: str, 
                             researcher_data: Dict) -> str:
        """Create realistic drafter user prompt."""
        return f"""
CANDIDATE EXPERIENCES (JSON):
{json.dumps(experiences, indent=2)}

TARGET JOB DESCRIPTION:
{job_ad[:1500]}

RESEARCH INSIGHTS:
- Key Metrics to Highlight: {', '.join(researcher_data.get('implied_metrics', [])[:3])}
- Domain Vocabulary: {', '.join(researcher_data.get('domain_vocab', [])[:5])}
- Implied Skills: {', '.join(researcher_data.get('implied_skills', [])[:5])}
- Gap Bridging Strategy: {researcher_data.get('action_plan', 'Emphasize relevant experience')}

TASK: Rewrite the candidate's experiences into STAR bullets that align with the target role.
Focus on: ML expertise, scalability, metrics, and deployment experience.
"""
    
    async def call_model(self, model_slug: str, system_prompt: str, 
                        user_prompt: str, max_tokens: int = 1500) -> Dict[str, Any]:
        """Call OpenRouter with realistic drafter prompt."""
        start_time = time.time()
        
        try:
            payload = {
                "model": model_slug,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,  # Drafter temperature
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"}
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
                
                return {
                    "success": True,
                    "content": content,
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "response_time": response_time
                }
            else:
                return {
                    "success": False,
                    "error": f"API Error {response.status_code}: {response.text[:200]}",
                    "response_time": response_time,
                    "input_tokens": 0,
                    "output_tokens": 0
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": time.time() - start_time,
                "input_tokens": 0,
                "output_tokens": 0
            }
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model_name: str) -> float:
        """Calculate cost for model."""
        model = MODELS[model_name]
        input_cost = (input_tokens / 1000) * model["input_price"]
        output_cost = (output_tokens / 1000) * model["output_price"]
        return input_cost + output_cost
    
    def evaluate_quality(self, response: str) -> float:
        """Evaluate response quality."""
        score = 0.0
        
        try:
            data = json.loads(response)
            
            # Check structure
            if "rewritten_experiences" in data:
                score += 0.3
            
            # Check bullets exist and have metrics
            experiences = data.get("rewritten_experiences", [])
            if experiences:
                score += 0.2
                for exp in experiences:
                    bullets = exp.get("bullets", [])
                    if bullets:
                        score += 0.1
                        # Check for metrics in bullets
                        for bullet in bullets:
                            if any(char in bullet for char in ['%', '$', 'x', '+']):
                                score += 0.15
                                break
            
            # Check quantification score
            if data.get("quantification_score", 0) > 0.8:
                score += 0.15
            
        except:
            score = 0.1  # Partial credit for any response
        
        return min(score, 1.0)
    
    def save_output(self, model_name: str, response: str, 
                   input_tokens: int, output_tokens: int, cost: float) -> str:
        """Save model output to file."""
        filename = f"{self.output_dir}/{model_name.replace(' ', '_')}_output.txt"
        
        try:
            data = json.loads(response)
            
            with open(filename, "w") as f:
                f.write(f"{'='*70}\n")
                f.write(f"MODEL: {model_name}\n")
                f.write(f"{'='*70}\n")
                f.write(f"Tokens: {input_tokens} in / {output_tokens} out\n")
                f.write(f"Cost: ${cost:.6f}\n")
                f.write(f"{'='*70}\n\n")
                
                # Write formatted resume
                f.write("REWRITTEN RESUME BULLETS:\n")
                f.write("-" * 70 + "\n\n")
                
                for exp in data.get("rewritten_experiences", []):
                    f.write(f"**{exp.get('role')}** at *{exp.get('company')}*\n")
                    for bullet in exp.get("bullets", []):
                        f.write(f"• {bullet}\n")
                    f.write("\n")
                
                f.write("\n" + "="*70 + "\n")
                f.write("RAW JSON OUTPUT:\n")
                f.write("="*70 + "\n")
                f.write(json.dumps(data, indent=2))
        
        except Exception as e:
            with open(filename, "w") as f:
                f.write(f"ERROR: {e}\n\n")
                f.write(f"Raw response:\n{response}")
        
        return filename
    
    async def run_test(self) -> Dict[str, Any]:
        """Run realistic drafter test."""
        print("="*70)
        print("REALISTIC DRAFTER TEST - Real Pipeline Inputs")
        print("="*70)
        print(f"Date: {datetime.now().isoformat()}")
        print(f"Output Directory: {self.output_dir}")
        print(f"\nTest Scenario: Data Scientist → AI Engineer (Armada)")
        print(f"Max Tokens: 1500 (realistic limit)")
        print(f"Temperature: 0.3 (consistent, structured output)")
        
        results = []
        
        # Create prompts
        system_prompt = DRAFTER_SYSTEM_PROMPT
        user_prompt = self.create_drafter_prompt(
            SAMPLE_1_EXPERIENCES,
            SAMPLE_1_JOB_AD,
            SAMPLE_1_RESEARCHER_DATA
        )
        
        print(f"\nPrompt Size: {len(user_prompt)} chars")
        print(f"Experiences: {len(SAMPLE_1_EXPERIENCES)}")
        print(f"Total Bullets: {sum(len(e['bullets']) for e in SAMPLE_1_EXPERIENCES)}")
        
        # Test each model
        for model_name, model_info in MODELS.items():
            print(f"\n{'='*70}")
            print(f"Testing: {model_name}")
            print(f"{'='*70}")
            
            result = await self.call_model(
                model_info["slug"],
                system_prompt,
                user_prompt,
                max_tokens=1500
            )
            
            if result["success"]:
                cost = self.calculate_cost(
                    result["input_tokens"],
                    result["output_tokens"],
                    model_name
                )
                quality = self.evaluate_quality(result["content"])
                output_file = self.save_output(
                    model_name,
                    result["content"],
                    result["input_tokens"],
                    result["output_tokens"],
                    cost
                )
                
                print(f"✅ Success!")
                print(f"   Response Time: {result['response_time']:.2f}s")
                print(f"   Tokens: {result['input_tokens']} in / {result['output_tokens']} out")
                print(f"   Cost: ${cost:.6f}")
                print(f"   Quality Score: {quality:.2f}/1.0")
                print(f"   Output File: {output_file}")
                
                test_result = DrafterTestResult(
                    model_name=model_name,
                    model_slug=model_info["slug"],
                    input_tokens=result["input_tokens"],
                    output_tokens=result["output_tokens"],
                    cost=cost,
                    response_time=result["response_time"],
                    success=True,
                    quality_score=quality,
                    output_file=output_file
                )
            else:
                print(f"❌ Failed: {result['error']}")
                test_result = DrafterTestResult(
                    model_name=model_name,
                    model_slug=model_info["slug"],
                    input_tokens=0,
                    output_tokens=0,
                    cost=0.0,
                    response_time=result["response_time"],
                    success=False,
                    quality_score=0.0,
                    error=result["error"]
                )
            
            results.append(test_result)
            await asyncio.sleep(1)  # Rate limiting
        
        # Generate summary
        return self.generate_summary(results)
    
    def generate_summary(self, results: List[DrafterTestResult]) -> Dict[str, Any]:
        """Generate test summary."""
        successful = [r for r in results if r.success]
        
        summary = {
            "timestamp": self.timestamp,
            "output_directory": self.output_dir,
            "test_scenario": "Data Scientist → AI Engineer (Armada)",
            "total_tests": len(results),
            "successful_tests": len(successful),
            "results": []
        }
        
        for result in results:
            summary["results"].append({
                "model": result.model_name,
                "slug": result.model_slug,
                "success": result.success,
                "tokens_in": result.input_tokens,
                "tokens_out": result.output_tokens,
                "cost": result.cost,
                "response_time": result.response_time,
                "quality_score": result.quality_score,
                "output_file": result.output_file,
                "error": result.error
            })
        
        if successful:
            best_cost = min(successful, key=lambda x: x.cost)
            best_quality = max(successful, key=lambda x: x.quality_score)
            fastest = min(successful, key=lambda x: x.response_time)
            
            summary["recommendations"] = {
                "best_cost": f"{best_cost.model_name} (${best_cost.cost:.6f})",
                "best_quality": f"{best_quality.model_name} ({best_quality.quality_score:.2f}/1.0)",
                "fastest": f"{fastest.model_name} ({fastest.response_time:.2f}s)"
            }
        
        return summary

async def main():
    """Main execution."""
    api_key = os.getenv("OPENROUTER_API_KEY_1") or os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("❌ ERROR: No OpenRouter API key found!")
        return
    
    test = RealisticDrafterTest(api_key)
    
    try:
        summary = await test.run_test()
        
        # Print summary
        print(f"\n{'='*70}")
        print("TEST SUMMARY")
        print(f"{'='*70}")
        print(f"Output Directory: {summary['output_directory']}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']}")
        
        if "recommendations" in summary:
            print(f"\nRecommendations:")
            print(f"  Best Cost: {summary['recommendations']['best_cost']}")
            print(f"  Best Quality: {summary['recommendations']['best_quality']}")
            print(f"  Fastest: {summary['recommendations']['fastest']}")
        
        print(f"\nDetailed Results:")
        for result in summary["results"]:
            status = "✅" if result["success"] else "❌"
            print(f"\n{status} {result['model']}")
            if result["success"]:
                print(f"   Tokens: {result['tokens_in']} in / {result['tokens_out']} out")
                print(f"   Cost: ${result['cost']:.6f}")
                print(f"   Quality: {result['quality_score']:.2f}/1.0")
                print(f"   Speed: {result['response_time']:.2f}s")
                print(f"   Output: {result['output_file']}")
            else:
                print(f"   Error: {result['error']}")
        
        # Save summary
        summary_file = f"{summary['output_directory']}/summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Summary saved to: {summary_file}")
        print(f"📁 All outputs saved to: {os.path.abspath(summary['output_directory'])}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
