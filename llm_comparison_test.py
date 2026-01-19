#!/usr/bin/env python3
"""
LLM Comparison Test - DeepSeek v3.2 vs Grok 4.1 Fast vs Xiaomi MiMo v2 Flash
Performs identical tests on each model and provides cost analysis

Model Pricing (OpenRouter):
- DeepSeek v3.2: $0.00015 input / $0.00075 output per 1K tokens (Ultra-cheap)
- Grok 4.1 Fast: $0.30 input / $1.00 output per 1K tokens (Agentic/fast, promo pricing)
- Xiaomi MiMo v2 Flash: $0.10 input / $0.30 output per 1K tokens (MoE beast, #1 open-src SWE-Bench)
"""

import json
import time
import asyncio
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import requests

# Test data - same for all models
TEST_RESUME = """
John Smith
Senior Software Engineer

EXPERIENCE:
TechCorp Inc. - Senior Developer (2020-2024)
• Led development of microservices architecture serving 1M+ users
• Optimized database queries reducing response time by 40%
• Mentored 5 junior developers and conducted code reviews

Skills: Python, React, AWS, Docker, Kubernetes
"""

TEST_JOB_AD = """
Senior Full Stack Engineer - Tech Innovations Inc.
Requirements:
- 5+ years experience with React and Node.js
- Cloud infrastructure (AWS/Azure)
- Microservices architecture
- Leadership experience
"""

TEST_STAR_DATA = {
    "experience": [
        {
            "company": "TechCorp Inc.",
            "role": "Senior Developer",
            "period": "2020-2024",
            "bullets": [
                "Led development of microservices architecture serving 1M+ users",
                "Optimized database queries reducing response time by 40%",
                "Mentored 5 junior developers and conducted code reviews"
            ]
        }
    ],
    "skills": ["Python", "React", "AWS", "Docker", "Kubernetes"]
}

# Model configurations
MODELS = {
    "DeepSeek v3.2": {
        "slug": "deepseek/deepseek-v3.2",
        "input_price": 0.00015,
        "output_price": 0.00075,
        "context": "128K",
        "notes": "Ultra-cheap coding/reasoning leader"
    },
    "Grok 4.1 Fast": {
        "slug": "x-ai/grok-4.1-fast",
        "input_price": 0.30,
        "output_price": 1.00,
        "context": "2M",
        "notes": "Agentic/fast; promo pricing"
    },
    "Xiaomi MiMo v2 Flash": {
        "slug": "xiaomi/mimo-v2-flash",
        "input_price": 0.10,
        "output_price": 0.30,
        "context": "256K",
        "notes": "MoE beast, #1 open-src SWE-Bench"
    },
    "Claude 3 Haiku": {
        "slug": "anthropic/claude-3-haiku",
        "input_price": 0.00025,
        "output_price": 0.00125,
        "context": "200K",
        "notes": "Reliable JSON/writing baseline"
    }
}

# Test prompts (same for all models)
TEST_PROMPTS = {
    "researcher": """Analyze the following resume and job ad. Identify critical skill gaps and create a gap bridging strategy.

Resume:
{resume}

Job Ad:
{job_ad}

Return JSON with gap_analysis, implied_skills, and action_plan.""",
    
    "drafter": """Convert this STAR data into professional resume bullets. Focus on metrics and impact.

STAR Data:
{star_data}

Return JSON with gap_bridging and metric_improvements.""",
    
    "star_editor": """Polish this resume data into clean, ATS-friendly Markdown. Remove all labels and ensure professional formatting.

Input Data:
{star_data}

Return JSON with final_markdown and editorial_notes."""
}

@dataclass
class TestResult:
    model_name: str
    model_slug: str
    stage: str
    input_tokens: int
    output_tokens: int
    cost: float
    response_time: float
    success: bool
    quality_score: float
    error: str = ""

class LLMComparisonTest:
    """Performs comprehensive comparison tests across multiple LLM models."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://imaginator-resume-cowriter.onrender.com",
            "X-Title": "Imaginator LLM Comparison Test"
        }
    
    async def call_model(self, model_slug: str, prompt: str, temperature: float = 0.7, stage: str = "test") -> Dict[str, Any]:
        """Make API call to OpenRouter with specific model and 2000 hard limit."""
        start_time = time.time()
        
        try:
            payload = {
                "model": model_slug,
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": 2000  # HARD LIMIT as requested
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
                
                # Save response to file for inspection
                filename = f"response_{model_slug.replace('/', '_')}_{stage}_{int(time.time())}.txt"
                with open(filename, "w") as f:
                    f.write(f"Model: {model_slug}\n")
                    f.write(f"Stage: {stage}\n")
                    f.write(f"Tokens Used: {usage.get('prompt_tokens', 0)} in / {usage.get('completion_tokens', 0)} out\n")
                    f.write(f"Response Time: {response_time:.2f}s\n")
                    f.write("="*50 + "\n")
                    f.write(content)
                
                return {
                    "success": True,
                    "content": content,
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "response_time": response_time,
                    "raw_response": result,
                    "saved_file": filename
                }
            else:
                return {
                    "success": False,
                    "error": f"API Error {response.status_code}: {response.text}",
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
        """Calculate cost in dollars for given token usage."""
        model = MODELS[model_name]
        input_cost = (input_tokens / 1000) * model["input_price"]
        output_cost = (output_tokens / 1000) * model["output_price"]
        return input_cost + output_cost
    
    def evaluate_quality(self, response: str, stage: str) -> float:
        """Simple quality evaluation based on response characteristics."""
        if not response or not response.strip():
            return 0.0
        
        score = 0.0
        
        # Basic quality checks
        if len(response.strip()) > 50:
            score += 0.2
        
        # JSON structure check
        if stage in ["researcher", "drafter", "star_editor"]:
            if "{" in response and "}" in response:
                score += 0.3
        
        # Content richness
        words = response.split()
        if len(words) > 20:
            score += 0.2
        
        # Specific checks per stage
        if stage == "researcher":
            if "gap" in response.lower() or "analysis" in response.lower():
                score += 0.15
            if "action" in response.lower() or "strategy" in response.lower():
                score += 0.15
        
        elif stage == "drafter":
            if "metric" in response.lower() or "improvement" in response.lower():
                score += 0.2
            if "skill" in response.lower() or "suggestion" in response.lower():
                score += 0.1
        
        elif stage == "star_editor":
            if "##" in response or "markdown" in response.lower():
                score += 0.2
            if "professional" in response.lower() or "experience" in response.lower():
                score += 0.1
        
        return min(score, 1.0)
    
    async def run_stage_test(self, model_name: str, stage: str, test_data: Dict[str, Any]) -> TestResult:
        """Run a single stage test for a specific model."""
        model = MODELS[model_name]
        prompt_template = TEST_PROMPTS[stage]
        
        # Format prompt with test data
        if stage == "researcher":
            prompt = prompt_template.format(
                resume=TEST_RESUME,
                job_ad=TEST_JOB_AD
            )
        elif stage == "drafter":
            prompt = prompt_template.format(star_data=json.dumps(TEST_STAR_DATA))
        else:  # star_editor
            prompt = prompt_template.format(star_data=json.dumps(TEST_STAR_DATA))
        
        print(f"\n🔍 Testing {model_name} - {stage.upper()} stage...")
        print(f"Model: {model['slug']}")
        print(f"Prompt length: {len(prompt)} chars")
        
        # Call model with stage name for file saving
        result = await self.call_model(model["slug"], prompt, stage=stage)
        
        if result["success"]:
            cost = self.calculate_cost(
                result["input_tokens"], 
                result["output_tokens"], 
                model_name
            )
            quality = self.evaluate_quality(result["content"], stage)
            
            print(f"✅ Success!")
            print(f"   Response time: {result['response_time']:.2f}s")
            print(f"   Tokens: {result['input_tokens']} in, {result['output_tokens']} out")
            print(f"   Cost: ${cost:.6f}")
            print(f"   Quality score: {quality:.2f}/1.0")
            print(f"   Saved to: {result.get('saved_file', 'N/A')}")
            
            return TestResult(
                model_name=model_name,
                model_slug=model["slug"],
                stage=stage,
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                cost=cost,
                response_time=result["response_time"],
                success=True,
                quality_score=quality
            )
        else:
            print(f"❌ Failed: {result['error']}")
            return TestResult(
                model_name=model_name,
                model_slug=model["slug"],
                stage=stage,
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                response_time=result["response_time"],
                success=False,
                quality_score=0.0,
                error=result["error"]
            )
    
    async def run_full_comparison(self) -> Dict[str, Any]:
        """Run complete comparison across all models and stages."""
        print("=" * 80)
        print("LLM COMPARISON TEST - 3 STAGE PIPELINE")
        print("=" * 80)
        print(f"Date: {datetime.now().isoformat()}")
        print("\nTesting Models:")
        for name, info in MODELS.items():
            print(f"  • {name} ({info['slug']}) - ${info['input_price']:.6f}/$${info['output_price']:.6f} per 1K tokens")
        
        all_results = []
        
        # Test each model across all stages
        for model_name in MODELS.keys():
            print(f"\n{'='*60}")
            print(f"TESTING: {model_name}")
            print(f"{'='*60}")
            
            for stage in ["researcher", "drafter", "star_editor"]:
                result = await self.run_stage_test(model_name, stage, {})
                all_results.append(result)
                await asyncio.sleep(1)  # Rate limiting
        
        # Generate comprehensive report
        report = self.generate_report(all_results)
        return report
    
    def generate_report(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "models_tested": list(MODELS.keys()),
            "stages": ["researcher", "drafter", "star_editor"],
            "results": [],
            "summary": {},
            "recommendations": []
        }
        
        # Group results by model
        model_results = {}
        for result in results:
            if result.model_name not in model_results:
                model_results[result.model_name] = []
            model_results[result.model_name].append(result)
        
        # Calculate per-model metrics
        for model_name, model_results_list in model_results.items():
            total_cost = sum(r.cost for r in model_results_list if r.success)
            total_tokens = sum(r.input_tokens + r.output_tokens for r in model_results_list if r.success)
            avg_response_time = sum(r.response_time for r in model_results_list if r.success) / len([r for r in model_results_list if r.success])
            avg_quality = sum(r.quality_score for r in model_results_list if r.success) / len([r for r in model_results_list if r.success])
            success_rate = len([r for r in model_results_list if r.success]) / len(model_results_list)
            
            report["results"].append({
                "model_name": model_name,
                "model_slug": MODELS[model_name]["slug"],
                "total_cost": total_cost,
                "total_tokens": total_tokens,
                "avg_response_time": avg_response_time,
                "avg_quality_score": avg_quality,
                "success_rate": success_rate,
                "cost_per_token": total_cost / total_tokens if total_tokens > 0 else 0,
                "stages": [{
                    "stage": r.stage,
                    "cost": r.cost,
                    "tokens": r.input_tokens + r.output_tokens,
                    "response_time": r.response_time,
                    "quality_score": r.quality_score,
                    "success": r.success,
                    "error": r.error
                } for r in model_results_list]
            })
        
        # Generate summary
        successful_results = [r for r in results if r.success]
        if successful_results:
            best_cost = min([r.cost for r in successful_results])
            best_quality = max([r.quality_score for r in successful_results])
            fastest = min([r.response_time for r in successful_results])
            
            report["summary"] = {
                "total_tests": len(results),
                "successful_tests": len(successful_results),
                "overall_success_rate": len(successful_results) / len(results),
                "best_cost_model": min(report["results"], key=lambda x: x["total_cost"])["model_name"],
                "best_quality_model": max(report["results"], key=lambda x: x["avg_quality_score"])["model_name"],
                "fastest_model": min(report["results"], key=lambda x: x["avg_response_time"])["model_name"],
                "most_cost_efficient": min(report["results"], key=lambda x: x["cost_per_token"])["model_name"]
            }
        
        # Generate recommendations
        report["recommendations"] = self.generate_recommendations(report)
        
        return report
    
    def generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if not report["results"]:
            return ["No successful tests to generate recommendations"]
        
        summary = report["summary"]
        results = report["results"]
        
        # Cost-focused recommendation
        cost_leader = next(r for r in results if r["model_name"] == summary["best_cost_model"])
        recommendations.append(
            f"💰 COST LEADER: {cost_leader['model_name']} at ${cost_leader['total_cost']:.6f} total "
            f"(${cost_leader['cost_per_token']:.8f} per token)"
        )
        
        # Quality-focused recommendation
        quality_leader = next(r for r in results if r["model_name"] == summary["best_quality_model"])
        recommendations.append(
            f"🏆 QUALITY LEADER: {quality_leader['model_name']} with {quality_leader['avg_quality_score']:.2f}/1.0 average quality"
        )
        
        # Speed-focused recommendation
        speed_leader = next(r for r in results if r["model_name"] == summary["fastest_model"])
        recommendations.append(
            f"⚡ SPEED LEADER: {speed_leader['model_name']} at {speed_leader['avg_response_time']:.2f}s average response time"
        )
        
        # Overall recommendation
        cost_efficient = next(r for r in results if r["model_name"] == summary["most_cost_efficient"])
        recommendations.append(
            f"🎯 BEST OVERALL: {cost_efficient['model_name']} - Best cost efficiency at ${cost_efficient['cost_per_token']:.8f} per token"
        )
        
        # Cost comparison for 1000 tests
        total_cost_1000 = {}
        for r in results:
            if r["model_name"] not in total_cost_1000:
                total_cost_1000[r["model_name"]] = 0
            total_cost_1000[r["model_name"]] += r["total_cost"]
        
        recommendations.append("\n📊 COST FOR 1000 PIPELINE RUNS:")
        for model_name, cost in sorted(total_cost_1000.items(), key=lambda x: x[1]):
            recommendations.append(f"   {model_name}: ${cost * 1000:.2f}")
        
        return recommendations

async def main():
    """Main execution function."""
    # Get API key from environment
    import os
    api_key = os.getenv("OPENROUTER_API_KEY_1") or os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("❌ ERROR: No OpenRouter API key found!")
        print("Please set OPENROUTER_API_KEY or OPENROUTER_API_KEY_1 environment variable")
        print("You can get one from: https://openrouter.ai/keys")
        return
    
    # Initialize test
    test = LLMComparisonTest(api_key)
    
    # Run comparison
    try:
        report = await test.run_full_comparison()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"llm_comparison_report_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE REPORT")
        print("=" * 80)
        
        # Print summary
        summary = report["summary"]
        print(f"\n📊 SUMMARY:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Success Rate: {summary['overall_success_rate']:.1%}")
        print(f"   Best Cost: {summary['best_cost_model']}")
        print(f"   Best Quality: {summary['best_quality_model']}")
        print(f"   Fastest: {summary['fastest_model']}")
        print(f"   Most Cost Efficient: {summary['most_cost_efficient']}")
        
        # Print detailed results
        print(f"\n📈 DETAILED RESULTS:")
        for result in report["results"]:
            print(f"\n   {result['model_name']} ({result['model_slug']}):")
            print(f"      Total Cost: ${result['total_cost']:.6f}")
            print(f"      Total Tokens: {result['total_tokens']}")
            print(f"      Avg Response Time: {result['avg_response_time']:.2f}s")
            print(f"      Avg Quality: {result['avg_quality_score']:.2f}/1.0")
            print(f"      Success Rate: {result['success_rate']:.1%}")
            print(f"      Cost per Token: ${result['cost_per_token']:.8f}")
        
        # Print recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"   {rec}")
        
        print(f"\n✅ Report saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())