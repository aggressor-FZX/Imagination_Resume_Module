#!/usr/bin/env python3
"""
Cost Calculation for Imaginator 3-Stage Pipeline
Compute actual LLM costs based on token usage and OpenRouter pricing
"""

# Pricing from pipeline_config.py (USD per 1K tokens)
PRICING = {
    "perplexity/sonar-pro": {"input": 0.003, "output": 0.015},  # Search model
    "perplexity/sonar": {"input": 0.001, "output": 0.001},
    "google/gemini-2.0-flash-001": {"input": 0.00025, "output": 0.0005},
    "google/gemini-2.0-pro": {"input": 0.00125, "output": 0.0025},
    "anthropic/claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
    "anthropic/claude-3-haiku": {"input": 0.00025, "output": 0.00125}
}

# Token estimates based on actual prompts and responses
# Average resume: 2 pages = ~1000 words = ~1500 tokens
# Average job ad: 500 words = ~750 tokens
# JSON responses: ~500-1000 tokens

TOKEN_ESTIMATES = {
    "researcher": {
        "model": "perplexity/sonar-pro",
        "input_tokens": 1900,  # System: 400 + Job ad: 750 + Experiences: 750 = 1900
        "output_tokens": 500   # JSON response
    },
    "drafter": {
        "model": "anthropic/claude-3.5-sonnet",
        "input_tokens": 4800,  # System: 800 + Experiences: 1500 + Job ad: 750 + Research: 750 + Golden bullets: 1000 = 4800
        "output_tokens": 1000  # JSON with rewritten bullets
    },
    "star_editor": {
        "model": "google/gemini-2.0-flash-001",
        "input_tokens": 3400,  # System: 600 + Drafted bullets: 1000 + Research: 750 + Instructions: 1050 = 3400
        "output_tokens": 800   # Final markdown resume
    }
}

def calculate_stage_cost(model, input_tokens, output_tokens):
    """Calculate cost for a single stage."""
    if model not in PRICING:
        # Default pricing
        input_cost = (input_tokens / 1000.0) * 0.0005
        output_cost = (output_tokens / 1000.0) * 0.0015
    else:
        pricing = PRICING[model]
        input_cost = (input_tokens / 1000.0) * pricing["input"]
        output_cost = (output_tokens / 1000.0) * pricing["output"]
    
    total_cost = input_cost + output_cost
    return total_cost, input_cost, output_cost

def calculate_pipeline_cost():
    """Calculate total cost for the 3-stage pipeline."""
    total_cost = 0.0
    stage_costs = {}
    
    print("=" * 60)
    print("IMAGINATOR 3-STAGE PIPELINE COST ANALYSIS")
    print("=" * 60)
    
    for stage_name, stage_data in TOKEN_ESTIMATES.items():
        model = stage_data["model"]
        input_tokens = stage_data["input_tokens"]
        output_tokens = stage_data["output_tokens"]
        
        stage_total, input_cost, output_cost = calculate_stage_cost(
            model, input_tokens, output_tokens
        )
        
        stage_costs[stage_name] = {
            "total": stage_total,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model
        }
        
        total_cost += stage_total
        
        print(f"\n{stage_name.upper()} Stage:")
        print(f"  Model: {model}")
        print(f"  Input tokens: {input_tokens:,} (${input_cost:.6f})")
        print(f"  Output tokens: {output_tokens:,} (${output_cost:.6f})")
        print(f"  Stage total: ${stage_total:.6f}")
    
    print("\n" + "=" * 60)
    print(f"TOTAL PIPELINE COST: ${total_cost:.6f}")
    print(f"TOTAL PIPELINE COST (rounded): ${total_cost:.4f}")
    print("=" * 60)
    
    # Calculate with 20% buffer for errors, retries, etc.
    buffered_cost = total_cost * 1.2
    print(f"\nWith 20% buffer (errors, retries): ${buffered_cost:.6f}")
    print(f"With 20% buffer (rounded): ${buffered_cost:.4f}")
    
    return total_cost, stage_costs

def calculate_other_service_costs():
    """Calculate costs for other services in the pipeline."""
    print("\n" + "=" * 60)
    print("OTHER SERVICE COSTS")
    print("=" * 60)
    
    # Document Reader Service (PDF/DOCX parsing)
    # Typically uses local ML models, minimal external API cost
    doc_reader_cost = 0.001  # $0.001 per document (local processing)
    
    # Hermes Service (Skill extraction)
    # Uses local NLP models, minimal external API cost
    hermes_cost = 0.002  # $0.002 per resume
    
    # FastSVM Service (Skill classification)
    # Uses local ML models, minimal external API cost
    fastsvm_cost = 0.001  # $0.001 per resume
    
    # Job Search Service (HasData API)
    # External API: $49/month for 200,000 credits = $0.000245 per credit
    # 5 credits per job search = $0.001225 per search
    # Average 5 searches per analysis = $0.006125
    job_search_cost = 0.006125
    
    other_costs = {
        "document_reader": doc_reader_cost,
        "hermes": hermes_cost,
        "fastsvm": fastsvm_cost,
        "job_search": job_search_cost
    }
    
    total_other = sum(other_costs.values())
    
    for service, cost in other_costs.items():
        print(f"{service.replace('_', ' ').title()}: ${cost:.6f}")
    
    print(f"\nTotal Other Services: ${total_other:.6f}")
    print(f"Total Other Services (rounded): ${total_other:.4f}")
    
    return total_other, other_costs

def calculate_storage_costs():
    """Calculate storage costs for file uploads."""
    print("\n" + "=" * 60)
    print("STORAGE AND FILE UPLOAD COSTS")
    print("=" * 60)
    
    # File upload limits: 10MB per file (from config.py)
    max_file_size_mb = 10
    
    # Storage costs (Render free tier includes some storage)
    # Backblaze B2: $0.01/GB/month
    # Average resume file: 500KB (0.5MB)
    # Storage per file per month: 0.5MB * $0.01/GB = $0.000005
    
    storage_cost_per_file_per_month = 0.000005
    
    # Processing storage (temporary during analysis)
    # Typically held for 24 hours then deleted
    processing_storage_cost = 0.000001  # Negligible
    
    print(f"Max file size: {max_file_size_mb}MB")
    print(f"Storage cost per file per month: ${storage_cost_per_file_per_month:.8f}")
    print(f"Processing storage cost: ${processing_storage_cost:.8f}")
    
    total_storage = storage_cost_per_file_per_month + processing_storage_cost
    
    return total_storage

def main():
    """Main cost analysis function."""
    print("\n" + "=" * 80)
    print("COMPLETE COST ANALYSIS FOR COGITO METRIC RESUME PIPELINE")
    print("=" * 80)
    
    # 1. Imaginator 3-stage pipeline costs
    imaginator_cost, stage_costs = calculate_pipeline_cost()
    
    # 2. Other service costs
    other_cost, other_costs = calculate_other_service_costs()
    
    # 3. Storage costs
    storage_cost = calculate_storage_costs()
    
    # 4. Total cost per analysis
    total_cost_per_analysis = imaginator_cost + other_cost + storage_cost
    
    print("\n" + "=" * 60)
    print("TOTAL COST PER ANALYSIS")
    print("=" * 60)
    print(f"Imaginator Pipeline: ${imaginator_cost:.6f}")
    print(f"Other Services: ${other_cost:.6f}")
    print(f"Storage: ${storage_cost:.8f}")
    print("-" * 40)
    print(f"TOTAL: ${total_cost_per_analysis:.6f}")
    print(f"TOTAL (rounded): ${total_cost_per_analysis:.4f}")
    
    # 5. Compare with current pricing
    current_price = 0.38  # $0.38 per analysis (33 credits)
    margin = current_price - total_cost_per_analysis
    margin_percentage = (margin / current_price) * 100
    
    print("\n" + "=" * 60)
    print("PRICING ANALYSIS")
    print("=" * 60)
    print(f"Current price per analysis: ${current_price:.2f}")
    print(f"Actual cost per analysis: ${total_cost_per_analysis:.4f}")
    print(f"Gross margin: ${margin:.4f}")
    print(f"Margin percentage: {margin_percentage:.1f}%")
    
    # 6. Credit package analysis
    print("\n" + "=" * 60)
    print("CREDIT PACKAGE ANALYSIS")
    print("=" * 60)
    
    credit_packages = {
        "$5": 435,    # credits (0% bonus)
        "$10": 914,   # credits (+5% bonus)
        "$25": 2393   # credits (+10% bonus)
    }
    
    for price, credits in credit_packages.items():
        analyses = credits / 33  # 33 credits per analysis
        revenue = float(price.replace('$', ''))
        cost = analyses * total_cost_per_analysis
        profit = revenue - cost
        profit_margin = (profit / revenue) * 100
        
        print(f"\n{price} package ({credits} credits):")
        print(f"  Analyses: {analyses:.1f}")
        print(f"  Revenue: ${revenue:.2f}")
        print(f"  Cost: ${cost:.4f}")
        print(f"  Profit: ${profit:.4f}")
        print(f"  Margin: {profit_margin:.1f}%")
    
    # 7. Monthly projections
    print("\n" + "=" * 60)
    print("MONTHLY PROJECTIONS (100 users)")
    print("=" * 60)
    
    users_per_month = 100
    analyses_per_user = 3  # Average 3 analyses per user
    total_analyses = users_per_month * analyses_per_user
    
    monthly_revenue = users_per_month * 15  # Average $15 per user
    monthly_cost = total_analyses * total_cost_per_analysis
    monthly_profit = monthly_revenue - monthly_cost
    monthly_margin = (monthly_profit / monthly_revenue) * 100
    
    print(f"Users per month: {users_per_month}")
    print(f"Analyses per user: {analyses_per_user}")
    print(f"Total analyses: {total_analyses}")
    print(f"Monthly revenue: ${monthly_revenue:.2f}")
    print(f"Monthly cost: ${monthly_cost:.4f}")
    print(f"Monthly profit: ${monthly_profit:.4f}")
    print(f"Monthly margin: {monthly_margin:.1f}%")
    
    return total_cost_per_analysis

if __name__ == "__main__":
    main()