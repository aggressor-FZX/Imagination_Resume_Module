#!/usr/bin/env python3
"""
Verify OpenRouter Pricing
Check actual OpenRouter pricing for models used in the pipeline
"""

import requests
import json

def check_openrouter_pricing():
    """Check OpenRouter pricing via their API or website."""
    
    # Models used in our pipeline
    models_to_check = [
        "perplexity/sonar-pro",
        "perplexity/sonar",
        "google/gemini-2.0-flash-001",
        "google/gemini-2.0-pro",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3-haiku",
        "deepseek/deepseek-chat-v3.1",
        "openai/gpt-3.5-turbo"
    ]
    
    print("=" * 80)
    print("OPENROUTER PRICING VERIFICATION")
    print("=" * 80)
    print("\nNote: This script attempts to check pricing via OpenRouter API.")
    print("Actual pricing may vary and should be verified on openrouter.ai")
    print("\n" + "=" * 80)
    
    # Try to get model list from OpenRouter
    try:
        print("\nAttempting to fetch model list from OpenRouter...")
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={
                "Content-Type": "application/json"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            models = data.get('data', [])
            
            print(f"Found {len(models)} models on OpenRouter")
            
            # Look for our models
            found_models = {}
            for model in models:
                model_id = model.get('id')
                if model_id in models_to_check:
                    pricing = model.get('pricing', {})
                    found_models[model_id] = {
                        'name': model.get('name', 'Unknown'),
                        'pricing': pricing,
                        'context_length': model.get('context_length', 'Unknown')
                    }
            
            if found_models:
                print("\nFound pricing for our models:")
                for model_id, info in found_models.items():
                    pricing = info['pricing']
                    print(f"\n{model_id}:")
                    print(f"  Name: {info['name']}")
                    print(f"  Context: {info['context_length']}")
                    if pricing:
                        print(f"  Pricing: {json.dumps(pricing, indent=4)}")
                    else:
                        print(f"  Pricing: Not available in API response")
            else:
                print("\nCould not find our specific models in the API response.")
                print("This is normal - the public API might not include pricing.")
                
        else:
            print(f"Failed to fetch models: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"Error fetching model list: {e}")
    
    # Show current pricing from our config
    print("\n" + "=" * 80)
    print("CURRENT PRICING IN OUR CONFIG (pipeline_config.py)")
    print("=" * 80)
    
    current_pricing = {
        "perplexity/sonar-pro": {"input": 0.003, "output": 0.015},
        "perplexity/sonar": {"input": 0.001, "output": 0.001},
        "google/gemini-2.0-flash-001": {"input": 0.00025, "output": 0.0005},
        "google/gemini-2.0-pro": {"input": 0.00125, "output": 0.0025},
        "anthropic/claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
        "anthropic/claude-3-haiku": {"input": 0.00025, "output": 0.00125}
    }
    
    for model, pricing in current_pricing.items():
        print(f"\n{model}:")
        print(f"  Input: ${pricing['input']}/1K tokens")
        print(f"  Output: ${pricing['output']}/1K tokens")
        cost_per_1k_tokens = pricing['input'] + pricing['output']
        print(f"  Total (1K in + 1K out): ${cost_per_1k_tokens}")
    
    # Compare with known market rates
    print("\n" + "=" * 80)
    print("KNOWN MARKET RATES (Approximate, Jan 2025)")
    print("=" * 80)
    
    market_rates = {
        "Google Gemini 2.0 Flash": {"input": 0.00025, "output": 0.0005, "source": "Google Cloud"},
        "Google Gemini 2.0 Pro": {"input": 0.00125, "output": 0.0025, "source": "Google Cloud"},
        "Anthropic Claude 3.5 Sonnet": {"input": 0.003, "output": 0.015, "source": "Anthropic"},
        "Anthropic Claude 3 Haiku": {"input": 0.00025, "output": 0.00125, "source": "Anthropic"},
        "DeepSeek Chat V3.1": {"input": 0.00014, "output": 0.00028, "source": "OpenRouter"},
        "GPT-3.5 Turbo": {"input": 0.0005, "output": 0.0015, "source": "OpenAI"},
        "Perplexity Sonar Pro": {"input": "Unknown", "output": "Unknown", "source": "Search model, likely higher"},
    }
    
    for model_name, info in market_rates.items():
        print(f"\n{model_name}:")
        print(f"  Input: ${info['input']}/1K tokens" if isinstance(info['input'], (int, float)) else f"  Input: {info['input']}")
        print(f"  Output: ${info['output']}/1K tokens" if isinstance(info['output'], (int, float)) else f"  Output: {info['output']}")
        print(f"  Source: {info['source']}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n1. **Verify Perplexity Sonar Pro Pricing**:")
    print("   - Check openrouter.ai/models for exact pricing")
    print("   - Search models typically cost more than regular LLMs")
    print("   - Consider if search capability is necessary for Researcher stage")
    
    print("\n2. **Consider Cost Optimization**:")
    print("   - Researcher: Switch to Gemini Flash (save ~$0.0076/analysis)")
    print("   - Drafter: Consider Claude Haiku vs Sonnet (save ~$0.0206/analysis)")
    print("   - Quality impact should be tested")
    
    print("\n3. **Monitor Actual Usage**:")
    print("   - Implement token tracking in production")
    print("   - Log actual costs per analysis")
    print("   - Adjust estimates based on real data")
    
    print("\n4. **Pricing Strategy**:")
    print("   - Current $0.38/analysis provides 85-96% margins")
    print("   - Room for price optimization if needed")
    print("   - Consider tiered pricing (basic vs premium)")

if __name__ == "__main__":
    check_openrouter_pricing()