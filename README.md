# Enhanced Imaginator - AI-Powered Resume Analysis with Structured Skill Processing

Advanced career development tool that integrates multiple AI repositories for creative, confidence-weighted gap analysis and personalized development recommendations.

## 🚀 Enhanced Features

### Structured Skill Analysis
- **Confidence-Based Filtering**: Processes skills with confidence scores from FastSVM and Hermes repositories
- **Domain-Aware Processing**: Incorporates industry-specific insights and market alignment data
- **Creative Gap Analysis**: Generates personalized development plans using structured data

### Repository Integration
- **FastSVM_Skill_Title_Extraction**: High-confidence skill extraction with SVM models
- **Hermes**: Domain insights, market alignment, and strategic recommendations
- **Resume_Document_Loader**: Structured text extraction and preprocessing

## 📋 Files

- `imaginator_flow.py`: Enhanced main script with structured data processing
- `requirements.txt`: Dependencies (openai, python-dotenv)
- `.env`: Environment file with API keys
- `sample_resume.txt`: Test resume text
- `sample_skills.json`: Sample structured skills data
- `sample_insights.json`: Sample domain insights from Hermes
- `sample_job_ad.txt`: Sample job description for testing

## 🛠 Setup

1. Activate virtual environment:
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Configure API keys in `.env`:
   ```bash
   OPENAI_API_KEY=your_key_here
   ```

## 🎯 Usage

### Basic Usage (Keyword-based)
```bash
python imaginator_flow.py --resume sample_resume.txt --target_job_ad "Senior Software Engineer role description"
```

### Enhanced Usage (Structured Data)
```bash
python imaginator_flow.py \
  --parsed_resume_text "$(cat sample_resume.txt)" \
  --extracted_skills_json sample_skills.json \
  --domain_insights_json sample_insights.json \
  --target_job_ad "$(cat sample_job_ad.txt)" \
  --confidence_threshold 0.8
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--resume` | Path to resume file | - |
| `--parsed_resume_text` | Resume text from Resume_Document_Loader | - |
| `--extracted_skills_json` | Skills data from FastSVM/Hermes | - |
| `--domain_insights_json` | Domain insights from Hermes | - |
| `--target_job_ad` | Job description (required) | - |
| `--confidence_threshold` | Minimum skill confidence | 0.7 |

## 📊 Output Format

```json
{
  "experiences": [...],
  "aggregate_skills": ["python", "aws", "docker"],
  "processed_skills": {
    "high_confidence_skills": ["python", "aws"],
    "skill_confidences": {"python": 0.95, "aws": 0.92},
    "categories": {...}
  },
  "domain_insights": {
    "domain": "tech",
    "insights": {
      "strengths": [...],
      "gaps": [...],
      "recommendations": [...]
    }
  },
  "gap_analysis": "Creative development recommendations..."
}
```

## Run Metrics, Token Usage, and Cost Estimation

The module now appends a `run_metrics` block to the final JSON output so callers can reliably parse token usage, per-call breakdowns, failures, and an estimated cost for the run.

Example output tail:

```json
{
  "suggested_experiences": { /* ... */ },
  "run_metrics": {
    "calls": [
      {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "prompt_tokens": 1499,
        "completion_tokens": 629,
        "total_tokens": 2128,
        "estimated_cost_usd": 0.001693
      },
      {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "prompt_tokens": 511,
        "completion_tokens": 366,
        "total_tokens": 877,
        "estimated_cost_usd": 0.000805
      }
    ],
    "total_prompt_tokens": 2853,
    "total_completion_tokens": 1201,
    "total_tokens": 4054,
    "estimated_cost_usd": 0.003228,
    "failures": []
  }
}
```

### Pricing configuration (override via environment)
Default per-1K token USD rates can be overridden:

- `OPENAI_PRICE_INPUT_PER_1K` (default: `0.0005`)
- `OPENAI_PRICE_OUTPUT_PER_1K` (default: `0.0015`)
- `ANTHROPIC_PRICE_INPUT_PER_1K` (default: `0.003`)
- `ANTHROPIC_PRICE_OUTPUT_PER_1K` (default: `0.015`)

Totals are computed by summing per-call estimated cost: `prompt_tokens/1000 * input_rate + completion_tokens/1000 * output_rate`.

### Failure and fallback reporting
- All provider call failures during retries are recorded in `run_metrics.failures` with `attempt`, `provider`, and `error` for post-mortem.
- JSON decode fallbacks in `run_generation` and `run_criticism` are explicitly logged to stderr.
- Progress logs print which provider/model was used per step.

### Parsing the cost in downstream tooling
Example (Python):

```python
import json
result = json.loads(output_text)
run_metrics = result.get("run_metrics", {})
estimated_cost = run_metrics.get("estimated_cost_usd", 0.0)
```

## 🔄 Core Functionality

1. **Skill Processing**: Filters and prioritizes skills by confidence scores
2. **Domain Integration**: Incorporates industry-specific insights and market data
3. **Creative Analysis**: Generates personalized, actionable development plans
4. **Confidence Weighting**: Prioritizes high-confidence skills for recommendations

## 🧪 Testing

Test with sample data:
```bash
python imaginator_flow.py \
  --parsed_resume_text "Senior developer with Python, AWS experience" \
  --extracted_skills_json sample_skills.json \
  --domain_insights_json sample_insights.json \
  --target_job_ad "Full-stack developer role with React, Node.js, cloud experience"
```
