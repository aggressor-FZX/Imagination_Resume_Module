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
