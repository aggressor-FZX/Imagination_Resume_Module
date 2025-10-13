# Generative Resume Co-Writer - Input/Output Specification

## Overview

The Generative Resume Co-Writer is a standalone Python application that analyzes resumes against target job descriptions using Large Language Models (LLMs) to provide personalized career development recommendations. The system processes resume data through a three-stage pipeline: Analysis, Generation, and Criticism.

## System Architecture

The application consists of three main processing stages:

1. **Analysis Stage** (`run_analysis`): Extracts skills, experiences, and performs gap analysis
2. **Generation Stage** (`run_generation`): Creates targeted resume improvement suggestions
3. **Criticism Stage** (`run_criticism`): Refines suggestions through adversarial review

## Input Specifications

### Required Inputs

| Input | Type | Description | Example |
|-------|------|-------------|---------|
| `resume_text` | String | Raw resume text content | "John Doe\nSoftware Engineer\n..." |
| `job_ad` | String | Target job description text | "Senior Python Developer\nRequirements: Python, Django..." |

### Optional Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `extracted_skills_json` | File Path | None | JSON file with structured skills data (confidence scores) |
| `domain_insights_json` | File Path | None | JSON file with domain-specific insights |
| `confidence_threshold` | Float | 0.7 | Minimum confidence score for skill filtering (0.0-1.0) |

### Command Line Interface

```bash
python imaginator_flow.py \
  --target_job_ad "Senior Software Engineer position description" \
  --resume sample_resume.txt \
  [--extracted_skills_json skills.json] \
  [--domain_insights_json insights.json] \
  [--confidence_threshold 0.8]
```

**Alternative input methods:**
- `--parsed_resume_text`: Direct text input instead of file
- `--resume`: File path to resume text file

### Structured Data Formats

#### Extracted Skills JSON Format
```json
{
  "title": "Software Engineer",
  "canonical_title": "Software Engineer",
  "skills": [
    {"skill": "Python", "confidence": 0.95},
    {"skill": "JavaScript", "confidence": 0.87},
    {"skill": "SQL", "confidence": 0.72}
  ]
}
```

#### Domain Insights JSON Format
```json
{
  "domain": "technology",
  "market_trends": ["cloud computing", "AI/ML"],
  "skill_priorities": {
    "high": ["Python", "AWS"],
    "medium": ["JavaScript", "Docker"],
    "low": ["PHP", "jQuery"]
  }
}
```

## Output Specifications

### Complete Output Schema

The system produces a validated JSON output conforming to this schema:

```json
{
  "type": "object",
  "properties": {
    "experiences": {
      "type": "array",
      "description": "Parsed work experiences from resume",
      "items": {
        "type": "object",
        "properties": {
          "title_line": {"type": "string", "description": "Job title and company"},
          "skills": {"type": "array", "items": {"type": "string"}, "description": "Skills mentioned in this experience"},
          "snippet": {"type": "string", "description": "Relevant experience description"}
        },
        "required": ["title_line", "skills", "snippet"]
      }
    },
    "aggregate_skills": {
      "type": "array",
      "description": "All unique skills found across the resume",
      "items": {"type": "string"}
    },
    "processed_skills": {
      "type": "object",
      "description": "Structured skill analysis with confidence scores and categories"
    },
    "domain_insights": {
      "type": "object",
      "description": "Industry-specific insights and market context"
    },
    "gap_analysis": {
      "type": "string",
      "description": "Narrative analysis of skill gaps and development opportunities"
    },
    "suggested_experiences": {
      "type": "object",
      "description": "Refined resume improvement suggestions",
      "properties": {
        "bridging_gaps": {
          "type": "array",
          "description": "Suggestions to address identified skill gaps",
          "items": {
            "type": "object",
            "properties": {
              "skill_focus": {"type": "string"},
              "refined_suggestions": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["skill_focus", "refined_suggestions"]
          }
        },
        "metric_improvements": {
          "type": "array",
          "description": "Suggestions to improve quantifiable metrics",
          "items": {
            "type": "object",
            "properties": {
              "skill_focus": {"type": "string"},
              "refined_suggestions": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["skill_focus", "refined_suggestions"]
          }
        }
      },
      "required": ["bridging_gaps", "metric_improvements"]
    },
    "run_metrics": {
      "type": "object",
      "description": "Usage statistics and performance metrics",
      "properties": {
        "calls": {"type": "array", "description": "Per-call usage data"},
        "total_prompt_tokens": {"type": "number"},
        "total_completion_tokens": {"type": "number"},
        "total_tokens": {"type": "number"},
        "estimated_cost_usd": {"type": "number"},
        "failures": {"type": "array", "description": "List of API call failures"}
      }
    }
  },
  "required": ["experiences", "aggregate_skills", "processed_skills", "domain_insights", "gap_analysis", "suggested_experiences"]
}
```

### Output Sections Explained

#### 1. Experiences
Parsed work experiences with associated skills:
```json
[
  {
    "title_line": "Software Engineer at TechCorp (2020-2023)",
    "skills": ["Python", "Django", "PostgreSQL"],
    "snippet": "Developed web applications using Python and Django framework..."
  }
]
```

#### 2. Aggregate Skills
Flat list of all unique skills found:
```json
["Python", "JavaScript", "SQL", "AWS", "Docker", "Git"]
```

#### 3. Processed Skills
Structured skill analysis with confidence scores and categorization:
```json
{
  "high_confidence": ["Python", "JavaScript"],
  "medium_confidence": ["SQL", "Git"],
  "inferred_skills": ["Django", "Flask"],
  "competencies": {
    "technical": ["Python", "JavaScript", "SQL"],
    "soft": ["Communication", "Leadership"]
  }
}
```

#### 4. Domain Insights
Industry context and market intelligence:
```json
{
  "primary_domain": "Software Development",
  "market_demand": ["Cloud Computing", "AI/ML"],
  "emerging_skills": ["Kubernetes", "MLOps"],
  "skill_gap_priority": "high"
}
```

#### 5. Gap Analysis
Narrative summary of skill gaps and development recommendations:
```
"Your resume demonstrates strong Python development skills, but shows gaps in cloud architecture and DevOps practices that are increasingly required for senior software engineering roles. Consider focusing on AWS certification and container orchestration skills."
```

#### 6. Suggested Experiences
Refined, actionable resume improvement suggestions:
```json
{
  "bridging_gaps": [
    {
      "skill_focus": "Cloud Architecture",
      "refined_suggestions": [
        "Led migration of legacy monolithic application to microservices architecture on AWS ECS",
        "Implemented Infrastructure as Code using Terraform and AWS CloudFormation"
      ]
    }
  ],
  "metric_improvements": [
    {
      "skill_focus": "Performance Optimization",
      "refined_suggestions": [
        "Optimized database queries resulting in 40% reduction in response times",
        "Implemented caching strategy reducing server load by 60%"
      ]
    }
  ]
}
```

#### 7. Run Metrics
Usage statistics for monitoring and cost tracking:
```json
{
  "calls": [
    {
      "provider": "openai",
      "model": "gpt-4",
      "prompt_tokens": 1500,
      "completion_tokens": 800,
      "cost_usd": 0.045,
      "duration_ms": 2500
    }
  ],
  "total_prompt_tokens": 1500,
  "total_completion_tokens": 800,
  "total_tokens": 2300,
  "estimated_cost_usd": 0.045,
  "failures": []
}
```

## Processing Pipeline

### Stage 1: Analysis (`run_analysis`)
**Input:** Resume text, job description, optional structured data
**Process:** 
- Parse work experiences and skills
- Apply confidence filtering if structured data provided
- Generate gap analysis using LLM
- Infer implied skills from skill adjacency data
**Output:** Experiences, skills analysis, gap analysis

### Stage 2: Generation (`run_generation`)
**Input:** Analysis results, job description
**Process:**
- Extract skill gaps from analysis
- Generate targeted improvement suggestions
- Create metric-focused recommendations
**Output:** Gap bridging and metric improvement suggestions

### Stage 3: Criticism (`run_criticism`)
**Input:** Generated suggestions, job description
**Process:**
- Apply adversarial review to suggestions
- Refine wording and improve specificity
- Validate alignment with job requirements
**Output:** Refined, production-ready suggestions

## Error Handling & Graceful Degradation

The system implements comprehensive error handling:

- **API Failures:** Automatic fallback from OpenAI to Anthropic
- **JSON Parsing:** Safe parsing with fallback structures
- **Network Issues:** Exponential backoff retry logic
- **Schema Validation:** Ensures output conforms to expected format
- **Partial Failures:** Continues processing with degraded but valid output

## Configuration & Environment

### Required Environment Variables
- `OPENAI_API_KEY`: Primary LLM provider
- `ANTHROPIC_API_KEY`: Fallback LLM provider (optional but recommended)

### Optional Environment Variables
- `OPENAI_PRICE_INPUT_PER_1K`: Custom pricing for OpenAI input tokens
- `OPENAI_PRICE_OUTPUT_PER_1K`: Custom pricing for OpenAI output tokens
- `ANTHROPIC_PRICE_INPUT_PER_1K`: Custom pricing for Anthropic input tokens
- `ANTHROPIC_PRICE_OUTPUT_PER_1K`: Custom pricing for Anthropic output tokens

## Dependencies

- `openai`: OpenAI API client
- `anthropic`: Anthropic API client
- `python-dotenv`: Environment variable management
- `jsonschema`: JSON schema validation
- `pytest`: Testing framework (development)

## File Structure

```
imaginator_flow.py          # Main application
requirements.txt            # Python dependencies
sample_resume.txt           # Sample input resume
sample_job_ad.txt           # Sample job description
skill_adjacency.json        # Skill relationship data
verb_competency.json        # Competency mapping data
tests/                      # Test suite
  ├── test_imaginator_flow_unit.py    # Unit tests
  └── test_imaginator_flow_e2e.py     # End-to-end tests
README.md                   # Documentation
```

## Backend System Integration

### Calling the Container

To integrate with the Generative Resume Co-Writer, backend systems should send a POST request to the `/analyze` endpoint of the containerized FastAPI application.

**Endpoint**: `http://<container_ip>:8000/analyze`

**Method**: `POST`

**Headers**:
- `Content-Type`: `application/json`

**Request Body**:
The request body should be a JSON object with the following structure:

```json
{
  "resume_text": "...",
  "job_ad": "...",
  "extracted_skills_json": "...",
  "domain_insights_json": "...",
  "confidence_threshold": 0.7
}
```

### Expected Response

The system will respond with a JSON object containing the analysis, generation, and criticism results.

**Success Response (200 OK)**:
```json
{
  "analysis": {
    "experiences": [...],
    "aggregate_skills": [...],
    "processed_skills": {...},
    "domain_insights": {...},
    "gap_analysis": "..."
  },
  "generation": {
    "gap_bridging": [...],
    "metric_improvements": [...]
  },
  "criticism": {
    "suggested_experiences": {
      "bridging_gaps": [...],
      "metric_improvements": [...]
    }
  },
  "run_metrics": {
    "total_tokens": 1234,
    "estimated_cost_usd": 0.0123,
    "failures": []
  }
}
```

**Error Response (4xx/5xx)**:
If an error occurs, the system will respond with a JSON object containing an error message.

```json
{
  "detail": "Error message describing the issue"
}
```