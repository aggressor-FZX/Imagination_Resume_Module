# Generative Resume Co-Writer - Input/Output Specification

## Overview

The Generative Resume Co-Writer is a standalone Python application that analyzes resumes against target job descriptions using Large Language Models (LLMs) to provide personalized career development recommendations. The system processes resume data through a three-stage pipeline: Analysis, Generation, and Criticism.

## System Architecture

The application consists of three main processing stages:

1. **Analysis Stage** (`run_analysis_async`): Extracts skills, experiences, and performs gap analysis
2. **Generation Stage** (`run_generation_async`): Creates targeted resume improvement suggestions
3. **Criticism Stage** (`run_criticism_async`): Refines suggestions through adversarial review
4. **Synthesis Stage** (`run_synthesis_async`): Incorporates critique into final written section

## Input Specifications

### Required Inputs

| Input | Type | Description | Example |
|-------|------|-------------|---------|
| `resume_text` | String | Raw resume text content | "John Doe\nSoftware Engineer\n..." |
| `job_ad` | String | Target job description text | "Senior Python Developer\nRequirements: Python, Django..." |

### Optional Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `extracted_skills_json` | File Path or JSON | None | JSON file/object with structured skills data (confidence scores) |
| `domain_insights_json` | File Path or JSON | None | JSON file/object with domain-specific insights |
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
      "description": "Structured skill analysis with confidence scores and categories",
      "properties": {
        "high_confidence": {"type": "array", "items": {"type": "string"}},
        "medium_confidence": {"type": "array", "items": {"type": "string"}},
        "low_confidence": {"type": "array", "items": {"type": "string"}},
        "inferred_skills": {"type": "array", "items": {"type": "string"}}
      },
      "required": ["high_confidence", "medium_confidence", "low_confidence", "inferred_skills"]
    },
    "domain_insights": {
      "type": "object",
      "description": "Industry-specific insights and market context",
      "properties": {
        "domain": {"type": "string"},
        "market_demand": {"type": "string"},
        "skill_gap_priority": {"type": "string"},
        "emerging_trends": {"type": "array", "items": {"type": "string"}},
        "insights": {"type": "array", "items": {"type": "string"}}
      }
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
  "domain": "Software Development",
  "market_demand": "High",
  "skill_gap_priority": "Medium",
  "emerging_trends": ["AI/ML", "Cloud Computing"],
  "insights": ["Strong demand for full-stack developers"]
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
- `OPENROUTER_API_KEY`: OpenRouter API key (single LLM provider)

### Optional Environment Variables
- `CONTEXT7_API_KEY`: Context7 API key for documentation features
- `OPENROUTER_PRICE_INPUT_PER_1K`: Custom pricing for OpenRouter input tokens (default: 0.0005)
- `OPENROUTER_PRICE_OUTPUT_PER_1K`: Custom pricing for OpenRouter output tokens (default: 0.0015)

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

### Calling the Live API

To integrate with the Generative Resume Co-Writer, backend systems should send a POST request to the `/analyze` endpoint of the deployed FastAPI application.

**Production Endpoint**: `https://imaginator-resume-cowriter.onrender.com/analyze`

**Local/Container Endpoint**: `http://localhost:8000/analyze`

**Method**: `POST`

**Headers**:
- `Content-Type`: `application/json`
- `X-API-Key`: `your-api-key-here` **(Required for production)**

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
  "experiences": [...],
  "aggregate_skills": [...],
  "processed_skills": {...},
  "domain_insights": {...},
  "gap_analysis": "...",
  "suggested_experiences": {
    "bridging_gaps": [...],
    "metric_improvements": [...]
  },
  "seniority_analysis": {...},
  "final_written_section": "...",
  "run_metrics": {
    "total_tokens": 1234,
    "estimated_cost_usd": 0.0123,
    "failures": []
  }
}
```

**Authentication Error (403 Forbidden)**:
If the `X-API-Key` header is missing or invalid:

```json
{
  "detail": "X-API-Key header is required"
}
```

**Validation Error (422 Unprocessable Entity)**:
If the request body is invalid:

```json
{
  "detail": [
    {
      "loc": ["body", "resume_text"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Server Error (500 Internal Server Error)**:
If an unexpected error occurs during processing:

```json
{
  "detail": "Error message describing the issue"
}
```

### Authentication

The production API requires authentication using the `X-API-Key` header:

```bash
curl -X POST https://imaginator-resume-cowriter.onrender.com/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"resume_text": "...", "job_ad": "..."}'
```

**Key Management**
Provider keys are configured server-side. BYOK headers are deprecated and not supported.

### Caching Behavior
The analysis stage uses an in-memory cache keyed by input content to improve performance.

- TTL configured via `ANALYSIS_CACHE_TTL_SECONDS` (default: `600` seconds)
- Cache key includes `resume_text`, `job_ad`, `extracted_skills_json`, `domain_insights_json`, and `confidence_threshold`
- Observe cache usage via `run_metrics.stages.analysis.cache_hit`
    "seniority_analysis": {
      "type": "object",
      "description": "Seniority level analysis",
      "properties": {
        "level": {"type": "string"},
        "confidence": {"type": "number"},
        "total_years_experience": {"type": "number"},
        "experience_quality_score": {"type": "number"},
        "leadership_score": {"type": "number"},
        "skill_depth_score": {"type": "number"},
        "achievement_complexity_score": {"type": "number"},
        "reasoning": {"type": "string"},
        "recommendations": {"type": "array", "items": {"type": "string"}}
      },
      "required": ["level", "confidence", "total_years_experience", "experience_quality_score", "leadership_score", "skill_depth_score", "achievement_complexity_score", "reasoning", "recommendations"]
    },
    "final_written_section": {
      "type": "string",
      "description": "Generated resume experience section text"
    },
#### 7. Seniority Analysis
Detected seniority level and supporting signals:
```json
{
  "level": "mid-level",
  "confidence": 0.83,
  "total_years_experience": 5.0,
  "experience_quality_score": 0.7,
  "leadership_score": 0.3,
  "skill_depth_score": 0.6,
  "achievement_complexity_score": 0.5,
  "reasoning": "5.0 years of experience demonstrates significant expertise",
  "recommendations": ["Focus on building technical depth", "Seek mentorship opportunities"]
}
```

#### 8. Final Written Section
Generated resume experience section text:
```json
"Led migration of legacy monolith to microservices on AWS ECS, improving deployment frequency by 3x and reducing rollback incidents by 40%."
```
