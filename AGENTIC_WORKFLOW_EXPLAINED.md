# Imaginator Agentic Workflow Explained

## Overview

Imaginator is a **multi-stage agentic system** that analyzes resumes against job descriptions to identify skill gaps, suggest improvements, and provide AI-enhanced resume sections. The system follows a **4-stage pipeline** with independent AI agents working on different aspects.

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER REQUEST (Resume + Job Ad)                │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │   STAGE 1: ANALYSIS AGENT       │
        │  (Parse & Extract Information)  │
        └────────────────┬────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │  STAGE 2: GENERATION AGENT      │
        │  (Create Enhancement Suggestions)│
        └────────────────┬────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │  STAGE 3: SYNTHESIS AGENT       │
        │  (Combine & Structure Results)   │
        └────────────────┬────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │  STAGE 4: CRITICISM AGENT       │
        │  (Validate & Refine Output)      │
        └────────────────┬────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │      RETURN TO USER             │
        │  (Gaps, Suggestions, Rewrite)    │
        └─────────────────────────────────┘
```

## Stage Breakdown

### Stage 1: ANALYSIS Agent
**Purpose**: Extract and normalize information from resume and job ad

**Input**:
- Resume text (extracted via Document Reader)
- Job ad text (user provided)
- Extracted skills (from FastSVM)
- Domain insights (optional)

**Process**:
1. Parse resume for:
   - Experience level
   - Technical skills
   - Soft skills
   - Leadership experience
   - Domain expertise
2. Parse job ad for:
   - Required skills
   - Preferred qualifications
   - Seniority level
   - Industry context
3. Identify overlaps and gaps

**Output**:
- Structured skill comparison
- Experience level assessment
- Missing qualifications list

### Stage 2: GENERATION Agent
**Purpose**: Create creative enhancement suggestions

**Input**: Analysis results from Stage 1

**Process**:
1. Use **creative prompt** to trigger `qwen/qwen3-30b-a3b` model
2. Generate:
   - Gap analysis narrative
   - Skill development suggestions
   - Experience narratives for gaps
   - Rewritten resume sections tailored to job ad
3. Focus on:
   - Actionable recommendations
   - Realistic skill development paths
   - ATS-friendly wording

**Output**:
- Gap analysis (structured)
- Suggested experiences (JSON)
- Rewritten resume sections

### Stage 3: SYNTHESIS Agent
**Purpose**: Combine all insights into cohesive output

**Input**:
- Analysis results
- Generated suggestions
- Seniority assessment

**Process**:
1. Structure final response as `AnalysisResponse`:
   ```python
   {
       "gap_analysis": str,              # Narrative of gaps
       "suggested_experiences": {},       # Dict of suggestions
       "seniority_analysis": {},          # Seniority assessment
       "final_written_section": str       # AI-enhanced resume section
   }
   ```
2. Ensure consistency across all sections
3. Validate JSON structure

**Output**: Fully structured `AnalysisResponse` object

### Stage 4: CRITICISM Agent
**Purpose**: Validate quality and refine output

**Input**: Synthesized response from Stage 3

**Process**:
1. Use **critic prompt** to trigger `deepseek/deepseek-chat-v3.1` model
2. Review for:
   - Accuracy (matches job requirements)
   - Realism (achievable suggestions)
   - Tone (professional, encouraging)
   - Completeness (addresses all gaps)
3. Identify issues and refine

**Output**: Validated, high-quality final response

## Model Selection Strategy

The system uses **intelligent model routing** based on stage requirements:

```python
def _get_openrouter_preferences(system_prompt, user_prompt):
    if "creative" in system_prompt or "generation" in user_prompt:
        # Stage 2: Generation stage needs creativity
        return ["qwen/qwen3-30b-a3b"]

    elif "critic" in system_prompt or "review" in user_prompt:
        # Stage 4: Criticism stage needs reasoning
        return ["deepseek/deepseek-chat-v3.1"]

    else:
        # Stages 1 & 3: General analysis/synthesis
        return ["anthropic/claude-3-haiku"]
```

**Model Characteristics**:
- **Claude 3 Haiku**: Fast, precise analysis and structuring
- **DeepSeek Chat v3.1**: Strong reasoning, great for critical review
- **Qwen 3 30B**: Creative, generates natural language well

## Failsafe Fallback Mechanism

If a model fails or is unavailable, the system automatically falls back:

```
Preferred Model
      ↓
   [FAILS]
      ↓
Fallback Chain:
   deepseek/deepseek-chat-v3.1    (always works)
   anthropic/claude-3-haiku        (reliable backup)
   openai/gpt-3.5-turbo           (last resort)
```

**Benefits**:
✅ Never returns empty/null responses
✅ Automatically handles model unavailability
✅ Logs all failures for debugging
✅ Uses most cost-effective available model

## Request/Response Format (Per OpenRouter API Spec)

### Proper Request Format

```python
{
    "model": "anthropic/claude-3-haiku",
    "messages": [
        {
            "role": "system",
            "content": "You are a resume analysis expert..."
        },
        {
            "role": "user",
            "content": "Analyze this resume against the job ad..."
        }
    ],
    "max_tokens": 2000,
    "temperature": 0.7,
    "stream": False
}
```

### Proper Response Extraction

```python
response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://imaginator-resume-cowriter.onrender.com"
    },
    json=payload
)

if response.status_code == 200:
    data = response.json()
    content = data['choices'][0]['message']['content']
```

## Metrics & Monitoring

The system tracks per-stage metrics:

```python
RUN_METRICS = {
    "calls": [
        {
            "stage": "analysis",
            "model": "anthropic/claude-3-haiku",
            "prompt_tokens": 1234,
            "completion_tokens": 567,
            "cost_usd": 0.0015
        }
    ],
    "total_prompt_tokens": 5000,
    "total_completion_tokens": 2000,
    "estimated_cost_usd": 0.0075,
    "failures": [],
    "stages": {
        "analysis": {"duration_ms": 1200},
        "generation": {"duration_ms": 2100},
        "synthesis": {"duration_ms": 800},
        "criticism": {"duration_ms": 1500}
    }
}
```

## Example Flow

```
INPUT:
  Resume: "Senior Software Engineer, 5 years experience..."
  Job Ad: "Principal Engineer - needs ML/AI expertise..."
  Skills: ["Python", "Go", "REST APIs"]

STAGE 1 (ANALYSIS):
  Models: Claude 3 Haiku
  Output: "Missing ML/AI knowledge, has core engineering skills"

STAGE 2 (GENERATION):
  Models: Qwen 3 30B (creative)
  Output: "Suggest: take ML course, highlight data-driven projects"

STAGE 3 (SYNTHESIS):
  Models: Claude 3 Haiku
  Output: {
    "gap_analysis": "2 gaps identified: ML/AI, Big data experience",
    "suggested_experiences": ["Build ML project", "Learn TensorFlow"],
    "final_written_section": "Enhanced resume intro..."
  }

STAGE 4 (CRITICISM):
  Models: DeepSeek v3.1
  Output: "Validated - all suggestions are realistic and actionable"

RETURN TO USER:
  Complete AnalysisResponse with gaps, suggestions, rewrite
```

## Environment Variables Required

```bash
# OpenRouter API Keys (use the safe client's fallback chain)
OPENROUTER_API_KEY_1=sk-or-v1-...
OPENROUTER_API_KEY_2=sk-or-v1-...  # Backup key

# Service Authentication
IMAGINATOR_AUTH_TOKEN=05c2765ea794c...  # X-API-Key for requests

# Configuration
ENVIRONMENT=production
CONFIDENCE_THRESHOLD=0.7
MAX_CONCURRENT_REQUESTS=10
```

## Key Features

✅ **Multi-Agent Architecture**: Independent agents for each stage
✅ **Intelligent Routing**: Different models for different tasks
✅ **Failsafe Fallback**: Never fails completely
✅ **Cost Optimization**: Uses cheapest viable model per stage
✅ **Metrics Tracking**: Logs all usage and costs
✅ **Proper API Format**: Per OpenRouter specification
✅ **Model Validation**: Checks availability before calling

## Integration with Resume Pipeline

```
Document Reader (extracts text)
        ↓
Hermes (extracts structured resume data)
        ↓
FastSVM (extracts skills & titles)
        ↓
Imaginator (analyzes against job ad) ← YOU ARE HERE
        ↓
Frontend (displays results to user)
```

## Error Handling

The `OpenRouterSafeClient` provides robust error handling:

1. **Model Validation**: Checks if model exists before calling
2. **Format Validation**: Ensures messages follow OpenRouter spec
3. **HTTP Errors**: Handles all HTTP status codes appropriately
4. **Timeout**: 30-second timeout with automatic retry on failure
5. **Fallback Chain**: Tries next model if current fails

Example error responses:
```python
{
    "success": False,
    "error": "All models failed. Last error: Rate limit exceeded (429)",
    "model_used": None
}
```

## Future Improvements

- [ ] Implement prompt caching for repeated analyses
- [ ] Add streaming support for long responses
- [ ] Implement budget tracking and alerts
- [ ] Add A/B testing for different prompts
- [ ] Cache common job ad analyses

