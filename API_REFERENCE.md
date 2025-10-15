# API Reference - Imaginator Resume Co-Writer

## Base URL

**Production**: `https://imaginator-resume-cowriter.onrender.com`  
**Local Development**: `http://localhost:8000`

---

## Authentication

All `/analyze` endpoints require authentication via the `X-API-Key` header:

```bash
X-API-Key: your-api-key-here
```

### Bring Your Own Key (BYOK)

Optionally provide your own LLM API keys to use your own credits:

```bash
X-OpenAI-API-Key: your-openai-key
X-Google-API-Key: your-google-key
X-Anthropic-API-Key: your-anthropic-key
```

---

## Endpoints

### 1. Health Check

Check if the service is running and healthy.

**Endpoint**: `GET /health`  
**Authentication**: None required

**Response** (200 OK):
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "production"
}
```

**Example**:
```bash
curl https://imaginator-resume-cowriter.onrender.com/health
```

---

### 2. Analyze Resume

Analyze a resume against a job description to identify skills, gaps, and generate recommendations.

**Endpoint**: `POST /analyze`  
**Authentication**: Required (`X-API-Key` header)

**Request Headers**:
```
Content-Type: application/json
X-API-Key: your-api-key-here
```

**Request Body**:
```json
{
  "resume_text": "string (required) - Full resume text content",
  "job_ad": "string (required) - Target job description",
  "confidence_threshold": "number (optional, default: 0.7) - Skill confidence threshold (0.0-1.0)",
  "extracted_skills_json": "string (optional) - JSON with structured skills data",
  "domain_insights_json": "string (optional) - JSON with domain insights"
}
```

**Response** (200 OK):
```json
{
  "experiences": [
    {
      "title_line": "Job Title at Company",
      "skills": ["skill1", "skill2"],
      "snippet": "Experience description..."
    }
  ],
  "aggregate_skills": ["Python", "AWS", "Docker", "..."],
  "processed_skills": {
    "high_confidence": ["Python", "SQL"],
    "medium_confidence": ["AWS"],
    "low_confidence": [],
    "inferred_skills": ["Django", "Flask"]
  },
  "domain_insights": {
    "primary_domain": "Software Development",
    "market_demand": ["Cloud Computing", "AI/ML"],
    "emerging_skills": ["Kubernetes"]
  },
  "gap_analysis": "Detailed narrative analysis of skill gaps...",
  "suggested_experiences": {
    "bridging_gaps": [
      {
        "skill_focus": "Cloud Architecture",
        "refined_suggestions": [
          "Led migration to AWS ECS...",
          "Implemented Infrastructure as Code..."
        ]
      }
    ],
    "metric_improvements": [
      {
        "skill_focus": "Performance",
        "refined_suggestions": [
          "Optimized queries reducing latency by 40%...",
          "Implemented caching reducing load by 60%..."
        ]
      }
    ]
  },
  "run_metrics": {
    "calls": [
      {
        "provider": "openai",
        "model": "gpt-4",
        "prompt_tokens": 1500,
        "completion_tokens": 800,
        "cost_usd": 0.045
      }
    ],
    "total_prompt_tokens": 1500,
    "total_completion_tokens": 800,
    "total_tokens": 2300,
    "estimated_cost_usd": 0.045,
    "failures": []
  },
  "processing_status": "COMPLETED",
  "processing_time_seconds": 45.23
}
```

**Error Responses**:

- **403 Forbidden** - Missing or invalid API key
  ```json
  {"detail": "X-API-Key header is required"}
  ```

- **422 Unprocessable Entity** - Invalid request body
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

- **500 Internal Server Error** - Processing error
  ```json
  {"detail": "Error message"}
  ```

**Example**:
```bash
curl -X POST https://imaginator-resume-cowriter.onrender.com/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "resume_text": "John Doe\nSoftware Engineer\nPython, AWS, Docker experience...",
    "job_ad": "Senior Developer\nRequires: Python, AWS, Kubernetes, CI/CD...",
    "confidence_threshold": 0.7
  }'
```

---

### 3. Analyze Resume File

Upload a resume file and analyze it against a job description.

**Endpoint**: `POST /analyze-file`  
**Authentication**: Required (`X-API-Key` header)

**Request Headers**:
```
Content-Type: multipart/form-data
X-API-Key: your-api-key-here
```

**Form Data**:
- `resume_file`: File (text/plain) - Resume text file
- `job_ad`: String (required) - Job description text
- `confidence_threshold`: Number (optional, default: 0.7)

**Response**: Same as `/analyze` endpoint

**Example**:
```bash
curl -X POST https://imaginator-resume-cowriter.onrender.com/analyze-file \
  -H "X-API-Key: your-api-key" \
  -F "resume_file=@resume.txt" \
  -F "job_ad=Senior Developer position requiring Python..."
```

---

## Code Examples

### Python

```python
import requests

API_URL = "https://imaginator-resume-cowriter.onrender.com"
API_KEY = "your-api-key-here"

# Analyze resume
response = requests.post(
    f"{API_URL}/analyze",
    headers={
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    },
    json={
        "resume_text": "Your resume text...",
        "job_ad": "Job description...",
        "confidence_threshold": 0.8
    },
    timeout=120  # Analysis can take time
)

if response.status_code == 200:
    result = response.json()
    print(f"Matched Skills: {len(result['aggregate_skills'])}")
    print(f"Gap Analysis: {result['gap_analysis']}")
    print(f"Cost: ${result['run_metrics']['estimated_cost_usd']:.4f}")
else:
    print(f"Error {response.status_code}: {response.text}")
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

const API_URL = 'https://imaginator-resume-cowriter.onrender.com';
const API_KEY = 'your-api-key-here';

async function analyzeResume(resumeText, jobAd) {
  try {
    const response = await axios.post(`${API_URL}/analyze`, {
      resume_text: resumeText,
      job_ad: jobAd,
      confidence_threshold: 0.7
    }, {
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY
      },
      timeout: 120000  // 120 second timeout
    });
    
    console.log('Skills:', response.data.aggregate_skills);
    console.log('Gap Analysis:', response.data.gap_analysis);
    console.log('Cost:', response.data.run_metrics.estimated_cost_usd);
    
    return response.data;
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
    throw error;
  }
}

// Usage
analyzeResume(
  'John Doe\nSoftware Engineer\nPython, AWS...',
  'Senior Developer position...'
);
```

### cURL

```bash
# Basic analysis
curl -X POST https://imaginator-resume-cowriter.onrender.com/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d @- <<EOF
{
  "resume_text": "Your resume here...",
  "job_ad": "Job description here...",
  "confidence_threshold": 0.7
}
EOF

# File upload
curl -X POST https://imaginator-resume-cowriter.onrender.com/analyze-file \
  -H "X-API-Key: your-api-key" \
  -F "resume_file=@resume.txt" \
  -F "job_ad=Senior Developer position..."
```

---

## Rate Limits & Performance

- **Processing Time**: 30-120 seconds per analysis (varies with resume/job complexity)
- **Timeout**: Set client timeout to at least 120 seconds
- **Concurrent Requests**: Service can handle multiple concurrent requests
- **Rate Limits**: No hard limits currently, but recommended: max 10 requests/minute

---

## Cost Estimation

The `run_metrics` in the response includes:
- `total_tokens`: Total tokens consumed across all LLM calls
- `estimated_cost_usd`: Estimated cost in USD for the analysis
- `calls`: Per-call breakdown with provider and model information

Typical costs:
- Simple analysis: $0.02 - $0.05
- Complex analysis: $0.05 - $0.15
- With BYOK: Costs charged to your own API keys

---

## Error Handling

Best practices for error handling:

```python
import requests
from requests.exceptions import Timeout, RequestException

def analyze_with_retry(resume_text, job_ad, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{API_URL}/analyze",
                headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
                json={"resume_text": resume_text, "job_ad": job_ad},
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                raise ValueError("Invalid API key")
            elif response.status_code == 422:
                raise ValueError(f"Invalid input: {response.json()}")
            else:
                print(f"Attempt {attempt + 1} failed: {response.status_code}")
                
        except Timeout:
            print(f"Timeout on attempt {attempt + 1}")
        except RequestException as e:
            print(f"Request error: {e}")
            
    raise Exception("Max retries exceeded")
```

---

## Support & Documentation

- **Full Documentation**: See `README.md` and `SYSTEM_IO_SPECIFICATION.md`
- **Interactive API Docs**: https://imaginator-resume-cowriter.onrender.com/docs
- **Service Status**: Check `/health` endpoint
- **Test Script**: Use `test_live_api.py` for comprehensive testing

---

## CORS Configuration

The API is configured to allow requests from:
- **Production**: `https://www.cogitometric.org`
- **Local Development**: `http://localhost:*`

If you need to access from a different origin, contact the service administrator.

---

## Changelog

### v1.0.0 (2025-10-15)
- ✅ Initial production deployment
- ✅ Docker-based deployment on Render
- ✅ API key authentication
- ✅ Health check endpoint
- ✅ Comprehensive error handling
- ✅ BYOK support for LLM API keys
- ✅ Auto-deploy from GitHub
