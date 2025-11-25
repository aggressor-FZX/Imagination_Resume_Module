# Generative Resume Co-Writer

AI-powered resume analysis tool that provides personalized career development recommendations by analyzing resumes against target job descriptions using advanced language models and structured skill processing.

## 🌐 Live Service

**Production API**: https://imaginator-resume-cowriter.onrender.com  
**Status**: ✅ Live and Operational  
**Documentation**: https://imaginator-resume-cowriter.onrender.com/docs

**Quick Start**: See [QUICKSTART.md](QUICKSTART.md) for 5-minute integration guide  
**API Reference**: See [API_REFERENCE.md](API_REFERENCE.md) for complete endpoint documentation

---

## 🚀 Features

### Three-Stage AI Pipeline
- **Analysis**: Extracts skills, experiences, and identifies gaps using structured data processing
- **Generation**: Creates targeted resume improvement suggestions based on job requirements
- **Criticism**: Refines suggestions through adversarial review for maximum impact

### Advanced Skill Processing
- **Structured Skill Analysis**: Processes skills with confidence scores and categorization
- **Confidence-Based Filtering**: Filters skills based on extraction confidence thresholds
- **Knowledge Base Integration**: Uses skill adjacency and competency mapping for intelligent analysis

### Web Service Architecture (FastAPI)
- **RESTful API**: Async endpoints with automatic OpenAPI documentation
- **Input Model**: The FrontEnd provides structured JSON payloads (parsed by a document loader). The service expects this structured JSON for analysis — file uploads are not required.
- **Health Monitoring**: Built-in health checks and metrics endpoints
- **Configuration Management**: Environment-based configuration with pydantic-settings
- **CORS Support**: Cross-origin resource sharing for web applications

### Repository Integration
- **FastSVM_Skill_Title_Extraction**: High-confidence skill extraction with SVM models
- **Hermes**: Domain insights, market alignment, and strategic recommendations
- **Resume_Document_Loader**: Structured text extraction and preprocessing

### Robust Architecture
- **Graceful Degradation**: Continues functioning even with API failures or partial data
- **Provider Fallback**: Automatic fallback from OpenAI to Google to Anthropic on failures
- **Schema Validation**: Ensures output conforms to standardized JSON format
- **Async Processing**: Concurrent LLM API calls for improved performance
- **Usage Tracking**: Comprehensive metrics for tokens, costs, and performance monitoring

## 📋 Files

### Core Application
- `app.py`: FastAPI web service with async endpoints
- `config.py`: Configuration management with pydantic-settings
- `models.py`: Pydantic models for API request/response validation
- `imaginator_flow.py`: Core analysis functions (CLI compatibility maintained)
- `requirements.txt`: Python dependencies

### Deployment & Configuration
- `Dockerfile`: Multi-stage container build with UV package manager
- `docker-compose.yml`: Local development setup
- `render.yaml`: Render deployment configuration (✅ deployed)
- `.env`: Environment file with API keys (create from .env.example)

### Testing
- `pytest.ini`: Test configuration
- `run_tests.py`: Test runner script
- `tests/test_app.py`: FastAPI endpoint tests
- `test_live_api.py`: Production API test suite
- `test_live_api.sh`: Bash test script
- `test/`: Additional test files and reports

### Sample Data
- `sample_resume.txt`: Sample resume for testing
- `sample_job_ad.txt`: Sample job description for testing

### Documentation
- 📖 **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- 📖 **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API documentation
- 📖 **[SYSTEM_IO_SPECIFICATION.md](SYSTEM_IO_SPECIFICATION.md)** - Input/output specification
- 📖 **[DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md)** - Deployment details and status
- 📖 **[deployment_readiness_log.md](deployment_readiness_log.md)** - Deployment history

## 🛠 Setup

### Local Development Setup

1. **Clone and navigate to the repository:**
   ```bash
   git clone <repository-url>
   cd imaginator
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
  Create a `.env` file in the project root (we use OpenRouter as the single LLM provider):
  ```bash
  OPENROUTER_API_KEY=your_openrouter_key_here
  API_KEY=your_service_api_key_here
  CONTEXT7_API_KEY=your_context7_key_here  # Optional, for documentation features

  # Optional: Custom pricing (defaults provided) for OpenRouter models
  OPENROUTER_PRICE_INPUT_PER_1K=0.0005
  OPENROUTER_PRICE_OUTPUT_PER_1K=0.0015
  ```

### Docker Setup

**Prerequisites:**
- Docker and Docker Compose installed
- `.env` file with API keys (see `.env.example`)

**Build and run with Docker Compose:**
```bash
# Build the container
docker-compose build

# Run the service (port 8000 by default, or 8001 if 8000 is in use)
docker-compose up

# Or run in background
docker-compose up -d

# Check health endpoint
curl http://localhost:8000/health
```

**Manual Docker build:**
```bash
# Build the image
docker build -t imaginator .

# Run the container
docker run -p 8000:8000 --env-file .env imaginator
```

**Container Features:**
- Multi-stage build with UV package manager for fast Python installs
- Non-root user for security
- Health checks configured
- Optimized for production with minimal image size
- Tested locally ✅

### Render Deployment

The application is **LIVE** and deployed to Render with production-ready settings.

**Live Service:**
- **URL**: https://imaginator-resume-cowriter.onrender.com
- **Service ID**: srv-d3nf73ur433s73bh9j00
- **Region**: Oregon (US West)
- **Plan**: Starter ($7/month - 0.5 CPU, 512 MB RAM)
- **Runtime**: Docker
- **Status**: ✅ Active and Healthy

**Deployment Configuration:**
1. **GitHub Integration**: Automatic deployment on push to master branch
2. **Health Monitoring**: `/health` endpoint checked every 30 seconds
3. **Environment Variables**: Configured via Render dashboard and `render.yaml`
4. **CORS**: Restricted to `https://www.cogitometric.org`
5. **API Authentication**: Requires `X-API-Key` header for `/analyze` endpoint

**Environment Variables (configured in Render):**
- `OPENROUTER_API_KEY` ✅
- `API_KEY` - Custom API key for endpoint authentication ✅
- `CORS_ORIGINS` - Set to cogitometric.org domain ✅
- `ENVIRONMENT` - Set to "production" ✅

**Deployment Features:**
- ✅ Auto-deploy on git push
- ✅ Docker container build and run
- ✅ Health check monitoring
- ✅ SSL/TLS encryption (HTTPS)
- ✅ Custom domain support ready
- ✅ Environment-based configuration
- ✅ API key authentication
- ✅ CORS security

**Status**: ✅ Deployed | ✅ Live | ✅ Tested

The `render.yaml` file handles all deployment configuration including health checks, environment variables, and scaling settings.

### Testing the Live API

The deployed service includes a comprehensive test suite for verifying functionality:

**Test Script**: `test_live_api.py`

```bash
# Run basic tests (health check and authentication)
python test_live_api.py

# Run full analysis test with your API key
export API_KEY='your-api-key-here'
python test_live_api.py
```

**Test Cases:**
1. ✅ **Health Check** - Verifies service is running and responding
2. ✅ **Authentication** - Confirms API key is required (403 without key)
3. ⏳ **Full Analysis** - Tests complete resume analysis workflow (requires API_KEY)

**Manual API Testing:**

```bash
# Health check (no auth required)
curl https://imaginator-resume-cowriter.onrender.com/health

# Analyze resume (requires API key)
curl -X POST https://imaginator-resume-cowriter.onrender.com/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "resume_text": "Your resume text here",
    "job_ad": "Job description here"
  }'
```

**Expected Responses:**
- Health: `{"status":"healthy","version":"1.0.0","environment":"production"}`
- No Auth: `{"detail":"X-API-Key header is required"}`
- Success: Full analysis JSON with skills, gaps, and recommendations

#### Transport Metrics

Responses include `run_metrics.transport.http_pool` summarizing pooled HTTP transport for feature‑flagged external calls:
- `requests_total`, `errors_total`, `timeouts_total`
- `status_counts` per HTTP status
- `latency_ms` with `last`, `p50`, `p95`

#### Connection Pooling

The service uses a shared `aiohttp` `ClientSession` with `TCPConnector` to reuse connections across requests.
- Config:
  - `MAX_CONCURRENT_REQUESTS` → `settings.max_concurrent_requests`
  - `REQUEST_TIMEOUT` → `settings.request_timeout`
- Lifecycle: initialized at startup and closed on shutdown; external calls in `imaginator_flow.py` reuse the shared session when available.

### Render MCP Server

This project is configured to use the Render MCP server, which allows for direct interaction with your Render account from within the development environment. This enables automated deployments, environment management, and monitoring.

To use the Render MCP server, you will need to:
1.  Have Docker installed and running.
2.  Set the `RENDER_API_KEY` in your `.env` file.
3.  Reload the VS Code window to activate the MCP server.

## 🎯 Usage

### Web Service API

The application provides a RESTful API with the following endpoints:

#### Start the Server

**Local development:**
```bash
# Using uvicorn directly
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Or using the provided script
python -m uvicorn app:app --reload
```

**Access the API:**

**Local Development:**
- **Interactive API Docs**: http://localhost:8000/docs (Swagger UI)
- **ReDoc Documentation**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

**Production (Live):**
- **API Base URL**: https://imaginator-resume-cowriter.onrender.com
- **Interactive API Docs**: https://imaginator-resume-cowriter.onrender.com/docs
- **Health Check**: https://imaginator-resume-cowriter.onrender.com/health

#### API Endpoints

##### POST `/analyze`
Analyze a resume against a job description.

**Authentication**: Requires `X-API-Key` header

**Request Headers:**
```
Content-Type: application/json
X-API-Key: your-api-key-here
```

**Request Body:**
```json
{
  "resume_text": "John Doe\nSoftware Engineer\nPython, SQL, AWS",
  "job_ad": "Senior Developer\nPython, AWS, Docker required",
  "confidence_threshold": 0.7,
  "extracted_skills_json": "{\"skills\": [\"python\", \"sql\"]}",
  "domain_insights_json": "{\"domain\": \"tech\"}"
}
```

**Response:**
```json
{
  "experiences": [],
  "aggregate_skills": ["Python", "SQL", "AWS"],
  "processed_skills": {
    "high_confidence": ["Python"],
    "medium_confidence": ["SQL"],
    "low_confidence": [],
    "inferred_skills": ["Django", "Flask"]
  },
  "domain_insights": {
    "domain": "software",
    "market_demand": "high",
    "skill_gap_priority": "medium"
  },
  "gap_analysis": "Analysis results...",
  "suggested_experiences": {
    "bridging_gaps": ["Learn Docker"],
    "metric_improvements": ["Add metrics"]
  },
  "seniority_analysis": {
    "level": "mid-level",
    "confidence": 0.83,
    "total_years_experience": 5.0,
    "experience_quality_score": 0.7,
    "leadership_score": 0.3,
    "skill_depth_score": 0.6,
    "achievement_complexity_score": 0.5,
    "reasoning": "5.0 years of experience demonstrates significant expertise",
    "recommendations": [
      "Focus on building technical depth",
      "Seek mentorship opportunities"
    ]
  },
  "final_written_section": "Generated resume experience section text",
  "run_metrics": {
    "total_tokens": 2300,
    "estimated_cost_usd": 0.045,
    "calls": [...],
    "failures": []
  },
  "processing_status": "COMPLETED",
  "processing_time_seconds": 2.34
}
```

> Note: This service expects structured JSON from the FrontEnd. The `POST /analyze` endpoint accepts the complete analysis payload as JSON (see `/analyze` above). File-upload endpoints are not required in the current integration.

##### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "development",
  "has_openai_key": true,
  "has_anthropic_key": true
}
```

##### GET `/config`
Get current configuration (for debugging).

##### GET `/docs/{library}`
Get library documentation using Context7 (if configured).

**Example:** `/docs/fastapi?version=0.104.1`

### Python Client Example

```python
import requests

# Production API (requires API key)
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
        "resume_text": "Your resume text here",
        "job_ad": "Job description here",
        "confidence_threshold": 0.8
    },
    timeout=120  # Analysis can take time
)

if response.status_code == 200:
    result = response.json()
    print(f"Skills found: {result['aggregate_skills']}")
    print(f"Skill gap: {result['gap_analysis']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### CLI Compatibility

The original CLI interface is still available for backward compatibility:

```bash
# Basic usage
python imaginator_flow.py --resume sample_resume.txt --target_job_ad "Senior Developer role"

# Advanced usage with structured data
python imaginator_flow.py \
  --parsed_resume_text "$(cat sample_resume.txt)" \
  --extracted_skills_json sample_skills.json \
  --domain_insights_json sample_insights.json \
  --target_job_ad "$(cat sample_job_ad.txt)" \
  --confidence_threshold 0.8
```

  --resume sample_resume.txt \| `--resume` | Path to resume file | - |

  --target_job_ad "Senior Python Developer with 5+ years experience in Django, React, and AWS"| `--parsed_resume_text` | Resume text from Resume_Document_Loader | - |

```| `--extracted_skills_json` | Skills data from FastSVM/Hermes | - |

| `--domain_insights_json` | Domain insights from Hermes | - |

### Advanced Usage with Structured Data| `--target_job_ad` | Job description (required) | - |

```bash| `--confidence_threshold` | Minimum skill confidence | 0.7 |

python imaginator_flow.py \

  --parsed_resume_text "$(cat sample_resume.txt)" \## 📊 Output Format

  --extracted_skills_json sample_skills.json \

  --domain_insights_json sample_insights.json \```json

  --target_job_ad "$(cat sample_job_ad.txt)" \{

  --confidence_threshold 0.8  "experiences": [...],

```  "aggregate_skills": ["python", "aws", "docker"],

  "processed_skills": {

### Input Options    "high_confidence_skills": ["python", "aws"],

    "skill_confidences": {"python": 0.95, "aws": 0.92},

| Parameter | Description | Required |    "categories": {...}

|-----------|-------------|----------|  },

| `--target_job_ad` | Job description text to analyze against | Yes |  "domain_insights": {

| `--resume` | Path to resume text file | Yes* |    "domain": "tech",

| `--parsed_resume_text` | Direct resume text input | Yes* |    "insights": {

| `--extracted_skills_json` | JSON file with structured skills data | No |      "strengths": [...],

| `--domain_insights_json` | JSON file with domain insights | No |      "gaps": [...],

| `--confidence_threshold` | Minimum skill confidence score (0.0-1.0) | No (default: 0.7) |      "recommendations": [...]

    }

*Either `--resume` or `--parsed_resume_text` must be provided.  },

  "gap_analysis": "Creative development recommendations..."

## 📊 Output Format}

```

The system produces validated JSON output with the following structure:

## Run Metrics, Token Usage, and Cost Estimation

```json

{The module now appends a `run_metrics` block to the final JSON output so callers can reliably parse token usage, per-call breakdowns, failures, and an estimated cost for the run.

  "experiences": [...],

  "aggregate_skills": [...],Example output tail:

  "processed_skills": {...},

  "domain_insights": {...},```json

  "gap_analysis": "...",{

  "suggested_experiences": {  "suggested_experiences": { /* ... */ },

    "bridging_gaps": [...],  "run_metrics": {

    "metric_improvements": [...]    "calls": [

  },      {

  "run_metrics": {        "provider": "openai",

    "total_tokens": 2300,        "model": "gpt-3.5-turbo",

    "estimated_cost_usd": 0.045,        "prompt_tokens": 1499,

    "calls": [...],        "completion_tokens": 629,

    "failures": [...]        "total_tokens": 2128,

  }        "estimated_cost_usd": 0.001693

}      },

```      {

        "provider": "openai",

See `SYSTEM_IO_SPECIFICATION.md` for complete input/output documentation.        "model": "gpt-3.5-turbo",

        "prompt_tokens": 511,

## 🔧 Configuration        "completion_tokens": 366,

        "total_tokens": 877,

### Environment Variables        "estimated_cost_usd": 0.000805

      }

| Variable | Description | Default | Example |

|----------|-------------|---------|---------|

| `OPENROUTER_API_KEY` | OpenRouter API key | Required | `sk-or-v1-...` |

| `API_KEY` | Service authentication key | Required | `your-secret-key` |

| `CONTEXT7_API_KEY` | Context7 documentation API key | Optional | `ctx7-...` |

| `OPENROUTER_PRICE_INPUT_PER_1K` | OpenRouter input token price | 0.0005 | `0.0005` |

| `OPENROUTER_PRICE_OUTPUT_PER_1K` | OpenRouter output token price | 0.0015 | `0.0015` |

```

### Custom Pricing

Default per-1K token USD rates can be overridden:

- `OPENROUTER_PRICE_INPUT_PER_1K` (default: `0.0005`)
- `OPENROUTER_PRICE_OUTPUT_PER_1K` (default: `0.0015`)

## 🧪 Testing

The project includes comprehensive tests for the FastAPI web service, including unit tests, integration tests, and API endpoint validation.

### Running Tests

#### Quick Test Runner (Recommended)
```bash
# Run all tests with coverage
./run_tests.py --coverage

# Run only unit tests
./run_tests.py --type unit

# Run with linting and type checking
./run_tests.py --lint --type-check
```

#### Using pytest Directly
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_app.py -v

# Run tests matching pattern
pytest -k "test_health" -v
```

#### Docker Testing
```bash
# Run tests in container
docker-compose run --rm app pytest

# Run tests with coverage in container
docker-compose run --rm app ./run_tests.py --coverage
```

### Test Structure

The test suite includes:

- **Unit Tests** (`tests/test_app.py`):
  - Health check endpoint validation
  - Configuration endpoint testing
  - Resume analysis endpoint testing
  - Structured JSON input validation (FrontEnd document-loader output)
  - Error handling and edge cases
  - Context7 integration (when available)

- **Integration Tests**:
  - Full API request/response cycles
  - Database interactions (if added)
  - External service integrations

- **Performance Tests**:
  - Response time validation
  - Memory usage monitoring
  - Concurrent request handling

### Test Configuration

Tests use the following configuration:
- `pytest.ini`: Test configuration and markers
- `tests/test_app.py`: Main test suite
- Mocked LLM API calls for consistent testing
- Sample data from `sample_resume.txt` and `sample_job_ad.txt`

### CI/CD Testing

The test suite is designed to run in CI/CD environments:
```bash
# Install test dependencies
pip install -r requirements.txt

# Run tests with coverage
./run_tests.py --coverage --type-check

# Generate coverage reports
coverage html
```

### Writing New Tests

When adding new functionality:

1. **Add tests to `tests/test_app.py`** for API endpoints
2. **Use descriptive test names** following the pattern `test_<feature>_<scenario>`
3. **Mock external dependencies** (LLM APIs, file I/O)
4. **Test both success and failure cases**
5. **Include edge cases and error conditions**

Example test structure:
```python
def test_successful_analysis(self):
    # Test successful analysis flow

def test_invalid_input_handling(self):
    # Test error handling for invalid inputs

def test_api_fallback_behavior(self):
    # Test provider fallback functionality
```

- `test_report.md`: Comprehensive test summary with performance metrics
- `test_results.json`: Raw test data for programmatic analysis
- `test_results.log`: Detailed execution logs
- `test_results.html`: Visual performance dashboard (if generated)

### Test Configuration

Tests use the following sample data:
- `sample_resume.txt`: Test resume content
- `sample_job_ad.txt`: Test job description
- `sample_skills.json`: Structured skills data
- `sample_insights.json`: Domain insights data

### Performance Benchmarks

Typical test results (based on recent runs):
- **Test Duration**: 11-18 seconds per test case
- **Memory Usage**: 0-16.3 MB peak consumption
- **Token Usage**: 3,600-4,000 tokens per complete test
- **API Cost**: $0.0029-$0.0034 per test execution
- **Success Rate**: 100% (all test cases passing)

### Adding New Tests

To extend the test suite:

1. Add test cases to `test/system_io_test.py`
2. Update input variations in the `test_cases` list
3. Run tests to validate new scenarios
4. Update `test/README.md` with new test documentation

estimated_cost = run_metrics.get("estimated_cost_usd", 0.0)

1. **Analysis Stage**```

   - Parses resume text and extracts experiences

   - Processes skills using confidence-based filtering## 🔄 Core Functionality

   - Generates gap analysis using LLM

1. **Skill Processing**: Filters and prioritizes skills by confidence scores

2. **Generation Stage**2. **Domain Integration**: Incorporates industry-specific insights and market data

   - Creates targeted improvement suggestions3. **Creative Analysis**: Generates personalized, actionable development plans

   - Focuses on gap bridging and metric improvements4. **Confidence Weighting**: Prioritizes high-confidence skills for recommendations



3. **Criticism Stage**## 🧪 Testing

   - Applies adversarial review to suggestions

   - Refines wording and improves specificityTest with sample data:

```bash

### Error Handlingpython imaginator_flow.py \

  --parsed_resume_text "Senior developer with Python, AWS experience" \

- **API Failures**: Automatic provider fallback (OpenAI → Anthropic)  --extracted_skills_json sample_skills.json \

- **Network Issues**: Exponential backoff retry logic  --domain_insights_json sample_insights.json \

- **Data Issues**: Graceful degradation with safe default structures  --target_job_ad "Full-stack developer role with React, Node.js, cloud experience"

- **Validation**: JSON schema validation ensures output consistency```


## 📈 Metrics & Monitoring

The system tracks comprehensive usage metrics:
- Token consumption (prompt/completion/total)
- API call success/failure rates
- Estimated costs by provider
- Processing duration and performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is open source. See individual files for license information.

**Key Management**

Provider keys are managed server-side. BYOK headers are deprecated and not supported.
