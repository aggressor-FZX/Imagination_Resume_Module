# Quick Start Guide - Imaginator Resume Co-Writer

Get up and running with the Imaginator Resume Co-Writer API in 5 minutes.

---

## 🚀 For Backend Developers

### 1. Get Your API Key

Contact the service administrator to get your `X-API-Key`.

### 2. Make Your First Request

```bash
curl -X POST https://imaginator-resume-cowriter.onrender.com/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY_HERE" \
  -d '{
    "resume_text": "John Doe\nSoftware Engineer\n\nEXPERIENCE:\nSenior Developer at TechCorp\n- Python, Django, PostgreSQL\n- AWS, Docker, CI/CD",
    "job_ad": "Senior Full Stack Developer\n\nREQUIREMENTS:\n- 5+ years Python\n- React.js experience\n- AWS and Kubernetes\n- Microservices architecture"
  }'
```

### 3. Handle the Response

```python
import requests

API_URL = "https://imaginator-resume-cowriter.onrender.com"
API_KEY = "your-api-key-here"

response = requests.post(
    f"{API_URL}/analyze",
    headers={
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    },
    json={
        "resume_text": "your resume text...",
        "job_ad": "job description..."
    },
    timeout=120
)

result = response.json()

# Extract key insights
skills = result["aggregate_skills"]
gap_analysis = result["gap_analysis"]
suggestions = result["suggested_experiences"]
cost = result["run_metrics"]["estimated_cost_usd"]

print(f"Skills found: {len(skills)}")
print(f"Analysis cost: ${cost:.4f}")
```

---

## 🧪 For Testers

### Quick Health Check

```bash
curl https://imaginator-resume-cowriter.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "production"
}
```

### Run Test Suite

```bash
# Clone the repository
git clone https://github.com/aggressor-FZX/Imagination_Resume_Module.git
cd Imagination_Resume_Module

# Set your API key
export API_KEY='your-api-key-here'

# Run tests
python test_live_api.py
```

---

## 💻 For Local Development

### 1. Clone and Setup

```bash
git clone https://github.com/aggressor-FZX/Imagination_Resume_Module.git
cd Imagination_Resume_Module

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:
```bash
OPENROUTER_API_KEY=your_openrouter_key
API_KEY=your_custom_api_key
CONTEXT7_API_KEY=your_context7_key  # Optional
```

### 3. Run Locally

```bash
# Start the server
uvicorn app:app --reload --port 8000

# In another terminal, test it
curl http://localhost:8000/health
```

### 4. Access API Documentation

Open in browser: http://localhost:8000/docs

---

## 🐳 For Docker Users

### Quick Docker Run

```bash
# Build
docker build -t imaginator .

# Run
docker run -p 8000:8000 --env-file .env imaginator

# Test
curl http://localhost:8000/health
```

### Using Docker Compose

```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 📊 Understanding the Response

### Minimal Response Structure

```json
{
  "aggregate_skills": ["Python", "AWS", "Docker"],
  "gap_analysis": "Your resume shows strong Python skills but lacks React.js and Kubernetes experience...",
  "suggested_experiences": {
    "bridging_gaps": [
      {
        "skill_focus": "React.js",
        "refined_suggestions": ["Built responsive dashboard using React.js and Redux..."]
      }
    ],
    "metric_improvements": [...]
  },
  "run_metrics": {
    "total_tokens": 3500,
    "estimated_cost_usd": 0.042
  }
}
```

### Key Fields

- **aggregate_skills**: All skills found in resume
- **gap_analysis**: Narrative comparison with job requirements
- **suggested_experiences.bridging_gaps**: Suggestions to address missing skills
- **suggested_experiences.metric_improvements**: Ways to strengthen existing experience
- **run_metrics**: Token usage and cost information

---

## ⚡ Common Use Cases

### 1. Basic Resume Analysis

```python
def analyze_resume(resume_text, job_ad):
    response = requests.post(
        f"{API_URL}/analyze",
        headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
        json={"resume_text": resume_text, "job_ad": job_ad},
        timeout=120
    )
    return response.json()
```

### 2. With Confidence Threshold

```python
# Only include high-confidence skills (0.8+)
result = requests.post(
    f"{API_URL}/analyze",
    headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
    json={
        "resume_text": resume_text,
        "job_ad": job_ad,
        "confidence_threshold": 0.8
    },
    timeout=120
).json()
```

> Note: The preferred integration method is for the FrontEnd to POST structured JSON to the `/analyze` endpoint. File uploads are not required.

### 4. Key Management

The service uses server-configured provider keys. Custom BYOK headers are deprecated and no longer supported.

---

## 🛠️ Troubleshooting

### 403 Forbidden Error

**Problem**: Missing or invalid API key

**Solution**: Ensure `X-API-Key` header is set correctly
```python
headers = {"X-API-Key": "your-actual-key"}
```

### Timeout Error

**Problem**: Request taking too long

**Solution**: Increase timeout (analysis can take 30-120 seconds)
```python
requests.post(..., timeout=120)  # 120 seconds
```

### 422 Validation Error

**Problem**: Invalid request body

**Solution**: Check required fields
```python
# Both fields are required
{
    "resume_text": "...",  # Required
    "job_ad": "..."        # Required
}
```

### Connection Error

**Problem**: Can't reach the service

**Solution**: Check service status
```bash
curl https://imaginator-resume-cowriter.onrender.com/health
```

---

## 📈 Performance Tips

1. **Set Appropriate Timeout**: Use 120 seconds minimum
2. **Handle Retries**: Implement retry logic for transient failures
3. **Cache Results**: Cache analyses for identical resume/job pairs
4. **Batch Requests**: Don't overwhelm the service (max 10/min recommended)
5. **Monitor Costs**: Check `run_metrics.estimated_cost_usd` in responses

---

## 📚 Additional Resources

- **Full Documentation**: [README.md](README.md)
- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)
- **I/O Specification**: [SYSTEM_IO_SPECIFICATION.md](SYSTEM_IO_SPECIFICATION.md)
- **Deployment Info**: [DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md)
- **Interactive Docs**: https://imaginator-resume-cowriter.onrender.com/docs

---

## 🆘 Support

- **GitHub Issues**: https://github.com/aggressor-FZX/Imagination_Resume_Module/issues
- **Health Status**: https://imaginator-resume-cowriter.onrender.com/health
- **Test Your Setup**: Run `python test_live_api.py`

---

## 🎯 Next Steps

1. ✅ Get your API key
2. ✅ Test the health endpoint
3. ✅ Make your first analysis request
4. ✅ Integrate into your application
5. ✅ Monitor usage and costs

Happy analyzing! 🚀
