#!/bin/bash
# Test the new 3-stage pipeline via live API

API_KEY="05c2765ea794c6e15374f2a63ac35da8e0e665444f6232225a3d4abfe5238c45"
BASE_URL="http://localhost:8000"

echo "=== Testing New 3-Stage Pipeline (Live API) ==="
echo ""

# Read sample resume
RESUME_TEXT=$(cat /home/skystarved/Render_Dockers/HF_Resume_Samples/sample_data_scientist.txt)
JOB_AD=$(cat /home/skystarved/Render_Dockers/Imaginator/sample_job_ad.txt)

# Call /analyze endpoint
echo "📤 Calling /analyze endpoint..."
curl -X POST "$BASE_URL/analyze" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{
    \"resume_text\": $(echo "$RESUME_TEXT" | jq -Rs .),
    \"job_ad\": $(echo "$JOB_AD" | jq -Rs .),
    \"confidence_threshold\": 0.7
  }" \
  -o new_pipeline_output.json \
  -w "\n📥 Response Status: %{http_code}\n" \
  2>&1

echo ""
echo "=== Response saved to new_pipeline_output.json ==="
echo ""

# Extract and display final resume
if [ -f new_pipeline_output.json ]; then
  echo "📄 Final Resume Content:"
  echo "========================"
  jq -r '.final_written_section_markdown // .final_written_section // "No resume content"' new_pipeline_output.json
  echo ""
  echo "📊 Metrics:"
  echo "========================"
  jq '{
    seniority: .seniority_level,
    word_count: (.final_written_section_markdown // "" | split(" ") | length),
    quantification_score: .quantification_analysis.quantification_score,
    domain_terms: (.domain_terms_used | length),
    pipeline_status: .pipeline_metrics.pipeline_status,
    duration_seconds: .pipeline_metrics.total_duration_seconds
  }' new_pipeline_output.json
fi
