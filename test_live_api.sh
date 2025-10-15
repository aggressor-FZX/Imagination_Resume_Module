#!/bin/bash
# Test script for the live Imaginator Resume Co-Writer API on Render

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# API Configuration
API_URL="https://imaginator-resume-cowriter.onrender.com"
API_KEY="${API_KEY:-your-api-key-here}"  # Set via environment or replace

echo "======================================"
echo "Testing Imaginator API on Render"
echo "======================================"
echo ""

# Test 1: Health Check
echo -e "${YELLOW}Test 1: Health Check${NC}"
HEALTH_RESPONSE=$(curl -s "${API_URL}/health")
echo "$HEALTH_RESPONSE" | jq .
if echo "$HEALTH_RESPONSE" | jq -e '.status == "healthy"' > /dev/null; then
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${RED}✗ Health check failed${NC}"
    exit 1
fi
echo ""

# Test 2: API without authentication (should fail with 403)
echo -e "${YELLOW}Test 2: API Call Without Authentication (should fail)${NC}"
NO_AUTH_RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" "${API_URL}/analyze" \
    -H "Content-Type: application/json" \
    -d '{
        "resume_text": "Test",
        "job_description": "Test"
    }')
HTTP_STATUS=$(echo "$NO_AUTH_RESPONSE" | grep "HTTP_STATUS:" | cut -d':' -f2)
if [ "$HTTP_STATUS" = "403" ]; then
    echo -e "${GREEN}✓ Authentication required (expected 403)${NC}"
else
    echo -e "${RED}✗ Expected 403, got $HTTP_STATUS${NC}"
fi
echo ""

# Test 3: Full API test with authentication
if [ "$API_KEY" != "your-api-key-here" ]; then
    echo -e "${YELLOW}Test 3: Full API Call With Authentication${NC}"
    echo "Using sample resume and job description..."
    
    # Create test payload
    cat > /tmp/test_payload.json << 'EOF'
{
    "resume_text": "John Doe\nSoftware Engineer\n\nExperience:\n- 5 years Python development\n- Django and Flask frameworks\n- REST API design\n\nSkills:\n- Python, JavaScript\n- Docker, Kubernetes\n- PostgreSQL, MongoDB",
    "job_description": "Senior Software Engineer\n\nRequired Skills:\n- 5+ years Python\n- React.js\n- Microservices architecture\n- Cloud platforms (AWS/GCP)\n- Strong communication skills"
}
EOF

    # Make API call
    API_RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" "${API_URL}/analyze" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: ${API_KEY}" \
        -d @/tmp/test_payload.json)
    
    HTTP_STATUS=$(echo "$API_RESPONSE" | grep "HTTP_STATUS:" | cut -d':' -f2)
    RESPONSE_BODY=$(echo "$API_RESPONSE" | sed '/HTTP_STATUS:/d')
    
    if [ "$HTTP_STATUS" = "200" ]; then
        echo -e "${GREEN}✓ API call successful (200)${NC}"
        echo ""
        echo "Response summary:"
        echo "$RESPONSE_BODY" | jq '{
            matched_skills: .matched_skills[0:3],
            missing_skills: .missing_skills[0:3],
            skill_gap_percentage: .skill_gap_percentage,
            confidence: .confidence
        }' 2>/dev/null || echo "$RESPONSE_BODY"
    else
        echo -e "${RED}✗ API call failed with status $HTTP_STATUS${NC}"
        echo "Response:"
        echo "$RESPONSE_BODY"
    fi
    
    rm -f /tmp/test_payload.json
else
    echo -e "${YELLOW}Test 3: Skipped (API_KEY not set)${NC}"
    echo "To test the full API, run:"
    echo "  export API_KEY='your-actual-api-key'"
    echo "  ./test_live_api.sh"
fi

echo ""
echo "======================================"
echo "Testing Complete"
echo "======================================"
