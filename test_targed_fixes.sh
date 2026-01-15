#!/bin/bash

# Targeted test for the fixes we just implemented
echo "=========================================="
echo "Testing Targeted Fixes"
echo "=========================================="
echo "Date: $(date)"
echo ""

# Wait for deployment to complete
echo "Waiting 60 seconds for deployment to complete..."
sleep 60

# Test 1: Check if "is an" is detected as narrative
echo ""
echo "Test 1: 'is an' narrative detection"
echo "-----------------------------------"
cat > /tmp/test_is_an.json << 'EOF'
{
  "resume_text": "John Smith is an accomplished software engineer with over 8 years of experience.",
  "job_ad": "Senior Software Engineer",
  "confidence_threshold": 0.7
}
EOF

echo "Testing 'is an' pattern..."
curl -X POST "https://imaginator-resume-cowriter.onrender.com/analyze" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: 05c2765ea794c6e15374f2a63ac35da8e0e665444f6232225a3d4abfe5238c45" \
  -d @/tmp/test_is_an.json \
  --max-time 60 2>&1 | grep -o '"experiences":\[[^]]*\]' || echo "No experiences found (good - narrative filtered)"

# Test 2: Mixed resume with narrative block
echo ""
echo "Test 2: Mixed resume with narrative block"
echo "-----------------------------------------"
cat > /tmp/test_mixed.json << 'EOF'
{
  "resume_text": "PROFESSIONAL EXPERIENCE\n\nSenior Software Engineer\nTechCorp Inc. | Jan 2020 - Present\n- Led development of cloud-native applications\n- Implemented CI/CD pipelines\n\nJohn is a dedicated professional. He has excellent communication skills.\n\nSoftware Developer\nStartupXYZ | Jun 2018 - Dec 2019\n- Developed REST APIs\n- Collaborated with product team",
  "job_ad": "Senior Software Engineer",
  "confidence_threshold": 0.7
}
EOF

echo "Testing mixed resume (should extract 2 experiences, filter narrative)..."
curl -X POST "https://imaginator-resume-cowriter.onrender.com/analyze" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: 05c2765ea794c6e15374f2a63ac35da8e0e665444f6232225a3d4abfe5238c45" \
  -d @/tmp/test_mixed.json \
  --max-time 60 2>&1 | tee /tmp/test_mixed_output.json

echo ""
echo "Checking experiences count..."
if [ -f /tmp/test_mixed_output.json ]; then
    EXP_COUNT=$(grep -o '"title_line"' /tmp/test_mixed_output.json | wc -l)
    echo "Experiences extracted: $EXP_COUNT (should be 2)"
    
    # Show what was extracted
    grep -o '"title_line":"[^"]*"' /tmp/test_mixed_output.json || echo "No title lines found"
fi

# Test 3: Education section should not be parsed as experience
echo ""
echo "Test 3: Education section filtering"
echo "-----------------------------------"
cat > /tmp/test_education.json << 'EOF'
{
  "resume_text": "PROFESSIONAL EXPERIENCE\n\nSoftware Developer\nCompany | 2020-2022\n- Developed applications\n\nEDUCATION\nBachelor of Science\nUniversity | 2016-2020",
  "job_ad": "Software Developer",
  "confidence_threshold": 0.7
}
EOF

echo "Testing education filtering (should extract 1 experience, not 2)..."
curl -X POST "https://imaginator-resume-cowriter.onrender.com/analyze" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: 05c2765ea794c6e15374f2a63ac35da8e0e665444f6232225a3d4abfe5238c45" \
  -d @/tmp/test_education.json \
  --max-time 60 2>&1 | tee /tmp/test_education_output.json

echo ""
echo "Checking experiences count..."
if [ -f /tmp/test_education_output.json ]; then
    EXP_COUNT=$(grep -o '"title_line"' /tmp/test_education_output.json | wc -l)
    echo "Experiences extracted: $EXP_COUNT (should be 1, not including EDUCATION)"
    
    # Show what was extracted
    grep -o '"title_line":"[^"]*"' /tmp/test_education_output.json || echo "No title lines found"
fi

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="