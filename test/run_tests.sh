#!/bin/bash
# Test Runner Script for System I/O Tests

echo "🚀 Running System I/O Test Suite"
echo "================================="

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Please activate with:"
    echo "   source .venv/bin/activate"
    exit 1
fi

# Check if required dependencies are installed
echo "📦 Checking dependencies..."
python -c "import psutil; print('✅ psutil available')" 2>/dev/null || {
    echo "❌ psutil not found. Installing..."
    pip install psutil
}

# Check if API keys are configured
echo "🔑 Checking API configuration..."
if [[ -f ".env" ]]; then
    echo "✅ .env file found"
    if grep -q "OPENAI_API_KEY" .env 2>/dev/null; then
        echo "✅ OpenAI API key configured"
    else
        echo "⚠️  OpenAI API key not found in .env"
    fi
else
    echo "❌ .env file not found. Please create it with API keys."
    exit 1
fi

# Run the tests
echo ""
echo "🧪 Starting test execution..."
cd test
python system_io_test.py

# Display results
echo ""
echo "📊 Test Results Summary:"
echo "========================"
if [[ -f "test_report.md" ]]; then
    head -10 test_report.md
    echo ""
    echo "📄 Full report available at: test/test_report.md"
    echo "📝 Detailed logs available at: test/test_results.log"
    echo "🔍 Raw data available at: test/test_results.json"
else
    echo "❌ Test report not generated"
fi