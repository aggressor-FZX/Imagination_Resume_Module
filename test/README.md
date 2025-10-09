# System I/O Test Suite

This directory contains comprehensive tests for the Generative Resume Co-Writer system, focusing on input/output validation, performance monitoring, and accuracy verification.

## Files

- `system_io_test.py`: Main test script that exercises all system inputs and outputs
- `run_tests.sh`: Bash script for automated test execution with environment checks
- `run_tests.py`: Python wrapper for programmatic test execution and result display
- `requirements-test.txt`: Additional dependencies for testing (psutil for performance monitoring)
- `test_report.md`: Generated test report with detailed results
- `test_results.json`: Raw test results in JSON format
- `test_results.log`: Detailed execution logs

## Test Coverage

The test suite validates:

### Input Variations
- **Basic text input**: Resume text + job description only
- **Structured skills**: With confidence-scored skills data
- **Domain insights**: With industry-specific context
- **Combined structured data**: All optional inputs together
- **Confidence thresholds**: Different filtering levels (0.5, 0.7, 0.9)

### Output Validation
- **Schema compliance**: Validates against the I/O specification
- **Required fields**: Ensures all mandatory outputs are present
- **Data structure**: Verifies correct types and nested structures
- **Content validation**: Checks for reasonable output content

### Performance Metrics
- **Execution time**: Duration for each processing stage
- **Memory usage**: RAM consumption tracking
- **CPU utilization**: Processor usage monitoring
- **Peak memory**: Maximum memory allocation

### Token Accuracy
- **Per-call validation**: Verifies individual API call token counts
- **Total aggregation**: Checks sum calculations across all calls
- **Cost estimation**: Validates pricing calculations
- **Discrepancy detection**: Identifies token counting errors

## Running Tests

### Quick Start (Recommended)
```bash
# From project root - automated environment check and execution
./test/run_tests.sh
```

### Python Runner
```bash
# Run full test suite
python test/run_tests.py

# Show summary of latest results
python test/run_tests.py --summary
```

### Manual Execution
```bash
# Install test dependencies
pip install -r test/requirements-test.txt

# Ensure API keys are configured in .env
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here

# Run tests manually
cd test
python system_io_test.py
```

## Test Results

After running, check:

- **`test_report.md`**: Human-readable summary with performance metrics
- **`test_results.json`**: Detailed raw data for analysis
- **`test_results.log`**: Execution logs and any errors

## Performance Benchmarks

Based on recent test executions with OpenAI GPT-4 and Anthropic Claude:

- **Test Duration**: 11.27-18.27 seconds per complete test case
- **Memory Usage**: 0-16.3 MB peak consumption
- **API Calls**: 3 calls per test (analysis + generation + criticism)
- **Token Usage**: 3,600-4,000 tokens total per test
- **API Cost**: $0.0029-$0.0034 per test execution
- **Success Rate**: 100% (all test cases passing)
- **Token Accuracy**: 100% (API-reported counts match calculations)

## Troubleshooting

### Common Issues

1. **Missing API keys**: Ensure `.env` file exists with valid keys
2. **Import errors**: Run from project root or adjust Python path
3. **Memory issues**: Tests may require 500MB+ available RAM
4. **Rate limits**: API rate limiting may cause delays or failures

### Debug Mode

For detailed debugging, modify the logging level in `system_io_test.py`:

```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## Test Data

The tests use sample data files from the project root:
- `sample_resume.txt`: Test resume content
- `sample_job_ad.txt`: Test job description
- `sample_skills.json`: Structured skills data
- `sample_insights.json`: Domain insights data

## Extending Tests

To add new test cases, modify the `test_cases` list in `run_test_suite()`:

```python
{
    'name': 'your_test_name',
    'config': {
        'resume_text': 'custom resume...',
        'job_ad': 'custom job...',
        'skills_json': {...},
        'confidence_threshold': 0.8
    }
}
```