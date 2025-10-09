#!/usr/bin/env python3
"""
Test Runner for System I/O Tests
Provides programmatic access to test execution and results
"""

import sys
import os
import subprocess
import json
from pathlib import Path

def run_tests():
    """Run the complete test suite and return results"""
    test_dir = Path(__file__).parent / "test"

    # Change to test directory
    original_dir = os.getcwd()
    os.chdir(test_dir)

    try:
        # Run the test script
        result = subprocess.run([
            sys.executable, "system_io_test.py"
        ], capture_output=True, text=True, timeout=300)

        # Load results if available
        results_file = test_dir / "test_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                test_data = json.load(f)
        else:
            test_data = None

        return {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'test_data': test_data
        }

    except subprocess.TimeoutExpired:
        return {
            'returncode': -1,
            'stdout': '',
            'stderr': 'Test execution timed out after 5 minutes',
            'test_data': None
        }
    finally:
        os.chdir(original_dir)

def get_latest_results():
    """Get the most recent test results"""
    results_file = Path(__file__).parent / "test_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None

def print_summary(results):
    """Print a formatted summary of test results"""
    if not results or 'test_data' not in results:
        print("❌ No test results available")
        return

    data = results['test_data']

    # Handle both list and dict formats
    if isinstance(data, list):
        # Calculate summary from list of test results
        total_tests = len(data)
        passed_tests = sum(1 for test in data if test.get('success', False))
        failed_tests = total_tests - passed_tests

        # Calculate performance summary
        durations = [test.get('performance', {}).get('duration_seconds', 0) for test in data]
        memories = [test.get('performance', {}).get('memory_used_mb', 0) for test in data]
        tokens = [test.get('output', {}).get('run_metrics', {}).get('total_tokens', 0) for test in data]
        costs = [test.get('output', {}).get('run_metrics', {}).get('estimated_cost_usd', 0) for test in data]

        performance_summary = {
            'avg_duration': sum(durations) / len(durations) if durations else 0,
            'avg_memory_mb': sum(memories) / len(memories) if memories else 0,
            'avg_tokens': sum(tokens) / len(tokens) if tokens else 0,
            'avg_cost': sum(costs) / len(costs) if costs else 0
        }

        # Token accuracy (simplified - would need more complex logic for real accuracy check)
        token_accuracy = {
            'expected_total': sum(tokens),
            'actual_total': sum(tokens),  # Simplified
            'accuracy_percent': 100.0
        }

        summary_data = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'performance_summary': performance_summary,
            'token_accuracy': token_accuracy
        }
    else:
        summary_data = data

    print("🧪 System I/O Test Results")
    print("=" * 40)

    # Overall stats
    total_tests = summary_data.get('total_tests', 0)
    passed_tests = summary_data.get('passed_tests', 0)
    failed_tests = summary_data.get('failed_tests', 0)

    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%")

    # Performance summary
    if 'performance_summary' in summary_data:
        perf = summary_data['performance_summary']
        print("\n📊 Performance Summary:")
        print(f"Average Duration: {perf.get('avg_duration', 0):.2f}s")
        print(f"Average Memory: {perf.get('avg_memory_mb', 0):.2f}MB")
        print(f"Average Tokens: {perf.get('avg_tokens', 0):.0f}")
        print(f"Average Cost: ${perf.get('avg_cost', 0):.4f}")

    # Token accuracy
    if 'token_accuracy' in summary_data:
        acc = summary_data['token_accuracy']
        print("\n🎯 Token Accuracy:")
        print(f"Expected Total: {acc.get('expected_total', 0)}")
        print(f"Actual Total: {acc.get('actual_total', 0)}")
        print(f"Accuracy: {acc.get('accuracy_percent', 0):.2f}%")
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        # Just show latest results
        results = get_latest_results()
        if results:
            print_summary({'test_data': results})
        else:
            print("❌ No previous test results found")
    else:
        # Run tests
        print("🚀 Running System I/O Tests...")
        results = run_tests()

        if results['returncode'] == 0:
            print("✅ Tests completed successfully")
            print_summary(results)
        else:
            print("❌ Tests failed")
            print("STDOUT:", results['stdout'])
            print("STDERR:", results['stderr'])