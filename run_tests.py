#!/usr/bin/env python3
"""
Test runner script for the Imaginator FastAPI application
Based on Context7 research findings for pytest configuration
"""

import subprocess
import sys
import os
from pathlib import Path


def run_tests(test_type="all", coverage=False, verbose=True):
    """Run tests with specified configuration"""

    # Ensure we're in the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Base pytest command
    cmd = ["python", "-m", "pytest"]

    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])

    if coverage:
        cmd.extend([
            "--cov=.",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-fail-under=80"
        ])

    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    # Add test directories and files
    cmd.extend(["tests/", "test_api.py"])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    return result.returncode == 0


def run_type_checking():
    """Run type checking with mypy if available"""
    try:
        import mypy  # type: ignore
        print("Running mypy type checking...")
        result = subprocess.run([
            "python", "-m", "mypy",
            "app.py", "config.py", "models.py",
            "--ignore-missing-imports"
        ], capture_output=False)
        return result.returncode == 0
    except ImportError:
        print("mypy not installed, skipping type checking")
        return True


def run_linting():
    """Run code linting with flake8/black if available"""
    success = True

    # Try black formatting check
    try:
        print("Checking code formatting with black...")
        result = subprocess.run([
            "python", "-m", "black", "--check", "--diff",
            "app.py", "config.py", "models.py", "tests/"
        ], capture_output=False)
        if result.returncode != 0:
            print("Code formatting issues found. Run 'black .' to fix.")
            success = False
    except ImportError:
        print("black not installed, skipping formatting check")

    # Try isort import sorting check
    try:
        print("Checking import sorting with isort...")
        result = subprocess.run([
            "python", "-m", "isort", "--check-only", "--diff",
            "app.py", "config.py", "models.py", "tests/"
        ], capture_output=False)
        if result.returncode != 0:
            print("Import sorting issues found. Run 'isort .' to fix.")
            success = False
    except ImportError:
        print("isort not installed, skipping import check")

    return success


def main():
    """Main test runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Run Imaginator tests")
    parser.add_argument(
        "--type", choices=["all", "unit", "integration", "fast"],
        default="all", help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage", action="store_true",
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Run tests quietly"
    )
    parser.add_argument(
        "--lint", action="store_true",
        help="Run linting checks"
    )
    parser.add_argument(
        "--type-check", action="store_true",
        help="Run type checking"
    )

    args = parser.parse_args()

    print("🚀 Running Imaginator Test Suite")
    print("=" * 40)

    all_passed = True

    # Run linting if requested
    if args.lint:
        print("\n📏 Running code quality checks...")
        if not run_linting():
            all_passed = False

    # Run type checking if requested
    if args.type_check:
        print("\n🔍 Running type checking...")
        if not run_type_checking():
            all_passed = False

    # Run tests
    print(f"\n🧪 Running {args.type} tests...")
    if not run_tests(args.type, args.coverage, not args.quiet):
        all_passed = False

    print("\n" + "=" * 40)
    if all_passed:
        print("✅ All checks passed!")
        sys.exit(0)
    else:
        print("❌ Some checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()