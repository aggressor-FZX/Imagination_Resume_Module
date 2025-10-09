#!/usr/bin/env python3
"""
Comprehensive System I/O Test Suite

Tests the Generative Resume Co-Writer system inputs and outputs,
measuring performance, timing, and token counting accuracy.
Records detailed logs and validates against the I/O specification.
"""

import json
import time
import logging
import os
import sys
from typing import Dict, Any, List
from datetime import datetime
import psutil
import tracemalloc

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imaginator_flow import (
    run_analysis, run_generation, run_criticism,
    validate_output_schema, OUTPUT_SCHEMA, RUN_METRICS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test/test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SystemIOTest')

class PerformanceMonitor:
    """Monitor system performance during tests"""

    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None

    def start(self):
        """Start performance monitoring"""
        tracemalloc.start()
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.start_cpu = psutil.cpu_percent(interval=None)

    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return metrics"""
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        end_cpu = psutil.cpu_percent(interval=None)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            'duration_seconds': end_time - self.start_time,
            'memory_used_mb': end_memory - self.start_memory,
            'peak_memory_mb': peak / 1024 / 1024,
            'cpu_percent': end_cpu
        }

class SystemIOTester:
    """Comprehensive tester for system I/O specification"""

    def __init__(self):
        self.test_results = []
        self.sample_data = self.load_sample_data()

    def load_sample_data(self) -> Dict[str, Any]:
        """Load sample test data"""
        data = {}

        # Load sample resume
        try:
            with open('sample_resume.txt', 'r', encoding='utf-8') as f:
                data['resume_text'] = f.read()
        except FileNotFoundError:
            data['resume_text'] = """
            John Doe
            Senior Software Engineer

            EXPERIENCE:
            Senior Software Engineer at TechCorp (2020-Present)
            - Developed web applications using Python, Django, and PostgreSQL
            - Led team of 5 developers on microservices architecture
            - Implemented CI/CD pipelines using Jenkins and Docker

            Software Engineer at StartupXYZ (2018-2020)
            - Built REST APIs using Flask and SQLAlchemy
            - Optimized database queries improving performance by 40%
            - Collaborated with cross-functional teams

            EDUCATION:
            BS Computer Science, University of Technology (2018)
            """

        # Load sample job ad
        try:
            with open('sample_job_ad.txt', 'r', encoding='utf-8') as f:
                data['job_ad'] = f.read()
        except FileNotFoundError:
            data['job_ad'] = """
            Senior Python Developer

            We are looking for an experienced Python developer to join our team.

            Requirements:
            - 5+ years of Python development experience
            - Strong knowledge of Django, Flask, or FastAPI
            - Experience with PostgreSQL or similar databases
            - Familiarity with Docker and containerization
            - Experience with AWS or cloud platforms
            - Knowledge of REST API design
            - Git version control
            - Agile development methodologies

            Nice to have:
            - Experience with React or JavaScript frameworks
            - Knowledge of machine learning libraries
            - DevOps experience
            """

        # Load structured skills data
        try:
            with open('sample_skills.json', 'r', encoding='utf-8') as f:
                data['skills_json'] = json.load(f)
        except FileNotFoundError:
            data['skills_json'] = {
                "title": "Software Engineer",
                "canonical_title": "Software Engineer",
                "skills": [
                    {"skill": "Python", "confidence": 0.95},
                    {"skill": "Django", "confidence": 0.90},
                    {"skill": "PostgreSQL", "confidence": 0.85},
                    {"skill": "Docker", "confidence": 0.80},
                    {"skill": "AWS", "confidence": 0.75},
                    {"skill": "JavaScript", "confidence": 0.70}
                ]
            }

        # Load domain insights
        try:
            with open('sample_insights.json', 'r', encoding='utf-8') as f:
                data['insights_json'] = json.load(f)
        except FileNotFoundError:
            data['insights_json'] = {
                "domain": "software_development",
                "market_trends": ["cloud_computing", "microservices", "devops"],
                "skill_priorities": {
                    "high": ["Python", "AWS", "Docker"],
                    "medium": ["Django", "PostgreSQL", "JavaScript"],
                    "low": ["PHP", "jQuery"]
                }
            }

        return data

    def validate_output_structure(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate output against I/O specification"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        # Check required fields
        required_fields = [
            'experiences', 'aggregate_skills', 'processed_skills',
            'domain_insights', 'gap_analysis', 'suggested_experiences'
        ]

        for field in required_fields:
            if field not in output:
                validation_result['errors'].append(f"Missing required field: {field}")
                validation_result['is_valid'] = False

        # Validate experiences structure
        if 'experiences' in output:
            if not isinstance(output['experiences'], list):
                validation_result['errors'].append("experiences must be a list")
                validation_result['is_valid'] = False
            else:
                for i, exp in enumerate(output['experiences']):
                    if not isinstance(exp, dict):
                        validation_result['errors'].append(f"experiences[{i}] must be a dict")
                        validation_result['is_valid'] = False
                    elif not all(k in exp for k in ['title_line', 'skills', 'snippet']):
                        validation_result['errors'].append(f"experiences[{i}] missing required keys")
                        validation_result['is_valid'] = False

        # Validate aggregate_skills
        if 'aggregate_skills' in output:
            if not isinstance(output['aggregate_skills'], list):
                validation_result['errors'].append("aggregate_skills must be a list")
                validation_result['is_valid'] = False

        # Validate suggested_experiences structure
        if 'suggested_experiences' in output:
            if not isinstance(output['suggested_experiences'], dict):
                validation_result['errors'].append("suggested_experiences must be a dict")
                validation_result['is_valid'] = False
            elif not all(k in output['suggested_experiences'] for k in ['bridging_gaps', 'metric_improvements']):
                validation_result['errors'].append("suggested_experiences missing required keys")
                validation_result['is_valid'] = False

        # Check run_metrics if present
        if 'run_metrics' in output:
            metrics = output['run_metrics']
            if not isinstance(metrics, dict):
                validation_result['warnings'].append("run_metrics should be a dict")
            else:
                expected_keys = ['calls', 'total_prompt_tokens', 'total_completion_tokens',
                               'total_tokens', 'estimated_cost_usd', 'failures']
                for key in expected_keys:
                    if key not in metrics:
                        validation_result['warnings'].append(f"run_metrics missing: {key}")

        return validation_result

    def test_token_accuracy(self, run_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Test token counting accuracy"""
        result = {
            'total_accuracy': True,
            'discrepancies': [],
            'calculated_total': 0,
            'reported_total': 0
        }

        if 'calls' not in run_metrics:
            result['discrepancies'].append("No calls data in run_metrics")
            result['total_accuracy'] = False
            return result

        calculated_prompt = 0
        calculated_completion = 0
        calculated_total = 0

        for call in run_metrics['calls']:
            if 'prompt_tokens' in call:
                calculated_prompt += call['prompt_tokens']
            if 'completion_tokens' in call:
                calculated_completion += call['completion_tokens']
            if 'total_tokens' in call:
                calculated_total += call['total_tokens']

        # Check prompt tokens
        reported_prompt = run_metrics.get('total_prompt_tokens', 0)
        if abs(calculated_prompt - reported_prompt) > 1:  # Allow small rounding differences
            result['discrepancies'].append(
                f"Prompt tokens mismatch: calculated={calculated_prompt}, reported={reported_prompt}")
            result['total_accuracy'] = False

        # Check completion tokens
        reported_completion = run_metrics.get('total_completion_tokens', 0)
        if abs(calculated_completion - reported_completion) > 1:
            result['discrepancies'].append(
                f"Completion tokens mismatch: calculated={calculated_completion}, reported={reported_completion}")
            result['total_accuracy'] = False

        # Check total tokens
        reported_total = run_metrics.get('total_tokens', 0)
        if abs(calculated_total - reported_total) > 1:
            result['discrepancies'].append(
                f"Total tokens mismatch: calculated={calculated_total}, reported={reported_total}")
            result['total_accuracy'] = False

        result['calculated_total'] = calculated_total
        result['reported_total'] = reported_total

        return result

    def run_single_test(self, test_name: str, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case"""
        logger.info(f"Starting test: {test_name}")

        # Reset RUN_METRICS
        RUN_METRICS.update({
            "calls": [],
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "failures": []
        })

        monitor = PerformanceMonitor()
        monitor.start()

        test_result = {
            'test_name': test_name,
            'config': test_config,
            'success': False,
            'error': None,
            'performance': {},
            'validation': {},
            'token_accuracy': {},
            'output': None
        }

        # Create temporary files for structured data
        temp_files = []
        skills_file = None
        insights_file = None

        try:
            # Write structured data to temporary files if provided
            if 'skills_json' in test_config:
                import tempfile
                skills_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                json.dump(test_config['skills_json'], skills_file)
                skills_file.close()
                temp_files.append(skills_file.name)

            if 'insights_json' in test_config:
                import tempfile
                insights_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                json.dump(test_config['insights_json'], insights_file)
                insights_file.close()
                temp_files.append(insights_file.name)

            # Stage 1: Analysis
            logger.info(f"Running analysis for {test_name}")
            analysis_result = run_analysis(
                resume_text=test_config.get('resume_text', self.sample_data['resume_text']),
                job_ad=test_config.get('job_ad', self.sample_data['job_ad']),
                extracted_skills_json=skills_file.name if skills_file else None,
                domain_insights_json=insights_file.name if insights_file else None,
                confidence_threshold=test_config.get('confidence_threshold', 0.7)
            )

            # Stage 2: Generation
            logger.info(f"Running generation for {test_name}")
            generation_result = run_generation(
                analysis_json=analysis_result,
                job_ad=test_config.get('job_ad', self.sample_data['job_ad'])
            )

            # Stage 3: Criticism
            logger.info(f"Running criticism for {test_name}")
            criticism_result = run_criticism(
                generated_suggestions=generation_result,
                job_ad=test_config.get('job_ad', self.sample_data['job_ad'])
            )

            # Assemble final output
            output = {
                **analysis_result,
                **criticism_result,
                "run_metrics": RUN_METRICS.copy()
            }

            # Validate output
            validation = self.validate_output_structure(output)
            token_accuracy = self.test_token_accuracy(RUN_METRICS)

            test_result.update({
                'success': True,
                'validation': validation,
                'token_accuracy': token_accuracy,
                'output': output
            })

            logger.info(f"Test {test_name} completed successfully")

        except Exception as e:
            logger.error(f"Test {test_name} failed: {str(e)}")
            test_result.update({
                'success': False,
                'error': str(e)
            })

        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

            # Stop performance monitoring
            test_result['performance'] = monitor.stop()

        return test_result

    def run_test_suite(self) -> List[Dict[str, Any]]:
        """Run comprehensive test suite"""
        logger.info("Starting comprehensive system I/O test suite")

        test_cases = [
            {
                'name': 'basic_text_only',
                'config': {}
            },
            {
                'name': 'with_structured_skills',
                'config': {
                    'skills_json': self.sample_data['skills_json']
                }
            },
            {
                'name': 'with_domain_insights',
                'config': {
                    'insights_json': self.sample_data['insights_json']
                }
            },
            {
                'name': 'with_all_structured_data',
                'config': {
                    'skills_json': self.sample_data['skills_json'],
                    'insights_json': self.sample_data['insights_json']
                }
            },
            {
                'name': 'high_confidence_threshold',
                'config': {
                    'skills_json': self.sample_data['skills_json'],
                    'confidence_threshold': 0.9
                }
            },
            {
                'name': 'low_confidence_threshold',
                'config': {
                    'skills_json': self.sample_data['skills_json'],
                    'confidence_threshold': 0.5
                }
            }
        ]

        results = []
        for test_case in test_cases:
            result = self.run_single_test(test_case['name'], test_case['config'])
            results.append(result)
            self.test_results.append(result)

        return results

    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("# System I/O Test Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")

        # Summary statistics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r['success'])
        failed_tests = total_tests - successful_tests

        report.append("## Summary")
        report.append(f"- Total Tests: {total_tests}")
        report.append(f"- Successful: {successful_tests}")
        report.append(f"- Failed: {failed_tests}")
        report.append(f"- Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        report.append("")

        # Performance summary
        durations = [r['performance']['duration_seconds'] for r in self.test_results if r['success']]
        if durations:
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            report.append("## Performance Summary")
            report.append(f"- Average Duration: {avg_duration:.2f}s")
            report.append(f"- Min Duration: {min_duration:.2f}s")
            report.append(f"- Max Duration: {max_duration:.2f}s")
            report.append("")

        # Detailed results
        report.append("## Detailed Results")
        for result in self.test_results:
            report.append(f"### {result['test_name']}")
            report.append(f"- Success: {result['success']}")

            if result['success']:
                perf = result['performance']
                report.append(f"- Duration: {perf['duration_seconds']:.2f}s")
                report.append(f"- Memory Used: {perf['memory_used_mb']:.1f}MB")
                report.append(f"- Peak Memory: {perf['peak_memory_mb']:.1f}MB")

                # Validation results
                val = result['validation']
                if val['errors']:
                    report.append(f"- Validation Errors: {len(val['errors'])}")
                    for error in val['errors'][:3]:  # Show first 3 errors
                        report.append(f"  - {error}")

                # Token accuracy
                token_acc = result['token_accuracy']
                if not token_acc['total_accuracy']:
                    report.append(f"- Token Accuracy Issues: {len(token_acc['discrepancies'])}")
                    for disc in token_acc['discrepancies'][:2]:  # Show first 2
                        report.append(f"  - {disc}")

                # Run metrics summary
                if 'output' in result and 'run_metrics' in result['output']:
                    metrics = result['output']['run_metrics']
                    report.append(f"- API Calls: {len(metrics.get('calls', []))}")
                    report.append(f"- Total Tokens: {metrics.get('total_tokens', 0)}")
                    report.append(f"- Estimated Cost: ${metrics.get('estimated_cost_usd', 0):.4f}")
            else:
                report.append(f"- Error: {result['error']}")

            report.append("")

        return "\n".join(report)

def main():
    """Main test execution"""
    print("🚀 Starting System I/O Test Suite")
    print("=" * 50)

    tester = SystemIOTester()
    results = tester.run_test_suite()

    # Generate and save report
    report = tester.generate_report()

    with open('test/test_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    # Save detailed results
    with open('test/test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    print("✅ Test suite completed!")
    print(f"📊 Results saved to test/test_report.md and test/test_results.json")
    print(f"📝 Logs saved to test/test_results.log")

    # Print summary
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    print(f"📈 Success Rate: {successful}/{total} ({(successful/total)*100:.1f}%)")

if __name__ == "__main__":
    main()