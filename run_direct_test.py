#!/usr/bin/env python3
"""
Direct test of Imaginator system using test files
"""

import os
import json
import sys

# Add current directory to path
sys.path.append('.')

from imaginator_flow import run_analysis

def main():
    # Read test files
    with open('test/analyst_programmer_resume.txt', 'r') as f:
        resume_text = f.read()

    with open('test/job_ad.txt', 'r') as f:
        job_ad_text = f.read()

    print('🚀 Starting Imaginator System Test')
    print('=' * 60)

    # INPUT SECTION
    print('\n📥 INPUT DATA INGESTED')
    print('-' * 30)
    print(f'Resume File: analyst_programmer_resume.txt')
    print(f'Resume Size: {len(resume_text)} characters')
    print(f'Job Ad File: job_ad.txt')
    print(f'Job Ad Size: {len(job_ad_text)} characters')
    print(f'Confidence Threshold: 0.7')

    print('\n📄 RESUME CONTENT PREVIEW (first 300 chars):')
    print('-' * 50)
    print(resume_text[:300] + '...' if len(resume_text) > 300 else resume_text)

    print('\n📋 JOB AD CONTENT PREVIEW (first 300 chars):')
    print('-' * 50)
    print(job_ad_text[:300] + '...' if len(job_ad_text) > 300 else job_ad_text)

    print('\n⚙️  PROCESSING CONFIGURATION')
    print('-' * 30)
    print('• Extracting work experiences from resume')
    print('• Identifying skills from experience descriptions')
    print('• Processing skills with confidence filtering')
    print('• Generating role suggestions based on skill matches')
    print('• Performing gap analysis between resume and job requirements')
    print('• Calculating run metrics (tokens, costs, failures)')

    print('\n🔄 RUNNING ANALYSIS...')
    print('-' * 30)

    try:
        # Run the analysis
        result = run_analysis(
            resume_text=resume_text,
            job_ad=job_ad_text,
            confidence_threshold=0.7
        )

        # OUTPUT SECTION
        print('\n📊 ANALYSIS RESULTS')
        print('=' * 60)

        print('\n🏢 EXPERIENCES EXTRACTED')
        print('-' * 30)
        for i, exp in enumerate(result.get('experiences', []), 1):
            print(f'{i}. {exp.get("title_line", "Unknown")}')
            skills = exp.get('skills', [])
            if skills:
                print(f'   Skills: {", ".join(skills)}')
            snippet = exp.get('snippet', '')
            if len(snippet) > 100:
                print(f'   Context: {snippet[:100]}...')
            print()

        print('\n🎯 AGGREGATE SKILLS IDENTIFIED')
        print('-' * 30)
        skills = result.get('aggregate_skills', [])
        if skills:
            for skill in skills:
                print(f'• {skill}')
        else:
            print('No skills extracted from keyword matching')

        print('\n📈 PROCESSED SKILLS (Confidence-Based)')
        print('-' * 30)
        processed = result.get('processed_skills', {})
        high_conf = processed.get('high_confidence_skills', [])
        med_conf = processed.get('medium_confidence_skills', [])
        low_conf = processed.get('low_confidence_skills', [])

        print(f'High Confidence (≥0.7): {len(high_conf)} skills')
        if high_conf:
            for skill in high_conf:
                print(f'  ✓ {skill}')

        print(f'Medium Confidence (0.5-0.7): {len(med_conf)} skills')
        if med_conf:
            for skill in med_conf:
                print(f'  ~ {skill}')

        print(f'Low Confidence (<0.5): {len(low_conf)} skills')
        if low_conf:
            for skill in low_conf:
                print(f'  ? {skill}')

        print('\n👔 ROLE SUGGESTIONS GENERATED')
        print('-' * 30)
        roles = result.get('role_suggestions', [])
        if roles:
            for role in roles:
                role_name = role.get('role', 'Unknown')
                score = role.get('score', 0)
                matched = role.get('matched_skills', [])
                print(f'• {role_name} (Match Score: {score:.2f})')
                if matched:
                    print(f'  Matching Skills: {", ".join(matched)}')
                print()
        else:
            print('No role suggestions generated')

        print('\n🔍 GAP ANALYSIS RESULTS')
        print('-' * 30)
        gap = result.get('gap_analysis', '')
        if gap:
            try:
                # Try to parse as JSON for better formatting
                gap_data = json.loads(gap)
                print('Gap Analysis (Structured):')
                if 'gap_analysis' in gap_data:
                    ga = gap_data['gap_analysis']
                    if 'critical_gaps' in ga and ga['critical_gaps']:
                        print(f'Critical Gaps: {", ".join(ga["critical_gaps"])}')
                    if 'nice_to_have_gaps' in ga and ga['nice_to_have_gaps']:
                        print(f'Nice-to-Have Gaps: {", ".join(ga["nice_to_have_gaps"])}')
                    if 'gap_bridging_strategy' in ga:
                        print(f'Strategy: {ga["gap_bridging_strategy"]}')
                else:
                    print(json.dumps(gap_data, indent=2))
            except json.JSONDecodeError:
                # Fallback to text display
                print('Gap Analysis (Text):')
                print(gap[:500] + '...' if len(gap) > 500 else gap)
        else:
            print('No gap analysis performed')

        print('\n📊 RUN METRICS')
        print('-' * 30)
        metrics = result.get('run_metrics', {})
        print(f'Total Tokens Used: {metrics.get("total_tokens", 0)}')
        print(f'Estimated Cost: ${metrics.get("estimated_cost_usd", 0):.4f}')
        failures = metrics.get('failures', [])
        print(f'API Failures: {len(failures)}')
        if failures:
            print('Failure Details:')
            for failure in failures[:3]:  # Show first 3 failures
                print(f'  • {failure}')

        # Save to JSON file
        with open('test_output.json', 'w') as f:
            json.dump(result, f, indent=2)

        # Also save a formatted text log
        with open('test_output.log', 'w') as f:
            f.write('IMAGINATOR SYSTEM TEST RESULTS\n')
            f.write('=' * 60 + '\n\n')
            f.write('INPUT DATA INGESTED:\n')
            f.write('-' * 30 + '\n')
            f.write(f'Resume: analyst_programmer_resume.txt ({len(resume_text)} chars)\n')
            f.write(f'Job Ad: job_ad.txt ({len(job_ad_text)} chars)\n')
            f.write('Confidence Threshold: 0.7\n\n')

            f.write('EXPERIENCES EXTRACTED:\n')
            for i, exp in enumerate(result.get('experiences', []), 1):
                f.write(f'{i}. {exp.get("title_line", "Unknown")}\n')
                skills = exp.get('skills', [])
                if skills:
                    f.write(f'   Skills: {", ".join(skills)}\n')
                f.write('\n')

            f.write('AGGREGATE SKILLS:\n')
            for skill in result.get('aggregate_skills', []):
                f.write(f'- {skill}\n')
            f.write('\n')

            f.write('ROLE SUGGESTIONS:\n')
            for role in result.get('role_suggestions', []):
                f.write(f'- {role.get("role", "Unknown")} (score: {role.get("score", 0):.2f})\n')
            f.write('\n')

            f.write('GAP ANALYSIS:\n')
            gap_text = result.get('gap_analysis', '')
            f.write(gap_text[:1000] + '...\n' if len(gap_text) > 1000 else gap_text + '\n')
            f.write('\n')

            f.write('RUN METRICS:\n')
            metrics = result.get('run_metrics', {})
            f.write(f'Total tokens: {metrics.get("total_tokens", 0)}\n')
            f.write(f'Estimated cost: ${metrics.get("estimated_cost_usd", 0):.4f}\n')
            f.write(f'Failures: {len(metrics.get("failures", []))}\n')

        print('\n✅ Test completed successfully!')
        print('📁 Output files saved:')
        print('   • test_output.json (complete JSON data)')
        print('   • test_output.log (formatted summary)')
        print('\n🎯 Key Insights:')
        print(f'   • {len(result.get("experiences", []))} experiences extracted')
        print(f'   • {len(result.get("aggregate_skills", []))} skills identified')
        print(f'   • {len(result.get("role_suggestions", []))} role suggestions generated')
        print(f'   • {len(metrics.get("failures", []))} API failures handled gracefully')

    except Exception as e:
        print(f'\n❌ Test failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()