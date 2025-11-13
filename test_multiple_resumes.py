#!/usr/bin/env python3
"""
Test multiple resume ingestion and processing flow
"""

import os
import json
import sys
import requests
from typing import List, Dict

# Add current directory to path
sys.path.append('.')

from imaginator_flow import run_analysis

def test_document_reader_service():
    """Test if document-reader-service is accessible"""
    try:
        response = requests.get('https://document-reader-service.onrender.com/health', timeout=10)
        return response.status_code == 200
    except:
        return False

def test_hermes_service():
    """Test if hermes-resume-extractor is accessible"""
    try:
        response = requests.get('https://hermes-resume-extractor.onrender.com/health', timeout=10)
        return response.status_code == 200
    except:
        return False

def test_fastsvm_service():
    """Test if FastSVM service is accessible"""
    try:
        response = requests.get('https://fast-svm-ml-tools-for-skill-and-job.onrender.com/api/v1/health', timeout=10)
        return response.status_code == 200
    except:
        return False

def test_imaginator_service():
    """Test if imaginator service is accessible"""
    try:
        response = requests.get('https://imaginator-resume-cowriter.onrender.com/health', timeout=10)
        return response.status_code == 200
    except:
        return False

def process_multiple_resumes():
    """Process multiple resumes to test the ingestion flow"""
    
    print('🚀 TESTING MULTI-RESUME PROCESSING FLOW')
    print('=' * 70)
    
    # Test service connectivity
    print('\n🔗 SERVICE CONNECTIVITY CHECK')
    print('-' * 40)
    
    services = {
        'document-reader-service': test_document_reader_service(),
        'hermes-resume-extractor': test_hermes_service(),
        'fastsvm-service': test_fastsvm_service(),
        'imaginator-service': test_imaginator_service()
    }
    
    for service, status in services.items():
        status_icon = '✅' if status else '❌'
        print(f'{status_icon} {service}: {'Accessible' if status else 'Not Accessible'}')
    
    # Load multiple resumes
    resume_files = [
        'test/analyst_programmer_resume.txt',
        'test/dogwood_resume.txt'
    ]
    
    job_ad_file = 'test/job_ad.txt'
    
    print(f'\n📥 RESUME FILES TO PROCESS')
    print('-' * 40)
    for resume_file in resume_files:
        if os.path.exists(resume_file):
            with open(resume_file, 'r') as f:
                content = f.read()
            print(f'• {resume_file} ({len(content)} chars)')
        else:
            print(f'❌ {resume_file} - File not found')
    
    if os.path.exists(job_ad_file):
        with open(job_ad_file, 'r') as f:
            job_ad_content = f.read()
        print(f'• {job_ad_file} ({len(job_ad_content)} chars)')
    else:
        print(f'❌ {job_ad_file} - File not found')
    
    # Process each resume
    all_results = {}
    
    for resume_file in resume_files:
        if not os.path.exists(resume_file):
            continue
            
        print(f'\n🔄 PROCESSING: {resume_file}')
        print('-' * 40)
        
        with open(resume_file, 'r') as f:
            resume_text = f.read()
        
        try:
            result = run_analysis(
                resume_text=resume_text,
                job_ad=job_ad_content,
                confidence_threshold=0.7
            )
            
            # Extract key metrics
            resume_name = os.path.basename(resume_file)
            all_results[resume_name] = {
                'experiences_count': len(result.get('experiences', [])),
                'skills_count': len(result.get('aggregate_skills', [])),
                'roles_count': len(result.get('role_suggestions', [])),
                'top_roles': [r['role'] for r in result.get('role_suggestions', [])[:3]],
                'top_skills': result.get('aggregate_skills', [])[:5]
            }
            
            print(f'✅ {resume_name} processed successfully')
            print(f'   • Experiences: {len(result.get("experiences", []))}')
            print(f'   • Skills: {len(result.get("aggregate_skills", []))}')
            print(f'   • Role suggestions: {len(result.get("role_suggestions", []))}')
            
        except Exception as e:
            print(f'❌ Failed to process {resume_file}: {e}')
    
    # Generate comparison report
    print(f'\n📊 MULTI-RESUME COMPARISON REPORT')
    print('=' * 70)
    
    for resume_name, metrics in all_results.items():
        print(f'\n📄 {resume_name}')
        print('-' * 30)
        print(f'Experiences: {metrics["experiences_count"]}')
        print(f'Skills: {metrics["skills_count"]}')
        print(f'Role Suggestions: {metrics["roles_count"]}')
        print(f'Top Roles: {", ".join(metrics["top_roles"])}')
        print(f'Top Skills: {", ".join(metrics["top_skills"])}')
    
    # Save comprehensive results
    output_file = 'multi_resume_test_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'service_status': services,
            'resume_results': all_results,
            'processing_flow': {
                'document_reader': 'https://document-reader-service.onrender.com',
                'hermes_extractor': 'https://hermes-resume-extractor.onrender.com',
                'fastsvm_processor': 'https://fast-svm-ml-tools-for-skill-and-job.onrender.com',
                'imaginator_analyzer': 'https://imaginator-resume-cowriter.onrender.com'
            }
        }, f, indent=2)
    
    print(f'\n📁 Results saved to: {output_file}')
    
    # Summary
    print(f'\n🎯 PROCESSING FLOW SUMMARY')
    print('=' * 70)
    print('1. Frontend uploads documents to document-reader-service')
    print('2. Document reader processes files and returns structured data')
    print('3. Data sent to hermes-resume-extractor for skill extraction')
    print('4. Extracted skills sent to FastSVM for ML processing')
    print('5. Processed data sent to imaginator for final analysis')
    print('6. Results returned to frontend for display')
    
    accessible_services = sum(1 for status in services.values() if status)
    print(f'\n📊 Service Status: {accessible_services}/4 services accessible')
    print(f'📄 Resumes Processed: {len(all_results)}/{len(resume_files)}')

if __name__ == '__main__':
    process_multiple_resumes()