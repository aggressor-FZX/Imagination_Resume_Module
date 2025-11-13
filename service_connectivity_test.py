#!/usr/bin/env python3
"""
Detailed service connectivity test showing exactly how each service was verified
"""

import requests
import json
from datetime import datetime

def test_service_with_details(url: str, service_name: str) -> dict:
    """Test a service and return detailed results"""
    result = {
        'service': service_name,
        'url': url,
        'accessible': False,
        'status_code': None,
        'response_time_ms': None,
        'error': None,
        'response_content': None
    }
    
    try:
        start_time = datetime.now()
        response = requests.get(url, timeout=10)
        end_time = datetime.now()
        
        result['status_code'] = response.status_code
        result['response_time_ms'] = round((end_time - start_time).total_seconds() * 1000, 2)
        result['accessible'] = response.status_code == 200
        
        # Try to get response content
        try:
            result['response_content'] = response.text[:200]  # First 200 chars
        except:
            result['response_content'] = "Unable to read response content"
            
    except requests.exceptions.Timeout:
        result['error'] = "Request timed out (10s)"
    except requests.exceptions.ConnectionError:
        result['error'] = "Connection failed - service may be down"
    except Exception as e:
        result['error'] = f"Unexpected error: {str(e)}"
    
    return result

def main():
    print('🔍 DETAILED SERVICE CONNECTIVITY TEST')
    print('=' * 70)
    print(f'Test performed at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()
    
    # Test each service with their specific health endpoints
    services_to_test = [
        {
            'name': 'document-reader-service',
            'url': 'https://document-reader-service.onrender.com/health',
            'description': 'Document processing service - handles file uploads'
        },
        {
            'name': 'hermes-resume-extractor', 
            'url': 'https://hermes-resume-extractor.onrender.com/health',
            'description': 'Resume skill extraction service'
        },
        {
            'name': 'fastsvm-service',
            'url': 'https://fast-svm-ml-tools-for-skill-and-job.onrender.com/api/v1/health',
            'description': 'ML skill processing service'
        },
        {
            'name': 'imaginator-service',
            'url': 'https://imaginator-resume-cowriter.onrender.com/health',
            'description': 'Final analysis and gap detection service'
        }
    ]
    
    results = []
    
    for service_info in services_to_test:
        print(f'🧪 Testing: {service_info["name"]}')
        print(f'   URL: {service_info["url"]}')
        print(f'   Purpose: {service_info["description"]}')
        
        result = test_service_with_details(service_info['url'], service_info['name'])
        results.append(result)
        
        if result['accessible']:
            print(f'   ✅ Status: ACCESSIBLE (HTTP {result["status_code"]})')
            print(f'   ⏱️  Response Time: {result["response_time_ms"]}ms')
            if result['response_content']:
                print(f'   📄 Response: {result["response_content"][:100]}...')
        else:
            print(f'   ❌ Status: NOT ACCESSIBLE')
            if result['error']:
                print(f'   💥 Error: {result["error"]}')
            elif result['status_code']:
                print(f'   📊 HTTP Status: {result["status_code"]}')
        print()
    
    # Summary
    print('📊 CONNECTIVITY SUMMARY')
    print('=' * 70)
    
    accessible_count = sum(1 for r in results if r['accessible'])
    total_count = len(results)
    
    print(f'✅ Accessible Services: {accessible_count}/{total_count}')
    print(f'❌ Unavailable Services: {total_count - accessible_count}/{total_count}')
    print()
    
    # Detailed results table
    print('📋 DETAILED RESULTS')
    print('-' * 70)
    print(f'| {'Service':<25} | {'Status':<10} | {'Response Time':<12} | {'Details':<20} |')
    print('|' + '-'*25 + '|' + '-'*10 + '|' + '-'*12 + '|' + '-'*20 + '|')
    
    for result in results:
        status = '✅ OK' if result['accessible'] else '❌ FAIL'
        response_time = f'{result["response_time_ms"]}ms' if result['response_time_ms'] else 'N/A'
        details = result['error'] or f'HTTP {result["status_code"]}' if result['status_code'] else 'No response'
        
        print(f'| {result["service"]:<25} | {status:<10} | {response_time:<12} | {details:<20} |')
    
    # Save detailed results
    output_file = 'service_connectivity_detailed.json'
    with open(output_file, 'w') as f:
        json.dump({
            'test_timestamp': datetime.now().isoformat(),
            'results': results,
            'summary': {
                'total_services': total_count,
                'accessible_services': accessible_count,
                'success_rate': f'{round(accessible_count/total_count*100, 1)}%'
            }
        }, f, indent=2)
    
    print(f'\n📁 Detailed results saved to: {output_file}')
    
    # Explain the verification method
    print(f'\n🔍 VERIFICATION METHODOLOGY')
    print('=' * 70)
    print('1. HTTP GET requests to each service\'s health endpoint')
    print('2. 10-second timeout to prevent hanging')
    print('3. Status code 200 = service is accessible')
    print('4. Response time measurement for performance monitoring')
    print('5. Error handling for connection failures and timeouts')
    print('6. Response content inspection to verify service functionality')

if __name__ == '__main__':
    main()