import os
import json
import requests
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_test():
    """Runs a test of the Imaginator system."""
    test_dir = 'test'
    resume_path = os.path.join(test_dir, 'analyst_programmer_resume.txt')
    job_ad_path = os.path.join(test_dir, 'job_ad.txt')
    output_path = os.path.join(test_dir, 'test_result.json')

    with open(resume_path, 'r') as f:
        resume_text = f.read()

    with open(job_ad_path, 'r') as f:
        job_ad_text = f.read()

    # --- IMPORTANT ---
    # Replace with your actual API key
    # You can also set this as an environment variable
    api_key = os.environ.get("IMAGINATOR_API_KEY", "your-api-key-here")
    # --- IMPORTANT ---

    headers = {
        'Content-Type': 'application/json',
        'X-API-Key': api_key
    }

    data = {
        "resume_text": resume_text,
        "job_ad": job_ad_text,
        "confidence_threshold": 0.7
    }

    # Assuming the app is running on localhost:8000
    url = "http://localhost:8000/analyze"

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)

        print(f"Test successful. Results saved to {output_path}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    run_test()
