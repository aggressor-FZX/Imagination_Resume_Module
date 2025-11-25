
import json
import os
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util


def calculate_semantic_similarity(generated_text, ideal_text):
    """Calculates the semantic similarity between two texts."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding1 = model.encode(generated_text, convert_to_tensor=True)
    embedding2 = model.encode(ideal_text, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding1, embedding2).item()


def calculate_actionability_score(suggestions):
    """Calculates an actionability score based on the presence of metrics and action verbs."""
    score = 0
    for suggestion in suggestions:
        # Reward suggestions with numbers (metrics)
        if any(char.isdigit() for char in suggestion):
            score += 1
        # Reward suggestions with strong action verbs
        action_verbs = ['developed', 'led', 'managed', 'created', 'implemented', 'optimized']
        if any(verb in suggestion.lower() for verb in action_verbs):
            score += 1
    return score


def calculate_keyword_coverage(generated_skills, ideal_skills):
    """Calculates the percentage of ideal skills covered by the generated skills."""
    return len(set(generated_skills) & set(ideal_skills)) / len(set(ideal_skills))


def evaluate_quality(generated_output, ideal_output):
    """Evaluates the quality of the generated output against the ideal output."""
    similarity = calculate_semantic_similarity(generated_output['gap_analysis'], ideal_output['gap_analysis'])
    
    # The API returns a list of suggestions. We need to iterate through them.
    suggestions = []
    if 'suggested_experiences' in generated_output and generated_output['suggested_experiences']:
        for experience in generated_output['suggested_experiences']:
            if 'refined_suggestions' in experience and experience['refined_suggestions']:
                suggestions.extend(experience['refined_suggestions'])

    actionability = calculate_actionability_score(suggestions)
    coverage = calculate_keyword_coverage(generated_output['aggregate_skills'], ideal_output['aggregate_skills'])

    return {
        'semantic_similarity': similarity,
        'actionability_score': actionability,
        'keyword_coverage': coverage
    }


if __name__ == '__main__':
    # Load environment variables from .env file
    load_dotenv()

    # Load the golden dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    golden_data_path = os.path.join(script_dir, '..', 'golden_dataset', 'golden_case_1.json')
    with open(golden_data_path) as f:
        golden_data = json.load(f)

    # Get API key from environment
    api_key = os.getenv("IMAGINATOR_AUTH_TOKEN")
    if not api_key:
        raise ValueError("IMAGINATOR_AUTH_TOKEN not found in .env file")

    # Define API endpoint and headers
    api_url = "http://127.0.0.1:8000/analyze"
    headers = {"X-API-Key": api_key}

    # Prepare request data
    request_data = {
        "resume_text": golden_data["input"]["resume_text"],
        "job_ad": golden_data["input"]["job_ad"]
    }

    # Run the model with the input from the golden data
    print("🔬 Calling the Imaginator API to get real output...")
    response = requests.post(api_url, headers=headers, json=request_data)

    if response.status_code == 200:
        generated_output = response.json()
        print("✅ Successfully received output from the API.")
        
        # The 'suggested_experiences' in the response is a list, not a dictionary.
        # We need to adapt the 'evaluate_quality' function to handle this.
        # For now, let's just print the raw output to see its structure.
        # print(json.dumps(generated_output, indent=4))

        quality_scores = evaluate_quality(generated_output, golden_data['ideal_output'])
        print("\n📊 Quality Evaluation Scores:")
        print(json.dumps(quality_scores, indent=4))
    else:
        print(f"❌ API call failed with status code {response.status_code}")
        print(response.text)
