# Imaginator Flow Documentation

## Overview
The Imaginator Flow is ### Creativity and Limitations
- **Creativity Sources**: LLM generates novel phrasing and suggestions; focuses on target job ad for relevance.
- **Enhancements from Pipeline**: Accurate skill extraction from FastSVM/Hermes improves input quality; parsed text from Resume_Document_Loader ensures clean data; target job ad provides focus.
- **Limitations**: Deterministic early stages; LLM outputs can vary; no structured validation.thon script (`imaginator_flow.py`) that processes a resume to extract skills, suggest job roles, and generate a gap analysis with resume improvement recommendations. It leverages OpenAI's GPT-3.5-turbo for creative, natural language-based suggestions. The script is designed to run after a pipeline of preprocessing modules to enhance its inputs and focus its creativity on a target job advertisement.

## Pipeline Integration
The Imaginator runs as the final step in a resume processing pipeline consisting of three preceding modules:

1. **Hermes** (https://github.com/aggressor-FZX/Hermes/tree/main)
   - Purpose: Resume processing pipeline using SVM (Support Vector Machine) for skill extraction and analysis. Achieves high accuracy through ML models and fuzzy matching.
   - Inputs: Structured resume data (text, sections).
   - Outputs: JSON with extracted title, skills (with confidence scores), domain classification, insights, and processing metadata.
   - How it works: Parses resumes, extracts skills/titles using SVM models, generates domain-aware insights and recommendations.

2. **Resume_Document_Loader** (https://github.com/aggressor-FZX/Resume_Document_Loader/tree/main)
   - Purpose: Document processing service that extracts text from various formats and prepares structured data for downstream analysis.
   - Inputs: Document files (PDF, DOCX, images, text).
   - Outputs: JSON with extracted text, parsed sections (summary, experience, education, skills), and Hermes-compatible payload.
   - How it works: Multi-layer text extraction (PyPDF2 → pdfplumber → OCR), intelligent section detection, and formatting for pipeline consumption.

3. **FastSVM_Skill_Title_Extraction** (https://github.com/aggressor-FZX/FastSVM_Skill_Title_Extraction/tree/main)
   - Purpose: High-accuracy skill and title extraction using SVM models, achieving 90%+ accuracy.
   - Inputs: Resume text.
   - Outputs: JSON with extracted title, canonical title, skills (with confidence), and processing metadata.
   - How it works: Hybrid ML + rule-based approach with TF-IDF, fuzzy matching, and confidence scoring.

By integrating outputs from these modules, Imaginator avoids redundant parsing and leverages advanced extraction for better creativity in suggestions.

## How It Works

### Inputs
- `--resume`: Path to a resume text file (fallback if parsed text not provided).
- `--parsed_resume_text`: Pre-parsed resume text from Resume_Document_Loader.
- `--extracted_skills_json`: JSON file from FastSVM_Skill_Title_Extraction containing extracted skills and titles.
- `--target_job_ad` (required): Target job ad text to focus the analysis (user-provided or from external source).

### Processing Steps

1. **Resume Text Acquisition**:
   - If `--parsed_resume_text` is provided, use it directly.
   - Otherwise, read from `--resume` file.

2. **Skill Extraction**:
   - If `--extracted_skills_json` is provided, load skills and experiences from the JSON.
   - Otherwise, parse experiences using regex splitting and extrapolate skills via keyword matching in `_SKILL_KEYWORDS`.

3. **Role Suggestion**:
   - Match extracted skills against predefined role requirements in `_ROLE_MAP`.
   - Score roles by the fraction of required skills present.

4. **Gap Analysis Generation**:
   - Construct a prompt including resume text, skills, role suggestions, and target job ad.
   - Call OpenAI GPT-3.5-turbo to generate natural language recommendations for resume improvements focused on the target job.

### Outputs
A JSON object containing:
- `experiences`: List of parsed experience blocks with skills and snippets.
- `aggregate_skills`: Sorted list of all extracted skills.
- `role_suggestions`: List of suggested roles with scores and matched skills.
- `target_job_ad`: The input job ad text.
- `gap_analysis`: LLM-generated improvement suggestions.

### Key Components

- **Keyword Mapping**: `_SKILL_KEYWORDS` maps skill categories to keywords for basic extraction.
- **Role Mapping**: `_ROLE_MAP` defines skill requirements for roles.
- **LLM Integration**: Uses OpenAI API for creative gap analysis, with temperature=0.7 for variety.

### Creativity and Limitations
- **Creativity Sources**: LLM generates novel phrasing and suggestions; focuses on target job ad for relevance.
- **Enhancements from Pipeline**: Accurate skill extraction from FastSVM improves input quality; parsed text from Resume_Document_Loader ensures clean data; target job ad from Hermes provides focus.
- **Limitations**: Deterministic early stages; LLM outputs can vary; no structured validation.

## Usage
```bash
python imaginator_flow.py --parsed_resume_text "Resume content..." --extracted_skills_json skills.json --target_job_ad "Job ad text..."
```

## Dependencies
- `openai`: For LLM calls.
- `python-dotenv`: For API key loading.
- Requires `OPENAI_API_KEY` in `.env`.

## Future Improvements
- Structured LLM outputs (JSON).
- Embeddings-based skill matching.
- Metrics for creativity evaluation.