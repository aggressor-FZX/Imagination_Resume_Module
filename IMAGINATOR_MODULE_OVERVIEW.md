# Imaginator Module: Architecture, Inputs, and Integration

## Overview
The **Imaginator Module** is a core component of the Cogito Metric platform, designed to enhance and co-write resumes by leveraging AI-driven enrichment and transformation. It operates as a web service, receiving structured data and documents, processing them through a pipeline of enrichment modules, and returning improved, context-aware resume content.

## What the Imaginator Receives
### Inputs
- **Resume Data**: Structured JSON representing a user's resume (work history, education, skills, etc.).
- **Raw Documents**: PDF, DOCX, or plain text files containing resume or career information.
- **User Metadata**: Optional information such as user ID, job target, or preferences.
- **API Requests**: HTTP POST requests (typically JSON payloads) from other services or frontends.

### Example Input: Imaginator Module

```json
{
  "resume_text": "John Doe\nSenior Software Engineer\nEXPERIENCE\nSoftware Engineer at Tech Corp (2020-2023)\n- Developed Python APIs using Flask and FastAPI\n- Built microservices with Docker and Kubernetes\n- Implemented CI/CD pipelines with Jenkins\n- Led team of 5 developers\nSKILLS\nPython, JavaScript, React, Docker, Kubernetes, AWS",
  "job_ad_text": "We are looking for a Senior Full Stack Developer with experience in Python, JavaScript, React, Docker, Kubernetes, AWS, and cloud platforms. Team leadership is a plus.",
  "confidence_threshold": 0.7,
  "target_role": "Senior Full Stack Developer"
}
```

**Optional fields:**
- `extracted_skills_json`: JSON string or object with pre-extracted skills (from Hermes or other modules)
- `domain_insights_json`: JSON string or object with domain-specific insights

**Minimal Example:**
```json
{
  "resume_text": "Jane Smith\nData Analyst...",
  "job_ad_text": "Seeking Data Analyst with SQL, Python..."
}
```

### Sources of Input
- **Document Reader Service**: Extracts and parses raw resume files, sending structured data to the Imaginator.
- **Hermes Resume Extractor**: Performs advanced entity extraction and normalization on resume text, forwarding enriched data.
- **Job Search API**: Supplies job descriptions or requirements to tailor the resume output.
- **Frontend Applications**: Users can upload resumes or input data directly via web interfaces.

## Pipeline Modules and Their Roles
### 1. Document Reader Service
- **Role**: Converts uploaded files (PDF, DOCX, TXT) into structured text.
- **Output**: Cleaned, segmented resume data (sections, bullet points, etc.).
- **Provides**: Initial input for downstream enrichment.
- **Example Output**:
```json
{
  "raw_text": "John Doe\nSenior Software Engineer...",
  "parsed_sections": {
    "summary": "Professional summary...",
    "experience": [
      {
        "title": "Senior Developer",
        "company": "TechCorp",
        "duration": "2020-2023",
        "description": "Job description..."
      }
    ],
    "education": [...],
    "skills": ["Python", "JavaScript", "React"]
  }
}
```

### 2. Hermes Resume Extractor
- **Role**: Extracts entities (skills, job titles, education, dates) and normalizes them using SVM models and O*NET taxonomy.
- **Output**: Annotated, structured resume data with standardized fields.
- **Provides**: Enhanced, machine-readable input for the Imaginator.
- **Example Output**:
```json
{
  "title": "Software Engineer",
  "title_confidence": 0.7,
  "skills": [
    {
      "skill": "python",
      "category": "technical",
      "confidence": 0.9,
      "source": "extracted"
    },
    {
      "skill": "react",
      "category": "framework",
      "confidence": 0.8,
      "source": "skills"
    }
  ],
  "skill_count": 15,
  "overall_confidence": 0.82,
  "domain": "tech",
  "domain_confidence": 0.85,
  "insights": {
    "strengths": ["Strong Python foundation", "Full-stack development experience"],
    "gaps": ["Version control systems (Git)", "Testing methodologies"],
    "recommendations": ["Consider learning advanced cloud architectures", "Expand DevOps skills"],
    "market_alignment": 0.75
  }
}
```

### 3. FastSVM Skill & Title Extraction
- **Role**: Extracts skills and canonical job titles from resume text using SVM models.
- **Output**: Structured skills with confidence scores and standardized job titles.
- **Provides**: Skill extraction for both resumes and job ads.
- **Example Output**:
```json
{
  "title": "Senior Software Engineer",
  "canonical_title": "Software Engineer",
  "skills": [
    { "skill": "Python", "confidence": 0.95 },
    { "skill": "Django", "confidence": 0.92 }
  ],
  "overall_confidence": 0.91,
  "processing_metadata": {
    "input_type": "pdf",
    "text_length": 1550,
    "sections_count": 5,
    "skills_count": 8
  }
}
```

### 4. Job Search API (with ATS Scoring)
- **Role**: Searches for jobs and provides ATS (Applicant Tracking System) scoring.
- **Output**: Job listings with match scores and detailed matching analysis.
- **Provides**: Job context and ATS scoring for resume optimization.
- **Example ATS Scoring Request**:
```json
{
  "resume_text": "Senior Python developer with 5 years experience...",
  "job_ad": "Looking for Senior Python Developer with AWS experience...",
  "skills": ["Python", "AWS", "Docker"],
  "experiences": [
    {
      "title": "Senior Developer",
      "skills": ["Python", "FastAPI", "PostgreSQL"]
    }
  ]
}
```
- **Example ATS Scoring Response**:
```json
{
  "match_score": 0.87,
  "matched_requirements": ["python", "aws"],
  "unmet_requirements": ["kubernetes", "terraform"]
}
```
- **Job Search Matching Request**:
```json
{
  "skills": [
    {"skill": "Python", "confidence": 0.95},
    {"skill": "FastAPI", "confidence": 0.90}
  ],
  "job_preferences": {
    "target_roles": ["Python Developer"],
    "preferred_locations": ["Remote"],
    "experience_level": "senior",
    "salary_expectations": {"min": 120000}
  }
}
```

### 5. Imaginator Module
- **Role**: AI-powered resume co-writer and enhancer.
- **Input**: Receives structured resume data, optionally with job context.
- **Output**: Improved, tailored resume content (rewritten sections, added achievements, keyword optimization).

## What the Imaginator Is For
- **Purpose**: To automatically enhance, rewrite, and optimize resumes for specific job applications or general improvement.
- **How It Works**: Uses AI models to:
  - Rephrase and enrich resume sections.
  - Insert relevant keywords and achievements.
  - Tailor content to match job descriptions.
  - Ensure clarity, conciseness, and impact.
- **Intended Outcome**: Higher-quality, more competitive resumes that increase job seeker success.

## How Job Search Interacts with the Imaginator
**You are correct!** The flow is actually:

1. **User uploads resume** → Document Reader extracts text
2. **Text goes to Hermes** → Extracts skills and domain insights
3. **Text goes to FastSVM** → Extracts skills and canonical titles
4. **All enriched data goes to Imaginator** → Generates improved resume
5. **Improved resume goes to Job Search API** → Gets ATS scoring and job matches

**Correction to previous understanding:**
- **Job Search API does NOT supply data to Imaginator** - it receives the improved resume FROM Imaginator
- **Imaginator writes the new resume FIRST**, then provides it to Job Search for matching and scoring
- **The pipeline is sequential**: Document Reader → Hermes → FastSVM → Imaginator → Job Search

**Actual Flow:**
```
[User Upload] → [Document Reader] → [Hermes] → [FastSVM] → [Imaginator] → [Job Search API]
      ↓                ↓               ↓           ↓           ↓              ↓
   Raw file    →  Structured text → Skills/Insights → Skills/Titles → Enhanced Resume → ATS Scores + Job Matches
```

## Data Flow: What Imaginator Receives from Each Module

### From Document Reader Service:
```json
{
  "raw_text": "Full resume text...",
  "parsed_sections": {
    "summary": "Professional summary...",
    "experience": [...],
    "education": [...],
    "skills": ["Python", "JavaScript", "React"]
  }
}
```

### From Hermes Resume Extractor:
```json
{
  "skills": [
    {"skill": "python", "category": "technical", "confidence": 0.9},
    {"skill": "react", "category": "framework", "confidence": 0.8}
  ],
  "domain": "tech",
  "domain_confidence": 0.85,
  "insights": {
    "strengths": ["Strong Python foundation"],
    "gaps": ["Version control systems"],
    "recommendations": ["Learn cloud architectures"],
    "market_alignment": 0.75
  }
}
```

### From FastSVM Skill Extraction:
```json
{
  "skills": [
    {"skill": "Python", "confidence": 0.95},
    {"skill": "Django", "confidence": 0.92}
  ],
  "title": "Senior Software Engineer",
  "canonical_title": "Software Engineer"
}
```

### From Job Search API (ATS Scoring):
```json
{
  "match_score": 0.87,
  "matched_requirements": ["python", "aws"],
  "unmet_requirements": ["kubernetes", "terraform"],
  "job_details": {
    "title": "Senior Python Developer",
    "description": "Looking for Senior Python Developer with AWS experience...",
    "required_skills": ["Python", "AWS", "Kubernetes", "Terraform"]
  }
}
```

## Complete Imaginator Input Example (Integrated Pipeline)
```json
{
  "resume_text": "John Doe\nSenior Software Engineer...",
  "job_ad_text": "Looking for Senior Full Stack Developer...",
  "extracted_skills_json": {
    "skills": [
      {"skill": "Python", "confidence": 0.95, "category": "technical"},
      {"skill": "React", "confidence": 0.88, "category": "framework"}
    ],
    "title": "Senior Software Engineer",
    "domain": "tech"
  },
  "domain_insights_json": {
    "domain": "tech",
    "market_demand": "High",
    "skill_gap_priority": "Medium",
    "emerging_trends": ["AI/ML", "DevOps"],
    "insights": ["Cloud skills in high demand", "Full-stack developers preferred"]
  },
  "confidence_threshold": 0.7,
  "target_role": "Senior Full Stack Developer"
}
```

## How Imaginator Works: Skill Matching and Transfer Inference

### Core Skill Processing Pipeline

The Imaginator uses a sophisticated multi-stage approach to match existing skills and infer transferable skills:

#### 1. **Basic Skill Extraction**
- **Keyword Matching**: Scans resume text for skill keywords using predefined mappings
- **Pattern Recognition**: Identifies skills through context patterns (e.g., "5+ years experience with Python")
- **Fallback Mechanism**: When structured data is unavailable, falls back to text-based extraction

#### 2. **Structured Skill Processing**
When skills come from upstream services (Hermes, FastSVM), the Imaginator:
- **Confidence Filtering**: Applies confidence thresholds (default 0.7) to filter reliable skills
- **Categorization**: Groups skills into high/medium/low confidence buckets
- **Source Attribution**: Tracks skill provenance (document reader, extracted, inferred)

#### 3. **Skill Transfer Inference (Adjacency Analysis)**

**Core Mechanism:**
```python
# Uses skill_adjacency.json to find related skills
adjacency_mappings = {
    "python": {
        "scripting": 0.95,           # High confidence transfer
        "automation": 0.90,          # Strong correlation
        "data_processing": 0.85,     # Moderate correlation
        "backend_development": 0.88, # Strong correlation
        "api_development": 0.82,     # Moderate correlation
        "testing": 0.78,             # Lower correlation
        "debugging": 0.92            # High confidence transfer
    }
}
```

**Confidence Decay Formula:**
```
inferred_skill_confidence = base_skill_confidence × adjacency_weight
```

**Example:**
- User has "Python" with confidence 0.85
- Adjacency mapping shows "scripting" with weight 0.95
- Inferred "scripting" skill gets confidence: 0.85 × 0.95 = 0.81

#### 4. **Transferable Skill Detection**

The system identifies transferable skills through:

**A. Semantic Relationships:**
- **Backend Development** → **API Development**, **Database Design**
- **Cloud Computing** → **Infrastructure Management**, **Scalability**
- **Containerization** → **Microservices**, **Deployment**

**B. Domain-Specific Transfers:**
```json
{
    "data_analysis": {
        "statistical_analysis": 0.90,
        "data_visualization": 0.85,
        "reporting": 0.80,
        "business_intelligence": 0.75
    },
    "project_management": {
        "team_leadership": 0.92,
        "stakeholder_communication": 0.88,
        "resource_planning": 0.85,
        "risk_management": 0.80
    }
}
```

**C. Technology Stack Inference:**
- **React** → **Component Architecture**, **State Management**, **UI Development**
- **AWS** → **Cloud Security**, **Serverless**, **Infrastructure**
- **Docker** → **Containerization**, **Microservices**, **Orchestration**

#### 5. **Action Verb Competency Extraction**

The system analyzes experience descriptions for action verbs to infer competencies:
- **"Led"** → Leadership, Team Management
- **"Implemented"** → Technical Implementation, Problem Solving
- **"Optimized"** → Performance Tuning, Efficiency
- **"Collaborated"** → Teamwork, Communication

#### 6. **Gap Analysis with Transfer Skills**

When comparing against job requirements:
1. **Exact Matches**: Direct skill matches from resume
2. **Inferred Matches**: Skills derived through adjacency analysis
3. **Transferable Skills**: Competencies from action verbs and experience
4. **Missing Skills**: True gaps requiring development

**Example Output:**
```json
{
  "processed_skills": {
    "high_confidence": ["Python", "AWS", "Docker"],
    "medium_confidence": ["Kubernetes", "React"],
    "low_confidence": ["Machine Learning"],
    "inferred_skills": ["Scripting", "Automation", "Containerization", "Cloud Architecture"]
  },
  "gap_analysis": {
    "critical_gaps": ["Kubernetes", "Machine Learning"],
    "transferable_skills": ["Scripting (from Python)", "Cloud Architecture (from AWS)"],
    "recommendations": ["Leverage scripting experience for automation roles", "Highlight cloud architecture knowledge from AWS projects"]
  }
}
```

### Key Benefits of Transfer Inference

1. **Comprehensive Skill Profile**: Captures implicit skills not explicitly stated
2. **Career Transition Support**: Identifies transferable skills for role changes
3. **Gap Mitigation**: Reduces apparent skill gaps through inference
4. **Personalized Recommendations**: Suggests roles matching both explicit and implicit skills

This approach ensures the Imaginator provides a complete, nuanced understanding of a candidate's capabilities, going beyond simple keyword matching to understand skill relationships and transfer potential.

## Summary Table
| Module                  | Input                        | Output                        | Provides to Imaginator? |
|-------------------------|------------------------------|-------------------------------|------------------------|
| Document Reader Service | Raw resume files (PDF, DOCX, TXT) | Structured text with parsed sections | Yes (raw_text, parsed_sections) |
| Hermes Resume Extractor | Structured text              | Annotated skills, domain insights, market alignment | Yes (skills, domain_insights_json) |
| FastSVM Skill Extraction| Resume/job ad text           | Skills with confidence scores, canonical titles | Yes (extracted_skills_json) |
| Job Search API          | Enhanced resume from Imaginator | Job listings with ATS scores, matched requirements | **No** - Receives FROM Imaginator |
| Imaginator Module       | All enriched data + job context | Enhanced, tailored resume with gap analysis and improvement suggestions | N/A (final output) |

## Complete Pipeline Flow
```
[User Upload] → [Document Reader] → [Hermes Extractor] → [FastSVM] → [Imaginator] → [Job Search API]
                     ↓                      ↓                ↓              ↓              ↓
              [Raw file]           [Skills/Insights]  [Skills/Titles]  [Enhanced Resume]  [ATS Scores + Jobs]
```

## Intended Usage
- **Users**: Upload resumes or input data via frontend.
- **System**: Processes and enriches data through the pipeline:
  1. Document Reader extracts structured text
  2. Hermes provides domain insights and normalized skills
  3. FastSVM extracts skills from both resume and job ads
  4. Job Search provides ATS scoring and job context
  5. Imaginator synthesizes all inputs for optimized resume
- **Output**: High-quality, job-targeted resume with gap analysis and improvement suggestions.

## Key Integration Points
1. **Document Reader → Hermes/FastSVM**: Provides foundational structured text for all downstream processing
2. **Hermes → Imaginator**: Provides `domain_insights_json` with market trends and skill priorities
3. **FastSVM → Imaginator**: Provides `extracted_skills_json` with confidence-scored skills
4. **Imaginator → Job Search**: Provides enhanced resume for ATS scoring and job matching
5. **Sequential Flow**: Each module processes data and passes it to the next - no parallel feedback loops

---
For more technical details, see the README in the Imagination_Resume_Module repository.