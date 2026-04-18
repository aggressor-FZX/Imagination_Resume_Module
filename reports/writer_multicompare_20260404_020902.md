# Writer (Drafter) comparison — multi-resume simulation

Generated: 2026-04-04T02:09:02.188255+00:00
Source file: `/mnt/c/Users/jeffd/latexRoot/My_resume_tests/combined_resumes.txt`
Parsed experience blocks (pre-dedupe): 19
Drafter input roles (deduped, capped): 10
Drafter `max_tokens`: 32768

## How to read this

- **Style / formatting**: bullet length, parallelism, weak verbs, stray markdown.
- **Rule following**: `distinctive_job_ad_leaks` (SkyHarbor / NebulaForge / QuantMesh / p99 phrasing),
  `company_field_mismatches_vs_allowlist` (rewritten `company` not matching parsed employers).
- **Quantification**: fraction of bullets with numbers (heuristic; does not prove factual accuracy).

## Simulated upstream services (review artifacts)

Hermes-style aggregate entities and FastSVM-style predictions are **synthetic**
reconstructions for documentation; the live Drafter prompt matches production.

```json
{
  "hermes_style": {
    "note": "Simulated structured extraction from document text",
    "aggregate_entities": {
      "organizations": [
        "Connect AI Club, Everett Community College",
        "Washington State Data Exchange for Public SafetyInternship",
        "National Oceanic and Atmospheric Administration (NOAA)",
        "Naval Research Laboratory"
      ],
      "roles": [
        "Operations Analyst",
        "President",
        "Operations Manager",
        "software integration.",
        "Research Physicist",
        "program directors on technical strategy.",
        "Commissioned Officer (National Oceanic & Atmospheric Administration) Newport",
        "Research Physicist (Naval Research Laboratory) Washington",
        "Faculty Assistant Researcher (University of Maryland College Park) College Park",
        "Aerospace Ground Equipment Mechanic (United States Air Force) World Wide 05/2002 - 01/2010"
      ],
      "skills_flat": [
        "AWS",
        "Azure",
        "C++",
        "Cross-functional Teams",
        "Data Analysis",
        "Deep Learning",
        "Docker",
        "Go",
        "Hadoop",
        "Java",
        "JavaScript",
        "Kubernetes",
        "Leadership",
        "Machine Learning",
        "MongoDB",
        "Project Management",
        "PyTorch",
        "Python",
        "Ruby",
        "Rust",
        "SQL",
        "Scikit-learn",
        "TensorFlow"
      ]
    }
  },
  "fastsvm_style": {
    "title": "Analytics / Data Engineering",
    "canonical_title": "Senior Analytics Engineer",
    "overall_confidence": 0.86,
    "skills": [
      {
        "skill": "aws",
        "confidence": 0.75
      },
      {
        "skill": "azure",
        "confidence": 0.77
      },
      {
        "skill": "c++",
        "confidence": 0.79
      },
      {
        "skill": "cross-functional_teams",
        "confidence": 0.81
      },
      {
        "skill": "data_analysis",
        "confidence": 0.83
      },
      {
        "skill": "deep_learning",
        "confidence": 0.75
      },
      {
        "skill": "docker",
        "confidence": 0.77
      },
      {
        "skill": "go",
        "confidence": 0.79
      },
      {
        "skill": "hadoop",
        "confidence": 0.81
      },
      {
        "skill": "java",
        "confidence": 0.83
      },
      {
        "skill": "javascript",
        "confidence": 0.75
      },
      {
        "skill": "kubernetes",
        "confidence": 0.77
      },
      {
        "skill": "leadership",
        "confidence": 0.79
      },
      {
        "skill": "machine_learning",
        "confidence": 0.81
      },
      {
        "skill": "mongodb",
        "confidence": 0.83
      },
      {
        "skill": "project_management",
        "confidence": 0.75
      },
      {
        "skill": "pytorch",
        "confidence": 0.77
      },
      {
        "skill": "python",
        "confidence": 0.79
      },
      {
        "skill": "ruby",
        "confidence": 0.81
      },
      {
        "skill": "rust",
        "confidence": 0.83
      },
      {
        "skill": "sql",
        "confidence": 0.75
      },
      {
        "skill": "scikit-learn",
        "confidence": 0.77
      },
      {
        "skill": "tensorflow",
        "confidence": 0.79
      }
    ]
  }
}
```

## Target job ad (fictitious company — leak test)

SkyHarbor Analytics — Senior Analytics Engineer (Remote, US)

We ingest 4B+ events/day into NebulaForge, our real-time lakehouse. You will own
pipelines (Python, SQL, Spark), QuantMesh metric stores, and sub-40ms p99 dashboards
for executive KPIs. Partner with ML on feature stores; enforce data contracts;
mentor two junior engineers. Requires AWS or GCP, Airflow or Dagster, and a track
record of measurable reliability wins — not slide decks.

Nice-to-have: experience with government or regulated telemetry, on-call rotation,
and cost guardrails for big batch + streaming stacks.

## Researcher output (shared across models)

```json
{
  "implied_metrics": [
    "Ingest 4B+ events/day (Context: Real-time lakehouse data ingestion)",
    "Sub-40ms p99 dashboards (Context: Executive KPI dashboard performance)",
    "Measurable reliability wins (Context: Track record of improving system stability and performance)"
  ],
  "domain_vocab": [
    "Analytics Engineer",
    "Lakehouse",
    "Pipelines",
    "Python",
    "SQL",
    "Spark",
    "QuantMesh",
    "Metric Stores",
    "p99 dashboards",
    "Executive KPIs",
    "ML",
    "Feature Stores",
    "Data Contracts",
    "AWS",
    "GCP",
    "Airflow",
    "Dagster",
    "Reliability",
    "Government telemetry",
    "Regulated telemetry",
    "On-call rotation",
    "Cost guardrails",
    "Batch processing",
    "Streaming stacks"
  ],
  "implied_skills": [
    "Real-time Data Ingestion & Processing (Implied by 'Architect and deploy end-to-end ETL pipelines in SQL and Python, integrating diverse data sources to power real-time' and 'analyzing sensor and sonar data to proactively mitigate risks')",
    "Data Lakehouse Management (Implied by experience with 'diverse data sources' and 'real-time' processing, suggesting familiarity with large-scale data architectures)",
    "Metric Store Design & Implementation (Implied by 'define key performance metrics, automated data pipelines' and 'data-driven' solutions)",
    "Low-latency Dashboarding (Implied by 'real-time' data processing and 'decision-support tools' for operational insights)",
    "Feature Store Collaboration (Implied by 'data-driven' solutions, 'automated data pipelines', and 'LLMs, cloud platforms, and vector databases' in projects, which often feed ML models)",
    "Data Contract Enforcement (Implied by 'rigorous quality standards' in projects and 'translating business requirements into data-driven' solutions, which requires data governance)",
    "Mentorship & Leadership (Explicitly stated 'Coach peers' and 'Led cross-discipline teams')",
    "Reliability Engineering / SRE (Implied by 'proactively mitigate risks', 'reducing manual reporting overhead', and 'rigorous quality standards')",
    "Cloud Platform Experience (Explicitly mentioned 'cloud platforms' in projects, aligning with AWS/GCP requirement)",
    "Workflow Orchestration (Implied by 'Architect and deploy end-to-end ETL pipelines' and 'automated data pipelines', suggesting experience with tools like Airflow/Dagster)",
    "Government/Regulated Telemetry Experience (Explicitly stated 'Partner with law-enforcement and EMS stakeholders' and 'National Oceanic and Atmospheric Administration (NOAA)' experience)",
    "Cost Optimization (Implied by 'reducing manual reporting overhead by 40%' and 'proactively mitigate risks', suggesting an awareness of operational efficiency and resource management)"
  ],
  "work_archetypes": [
    "Scaling/Optimization",
    "Greenfield Build"
  ],
  "has_projects_section": true,
  "is_student_or_career_changer": true,
  "insider_tips": "Frame past operational analysis and ETL pipeline work as direct experience in building and optimizing data infrastructure. Emphasize the 'real-time' aspects of previous projects and how they align with the low-latency requirements of the target role."
}
```

## Model: `xiaomi/mimo-v2-flash`

| Metric | Value |
| --- | --- |
| bullet_count | 21 |
| quantification_score | 0.381 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 4 |
| avg_bullet_len | 124.0 |

Wall time: 7.98s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Assisted in mapping critical workflows for law enforcement and EMS stakeholders, translating business requirements into data-driven solutions.
- Contributed to architecting and deploying end-to-end ETL pipelines in SQL and Python, integrating diverse data sources to power real-time analytics.

**President** — *Connect AI Club, Everett Community College*
- Built student consulting projects using LLMs, cloud platforms, and vector databases to deliver proof-of-concepts solving client problems.
- Collaborated on client-style workshops and code reviews, ensuring rigorous quality standards and fostering collaborative innovation.
- Helped peers with prompt engineering, iterative model tuning, and data storytelling to communicate AI outcomes effectively.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Supported directed enterprise operations across 13 research vessels, analyzing sensor and sonar data to proactively mitigate risks.
- Collaborated on leading cross-discipline teams to design Python-based decision-support tools, reducing manual reporting overhead by 40%.
- Assisted in launching a novel Government Official Travel database and user interface, achieving 85% administrator adoption within a short timeframe.
- Contributed to consulting with operations leads to define key performance metrics and automate data pipelines for weather and oceanographic data.

**Research Physicist** — *Naval Research Laboratory*
- Collaborated on multi-million-dollar R&D projects, guiding model optimization for radiation detection and advising on technical strategy.

**Commissioned Officer (National Oceanic & Atmospheric Administration) Newport** — *Unknown*
- Supported the acquisition of hydrographic data using remote sensing and automated systems.
- Applied coding skills to automate DVD library inventory tracking and implement a GUI-based Sea Service letter generator.
- Served as Ancillary Data Manager; configured the Science Computing System and connected sensors to SCS servers.

**Research Physicist (Naval Research Laboratory) Washington** — *Unknown*
- Utilized high-performance computing to conduct critical studies on radiation detection for border security applications.
- Supported software development to predict active interrogation procedures and passive detection of WMD/radiological devices.
- Designed a graphic user interface for fast implementation of scenario-based radiation simulation software on Linux.

**Faculty Assistant Researcher (University of Maryland College Park) College Park** — *Unknown*
- Designed software for data analysis and publication-quality plots in particle physics experiments.
- Co-authored publications on experiments at the Large Hadron Collider, contributing to scientific knowledge.
- Developed simulations for radiation studies and efficiency quantifications with programming skills in C++ and Python.

**Aerospace Ground Equipment Mechanic (United States Air Force) World Wide** — *Unknown*
- Administered equipment databases and supervised maintenance operations to ensure aircraft data availability.
- Maintained aircraft maintenance records for F-16 fighter jets and presented readiness statistics to base leadership.

---

## Model: `qwen/qwen-3.6-plus-preview`

| Metric | Value |
| --- | --- |
| bullet_count | 0 |
| quantification_score | 0.0 |
| has_placeholders | False |
| fallback | True |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 0 |
| avg_bullet_len | 0.0 |

Wall time: 0.0s

### Rewritten experiences (markdown preview)

---

## Model: `qwen/qwen3.6-plus:free`

| Metric | Value |
| --- | --- |
| bullet_count | 23 |
| quantification_score | 0.391 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 6 |
| avg_bullet_len | 131.5 |

Wall time: 166.64s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Built end-to-end ETL pipelines using SQL and Python to integrate diverse public safety data for real-time analytics.
- Supported cross-functional stakeholders in mapping operational workflows and translating business requirements into structured data models.

**President** — *Connect AI Club, Everett Community College*
- Built proof-of-concept AI solutions utilizing LLMs, cloud platforms, and vector databases to solve complex client data challenges.
- Supported peer development through structured code reviews and coaching on prompt engineering and iterative model tuning.
- Collaborated on data storytelling initiatives to translate complex AI model outputs into clear, actionable business insights.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Built Python-based decision-support tools that automated manual reporting workflows, reducing operational overhead by 40 percent.
- Supported cross-discipline teams in defining key performance metrics and automating weather data pipelines for tracking.
- Collaborated on a centralized travel database launch, achieving 85 percent administrator adoption within target timelines.
- Assisted in streamlining vessel sensor data ingestion by developing Python GUIs and automated reporting scripts.

**software integration.** — *Unknown*

**Research Physicist** — *Naval Research Laboratory*
- Collaborated on multi-million-dollar R&D initiatives to optimize radiation detection models and improve overall analytical accuracy across teams.

**program directors on technical strategy.** — *Unknown*
- Assisted in translating complex simulation data into executive briefings, securing $500K in incremental project funding.

**Commissioned Officer (National Oceanic & Atmospheric Administration) Newport** — *Unknown*
- Built automated tracking systems and GUI-based reporting tools to streamline administrative data workflows and reduce manual entry.
- Supported hydrographic data acquisition by configuring Science Computing Systems and integrating remote sensor networks for research.
- Collaborated on data management protocols to ensure reliable sensor-to-server connectivity and maintain high-quality oceanographic datasets.

**Research Physicist (Naval Research Laboratory) Washington** — *Unknown*
- Contributed to high-performance computing studies on radiation detection, optimizing simulation parameters for enhanced border security analytics.
- Built Linux-based graphical interfaces to accelerate scenario-based radiation simulation software deployment and improve user accessibility.
- Supported predictive software development for radiological detection, improving analytical model accuracy and reducing processing latency.

**Faculty Assistant Researcher (University of Maryland College Park) College Park** — *Unknown*
- Built C++ and Python simulation models to quantify experimental efficiency and analyze large-scale particle physics datasets.
- Contributed to custom data analysis software development to generate publication-quality visualizations for Large Hadron Collider datasets.
- Collaborated on peer-reviewed research publications, translating complex experimental data into actionable scientific insights for stakeholders.

**Aerospace Ground Equipment Mechanic (United States Air Force) World Wide 05/2002 - 01/2010** — *Unknown*
- Supported fleet readiness tracking by maintaining equipment databases and ensuring complete data availability for daily operations.
- Assisted in leadership decision-making by compiling and presenting precise aircraft readiness statistics and maintenance performance metrics.
- Collaborated with technical teams to interpret complex schematics and standardize equipment maintenance documentation for operational use.

---

## Model: `minimax/minimax-m2.1`

| Metric | Value |
| --- | --- |
| bullet_count | 6 |
| quantification_score | 0.3 |
| has_placeholders | False |
| fallback | True |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 0 |
| avg_bullet_len | 55.5 |

Wall time: 339.71s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Applied technical skills to achieve measurable results
- Contributed to team success through collaborative efforts

**President** — *Connect AI Club, Everett Community College*
- Applied technical skills to achieve measurable results
- Contributed to team success through collaborative efforts

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Applied technical skills to achieve measurable results
- Contributed to team success through collaborative efforts

---

## Run notes (automated)

- **`minimax/minimax-m2.1`**: Drafter fell back (often empty JSON when the provider returns reasoning-only or truncated `message.content` at large prompt sizes). Try `--drafter-max-tokens 32000` or a different MiniMax provider route.
- Compare **Xiaomi** vs **Qwen** (and others) by reading the bullet sections above; rubric scores are heuristics only.
