# Writer (Drafter) comparison — multi-resume simulation

Generated: 2026-04-04T03:28:56.514568+00:00
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
        "National Oceanic and Atmospheric Administration (NOAA)",
        "Naval Research Laboratory",
        "Washington State Data Exchange for Public SafetyInternship",
        "Connect AI Club, Everett Community College"
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
    "Must be able to handle 4B+ events/day (Context: Ingesting data into a real-time lakehouse)",
    "Must be able to deliver sub-40ms p99 dashboards (Context: Executive KPIs)",
    "Must be able to reduce manual reporting overhead by 40% (Context: User's past achievement, demonstrating efficiency gains)",
    "Must be able to achieve 85% administrator adoption (Context: User's past achievement, demonstrating successful deployment and user engagement)"
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
    "Reliability Wins",
    "Government Telemetry",
    "Regulated Telemetry",
    "On-call Rotation",
    "Cost Guardrails",
    "Batch Processing",
    "Streaming Stacks"
  ],
  "implied_skills": [
    "Real-time Data Ingestion & Processing (Implied by 'Architect and deploy end-to-end ETL pipelines in SQL and Python, integrating diverse data sources to power real-time' and 'analyzing sensor and sonar data to proactively mitigate risks')",
    "Data Lakehouse Management (Implied by experience with 'diverse data sources' and 'ETL pipelines' which are foundational for lakehouse architectures)",
    "Metric Store Design & Implementation (Implied by 'define key performance metrics' and 'automated data pipelines' for weather and oceanographic data, which requires structured metric storage)",
    "Low-latency Dashboarding (Implied by 'real-time' data processing and 'decision-support tools' which often require quick data visualization for operational insights)",
    "Feature Store Collaboration (Implied by 'data-driven' solutions, 'automated data pipelines', and 'LLMs, cloud platforms, and vector databases' in projects, indicating familiarity with data preparation for ML)",
    "Data Contract Enforcement (Implied by 'rigorous quality standards' in projects and 'translating business requirements into data-driven' solutions, which necessitates data governance)",
    "Mentorship & Leadership (Explicitly stated 'mentor two junior engineers' in job, and user has 'Coach peers' and 'Led cross-discipline teams' experience)",
    "Cloud Platform Experience (Implied by 'cloud platforms' in projects and general data engineering work, though specific AWS/GCP not named)",
    "Workflow Orchestration (Implied by 'Architect and deploy end-to-end ETL pipelines' and 'automated data pipelines', suggesting experience with tools like Airflow/Dagster even if not named)",
    "Measurable Reliability Wins (Implied by 'reducing manual reporting overhead by 40%' and 'achieving 85% administrator adoption', demonstrating impact and reliability improvements)",
    "Government/Regulated Telemetry Experience (Explicitly stated 'Partner with law-enforcement and EMS stakeholders' and 'National Oceanic and Atmospheric Administration (NOAA)' experience)",
    "Cost Optimization for Big Data (Implied by 'reducing manual reporting overhead' and 'decision-support tools' which often have cost-saving as a goal, though not explicitly 'cost guardrails')"
  ],
  "work_archetypes": [
    "Scaling/Optimization",
    "Greenfield Build"
  ],
  "has_projects_section": true,
  "is_student_or_career_changer": true,
  "insider_tips": "Frame the NOAA and Washington State Data Exchange experience as direct parallels to regulated telemetry and high-stakes data environments. Emphasize the 'real-time' aspects of past ETL and sensor data analysis to align with the job's low-latency requirements."
}
```

## Model: `anthropic/claude-3.5-haiku`

| Metric | Value |
| --- | --- |
| bullet_count | 9 |
| quantification_score | 0.333 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 0 |
| avg_bullet_len | 151.2 |

Wall time: 15.44s

### Rewritten experiences (markdown preview)

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Collaborated with cross-discipline teams to architect Python-based decision-support tools, reducing manual reporting overhead by 40% and enhancing operational efficiency
- Directed enterprise operations across 13 research vessels, analyzing sensor and sonar data to proactively develop real-time risk mitigation strategies
- Consulted with operations leads to define and automate key performance metrics for weather and oceanographic data pipelines, enabling data-driven decision making

**Operations Analyst** — *Washington State Data Exchange for Public Safety*
- Collaborated with law enforcement and EMS stakeholders to architect end-to-end ETL pipelines in SQL and Python, integrating diverse data sources for real-time operational insights
- Translated complex business requirements into data-driven workflow solutions, bridging technical and operational domains

**Research Physicist** — *Naval Research Laboratory*
- Utilized high-performance computing to develop radiation detection models, supporting critical national security data analysis projects
- Designed graphic user interfaces for scenario-based radiation simulation software, demonstrating technical versatility in data visualization and computational research

**President** — *Connect AI Club, Everett Community College*
- Spearheaded student consulting projects leveraging LLMs, cloud platforms, and vector databases to deliver proof-of-concept solutions
- Facilitated technical workshops and code reviews, ensuring rigorous quality standards and fostering collaborative innovation in data-driven projects

---

## Model: `xiaomi/mimo-v2-flash`

| Metric | Value |
| --- | --- |
| bullet_count | 22 |
| quantification_score | 0.364 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 4 |
| avg_bullet_len | 121.3 |

Wall time: 8.08s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Collaborated on mapping critical workflows with law enforcement and EMS stakeholders, translating business requirements into data-driven solutions.
- Built end-to-end ETL pipelines in SQL and Python, integrating diverse data sources to power real-time analytics for public safety operations.

**President** — *Connect AI Club, Everett Community College*
- Built student consulting projects using LLMs, cloud platforms, and vector databases to deliver proof-of-concepts solving client challenges.
- Coached peers on prompt engineering and iterative model tuning, effectively communicating AI outcomes to non-technical audiences.
- Facilitated client-style workshops and code reviews, ensuring rigorous quality standards and fostering collaborative innovation.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Supported enterprise operations across 13 research vessels, analyzing sensor and sonar data to proactively mitigate risks.
- Led cross-discipline teams to design Python-based decision-support tools, reducing manual reporting overhead by 40%.
- Launched a Government Official Travel database and user interface, achieving 85% administrator adoption within the first year.
- Consulted with operations leads to define key performance metrics and automated data pipelines for weather and oceanographic data.
- Developed Python GUIs and reporting scripts to streamline vessel data ingestion and fostered cross-team collaboration.

**Research Physicist** — *Naval Research Laboratory*
- Collaborated on multi-million-dollar R&D projects, guiding model optimization for radiation detection and advising on technical strategy.

**Commissioned Officer (National Oceanic & Atmospheric Administration) Newport** — *Unknown*
- Supported the acquisition of hydrographic data using remote sensing and automated systems.
- Applied coding skills to automate DVD library inventory tracking and implemented a GUI-based Sea Service letter generator.
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
- Maintained aircraft maintenance records supporting F-16 fighter jets and presented readiness statistics to base leadership.

---

## Model: `qwen/qwen3.6-plus:free`

| Metric | Value |
| --- | --- |
| bullet_count | 22 |
| quantification_score | 0.5 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 4 |
| avg_bullet_len | 125.0 |

Wall time: 146.18s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Collaborated on stakeholder workflow mapping to translate complex business requirements into scalable data pipeline architectures.
- Built end-to-end ETL pipelines in SQL and Python, integrating diverse sources to enable real-time analytics.

**President** — *Connect AI Club, Everett Community College*
- Built student consulting projects utilizing LLMs, cloud platforms, and vector databases to deliver functional data PoCs.
- Supported peer development by coaching on prompt engineering, model tuning, and data storytelling for technical presentations.
- Helped establish code review standards and client workshops, ensuring rigorous data quality and collaborative delivery.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Assisted in analyzing sensor and sonar telemetry across 13 research vessels to proactively identify operational risks.
- Built Python-based decision support tools that automated reporting workflows, successfully reducing manual overhead by 40%.
- Collaborated on defining KPI frameworks and automating weather data pipelines, achieving 85% administrator adoption rates.
- Supported cross-team data ingestion by developing Python GUIs and reporting scripts for streamlined vessel telemetry.

**Research Physicist** — *Naval Research Laboratory*
- Contributed to multi-million dollar R&D initiatives by guiding model optimization and advising program directors on strategy.

**Commissioned Officer (National Oceanic & Atmospheric Administration) Newport** — *Unknown*
- Assisted in acquiring hydrographic data using remote sensing and automated telemetry systems for scientific analysis.
- Built automated Python scripts to track inventory and generate standardized reports, eliminating manual administrative tasks.
- Supported sensor-to-server integrations by configuring the Science Computing System and managing critical ancillary data pipelines.

**Research Physicist (Naval Research Laboratory) Washington** — *Unknown*
- Collaborated on high-performance computing clusters to execute large-scale radiation detection simulations for critical security applications.
- Supported predictive modeling development for active interrogation and passive radiological detection workflows using Python and C++.
- Built graphical user interfaces on Linux to accelerate scenario-based radiation simulation deployment and data analysis.

**Faculty Assistant Researcher (University of Maryland College Park) College Park** — *Unknown*
- Assisted in developing C++ and Python software for particle physics data analysis and high-fidelity visualization.
- Contributed to Large Hadron Collider research by co-authoring technical publications on experimental data analysis and findings.
- Helped engineer simulation workflows to quantify radiation efficiency, optimizing computational resource allocation for research teams.

**Aerospace Ground Equipment Mechanic (United States Air Force) World Wide 05/2002 - 01/2010** — *Unknown*
- Supported aircraft readiness by administering maintenance databases and ensuring 100% data availability for operational tracking.
- Assisted in compiling F-16 maintenance records and generating readiness statistics for executive leadership operational reviews.
- Collaborated on technical documentation and schematic interpretation to streamline equipment maintenance workflows and operational reporting.

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

Wall time: 474.14s

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
