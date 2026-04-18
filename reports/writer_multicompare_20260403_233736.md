# Writer (Drafter) comparison — multi-resume simulation

Generated: 2026-04-03T23:37:36.412391+00:00
Source file: `/mnt/c/Users/jeffd/latexRoot/My_resume_tests/combined_resumes.txt`
Parsed experience blocks (pre-dedupe): 19
Drafter input roles (deduped, capped): 10

## Simulated upstream services (review artifacts)

Hermes-style aggregate entities and FastSVM-style predictions are **synthetic**
reconstructions for documentation; the live Drafter prompt matches production.

```json
{
  "hermes_style": {
    "note": "Simulated structured extraction from document text",
    "aggregate_entities": {
      "organizations": [
        "MD",
        "Naval Research Laboratory",
        "DC",
        "OR",
        "Connect AI Club, Everett Community College",
        "National Oceanic and Atmospheric Administration (NOAA)",
        "Washington State Data Exchange for Public SafetyInternship"
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
    "Must be able to mentor two junior engineers (Context: Team leadership and development)"
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
    "On-call rotation",
    "Cost Guardrails",
    "Batch Processing",
    "Streaming Stacks"
  ],
  "implied_skills": [
    "Real-time Data Ingestion & Processing (Implied by 'Architect and deploy end-to-end ETL pipelines in SQL and Python, integrating diverse data sources to power real-time' and 'analyzing sensor and sonar data to proactively mitigate risks')",
    "Data Lakehouse Management (Implied by experience with 'diverse data sources' and 'real-time' processing, suggesting familiarity with large-scale data architectures)",
    "Metric Store Design & Implementation (Implied by 'define key performance metrics' and 'automated data pipelines' for NOAA, which requires structured storage for metrics)",
    "Low-Latency Dashboarding (Implied by 'real-time' ETL pipelines and the need to 'effectively communicate AI outcomes' and 'data storytelling', which often involves rapid visualization)",
    "Feature Store Collaboration (Implied by 'Spearhead student consulting projects, employing LLMs, cloud platforms, and vector databases to deliver PoCs' and 'iterative model tuning', which necessitates structured data for ML models)",
    "Data Contract Enforcement (Implied by 'rigorous quality standards' in code reviews and 'translate business requirements into data-driven' solutions, requiring clear data definitions and validation)",
    "Mentorship & Leadership (Explicitly stated as 'President' of Connect AI Club, 'Coach peers', 'Facilitate client-style workshops', and 'Led cross-discipline teams')",
    "Cloud Platform Experience (Explicitly mentioned 'cloud platforms' in Connect AI Club projects, aligning with AWS/GCP requirement)",
    "Workflow Orchestration (Implied by 'Architect and deploy end-to-end ETL pipelines' and 'automated data pipelines', suggesting experience with tools like Airflow/Dagster)",
    "Measurable Reliability Wins (Implied by 'reducing manual reporting overhead by 40%' and 'achieving 85% administrator adoption', demonstrating impact and stability)",
    "Government/Regulated Telemetry Experience (Explicitly stated 'Partner with law-enforcement and EMS stakeholders' and 'National Oceanic and Atmospheric Administration (NOAA)' experience)",
    "Cost Optimization for Data Stacks (Implied by 'reducing manual reporting overhead by 40%', indicating an awareness of efficiency and resource management)"
  ],
  "work_archetypes": [
    "Scaling/Optimization",
    "Greenfield Build"
  ],
  "has_projects_section": true,
  "is_student_or_career_changer": true,
  "insider_tips": "Frame the NOAA and Washington State Data Exchange experience as direct parallels to regulated telemetry and large-scale data operations. Emphasize the leadership and project work from the AI Club as demonstrating proactive problem-solving and mentorship capabilities relevant to the Senior role."
}
```

## Model: `xiaomi/mimo-v2-flash`

| Metric | Value |
| --- | --- |
| bullet_count | 22 |
| quantification_score | 0.409 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 3 |
| avg_bullet_len | 116.4 |

Wall time: 7.85s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Collaborated on mapping critical workflows with law enforcement and EMS stakeholders, translating business requirements into data-driven solutions.
- Built end-to-end ETL pipelines in SQL and Python, integrating diverse data sources to power real-time analytics for public safety operations.

**President** — *Connect AI Club, Everett Community College*
- Built student consulting projects using LLMs, cloud platforms, and vector databases to deliver proof-of-concepts solving real-world problems.
- Helped peers on prompt engineering and model tuning, coaching data storytelling to effectively communicate AI outcomes to clients.
- Supported client-style workshops and code reviews, ensuring rigorous quality standards and fostering collaborative innovation.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Supported enterprise operations across 13 research vessels, analyzing sensor and sonar data to proactively mitigate risks.
- Contributed to designing Python-based decision-support tools, reducing manual reporting overhead by 40%.
- Built a Government Official Travel database and user interface, achieving 85% administrator adoption within a short timeframe.
- Collaborated with operations leads to define key performance metrics and automate data pipelines for weather and oceanographic data.

**Research Physicist** — *Naval Research Laboratory*
- Collaborated on multi-million-dollar R&D projects, guiding model optimization for radiation detection and advising on technical strategy.

**software integration.** — *Unknown*

**program directors on technical strategy.** — *Unknown*
- Contributed to translating complex simulation results into executive briefings, securing $500K in incremental funding.

**Commissioned Officer (National Oceanic & Atmospheric Administration) Newport** — *OR*
- Supported hydrographic data acquisition using remote sensing and automated systems.
- Helped automate DVD library inventory tracking and implemented a GUI-based Sea Service letter generator.
- Served as Ancillary Data Manager; configured the Science Computing System and connected sensors to SCS servers.

**Research Physicist (Naval Research Laboratory) Washington** — *DC*
- Utilized high-performance computing to conduct critical studies on radiation detection for border security applications.
- Supported software development to predict active interrogation procedures and passive detection of WMD/radiological devices.
- Designed a graphic user interface for fast implementation of scenario-based radiation simulation software on Linux.

**Faculty Assistant Researcher (University of Maryland College Park) College Park** — *MD*
- Built software for data analysis and publication-quality plots in particle physics experiments.
- Contributed to co-authoring publications on experiments at the Large Hadron Collider.
- Developed simulations for radiation studies using C++ and Python.

**Aerospace Ground Equipment Mechanic (United States Air Force) World Wide 05/2002 - 01/2010** — *Unknown*
- Supported equipment database administration and supervised maintenance operations to ensure aircraft data availability.
- Maintained aircraft maintenance records for F-16 fighter jets and presented readiness statistics to base leadership.

---

## Model: `qwen/qwen3.6-plus:free`

| Metric | Value |
| --- | --- |
| bullet_count | 17 |
| quantification_score | 0.529 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 2 |
| avg_bullet_len | 137.5 |

Wall time: 138.47s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Built end-to-end ETL pipelines in SQL and Python, integrating diverse sources to enable real-time public safety analytics.
- Assisted in mapping critical stakeholder workflows, translating operational requirements into structured data models for cross-agency reporting.

**President** — *Connect AI Club, Everett Community College*
- Supported student consulting projects by deploying LLMs and vector databases on cloud platforms to deliver actionable PoCs.
- Helped peers master prompt engineering and iterative model tuning, translating complex AI outputs into clear data storytelling frameworks.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Contributed to automated weather and oceanographic data pipelines, reducing manual reporting overhead by 40% through Python scripting.
- Built a centralized travel database and UI, achieving 85% administrator adoption by streamlining data ingestion workflows.
- Collaborated on defining key performance metrics and developing Python-based decision support tools for cross-disciplinary operational analysis.

**Research Physicist** — *Naval Research Laboratory*
- Collaborated on multi-million-dollar R&D initiatives, optimizing radiation detection models and advising program directors on technical data strategy.

**program directors on technical strategy.** — *Unknown*
- Supported executive decision-making by translating complex simulation results into clear briefings, securing $500K in incremental project funding.

**Commissioned Officer (National Oceanic & Atmospheric Administration) Newport** — *OR*
- Assisted in configuring the Science Computing System, integrating sensor data streams to ensure reliable hydrographic telemetry collection.
- Helped automate legacy inventory tracking and reporting workflows using custom Python scripts and GUI-based application development.

**Research Physicist (Naval Research Laboratory) Washington** — *DC*
- Supported software development for radiation detection models, leveraging high-performance computing to accelerate scenario-based simulation execution.
- Built a Linux-based graphical user interface to streamline simulation software deployment and improve researcher data analysis workflows.

**Faculty Assistant Researcher (University of Maryland College Park) College Park** — *MD*
- Contributed to C++ and Python simulation development, quantifying experimental efficiency for large-scale particle physics data analysis.
- Built custom data analysis and visualization software to generate publication-quality plots for Large Hadron Collider experiments.

**Aerospace Ground Equipment Mechanic (United States Air Force) World Wide 05/2002 - 01/2010** — *Unknown*
- Supported aircraft readiness reporting by maintaining comprehensive maintenance databases and presenting key operational statistics to leadership.
- Helped administer equipment tracking databases to ensure complete data availability and streamline maintenance scheduling for F-16 operations.

---

## Model: `minimax/minimax-m2.1`

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

Wall time: 41.08s

### Rewritten experiences (markdown preview)

---
