# Writer (Drafter) comparison — multi-resume simulation

Generated: 2026-04-04T04:44:04.375851+00:00
Source file: `/mnt/c/Users/jeffd/latexRoot/My_resume_tests/combined_resumes.txt`
Parsed experience blocks (pre-dedupe): 15
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
        "National Oceanic and Atmospheric Administration (NOAA)",
        "Naval Research Laboratory",
        "University of Maryland College Park",
        "Washington State Data Exchange for Public SafetyInternship"
      ],
      "roles": [
        "Operations Analyst",
        "Professional",
        "Operations Manager",
        "Research Physicist",
        "Commissioned Officer (National Oceanic & Atmospheric Administration) Newport",
        "Research Physicist (Naval Research Laboratory) Washington",
        "Faculty Assistant Researcher (University of Maryland College Park) College Park",
        "Aerospace Ground Equipment Mechanic (United States Air Force) World Wide 05/2002 - 01/2010",
        "Faculty Assistant Researcher",
        "Operations Manager of Marine Operations Center"
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
    "Must be able to handle 4B+ events/day (Context: Ingesting data into real-time lakehouse)",
    "Must be able to deliver sub-40ms p99 dashboards (Context: Executive KPIs)",
    "Must have a track record of measurable reliability wins (Context: Data pipelines and systems)"
  ],
  "domain_vocab": [
    "Python",
    "SQL",
    "Spark",
    "NebulaForge",
    "QuantMesh",
    "AWS",
    "GCP",
    "Airflow",
    "Dagster",
    "Lakehouse",
    "ETL",
    "Feature Stores",
    "Data Contracts",
    "Telemetry",
    "Streaming Stacks",
    "Batch Stacks"
  ],
  "implied_skills": [
    "Data Governance & Contracts (Implied by 'enforce data contracts' in job and user's 'translate business requirements into data-driven' and 'rigorous quality standards' in projects)",
    "Real-time Data Processing (Implied by job's '4B+ events/day into NebulaForge, our real-time lakehouse' and user's 'Architect and deploy end-to-end ETL pipelines... to power real-time')",
    "Metric Store Management (Implied by job's 'QuantMesh metric stores' and user's 'define key performance metrics, automated data pipelines')",
    "High-Performance Dashboarding (Implied by job's 'sub-40ms p99 dashboards' and user's 'Python-based decision-support tools' and 'reporting scripts to streamline vessel data ingestion')",
    "Feature Store Collaboration (Implied by job's 'Partner with ML on feature stores' and user's 'employing LLMs, cloud platforms, and vector databases to deliver PoCs' and 'iterative model tuning')",
    "Mentorship & Leadership (Implied by job's 'mentor two junior engineers' and user's 'President' role, 'Coach peers', and 'Led cross-discipline teams')",
    "Cloud Platform Experience (Implied by job's 'AWS or GCP' and user's 'employing LLMs, cloud platforms' in projects)",
    "Airflow/Dagster equivalent (Implied by job's 'Airflow or Dagster' and user's 'Architect and deploy end-to-end ETL pipelines' and 'automated data pipelines')",
    "Reliability Engineering (Implied by job's 'measurable reliability wins' and user's 'analyzing sensor and sonar data to proactively mitigate risks' and 'rigorous quality standards')",
    "Time-Series Anomaly Detection (Implied by user's 'analyzing sensor and sonar data to proactively mitigate risks' - handling noisy, continuous signal data for predictive insights)",
    "Cost Optimization for Data Stacks (Implied by job's 'cost guardrails for big batch + streaming stacks' and user's 'reducing manual reporting overhead by 40%' through automation)"
  ],
  "work_archetypes": [
    "Scaling/Optimization",
    "Greenfield Build"
  ],
  "has_projects_section": true,
  "is_student_or_career_changer": true,
  "insider_tips": "Frame past operational and research roles as direct experience in building robust, data-driven systems under high-stakes conditions. Emphasize how leading projects and teams, even in academic or government settings, translates to mentoring and driving measurable impact in a fast-paced analytics environment."
}
```

## Model: `qwen/qwen3.5-9b`

| Metric | Value |
| --- | --- |
| bullet_count | 21 |
| quantification_score | 0.381 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 5 |
| avg_bullet_len | 114.4 |

Wall time: 79.57s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Built end-to-end ETL pipelines in SQL and Python to integrate diverse data sources for real-time analytics.
- Assisted stakeholders in mapping critical workflows to translate business requirements into data-driven solutions.

**Professional** — *Connect AI Club, Everett Community College*
- Contributed to student consulting projects using LLMs, cloud platforms, and vector databases to deliver PoCs.
- Collaborated on iterative model tuning and data storytelling to communicate AI outcomes effectively to clients.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Built Python-based decision-support tools that reduced manual reporting overhead by 40% for cross-discipline teams.
- Supported operations leads in defining key performance metrics and automating data pipelines for weather data.
- Built a Government Official Travel database achieving 85% administrator adoption within the first year.

**Research Physicist** — *Naval Research Laboratory*
- Collaborated on multi-million-dollar R&D projects guiding model optimization for radiation detection and advisory systems.
- Assisted in translating complex simulation results into executive briefings to secure $500K in incremental funding.

**Commissioned Officer (National Oceanic & Atmospheric Administration) Newport** — *Unknown*
- Contributed to coding automation for DVD library inventory tracking and implemented GUI-based letter generators.
- Supported Science Computing System configuration and connected sensors to servers for efficient data management.

**Research Physicist (Naval Research Laboratory) Washington** — *Unknown*
- Contributed to high performance computing studies on radiation detection for border security applications.
- Built graphic user interface for fast implementation of scenario-based radiation simulation software running on Linux.

**Faculty Assistant Researcher (University of Maryland College Park) College Park** — *Unknown*
- Built software for data analysis and publication quality plots in particle physics experiments using C++ and Python.
- Contributed to publications on experiments conducted at the Large Hadron Collider contributing to scientific knowledge.

**Aerospace Ground Equipment Mechanic (United States Air Force) World Wide 05/2002 - 01/2010** — *Unknown*
- Supported equipment database administration and Supported maintenance operations to ensure aircraft data availability.
- Supported aircraft maintenance records supporting F-16 fighter jets and Supported readiness statistics to leadership.

**Faculty Assistant Researcher** — *University of Maryland College Park*
- Contributed to simulation studies and Built gradient-based solvers using Hadoop and LHC computing grid infrastructure.
- Built software and Contributed to distributed computing to analyze particle accelerator data for publication quality outputs.

**Operations Manager of Marine Operations Center** — *Unknown*
- Built metric tracking tools and Supported status briefings for fleet leadership managing 13 oceanographic research vessels.
- Supported data ingestion architecture including collection, ETL, and dashboard creation for real-time visibility.

---

## Model: `inception/mercury-2`

| Metric | Value |
| --- | --- |
| bullet_count | 28 |
| quantification_score | 1.0 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 5 |
| avg_bullet_len | 117.8 |

Wall time: 4.45s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Assisted in mapping law‑enforcement and EMS stakeholder workflows to data‑driven requirements, reducing manual hand‑offs by 30%.
- Built end‑to‑end ETL pipelines in SQL and Python, integrating 12 disparate data sources to enable real‑time incident dashboards.

**Professional** — *Connect AI Club, Everett Community College*
- Collaborated on student consulting projects delivering PoCs with LLMs, cloud platforms, and vector databases, achieving 4 client prototypes in 6 months.
- Helped coach peers on prompt engineering and model tuning, improving prototype accuracy by 15% on average.
- Assisted in facilitating client‑style workshops and code reviews, maintaining 100% compliance with quality standards.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Assisted in directing operations across 13 research vessels, integrating sensor and sonar data to proactively mitigate risks, decreasing incident response time by 25%.
- Built Python‑based decision‑support tools that automated reporting, cutting manual overhead by 40%.
- Contributed to launching a Government Official Travel database, achieving 85% administrator adoption within 2 months.

**Research Physicist** — *Naval Research Laboratory*
- Collaborated on multi‑million‑dollar R&D projects, optimizing radiation‑detection models that improved detection sensitivity by 18%.
- Assisted in translating simulation results into executive briefings, securing $500K incremental funding.

**Commissioned Officer (National Oceanic & Atmospheric Administration) Newport** — *Unknown*
- Assisted in acquiring hydrographic data using remote sensing and automated systems, increasing data collection volume by 22%.
- Built GUI‑based Sea Service letter generator and automated DVD inventory tracking, reducing processing time by 35%.
- Supported configuration of Science Computing System and sensor integration, improving data pipeline reliability by 12%.

**Research Physicist (Naval Research Laboratory) Washington** — *Unknown*
- Collaborated on high‑performance computing studies for radiation detection, delivering simulation runs 3× faster than prior methods.
- Assisted in software development for active interrogation prediction, enhancing detection accuracy by 10%.
- Built Linux GUI for scenario‑based radiation simulations, shortening setup time by 40%.

**Faculty Assistant Researcher (University of Maryland College Park) College Park** — *Unknown*
- Built software for data analysis and publication‑quality plots in particle physics, reducing analysis time by 28%.
- Contributed to simulations for radiation studies using C++ and Python, improving efficiency quantification precision by 12%.
- Assisted in co‑authoring LHC experiments papers, supporting 5 peer‑reviewed publications.

**Aerospace Ground Equipment Mechanic (United States Air Force) World Wide** — *Unknown*
- Assisted in administering equipment databases and supervising maintenance, achieving 99% aircraft data availability.
- Built maintenance record system for F‑16 fleet, delivering readiness statistics to leadership with 0% reporting errors.
- Supported interpretation of technical schematics, reducing troubleshooting time by 20%.

**Faculty Assistant Researcher** — *University of Maryland College Park*
- Collaborated on simulation studies using Hadoop and LHC computing grid, processing 2 TB of accelerator data per run.
- Built distributed‑computing software for particle accelerator analysis, cutting processing latency by 30%.
- Assisted in authoring publications, contributing to 4 peer‑reviewed papers.

**Operations Manager of Marine Operations Center** — *Unknown*
- Assisted in providing in‑house software and project‑management support for NOAA shore‑side operations, improving project delivery speed by 18%.
- Built quantitative research tools and surveys, generating actionable insights for 13 vessels and increasing metric tracking adoption by 25%.
- Contributed to launching Government Official Travel database and ingestion architecture, handling 1,200 daily records with 99.5% uptime.

---

## Model: `minimax/minimax-m2-7`

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

## Model: `openai/gpt-5.4-mini-medium`

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

## Model: `z-ai/glm-5`

| Metric | Value |
| --- | --- |
| bullet_count | 30 |
| quantification_score | 0.3 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 5 |
| avg_bullet_len | 92.9 |

Wall time: 327.76s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Collaborated with law enforcement and EMS stakeholders to map workflows and translate requirements into data solutions
- Built ETL pipelines using SQL and Python, integrating diverse data sources for real-time analytics support

**President** — *Connect AI Club, Everett Community College*
- Built proof-of-concept solutions using LLMs, cloud platforms, and vector databases for student consulting projects
- Helped coach peers on prompt engineering and model tuning, enhancing team AI capabilities
- Collaborated on workshops and code reviews, ensuring quality standards and fostering innovation

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Supported operations across 13 research vessels, analyzing sensor and sonar data to mitigate operational risks
- Collaborated on Python-based decision-support tools, reducing manual reporting overhead by 40%
- Built Government Official Travel database achieving 85% administrator adoption within first quarter
- Contributed to automated data pipelines for weather and oceanographic metrics, streamlining data ingestion
- Built Python GUIs and reporting scripts to streamline vessel data workflows and cross-team collaboration

**Research Physicist** — *Naval Research Laboratory*
- Collaborated on multi-million-dollar R&D projects, guiding model optimization for radiation detection
- Supported executive briefings on simulation results, securing $500K in incremental funding

**Commissioned Officer (NOAA)** — *Unknown*
- Supported hydrographic data acquisition using remote sensing and automated systems
- Built GUI-based automation tools for inventory tracking and Sea Service letter generation
- Contributed to Science Computing System configuration and sensor integrations as Ancillary Data Manager

**Research Physicist (Naval Research Laboratory)** — *Unknown*
- Built radiation detection studies using high-performance computing for border security applications
- Supported software development for WMD detection prediction and passive detection systems
- Built GUI for radiation simulation software running on Linux systems

**Faculty Assistant Researcher (University of Maryland)** — *Unknown*
- Built simulation studies and gradient-based solvers using Hadoop and LHC computing grid
- Contributed to software design using distributed computing for particle accelerator data analysis
- Co-authored publications on Large Hadron Collider experiments

**Aerospace Ground Equipment Mechanic (USAF)** — *Unknown*
- Supported equipment database administration and maintenance operations for F-16 fighter jets
- Built maintenance records and presented readiness statistics to base leadership

**Faculty Assistant Researcher** — *University of Maryland College Park*
- Built simulation studies using Hadoop and LHC computing grid for particle physics data analysis
- Contributed to software design using distributed computing to analyze particle accelerator data
- Co-authored publications on Large Hadron Collider experiments

**Operations Manager of Marine Operations Center** — *Unknown*
- Supported software and project management for Regional Director's shore-side operations
- Built metric tracking tools and supported fleet briefings for 13 oceanographic research vessels
- Contributed to database management, security infrastructure, and real-time dashboard creation
- Built Government Official Travel database with comprehensive data ingestion architecture

---

## Model: `anthropic/claude-sonnet-4-6-non-reasoning-low-effort`

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

## Model: `openai/gpt-5.4-nano-medium`

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

## Model: `openai/gpt-5.4-non-reasoning`

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

## Model: `z-ai/glm-5-non-reasoning`

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

## Model: `google/gemini-3-flash-reasoning`

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

## Model: `kimi/kimi-k2.5-non-reasoning`

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

## Run notes (automated)

- **`minimax/minimax-m2-7`**: Drafter fell back (often empty JSON when the provider returns reasoning-only or truncated `message.content` at large prompt sizes). Try `--drafter-max-tokens 32000` or a different MiniMax provider route.
- Compare **Xiaomi** vs **Qwen** (and others) by reading the bullet sections above; rubric scores are heuristics only.
