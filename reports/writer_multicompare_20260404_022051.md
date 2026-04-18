# Writer (Drafter) comparison — multi-resume simulation

Generated: 2026-04-04T02:20:51.879965+00:00
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
        "Washington State Data Exchange for Public SafetyInternship",
        "National Oceanic and Atmospheric Administration (NOAA)",
        "Connect AI Club, Everett Community College",
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
    "Must be able to handle 4B+ events/day (Context: Ingesting data into NebulaForge)",
    "Must be able to deliver sub-40ms p99 dashboards (Context: Executive KPIs)",
    "Must be able to reduce manual reporting overhead by 40% (Context: NOAA experience)"
  ],
  "domain_vocab": [
    "Python",
    "SQL",
    "Spark",
    "AWS",
    "GCP",
    "Airflow",
    "Dagster",
    "ETL pipelines",
    "real-time lakehouse",
    "QuantMesh metric stores",
    "feature stores",
    "data contracts",
    "LLMs",
    "cloud platforms",
    "vector databases",
    "sensor data",
    "sonar data"
  ],
  "implied_skills": [
    "Real-time Data Ingestion & Processing (Implied by 'Architect and deploy end-to-end ETL pipelines in SQL and Python, integrating diverse data sources to power real-time' and 'analyzing sensor and sonar data to proactively mitigate risks')",
    "Data Lakehouse Management (Implied by experience with 'diverse data sources' and 'real-time' processing, suggesting familiarity with large-scale data architectures)",
    "Metric Store Design & Implementation (Implied by 'define key performance metrics, automated data pipelines' and 'data-driven' solutions, which require structured metric definition)",
    "Low-latency Dashboarding (Implied by 'real-time' ETL and 'decision-support tools' for operations, where quick insights are crucial)",
    "Feature Store Collaboration (Implied by 'data-driven' solutions and 'LLMs, cloud platforms, and vector databases' in projects, indicating understanding of data preparation for ML)",
    "Data Contract Enforcement (Implied by 'rigorous quality standards' in projects and 'translate business requirements into data-driven' solutions, which necessitates data governance)",
    "Mentorship & Leadership (Explicitly stated 'Coach peers' and 'Led cross-discipline teams')",
    "Reliability Engineering / SRE Principles (Implied by 'proactively mitigate risks' and 'reducing manual reporting overhead by 40%', demonstrating a focus on system stability and efficiency)",
    "Cost Optimization for Data Stacks (Implied by 'reducing manual reporting overhead by 40%' and general operational efficiency focus, suggesting an awareness of resource utilization)",
    "Government/Regulated Telemetry Experience (Explicitly stated 'Partner with law-enforcement and EMS stakeholders' and 'National Oceanic and Atmospheric Administration (NOAA)')"
  ],
  "work_archetypes": [
    "Scaling/Optimization",
    "Greenfield Build"
  ],
  "has_projects_section": true,
  "is_student_or_career_changer": true,
  "insider_tips": "Frame the NOAA and Public Safety experience as direct parallels to regulated telemetry and high-stakes data environments. Emphasize the 'real-time' aspects of past ETL work and how it translates to low-latency dashboard requirements."
}
```

## Model: `xiaomi/mimo-v2-flash`

| Metric | Value |
| --- | --- |
| bullet_count | 18 |
| quantification_score | 0.389 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 4 |
| avg_bullet_len | 120.8 |

Wall time: 12.02s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Assisted in mapping critical workflows with law enforcement and EMS stakeholders, translating business requirements into data-driven solutions.
- Contributed to building end-to-end ETL pipelines in SQL and Python, integrating diverse data sources to power real-time analytics.

**President** — *Connect AI Club, Everett Community College*
- Built student consulting projects using LLMs, cloud platforms, and vector databases to deliver proof-of-concepts solving client problems.
- Collaborated on coaching peers in prompt engineering and model tuning, fostering effective communication of AI outcomes to stakeholders.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Supported enterprise operations across 13 research vessels, analyzing sensor and sonar data to proactively mitigate risks.
- Led cross-discipline teams to design Python-based decision-support tools, reducing manual reporting overhead by 40%.
- Launched a Government Official Travel database and UI, achieving 85% administrator adoption within a short timeframe.
- Consulted with operations leads to define key performance metrics and automate data pipelines for weather and oceanographic data.

**Research Physicist** — *Naval Research Laboratory*
- Collaborated on multi-million-dollar R&D projects, guiding model optimization for radiation detection and advising on technical strategy.

**Commissioned Officer (National Oceanic & Atmospheric Administration) Newport** — *Unknown*
- Supported the acquisition of hydrographic data using remote sensing and automated systems.
- Applied coding skills to automate inventory tracking and implement GUI-based tools, fostering cross-team collaboration.

**Research Physicist (Naval Research Laboratory) Washington** — *Unknown*
- Utilized high-performance computing to conduct critical studies on radiation detection for border security applications.
- Supported software development to predict active interrogation procedures and passive detection of WMD/radiological devices.
- Designed a graphic user interface for fast implementation of scenario-based radiation simulation software on Linux.

**Faculty Assistant Researcher (University of Maryland College Park) College Park** — *Unknown*
- Designed software for data analysis and publication-quality plots in particle physics experiments.
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
| bullet_count | 24 |
| quantification_score | 0.542 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 6 |
| avg_bullet_len | 128.2 |

Wall time: 137.09s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Built SQL and Python ETL pipelines to integrate diverse public safety data sources for real-time analytics.
- Collaborated on workflow mapping with law enforcement stakeholders to translate operational needs into structured data models.

**President** — *Connect AI Club, Everett Community College*
- Supported student consulting teams in deploying LLMs and vector databases on cloud platforms for client PoCs.
- Helped peers optimize prompt engineering and iterative model tuning to clearly communicate complex AI project outcomes.
- Built standardized code review frameworks to maintain software quality and foster collaborative development practices across the club.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Built Python decision-support tools that automated sensor data analysis, successfully cutting manual reporting overhead by 40%.
- Supported the launch of an official travel database, driving 85% administrator adoption across distributed operations teams.
- Assisted in defining KPIs and automating weather data pipelines to streamline complex oceanographic research workflows.
- Helped develop Python GUIs and reporting scripts to standardize vessel data ingestion across 13 active research vessels.

**Software Integration Analyst** — *Unknown*

**Research Physicist** — *Naval Research Laboratory*
- Collaborated on multi-million dollar R&D initiatives to optimize radiation detection models for critical security applications.

**Technical Strategy Advisor** — *Unknown*
- Supported technical strategy development by translating complex simulation results into actionable executive briefings for leadership.
- Helped secure $500K in incremental research funding by demonstrating clear, data-driven project outcomes to key stakeholders.

**Commissioned Officer (NOAA)** — *Unknown*
- Assisted in hydrographic data acquisition by deploying remote sensing tools and automated collection systems for field teams.
- Built Python automation scripts to streamline inventory tracking and standardize administrative reporting workflows across the unit.
- Supported sensor-to-server integrations by configuring computing systems to ensure reliable, continuous oceanographic data pipeline operations.

**Research Physicist (Naval Research Laboratory)** — *Unknown*
- Built Linux-based graphical interfaces to rapidly accelerate scenario-based radiation simulation testing for critical border security applications.
- Supported software development for predictive modeling of radiological detection and complex active interrogation workflows across teams.
- Contributed to high-performance computing studies that optimized radiation detection algorithms for advanced border security protocols.

**Faculty Assistant Researcher (University of Maryland College Park)** — *Unknown*
- Built C++ and Python analysis tools to generate publication-quality visualizations for complex particle physics experiments.
- Supported large-scale simulation development to accurately quantify radiation efficiency and validate experimental data models for publication.
- Contributed to peer-reviewed research publications by documenting Large Hadron Collider experimental data workflows and methodologies.

**Aerospace Ground Equipment Mechanic (United States Air Force)** — *Unknown*
- Built maintenance tracking databases to centralize aircraft readiness metrics and streamline daily operational reporting processes.
- Supported F-16 fleet operations by managing detailed maintenance records and presenting readiness statistics to base leadership.
- Assisted in technical documentation review to ensure precise equipment schematics and strict compliance with safety standards.

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

Wall time: 505.27s

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
