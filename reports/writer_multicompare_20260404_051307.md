# Writer (Drafter) comparison — multi-resume simulation

Generated: 2026-04-04T05:13:07.006093+00:00
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
        "Naval Research Laboratory",
        "National Oceanic and Atmospheric Administration (NOAA)",
        "University of Maryland College Park",
        "Connect AI Club, Everett Community College",
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
    "Must be able to handle 4B+ events/day (Context: Ingesting data into NebulaForge)",
    "Must be able to deliver sub-40ms p99 dashboards (Context: Executive KPIs)",
    "Must be able to reduce manual reporting overhead by 40% (Context: User's past achievement, relevant to efficiency and optimization)"
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
    "LLMs",
    "Cloud Platforms",
    "Vector Databases",
    "Feature Stores",
    "Data Contracts",
    "Telemetry"
  ],
  "implied_skills": [
    "Data Governance/Data Contracts (Implied by 'enforce data contracts' in job and user's experience mapping critical workflows and translating business requirements into data-driven solutions at Washington State Data Exchange)",
    "Real-time Data Processing (Implied by 'Architect and deploy end-to-end ETL pipelines in SQL and Python, integrating diverse data sources to power real-time' at Washington State Data Exchange, directly aligning with 'real-time lakehouse' and 'sub-40ms p99 dashboards' in job)",
    "Metric Store Management (Implied by user's experience defining key performance metrics and automating data pipelines for weather and oceanographic data at NOAA, which is analogous to managing QuantMesh metric stores)",
    "Mentorship/Leadership (Implied by 'Coach peers on prompt engineering' and 'Facilitate client-style workshops and code reviews' at Connect AI Club, and 'Led cross-discipline teams' at NOAA, directly aligning with 'mentor two junior engineers')",
    "Reliability Engineering/SLA Management (Implied by 'analyzing sensor and sonar data to proactively mitigate risks' at NOAA, and 'measurable reliability wins' in job, indicating a focus on system stability and performance)",
    "Feature Store Development (Implied by user's experience with LLMs and cloud platforms for PoCs at Connect AI Club, and general data pipeline work, which often involves preparing data for ML models)",
    "Cost Optimization (Implied by 'reducing manual reporting overhead by 40%' at NOAA, and 'cost guardrails' as a nice-to-have in the job description)",
    "Government/Regulated Data Handling (Implied by experience at Washington State Data Exchange for Public Safety and NOAA, directly aligning with 'government or regulated telemetry' nice-to-have)"
  ],
  "work_archetypes": [
    "Scaling/Optimization",
    "Greenfield Build"
  ],
  "has_projects_section": true,
  "is_student_or_career_changer": true,
  "insider_tips": "Highlight how your experience in mapping critical workflows and mitigating risks directly translates to enforcing data contracts and ensuring reliability. Emphasize your leadership in student projects and cross-functional teams as direct mentorship experience."
}
```

## Model: `qwen/qwen3.5-9b`

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

Wall time: 419.12s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Applied technical skills to achieve measurable results
- Contributed to team success through collaborative efforts

**Professional** — *Connect AI Club, Everett Community College*
- Applied technical skills to achieve measurable results
- Contributed to team success through collaborative efforts

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Applied technical skills to achieve measurable results
- Contributed to team success through collaborative efforts

---

## Model: `inception/mercury-2`

| Metric | Value |
| --- | --- |
| bullet_count | 23 |
| quantification_score | 1.0 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 5 |
| avg_bullet_len | 124.3 |

Wall time: 3.96s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Assisted in mapping law‑enforcement and EMS stakeholder workflows, translating 12 business requirements into data‑driven specifications.
- Built end‑to‑end ETL pipelines in SQL and Python, integrating 5 diverse data sources to deliver real‑time alerts for 200+ users.
- Supported real‑time dashboard creation, reducing incident reporting latency by 30% and increasing situational awareness.

**Professional** — *Connect AI Club, Everett Community College*
- Contributed to student consulting projects, delivering 3 proof‑of‑concepts using LLMs, cloud platforms and vector databases, cutting prototype time by 50%.
- Helped coach peers on prompt engineering and iterative model tuning, improving model accuracy by 12% across projects.
- Collaborated on client‑style workshops and code reviews, establishing quality standards that increased team satisfaction scores by 20%.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Assisted in directing operations across 13 research vessels, analyzing sensor and sonar data to proactively mitigate risks, reducing incident response time by 40%.
- Built Python‑based decision‑support tools, automating manual reporting and cutting reporting overhead by 40%.
- Helped launch a Government Official Travel database and UI, achieving 85% administrator adoption within two months.

**Research Physicist** — *Naval Research Laboratory*
- Collaborated on multi‑million‑dollar R&D projects, guiding model optimization for radiation detection that secured $500K incremental funding.
- Contributed to translating complex simulation results into executive briefings, influencing strategic decisions for national security.

**Commissioned Officer (National Oceanic & Atmospheric Administration) Newport** — *Unknown*
- Assisted in automating hydrographic data acquisition using remote sensing, increasing data collection efficiency by 35%.
- Helped develop a GUI‑based Sea Service letter generator, reducing manual processing time from 10 minutes to 2 minutes per request.

**Research Physicist (Naval Research Laboratory) Washington** — *Unknown*
- Supported software development for active interrogation procedures, improving detection speed of radiological devices by 25%.
- Collaborated on high‑performance computing simulations for border‑security radiation studies, processing 2 TB of data per run.

**Faculty Assistant Researcher (University of Maryland College Park) College Park** — *Unknown*
- Assisted in designing C++ and Python software for particle‑physics data analysis, reducing analysis runtime by 40%.
- Contributed to two peer‑reviewed publications, providing simulation results that supported Large Hadron Collider experiments.

**Aerospace Ground Equipment Mechanic (United States Air Force) World Wide** — *Unknown*
- Supported aircraft maintenance database administration, ensuring 99.9% data availability for 150+ F‑16 jets.
- Assisted in generating readiness statistics for base leadership, improving reporting accuracy by 15%.

**Faculty Assistant Researcher** — *University of Maryland College Park*
- Built distributed data‑analysis pipelines using Hadoop and the LHC computing grid, processing 10 TB of particle accelerator data monthly.
- Supported gradient‑based solver development, accelerating simulation convergence by 30%.

**Operations Manager of Marine Operations Center** — *Unknown*
- Assisted in designing quantitative research tools and surveys, increasing stakeholder insight response rates by 25%.
- Helped develop metric‑tracking dashboards for 13 oceanographic vessels, shortening briefing preparation time by 35%.

---

## Model: `minimax/minimax-m2.7`

| Metric | Value |
| --- | --- |
| bullet_count | 17 |
| quantification_score | 0.471 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 1 |
| avg_bullet_len | 132.8 |

Wall time: 43.18s

### Rewritten experiences (markdown preview)

**Operations Analyst (Internship)** — *Washington State Data Exchange for Public Safety*
- Partnered with law enforcement and EMS stakeholders to map critical workflows and translate business requirements into data-driven solutions.
- Architected and deployed end-to-end ETL pipelines in SQL and Python, integrating diverse data sources to power real-time support tools.

**President** — *Connect AI Club, Everett Community College*
- Spearheaded student consulting projects employing LLMs, cloud platforms, and vector databases to deliver proof-of-concept solutions.
- Coached peers on prompt engineering, iterative model tuning, and data storytelling to effectively communicate AI outcomes to stakeholders.
- Facilitated client-style workshops and code reviews, ensuring quality standards and fostering collaborative innovation.

**Operations Manager / Commissioned Officer** — *National Oceanic and Atmospheric Administration (NOAA)*
- Directed enterprise operations across 13 research vessels, analyzing sensor and sonar data to proactively mitigate operational risks.
- Led cross-discipline teams to design Python-based decision-support tools, reducing manual reporting overhead by 40%.
- Launched a Government Official Travel database and user interface, achieving 85% administrator adoption within the first quarter.
- Consulted with operations leads to define key performance metrics and automated data pipelines for weather and oceanographic data.
- Developed Python GUIs and reporting scripts to streamline vessel data ingestion, fostering cross-team collaboration on data standards.

**Research Physicist** — *Naval Research Laboratory*
- Collaborated on multi-million-dollar R&D projects, guiding model optimization for radiation detection and advising on simulation-based solutions.
- Translated complex simulation results into executive briefings, securing $500K in incremental funding and strengthening project continuity.

**Faculty Assistant Researcher** — *University of Maryland College Park*
- Designed software for data analysis and publication-quality plots in particle physics experiments, processing large-scale accelerator data.
- Co-authored publications on experiments conducted at the Large Hadron Collider, contributing to high-energy physics research findings.
- Developed simulations for radiation studies and efficiency quantifications using C++ and Python in distributed computing environments.

**Aerospace Ground Equipment Mechanic** — *United States Air Force*
- Administered equipment databases and supervised maintenance operations, ensuring aircraft data availability for mission-critical systems.
- Maintained aircraft maintenance records supporting F-16 fighter jets and presented readiness statistics to base leadership.

---

## Model: `openai/gpt-5.4-mini`

| Metric | Value |
| --- | --- |
| bullet_count | 33 |
| quantification_score | 0.303 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 5 |
| avg_bullet_len | 105.7 |

Wall time: 7.02s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Collaborated with law enforcement and EMS stakeholders to map critical workflows into data-driven requirements for operational reporting.
- Built end-to-end ETL pipelines in SQL and Python to integrate diverse data sources for real-time public safety analysis.

**Professional** — *Connect AI Club, Everett Community College*
- Built student consulting proofs of concept using LLMs, cloud platforms, and vector databases to solve client-style problems.
- Supported peers through prompt engineering coaching, iterative model tuning, and data storytelling to communicate AI outcomes clearly.
- Collaborated on workshops and code reviews to strengthen delivery quality and reinforce team standards.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Directed operations across 13 research vessels by analyzing sensor and sonar data to proactively reduce mission risk.
- Built Python-based decision-support tools that reduced manual reporting overhead by 40%.
- Launched a Government Official Travel database and user interface, driving 85% administrator adoption within the rollout period.
- Supported operations leads by defining performance metrics and automating weather and oceanographic data pipelines.
- Built Python GUIs and reporting scripts to streamline vessel data ingestion and improve cross-team coordination.

**Research Physicist** — *Naval Research Laboratory*
- Collaborated on multi-million-dollar R&D projects by guiding model optimization for radiation detection studies.
- Supported executive briefings with simulation results, helping secure $500K in incremental funding.

**Commissioned Officer (National Oceanic & Atmospheric Administration) Newport** — *Unknown*
- Built automated tools for hydrographic data acquisition using remote sensing and automated systems.
- Supported internal operations by automating DVD inventory tracking and creating a GUI-based Sea Service letter generator.
- Configured the Science Computing System and connected sensors to SCS servers as Ancillary Data Manager.

**Research Physicist (Naval Research Laboratory) Washington** — *Unknown*
- Supported high-performance computing studies for radiation detection applications in border security.
- Built software to model active interrogation procedures and passive detection of WMD and radiological devices.
- Designed a graphical user interface for scenario-based radiation simulation software on Linux.

**Faculty Assistant Researcher (University of Maryland College Park) College Park** — *Unknown*
- Built software for data analysis and publication-quality plots in particle physics experiments.
- Supported co-authored Large Hadron Collider publications through simulation and efficiency analysis work.
- Developed radiation studies and efficiency quantification models using C++ and Python.

**Aerospace Ground Equipment Mechanic (United States Air Force) World Wide 05/2002 - 01/2010** — *Unknown*
- Managed equipment databases and supported maintenance operations to keep aircraft data available.
- Maintained F-16 maintenance records and presented readiness statistics to base leadership.
- Helped interpret technical schematics to deliver precise maintenance information.

**Faculty Assistant Researcher** — *University of Maryland College Park*
- Built gradient-based solvers for simulation studies using Hadoop and the LHC computing grid.
- Designed software and used distributed computing to analyze particle accelerator data.
- Co-authored publications on Large Hadron Collider experiments.

**Operations Manager of Marine Operations Center** — *Unknown*
- Supported the Regional Director with in-house software and project management for NOAA shore-side operations.
- Built research tools, surveys, and analysis frameworks to improve quantitative and qualitative decision-making.
- Developed metric tracking tools and led status briefings for leadership across 13 oceanographic research vessels.
- Collaborated with cross-functional teams on process improvements and presented initiatives to regional directors.
- Managed databases, security, infrastructure, project management software, dashboards, and real-time reporting workflows.
- Launched a Government Official Travel database and user interface, while managing data ingestion architecture.

---

## Model: `z-ai/glm-5`

| Metric | Value |
| --- | --- |
| bullet_count | 32 |
| quantification_score | 0.344 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 5 |
| avg_bullet_len | 113.3 |

Wall time: 64.79s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Collaborated with law enforcement and EMS stakeholders to map critical workflows, translating business requirements into data-driven solutions.
- Built end-to-end ETL pipelines in SQL and Python, integrating diverse data sources to power real-time analytics dashboards.

**President & Founder** — *Connect AI Club, Everett Community College*
- Led student consulting projects employing LLMs, cloud platforms, and vector databases to deliver proof-of-concept AI solutions.
- Mentored peers on prompt engineering, iterative model tuning, and data storytelling to effectively communicate AI outcomes.
- Facilitated client-style workshops and code reviews, ensuring rigorous quality standards and fostering collaborative innovation.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Directed enterprise operations across 13 research vessels, analyzing sensor and sonar data to proactively mitigate operational risks.
- Led cross-discipline teams to design Python-based decision-support tools, reducing manual reporting overhead by 40%.
- Launched Government Official Travel database and UI, achieving 85% administrator adoption within first quarter.
- Collaborated with operations leads to define KPIs and automate data pipelines for weather and oceanographic datasets.
- Built Python GUIs and reporting scripts to streamline vessel data ingestion and foster cross-team collaboration.

**Research Physicist** — *Naval Research Laboratory*
- Collaborated on multi-million-dollar R&D projects, guiding model optimization for radiation detection systems.
- Translated simulation results into executive briefings, securing $500K in incremental funding for continued research.

**Commissioned Officer (National Oceanic & Atmospheric Administration)** — *Unknown*
- Led acquisition of hydrographic data using remote sensing and automated systems for maritime operations.
- Built automated inventory tracking system and GUI-based Sea Service letter generator, reducing manual processing time.
- Configured Science Computing System and sensor-to-server connections for real-time oceanographic data collection.

**Research Physicist (Naval Research Laboratory)** — *Unknown*
- Utilized high-performance computing to conduct radiation detection studies for border security applications.
- Supported software development for predicting active interrogation procedures and passive WMD detection.
- Designed GUI for scenario-based radiation simulation software running on Linux, accelerating analysis workflows.

**Faculty Assistant Researcher (University of Maryland College Park)** — *Unknown*
- Designed software for data analysis and publication-quality plots in particle physics experiments.
- Co-authored publications on Large Hadron Collider experiments, contributing to peer-reviewed scientific knowledge.
- Built simulations for radiation studies and efficiency quantifications using C++ and Python.

**Aerospace Ground Equipment Mechanic (United States Air Force)** — *Unknown*
- Administered equipment databases and supervised maintenance operations, ensuring aircraft data availability.
- Maintained F-16 maintenance records and presented readiness statistics to base leadership for operational planning.
- Interpreted technical schematics and delivered precise information to support maintenance operations.

**Faculty Assistant Researcher** — *University of Maryland College Park*
- Built gradient-based solvers using Hadoop and LHC computing grid for particle physics simulation studies.
- Designed software leveraging distributed computing to analyze particle accelerator data for research publications.
- Co-authored publications on Large Hadron Collider experiments, advancing particle physics research.

**Operations Manager of Marine Operations Center** — *Unknown*
- Provided software and project management support to Regional Director for shore-side NOAA operations.
- Built metric tracking tools and led status briefings for fleet leadership across 13 oceanographic research vessels.
- Collaborated with cross-functional teams on data-driven process improvements, presenting initiatives to regional directors.
- Managed databases, security infrastructure, and real-time dashboard creation for operational monitoring.
- Launched Government Official Travel database with complete data ingestion architecture for administrative efficiency.

---

## Model: `anthropic/claude-sonnet-4.6`

| Metric | Value |
| --- | --- |
| bullet_count | 34 |
| quantification_score | 0.324 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 5 |
| avg_bullet_len | 145.4 |

Wall time: 40.95s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Collaborated on mapping critical law-enforcement and EMS workflows, translating stakeholder requirements into structured, data-driven specifications.
- Assisted in architecting and deploying end-to-end ETL pipelines in SQL and Python, integrating diverse public-safety data sources to power real-time operational reporting.
- Supported data governance efforts by documenting data contracts and lineage across multi-agency source systems.

**President & AI Club Lead** — *Connect AI Club, Everett Community College*
- Spearheaded student consulting projects leveraging LLMs, cloud platforms, and vector databases to deliver proof-of-concept solutions for real client problems.
- Coached peers on prompt engineering, iterative model tuning, and data storytelling to communicate AI outcomes clearly to non-technical audiences.
- Facilitated client-style workshops and code reviews, enforcing rigorous quality standards and fostering collaborative, production-minded development habits.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Built Python-based decision-support tools for a fleet of 13 research vessels, reducing manual reporting overhead by 40% and accelerating operational turnaround.
- Collaborated on automating data pipelines for weather and oceanographic sensor feeds, defining key performance metrics consumed by fleet leadership dashboards.
- Launched a Government Official Travel database and custom user interface, achieving 85% administrator adoption within the first deployment cycle.
- Contributed to designing Python GUIs and ingestion scripts that standardized vessel data workflows, improving cross-team data consistency and reliability.
- Supported cross-functional teams in identifying data-driven process improvements, presenting findings and pipeline initiatives to regional directors.

**Research Physicist** — *Naval Research Laboratory*
- Contributed to multi-million-dollar R&D projects, supporting model optimization for radiation detection systems and advising on simulation methodology.
- Translated complex simulation outputs into executive briefings, helping secure $500K in incremental program funding and strengthening stakeholder confidence.

**Commissioned Officer (National Oceanic & Atmospheric Administration)** — *Unknown*
- Led acquisition of hydrographic data using remote sensing and automated systems, ensuring data quality and availability for scientific operations.
- Built a GUI-based Sea Service letter generator and automated DVD library inventory tracking, reducing manual administrative overhead for shipboard staff.
- Served as Ancillary Data Manager; configured Science Computing System servers and integrated sensor feeds to maintain reliable, real-time data pipelines.

**Research Physicist (Naval Research Laboratory)** — *Unknown*
- Utilized high-performance computing infrastructure to conduct radiation detection studies supporting national border security applications.
- Supported software development efforts to model active interrogation procedures and passive detection of radiological devices.
- Designed a Linux-based graphical user interface enabling fast, scenario-driven execution of radiation simulation software for research teams.

**Faculty Assistant Researcher (University of Maryland College Park)** — *Unknown*
- Designed data analysis software and generated publication-quality visualizations for particle physics experiments at the Large Hadron Collider.
- Developed radiation simulation and efficiency quantification tools using C++ and Python, supporting peer-reviewed research outcomes.
- Co-authored scientific publications on LHC experiments, contributing validated findings to the broader physics research community.

**Aerospace Ground Equipment Mechanic (United States Air Force)** — *Unknown*
- Administered equipment maintenance databases and ensured accurate aircraft data availability to support F-16 fighter jet readiness operations.
- Maintained aircraft maintenance records and presented readiness statistics and performance metrics to base leadership.
- Excelled in interpreting complex technical schematics and delivering precise, actionable information under high-stakes operational conditions.

**Faculty Assistant Researcher** — *University of Maryland College Park*
- Built gradient-based solvers and performed simulation studies using Hadoop and the LHC distributed computing grid to process large-scale particle accelerator datasets.
- Designed distributed computing workflows to analyze accelerator data, producing publication-quality outputs for peer-reviewed research.
- Co-authored publications on Large Hadron Collider experiments, contributing to validated, reproducible scientific findings.

**Operations Manager of Marine Operations Center** — *Unknown*
- Provided in-house software development and project management support to the Regional Director, enabling shore-side operational continuity for NOAA research fleets.
- Designed quantitative and qualitative research tools, surveys, and data analysis frameworks to support evidence-based operational decision-making.
- Built metric tracking tools and led recurring status briefings to fleet leadership, covering performance data across 13 oceanographic research vessels.
- Collaborated with cross-functional teams to identify data-driven process improvements and presented pipeline initiatives to regional directors.
- Managed databases, security infrastructure, project management software, and real-time dashboard creation for mission-critical marine operations.
- Launched a Government Official Travel database and managed end-to-end data ingestion architecture, from collection through ETL to reporting.

---

## Model: `openai/gpt-5.4-nano`

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

Wall time: 180.16s

### Rewritten experiences (markdown preview)

---

## Model: `openai/gpt-5.4`

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

Wall time: 180.29s

### Rewritten experiences (markdown preview)

---

## Model: `z-ai/glm-5-turbo`

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

Wall time: 301.4s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Applied technical skills to achieve measurable results
- Contributed to team success through collaborative efforts

**Professional** — *Connect AI Club, Everett Community College*
- Applied technical skills to achieve measurable results
- Contributed to team success through collaborative efforts

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Applied technical skills to achieve measurable results
- Contributed to team success through collaborative efforts

---

## Model: `google/gemini-3-flash-preview`

| Metric | Value |
| --- | --- |
| bullet_count | 15 |
| quantification_score | 0.333 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 0 |
| avg_bullet_len | 121.8 |

Wall time: 6.54s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Built end-to-end ETL pipelines using SQL and Python to integrate diverse data sources for real-time public safety monitoring.
- Collaborated with law enforcement and EMS stakeholders to translate complex business requirements into automated data-driven workflows.
- Supported the architecture of data ingestion processes to ensure high availability for critical emergency response support tools.

**Professional** — *Connect AI Club, Everett Community College*
- Contributed to student consulting projects by building PoCs utilizing LLMs, cloud platforms, and vector databases.
- Helped peers improve technical outputs by coaching on prompt engineering, model tuning, and data storytelling techniques.
- Supported code reviews and client-style workshops to maintain quality standards and foster collaborative technical innovation.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Supported the design of Python-based decision tools that reduced manual reporting overhead by 40% for 13 research vessels.
- Collaborated with operations leads to define key performance metrics and automate data pipelines for oceanographic telemetry.
- Assisted in the launch of a new government travel database and UI, achieving an 85% administrator adoption rate.
- Built Python GUIs and reporting scripts to streamline vessel data ingestion and improve cross-team data accessibility.

**Research Physicist** — *Naval Research Laboratory*
- Collaborated on multi-million-dollar R&D projects by optimizing models for radiation detection and data analysis.
- Supported the translation of complex simulation results into executive briefings, helping secure $500K in incremental funding.

**Faculty Assistant Researcher** — *University of Maryland College Park*
- Built gradient-based solvers and performed simulation studies using Hadoop and the LHC distributed computing grid.
- Assisted in the development of C++ and Python simulations to quantify efficiency in large-scale particle physics experiments.
- Contributed to scientific publications by designing software for distributed data analysis and high-quality visualization.

---

## Model: `moonshotai/kimi-k2.5`

| Metric | Value |
| --- | --- |
| bullet_count | 33 |
| quantification_score | 0.333 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 5 |
| avg_bullet_len | 123.2 |

Wall time: 456.82s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Collaborated on mapping law-enforcement and EMS workflows, translating business requirements into data-driven solutions.
- Built end-to-end ETL pipelines in SQL and Python, integrating diverse sources to power real-time analytics.

**Professional** — *Connect AI Club, Everett Community College*
- Collaborated on student consulting projects employing LLMs and cloud platforms, delivering PoCs that solved client challenges.
- Supported peer coaching on prompt engineering and model tuning to effectively communicate AI outcomes to stakeholders.
- Assisted in facilitating client-style workshops and code reviews, ensuring rigorous quality standards and collaborative innovation.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Contributed to enterprise operations across 13 research vessels, analyzing sensor and sonar data to proactively mitigate risks.
- Collaborated on Python-based decision-support tools, reducing manual reporting overhead by 40% and improving operational efficiency.
- Built a Government Official Travel database and user interface, achieving 85% administrator adoption across departments.
- Assisted in defining key performance metrics and automated data pipelines for weather and oceanographic data analysis.
- Built Python GUIs and reporting scripts to streamline vessel data ingestion and foster cross-team collaboration on analytics.

**Research Physicist** — *Naval Research Laboratory*
- Collaborated on multi-million-dollar R&D projects, guiding model optimization for radiation detection and advising on technical strategy.
- Contributed to translating complex simulation results into executive briefings, securing $500K in incremental research funding.

**Commissioned Officer (National Oceanic & Atmospheric Administration) Newport** — *Unknown*
- Contributed to acquiring hydrographic data using remote sensing and automated systems for maritime safety operations.
- Built automated inventory tracking and GUI-based Sea Service letter generator tools to streamline administrative workflows.
- Supported Ancillary Data Manager operations by configuring Science Computing Systems and sensor connections to SCS servers.

**Research Physicist (Naval Research Laboratory) Washington** — *Unknown*
- Contributed to high-performance computing studies on radiation detection for border security applications using advanced simulations.
- Supported software development to predict active interrogation procedures and passive detection of WMD and radiological devices.
- Built graphic user interface for scenario-based radiation simulation software, enabling fast implementation on Linux systems.

**Faculty Assistant Researcher (University of Maryland College Park) College Park** — *Unknown*
- Built software for data analysis and publication-quality plots supporting particle physics experiments at the research facility.
- Contributed to scientific publications on Large Hadron Collider experiments, advancing knowledge in particle physics research.
- Assisted in developing simulations for radiation studies and efficiency quantifications using C++ and Python programming.

**Aerospace Ground Equipment Mechanic (United States Air Force) World Wide 05/2002 - 01/2010** — *Unknown*
- Supported equipment database administration and maintenance operations, ensuring accurate aircraft data availability for missions.
- Contributed to maintaining F-16 fighter jet maintenance records and presented readiness statistics to base leadership.
- Assisted in interpreting technical schematics and delivering precise information to support aircraft maintenance operations.

**Faculty Assistant Researcher** — *University of Maryland College Park*
- Contributed to simulation studies and constructed gradient-based solvers using Hadoop and LHC computing grid infrastructure.
- Built software using distributed computing to analyze particle accelerator data and generate publication-quality visualizations.
- Collaborated on publications detailing experiments conducted at the Large Hadron Collider, contributing to scientific discovery.

**Operations Manager of Marine Operations Center** — *Unknown*
- Supported in-house software and project management initiatives for shore-side support of NOAA fleet operations.
- Assisted in designing quantitative research tools, surveys, and data analysis frameworks for operational improvement.
- Built metric tracking tools and supported status briefings to fleet leadership for 13 oceanographic research vessels.
- Contributed to data-driven process improvements with cross-functional teams, presenting initiatives to regional directors.
- Supported database, security, infrastructure, and real-time dashboard creation for marine operations monitoring.
- Built Government Official Travel database and user interface, managing data ingestion architecture and collection workflows.

---

## Run notes (automated)

- Compare **Xiaomi** vs **Qwen** (and others) by reading the bullet sections above; rubric scores are heuristics only.
