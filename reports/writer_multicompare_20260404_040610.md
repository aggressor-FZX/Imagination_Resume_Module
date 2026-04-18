# Writer (Drafter) comparison — multi-resume simulation

Generated: 2026-04-04T04:06:10.235034+00:00
Source file: `/mnt/c/Users/jeffd/latexRoot/My_resume_tests/combined_resumes.txt`
Parsed experience blocks (pre-dedupe): 17
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
        "University of Maryland College Park",
        "Washington State Data Exchange for Public SafetyInternship",
        "Naval Research Laboratory",
        "APIs databases and Smartsheet Project",
        "National Oceanic and Atmospheric Administration (NOAA)",
        "Connect AI Club, Everett Community College"
      ],
      "roles": [
        "Operations Analyst",
        "President",
        "Operations Manager",
        "Research Physicist",
        "Commissioned Officer (National Oceanic & Atmospheric Administration) Newport",
        "Research Physicist (Naval Research Laboratory) Washington",
        "Faculty Assistant Researcher (University of Maryland College Park) College Park",
        "Aerospace Ground Equipment Mechanic (United States Air Force) World Wide 05/2002 - 01/2010",
        "Director for shore-side support of 13 oceanographic research. Used Python",
        "Faculty Assistant Researcher"
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
    "Must be able to handle 4B+ events/day (Context: Real-time lakehouse ingestion)",
    "Must be able to deliver sub-40ms p99 dashboards (Context: Executive KPI reporting)",
    "Must have a track record of measurable reliability wins (Context: General requirement for Senior Analytics Engineer)"
  ],
  "domain_vocab": [
    "Python",
    "SQL",
    "Spark",
    "QuantMesh",
    "NebulaForge",
    "AWS",
    "GCP",
    "Airflow",
    "Dagster",
    "Lakehouse",
    "ETL",
    "Feature Stores",
    "Data Contracts",
    "Telemetry",
    "On-call rotation",
    "Cost guardrails",
    "LLMs",
    "Vector Databases"
  ],
  "implied_skills": [
    "Real-time Data Ingestion & Processing (Implied by 'Architect and deploy end-to-end ETL pipelines in SQL and Python, integrating diverse data sources to power real-time' and 'analyzing sensor and sonar data to proactively mitigate risks')",
    "Data Lakehouse Management (Implied by experience with ETL pipelines, diverse data sources, and real-time processing, which are foundational to lakehouse architectures)",
    "Metric Store Design & Implementation (Implied by 'define key performance metrics' and 'automated data pipelines for weather and oceanographic' combined with ETL experience)",
    "Low-latency Dashboarding (Implied by 'real-time' data processing and 'decision-support tools' for operations, which often require quick data visualization)",
    "Feature Store Collaboration (Implied by 'data-driven' solutions, 'automated data pipelines', and 'LLMs, cloud platforms, and vector databases' in projects, indicating familiarity with data preparation for ML)",
    "Data Contract Enforcement (Implied by 'rigorous quality standards' in projects and 'translate business requirements into data-driven' solutions, which necessitate data governance)",
    "Mentorship & Leadership (Explicitly stated 'Coach peers' and 'Facilitate client-style workshops' and 'Led cross-discipline teams')",
    "Cloud Platform Experience (Explicitly stated 'cloud platforms' in projects, aligning with AWS/GCP requirement)",
    "Workflow Orchestration (Implied by 'Architect and deploy end-to-end ETL pipelines' and 'automated data pipelines', which often utilize tools like Airflow/Dagster)",
    "Data Reliability & Quality Assurance (Implied by 'rigorous quality standards', 'proactively mitigate risks', and 'reducing manual reporting overhead')",
    "Cost Optimization for Data Stacks (Implied by 'reducing manual reporting overhead by 40%' and general operational efficiency focus, which often includes resource management)",
    "Government/Regulated Telemetry Experience (Explicitly stated 'Partner with law-enforcement and EMS stakeholders' and 'National Oceanic and Atmospheric Administration (NOAA)' and 'Naval Research Laboratory')"
  ],
  "work_archetypes": [
    "Scaling/Optimization",
    "Greenfield Build"
  ],
  "has_projects_section": true,
  "is_student_or_career_changer": true,
  "insider_tips": "Frame past operational and research roles as direct experience in building robust, high-performance data systems. Emphasize how 'reducing manual reporting overhead' and 'proactively mitigating risks' directly translate to 'measurable reliability wins' and 'cost guardrails' in a data engineering context."
}
```

## Model: `anthropic/claude-3.5-haiku`

| Metric | Value |
| --- | --- |
| bullet_count | 11 |
| quantification_score | 0.455 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 0 |
| avg_bullet_len | 148.3 |

Wall time: 16.94s

### Rewritten experiences (markdown preview)

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Collaborated with cross-discipline teams to design Python-based decision-support tools, reducing manual reporting overhead by 40% and enhancing operational efficiency
- Consulted with operations leads to define and automate key performance metrics for weather and oceanographic data pipelines, enabling real-time insights across 13 research vessels
- Launched enterprise travel database with user interface achieving 85% administrator adoption, streamlining data management and reporting processes

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Collaborated with law enforcement and EMS stakeholders to architect end-to-end ETL pipelines using SQL and Python, integrating diverse data sources for real-time operational insights
- Translated complex business requirements into data-driven workflow solutions, enabling more efficient cross-agency information exchange

**Director for shore-side support of 13 oceanographic research** — *APIs databases and Smartsheet Project*
- Developed data infrastructure using Python, ArcGIS, and CARIS to visualize ocean floor data and implement unsupervised learning techniques for advanced pattern detection
- Instituted metric tracking systems and led status briefings for fleet leadership, synthesizing complex research data into actionable insights
- Managed databases, security infrastructure, and project management software to support cross-functional data analysis initiatives

**President** — *Connect AI Club, Everett Community College*
- Spearheaded student consulting projects utilizing LLMs, cloud platforms, and vector databases to develop proof-of-concept solutions
- Coached peers on prompt engineering and iterative model tuning, demonstrating advanced data storytelling and technical communication skills
- Facilitated code reviews and client-style workshops to maintain rigorous quality standards in data and AI projects

---

## Model: `x-ai/grok-4.1-fast`

| Metric | Value |
| --- | --- |
| bullet_count | 33 |
| quantification_score | 0.364 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 3 |
| avg_bullet_len | 97.7 |

Wall time: 40.12s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Collaborated on mapping critical workflows with law-enforcement and EMS stakeholders to translate business requirements into data-driven solutions.
- Built end-to-end ETL pipelines using SQL and Python, integrating diverse data sources to power real-time analytics.

**President** — *Connect AI Club, Everett Community College*
- Built student consulting projects using LLMs, cloud platforms, and vector databases to deliver PoCs solving real-world problems.
- Helped peers with prompt engineering, iterative model tuning, and data storytelling to communicate AI outcomes effectively.
- Supported workshops and code reviews to enforce quality standards and foster collaborative innovation.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Supported operations across 13 research vessels, analyzing sensor and sonar data to mitigate risks proactively.
- Collaborated on Python-based decision-support tools, reducing manual reporting overhead by 40%.
- Built Government Official Travel database and UI, achieving 85% administrator adoption.
- Contributed to key performance metrics and automated data pipelines for weather and oceanographic data.
- Built Python GUIs and reporting scripts to streamline vessel data ingestion and cross-team collaboration.

**Research Physicist** — *Naval Research Laboratory*
- Collaborated on multi-million-dollar R&D projects, optimizing models for radiation detection simulations.
- Translated simulation results into executive briefings, securing $500K in incremental funding.
- Supported high-performance computing studies on radiation detection for border security applications.
- Contributed to software development for predicting WMD detection procedures.
- Built GUI for scenario-based radiation simulation software running on Linux.

**Commissioned Officer (National Oceanic & Atmospheric Administration) Newport** — *Unknown*
- Supported hydrographic data acquisition using remote sensing and automated systems.
- Built automations for DVD library inventory tracking and GUI-based Sea Service letter generator.
- Configured Science Computing System and sensor connections to SCS servers as Ancillary Data Manager.

**Faculty Assistant Researcher (University of Maryland College Park) College Park** — *Unknown*
- Built software for data analysis and publication-quality plots in particle physics experiments.
- Contributed to co-authored publications on Large Hadron Collider experiments.
- Developed C++ and Python simulations for radiation studies and efficiency quantifications.
- Supported distributed computing analysis of particle accelerator data using Hadoop and LHC grid.

**Aerospace Ground Equipment Mechanic (United States Air Force) World Wide 05/2002 - 01/2010** — *Unknown*
- Supported equipment databases and maintenance operations to ensure aircraft data availability.
- Maintained F-16 maintenance records and presented readiness statistics to base leadership.

**Director for shore-side support of 13 oceanographic research. Used Python** — *APIs databases and Smartsheet Project*
- Contributed to metric tracking systems and status briefings for 13 research vessels.
- Supported data-driven process improvements and executive presentations with cross-functional teams.
- Managed databases, security, infrastructure, Smartsheet dashboards, and real-time monitoring.
- Built Government Official Travel database and ingestion architecture for data collection.
- Automated repetitive tasks to resolve detected workflow inefficiencies.
- Built ocean floor visualizations using Python, ArcGIS, and CARIS; researched unsupervised learning detection techniques.

**Faculty Assistant Researcher** — *University of Maryland College Park*
- Supported simulation studies and gradient-based solvers using Hadoop and LHC computing grid.
- Built software for distributed computing analysis of particle accelerator data and publication-quality plots.
- Contributed to co-authored publications on Large Hadron Collider experiments.

---

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
| avg_bullet_len | 121.7 |

Wall time: 11.18s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Assisted in mapping critical workflows with law enforcement and EMS stakeholders, translating business requirements into data-driven solutions.
- Contributed to building end-to-end ETL pipelines in SQL and Python, integrating diverse data sources to power real-time analytics.

**President** — *Connect AI Club, Everett Community College*
- Built student consulting projects using LLMs, cloud platforms, and vector databases to deliver proof-of-concepts that solve client problems.
- Collaborated on coaching peers in prompt engineering and model tuning, effectively communicating AI outcomes to non-technical audiences.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Supported enterprise operations across 13 research vessels by analyzing sensor and sonar data to proactively mitigate risks.
- Helped design Python-based decision-support tools, reducing manual reporting overhead by 40% and streamlining data ingestion.
- Launched a Government Official Travel database and user interface, achieving 85% administrator adoption within the first year.

**Research Physicist** — *Naval Research Laboratory*
- Collaborated on multi-million-dollar R&D projects, guiding model optimization for radiation detection and advising on technical strategies.
- Translated complex simulation results into executive briefings, securing $500K in incremental funding and strengthening project support.

**Commissioned Officer (National Oceanic & Atmospheric Administration) Newport** — *Unknown*
- Supported the acquisition of hydrographic data using remote sensing and automated systems.
- Helped automate DVD library inventory tracking and implemented a GUI-based Sea Service letter generator.

**Research Physicist (Naval Research Laboratory) Washington** — *Unknown*
- Utilized high-performance computing to conduct critical studies on radiation detection for border security applications.
- Supported software development to predict active interrogation procedures and passive detection of WMD/radiological devices.

**Faculty Assistant Researcher (University of Maryland College Park) College Park** — *Unknown*
- Built software for data analysis and publication-quality plots in particle physics experiments.
- Contributed to co-authoring publications on experiments conducted at the Large Hadron Collider.

**Aerospace Ground Equipment Mechanic (United States Air Force) World Wide 05/2002 - 01/2010** — *Unknown*
- Supported equipment database administration and supervised maintenance operations to ensure aircraft data availability.
- Maintained aircraft maintenance records for F-16 fighter jets and presented readiness statistics to base leadership.

**Director for shore-side support of 13 oceanographic research. Used Python** — *APIs databases and Smartsheet Project*
- Helped institute metric tracking systems and lead status briefings for fleet leadership, synthesizing data for decision-making.
- Collaborated on data-driven process improvements with cross-functional teams and prepared presentations for senior executives.

**Faculty Assistant Researcher** — *University of Maryland College Park*
- Assisted in performing simulation studies and constructing gradient-based solvers using Hadoop and LHC computing grid.
- Built software and used distributed computing to analyze particle accelerator data, generating publication-quality results.

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
| bullet_count | 21 |
| quantification_score | 0.429 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 4 |
| avg_bullet_len | 132.7 |

Wall time: 132.01s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Built end-to-end ETL pipelines using SQL and Python, integrating diverse data sources to enable real-time operational reporting.
- Collaborated on workflow mapping with law enforcement teams, translating business requirements into structured data-driven analytics solutions.

**President** — *Connect AI Club, Everett Community College*
- Contributed to student consulting projects by deploying cloud platforms and vector databases to deliver functional AI solutions.
- Supported peer development in prompt engineering and iterative tuning, improving team ability to communicate complex AI outcomes.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Built Python-based decision-support tools for cross-discipline teams, reducing manual reporting overhead by 40% across operations.
- Assisted in defining key performance metrics and automating weather data pipelines, streamlining daily operational tracking processes.
- Contributed to a new travel database and UI launch, achieving 85% administrator adoption within the quarter.

**Research Physicist** — *Naval Research Laboratory*
- Collaborated on multi-million-dollar R&D initiatives, optimizing radiation detection models and advising technical stakeholders on deployment.
- Supported executive decision-making by translating simulation data into clear briefings, securing $500K in additional project funding.

**Commissioned Officer (National Oceanic & Atmospheric Administration) Newport** — *Unknown*
- Built automated tracking scripts for inventory and documentation, eliminating manual data entry processes across administrative workflows.
- Assisted in configuring the Science Computing System and connecting sensor arrays, ensuring reliable hydrographic data acquisition.

**Research Physicist (Naval Research Laboratory) Washington** — *Unknown*
- Supported software development for radiation detection models, enhancing predictive accuracy for critical border security applications.
- Built a Linux-based graphical interface for simulation software, accelerating scenario testing and deployment workflows for researchers.

**Faculty Assistant Researcher (University of Maryland College Park) College Park** — *Unknown*
- Contributed to particle physics data analysis by developing C++ and Python simulations for radiation efficiency studies.
- Built analytical software for Large Hadron Collider experiments, generating publication-quality visualizations for peer-reviewed research.

**Aerospace Ground Equipment Mechanic (United States Air Force) World Wide 05/2002 - 01/2010** — *Unknown*
- Assisted in administering maintenance databases for aircraft fleets, ensuring accurate readiness reporting for base leadership.
- Supported technical operations by interpreting complex schematics and maintaining precise equipment tracking records across deployments.

**Director for shore-side support of 13 oceanographic research. Used Python** — *APIs databases and Smartsheet Project*
- Built real-time dashboards and managed database infrastructure, improving cross-functional team visibility into operational project metrics.
- Contributed to workflow automation using Python, reducing repetitive manual tasks and streamlining executive reporting processes.

**Faculty Assistant Researcher** — *University of Maryland College Park*
- Supported distributed computing workflows using Hadoop, enabling large-scale particle accelerator data analysis and simulation modeling.
- Built gradient-based solvers for high-energy physics datasets, accelerating computational modeling and subsequent research publication timelines.

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

Wall time: 511.31s

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
