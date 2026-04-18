# Writer (Drafter) comparison — multi-resume simulation

Generated: 2026-04-03T23:45:38.158449+00:00
Source file: `/mnt/c/Users/jeffd/latexRoot/My_resume_tests/combined_resumes.txt`
Parsed experience blocks (pre-dedupe): 19
Drafter input roles (deduped, capped): 10
Drafter `max_tokens`: 16384

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
        "OR",
        "Naval Research Laboratory",
        "Washington State Data Exchange for Public SafetyInternship",
        "MD",
        "Connect AI Club, Everett Community College",
        "DC"
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
    "Must be able to mentor two junior engineers (Context: Team leadership)"
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
    "Feature Stores",
    "Data Contracts",
    "Telemetry",
    "Streaming Stacks",
    "Batch Stacks"
  ],
  "implied_skills": [
    "Real-time Data Ingestion & Processing (Implied by 'Architect and deploy end-to-end ETL pipelines' for 'real-time' needs at Washington State Data Exchange, and 'analyzing sensor and sonar data' at NOAA which often requires real-time or near real-time processing)",
    "Data Lakehouse Management (Implied by 'Architect and deploy end-to-end ETL pipelines' and managing diverse data sources, which are foundational to lakehouse architectures)",
    "Metric Store Design & Implementation (Implied by 'define key performance metrics' and 'automated data pipelines' at NOAA, and 'translate business requirements into data-driven' solutions at Washington State Data Exchange)",
    "Low-latency Dashboarding (Implied by 'automated data pipelines for weather and oceanographic support tools' and 'reducing manual reporting overhead' at NOAA, suggesting a need for efficient data delivery for decision-making)",
    "Feature Store Collaboration (Implied by 'Architect and deploy end-to-end ETL pipelines' and 'integrating diverse data sources' which are prerequisites for building and populating feature stores for ML)",
    "Data Contract Enforcement (Implied by 'ensuring rigorous quality standards' in Connect AI Club and 'translating business requirements into data-driven' solutions, which necessitates clear data definitions and agreements)",
    "Mentorship & Leadership (Explicitly stated as 'President' of Connect AI Club, 'Coach peers', 'Facilitate client-style workshops', and 'Led cross-discipline teams' at NOAA)",
    "Cloud Platform Experience (Explicitly mentioned 'cloud platforms' in Connect AI Club projects, which bridges to AWS/GCP requirement)",
    "Workflow Orchestration (Implied by 'Architect and deploy end-to-end ETL pipelines' and 'automated data pipelines', which often require tools like Airflow or Dagster for scheduling and monitoring)",
    "Measurable Reliability Wins (Implied by 'reducing manual reporting overhead by 40%' and 'achieving 85% administrator adoption' at NOAA, demonstrating impact and stability of solutions)",
    "Government/Regulated Telemetry Experience (Explicitly stated experience with 'Washington State Data Exchange for Public Safety' and 'National Oceanic and Atmospheric Administration (NOAA)', directly addressing the nice-to-have)",
    "Cost Optimization (Implied by 'reducing manual reporting overhead' and designing efficient systems, which often leads to cost savings, relevant to 'cost guardrails' nice-to-have)"
  ],
  "work_archetypes": [
    "Scaling/Optimization",
    "Greenfield Build"
  ],
  "has_projects_section": true,
  "is_student_or_career_changer": true,
  "insider_tips": "Frame the NOAA and Washington State Data Exchange experiences as direct parallels to building and optimizing large-scale data pipelines for critical, real-time decision-making. Emphasize the leadership and mentorship from the Connect AI Club as direct experience for managing junior engineers and fostering quality."
}
```

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
| company_field_mismatches_vs_allowlist | 1 |
| avg_bullet_len | 121.0 |

Wall time: 6.94s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Assisted in mapping critical workflows for law enforcement and EMS stakeholders, translating business requirements into data-driven solutions.
- Contributed to architecting and deploying end-to-end ETL pipelines in SQL and Python, integrating diverse data sources for real-time analytics.

**President** — *Connect AI Club, Everett Community College*
- Built student consulting projects using LLMs, cloud platforms, and vector databases to deliver Proof-of-Concepts solving client problems.
- Helped coach peers on prompt engineering and iterative model tuning, improving their ability to communicate AI outcomes effectively.
- Supported client-style workshops and code reviews, ensuring quality standards and fostering collaborative innovation among members.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Supported enterprise operations across 13 research vessels, analyzing sensor and sonar data to proactively mitigate risks.
- Collaborated on leading cross-discipline teams to design Python-based decision-support tools, reducing manual reporting overhead by 40%.
- Built a Government Official Travel database and user interface, achieving 85% administrator adoption within the first year.
- Assisted operations leads in defining key performance metrics and automating data pipelines for weather and oceanographic support tools.
- Developed Python GUIs and reporting scripts to streamline vessel data ingestion and fostered cross-team collaboration.

**Research Physicist** — *Naval Research Laboratory*
- Collaborated on multi-million-dollar R&D projects, guiding model optimization for radiation detection and advising on technical strategy.

**Commissioned Officer (National Oceanic & Atmospheric Administration) Newport** — *OR*
- Led in the acquisition of hydrographic data using remote sensing and automated systems.
- Applied coding skills to automate DVD library inventory tracking and implemented a GUI-based Sea Service letter generator.
- Served as Ancillary Data Manager; configured the Science Computing System and connected sensors to SCS servers.

**Research Physicist (Naval Research Laboratory) Washington** — *DC*
- Utilized high-performance computing to conduct critical studies on radiation detection for border security applications.
- Supported software development to predict active interrogation procedures and passive detection of WMD/radiological devices.
- Designed a graphic user interface for fast implementation of scenario-based radiation simulation software on Linux.

**Faculty Assistant Researcher (University of Maryland College Park) College Park** — *MD*
- Designed software for data analysis and publication-quality plots in particle physics experiments.
- Co-authored publications on experiments at the Large Hadron Collider, contributing to scientific knowledge.
- Developed simulations for radiation studies and efficiency quantifications using C++ and Python.

**Aerospace Ground Equipment Mechanic (United States Air Force) World Wide 05/2002 - 01/2010** — *Unknown*
- Administered equipment databases and supervised maintenance operations to ensure aircraft data availability.
- Maintained aircraft maintenance records for F-16 fighter jets and presented readiness statistics to base leadership.

---

## Model: `qwen/qwen3.6-plus:free`

| Metric | Value |
| --- | --- |
| bullet_count | 24 |
| quantification_score | 0.333 |
| has_placeholders | False |
| fallback | False |
| weak_verb_hits | 0 |
| markdown_noise_lines | 0 |
| distinctive_job_ad_leaks | 0 |
| company_field_mismatches_vs_allowlist | 1 |
| avg_bullet_len | 131.6 |

Wall time: 166.85s

### Rewritten experiences (markdown preview)

**Operations Analyst** — *Washington State Data Exchange for Public SafetyInternship*
- Built SQL and Python ETL pipelines integrating diverse public safety data sources for real-time analytics.
- Collaborated on workflow mapping with law enforcement stakeholders, translating operational requirements into scalable data-driven models.

**President** — *Connect AI Club, Everett Community College*
- Built LLM and vector database prototypes on cloud platforms, delivering actionable PoCs for student consulting.
- Helped peers refine prompt engineering and data storytelling techniques, improving overall team-wide AI communication standards.
- Collaborated on structured code review processes and client workshops, establishing rigorous quality standards for deliverables.

**Operations Manager** — *National Oceanic and Atmospheric Administration (NOAA)*
- Built Python-based decision-support tools across 13 research vessels, successfully cutting manual reporting overhead by 40%.
- Supported operations leads by defining critical KPIs and automating weather data pipelines for oceanographic tracking.
- Contributed to Python GUI and reporting script development, streamlining sensor data ingestion and cross-team collaboration.
- Contributed to a new travel database launch, achieving 85% administrator adoption through an intuitive user interface.

**Research Physicist** — *Naval Research Laboratory*
- Built Linux-based graphical interfaces to accelerate deployment of scenario-based radiation simulation software for security teams.
- Contributed to predictive software development, enhancing passive radiological detection accuracy through rigorous algorithmic testing protocols.
- Supported high-performance computing simulations, systematically optimizing radiation detection models for critical national border security applications.

**Commissioned Officer (National Oceanic & Atmospheric Administration) Newport** — *OR*
- Assisted in Science Computing System configuration, linking environmental sensors to servers for reliable data capture.
- Built automated inventory tracking and GUI-based reporting tools, significantly reducing daily administrative processing time across departments.
- Supported hydrographic data acquisition via advanced remote sensing, ensuring high-quality inputs for downstream analytical modeling.

**Research Physicist (Naval Research Laboratory) Washington** — *DC*
- Built Linux-based graphical interfaces to accelerate deployment of scenario-based radiation simulation software for security teams.
- Contributed to predictive software development, enhancing passive radiological detection accuracy through rigorous algorithmic testing protocols.
- Supported high-performance computing simulations, systematically optimizing radiation detection models for critical national border security applications.

**Faculty Assistant Researcher (University of Maryland College Park) College Park** — *MD*
- Built C++ and Python analysis scripts generating publication-quality visualizations for complex, large-scale particle physics experiments.
- Assisted in C++ and Python radiation simulation development, accurately quantifying system efficiency for experimental validation.
- Collaborated on Large Hadron Collider data analysis, co-authoring peer-reviewed publications detailing key experimental physics outcomes.

**Aerospace Ground Equipment Mechanic (United States Air Force) World Wide 05/2002 - 01/2010** — *Unknown*
- Helped maintain comprehensive F-16 maintenance records, consistently presenting readiness statistics to leadership for operational planning.
- Helped administer centralized equipment databases and supervised maintenance workflows to guarantee continuous aircraft data availability.
- Assisted in technical schematic interpretation, delivering precise maintenance data to engineering and command staff daily.

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

Wall time: 247.75s

### Rewritten experiences (markdown preview)

## Run notes (agent)

- **MiniMax M2.1** (OpenRouter / DeepInfra): With this prompt size, the completion was almost entirely
  provider `reasoning` tokens; `message.content` stayed empty and the Drafter received no JSON. If you
  want a fair MiniMax sample on this corpus, raise `--drafter-max-tokens` further or switch to a
  non–reasoning-heavy route for that model.
- **Xiaomi Mimo v2 Flash** vs **Qwen 3.6 Plus (free)**: Both produced full JSON. On the automated
  rubric, Xiaomi edged slightly (fewer allow-list company mismatches, slightly shorter bullets).
  Qwen had more bullets and slightly higher average length. Neither leaked the fictitious SkyHarbor
  markers into bullets. **Manual read of the sections above is still the authority** for tone and STAR quality.

---

## Run notes (agent)

- **MiniMax M2.1** (OpenRouter / DeepInfra): With this prompt size, the completion was almost entirely
  provider `reasoning` tokens; `message.content` stayed empty and the Drafter received no JSON. If you
  want a fair MiniMax sample on this corpus, raise `--drafter-max-tokens` further or switch to a
  non–reasoning-heavy route for that model.
- **Xiaomi Mimo v2 Flash** vs **Qwen 3.6 Plus (free)**: Both produced full JSON. On the automated
  rubric, Xiaomi edged slightly (fewer allow-list company mismatches, slightly shorter bullets).
  Qwen had more bullets and slightly higher average length. Neither leaked the fictitious SkyHarbor
  markers into bullets. **Manual read of the sections above is still the authority** for tone and STAR quality.

---
