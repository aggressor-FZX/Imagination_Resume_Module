import re, json
resume='Senior Data Scientist with Python, AWS, SQL; built dashboards and ETL pipelines; no mention of Kubernetes or Spark.'
job='Senior Data Scientist - Required: Python, Kubernetes, AWS, Spark, distributed systems.'
aggregate_skills=['Python','AWS','SQL','Dashboards']
inferred_skills=['data-analysis']
processed_skills={'high_confidence':['python','aws'],'medium_confidence':[],'low_confidence':[]}

def _normalize(s): return re.sub(r'[^a-z0-9]','', (s or '').lower())

candidate_skills=set()
for s in aggregate_skills: candidate_skills.add(_normalize(s))
for s in inferred_skills: candidate_skills.add(_normalize(s))
for arr in (processed_skills.get('high_confidence',[]), processed_skills.get('medium_confidence',[]), processed_skills.get('low_confidence',[])):
  for s in arr: candidate_skills.add(_normalize(s))

synonyms_map = {
    'go': ['go', 'golang'],
    'ai': ['ai', 'machinelearning', 'machine-learning', 'ml'],
    'data': ['data', 'dataanalysis', 'data-analysis', 'data-analytics', 'analytics'],
    'analytics': ['analytics', 'dataanalytics', 'data-analytics'],
    'javascript': ['javascript', 'js'],
    'typescript': ['typescript', 'ts'],
    'postgres': ['postgres', 'postgresql'],
    'mysql': ['mysql'],
    'mongodb': ['mongodb', 'mongo'],
    'kubernetes': ['kubernetes', 'k8s'],
    'docker': ['docker'],
    'devops': ['devops'],
    'ml': ['ml', 'machinelearning', 'machine-learning'],
    'security': ['security', 'infosec', 'cybersecurity'],
    'leadership': ['leadership', 'management'],
}

required_keywords = [
    'python', 'java', 'c++', 'go', 'javascript', 'typescript',
    'react', 'node', 'sql', 'postgres', 'mysql', 'mongodb',
    'aws', 'gcp', 'azure', 'kubernetes', 'docker', 'devops',
    'ml', 'ai', 'data', 'analytics', 'security', 'leadership',
    'communication', 'testing', 'ci/cd'
]

job_text = job.lower()
job_required = [kw for kw in required_keywords if kw in job_text]

def _keyword_present(kw):
  variants = synonyms_map.get(kw, [kw])
  for v in variants:
    v_norm = _normalize(v)
    if any((v_norm in s) or (s in v_norm) for s in candidate_skills):
      return True
  return False

present_skills = [kw for kw in job_required if _keyword_present(kw)]
missing_skills = [kw for kw in job_required if kw not in present_skills]

required_count = len(job_required)
missing_count = len(missing_skills)
match_percent = int(round(100 * (required_count - missing_count) / required_count)) if required_count else 100

# Simulate fastsvm_output absent
fastsvm_output = {}

if missing_skills:
  gap_analysis = {
    'required_keywords_in_job': job_required,
    'found_keywords': present_skills,
    'missing_keywords': missing_skills,
    'missing_count': missing_count,
    'required_count': required_count,
    'match_percent': match_percent,
    'critical_gaps': missing_skills[:5],
    'nice_to_have_gaps': [],
    'gap_bridging_strategy': [f'Project/demo proving {skill} impact (1-2wks)' for skill in missing_skills[:3]],
    'summary': f"{match_percent}% match - missing {missing_count}/{required_count} req skills"
  }
else:
  skills_present = candidate_skills
  job_skills_set = job_required
  nice_to_have = [s for s in job_skills_set if _normalize(s) not in skills_present][:8]
  has_skills_section = bool(fastsvm_output and fastsvm_output.get('skills'))
  sections_missing = []
  if not has_skills_section:
    sections_missing.append('skills_section')
  bridging_suggestions = []
  if nice_to_have:
    bridging_suggestions.extend([f"Add a short project or resume bullet demonstrating {s} with a measurable outcome (e.g., percent improvement, scale, or time saved)." for s in nice_to_have[:3]])
  bridging_suggestions.append("Add a dedicated 'Skills' section that lists core tools and keywords used in the role.")
  bridging_suggestions.append("Quantify achievements across bullets (metrics, scope, and outcomes).")
  bridging_suggestions.append("Add 1–2 short 'Project Briefs' that show end-to-end impact for high-value skills.")
  gap_analysis = {
    'required_keywords_in_job': job_required,
    'found_keywords': present_skills,
    'missing_keywords': [],
    'missing_count': 0,
    'required_count': required_count,
    'match_percent': 100,
    'critical_gaps': [],
    'nice_to_have_gaps': nice_to_have,
    'gap_bridging_strategy': bridging_suggestions,
    'summary': f"No critical gaps detected in job ad keywords. Opportunity areas: {', '.join(nice_to_have[:3]) or 'none'}. See gap_bridging_strategy for concrete next steps.",
    'sections_missing': sections_missing,
    'implied_skills': dict((s, {'confidence':0.5, 'evidence':'inferred'}) for s in inferred_skills) if inferred_skills else {},
  }

print(json.dumps({'job_required':job_required,'present_skills':present_skills,'missing_skills':missing_skills,'gap_analysis':gap_analysis}, indent=2))
