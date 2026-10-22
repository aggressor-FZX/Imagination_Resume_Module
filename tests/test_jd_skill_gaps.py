"""GAPS-01 regression: gap analysis must reflect the actual job ad's skills.

The prior implementation matched only a hardcoded data-governance vocabulary
(Collibra, ITAR, data lineage, ...), so a Machine Learning Engineer job ad
produced empty critical_gaps and a generic filler summary even though the
candidate was missing real, job-specific skills.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imaginator_new_integration import _compute_jd_skill_gaps


MLE_JOB_AD = """
Machine Learning Engineer
Responsibilities:
- Deploy machine learning models to production using Kubernetes.
- Build training pipelines with PyTorch; manage experiments.
- Write production Python services.
Requirements: PyTorch, Kubernetes, MLOps, model monitoring.
Nice to have: LangChain, vector databases.
"""

CANDIDATE_SKILLS = ["python", "sql", "data analysis", "aws glue", "power bi"]
CANDIDATE_RESUME = (
    "Data Analyst. Built ETL pipelines in Python and SQL on AWS Glue. "
    "Created Power BI dashboards. No mention of deep learning frameworks."
)


def test_mle_job_ad_produces_real_gaps():
    critical, nice = _compute_jd_skill_gaps(
        job_ad=MLE_JOB_AD,
        candidate_skills=CANDIDATE_SKILLS,
        resume_text=CANDIDATE_RESUME,
    )
    # The job ad's core requirements the candidate lacks must surface.
    joined = " ".join(critical + nice).lower()
    assert "pytorch" in joined, f"pytorch missing from gaps: {critical + nice}"
    assert "kubernetes" in joined, f"kubernetes missing from gaps: {critical + nice}"


def test_present_skills_are_not_gaps():
    critical, nice = _compute_jd_skill_gaps(
        job_ad=MLE_JOB_AD,
        candidate_skills=CANDIDATE_SKILLS + ["pytorch", "kubernetes", "docker", "airflow"],
        resume_text=CANDIDATE_RESUME + " Deployed models with PyTorch on Kubernetes.",
    )
    joined = " ".join(critical + nice).lower()
    assert "pytorch" not in joined
    assert "kubernetes" not in joined


def test_empty_job_ad_returns_no_gaps():
    assert _compute_jd_skill_gaps("", CANDIDATE_SKILLS, CANDIDATE_RESUME) == ([], [])


def test_governance_vocab_still_works():
    """The original data-governance terms must keep working (no regression)."""
    gov_ad = (
        "Data Governance Analyst: manage Collibra data catalog, data lineage, "
        "metadata, business glossary, MDM, and data quality KPIs."
    )
    critical, _ = _compute_jd_skill_gaps(
        job_ad=gov_ad,
        candidate_skills=["python"],
        resume_text="Python developer.",
    )
    joined = " ".join(critical).lower()
    assert "collibra" in joined
    assert "data lineage" in joined