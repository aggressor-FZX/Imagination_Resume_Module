from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from config import settings
import json
import os

router = APIRouter()


class ExperienceItem(BaseModel):
    title: Optional[str] = None
    skills: List[str] = Field(default_factory=list)


class SkillItem(BaseModel):
    name: str
    confidence: Optional[float] = None
    category: Optional[str] = None


class WorkExperience(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    duration_months: Optional[int] = None
    skills: List[SkillItem] = Field(default_factory=list)


class SimpleATSRequest(BaseModel):
    resume_text: Optional[str] = None
    job_ad: Optional[str] = None
    skills: Union[List[str], List[SkillItem]] = Field(default_factory=list)
    experiences: Union[List[ExperienceItem], List[WorkExperience]] = Field(default_factory=list)


class SimpleATSResponse(BaseModel):
    match_score: float
    matched_requirements: List[str]
    unmet_requirements: List[str]


def _auth_check(request: Request) -> None:
    token = (
        getattr(settings, "ATS_SERVICE_TOKEN", None)
        or getattr(settings, "JOB_SEARCH_AUTH_TOKEN", None)
        or os.environ.get("ATS_SERVICE_TOKEN")
        or os.environ.get("JOB_SEARCH_API_TOKEN")
    )
    if not token:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    provided = auth.split(" ", 1)[1].strip()
    if provided != token:
        raise HTTPException(status_code=403, detail="Invalid token")


def _extract_requirements(job_ad: Optional[str]) -> List[str]:
    if not job_ad:
        return []
    text = job_ad.lower()
    core = [
        "python", "react", "aws", "gcp", "kubernetes", "docker", "microservices",
        "leadership", "mentoring", "sql", "fastapi", "node", "typescript",
        "machine learning", "data science", "devops", "cloud", "api"
    ]
    found = []
    for r in core:
        if r in text:
            found.append(r)
    return found


def _normalize_skills(
    skills: Union[List[str], List[SkillItem]],
    experiences: Union[List[ExperienceItem], List[WorkExperience]],
) -> List[str]:
    z: List[str] = []
    for s in skills:
        if isinstance(s, str):
            z.append(s.lower())
        elif isinstance(s, SkillItem):
            z.append(s.name.lower())
    for exp in experiences:
        if isinstance(exp, ExperienceItem):
            for s in exp.skills:
                if isinstance(s, str):
                    z.append(s.lower())
        elif isinstance(exp, WorkExperience):
            for s in exp.skills:
                z.append(s.name.lower())
    return sorted(list(set(z)))


def _load_skill_adjacency() -> Dict[str, Dict[str, float]]:
    candidates = [
        os.path.join(os.getcwd(), "skill_adjacency.json"),
        os.path.join(
            os.path.dirname(
                os.path.dirname(__file__)
            ),
            "skill_adjacency.json",
        ),
        os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(__file__)
                    )
                )
            ),
            "skill_adjacency.json",
        ),
    ]
    for p in candidates:
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, dict) and "mappings" in data:
                        return data.get("mappings", {})
                    return data if isinstance(data, dict) else {}
        except Exception:
            pass
    return {
        "python": {"backend_development": 0.9, "api_development": 0.85, "data_processing": 0.8},
        "aws": {"cloud_computing": 0.9, "lambda": 0.8, "s3": 0.8},
        "docker": {"containerization": 0.95, "kubernetes": 0.75},
        "react": {"frontend_development": 0.9, "jsx": 0.9},
    }


def _expand_skills_with_adjacency(skills: List[str]) -> List[str]:
    adj = _load_skill_adjacency()
    expanded = set(skills)
    for s in skills:
        for k, v in adj.get(s, {}).items():
            if v >= 0.75:
                expanded.add(k)
    return sorted(list(expanded))


def _lexical_similar(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def compute_simple_match(req: SimpleATSRequest) -> SimpleATSResponse:
    reqs = _extract_requirements(req.job_ad)
    base_skills = _normalize_skills(req.skills, req.experiences)
    expanded_skills = set(_expand_skills_with_adjacency(base_skills))

    resume_text = (req.resume_text or "").lower()
    exp_texts: List[str] = []
    for exp in req.experiences:
        if isinstance(exp, WorkExperience):
            if exp.title:
                exp_texts.append(exp.title.lower())
            if exp.description:
                exp_texts.append(exp.description.lower())
        elif isinstance(exp, ExperienceItem):
            if exp.title:
                exp_texts.append(exp.title.lower())

    matched: List[str] = []
    unmet: List[str] = []
    for r in reqs:
        if r in expanded_skills:
            matched.append(r)
            continue
        lex = max((_lexical_similar(r, s) for s in expanded_skills), default=0.0)
        in_text = r in resume_text or any(r in t for t in exp_texts)
        if lex >= 0.5 or in_text:
            matched.append(r)
        else:
            unmet.append(r)

    denom = len(matched) + len(unmet)
    score = round((len(matched) / denom) if denom else 0.0, 2)
    return SimpleATSResponse(match_score=score, matched_requirements=matched, unmet_requirements=unmet)


@router.post("")
async def simple_match(request: Request, body: SimpleATSRequest) -> SimpleATSResponse:
    _auth_check(request)
    return compute_simple_match(body)
