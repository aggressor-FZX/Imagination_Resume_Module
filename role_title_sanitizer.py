"""
Keep in sync with Hermes/role_title_sanitizer.py (same rules).

Used by imaginator_flow.parse_experiences so local parsing matches Hermes hygiene.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# US / territory abbreviations often mistaken for employers when a comma splits "Role, ST".
_US_STATE_OR_TERRITORY_ABBREV = frozenset(
    {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
        "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV",
        "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
        "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC", "PR", "VI", "GU", "AS", "MP",
    }
)

# Core tokens that indicate a real job / role title (not arbitrary noun phrases).
_JOB_TITLE_PATTERN = re.compile(
    r"\b("
    r"engineer|engineers|developer|developers|analyst|analysts|scientist|scientists|"
    r"architect|architects|manager|managers|director|directors|lead\b|principal|staff\b|"
    r"coordinator|coordinators|specialist|specialists|consultant|consultants|"
    r"officer|officers|physicist|physicists|researcher|researchers|assistant|assistants|"
    r"intern|interns|internship|technician|technicians|mechanic|mechanics|craftsman|"
    r"president|partner|partners|designer|designers|administrator|administrators|"
    r"programmer|programmers|associate|associates|faculty|hydrographer|hydrographers|"
    r"commissioned|trainer|trainers|coach|coaches|supervisor|supervisors|"
    r"executive|executives|strategist|strategists|contributor|contributors|"
    r"contractor|contractors|editor|editors|writer|writers|devops|sre\b|founder|founders|"
    r"head\s+of|vp\b|ceo|cfo|cto|ciso|cmo|chro|graduate|undergraduate|student|students|"
    r"scientist|pathologist|nurse|physician|therapist|attorney|paralegal|accountant|"
    # Trades and licensed occupations. Their absence meant a real trade title
    # ("Journeyman Electrician", "Electrical Apprentice") failed validation and was
    # destroyed by sanitize_experience_record, which then invented "Professional".
    r"electrician|electricians|journeyman|journeyperson|apprentice|apprentices|"
    r"lineman|linemen|welder|welders|machinist|machinists|plumber|plumbers|carpenter|"
    r"carpenters|millwright|millwrights|pipefitter|pipefitters|fitter|fitters|"
    r"operator|operators|installer|installers|inspector|inspectors|surveyor|surveyors|"
    r"estimator|estimators|foreman|foremen|superintendent|superintendents|"
    r"laborer|laborers|technologist|technologists|draftsman|drafter|drafters"
    r")\b",
    re.IGNORECASE,
)

# A single word is only accepted as a role when the whole string is one of these
# occupation nouns. Kept deliberately tight: membership/seniority words ("Associate",
# "Student", "Intern", "Professional") are usually an education level or a fragment,
# so they must not become a role title on their own.
_STANDALONE_OCCUPATIONS = frozenset({
    "electrician", "journeyman", "journeyperson", "apprentice", "lineman",
    "welder", "machinist", "plumber", "carpenter", "millwright", "pipefitter",
    "foreman", "superintendent", "inspector", "surveyor", "estimator",
    "engineer", "developer", "analyst", "technician", "mechanic", "designer",
    "architect", "manager", "supervisor", "coordinator", "specialist", "consultant",
    "accountant", "attorney", "nurse", "physician", "therapist", "paralegal",
})

# Phone/contact shapes, used to reject contact lines promoted to a role field.
_PHONE_PATTERN = re.compile(
    r"(\(\d{3}\)\s*\d{3}[-.\s]?\d{4})|(\b\d{3}[-.]\d{3}[-.]\d{4}\b)"
)


def is_valid_extracted_job_title(title: Optional[str]) -> bool:
    """
    Return False for fragments that must not be promoted to role/title fields.

    Rules:
    - Empty / too short
    - Sentence-ending period (trailing '.' or '. ' before lowercase continuation)
    - Fewer than 3 words unless a recognizable job-title token is present
    - No recognizable job-title token
    """
    if not title:
        return False
    t = title.strip()
    if len(t) < 4:
        return False

    # A role title never carries contact details. Without this, expanding the
    # occupation vocabulary below made a contact line such as
    # "Example City, WA | (202) 555-0101 | alex.electrician@example.com" validate,
    # because the local part of the e-mail address contains an occupation word.
    if "@" in t or _PHONE_PATTERN.search(t) or "http" in t.lower():
        return False

    if t.endswith("."):
        return False
    # Mid-line ". Word" looks like two sentences (e.g. "...research. Used Python")
    # Use capital-after-period to avoid "ph.d. in" style abbreviations at short lengths.
    if len(t) > 40 and re.search(r"\.\s+[A-Z][a-z]{2,}", t):
        return False

    words = t.split()
    if len(words) < 1:
        return False

    if not _JOB_TITLE_PATTERN.search(t):
        return False

    # Require at least 3 words unless we already matched a strong role token alone
    # (e.g. "Data Analyst", "Software Engineer" — 2 words, both substantive).
    if len(words) < 3:
        # Two-word titles: second word should often be a role class; pattern already matched.
        # Single-word "Engineer" still matches pattern but is weak — reject 1-word titles.
        if len(words) < 2:
            # A single word is accepted only when the ENTIRE string is a strong
            # occupation noun ("Electrician", "Millwright"). Membership words
            # ("Associate", "Student", "Intern") stay rejected because they are far
            # more often an education level or a fragment than a standalone role.
            return t.lower() in _STANDALONE_OCCUPATIONS

    return True


def is_plausible_employer_name(name: Optional[str]) -> bool:
    """Reject state codes, empty strings, and other token noise mistaken for companies."""
    if not name:
        return False
    s = name.strip()
    if not s or s.lower() in {"unknown", "n/a", "na", "none"}:
        return False
    us = s.upper()
    if len(us) == 2 and us in _US_STATE_OR_TERRITORY_ABBREV:
        return False
    if re.fullmatch(r"[A-Z]{2}", s):
        return False
    return True


def sanitize_experience_record(exp: Dict[str, Any]) -> Dict[str, Any]:
    """
    If title/position/role/title_line is invalid, move text into description, bullets, or body.
    Supports Imaginator parse_experiences keys (title_line, body, snippet, raw).
    """
    out = dict(exp)
    title = ""
    for k in ("title_line", "title", "position", "role"):
        v = (out.get(k) or "").strip()
        if v:
            title = v
            break
    if not title:
        return out
    if is_valid_extracted_job_title(title):
        if "title_line" in out:
            out["title_line"] = title
        return out

    bullets = out.get("bullets")
    body = (out.get("body") or out.get("snippet") or "").strip()

    if isinstance(bullets, list) and bullets:
        out["bullets"] = [title] + [str(b) for b in bullets]
    elif body or "title_line" in out:
        merged = f"{title} {body}".strip() if body else title
        out["body"] = merged
        out["snippet"] = merged
        prev_desc = (out.get("description") or "").strip()
        out["description"] = (
            f"{merged} {prev_desc}".strip() if prev_desc else merged
        )
        if out.get("raw"):
            out["raw"] = f"Professional\n{merged}"
    else:
        desc = (out.get("description") or "").strip()
        out["description"] = f"{title} {desc}".strip() if desc else title

    for k in ("title_line", "title", "position", "role"):
        if k in out:
            out[k] = ""

    has_any_title_key = any(
        k in out for k in ("title_line", "title", "position", "role")
    )
    if has_any_title_key and not (
        (out.get("title_line") or "").strip()
        or (out.get("title") or "").strip()
        or (out.get("role") or "").strip()
        or (out.get("position") or "").strip()
    ):
        # A rejected title must leave the field EMPTY. It must never be replaced with
        # invented text: this used to write the literal word "Professional", which the
        # writer and reviewer then treated as the candidate's real role. Live delivery
        # run 4 lost both of its job titles to exactly that substitution, and no later
        # stage can restore a fact that was overwritten with a fabrication.
        #
        # The original text is preserved above in description/snippet/body, so nothing
        # is lost; it simply is not promoted to a role field it failed to satisfy.
        pass

    return out


def sanitize_experience_list(experiences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sanitize a list of Hermes-style experience dicts."""
    if not experiences:
        return []
    return [sanitize_experience_record(dict(e)) for e in experiences]
