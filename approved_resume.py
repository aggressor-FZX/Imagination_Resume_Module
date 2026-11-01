"""Deterministic approved-Markdown -> render payload (no model or source aliases).

The approved document is the sole authority here. Extracted experiences and
generated `source_*` fields are not alternate authorities. Unknown career
headings are refused rather than guessed across employer boundaries.
"""
import re

from publication_gate import approved_segments


def plain(text):
    return re.sub(r"[*`_]", "", re.sub(r"^\s*[-•·▪▸]\s+", "", text)).strip()


def export_payload(markdown):
    segments = approved_segments(markdown)
    payload = {"headline": "", "job_title": "", "location": "", "summary": "",
               "skills": [], "experience": [], "projects": [], "education": [],
               "certifications": [], "publication": ""}
    labels = {
        "professionalexperience": "experience", "workexperience": "experience",
        "employmentexperience": "experience", "experience": "experience",
        "relevantexperience": "experience", "careerhistory": "experience",
        "projects": "projects", "education": "education", "academic": "education",
        "skills": "skills", "technicalskills": "skills", "certifications": "certifications",
        "summary": "summary", "professionalsummary": "summary",
        "publications": "publication",
    }
    section = None
    for segment in segments:
        heading = plain(segment["heading"])
        key = re.sub(r"[^a-z]", "", heading.lower())
        if not segment["is_block"]:
            section = labels.get(key)
            if section is None and segment["bullets"]:
                raise ValueError("unsupported approved section")
        elif section is None:
            # A document-level heading/name is not a qualification claim.
            # Do not infer a headline from a candidate name.
            if segment["bullets"]:
                raise ValueError("unscoped approved content")
            continue
        bullets = [plain(line) for line in segment["bullets"] if plain(line)]
        if section in {"experience", "projects"}:
            if not segment["is_block"]:
                if bullets:
                    raise ValueError("ambiguous approved career heading")
                continue
            # Preserve the entire approved heading (role/company/dates/location)
            # verbatim, rather than reconstructing its component relationships.
            payload[section].append({"title": heading, "bullets": bullets})
        elif section in {"education", "certifications", "skills"}:
            if segment["is_block"] and heading:
                payload[section].append(heading)
            payload[section].extend(bullets)
        elif section in {"summary", "publication"}:
            parts = ([heading] if segment["is_block"] else []) + bullets
            payload[section] = " ".join(filter(None, [payload[section], *parts]))
    if not payload["experience"]:
        raise ValueError("approved resume has no explicit experience blocks")
    return payload


def unsupported_entities(markdown, source):
    """Conservative explicit-heading check, not semantic verification.

    Named entities are the bold leading names under career/education sections.
    They must occur in the original input text, never generated source echoes.
    Role/meaning attribution still belongs to the supplied source, not this
    deterministic lexical check.
    """
    normalized = re.sub(r"[^a-z0-9]", "", (source or "").lower())

    def supported_education_pair(text):
        # Only a comma-separated degree/institution heading is eligible, and
        # both components must occur in one bounded source education section.
        parts = text.split(", ")
        if len(parts) != 2:
            return False
        degree, institution = (part.strip() for part in parts)
        if not re.search(r"\b(?:degree|bachelor|master|doctor|associate|a\.?a\.?s\.?|b\.?s\.?|m\.?s\.?)\b", degree, re.I):
            return False
        if not re.search(r"\b(?:college|university|institute|school)\b", institution, re.I):
            return False
        source_sections = re.split(r"\n\s*---\s*\n", source or "")
        for document in source_sections:
            match = re.search(
                r"(?im)^\s*(?:#{1,3}\s*)?(?:education|academic background)\s*:?\s*$"
                r"([\s\S]*?)(?=^\s*(?:#{1,3}\s*)?(?:professional experience|work experience|experience|skills|technical skills|projects|certifications)\s*:?\s*$|\Z)",
                document,
            )
            if match:
                body = re.sub(r"[^a-z0-9]", "", match.group(1).lower())
                if all(re.sub(r"[^a-z0-9]", "", part.lower()) in body for part in (degree, institution)):
                    return True
        return False
    current = None
    missing = []
    for raw in (markdown or "").splitlines():
        line = raw.strip()
        label = re.sub(r"[^a-z]", "", plain(line.lstrip("#")).lower())
        # A different section must end the previous entity-checking context.
        # For example, bold skill-category labels are not employers or schools.
        if re.match(r"^#{1,6}\s+", line):
            current = label if label in {
                "professionalexperience", "workexperience", "experience",
                "projects", "education", "certifications"
            } else None
            continue
        if label in {"professionalexperience", "workexperience", "experience",
                     "projects", "education", "certifications"}:
            current = label
            continue
        match = re.match(r"^(?:#{1,6}\s*)?\*\*(.+?)\*\*", line)
        if match and current:
            entity = re.sub(r"[^a-z0-9]", "", match.group(1).lower())
            if entity and entity not in normalized and not (
                current == "education" and supported_education_pair(match.group(1))
            ):
                missing.append(match.group(1))
    return missing