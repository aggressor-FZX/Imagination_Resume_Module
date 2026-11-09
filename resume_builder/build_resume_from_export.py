#!/usr/bin/env python3
"""Build a resume PDF from an Imaginator export bundle and a LaTeX template.

The script accepts either a raw export-results text file or a JSON file that
contains the structured objects emitted by the Imaginator pipeline. It extracts
the most relevant data, maps it into the section structure used by
main_template.tex, writes a generated .tex file, and compiles it to PDF.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_TEMPLATE = Path(__file__).parent / "main_template.tex"
DEFAULT_OUTPUT_DIR = Path("/tmp/generated_resumes")


@dataclass
class ResumeSection:
    title: str
    body_lines: List[str] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""
    date_string: str = ""  # pre-formatted date range if provided directly


@dataclass
class ResumePayload:
    name: str
    location: str = ""
    phone: str = ""
    email: str = ""
    linkedin: str = ""
    github: str = ""
    headline: str = ""
    summary: str = ""
    skills: List[Tuple[str, List[str]]] = field(default_factory=list)
    experience: List[ResumeSection] = field(default_factory=list)
    projects: List[ResumeSection] = field(default_factory=list)
    education: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    publication: str = ""


LATEX_SPECIAL_CHARS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def sanitize_export_text(text: str) -> str:
    """Normalize LLM/smart punctuation so LaTeX/DOCX generators stay ASCII-safe.

    Also repairs Windows-1252 bytes that were mis-decoded as Latin-1 (they show up
    as C1 controls like U+0091 / U+0080 and crash pdflatex).
    """
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # EXPORT-01/DOCX-01: repair double-encoded UTF-8 ("mojibake") that was
    # decoded as cp1252 then stored as UTF-8. The stored run-3 text contains
    # e.g. "04/2025 â€“ Present" (U+00E2 U+0080 U+0093) where a plain en-dash
    # was intended. This MUST run before the C1-control remap below, which
    # would otherwise convert U+0080/U+0093 to €/" and break the sequence.
    mojibake_repair = {
        "\u00e2\u0080\u0093": "-",    # â€“  ← UTF-8 e2 80 93 (en dash)
        "\u00e2\u0080\u0094": "--",  # â€"  ← UTF-8 e2 80 94 (em dash)
        "\u00e2\u0080\u0099": "'",   # â€™  ← UTF-8 e2 80 99 (right quote)
        "\u00e2\u0080\u0098": "'",   # â€˜  ← UTF-8 e2 80 98 (left quote)
        "\u00e2\u0080\u009c": '"',   # â€œ  ← UTF-8 e2 80 9c (left dquote)
        "\u00e2\u0080\u009d": '"',   # â€<9d> ← UTF-8 e2 80 9d (right dquote)
        "\u00e2\u0080\u00a2": "-",    # â€¢  ← UTF-8 e2 80 a2 (bullet)
        "\u00e2\u0080\u00af": "-",   # â€¯  ← UTF-8 e2 80 af (narrow nbsp → hyphen)
        "\u00e2\u0080\u0091": "-",   # non-breaking hyphen (mojibake)
        "\u00c2\u00a0": " ",          # Â    ← UTF-8 c2 a0 (nbsp)
        "\u00c2\u00ad": "-",          # Â­   ← UTF-8 c2 ad (soft hyphen)
    }
    for original, replacement in mojibake_repair.items():
        text = text.replace(original, replacement)

    # Repair C1 controls that are actually Windows-1252 bytes mislabeled as Unicode.
    # Remap only those codepoints — do not latin-1-encode the whole string (drops U+2014 etc).
    if any(0x80 <= ord(ch) <= 0x9F for ch in text):
        repaired: List[str] = []
        for ch in text:
            code = ord(ch)
            if 0x80 <= code <= 0x9F:
                try:
                    repaired.append(bytes([code]).decode("cp1252", errors="ignore"))
                except Exception:
                    repaired.append("")
            else:
                repaired.append(ch)
        text = "".join(repaired)

    replacements = {
        "\u2018": "'",  # ‘
        "\u2019": "'",  # ’
        "\u201a": "'",  # ‚
        "\u201c": '"',  # “
        "\u201d": '"',  # ”
        "\u201e": '"',  # „
        "\u2013": "--",  # –
        "\u2014": "--",  # —
        "\u2212": "-",  # −
        "\u2022": "-",  # •
        "\u00a0": " ",  # nbsp
        "\u2026": "...",  # …
        "\u00b7": "-",  # ·
        "\u202f": " ",  # narrow nbsp
        # C1 controls that survive mis-decode (pdflatex: "Unicode character U+0091")
        "\u0080": "",
        "\u0085": "...",
        "\u0091": "'",
        "\u0092": "'",
        "\u0093": '"',
        "\u0094": '"',
        "\u0096": "--",
        "\u0097": "--",
        "\u0099": "",
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)

    # Drop remaining C0/C1 controls (keep \n \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Strip stray markdown section markers that merged onto content lines
    # ("... diverse scientific audiences. ##" — from "##\n\n## Projects").
    text = re.sub(r"\s*#{1,6}\s*$", "", text)
    return text


def latex_escape(text: str) -> str:
    return "".join(LATEX_SPECIAL_CHARS.get(char, char) for char in sanitize_export_text(text))


def normalize_dash(text: str) -> str:
    return sanitize_export_text(text).replace("–", "--").replace("—", "--")


def split_list_block(text: str) -> List[str]:
    items: List[str] = []
    for raw in re.split(r"\s*(?:\||\n|•|\*)\s*", text):
        item = raw.strip().strip("-").strip()
        if item:
            items.append(item)
    return items


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    marker = text.find('"exportedAt"')
    if marker == -1:
        marker = text.find("{\n  \"exportedAt\"")
    if marker == -1:
        marker = text.find("{\n\"exportedAt\"")
    if marker == -1:
        return None
    start = text.rfind("{", 0, marker)
    if start == -1:
        start = marker

    candidate = text[start:]
    for offset in range(len(candidate)):
        try:
            obj, end = decoder.raw_decode(candidate[offset:])
            if isinstance(obj, dict) and "data" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None


def get_nested(data: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def coerce_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def first_nonempty(mapping: Dict[str, Any], keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def format_entry_title(entry: Dict[str, Any], fallback: str = "") -> str:
    return first_nonempty(
        entry,
        ["title_line", "title", "role", "name", "project", "label"],
        fallback,
    )


def format_entry_body(entry: Dict[str, Any]) -> List[str]:
    bullets = []
    for key in ["bullets", "highlights", "items", "achievements", "responsibilities"]:
        value = entry.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    bullets.append(item.strip())
        elif isinstance(value, str) and value.strip():
            bullets.extend([line.strip() for line in re.split(r"[\n;]", value) if line.strip()])

    if not bullets:
        for key in ["snippet", "summary", "description", "body", "text"]:
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                bullets.extend([line.strip() for line in re.split(r"[\n;]", value) if line.strip()])
                break

    return bullets


def format_education_entry(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if not isinstance(entry, dict):
        return ""

    degree = first_nonempty(entry, ["degree", "title", "name"], "")
    field = first_nonempty(entry, ["field", "major"], "")
    institution = first_nonempty(entry, ["institution", "school", "university"], "")
    year = first_nonempty(entry, ["year", "date", "date_completed"], "")
    gpa = first_nonempty(entry, ["gpa"], "")

    parts = [part for part in [degree, field, institution, year] if part]
    line = ", ".join(parts)
    if gpa:
        line = f"{line} | GPA: {gpa}" if line else f"GPA: {gpa}"
    return line


def format_certification_entry(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if not isinstance(entry, dict):
        return ""
    name = first_nonempty(entry, ["certification", "name", "title"], "")
    issuer = first_nonempty(entry, ["issuer", "organization", "company"], "")
    date = first_nonempty(entry, ["date", "year", "issued"], "")
    parts = [part for part in [name, issuer, date] if part]
    return " ".join(parts)


def _strip_md_markers(text: str) -> str:
    """Remove markdown emphasis/bullet markers from a skill string.

    The frontend markdown fallback can leave '**' bold markers on skill
    lines ("** Python, Java, ** PyTorch, ..."); strip them so the rendered
    PDF/DOCX shows clean text.
    """
    return text.replace("**", "").replace("*", "").strip()


def parse_skill_groups(source: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    structured = source.get("skills")
    if isinstance(structured, list):
        if structured and all(isinstance(item, str) for item in structured):
            return [("Skills", [_strip_md_markers(item) for item in structured if _strip_md_markers(item)])]
        if structured and all(isinstance(item, dict) for item in structured):
            groups: List[Tuple[str, List[str]]] = []
            for entry in structured:
                label = first_nonempty(entry, ["category", "title", "name", "label"], "Skills")
                values: List[str] = []
                for key in ["items", "skills", "values"]:
                    raw = entry.get(key)
                    if isinstance(raw, list):
                        values.extend([_strip_md_markers(item) for item in raw if isinstance(item, str) and _strip_md_markers(item)])
                if not values:
                    skill = entry.get("skill")
                    values = [_strip_md_markers(skill)] if isinstance(skill, str) and _strip_md_markers(skill) else []
                groups.append((label, values))
            return groups

    flat = source.get("user_skills") or source.get("aggregate_skills") or source.get("skills_list")
    if isinstance(flat, list):
        items = [_strip_md_markers(item) for item in flat if isinstance(item, str) and _strip_md_markers(item)]
        if items:
            return [("Skills", items)]

    return []


def format_date_range(entry: Dict[str, Any]) -> str:
    """Extract a date range from an entry dict.

    Supports: duration (drafter output), date_string, start_date/end_date, dates, date, year.
    Returns a formatted string like '12/2019 -- 05/2023' or ''.

    The Imaginator Drafter is the source of truth for dates and writes them into the
    "duration" field (already formatted as "MM/YYYY -- MM/YYYY" or "MM/YYYY -- Present").
    The start_date/end_date fallback path only fires when "duration" is absent; it
    does NOT default a missing end_date to "Present" because we cannot know at this
    level whether a role is current. The Drafter already handles that rule (only
    index-0 gets "Present" when end_date is unknown).
    """
    # "duration" is the primary field the Imaginator drafter writes dates into.
    # Check it FIRST so drafter output takes full precedence.
    for key in ["duration", "date_string", "dates", "date_range"]:
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            # Normalise en-dash/em-dash variants to LaTeX double-hyphen
            normalised = val.strip().replace("\u2013", "--").replace("\u2014", "--")
            return normalised

    start = first_nonempty(entry, ["start_date", "start", "from", "date_start"], "")
    end = first_nonempty(entry, ["end_date", "end", "to", "date_end"], "")

    if start and end:
        return f"{start} -- {end}"
    if start:
        # Do NOT default to "Present" — only the Drafter knows which role is current.
        return start
    if end:
        return end

    # Single date/year fallback
    for key in ["date", "year"]:
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    return ""


def _build_section_title(entry: Dict[str, Any], fallback: str = "") -> str:
    """Build a clean 'Role -- Company -- Location' heading for a resume section.

    The drafter returns separate ``role`` and ``company`` fields, which are
    preferred. A raw ``title_line`` field (legacy LLM-parser format like
    "Software Engineer | Acme Corp | Seattle, WA | 2020 – Present") is used
    as a last resort — only the first pipe-segment is extracted to avoid
    embedding the full multi-part blob into a LaTeX bold line.

    Priority order:
      1. ``role`` + ``company`` (drafter output)  → "Role -- Company"
      2. ``title`` + ``company``                   → "Title -- Company"
      3. ``title`` or ``role`` alone               → just the role
      4. First ``|``-segment of ``title_line``      → avoid the full multi-part blob
      5. fallback

    Append an explicitly supplied entry ``location`` to the selected heading.
    Role locations never populate the top-level profile location.
    """
    role = (
        first_nonempty(entry, ["role", "title", "name"], "")
    )
    company = first_nonempty(entry, ["company", "organization", "employer"], "")

    if role and company:
        title = f"{role} -- {company}"
    elif role:
        title = role
    elif company:
        title = company
    else:
        # Last resort: keep only the first pipe-segment of title_line.
        title_line = entry.get("title_line", "")
        title = title_line.split("|")[0].strip() if isinstance(title_line, str) else ""
        title = title or fallback

    location = first_nonempty(entry, ["location"], "")
    return f"{title} -- {location}" if title and location else title or location


def parse_sections(source: Dict[str, Any], keys: Sequence[str]) -> List[ResumeSection]:
    """Parse experience/project entries with date support."""
    for key in keys:
        value = source.get(key)
        if isinstance(value, list) and value:
            sections: List[ResumeSection] = []
            for entry in value:
                if isinstance(entry, dict):
                    title = _build_section_title(entry)
                    body = format_entry_body(entry)
                    date_str = format_date_range(entry)
                    if not title and not body and not date_str:
                        continue
                    sections.append(ResumeSection(
                        title=title,
                        body_lines=body,
                        date_string=date_str,
                    ))
                elif isinstance(entry, str) and entry.strip():
                    sections.append(ResumeSection(
                        title="",
                        body_lines=[str(entry).strip()],
                    ))
            return sections
    return []


def extract_text_bundle(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    payload = extract_json_block(text)
    if payload:
        return payload

    raise ValueError(f"Could not locate structured JSON payload in {path}")


def build_fallback_payload(source: Dict[str, Any]) -> ResumePayload:
    """Compatibility wrapper: missing fields stay empty, never sample-filled."""
    return parse_resume_payload(source)


def parse_resume_payload(bundle: Dict[str, Any]) -> ResumePayload:
    """Parse explicitly supplied structured fields and aliases without inventing content.

    Raw resume/markdown text is not a substitute for structured career fields.
    Callers must parse that text before handing the canonical payload to rendering.
    """
    source = bundle.get("data", bundle)

    skills = parse_skill_groups(source)
    if not skills:
        # Preserve the existing scalar user_skills alias as explicit input.
        values = [skill for skill in coerce_list(source.get("user_skills")) if isinstance(skill, str) and skill.strip()]
        skills = [("Skills", values)] if values else []

    education_raw = coerce_list(source.get("education"))
    certifications_raw = coerce_list(source.get("certifications"))

    return ResumePayload(
        name=first_nonempty(source, ["name", "full_name"]),
        location=first_nonempty(source, ["location", "job_location", "preferred_location", "city"]),
        phone=first_nonempty(source, ["phone", "phone_number"]),
        email=first_nonempty(source, ["email", "email_address"]),
        linkedin=first_nonempty(source, ["linkedin", "linkedin_url"]),
        github=first_nonempty(source, ["github", "github_url"]),
        headline=first_nonempty(source, ["job_title", "target_job_title", "headline"]),
        summary=first_nonempty(source, ["summary", "professional_summary", "profile_summary"]),
        skills=skills,
        experience=parse_sections(source, ["experience", "professional_experience", "experiences"]),
        projects=parse_sections(source, ["projects"]),
        education=[format_education_entry(entry) for entry in education_raw if format_education_entry(entry)],
        certifications=[format_certification_entry(entry) for entry in certifications_raw if format_certification_entry(entry)],
        publication=first_nonempty(source, ["publication", "publications", "publication_line"]),
    )


def latex_list(items: Iterable[str]) -> str:
    return "\\begin{itemize}[leftmargin=0.18in, itemsep=0.2em]\n" + "\n".join(
        f"  \\item {latex_escape(normalize_dash(item))}" for item in items if item.strip()
    ) + "\n\\end{itemize}"


def split_name(full_name: str) -> Tuple[str, str]:
    cleaned = full_name.strip()
    if not cleaned:
        return "", ""
    parts = cleaned.split()
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def split_location(location: str) -> Tuple[str, str]:
    cleaned = location.strip()
    if not cleaned:
        return "", ""
    if "," in cleaned:
        city, state = cleaned.split(",", 1)
        return city.strip(), state.strip()
    return cleaned, ""


def replace_pattern(text: str, pattern: str, replacement: str) -> str:
    return re.sub(pattern, lambda _match: replacement, text, count=1, flags=re.MULTILINE)


def replace_between_markers(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = text.find(start_marker)
    if start == -1:
        raise ValueError(f"Start marker not found: {start_marker}")
    end = text.find(end_marker, start)
    if end == -1:
        raise ValueError(f"End marker not found: {end_marker}")
    return text[:start] + replacement + text[end:]


def build_summary_block(resume: ResumePayload) -> str:
    """Render the summary section, heading included.

    The heading is emitted only when there is summary text, matching
    ``ensure_required_headings``: an empty summary heading would claim a summary
    the candidate never wrote.

    Measured 2026-11-07: the delivered markdown carried
    ``## Professional Summary`` but the export dropped the label, so the summary
    text appeared as an unlabelled paragraph under the name. The text survived;
    the section did not.
    """
    parts = ["% TAGLINE / SUMMARY-------------------------------\n"]
    parts.append("\\vspace*{-9mm}\n")
    if resume.summary:
        parts.append("\\section{Professional Summary}\n")
    parts.append(
        "\\begin{center}\n"
        f"\\textit{{{latex_escape(resume.headline)}}}\n"
        "\\end{center}\n"
        "\\vspace*{1mm}\n\n"
    )
    if resume.summary:
        parts.append(f"{latex_escape(resume.summary)}\n")
    return "".join(parts)


def build_skill_block(resume: ResumePayload) -> str:
    lines: List[str] = []
    for label, values in resume.skills:
        if not values:
            continue
        lines.append(f"\\textbf{{{latex_escape(label)}:}} {latex_escape(', '.join(values))}")
    if not lines:
        return ""
    return (
        "% TECHNICAL SKILLS-------------------------------\n"
        "\\vspace*{1mm}\n"
        "\\section{Technical Skills}\n"
        "\\vspace*{1mm}\n"
        + "\n\n".join(lines)
        + "\n"
    )


def render_template(template_text: str, resume: ResumePayload) -> str:
    rendered = template_text
    first_name, last_name = split_name(resume.name)
    city, state = split_location(resume.location)
    rendered = replace_pattern(rendered, r"^\\name\{.*?\}\{.*?\}$", f"\\name{{{latex_escape(first_name)}}}{{{latex_escape(last_name)}}}")
    rendered = replace_pattern(rendered, r"^\\address\{.*?\}\{.*?\}$", f"\\address{{{latex_escape(city)}}}{{{latex_escape(state)}}}" if city or state else "")
    rendered = replace_pattern(rendered, r"^\\phone\{.*?\}$", f"\\phone{{{latex_escape(resume.phone)}}}" if resume.phone else "")
    rendered = replace_pattern(rendered, r"^\\email\{.*?\}$", f"\\email{{{latex_escape(resume.email)}}}" if resume.email else "")
    rendered = replace_pattern(rendered, r"^\\social\[linkedin\]\{.*?\}$", f"\\social[linkedin]{{{latex_escape(resume.linkedin.replace('https://', '').replace('http://', '').replace('www.', ''))}}}" if resume.linkedin else "")
    rendered = replace_pattern(rendered, r"^\\social\[github\]\{.*?\}$", f"\\social[github]{{{latex_escape(resume.github.replace('https://', '').replace('http://', '').replace('www.', ''))}}}" if resume.github else "")

    rendered = replace_between_markers(rendered, "% TAGLINE / SUMMARY-------------------------------", "% PROFESSIONAL EXPERIENCE-------------------------------", build_summary_block(resume))
    rendered = replace_between_markers(rendered, "% PROFESSIONAL EXPERIENCE-------------------------------", "% ML AND ANALYTICS PROJECTS-------------------------------", generate_experience_block(resume))
    # Slot order here MUST match the physical marker order in main_template.tex,
    # which follows ``source_preservation.REQUIRED_HEADINGS``:
    #   Summary -> Experience -> Projects -> Education -> Certifications -> Skills
    # Measured 2026-11-07: the template used to place Technical Skills before
    # Education/Certifications, so the PDF silently reordered the delivered
    # markdown while the DOCX kept it. The two exports disagreed about section
    # order for the same run.
    rendered = replace_between_markers(rendered, "% ML AND ANALYTICS PROJECTS-------------------------------", "% EDUCATION-------------------------------", generate_projects_block(resume))
    rendered = replace_between_markers(rendered, "% EDUCATION-------------------------------", "% CERTIFICATIONS-------------------------------", generate_education_block(resume))
    rendered = replace_between_markers(rendered, "% CERTIFICATIONS-------------------------------", "% TECHNICAL SKILLS-------------------------------", generate_certification_block(resume))
    rendered = replace_between_markers(rendered, "% TECHNICAL SKILLS-------------------------------", "% PUBLICATION-------------------------------", build_skill_block(resume))
    rendered = replace_between_markers(rendered, "% PUBLICATION-------------------------------", "\\end{document}", generate_publication_block(resume))
    return rendered


def bullet_block(lines: Sequence[str]) -> str:
    # Filter out empty lines
    filtered_lines = [line for line in lines if line and line.strip()]
    if not filtered_lines:
        return ""
    items = "\n".join(f"  \\item {latex_escape(normalize_dash(line))}" for line in filtered_lines)
    return f"\\begin{{itemize}}[label={{-}}, itemsep=-1.0ex]\n{items}\n\\end{{itemize}}"


def generate_section_body(section: ResumeSection) -> str:
    """Render a section entry with optional right-aligned date."""
    title = latex_escape(section.title)
    bullet_content = bullet_block(section.body_lines)
    if not title:
        date = f"{latex_escape(section.date_string)}\n" if section.date_string else ""
        return f"{date}{bullet_content}\n"
    if section.date_string:
        return f"\\textbf{{{title}}} \\hfill {latex_escape(section.date_string)}\n\\vspace*{{-2mm}}\n{bullet_content}\n"
    return f"\\textbf{{{title}}}\n\\vspace*{{-2mm}}\n{bullet_content}\n"


def generate_experience_block(resume: ResumePayload) -> str:
    if not resume.experience:
        return ""
    blocks = []
    blocks.append("% PROFESSIONAL EXPERIENCE-------------------------------\n\\vspace*{2mm}\n\\section{Professional Experience}\n\\vspace*{1mm}\n")
    for section in resume.experience:
        blocks.append(generate_section_body(section))
    return "".join(blocks)


def generate_projects_block(resume: ResumePayload) -> str:
    """Render the Projects section, heading included even when it is empty.

    ``REQUIRED_HEADINGS`` promises a Projects section and the delivered markdown
    keeps an empty one, so dropping the heading here made the two documents
    disagree about which sections exist. An empty section is the correct
    representation of "this section exists and the candidate has nothing for
    it"; nothing is invented to fill it. Measured 2026-11-07.
    """
    blocks = ["% PROJECTS-------------------------------\n\\vspace*{2mm}\n\\section{Projects}\n\\vspace*{1mm}\n"]
    for section in resume.projects:
        blocks.append(generate_section_body(section))
    return "".join(blocks)


def generate_education_block(resume: ResumePayload) -> str:
    if not resume.education:
        return ""
    lines = [latex_escape(line) for line in resume.education]
    body = "\\\\\n".join(lines)
    return f"% EDUCATION-------------------------------\n\\vspace*{{2mm}}\n\\section{{Education}}\n\\vspace*{{1mm}}\n{body}\n"


def generate_certification_block(resume: ResumePayload) -> str:
    if not resume.certifications:
        return ""
    lines = [latex_escape(line) for line in resume.certifications]
    body = "\\\\\n".join(lines)
    return f"% CERTIFICATIONS-------------------------------\n\\vspace*{{2mm}}\n\\section{{Certifications}}\n\\vspace*{{1mm}}\n{body}\n"


def generate_publication_block(resume: ResumePayload) -> str:
    if not resume.publication:
        return ""
    return (
        "% PUBLICATION-------------------------------\n"
        "\\vspace*{2mm}\n\\section{Publication}\n\\vspace*{1mm}\n"
        f"{latex_escape(resume.publication)}\n"
    )


def compile_latex(tex_path: Path, working_dir: Path) -> Path:
    """Compile a .tex file to PDF using pdflatex (from TeX Live).

    Runs pdflatex twice for proper cross-references and ToC resolution.
    Uses -interaction=nonstopmode to avoid hanging on errors.
    """
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-output-directory", str(working_dir),
        str(tex_path),
    ]
    def _decode_proc_output(raw: Optional[bytes]) -> str:
        # pdflatex often emits latin-1 / mixed bytes (e.g. 0xa2). Never use
        # subprocess text=True here — it raises UnicodeDecodeError mid-compile.
        return (raw or b"").decode("utf-8", errors="replace")

    def _log_tail() -> str:
        log_path = tex_path.with_suffix(".log")
        if not log_path.exists():
            return ""
        return log_path.read_text(encoding="utf-8", errors="replace")[-3000:]

    # First pass: generate .aux files
    result1 = subprocess.run(command, cwd=working_dir, capture_output=True)
    if result1.returncode != 0:
        error_output = _decode_proc_output(result1.stdout) + "\n" + _decode_proc_output(result1.stderr)
        log_tail = _log_tail()
        if log_tail:
            error_output += f"\n\n--- Log file tail ---\n{log_tail}"
        raise RuntimeError(f"pdflatex first pass failed (exit code {result1.returncode}):\n{error_output}")

    # Second pass: resolve references
    result2 = subprocess.run(command, cwd=working_dir, capture_output=True)
    if result2.returncode != 0:
        error_output = _decode_proc_output(result2.stdout) + "\n" + _decode_proc_output(result2.stderr)
        log_tail = _log_tail()
        if log_tail:
            error_output += f"\n\n--- Log file tail ---\n{log_tail}"
        raise RuntimeError(f"pdflatex second pass failed (exit code {result2.returncode}):\n{error_output}")

    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        log_tail = _log_tail()
        if log_tail:
            raise FileNotFoundError(f"pdflatex failed to produce PDF. Log tail:\n{log_tail}")
        raise FileNotFoundError(f"Expected PDF not produced: {pdf_path}")
    return pdf_path


def build_docx(resume: ResumePayload, output_dir: Path, safe_title: str) -> Path:
    """Generate a DOCX that visually matches the moderncv banking PDF style."""
    from docx import Document as DocxDocument
    from docx.shared import Pt, RGBColor as DocxRGB, Emu
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn as docx_qn
    
    doc = DocxDocument()
    
    # Page setup matching the PDF (Letter: 8.5x11 inches, with 0.3in margins)
    sec = doc.sections[0]
    # Letter dimensions in EMU: 8.5 x 914400 = 7772400, 11 x 914400 = 10058400
    sec.page_width = Emu(7772400)
    sec.page_height = Emu(10058400)
    # 0.3 inch margins = 0.3 x 914400 = 274320 EMU
    sec.left_margin = Emu(274320)
    sec.right_margin = Emu(274320)
    sec.top_margin = Emu(274320)
    sec.bottom_margin = Emu(274320)
    
    FONT = "Arial"
    COLOR_ACCENT = DocxRGB(0x1F, 0x4E, 0x79)  # dark blue, matches template
    COLOR_BODY = DocxRGB(0x1A, 0x1A, 0x1A)
    COLOR_MUTED = DocxRGB(0x44, 0x44, 0x44)
    
    def set_run(run, size=10, bold=False, italic=False, color=COLOR_BODY):
        # Keep text sanitized; Arial covers the remaining BMP range for DOCX.
        if run.text:
            run.text = sanitize_export_text(run.text)
        run.font.name = FONT
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
        run.font.color.rgb = color
        rPr = run._element.get_or_add_rPr()
        rFonts = rPr.find(docx_qn('w:rFonts'))
        if rFonts is None:
            rFonts = run._element.makeelement(docx_qn('w:rFonts'), {})
            rPr.insert(0, rFonts)
        rFonts.set(docx_qn('w:ascii'), FONT)
        rFonts.set(docx_qn('w:hAnsi'), FONT)
        rFonts.set(docx_qn('w:cs'), FONT)
        rFonts.set(docx_qn('w:eastAsia'), FONT)
    
    def add_para(sp_before=0, sp_after=2, alignment=None):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(sp_before)
        p.paragraph_format.space_after = Pt(sp_after)
        if alignment is not None:
            p.alignment = alignment
        r = p.add_run("")
        return p, r

    def add_section_heading(title):
        """Emit one section heading with the accent rule under it.

        Extracted so every section renders its heading identically, and so a
        section with no content can still emit its heading (see the Projects
        block below).
        """
        p, r = add_para(sp_before=8, sp_after=1)
        r.text = title
        set_run(r, size=12, bold=True, color=COLOR_ACCENT)
        pPr = p._element.get_or_add_pPr()
        pBdr = pPr.makeelement(docx_qn('w:pBdr'), {})
        bottom = pBdr.makeelement(docx_qn('w:bottom'), {
            docx_qn('w:val'): 'single',
            docx_qn('w:sz'): '6',
            docx_qn('w:space'): '1',
            docx_qn('w:color'): '1F4E79'
        })
        pBdr.append(bottom)
        pPr.append(pBdr)
        return p
    
    # ─── HEADER (centered, like moderncv banking) ───
    p, r = add_para(sp_before=0, sp_after=0, alignment=WD_ALIGN_PARAGRAPH.CENTER)
    r.text = resume.name
    set_run(r, size=16, bold=True, color=COLOR_ACCENT)
    
    # Location line (centered, under name)
    if resume.location:
        p, r = add_para(sp_before=0, sp_after=0, alignment=WD_ALIGN_PARAGRAPH.CENTER)
        r.text = resume.location
        set_run(r, size=9, color=COLOR_MUTED)
    
    # Contact line (centered)
    contacts = [value for value in (resume.phone, resume.email, resume.linkedin, resume.github) if value]
    if contacts:
        p, r = add_para(sp_before=2, sp_after=0, alignment=WD_ALIGN_PARAGRAPH.CENTER)
        r.text = "  |  ".join(contacts)
        set_run(r, size=8, color=COLOR_MUTED)

    if resume.headline:
        p, r = add_para(sp_before=4, sp_after=1, alignment=WD_ALIGN_PARAGRAPH.CENTER)
        r.text = resume.headline
        set_run(r, size=9, italic=True)
    
    # ─── SUMMARY ───
    # The heading is emitted only when there is summary text, matching
    # ``ensure_required_headings``: an empty summary heading would claim a
    # summary the candidate never wrote. Measured 2026-11-07: the markdown
    # carried ``## Professional Summary`` but the export dropped the label, so
    # the text appeared as an unlabelled paragraph under the name.
    if resume.summary:
        add_section_heading("Professional Summary")
        p, r = add_para(sp_before=2, sp_after=4)
        r.text = resume.summary
        set_run(r, size=8)
    
    # ─── PROFESSIONAL EXPERIENCE ───
    if resume.experience:
        add_section_heading("Professional Experience")
        
        for section in resume.experience:
            p, r = add_para(sp_before=4, sp_after=0)
            r.text = section.title
            set_run(r, size=9, bold=True)
            
            if section.date_string:
                run2 = p.add_run(f"    {section.date_string}")
                set_run(run2, size=9, italic=True, color=COLOR_MUTED)
            
            for bullet in section.body_lines:
                p, r = add_para(sp_before=0, sp_after=0)
                r.text = f"- {bullet}"
                set_run(r, size=9)
    
    # ─── PROJECTS ───
    # The heading is emitted even when the section is empty: REQUIRED_HEADINGS
    # promises it and the delivered markdown keeps it, so dropping it here made
    # the two documents disagree about which sections exist. Measured 2026-11-07.
    #
    # This asymmetry with Education/Certifications is DELIBERATE and is recorded
    # in ``section_contract.EMPTY_SECTION_APPEARANCE``. Making Projects
    # conditional to match them was tried on 2026-11-08 and reverted: it broke
    # the accepted guarantee that every promised heading reaches the exported
    # file, which is pinned by tests/test_export_section_headings.py. A section
    # set that disagrees between Markdown and the rendered document is the worse
    # defect, so the promised-heading rule wins and the asymmetry stands.
    add_section_heading("Projects")
    if resume.projects:
        for section in resume.projects:
            p, r = add_para(sp_before=3, sp_after=0)
            r.text = section.title
            set_run(r, size=9, bold=True)
            
            if section.date_string:
                run2 = p.add_run(f"    {section.date_string}")
                set_run(run2, size=9, italic=True, color=COLOR_MUTED)
            
            for bullet in section.body_lines:
                p, r = add_para(sp_before=0, sp_after=0)
                r.text = f"- {bullet}"
                set_run(r, size=9)
    
    # ─── EDUCATION ───
    if resume.education:
        add_section_heading("Education")
        
        for line in resume.education:
            p, r = add_para(sp_before=1, sp_after=1)
            r.text = line
            set_run(r, size=9)
    
    # ─── CERTIFICATIONS ───
    if resume.certifications:
        add_section_heading("Certifications")
        
        for line in resume.certifications:
            p, r = add_para(sp_before=1, sp_after=1)
            r.text = line
            set_run(r, size=9)
    
    # ─── TECHNICAL SKILLS (last section, after experience/projects/education/certs) ───
    if any(values for _, values in resume.skills):
        add_section_heading("Technical Skills")
        
        for label, values in resume.skills:
            if not values:
                continue
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            r1 = p.add_run(f"{label}: ")
            set_run(r1, size=9, bold=True)
            r2 = p.add_run(", ".join(values))
            set_run(r2, size=9, bold=False)

    if resume.publication:
        p, r = add_para(sp_before=8, sp_after=1)
        r.text = "Publication"
        set_run(r, size=12, bold=True, color=COLOR_ACCENT)
        p, r = add_para(sp_before=1, sp_after=1)
        r.text = resume.publication
        set_run(r, size=9)
    
    docx_path = output_dir / f"{safe_title}.docx"
    doc.save(str(docx_path))
    return docx_path


def build_resume_from_payload(payload: Dict[str, Any], output_dir: Optional[Path] = None, job_title: Optional[str] = None) -> Tuple[Path, Path]:
    """Build PDF + DOCX from a structured JSON payload (the Imaginator output schema).
    
    Args:
        payload: Dict with name, location, phone, email, linkedin, github,
                 headline, summary, skills, experience, projects, education, certifications
        output_dir: Where to write generated files (defaults to temp dir)
        job_title: Optional headline override
    
    Returns:
        (pdf_path, docx_path) tuple
    """
    template_path = Path(__file__).parent / "main_template.tex"
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    resume = parse_resume_payload({"data": payload} if "data" not in payload else payload)
    if job_title:
        resume.headline = job_title

    template_text = template_path.read_text(encoding="utf-8")
    rendered_tex = render_template(template_text, resume)

    safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", job_title or resume.headline or "resume")
    tex_path = output_dir / f"{safe_title}.tex"
    tex_path.write_text(rendered_tex, encoding="utf-8")

    pdf_path = compile_latex(tex_path, output_dir)
    docx_path = build_docx(resume, output_dir, safe_title)
    return pdf_path, docx_path


def build_resume_from_payload_base64(payload: Dict[str, Any], job_title: Optional[str] = None) -> Dict[str, str]:
    """Build PDF + DOCX and return as base64-encoded strings.
    
    Returns:
        Dict with 'pdf_base64', 'docx_base64', 'filename' keys
    """
    # A per-call private directory keeps sensitive files and compiler artifacts
    # isolated, including for concurrent requests with the same title.
    with tempfile.TemporaryDirectory(prefix="resume-export-") as directory:
        output_dir = Path(directory)
        pdf_path, docx_path = build_resume_from_payload(payload, output_dir, job_title)

        # The builder already chose a sanitized filename from the rendered title.
        # In particular, without an override this is the receipt-bound headline.
        safe_title = pdf_path.stem

        return {
            "pdf_base64": base64.b64encode(pdf_path.read_bytes()).decode("ascii"),
            "docx_base64": base64.b64encode(docx_path.read_bytes()).decode("ascii"),
            "filename": safe_title,
        }


def build_resume(input_path: Path, template_path: Path, output_dir: Path, job_title: Optional[str] = None) -> Tuple[Path, Path]:
    """Build both PDF and DOCX from the input. Returns (pdf_path, docx_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() in {".json"}:
        bundle = json.loads(input_path.read_text(encoding="utf-8"))
    else:
        bundle = extract_text_bundle(input_path)

    resume = parse_resume_payload(bundle)
    if job_title:
        resume.headline = job_title

    template_text = template_path.read_text(encoding="utf-8")
    rendered_tex = render_template(template_text, resume)

    safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", job_title or resume.headline or "resume")
    tex_path = output_dir / f"{safe_title}.tex"
    tex_path.write_text(rendered_tex, encoding="utf-8")

    pdf_path = compile_latex(tex_path, output_dir)
    docx_path = build_docx(resume, output_dir, safe_title)
    return pdf_path, docx_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a resume PDF from an Imaginator export bundle.")
    parser.add_argument("input", type=Path, help="Path to the export_results bundle text file or JSON file")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="LaTeX template to populate")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated .tex and .pdf files")
    parser.add_argument("--job-title", type=str, default=None, help="Optional headline/job title override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path, docx_path = build_resume(args.input, args.template, args.output_dir, args.job_title)
    print(f"Generated PDF: {pdf_path}")
    print(f"Generated DOCX: {docx_path}")


if __name__ == "__main__":
    main()