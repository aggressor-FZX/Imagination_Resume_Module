"""A completeness contract between the candidate's source and the approved résumé.

The system can already produce complete, correctly ordered résumés, and it has
meaningful safety checks. It cannot yet claim that every submission reliably
carries every source-backed section, and this module is the honest boundary
between those two statements.

What this checks, decided mechanically because each part is decidable:

  * every section the SOURCE provides is still non-empty in the approved
    markdown — Education, Certifications, Skills, Projects, Experience;
  * a section that is present but EMPTY in the output while the source provides
    content for it is a loss, not a formatting choice.

What this deliberately does NOT do:

  * It does not demand that a section exist when the source has none. A
    candidate with no projects gets an empty Projects section, and that is
    correct, not a failure.
  * It does not compare prose. Whether a bullet is a faithful paraphrase is a
    semantic question answered by the reviewer and the preservation rules; this
    contract only asks whether the section's content is still there at all.
  * It does not repair anything. Repair happens in ``source_preservation``; this
    module answers one question — should this document be published?

Why a contract rather than more filling rules: the existing restorers cover
summary, education, skills and certifications. Experience and Projects have no
restorer, and inventing one would compose prose from employer records, which is
exactly what the preservation rules refuse to do. The honest handling for a
dropped Experience section is to REFUSE PUBLICATION, not to synthesize a
replacement.

Measured 2026-11-08 on ``realistic_candidate``: a writer draft that omits
Professional Experience receives an empty heading from
``ensure_required_headings`` and the employer records vanish, with nothing
deterministic stopping publication.
"""
from __future__ import annotations

from typing import Any

import source_preservation as sp

# Sections this contract holds to the source. "Professional Summary" is
# conditional by product design and is checked only for presence-when-sourced.
CONTRACTED = (
    "Professional Summary",
    "Professional Experience",
    "Projects",
    "Education",
    "Certifications",
    "Technical Skills",
)

# ── The appearance decision, stated once ─────────────────────────────────
#
# How a genuinely empty section should appear, decided here so it is a rule
# rather than an accident of which renderer ran:
#
#   MARKDOWN — the heading is emitted with no body. ``ensure_required_headings``
#              promises six named sections and the delivered Markdown carries
#              them, so a consumer parsing the Markdown sees a section it can
#              recognise even when the candidate has nothing for it.
#
#   PDF/DOCX — an empty section is OMITTED, heading and all. Measured
#              2026-11-08 on a sparse payload: the DOCX carried Summary,
#              Experience, Projects and Technical Skills and simply had no
#              Education or Certifications headings, because
#              ``generate_education_block`` and the DOCX equivalent return early
#              on an empty list.
#
# Those two are deliberately different, and the difference is the point: the
# Markdown is the interchange form (structure is information there), while the
# exported document is what a human reads, and an empty heading with a rule under
# it reads as a mistake or a gap in the candidate's history. Omitting it is the
# honest appearance.
#
# KNOWN ASYMMETRY, deliberately left in place.
#
# ``Projects`` is emitted unconditionally by the DOCX while Education and
# Certifications are conditional, so a candidate with no projects gets an empty
# Projects heading in the exported file and a candidate with no education does
# not. Making Projects conditional to match was tried on 2026-11-08 and REVERTED:
# it broke the accepted guarantee that every promised heading reaches the exported
# file, pinned by ``tests/test_export_section_headings.py``.
#
# The trade was decided this way on purpose. A section set that disagrees between
# the Markdown and the rendered document is the worse defect — it means the two
# artefacts of one run claim different things about which sections exist — so the
# promised-heading rule wins and this asymmetry stands until both can be satisfied
# together.
#
# IMPORTANT: this constant is documentation of measured behaviour, not a switch.
EMPTY_SECTION_APPEARANCE = {
    "markdown": "heading_without_body",
    "pdf": "heading_without_body",
    "docx": "heading_without_body",
    "note": "Projects is unconditional in DOCX; Education/Certifications are "
            "conditional. Reverted on 2026-11-08 - see above.",
}

_ALIASES = {
    "summary": "Professional Summary",
    "profile": "Professional Summary",
    "objective": "Professional Summary",
    "professional summary": "Professional Summary",
    "experience": "Professional Experience",
    "work experience": "Professional Experience",
    "employment": "Professional Experience",
    "employment history": "Professional Experience",
    "professional experience": "Professional Experience",
    "projects": "Projects",
    "education": "Education",
    "academic background": "Education",
    "certifications": "Certifications",
    "certificates": "Certifications",
    "licenses": "Certifications",
    "certifications and licenses": "Certifications",
    "skills": "Technical Skills",
    "technical skills": "Technical Skills",
    "core competencies": "Technical Skills",
}


def canonical_sections(markdown: str) -> dict[str, list[str]]:
    """Map canonical section name -> its non-empty body lines.

    Uses the same splitter as preservation, so a bare extracted heading and a
    Markdown ``##`` heading are recognised identically. That matters because the
    source documents and the delivered markdown use different conventions, and a
    contract that disagreed with the splitter would report phantom losses.
    """
    out: dict[str, list[str]] = {}
    for section in sp._split_sections(markdown or ""):
        heading = (section.get("heading") or "").strip().lower()
        if not heading:
            continue
        canonical = _ALIASES.get(heading)
        if canonical is None:
            continue
        body = [line.strip() for line in section.get("lines") or []
                if line.strip()]
        # A repeated heading must not erase the first one's body.
        out.setdefault(canonical, []).extend(body)
    return out


def source_backed_sections(source_text: str) -> dict[str, int]:
    """Canonical sections the candidate's own documents actually populate."""
    sections = canonical_sections(source_text)
    return {name: len(body) for name, body in sections.items() if body}


def shortfall(markdown: str, source_text: str) -> list[dict[str, Any]]:
    """Sections the source populates that the output left empty.

    Returns one entry per lost section: ``{"section", "source_lines"}``. An empty
    list means the contract holds.

    Only sections the SOURCE provides can be lost. A source with no Projects
    section produces no entry, so an empty Projects heading in the output is not
    reported — that is the correct appearance for "nothing to show".
    """
    delivered = canonical_sections(markdown)
    backed = source_backed_sections(source_text)
    losses: list[dict[str, Any]] = []
    for name in CONTRACTED:
        if not backed.get(name):
            continue
        if delivered.get(name):
            continue
        losses.append({"section": name, "source_lines": backed[name]})
    return losses
