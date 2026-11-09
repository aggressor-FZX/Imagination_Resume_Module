"""A WHOLE-SECTION presence check between the candidate's source and the résumé.

Read the name carefully: this is a *section* contract, not a completeness proof.

What it establishes — and only this:

  * a section the SOURCE populates is not left EMPTY in the approved markdown
    (Professional Experience, Projects, Education, Certifications, Technical
    Skills, and the conditional Professional Summary);
  * an empty section where the source had content is a loss, not a formatting
    choice.

What it does NOT establish, and must never be described as establishing:

  * **record-level completeness.** A section with ONE surviving record satisfies
    this check while another employer is silently dropped. That case is pinned by
    ``test_a_section_with_one_surviving_record_satisfies_the_contract``, which
    asserts the check PASSES while a record is lost. Record-level loss belongs to
    the per-item coverage check, not here.
  * that every qualification survived. It counts sections, not facts.
  * that the wording is faithful. A paraphrase, an omission inside a bullet and a
    re-attributed metric are all invisible to this module.

So a document passing this contract proves its five core sections carry content.
It does not prove the résumé is complete.

Other things it deliberately does not do:

  * It does not demand a section exist when the source has none. A candidate with
    no projects gets an empty Projects section, and that is correct.
  * It does not compare prose. Whether a bullet is a faithful paraphrase is a
    semantic question answered by the reviewer and the preservation rules.
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

# ── How an empty section appears: one rule, all three formats ────────────
#
# RESOLVED 2026-11-08. Measured before the decision, on a candidate with no
# experience, education, projects, certifications or skills:
#
#   Markdown  Professional Experience, Projects, Education, Certifications,
#             Technical Skills
#   PDF       Projects only
#   DOCX      Projects only
#
# Three artefacts of ONE run disagreed about which sections exist. The project
# requirement is that all five core sections remain present without inventing
# content, so the renderers were changed to match the Markdown: a section the
# candidate has nothing for is shown as a named, empty section, and nothing is
# fabricated to fill it.
#
# An earlier attempt went the other way (make ``Projects`` conditional to match
# Education/Certifications). That was reverted: it satisfied an asymmetry
# instead of removing it, and it broke the accepted guarantee that every
# promised heading reaches the exported file.
#
CORE_SECTIONS = (
    "Professional Experience",
    "Projects",
    "Education",
    "Certifications",
    "Technical Skills",
)

# "heading_without_body" in every format: the section is named, and empty.
# "Professional Summary" is the one conditional heading in all three, because an
# empty summary heading would claim a summary the candidate never wrote; that
# asymmetry is by product design, not an oversight.
EMPTY_SECTION_APPEARANCE = {
    "markdown": "heading_without_body",
    "pdf": "heading_without_body",
    "docx": "heading_without_body",
    "note": "All five CORE_SECTIONS are unconditional in every format. Summary "
            "is conditional everywhere: an empty summary heading claims a "
            "summary the candidate never wrote.",
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
