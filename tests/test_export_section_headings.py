"""Every promised section heading reaches the exported files.

Measured 2026-11-07 by extracting the ACTUAL downloaded PDF and DOCX from a live
website run, not by reading the approved markdown. The markdown carried all six
headings from ``source_preservation.REQUIRED_HEADINGS``; the files did not:

  * ``Professional Summary`` — the summary TEXT was rendered, but with no
    heading, so it appeared as an unlabelled paragraph under the candidate's
    name. ``build_summary_block`` wrote the text with no ``\\section{...}`` and
    the DOCX path wrote ``r.text = resume.summary`` with no heading paragraph.
  * ``Projects`` — dropped entirely when the list was empty, because both
    renderers guarded the heading with ``if resume.projects:``. The markdown
    keeps an empty Projects section, so the two documents disagreed about which
    sections exist.

Both are content-level defects that byte-level assertions cannot see, so these
tests render real files and read them back. The LaTeX half skips cleanly when no
toolchain is installed; it never turns into a pass on mocked bytes.
"""

import base64
import io
import shutil
import subprocess

from docx import Document
import pytest

from resume_builder.build_resume_from_export import build_resume_from_payload_base64

REQUIRED = (
    "Professional Summary",
    "Professional Experience",
    "Projects",
    "Education",
    "Certifications",
    "Technical Skills",
)

# A payload with a summary, one experience record, and NO projects — the exact
# shape that lost both headings on the live run.
PAYLOAD = {
    "name": "Synthetic Candidate",
    "headline": "Operations Analyst",
    "summary": ("Operations analyst with 9 years of experience in depot "
                "reporting and fuel cost control."),
    "skills": [{"category": "Analysis", "items": ["SQL", "reporting"]}],
    "experience": [
        {
            "title": "Fuel Reporting Analyst",
            "company": "Harborline Transit Authority",
            "duration": "04/2019 -- Present",
            "bullets": [
                "Tracked monthly fuel consumption across 14 depots.",
            ],
        }
    ],
    "education": ["Example Technical College -- 2015"],
    "certifications": ["Certified Production and Inventory Management (CPIM)"],
    "projects": [],
}


def _docx_text(encoded):
    document = Document(io.BytesIO(base64.b64decode(encoded)))
    return "\n".join(p.text for p in document.paragraphs)


def _pdf_text(encoded, tmp_path):
    pdf = tmp_path / "headings.pdf"
    pdf.write_bytes(base64.b64decode(encoded))
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout


def test_docx_carries_every_required_heading():
    """The DOCX must label the summary and keep an empty Projects section."""
    result = build_resume_from_payload_base64(dict(PAYLOAD))
    text = _docx_text(result["docx_base64"])
    for heading in REQUIRED:
        assert heading in text, f"DOCX dropped the {heading!r} heading"


def test_docx_summary_heading_precedes_the_summary_text():
    """The label must be attached to the text, not merely present somewhere."""
    result = build_resume_from_payload_base64(dict(PAYLOAD))
    document = Document(io.BytesIO(base64.b64decode(result["docx_base64"])))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    summary_index = paragraphs.index("Professional Summary")
    assert paragraphs[summary_index + 1].startswith("Operations analyst with 9")


def test_docx_keeps_an_empty_projects_heading():
    """An empty section is the correct representation of "nothing to show".

    Nothing is invented to fill it: the heading is followed by the next section
    heading, not by fabricated content.
    """
    result = build_resume_from_payload_base64(dict(PAYLOAD))
    document = Document(io.BytesIO(base64.b64decode(result["docx_base64"])))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    projects_index = paragraphs.index("Projects")
    assert paragraphs[projects_index + 1] == "Education"


def test_docx_omits_the_summary_heading_when_there_is_no_summary():
    """An empty summary heading would claim a summary the candidate never wrote.

    This mirrors ``ensure_required_headings``, which treats the summary as the
    one conditional heading.
    """
    payload = dict(PAYLOAD)
    payload["summary"] = ""
    result = build_resume_from_payload_base64(payload)
    text = _docx_text(result["docx_base64"])
    assert "Professional Summary" not in text
    # The other five are unconditional.
    for heading in REQUIRED[1:]:
        assert heading in text, f"DOCX dropped the {heading!r} heading"


@pytest.mark.skipif(
    not (shutil.which("pdflatex") and shutil.which("pdftotext")),
    reason="real renderer toolchain (pdflatex/pdftotext) not installed",
)
def test_pdf_carries_every_required_heading(tmp_path):
    """The PDF must agree with the DOCX about which sections exist."""
    result = build_resume_from_payload_base64(dict(PAYLOAD))
    text = _pdf_text(result["pdf_base64"], tmp_path)
    for heading in REQUIRED:
        assert heading in text, f"PDF dropped the {heading!r} heading"


@pytest.mark.skipif(
    not (shutil.which("pdflatex") and shutil.which("pdftotext")),
    reason="real renderer toolchain (pdflatex/pdftotext) not installed",
)
def test_pdf_and_docx_agree_on_the_section_set(tmp_path):
    """Neither file may carry a section the other omits.

    This is the property that failed: the markdown had six sections, the files
    had four, and nothing compared them.
    """
    result = build_resume_from_payload_base64(dict(PAYLOAD))
    docx_text = _docx_text(result["docx_base64"])
    pdf_text = _pdf_text(result["pdf_base64"], tmp_path)
    in_docx = {h for h in REQUIRED if h in docx_text}
    in_pdf = {h for h in REQUIRED if h in pdf_text}
    assert in_docx == in_pdf == set(REQUIRED)
