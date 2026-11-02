"""Real renderer content checks against the answer key.

**[renderer] This is a real-renderer test: it invokes the actual LaTeX compiler and the
actual DOCX writer, then extracts text from both generated files and asserts the content.
It does not mock PDF or DOCX bytes.**

Why this exists: the export path had content-level defects that byte-level assertions
cannot see (mojibake reaching the page, a fabricated title rendered as if real). This test
takes a document built from the answer key's facts, renders it, and reads it back.

Skips cleanly when the local machine has no LaTeX toolchain.
"""
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pytest

from tests.fixtures.fidelity import expected_facts as facts

pdflatex = pytest.mark.skipif(
    shutil.which("pdflatex") is None, reason="no LaTeX toolchain on this machine")
pytestmark = pdflatex


def _payload():
    """A resume payload built ONLY from facts the source supports."""
    cascade, harborline = facts.EMPLOYMENT_RECORDS
    return {
        "name": "Synthetic Candidate",
        "location": cascade["location"],
        "phone": "",
        "email": "",
        "linkedin": "",
        "github": "",
        "headline": "Journeyman Electrician",
        "summary": ("Journeyman electrician with 8+ years of commercial and industrial "
                    "electrical experience across construction, facilities maintenance "
                    "and power distribution."),
        "skills": ["Commercial electrical systems", "Conduit bending", "Panel labeling",
                   "NFPA 70E", "Lockout/tagout", "Inspection checklists"],
        "experience": [
            {
                "title": f"{cascade['role']} -- {cascade['employer']}",
                "duration": f"{cascade['start']} -- {cascade['end']}",
                "bullets": [
                    "Installed and maintained commercial electrical systems across 40+ job sites.",
                    "Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices.",
                    "Cut rework by 18% by standardizing panel labeling and inspection checklists.",
                ],
            },
            {
                "title": f"{harborline['role']} -- {harborline['employer']}",
                "duration": f"{harborline['start']} -- {harborline['end']}",
                "bullets": [
                    "Assisted with conduit bending, wire pulling, and panel terminations on 25 projects.",
                    "Completed 8,000 hours of supervised field work toward licensure.",
                ],
            },
        ],
        "education": [{
            "degree": facts.EDUCATION_RECORD["degree"],
            "field": facts.EDUCATION_RECORD["field"],
            "institution": facts.EDUCATION_RECORD["institution"],
            "year": facts.EDUCATION_RECORD["year"],
        }],
        "certifications": list(facts.CREDENTIALS),
    }


def _pdf_text(path: Path) -> str:
    result = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[:400]
    return result.stdout


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    import re
    return re.sub(r"<[^>]+>", " ", xml)


@pytest.fixture(scope="module")
def rendered():
    from resume_builder.build_resume_from_export import build_resume_from_payload

    with tempfile.TemporaryDirectory(prefix="fidelity-render-") as directory:
        pdf_path, docx_path = build_resume_from_payload(
            _payload(), Path(directory), job_title="Senior Electrician")
        assert pdf_path.exists() and pdf_path.stat().st_size > 1000
        assert docx_path.exists() and docx_path.stat().st_size > 1000
        yield {"pdf": _pdf_text(pdf_path), "docx": _docx_text(docx_path),
               "pdf_size": pdf_path.stat().st_size,
               "docx_size": docx_path.stat().st_size}


def test_rendered_pdf_contains_the_real_role_titles(rendered):
    """[renderer] Both real titles reach the page — not the placeholder.

    The check is on title *lines*, not on any occurrence of the word: the section
    heading legitimately reads "Professional Experience", and a bare substring search
    would fail on correct output.
    """
    pdf = rendered["pdf"]
    for record in facts.EMPLOYMENT_RECORDS:
        assert record["role"] in pdf, f"{record['role']} missing from the PDF"
    # The run-4 defect rendered as a title line "<placeholder> -- <employer>".
    for placeholder in facts.FABRICATED_TITLE_PLACEHOLDERS:
        for employer in (r["employer"] for r in facts.EMPLOYMENT_RECORDS):
            assert f"{placeholder} -- {employer}" not in pdf
            assert f"{placeholder} |" not in pdf


def test_rendered_docx_contains_the_real_role_titles(rendered):
    """[renderer] Same assertion for the DOCX producer."""
    docx = rendered["docx"]
    for record in facts.EMPLOYMENT_RECORDS:
        assert record["role"] in docx, f"{record['role']} missing from the DOCX"


def test_rendered_output_has_no_mojibake(rendered):
    """[renderer] No UTF-8-read-as-Latin-1 artefacts reach either file."""
    for kind in ("pdf", "docx"):
        text = rendered[kind]
        assert "\u00e2\u0080" not in text, f"mojibake in {kind}"
        assert "\u00e2" not in text, f"mojibake in {kind}"


def test_rendered_output_preserves_quantified_facts(rendered):
    """[renderer] Every source number survives rendering, with its own subject."""
    for kind in ("pdf", "docx"):
        text = rendered[kind]
        for fact in facts.QUANTIFIED_FACTS:
            if fact["value"] == "8+":  # summary wording, not a bullet
                continue
            assert fact["value"] in text, f"{fact['value']} missing from {kind}"
        # "6 apprentices" must render as people, never as crews.
        assert "6 apprentices" in text
        assert "six apprentice crews" not in text.lower()


def test_rendered_output_contains_no_invented_content(rendered):
    """[renderer] The forbidden inventions do not appear in a source-faithful render."""
    for kind in ("pdf", "docx"):
        text = rendered[kind]
        lowered = text.lower()
        for project in facts.ABSENT_FROM_SOURCE["projects_section_content"]:
            assert project.lower() not in lowered, f"invented project in {kind}"
        for measure in facts.ABSENT_FROM_SOURCE["invented_measurements"]:
            assert measure not in text, f"invented measurement in {kind}"
        for technology in facts.ABSENT_FROM_SOURCE["invented_technologies"]:
            assert technology.lower() not in lowered, f"invented technology in {kind}"
        assert "gpa" not in lowered


def test_rendered_output_preserves_education_and_credentials(rendered):
    """[renderer] Degree, school, year and both credential names survive."""
    record = facts.EDUCATION_RECORD
    for kind in ("pdf", "docx"):
        text = rendered[kind]
        assert record["institution"] in text
        assert record["year"] in text
        assert "Electrical Systems Technology" in text
        for credential in facts.CREDENTIALS:
            assert credential in text, f"{credential} missing from {kind}"
