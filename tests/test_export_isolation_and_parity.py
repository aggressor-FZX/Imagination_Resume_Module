"""Offline integration checks for private export files and representative content."""

import base64
from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import re
import shutil
import stat
import subprocess
import threading

from docx import Document
import pytest

from resume_builder import build_resume_from_export as renderer


@pytest.fixture
def pdf_compile(monkeypatch, request):
    """Use real pdflatex when present; otherwise inspect the actual generated TeX."""
    if getattr(request, "param", None) != "double" and shutil.which("pdflatex"):
        return True

    def compile_double(tex_path, working_dir):
        pdf_path = tex_path.with_suffix(".pdf")
        pdf_path.write_bytes(b"TEST-DOUBLE-TEX\n" + tex_path.read_bytes())
        return pdf_path

    monkeypatch.setattr(renderer, "compile_latex", compile_double)
    return False


def docx_text(encoded):
    document = Document(io.BytesIO(base64.b64decode(encoded)))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def pdf_text(encoded, real_pdf, tmp_path):
    data = base64.b64decode(encoded)
    if not real_pdf:
        # TeX stores the first and last name in separate command arguments.
        return re.sub(r"\\name\{([^{}]*)\}\{([^{}]*)\}", r"\1 \2", data.decode("utf-8"))
    if not shutil.which("pdftotext"):
        pytest.skip("pdftotext is required to inspect real PDF content")
    pdf = tmp_path / "inspect.pdf"
    pdf.write_bytes(data)
    result = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True, capture_output=True)
    return result.stdout.decode("utf-8")


@pytest.mark.parametrize("pdf_compile", ["auto", "double"], indirect=True)
def test_both_formats_retain_representative_supplied_content(pdf_compile, tmp_path):
    payload = {
        "name": "Synthetic Candidate",
        "email": "person@example.test",
        "headline": "Approved Systems Specialist",
        "summary": "Supplied summary for testing.",
        "experience": [{"role": "Analyst", "company": "Example Labs", "duration": "2021 -- 2023",
                        "bullets": ["Built a synthetic report."]}],
        "projects": [{"title": "Community Archive", "bullets": ["Catalogued documents."]}],
        "skills": [{"category": "Languages", "items": ["Python", "SQL"]}],
        "education": ["Synthetic College"],
        "certifications": ["Synthetic Certificate"],
        "publication": "Example Study & Collaborators (2024)",
    }
    result = renderer.build_resume_from_payload_base64(payload)
    assert result["filename"] == "Approved_Systems_Specialist"
    texts = [docx_text(result["docx_base64"]), pdf_text(result["pdf_base64"], pdf_compile, tmp_path)]
    for text in texts:
        for expected in ("Synthetic Candidate", "person@example.test", "Approved Systems Specialist",
                         "Supplied summary for testing.", "Analyst", "Example Labs", "2021",
                         "2023", "Built a synthetic report.", "Projects", "Community Archive",
                         "Catalogued documents.", "Languages", "Python", "SQL", "Synthetic College",
                         "Synthetic Certificate", "Publication", "Example Study", "Collaborators (2024)"):
            assert expected in text
        assert "ML and Analytics Projects" not in text
        assert "None provided" not in text
    if pdf_compile:
        assert "Example Study & Collaborators (2024)" in texts[1]
    else:
        assert r"Example Study \& Collaborators (2024)" in texts[1]


@pytest.mark.parametrize("payload", [
    {},
    {"name": "Synthetic Candidate", "email": "only@example.test", "skills": []},
    {"skills": [{"category": "Empty", "items": []}], "projects": [{}], "experience": [{}]},
    {"skills": [{"category": "Empty", "items": [None, {}]}], "education": [{}],
     "certifications": [None, {}], "experience": [{"text": {"unsupported": "value"}}]},
    {"projects": ["A supplied project detail"]},
])
def test_sparse_outputs_have_no_invented_placeholders(pdf_compile, tmp_path, payload):
    result = renderer.build_resume_from_payload_base64(payload)
    docx = docx_text(result["docx_base64"])
    pdf = pdf_text(result["pdf_base64"], pdf_compile, tmp_path)
    for text in (docx, pdf):
        assert "None provided" not in text
        assert "ML and Analytics Projects" not in text
        assert "Entry 1" not in text
        assert "Technical Skills" not in text
        assert "None" not in text
        assert "{}" not in text
    if "email" in payload:
        assert "only@example.test" in docx
        assert "|" not in docx
    if payload.get("projects") == ["A supplied project detail"]:
        assert "A supplied project detail" in docx and "A supplied project detail" in pdf


@pytest.mark.parametrize("headline,override,filename", [
    ("Receipt Bound Title", None, "Receipt_Bound_Title"),
    ("Receipt Bound Title", "Override Title", "Override_Title"),
    ("", None, "resume"),
])
def test_filename_and_rendered_headline_follow_builder(pdf_compile, tmp_path, headline, override, filename):
    result = renderer.build_resume_from_payload_base64({"headline": headline}, override)
    assert result["filename"] == filename
    expected = override or headline
    if expected:
        assert expected in docx_text(result["docx_base64"])
        assert expected in pdf_text(result["pdf_base64"], pdf_compile, tmp_path)


def test_concurrent_same_title_is_private_and_isolated(monkeypatch, pdf_compile, tmp_path):
    barrier = threading.Barrier(2, timeout=60)
    original_compile = renderer.compile_latex
    seen = []
    lock = threading.Lock()
    shared = tmp_path / "shared"
    monkeypatch.setattr(renderer, "DEFAULT_OUTPUT_DIR", shared)

    def synchronized_compile(tex_path, directory):
        with lock:
            seen.append((directory, stat.S_IMODE(directory.stat().st_mode)))
        path = original_compile(tex_path, directory)
        barrier.wait()
        return path

    monkeypatch.setattr(renderer, "compile_latex", synchronized_compile)
    def export(name):
        return renderer.build_resume_from_payload_base64({"name": name, "headline": "Same Title"})

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(export, ["Alice Synthetic", "Bob Synthetic"]))
    assert len(seen) == 2 and len({directory for directory, _ in seen}) == 2
    assert all(mode == 0o700 and not directory.exists() for directory, mode in seen)
    assert not shared.exists()
    for result, own, other in zip(results, ["Alice Synthetic", "Bob Synthetic"], ["Bob Synthetic", "Alice Synthetic"]):
        assert result["filename"] == "Same_Title"
        for text in (docx_text(result["docx_base64"]), pdf_text(result["pdf_base64"], pdf_compile, tmp_path)):
            assert own in text and other not in text


@pytest.mark.parametrize("failure", ["compile", "docx", "read"])
def test_private_directory_is_removed_after_failures(monkeypatch, pdf_compile, failure):
    seen = []
    original_compile = renderer.compile_latex
    original_read = Path.read_bytes

    def compile_capture(tex_path, directory):
        seen.append(directory)
        if failure == "compile":
            (directory / "sensitive.aux").write_text("synthetic", encoding="utf-8")
            raise RuntimeError("compile failed")
        return original_compile(tex_path, directory)

    def failed_docx(*args, **kwargs):
        raise RuntimeError("docx failed")

    def failed_read(path):
        if path.suffix == ".docx":
            raise RuntimeError("read failed")
        return original_read(path)

    monkeypatch.setattr(renderer, "compile_latex", compile_capture)
    if failure == "docx":
        monkeypatch.setattr(renderer, "build_docx", failed_docx)
    if failure == "read":
        monkeypatch.setattr(Path, "read_bytes", failed_read)
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        renderer.build_resume_from_payload_base64({"name": "Synthetic Candidate"})
    assert len(seen) == 1 and not seen[0].exists()


def test_latex_special_characters_are_escaped_once():
    assert renderer.latex_escape(r"C:\Work & 50%_done") == (
        r"C:\textbackslash{}Work \& 50\%\_done"
    )