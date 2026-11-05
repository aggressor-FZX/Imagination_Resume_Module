"""Phase D.7 — export semantics: no generation, and failure is never shaped as success.

Two properties the spec calls out for the download endpoint (§D.7):

* **Export performs zero model/taxonomy calls.** It renders approved bytes; it
  must not become an accidental path to a second analysis (or a second charge).
* **A builder failure must not return a success-shaped response.** A 200 carrying
  empty or missing base64 would look like a completed download to the UI and
  produce a zero-byte file.

Both are asserted at the renderer/route seam rather than by mocking the whole
handler, so a regression in either direction is visible.

The credit-debit property is NOT asserted here: `export_resume` has no credit
dependency of its own, and inventing an assertion about middleware this file does
not exercise would be a claim without evidence. What is proven is the narrower,
checkable fact — the export path imports no pipeline module and calls no stage.
"""

import base64
import inspect
import sys

import pytest

import app as imaginator_app
from resume_builder import build_resume_from_export as renderer


# Stage/provider modules that must never be imported by, or reached from, export.
_PIPELINE_MODULES = (
    "orchestrator",
    "stages.researcher",
    "stages.drafter",
    "stages.star_editor",
    "source_correction",
    "correction_protocol",
)


def test_export_handler_references_no_pipeline_stage():
    """Static proof: the handler's own source names no stage or provider call."""
    source = inspect.getsource(imaginator_app.export_resume)
    for module in _PIPELINE_MODULES:
        root = module.split(".")[0]
        assert f"import {root}" not in source, f"export must not import {root}"
        assert f"from {root}" not in source, f"export must not import from {root}"
    # No provider client, no LLM call, no taxonomy lookup.
    for forbidden in ("openrouter", "chat.completions", "call_model",
                      "run_stage", "O*NET", "onet"):
        assert forbidden not in source, f"export must not reference {forbidden}"


def test_export_route_declares_no_credit_dependency():
    """Export must not opt into the analysis debit path.

    The handler takes only the API key and the request. A credit dependency
    added later would show up here as an extra parameter.
    """
    params = list(inspect.signature(imaginator_app.export_resume).parameters)
    assert params == ["payload", "request", "api_key"], (
        f"export signature changed: {params} — a new dependency needs a "
        "deliberate review that it does not debit an analysis"
    )


def test_rendering_calls_no_stage_even_with_pipeline_modules_loaded(monkeypatch, tmp_path):
    """Dynamic proof: rendering a real payload reaches no pipeline stage.

    The modules are imported first so the assertion is meaningful — an unimported
    module cannot be "not called". With them present, every stage entry point is
    replaced with a tripwire.

    Stage entry points are CLASS METHODS (`Researcher.analyze`,
    `Drafter.write_section`, ...), not module functions. Patching only module
    functions would have made this test vacuous: it would have asserted that
    nothing was called among things the pipeline never exposes that way. Class
    methods and module functions are both covered below, and the test proves at
    least one tripwire was actually installed.
    """
    import importlib

    loaded = {}
    for name in _PIPELINE_MODULES:
        try:
            loaded[name] = importlib.import_module(name)
        except Exception:  # noqa: BLE001 - absence is fine, presence is the point
            continue

    called = []
    armed = []

    def _tripwire(label):
        def _fail(*args, **kwargs):
            called.append(label)
            raise AssertionError(f"render reached pipeline stage {label}")
        return _fail

    def _interesting(attr):
        return (attr.startswith("run") or attr.startswith("call")
                or attr.startswith("analyze") or attr.startswith("invoke")
                or attr.startswith("write") or attr.startswith("research")
                or attr.startswith("edit"))

    for name, module in loaded.items():
        for attr in dir(module):
            if attr.startswith("_"):
                continue
            value = getattr(module, attr)

            # Module-level functions and callables.
            if inspect.isfunction(value) and _interesting(attr):
                monkeypatch.setattr(module, attr, _tripwire(f"{name}.{attr}"),
                                    raising=False)
                armed.append(f"{name}.{attr}")
                continue

            # Stage classes: patch their public entry-point methods.
            if inspect.isclass(value) and value.__module__ == module.__name__:
                for method_name, method in vars(value).items():
                    if method_name.startswith("_") or not _interesting(method_name):
                        continue
                    if not (inspect.isfunction(method) or inspect.iscoroutinefunction(method)):
                        continue
                    monkeypatch.setattr(
                        value, method_name,
                        _tripwire(f"{name}.{attr}.{method_name}"), raising=False)
                    armed.append(f"{name}.{attr}.{method_name}")

    assert armed, (
        "no tripwire was installed, so this test would pass vacuously — the "
        "stage entry points have moved and the assertion needs updating"
    )

    payload = {
        "name": "Synthetic Candidate",
        "headline": "Data Analyst",
        "summary": "Analyst with measurement and reporting experience.",
        "skills": [{"category": "Analysis", "items": ["trend analysis"]}],
        "experience": [{"title": "Data Analyst", "company": "Cascade Power Solutions",
                        "bullets": ["Compared weekly sales figures to identify trends."]}],
        "education": ["Example Technical College -- 2016"],
        "certifications": [],
        "projects": [],
    }
    result = renderer.build_resume_from_payload_base64(payload)
    assert called == [], f"export invoked {called}"
    assert result["pdf_base64"] and result["docx_base64"]


# ── Failure is an error, never a success-shaped response ──────────────────


def test_latex_failure_raises_rather_than_returning_empty_bytes(monkeypatch, tmp_path):
    """A failed compile must not become a 200 with no PDF."""
    def explode(tex_path, working_dir):
        raise RuntimeError("pdflatex exploded")

    monkeypatch.setattr(renderer, "compile_latex", explode)
    with pytest.raises(Exception):
        renderer.build_resume_from_payload_base64(
            {"name": "Synthetic Candidate", "headline": "Data Analyst",
             "summary": "Analyst with reporting experience."}
        )


def test_docx_failure_raises_rather_than_returning_a_pdf_only_success(monkeypatch, tmp_path):
    """A DOCX failure must not yield a payload the UI treats as complete."""
    real_compile = renderer.compile_latex

    def fake_compile(tex_path, working_dir):
        # Produce a real-looking PDF so the failure is isolated to the DOCX step.
        pdf_path = tex_path.with_suffix(".pdf")
        pdf_path.write_bytes(b"%PDF-1.4 fake\n")
        return pdf_path

    def explode(*args, **kwargs):
        raise RuntimeError("python-docx exploded")

    monkeypatch.setattr(renderer, "compile_latex", fake_compile)
    monkeypatch.setattr(renderer, "build_docx", explode)
    try:
        with pytest.raises(Exception):
            renderer.build_resume_from_payload_base64(
                {"name": "Synthetic Candidate", "headline": "Data Analyst",
                 "summary": "Analyst with reporting experience."}
            )
    finally:
        monkeypatch.setattr(renderer, "compile_latex", real_compile)


def test_a_returned_result_always_carries_real_bytes(tmp_path):
    """Positive half: every key must hold decodable, non-empty content."""
    result = renderer.build_resume_from_payload_base64(
        {"name": "Synthetic Candidate", "headline": "Data Analyst",
         "summary": "Analyst with measurement and reporting experience.",
         "skills": [{"category": "Analysis", "items": ["trend analysis"]}]}
    )
    for key in ("pdf_base64", "docx_base64"):
        raw = base64.b64decode(result[key], validate=True)
        assert raw, f"{key} was empty"
    assert result["filename"]


def test_the_private_render_directory_is_removed_after_a_failure(monkeypatch, tmp_path):
    """Temporary artifacts must not survive a failed build.

    `TemporaryDirectory` owns cleanup, so the observable guarantee is that the
    directory it created no longer exists once the exception has propagated.
    """
    import tempfile
    from pathlib import Path

    created = []
    real_factory = tempfile.TemporaryDirectory

    class Traced:
        def __init__(self, *args, **kwargs):
            self._inner = real_factory(*args, **kwargs)
            created.append(Path(self._inner.name))

        def __enter__(self):
            return self._inner.__enter__()

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

    monkeypatch.setattr(renderer.tempfile, "TemporaryDirectory", Traced)
    monkeypatch.setattr(renderer, "compile_latex",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(Exception):
        renderer.build_resume_from_payload_base64(
            {"name": "Synthetic Candidate", "headline": "Data Analyst",
             "summary": "Analyst with reporting experience."}
        )
    assert created, "no temporary directory was created"
    for path in created:
        assert not path.exists(), f"{path} survived a failed build"
