"""Bare extracted section headings are structural, not uncertain facts.

Measured on Conservative website run ``3403b994`` (2026-11-07, revision
``823c1b3``). The source text comes from the document reader, where section
titles are styled paragraphs with no ``#`` prefix:

  Professional Summary
  SYNTHETIC-ONLY Operations analyst with 9 years ...
  Professional Experience
  ...

The inventory model classified four bare headings ``uncertain``:

  'Professional Summary\\n', 'Professional Experience\\n', 'Education\\n',
  'Skills\\n'

Every draft line was ``supported`` and no source fact was missing, yet the run
withheld as ``source_completeness_uncertain``. The existing structural-heading
rule required a leading ``#``, so it was unreachable on exactly the input shape
production receives.

These tests pin the extension and the paired refusals that keep it narrow.
"""
import pytest

import source_correction as sc


@pytest.mark.parametrize("heading", [
    "Professional Summary\n",
    "Professional Experience\n",
    "Education\n",
    "Skills\n",
    "Certifications\n",
    "Technical Skills\n",
    "Work Experience\n",
    "Professional Summary",
])
def test_bare_extracted_headings_are_structural(heading):
    assert sc._is_section_heading(heading) is True


@pytest.mark.parametrize("heading", [
    "## Professional Summary\n",
    "## Education\n",
])
def test_markdown_headings_still_work(heading):
    assert sc._is_section_heading(heading) is True


@pytest.mark.parametrize("content", [
    # A substantive sentence contains words outside the closed vocabulary.
    "Operations analyst supporting stock planning and parts availability.\n",
    "Maintain parts availability records for 7 service branches.\n",
    # A heading carrying a figure is content, not a label.
    "Education 2020\n",
    "Professional Experience 2019 - Present\n",
    # A multi-line span is not a single label.
    "Professional Summary\nOperations analyst with 9 years of experience.\n",
    # A name is candidate-specific.
    "Professional Summary for Jordan Example\n",
    "",
    "   \n",
])
def test_content_is_never_reclassified_as_structural(content):
    assert sc._is_section_heading(content) is False


def _inventory(*items):
    """Build a valid, fully partitioned inventory for one source."""
    text = "".join(quote for _, quote in items)
    claims, position = [], 0
    for classification, quote in items:
        end = position + len(quote)
        claims.append({"id": f"doc:{position}:{end}", "source_id": "doc",
                       "start": position, "end": end, "quote": quote,
                       "classification": classification})
        position = end
    return {"source_claims": claims}, {"doc": text}


def test_uncertain_bare_heading_is_corrected_to_non_resume():
    response, sources = _inventory(
        ("uncertain", "Professional Summary\n"),
        ("required", "Operations analyst with 9 years.\n"))
    out = sc._validate_inventory(response, sources)
    assert out[0]["classification"] == "non_resume"
    # The substantive span keeps its classification.
    assert out[1]["classification"] == "required"


def test_uncertain_substantive_span_still_withholds():
    response, sources = _inventory(
        ("uncertain", "Operations analyst supporting stock planning.\n"))
    out = sc._validate_inventory(response, sources)
    assert out[0]["classification"] == "uncertain"
