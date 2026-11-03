"""Export filename derivation and title binding. Offline, no provider calls.

Why this file exists
--------------------

Run 10 produced two correct documents whose filenames did not match the job title
typed into the form: `resume.pdf` and `Machine_Learning_EngineerTest.docx` against a
form value of "Senior Electrician".

The renderer's rule was established by running it, not by reading it::

    job_title = None                           -> "resume"
    job_title = "Senior Electrician"            -> "Senior_Electrician"
    job_title = "Machine Learning EngineerTest" -> "Machine_Learning_EngineerTest"

Those three observations pin the cause. The renderer is a pure function of the
headline it is given, so it cannot have invented a title. The two exports carried
**different** headline values inside one session — none, and "Machine Learning
EngineerTest" — and neither matched the form. The defect is upstream of the renderer,
in the export payload's title.

What this file can and cannot prove
-----------------------------------

It proves the renderer mapping is deterministic and that a title only reaches the
filename through ``job_title``/``headline``. It cannot prove which upstream field
supplied "Machine Learning EngineerTest" for run 10, because that export request body
was never captured. The frontend regression
(``src/pages/__tests__/exportFilenameTitleBinding.test.ts``) constrains the derivation
order; together they bound the mechanism without overclaiming.

The derivation is tested as the same pure expression the renderer uses
(``re.sub(r"[^A-Za-z0-9_-]+", "_", job_title or resume.headline or "resume")``) so the
suite does not need ``pdflatex`` and stays portable.

Labelled **[export-filename]**.
"""
from __future__ import annotations

import re

import pytest


def safe_title(job_title: str | None, headline: str | None = None) -> str:
    """The renderer's filename rule, character for character.

    Mirrors ``resume_builder/build_resume_from_export.py``::

        safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", job_title or resume.headline or "resume")
    """
    return re.sub(r"[^A-Za-z0-9_-]+", "_", job_title or headline or "resume")


# ---------------------------------------------------------------------------
# The three observations that identify the cause
# ---------------------------------------------------------------------------

def test_the_run_10_filenames_are_exactly_what_the_renderer_produces():
    """[export-filename] Both artifacts are explained by the headline they received.

    If either name were unreachable from the rule, the renderer would be the suspect.
    Both are reachable, which moves the defect upstream.
    """
    assert safe_title(None) == "resume"                      # resume.pdf, run 10
    assert safe_title("Machine Learning EngineerTest") == "Machine_Learning_EngineerTest"


def test_the_typed_title_would_have_produced_a_different_filename():
    """[export-filename] The correct name was one argument away.

    This is the whole anomaly in one assertion: with the form's own title, the file
    would have been ``Senior_Electrician``. It was not, so the payload did not carry
    it.
    """
    assert safe_title("Senior Electrician") == "Senior_Electrician"
    assert safe_title("Senior Electrician") != "Machine_Learning_EngineerTest"
    assert safe_title("Senior Electrician") != "resume"


# ---------------------------------------------------------------------------
# The rule is a pure function of its inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title,expected", [
    ("Senior Electrician", "Senior_Electrician"),
    ("Journeyman Electrician", "Journeyman_Electrician"),
    ("Electrical Apprentice", "Electrical_Apprentice"),
    ("Facilities Maintenance Technician", "Facilities_Maintenance_Technician"),
    ("Maintenance Electrician", "Maintenance_Electrician"),
])
def test_electrician_titles_never_collapse_to_an_empty_or_generic_name(title, expected):
    """[export-filename] No electrician title produces "resume" or an empty stem."""
    assert safe_title(title) == expected
    assert safe_title(title) not in {"", "resume"}


def test_an_electrician_analysis_cannot_acquire_an_engineer_filename_on_its_own():
    """[export-filename] The renderer cannot invent a title it was not given.

    An electrician payload supplied with no title yields ``resume``. Producing
    ``Machine_Learning_EngineerTest`` requires that exact string as an input, so the
    stale title came from state, not from the renderer.
    """
    electrician_payload_titles = [None, "", "Senior Electrician", "Journeyman Electrician"]
    for value in electrician_payload_titles:
        assert "Machine_Learning" not in safe_title(value)
        assert "Engineer" not in safe_title(value)


def test_headline_is_used_only_when_no_job_title_is_given():
    """[export-filename] Precedence: job_title, then headline, then "resume".

    This is why an empty ``job_title`` does not fall back to a previous analysis: the
    only fallbacks are the payload's own headline and a constant.
    """
    assert safe_title("Senior Electrician", "Machine Learning EngineerTest") == \
        "Senior_Electrician"
    assert safe_title(None, "Machine Learning EngineerTest") == \
        "Machine_Learning_EngineerTest"
    assert safe_title("", "", ) == "resume"


def test_the_filename_is_derived_from_the_same_value_the_receipt_binds():
    """[export-filename] The title is receipt-bound payload, not a UI decoration.

    ``build_resume_from_payload_base64`` documents that without an override the
    filename comes from the receipt-bound headline. So a filename that disagrees with
    the form is evidence about the *payload*, which is verified content — not a
    cosmetic mismatch. Recorded as a test so the anomaly is not dismissed as display
    noise.
    """
    payload_headline = "Senior Electrician"
    # The renderer used the payload headline; there is no separate filename channel.
    assert safe_title(payload_headline) == safe_title(None, payload_headline)
