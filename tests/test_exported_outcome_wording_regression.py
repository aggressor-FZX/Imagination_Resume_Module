"""Regression for the exported unsupported outcome wording and omitted qualifier.

Measured on the retained Imaginative run ``de375f60`` (capture
``captures/restoration-mapping-validation/imaginative-535341063cd245f7be18ab5ac0132de1``).
The run was APPROVED (``requires_review: false``, receipt issued, export 200) and
its compiled PDF contains wording the source never states, while dropping a
source qualifier.

Exact source -> output comparison (frozen fixture, verbatim):

  SOURCE  "- SYNTHETIC-ONLY Reconciled 1,180 fuel invoices against depot
           dispensing records each quarter."
  OUTPUT  "- Reconciled 1,180 fuel invoices quarterly against depot records,
           ensuring accurate fuel expense reporting."
  ADDED   "ensuring accurate fuel expense reporting" -- an outcome/purpose
          clause no source states.

  SOURCE  "- SYNTHETIC-ONLY Prepared the monthly cost summary reviewed by the
           regional manager."
  OUTPUT  "- Prepared monthly cost summaries for regional management review,
           highlighting key cost drivers."
  ADDED   "highlighting key cost drivers" -- an outcome clause no source states.

  SOURCE  "- SYNTHETIC-ONLY Compared quarterly parts usage against the prior
           four quarters to set reorder points."
  OUTPUT  "- Analyzed quarterly parts usage with historical comparison to set
           data-driven reorder points."
  DROPPED "against the prior four quarters" -- the specific comparison basis.

WHY APPROVAL ALLOWED THEM (verified directly against the real bullets):

* ``_introduces_beyond`` returns ``[]`` for all three. It detects only
  capitalised proper nouns (tools, platforms, institutions) and unbacked
  figures. These additions are LOWERCASE outcome words ("ensuring", "accurate",
  "fuel", "expense", "reporting", "highlighting", "key", "cost", "drivers"), so
  the entity scan skips them; no new figure is introduced either.
* ``_quote_shares_content`` returns ``True`` for each, because the rewritten
  bullet still shares content words with its source item ("reconciled", "fuel",
  "invoices", "prepared", "cost", "summary", "quarterly", "parts", "usage").
  A shared word proves relatedness, not that the whole meaning is preserved.
* The reviewer marked these lines ``supported``, so the engine's only defence
  for this shape -- flag then repair -- was never engaged. The completeness
  check asks whether each required source item is *represented*, not whether
  the draft introduced extra outcome claims or dropped a qualifier.

This test is a REGRESSION RECORD, not a fix. It pins the observed behaviour so a
future change to the addition check can be measured against these exact cases.
It does NOT weaken any factual check, does not reshape the frozen fixture, and
does not assert that these bullets should be approved. The open remediation
(extending the addition check to lowercase outcome clauses, or requiring an
exact source span per bullet) is deliberately NOT implemented here -- that is a
separate, authorized change.
"""
import json
import re
from pathlib import Path

import pytest

import source_correction as sc

BASE = (Path(__file__).resolve().parent.parent / "captures"
        / "restoration-mapping-validation"
        / "imaginative-535341063cd245f7be18ab5ac0132de1")

# Verbatim source bullets from tests/fixtures/bridge_opportunity/*.txt
SOURCE_RECONCILED = ("- SYNTHETIC-ONLY Reconciled 1,180 fuel invoices against "
                     "depot dispensing records each quarter.")
SOURCE_COST_SUMMARY = ("- SYNTHETIC-ONLY Prepared the monthly cost summary "
                       "reviewed by the regional manager.")
SOURCE_REORDER = ("- SYNTHETIC-ONLY Compared quarterly parts usage against the "
                  "prior four quarters to set reorder points.")

# Verbatim output bullets from the approved final_written_section_markdown
OUTPUT_RECONCILED = ("- Reconciled 1,180 fuel invoices quarterly against depot "
                     "records, ensuring accurate fuel expense reporting.")
OUTPUT_COST_SUMMARY = ("- Prepared monthly cost summaries for regional "
                       "management review, highlighting key cost drivers.")
OUTPUT_REORDER = ("- Analyzed quarterly parts usage with historical comparison "
                  "to set data-driven reorder points.")

CASES = [
    ("ensuring accurate fuel expense reporting", SOURCE_RECONCILED, OUTPUT_RECONCILED),
    ("highlighting key cost drivers", SOURCE_COST_SUMMARY, OUTPUT_COST_SUMMARY),
    ("omitted 'against the prior four quarters'", SOURCE_REORDER, OUTPUT_REORDER),
]


@pytest.mark.parametrize("label,source,output", CASES,
                         ids=[c[0] for c in CASES])
def test_the_addition_check_does_not_flag_the_observed_wording(label, source, output):
    """Why approval allowed it: the check sees no introduced name or figure."""
    assert sc._introduces_beyond(output, [source], source) == [], label


@pytest.mark.parametrize("label,source,output", CASES,
                         ids=[c[0] for c in CASES])
def test_the_rewritten_bullet_still_shares_content_with_its_source(label, source, output):
    """Why approval allowed it: relatedness is not fidelity."""
    assert sc._quote_shares_content(output, source) is True, label


def test_the_added_outcome_clauses_are_not_in_the_source():
    """The additions are genuinely unbacked, not paraphrases of source words."""
    all_source = SOURCE_RECONCILED + " " + SOURCE_COST_SUMMARY + " " + SOURCE_REORDER
    for phrase in ("ensuring accurate fuel expense reporting",
                   "highlighting key cost drivers"):
        assert phrase not in all_source
        # And they are lowercase outcome words, which is exactly the blind spot.
        assert all(word[0].islower() for word in phrase.split())


def test_the_dropped_qualifier_is_a_real_source_fact():
    """The omitted qualifier is a specific comparison basis, not a stop word."""
    assert "against the prior four quarters" in SOURCE_REORDER
    assert "against the prior four quarters" not in OUTPUT_REORDER
    assert "historical comparison" in OUTPUT_REORDER  # the vague replacement


@pytest.mark.skipif(not (BASE / "analysis.raw").exists(),
                    reason="protected capture not present")
def test_the_approved_response_really_contains_the_wording():
    """The observed case is real: the approved markdown carries these bullets."""
    raw = json.loads((BASE / "analysis.raw").read_text())
    assert raw.get("requires_review") is False
    assert raw.get("publication_receipt")
    markdown = raw["final_written_section_markdown"]
    assert "ensuring accurate fuel expense reporting" in markdown
    assert "highlighting key cost drivers" in markdown
    assert "against the prior four quarters" not in markdown


@pytest.mark.skipif(not (BASE / "export.raw").exists(),
                    reason="protected capture not present")
def test_the_wording_reached_the_compiled_pdf():
    """Export 200 is not factual acceptance: the PDF carries the same wording."""
    import base64
    import zlib

    data = json.loads((BASE / "export.raw").read_bytes())
    pdf = base64.b64decode(data["pdf_base64"])
    blob = ""
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", pdf, re.DOTALL):
        try:
            blob += zlib.decompress(match.group(1)).decode("latin-1") + "\n"
        except Exception:
            pass
    fragments = re.findall(r"\((?:[^()\\]|\\.)*\)", blob)
    compact = re.sub(r"\s+", "", "".join(f[1:-1] for f in fragments)).lower()

    def present(phrase):
        return re.sub(r"\s+", "", phrase).lower() in compact

    assert present("ensuring accurate fuel expense reporting")
    assert present("highlighting key cost drivers")
    assert present("historical comparison")
    assert not present("against the prior four quarters")
