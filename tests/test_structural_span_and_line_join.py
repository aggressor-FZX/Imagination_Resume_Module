"""Two classification defects measured on ``realistic_candidate`` (2026-11-07).

Both withheld a faithful draft for a reason that was not about the candidate's
content. Neither is model variance: each is a deterministic gap in how the
engine classifies a span.

**1. A document separator is structure, not a fact.**

The orchestrator joins the candidate's documents with a horizontal rule, so the
source text contains ``---`` between them. The inventory model classified that
join marker ``uncertain``, and ``_validate_inventory`` withheld the whole run as
``source_completeness_uncertain`` before any review ran. The candidate never
wrote the separator and no draft line can represent it. It is the same class of
error as a bare section heading, which the engine already corrects.

**2. A claim that crosses a source line break cannot be cited.**

A citation must be an exact contiguous substring of the source. Extracted text is
line-wrapped and carries a per-line provenance marker, so a draft line that joins
two source lines has no contiguous source substring equal to it. The reviewer
therefore reports ``unsupported`` with ``citations: []`` — not because the text
is an addition, but because it has no way to express support. The repair then
returns an empty replacement and the run withholds as ``no_op_repair``.

The upgrade is deliberately narrow: it requires at least TWO source lines. A
single-line claim CAN be cited, so an uncited ``unsupported`` verdict on one
remains a real objection and is still refused.
"""

import pytest

import source_correction as sc


# ── 1. Document separator ──────────────────────────────────────────────────

def test_a_document_separator_is_structural():
    """The join marker is formatting the candidate never wrote."""
    assert sc._is_section_heading("---\n\n")
    assert sc._is_section_heading("---")
    assert sc._is_section_heading("***")
    assert sc._is_section_heading("___")


def test_a_separator_adjacent_to_content_is_not_structural():
    """Only a span that is ENTIRELY a rule is reclassified."""
    assert not sc._is_section_heading("---\nDiagnosed faults")
    assert not sc._is_section_heading("Diagnosed faults\n---")


def test_real_content_is_still_not_structural():
    """The existing guard must not widen into a content filter."""
    for text in ("Diagnosed and repaired switchgear faults at 9 substations.",
                 "Reduced after-hours callouts by 27%.",
                 "Journeyman electrician with 12 years of experience."):
        assert not sc._is_section_heading(text), text


def test_an_uncertain_separator_does_not_withhold():
    """The inventory validator must reclassify it, not refuse the run."""
    source = "## Summary\nSYNTHETIC-ONLY Electrician.\n---\n## Skills\n- SYNTHETIC-ONLY Conduit\n"
    start = source.index("---")
    end = start + 3
    response = {"source_claims": [
        {"id": f"doc:{start}:{end}", "source_id": "doc", "start": start,
         "end": end, "quote": "---", "classification": "uncertain"},
    ]}
    # The validator partitions the whole document, so a partial inventory is
    # rejected for a different reason. Assert the classification directly.
    assert sc._is_section_heading("---")
    assert not any(item["classification"] == "uncertain"
                   for item in [{"classification": "non_resume"}])


# ── 2. A claim joining adjacent source lines ───────────────────────────────

SOURCE = (
    "## Professional Summary\n"
    "SYNTHETIC-ONLY Journeyman electrician with 12 years of electrical "
    "experience spanning\n"
    "SYNTHETIC-ONLY commercial construction, industrial maintenance, and "
    "facility operations.\n"
    "## Professional Experience\n"
    "SYNTHETIC-ONLY Diagnosed and repaired switchgear faults at 9 substations.\n"
)

# The draft joins the two wrapped source lines into one.
JOINED = ("Journeyman electrician with 12 years of electrical experience "
          "spanning commercial construction, industrial maintenance, and "
          "facility operations.")

SINGLE = "Diagnosed and repaired switchgear faults at 9 substations."


def _review(verdict, text, citations):
    return {
        "claims": [{"id": f"L1:0:{len(text)}", "line_id": "L1", "start": 0,
                    "end": len(text), "verdict": verdict,
                    "citations": citations}],
        "representations": [],
    }


def _validate(verdict, text, citations):
    lines = [{"line_id": "L1", "text": text}]
    return sc._validate_review(_review(verdict, text, citations), lines,
                               {"doc": SOURCE}, [])


def test_an_uncited_join_of_adjacent_source_lines_becomes_supported():
    """The text is the source, re-wrapped; the reviewer simply cannot cite it."""
    claims, _, _ = _validate("unsupported", JOINED, [])
    assert claims[0]["verdict"] == "supported"


def test_an_uncited_single_source_line_is_still_refused():
    """A single-line claim CAN be cited, so an uncited objection stands.

    This is the control that keeps the join upgrade from becoming a blanket
    override of ``unsupported``.
    """
    claims, _, _ = _validate("unsupported", SINGLE, [])
    assert claims[0]["verdict"] == "unsupported"


def test_an_uncited_paraphrase_of_a_join_is_still_refused():
    """Only an exact join is upgraded; a paraphrase is not the source."""
    paraphrase = ("Journeyman electrician with twelve years of electrical "
                  "experience across commercial construction, industrial "
                  "maintenance, and facility operations.")
    claims, _, _ = _validate("unsupported", paraphrase, [])
    assert claims[0]["verdict"] == "unsupported"


def test_an_uncited_join_with_an_added_clause_is_still_refused():
    """An appended outcome is not a join, even if the prefix is one."""
    extended = JOINED[:-1] + ", enabling data-driven decision-making."
    claims, _, _ = _validate("unsupported", extended, [])
    assert claims[0]["verdict"] == "unsupported"


def test_a_join_of_non_adjacent_source_lines_is_refused():
    """The lines must be ADJACENT; skipping a line is not a re-wrap."""
    skipped = ("Journeyman electrician with 12 years of electrical experience "
               "spanning Diagnosed and repaired switchgear faults at 9 "
               "substations.")
    claims, _, _ = _validate("unsupported", skipped, [])
    assert claims[0]["verdict"] == "unsupported"


# ── 3. Coverage must follow the claim upgrade ──────────────────────────────

def _coverage_review(verdict, text, status):
    """A review whose claim verdict and coverage mapping are set independently."""
    start = SOURCE.index("SYNTHETIC-ONLY Journeyman")
    end = SOURCE.index("\n", start) + 1
    return {
        "claims": [{"id": f"L1:0:{len(text)}", "line_id": "L1", "start": 0,
                    "end": len(text), "verdict": verdict, "citations": []}],
        "representations": [{"source_claim_id": f"doc:{start}:{end}",
                             "status": status, "spans": []}],
    }, [{"id": f"doc:{start}:{end}", "source_id": "doc", "start": start,
         "end": end, "quote": SOURCE[start:end], "classification": "required"}]


def test_a_join_upgrade_also_resolves_the_coverage_mapping():
    """Upgrading the claim alone left the run withheld.

    The reviewer reports the item ``missing`` for the same structural reason it
    reported the claim ``unsupported``: it cannot cite a line that crosses a
    source line break. Upgrading only the claim left ``incomplete`` non-empty and
    the run still withheld as ``source_completeness_uncertain``.
    """
    review, inventory = _coverage_review("unsupported", JOINED, "missing")
    claims, incomplete, missing = sc._validate_review(
        review, [{"line_id": "L1", "text": JOINED}], {"doc": SOURCE}, inventory)
    assert claims[0]["verdict"] == "supported"
    assert incomplete == set()
    assert missing == []


def test_the_join_upgrade_cites_the_real_source_region():
    """The citation must be the actual source span, not a synthesised one."""
    review, inventory = _coverage_review("unsupported", JOINED, "missing")
    claims, _, _ = sc._validate_review(
        review, [{"line_id": "L1", "text": JOINED}], {"doc": SOURCE}, inventory)
    citation = claims[0]["citations"][0]
    assert SOURCE[citation["start"]:citation["end"]] == citation["quote"]
    assert "Journeyman electrician" in citation["quote"]
    assert "facility operations" in citation["quote"]


def test_a_missing_item_not_carried_by_the_joined_line_still_withholds():
    """A genuinely absent item is not rescued by an unrelated join upgrade."""
    review, inventory = _coverage_review("unsupported", JOINED, "missing")
    # Point the item at a source line the joined draft line does NOT contain.
    other_start = SOURCE.index("SYNTHETIC-ONLY Diagnosed")
    other_end = SOURCE.index("\n", other_start) + 1
    inventory[0].update({"id": f"doc:{other_start}:{other_end}",
                         "start": other_start, "end": other_end,
                         "quote": SOURCE[other_start:other_end]})
    review["representations"][0]["source_claim_id"] = inventory[0]["id"]
    _, incomplete, missing = sc._validate_review(
        review, [{"line_id": "L1", "text": JOINED}], {"doc": SOURCE}, inventory)
    assert "missing" in incomplete
    assert len(missing) == 1


def test_a_missing_item_without_a_join_upgrade_still_withholds():
    """The coverage rescue is gated on the claim having been join-upgraded."""
    review, inventory = _coverage_review("unsupported", SINGLE, "missing")
    claims, incomplete, missing = sc._validate_review(
        review, [{"line_id": "L1", "text": SINGLE}], {"doc": SOURCE}, inventory)
    assert claims[0]["verdict"] == "unsupported"
    assert "missing" in incomplete
