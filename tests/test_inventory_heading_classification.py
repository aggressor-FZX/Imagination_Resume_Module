"""A bare section heading is formatting, not an uncertain candidate fact.

Measured on protected capture ``2dc8b2b7-d3c7-4b24-bf66-fff4018833e9``
(Conservative, frozen fixture, direct ``/analyze``). The run withheld as
``source_completeness_uncertain`` with ``review_calls: 0`` -- before any draft
line was judged -- because the inventory model classified two supplement spans
``uncertain``:

1. ``## Professional Summary\\n`` -- a bare Markdown section heading. The SAME
   heading in the main document was classified ``non_resume`` on the same run,
   and every other heading in both documents was ``non_resume``. This is pure
   formatting with no candidate claim: a **structural** span mislabelled
   ``uncertain``.
2. ``SYNTHETIC-ONLY Operations analyst supporting stock planning and parts
   availability for a regional equipment service network.\\n\\n`` -- a
   substantive candidate fact (a role/summary statement). It is neither
   formatting nor mixed; it is a real claim whose ownership/relevance the model
   declined to confirm. It is **genuinely ambiguous** and must KEEP its
   ``uncertain`` classification and still withhold.

The narrow correction: when the inventory model calls a BARE section heading
``uncertain``, reclassify only that word to ``non_resume``. It does not waive
the span (it is still partitioned and accounted for in every review), does not
touch any substantive or mixed span, and does not turn all uncertain spans into
structural content. A heading carrying any candidate-specific word (a name, a
number, a metric, a verb) is left exactly as classified.
"""
import collections
import copy
import json
from pathlib import Path

import pytest

import source_correction as sc

CAPTURE = (Path(__file__).resolve().parent.parent
           / "captures" / "recovery-validation" / "post-recovery-fix" / "capture.json")

# The two captured spans, verbatim. Kept here so the tests do not depend on the
# private capture being present on every machine.
HEADING = "## Professional Summary\n"
SUBSTANTIVE = ("SYNTHETIC-ONLY Operations analyst supporting stock planning and "
               "parts availability for a regional equipment service network.\n\n")


def _inventory(*items):
    """Build a valid, fully partitioned inventory for one source."""
    text = "".join(item[1] for item in items)
    claims, position = [], 0
    for classification, quote in items:
        end = position + len(quote)
        claims.append({"id": f"doc:{position}:{end}", "source_id": "doc",
                       "start": position, "end": end, "quote": quote,
                       "classification": classification})
        position = end
    return {"source_claims": claims}, {"doc": text}


# --- 1. The classification of each captured span ---------------------------

def test_a_bare_section_heading_is_structural_not_a_candidate_fact():
    """Span 1: formatting with no candidate claim."""
    assert sc._is_section_heading(HEADING) is True
    # It is not substantive, not mixed, and not genuinely ambiguous: it is a
    # known section label and nothing else.
    assert HEADING.strip().lstrip("#").strip() == "Professional Summary"


def test_a_substantive_summary_sentence_is_not_a_section_heading():
    """Span 2: a substantive candidate fact, so it must NOT be reclassified."""
    assert sc._is_section_heading(SUBSTANTIVE) is False


@pytest.mark.parametrize("heading", [
    "## Professional Summary\n", "## Professional Experience\n\n",
    "## Education\n", "## Skills\n", "## Certifications\n",
    "## Technical Skills\n", "## Work Experience\n", "## Core Competencies\n",
])
def test_known_generic_headings_are_structural(heading):
    assert sc._is_section_heading(heading) is True


@pytest.mark.parametrize("mixed", [
    "## Professional Summary for Jordan Example\n",  # a name
    "## Increased revenue 30%\n",                    # a metric
    "## Managed 14 depots\n",                        # a verb + figure
    "## Professional Summary\nSYNTHETIC-ONLY Operations analyst\n",  # heading + fact
    "SYNTHETIC-ONLY Operations analyst supporting stock planning\n",  # no '#'
])
def test_a_heading_carrying_content_is_not_structural(mixed):
    """Mixed / substantive / ambiguous spans keep their classification."""
    assert sc._is_section_heading(mixed) is False


# --- 2. The narrow correction ----------------------------------------------

def test_only_the_uncertain_bare_heading_is_corrected():
    """The demonstrated fix: one word, one span, nothing else."""
    response, sources = _inventory(
        ("uncertain", HEADING),
        ("uncertain", SUBSTANTIVE),
        ("required", "- SYNTHETIC-ONLY Maintained parts records.\n"),
    )
    validated = sc._validate_inventory(response, sources)
    by_quote = {item["quote"]: item["classification"] for item in validated}

    assert by_quote[HEADING] == "non_resume", "bare heading must be corrected"
    assert by_quote[SUBSTANTIVE] == "uncertain", "substantive fact must be preserved"
    assert by_quote["- SYNTHETIC-ONLY Maintained parts records.\n"] == "required"


def test_a_required_heading_is_also_structural():
    """A heading is structure, not a fact to be represented.

    Measured on Conservative website run ``b2198062`` (2026-11-07): the inventory
    marked the bare headings ``Professional Summary`` and ``Professional
    Experience`` ``required``. The draft carries them as ``## Professional
    Summary``, so the reviewer could not map the bare quote to the marked line,
    reported both ``missing``, and the run withheld as ``source_facts_missing``
    with every heading and fact present.
    """
    response, sources = _inventory(
        ("required", HEADING),
        ("non_resume", "## Education\n"),
    )
    validated = sc._validate_inventory(response, sources)
    assert [item["classification"] for item in validated] == ["non_resume", "non_resume"]


def test_a_required_substantive_span_is_left_untouched():
    """The correction is not a blanket rewrite of required spans."""
    response, sources = _inventory(
        ("required", SUBSTANTIVE),
        ("non_resume", "## Education\n"),
    )
    validated = sc._validate_inventory(response, sources)
    assert [item["classification"] for item in validated] == ["required", "non_resume"]


def test_a_mixed_heading_keeps_its_uncertain_classification():
    """A heading with candidate content is not silently waived."""
    mixed = "## Professional Summary for Jordan Example\n"
    response, sources = _inventory(("uncertain", mixed))
    validated = sc._validate_inventory(response, sources)
    assert validated[0]["classification"] == "uncertain"


def test_the_span_is_still_partitioned_and_accounted_for():
    """Reclassifying is not waiving: the heading still covers its exact offsets."""
    response, sources = _inventory(
        ("uncertain", HEADING),
        ("required", "- SYNTHETIC-ONLY Maintained parts records.\n"),
    )
    validated = sc._validate_inventory(response, sources)
    assert validated[0]["start"] == 0 and validated[0]["end"] == len(HEADING)
    assert validated[0]["quote"] == HEADING
    assert validated[0]["id"] == f"doc:0:{len(HEADING)}"


def test_a_reclassified_heading_is_not_silently_waived():
    """The review must still account for the span; omitting it is malformed.

    ``non_resume`` is a real classification, not a waiver: ``_validate_review``
    requires every inventory item -- including non_resume ones -- to appear in
    ``representations``. A review that drops the heading is rejected, so the
    correction cannot be used to skip a span.
    """
    response, sources = _inventory(
        ("uncertain", HEADING),
        ("required", "- SYNTHETIC-ONLY Maintained parts records.\n"),
    )
    inventory = sc._validate_inventory(response, sources)
    heading_id = inventory[0]["id"]
    required_id = inventory[1]["id"]

    line_text = "## Professional Summary"
    lines = [{"line_id": "L000001", "text": line_text, "absolute_start": 0}]
    review = {
        "claims": [{"id": f"L000001:0:{len(line_text)}", "line_id": "L000001",
                    "start": 0, "end": len(line_text), "verdict": "structural",
                    "citations": []}],
        # The heading's representation is deliberately omitted.
        "representations": [{"source_claim_id": required_id, "status": "missing",
                             "spans": []}],
    }
    with pytest.raises(sc._Invalid):
        sc._validate_review(review, lines, sources, inventory)

    # With the heading accounted for, the same review is well-formed.
    review["representations"].append(
        {"source_claim_id": heading_id, "status": "non_resume", "spans": []})
    claims, incomplete, missing = sc._validate_review(review, lines, sources, inventory)
    assert claims and "missing" in incomplete and missing


# --- 3. End-to-end: the gate still withholds on a real uncertain fact ------

def test_a_genuinely_uncertain_fact_still_withholds_before_review():
    """Preserving review of candidate facts: the substantive span still blocks."""
    import asyncio

    async def callback(phase, payload):
        assert phase == "source_inventory", "no review may run on an uncertain fact"
        return _inventory(("uncertain", SUBSTANTIVE))[0]

    result = asyncio.run(sc.correct(
        "## Professional Summary\nSYNTHETIC-ONLY Operations analyst\n",
        (sc.SourceDocument("doc", SUBSTANTIVE),), callback, timeout_seconds=30))
    assert result.status == "withheld"
    assert result.reason == "source_completeness_uncertain"
    assert result.review_calls == 0


def test_correcting_the_heading_alone_does_not_approve_a_draft():
    """The fix removes one false block; it does not grant approval."""
    import asyncio

    async def callback(phase, payload):
        if phase == "source_inventory":
            return _inventory(("uncertain", HEADING), ("required", SUBSTANTIVE))[0]
        # A review that cannot substantiate the required fact must still withhold.
        return {"claims": [], "representations": []}

    result = asyncio.run(sc.correct(
        "## Professional Summary\n",
        (sc.SourceDocument("doc", HEADING + SUBSTANTIVE),), callback,
        timeout_seconds=30))
    assert result.status == "withheld"


# --- 4. Replay the exact captured inventory --------------------------------

@pytest.mark.skipif(not CAPTURE.exists(), reason="protected capture not present")
def test_the_captured_inventory_loses_exactly_one_false_uncertain():
    """Replay all 35 captured claims: only the bare heading changes."""
    data = json.loads(CAPTURE.read_text())
    phase = data["phases"][0]
    sources = {s["source_id"]: s["text"] for s in phase["payload"]["sources"]}
    before = copy.deepcopy(phase["result"]["source_claims"])
    after = sc._validate_inventory({"source_claims": copy.deepcopy(before)}, sources)

    assert collections.Counter(c["classification"] for c in before) == {
        "required": 25, "non_resume": 8, "uncertain": 2}
    assert collections.Counter(c["classification"] for c in after) == {
        "required": 25, "non_resume": 9, "uncertain": 1}

    changed = [(b["id"], b["classification"], a["classification"])
               for b, a in zip(before, after)
               if b["classification"] != a["classification"]]
    assert changed == [("candidate-supplement.txt:0:24", "uncertain", "non_resume")]
    # The substantive summary sentence is still uncertain and still withholds.
    assert any(c["classification"] == "uncertain" and c["quote"] == SUBSTANTIVE
               for c in after)
