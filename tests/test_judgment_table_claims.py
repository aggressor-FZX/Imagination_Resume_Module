"""Lock the numeric claims made in the compact judgment table.

The table is a document; these tests are the check that it matches the data. Without
them a later fixture edit or classifier change could silently make the prose wrong,
which is exactly how the earlier "8 agree" figure survived a packet build.

Every claim asserted here is one the table states in words.
"""

from __future__ import annotations

import json
from pathlib import Path

import decision_contract as dc
from experiment.cases_v2 import SOURCES
from experiment.judgments import classify

ROOT = Path(__file__).resolve().parents[1]
RETAINED = ROOT / "results/comparison-v2b-20261109/attempts.json"

APPROVE_CASES = {"faithful_paraphrase", "supported_method", "transferable_skill"}


def _verdicts() -> dict[tuple[str, str], str]:
    attempts = json.loads(RETAINED.read_text())
    out: dict[tuple[str, str], str] = {}
    for attempt in attempts:
        raw = attempt.get("provider_response") or ""
        if not raw or raw.lstrip().startswith("{"):
            continue
        decision = dc.decide_from_text(raw, attempt["draft_after_trim"], SOURCES)
        out[(attempt["case_id"], attempt["cell"])] = classify(
            attempt["case_id"], decision)["verdict"]
    return out


def test_the_totals_match_the_table():
    verdicts = _verdicts()
    assert len(verdicts) == 14
    counts: dict[str, int] = {}
    for verdict in verdicts.values():
        counts[verdict] = counts.get(verdict, 0) + 1
    assert counts == {
        "detected": 5,
        "detected_with_a_miscitation": 2,
        "not_detected": 1,
        "invented_objection": 6,
    }, counts


def test_every_invented_objection_is_a_faithful_document():
    """The table claims all 6 invented objections are on the 3 clean drafts."""
    invented = {key for key, verdict in _verdicts().items()
                if verdict == "invented_objection"}
    assert {case_id for case_id, _ in invented} == APPROVE_CASES
    # ...in BOTH arms, not just one.
    assert {cell for _, cell in invented} == {"revised+advisory", "revised+enabled"}


def test_the_single_miss_is_the_advisory_missing_fact_cell():
    verdicts = _verdicts()
    misses = {key for key, verdict in verdicts.items() if verdict == "not_detected"}
    assert misses == {("missing_fact", "revised+advisory")}
    # The table says it is not systematic: the enabled arm catches it.
    assert verdicts[("missing_fact", "revised+enabled")] == "detected"


def test_the_present_overlap_is_sourced_not_invented():
    """The table's key claim: the reviewer invents the timeline conflict.

    Each source states its OWN role as `Present`. Neither source asserts that
    concurrent employment is impossible, and the draft introduces no date the
    sources lack — so an "impossible overlap" objection cannot be grounded.
    """
    main = SOURCES["candidate-main.txt"]
    supplement = SOURCES["candidate-supplement.txt"]
    assert "04/2019 - Present" in main
    assert "08/2023 - Present" in supplement
    # No source says anything about overlap, conflict or impossibility.
    both = (main + supplement).lower()
    for word in ("overlap", "concurrent", "impossible", "conflict"):
        assert word not in both, f"source unexpectedly says {word!r}"


def test_the_miscitation_cells_really_quote_absent_text():
    """The table says Ridgeline is quoted as present while absent in fact."""
    attempts = json.loads(RETAINED.read_text())
    for cell in ("revised+advisory", "revised+enabled"):
        attempt = next(a for a in attempts
                       if a["case_id"] == "omitted_record" and a["cell"] == cell)
        draft = attempt["draft_after_trim"]
        assert "Ridgeline" not in draft
        assert "Inventory Analyst" not in draft
        # ...yet the reviewer quotes it.
        assert "Ridgeline" in attempt["provider_response"]


def test_the_missing_figure_is_absent_from_one_arm_only():
    attempts = json.loads(RETAINED.read_text())
    advisory = next(a for a in attempts if a["case_id"] == "missing_fact"
                    and a["cell"] == "revised+advisory")
    enabled = next(a for a in attempts if a["case_id"] == "missing_fact"
                   and a["cell"] == "revised+enabled")
    assert "1,180" not in advisory["provider_response"]
    assert "1,180" in enabled["provider_response"]
