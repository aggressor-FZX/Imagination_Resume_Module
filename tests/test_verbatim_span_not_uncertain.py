"""A verbatim source span cannot be an unsupported addition.

Measured on Conservative website run ``d0b19921`` (2026-11-07, revision
``3778d0c``). The reviewer marked the supplement's summary line ``uncertain``
even though it is copied character-for-character from the source:

  source  "SYNTHETIC-ONLY Operations analyst supporting stock planning and parts
           availability for a regional equipment service network."
  draft   "Operations analyst supporting stock planning and parts availability
           for a regional equipment service network."

The repair then deleted it (the model returned an empty replacement), the
re-review correctly reported the source fact missing, and the run withheld as
``source_completeness_uncertain`` — after destroying a sourced fact.

The correction is narrow: only an exact, whitespace-normalised substring of a
document the claim itself cites is upgraded. These tests pin that and the
refusals that keep it from becoming a blanket override.
"""
import pytest

import source_correction as sc

SOURCE = ("Professional Summary\n"
          "SYNTHETIC-ONLY Operations analyst supporting stock planning and parts "
          "availability for a regional equipment service network.\n"
          "Professional Experience\n"
          "SYNTHETIC-ONLY Maintain parts availability records for 7 service branches.\n")

VERBATIM = ("Operations analyst supporting stock planning and parts availability "
            "for a regional equipment service network.")


ITEM_START = SOURCE.index("SYNTHETIC-ONLY Operations analyst")
ITEM_END = SOURCE.index("\n", ITEM_START) + 1
ITEM_QUOTE = SOURCE[ITEM_START:ITEM_END]


def _review(verdict, text, quote):
    # A "represented" mapping requires a supported claim, so a flagged claim is
    # paired with the matching non-represented status. That is the shape the
    # engine actually receives. ``unsupported`` is not a valid mapping status, so
    # it maps to ``uncertain``.
    status = "represented" if verdict == "supported" else "uncertain"
    mapping = {"source_claim_id": f"doc:{ITEM_START}:{ITEM_END}", "status": status,
               "spans": ([{"line_id": "L1", "start": 0, "end": len(text)}]
                         if status == "represented" else [])}
    return {
        "claims": [{"id": f"L1:0:{len(text)}", "line_id": "L1", "start": 0,
                    "end": len(text), "verdict": verdict,
                    "citations": [{"source_id": "doc", "start": 0,
                                   "end": len(SOURCE), "quote": SOURCE}]}],
        "representations": [mapping],
    }


def _inventory():
    return [{"id": f"doc:{ITEM_START}:{ITEM_END}", "source_id": "doc",
             "start": ITEM_START, "end": ITEM_END, "quote": ITEM_QUOTE,
             "classification": "required"}]


def _validate(verdict, text):
    lines = [{"line_id": "L1", "text": text}]
    return sc._validate_review(_review(verdict, text, SOURCE), lines,
                               {"doc": SOURCE}, _inventory())


def test_a_verbatim_source_span_marked_uncertain_becomes_supported():
    claims, incomplete, missing = _validate("uncertain", VERBATIM)
    assert claims[0]["verdict"] == "supported"
    assert "uncertain" not in incomplete
    assert missing == []


def test_a_verbatim_source_span_marked_unsupported_becomes_supported():
    claims, incomplete, _ = _validate("unsupported", VERBATIM)
    assert claims[0]["verdict"] == "supported"
    assert "unsupported" not in incomplete


def test_an_appended_clause_is_still_uncertain():
    text = VERBATIM + " Ensuring flawless regulatory compliance."
    claims, incomplete, _ = _validate("uncertain", text)
    assert claims[0]["verdict"] == "uncertain"
    assert "uncertain" in incomplete


def test_a_paraphrase_is_still_uncertain():
    text = ("Operations analyst who supports stock planning and parts availability "
            "for a regional equipment service network.")
    claims, incomplete, _ = _validate("uncertain", text)
    assert claims[0]["verdict"] == "uncertain"


def test_an_uncited_whole_source_line_is_upgraded():
    """Measured on run ``9936f4e2``: the reviewer cited nothing at all.

    The span is an ENTIRE source line, so it cannot be an addition. A partial
    span or a paraphrase is not a whole line and is unaffected (see the two
    negative controls below).
    """
    review = _review("uncertain", VERBATIM, SOURCE)
    review["claims"][0]["citations"] = []
    lines = [{"line_id": "L1", "text": VERBATIM}]
    claims, incomplete, _ = sc._validate_review(review, lines, {"doc": SOURCE},
                                                _inventory())
    assert claims[0]["verdict"] == "supported"


def _uncited(verdict, text):
    review = _review(verdict, text, SOURCE)
    review["claims"][0].update(id=f"L1:0:{len(text)}", end=len(text), citations=[])
    lines = [{"line_id": "L1", "text": text}]
    return sc._validate_review(review, lines, {"doc": SOURCE}, _inventory())


def test_an_uncited_partial_span_is_not_upgraded():
    partial = "Operations analyst supporting stock planning"
    claims, _, _ = _uncited("uncertain", partial)
    assert claims[0]["verdict"] == "uncertain"


def test_an_uncited_paraphrase_is_not_upgraded():
    paraphrase = ("Operations analyst who supports stock planning and parts "
                  "availability for a regional equipment service network.")
    claims, _, _ = _uncited("uncertain", paraphrase)
    assert claims[0]["verdict"] == "uncertain"


def test_an_uncited_unsupported_verdict_is_not_upgraded():
    """An ``unsupported`` verdict is a named defect, not a declined confirmation."""
    review = _review("unsupported", VERBATIM, SOURCE)
    review["claims"][0]["citations"] = []
    lines = [{"line_id": "L1", "text": VERBATIM}]
    claims, incomplete, _ = sc._validate_review(review, lines, {"doc": SOURCE},
                                                _inventory())
    assert claims[0]["verdict"] == "unsupported"


def test_a_whole_line_reported_missing_still_withholds():
    """A ``missing`` verdict is a real objection and is never overridden.

    Measured on run ``40957281``: the reviewer said ``missing`` for a line the
    draft carries character-for-character. Overriding that broke six withhold
    controls, including the negative control that proves an omitted fact is never
    silently accepted. A withhold on a reviewer error is safe; a false approval
    is not, so the engine keeps the withhold and reports it.
    """
    review = _review("supported", VERBATIM, SOURCE)
    review["representations"][0].update(status="missing", spans=[])
    lines = [{"line_id": "L1", "text": VERBATIM}]
    claims, incomplete, missing = sc._validate_review(
        review, lines, {"doc": SOURCE}, _inventory())
    assert "missing" in incomplete
    assert missing


def test_a_span_from_an_uncited_document_is_not_upgraded():
    other = "A completely different document with unrelated wording."
    review = _review("uncertain", VERBATIM, SOURCE)
    review["claims"][0]["citations"] = [{"source_id": "other", "start": 0,
                                         "end": len(other), "quote": other}]
    lines = [{"line_id": "L1", "text": VERBATIM}]
    claims, incomplete, _ = sc._validate_review(review, lines,
                                                {"doc": SOURCE, "other": other},
                                                _inventory())
    assert claims[0]["verdict"] == "uncertain"
