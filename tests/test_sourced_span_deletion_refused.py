"""A repair may not delete the candidate's own verbatim text.

Measured on Conservative website run ``c0b26422`` (2026-11-07, revision
``b7fbecf``). The reviewer marked the supplement's summary line ``uncertain``
with no citations:

  "Operations analyst supporting stock planning and parts availability for a
   regional equipment service network."

That line is copied character-for-character from the source. The repair returned
an empty replacement, so the engine deleted it; the re-review then correctly
reported the source fact missing and the run withheld as ``no_op_repair`` /
``source_completeness_uncertain`` — after the fact was already gone.

The refusal is narrow: only an ``uncertain`` verdict, only an empty replacement,
and only when the span is verbatim in a source document. An ``unsupported``
verdict still permits deletion, because that is the legitimate repair for a
misattributed line.
"""
import pytest

import source_correction as sc

SOURCE = ("Professional Summary\n"
          "SYNTHETIC-ONLY Operations analyst supporting stock planning and parts "
          "availability for a regional equipment service network.\n")

VERBATIM = ("Operations analyst supporting stock planning and parts availability "
            "for a regional equipment service network.")


KEEP = "\n## Education\n**Example College** -- City, ST\n"


def _lines(text):
    return [{"line_id": "L1", "text": text, "absolute_start": 0},
            {"line_id": "L2", "text": KEEP, "absolute_start": len(text)}]


def _claims(verdict, text):
    return [{"id": f"L1:0:{len(text)}", "line_id": "L1", "start": 0,
             "end": len(text), "verdict": verdict, "citations": []}]


def _apply(verdict, replacement, text=VERBATIM):
    return sc._apply_edits(
        text + KEEP, {"edits": [{"claim_id": f"L1:0:{len(text)}",
                                 "replacement": replacement}]},
        _claims(verdict, text), _lines(text), source_map={"doc": SOURCE})


def test_deleting_a_verbatim_span_marked_uncertain_is_refused():
    with pytest.raises(sc.SourcedSpanDeletion):
        _apply("uncertain", "")


def test_deleting_a_verbatim_span_marked_unsupported_is_allowed():
    assert _apply("unsupported", "") == KEEP


def test_replacing_a_verbatim_span_with_source_wording_is_allowed():
    # A non-empty replacement is a correction, not a deletion.
    out = _apply("uncertain", "Operations analyst supporting stock planning.")
    assert "stock planning" in out


def test_deleting_a_paraphrase_marked_uncertain_is_allowed():
    paraphrase = ("Operations analyst who supports stock planning and parts "
                  "availability for a regional equipment service network.")
    assert _apply("uncertain", "", text=paraphrase) == KEEP


def test_deleting_an_appended_clause_marked_uncertain_is_allowed():
    text = VERBATIM + " Ensuring flawless regulatory compliance."
    assert _apply("uncertain", "", text=text) == KEEP


def test_without_a_source_map_the_old_behaviour_is_unchanged():
    text = VERBATIM
    out = sc._apply_edits(
        text + KEEP,
        {"edits": [{"claim_id": f"L1:0:{len(text)}", "replacement": ""}]},
        _claims("uncertain", text), _lines(text))
    assert out == KEEP
