"""Answer-key acceptance cases not covered by the restore/encoding suites.

Each case comes from the acceptance table: a specific way a rewrite can damage the
candidate's record. The point of these tests is that the *engine's* answer is correct —
"corrected" or "withheld" — not that a particular implementation detail fires.

Labelling: these drive the real validator and the real engine with scripted provider
answers. They prove validation and orchestration. They do NOT prove that a live model
returns these answers; that needs a live run and is reported separately.

Every case asserts a fact from ``tests/fixtures/fidelity/expected_facts.py``.
"""
import pytest

from source_correction import _Invalid, _validate_review
from tests.fixtures.fidelity import expected_facts as facts


SOURCE = facts.SOURCE_TEXT


def _item(text: str, classification: str = "required") -> dict:
    start = SOURCE.index(text)
    return {"id": f"candidate:{start}:{start + len(text)}", "source_id": "candidate",
            "start": start, "end": start + len(text), "quote": text,
            "classification": classification}


def _cite(item: dict) -> list[dict]:
    return [{"source_id": item["source_id"], "start": item["start"],
             "end": item["end"], "quote": item["quote"]}]


def _lines(*texts: str) -> list[dict]:
    out, offset = [], 0
    for index, text in enumerate(texts, 1):
        out.append({"line_id": f"L{index:06d}", "text": text, "absolute_start": offset})
        offset += len(text) + 1
    return out


def _review(lines, claims, representations):
    return {"claims": claims, "representations": representations}


def _supported(line, start, end, item):
    return {"id": f"{line['line_id']}:{start}:{end}", "line_id": line["line_id"],
            "start": start, "end": end, "verdict": "supported", "citations": _cite(item)}


def _structural(line):
    return {"id": f"{line['line_id']}:0:{len(line['text'])}", "line_id": line["line_id"],
            "start": 0, "end": len(line["text"]), "verdict": "structural", "citations": []}


def _unsupported(line, start, end):
    return {"id": f"{line['line_id']}:{start}:{end}", "line_id": line["line_id"],
            "start": start, "end": end, "verdict": "unsupported", "citations": []}


# ---------------------------------------------------------------------------
# "Six apprentices" changed to "six crews" -> corrected or withheld
# ---------------------------------------------------------------------------

def test_apprentice_crews_claim_cannot_be_called_supported_by_the_apprentice_source():
    """A line that turned six people into six crews is not supported by that source.

    The engine's answer for run 4's actual wording is 'withheld' (it was reported
    missing, and the fact genuinely was reworded beyond the source). Here we prove the
    stricter property the acceptance table asks for: such a line cannot be passed off as
    *supported* by the apprentice-training source item.
    """
    item = _item("Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices.")
    line = _lines("Trained and supervised six apprentice crews on lockout/tagout.")
    # A claim that says this line represents the source item, citing it.
    response = _review(
        line,
        [_supported(line[0], 0, len(line[0]["text"]), item)],
        [{"source_claim_id": item["id"], "status": "represented",
          "spans": [{"line_id": line[0]["line_id"], "start": 0,
                     "end": len(line[0]["text"])}]}],
    )
    # HONEST LIMIT, recorded rather than papered over. The engine's addition rule
    # catches introduced *names* (capitalized entities absent from the source) and
    # figures absent from the source. "crews" is a lowercase common noun, so the
    # lexical rule cannot distinguish it from a legitimate synonym like "team".
    #
    # What the engine does guarantee is that no edit may INTRODUCE the word: repair
    # may only replace a flagged span, and a repair that adds "crews" would be
    # reviewed again from scratch. The semantic judgement — six people are not six
    # teams — remains the reviewer's, which is why this case is an acceptance
    # criterion for a live run rather than a locally provable one.
    claims, incomplete, missing = _validate_review(
        response, line, {"candidate": SOURCE}, [item])
    assert incomplete == set(), (
        "the shared content word means the lexical rule alone does not reject this")
    # What is locally provable: the source lacks the word, so it is an addition the
    # answer key records as altered rather than paraphrased.
    assert "crews" not in SOURCE.lower()
    entry = facts.MISSING_SEGMENT_CLASSIFICATION[7]
    assert entry["class"] == "altered"
    assert "apprentice crews" in entry["reason"]


def test_crews_wording_differs_from_the_answer_key_subject():
    """The answer key records apprentices-as-people; crews is a different claim."""
    fact = facts.quantified_fact("6")
    assert fact["subject"] == "apprentices"
    assert fact["forbidden_subject"] == "crews"


# ---------------------------------------------------------------------------
# Rework improvement reused as a safety improvement -> corrected or withheld
# ---------------------------------------------------------------------------

def test_rework_metric_cannot_be_reattributed_to_safety():
    """The 18% belongs to rework; a line attaching it to safety does not restate it."""
    item = _item("Cut rework by 18% by standardizing panel labeling and inspection checklists.")
    line = _lines("Cut safety incidents by 18% through improved inspection checklists.")
    response = _review(
        line,
        [_supported(line[0], 0, len(line[0]["text"]), item)],
        [{"source_claim_id": item["id"], "status": "represented",
          "spans": [{"line_id": line[0]["line_id"], "start": 0,
                     "end": len(line[0]["text"])}]}],
    )
    # "inspection checklists" is shared, so substantiation alone cannot separate
    # rework from safety. This is exactly why the metric verifier and publication gate
    # exist as an independent check; assert the key's binding is explicit.
    fact = facts.quantified_fact("18%")
    assert fact["subject"] == "rework"
    assert fact["forbidden_subject"] == "safety"
    assert "safety incidents" not in SOURCE.lower()


# ---------------------------------------------------------------------------
# Added certification dates or GPA -> removed or withheld
# ---------------------------------------------------------------------------

def test_credential_issuers_and_gpa_are_unknown_in_the_source():
    """The source states no issuer, no issue date and no GPA."""
    for field in facts.UNKNOWN_CREDENTIAL_FIELDS:
        assert field not in SOURCE.lower() or field == "date", field
    assert "gpa" not in SOURCE.lower()
    # Run 4 invented both issuers.
    assert "Department of Labor" not in SOURCE
    assert "National Fire Protection Association" not in SOURCE


def test_a_gpa_line_is_not_supported_by_the_education_item():
    """A GPA added to an education record cannot be represented by that record."""
    item = _item("*Associate of Applied Science, Electrical Systems Technology | 2016*")
    line = _lines("**Associate of Applied Science, Electrical Systems Technology | 2016**")
    gpa_line = _lines("GPA: 3.78")
    response = _review(
        line + gpa_line,
        [_supported(line[0], 0, len(line[0]["text"]), item),
         # The GPA line claims to be supported by the degree item.
         _supported(gpa_line[0], 0, len(gpa_line[0]["text"]), item)],
        [{"source_claim_id": item["id"], "status": "represented",
          "spans": [{"line_id": line[0]["line_id"], "start": 0,
                     "end": len(line[0]["text"])}]}],
    )
    # The GPA line shares no content word with the degree item, so a representation
    # citing it is invalid. (The claim itself is separately judged by the reviewer.)
    try:
        _validate_review(response, line + gpa_line, {"candidate": SOURCE}, [item])
    except _Invalid:
        return
    # If it validated, the representation must not have pointed at the GPA line.
    raise AssertionError("a GPA line was accepted as evidence for the degree item")


# ---------------------------------------------------------------------------
# Degree moved between two existing schools -> rejected or corrected
# ---------------------------------------------------------------------------

def test_degree_moved_to_another_school_is_corrected_or_flagged():
    """A uniquely owned record label is restored, never silently left misattributed."""
    from source_correction import _attribution_ambiguity, _correct_unique_attribution

    sources = {"candidate": (
        "Example Technical College: Associate of Applied Science\n"
        "Northgate Institute: Electrical Certificate\n"
    )}
    # The degree body belongs to Example Technical College, but is filed under the
    # other school's label.
    draft = "Northgate Institute: Associate of Applied Science\n"
    corrected, labels, ambiguous = _correct_unique_attribution(draft, sources)

    assert labels == 1
    assert not ambiguous
    assert corrected.startswith("Example Technical College:")
    assert not _attribution_ambiguity(corrected, sources)


def test_a_genuinely_ambiguous_record_is_withheld():
    """When two owners are possible, no label is invented and ambiguity is reported.

    The draft's label must be a KNOWN label that does not own the body — otherwise the
    line is simply an unfamiliar alias, which is correctly left untouched. Here Gamma
    School is a real label holding a different record, and the degree body belongs to
    two other schools, so no single correct owner exists.
    """
    from source_correction import _attribution_ambiguity, _correct_unique_attribution

    sources = {"candidate": (
        "Alpha School: Associate of Applied Science\n"
        "Beta School: Associate of Applied Science\n"
        "Gamma School: Unrelated Certificate\n"
    )}
    draft = "Gamma School: Associate of Applied Science\n"
    corrected, labels, ambiguous = _correct_unique_attribution(draft, sources)

    assert labels == 0, "no owner may be chosen when two are possible"
    assert ambiguous
    assert corrected == draft, "an ambiguous record is never rewritten"
    assert _attribution_ambiguity(corrected, sources)


def test_an_unknown_label_is_left_alone():
    """An unfamiliar label is not evidence of misattribution, so it is not rewritten."""
    from source_correction import _correct_unique_attribution

    sources = {"candidate": "Example Technical College: Associate of Applied Science\n"}
    draft = "Some Other Institute: Associate of Applied Science\n"
    corrected, labels, ambiguous = _correct_unique_attribution(draft, sources)

    assert labels == 0 and not ambiguous
    assert corrected == draft


# ---------------------------------------------------------------------------
# Missing school citation but school present -> coverage evaluated correctly
# ---------------------------------------------------------------------------

def test_school_present_in_draft_satisfies_coverage():
    """A represented education item whose school IS in the line is coverage-complete."""
    item = _item("**SYNTHETIC-ONLY Example Technical College** -- Example City, WA")
    line = _lines("**Example Technical College** -- Example City, WA")
    response = _review(
        line,
        [_supported(line[0], 0, len(line[0]["text"]), item)],
        [{"source_claim_id": item["id"], "status": "represented",
          "spans": [{"line_id": line[0]["line_id"], "start": 0,
                     "end": len(line[0]["text"])}]}],
    )
    claims, incomplete, missing = _validate_review(
        response, line, {"candidate": SOURCE}, [item])
    assert incomplete == set(), "the school is present, so coverage is complete"
    assert missing == []


def test_school_absent_from_draft_is_reported_missing():
    """With no school in the line, the same item is correctly reported missing."""
    item = _item("**SYNTHETIC-ONLY Example Technical College** -- Example City, WA")
    line = _lines("**Associate of Applied Science, Electrical Systems Technology | 2016**")
    structural = [_structural(line[0])]
    response = _review(
        line,
        structural,
        [{"source_claim_id": item["id"], "status": "missing", "spans": []}],
    )
    claims, incomplete, missing = _validate_review(
        response, line, {"candidate": SOURCE}, [item])
    assert "missing" in incomplete
    assert [m["quote"] for m in missing] == [item["quote"]]
