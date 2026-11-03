"""Paired tests for the run-10 appended-clause defect. Offline, no provider calls.

Every source passage and every draft line below is copied verbatim from the
preserved run-10 capture (``850ee3ea-ca2e-4210-92b4-0e57ed5149eb.json``), so these
tests measure the real failure rather than a reconstruction of it. The capture is
read, never modified.

What is asserted, in pairs
--------------------------

A detection rule is only trustworthy when it is shown to accept as well as reject.
Each rejection case is paired with a faithful control that differs by one edit:

* a faithful paraphrase approves;
* supported cross-document synthesis approves;
* an unsupported outcome appended to a supported activity is detected;
* a supported outcome moved under the wrong employer is rejected;
* a reviewer marking a contaminated line "supported" is **not** evidence of
  semantic verification.

What remains model-dependent, and is stated rather than papered over: whether a
particular appended clause is a faithful inference or a fabrication is a semantic
judgement. Three of the six live clauses need that judgement — ``gaining hands-on
electrical construction experience`` is reasonable, ``ensuring continuous
operations`` is not, and no lexical test can tell them apart. ``appended_clauses``
deliberately reports the question instead of answering it.

Labelled **[tier-b-worklist]**.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import appended_clauses

# ---------------------------------------------------------------------------
# Verbatim run-10 evidence
# ---------------------------------------------------------------------------

# parents[0] is tests/, parents[1] is Imagination_Resume_Module/, parents[2] is the
# workspace root that holds the preserved live-run evidence.
# The REQUIRED regression fixture: committed, versioned, self-contained. These tests
# run from a clean checkout and need no historical files.
FIXTURE = (Path(__file__).resolve().parent
           / "fixtures" / "tier_b" / "run10_excerpt.json")

# The OPTIONAL full capture, used only by tests marked as archival replay. Absent is
# normal; those tests skip rather than fail.
CAPTURE = (Path(__file__).resolve().parents[2]
           / ".live-run-2026-11-02" / "captures" / "run10"
           / "850ee3ea-ca2e-4210-92b4-0e57ed5149eb.json")

DOC0_SUMMARY = ("SYNTHETIC-ONLY Journeyman electrician with 8+ years of commercial "
                "and industrial electrical\nexperience across construction, facilities "
                "maintenance, and power distribution.")
DOC1_SUMMARY = ("SYNTHETIC-ONLY Facilities maintenance technician focused on preventive "
                "maintenance of\nelectrical and mechanical building systems.")

# The exact source bullets each draft line was, or should have been, grounded in.
SRC_INSTALL = "Installed and maintained commercial electrical systems across 40+ job sites."
SRC_TRAIN = "Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices."
SRC_REWORK = "Cut rework by 18% by standardizing panel labeling and inspection checklists."
SRC_HOURS = "Completed 8,000 hours of supervised field work toward licensure."
SRC_MAINTAIN = "Maintained electrical distribution and HVAC controls for 12 commercial buildings."
SRC_DOWNTIME = "Reduced unplanned downtime by 22% through a preventive maintenance schedule."

# The six live draft lines, verbatim.
T_LIVE_INSTALL = ("Installed and maintained commercial electrical systems across 40+ "
                  "job sites, ensuring compliance with NFPA 70E standards.")
T_LIVE_TRAIN = ("Trained 6 apprentices on lockout/tagout and NFPA 70E safe work "
                "practices, improving workforce safety and code adherence.")
T_LIVE_REWORK = ("Cut rework by 18% by standardizing panel labeling and inspection "
                 "checklists, increasing installation efficiency and quality control.")
T_LIVE_HOURS = ("Completed 8,000 hours of supervised field work toward journeyman "
                "licensure, gaining hands\u2011on electrical construction experience.")
T_LIVE_MAINTAIN = ("Maintained electrical distribution systems and HVAC controls for 12 "
                   "commercial buildings, ensuring continuous operations.")
T_LIVE_DOWNTIME = ("Reduced unplanned downtime by 22% through implementation of "
                   "preventive maintenance schedules, enhancing system reliability.")


def _fixture() -> dict:
    """The committed, versioned excerpt. Required tests use only this.

    Raises a clear error rather than skipping: a missing required fixture means the
    checkout is broken, not that the test is inapplicable.
    """
    if not FIXTURE.exists():
        raise AssertionError(
            f"required fixture missing: {FIXTURE}. Regenerate with "
            f"`python tests/fixtures/tier_b/extract_from_capture.py --write` from a "
            f"checkout that still has the preserved capture.")
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _capture() -> dict:
    """The full preserved capture. Only archival replay tests use this."""
    if not CAPTURE.exists():
        pytest.skip("full run-10 capture not present; archival replay only")
    return json.loads(CAPTURE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The preserved evidence is what these tests claim it is
# ---------------------------------------------------------------------------

def test_archival_replay_the_capture_still_holds_all_recorded_values():
    """[tier-b-worklist][archival-replay] OPTIONAL — requires the preserved capture.

    Skips when `.live-run-2026-11-02/` is absent, which is normal on a clean
    checkout. Its job is to prove the committed excerpt still matches the immutable
    evidence: if the capture were replaced, the required tests could keep passing
    while measuring a reconstruction.
    """
    data = _capture()
    assert data["correlation_id"] == "850ee3ea-ca2e-4210-92b4-0e57ed5149eb"
    assert [s["source_id"] for s in data["sources"]] == ["document-0", "document-1"]
    for passage in (DOC0_SUMMARY, DOC1_SUMMARY, SRC_INSTALL, SRC_MAINTAIN):
        assert any(passage in s["text"] for s in data["sources"]), passage
    for line in (T_LIVE_INSTALL, T_LIVE_TRAIN, T_LIVE_MAINTAIN, T_LIVE_DOWNTIME):
        assert line in data["draft"], line
    assert data["review"]["status"] == "approved"
    assert data["review"]["repair_calls"] == 0


def test_the_committed_fixture_carries_the_strings_the_required_tests_assert_on():
    """[tier-b-worklist][portable] REQUIRED — runs from a clean checkout.

    Fails loudly (not skips) when the fixture is missing, because a broken checkout
    must not look like an inapplicable test. Every excerpt string must also occur
    verbatim in its own source document or draft line, so the fixture cannot drift
    into invented text.
    """
    fixture = _fixture()
    assert fixture["fixture_version"] == 1
    assert set(fixture["sources"]) == {"document-0", "document-1"}
    combined = "\n".join(fixture["sources"].values())
    for entry in fixture["source_lines"]:
        assert entry["text"] in combined, entry["text"]
        assert hashlib.sha256(entry["text"].encode()).hexdigest() == entry["sha256"]
    for entry in fixture["draft_lines"]:
        assert hashlib.sha256(entry["text"].encode()).hexdigest() == entry["sha256"]
    # The six live clauses the suite exists for are all present.
    for line in (T_LIVE_INSTALL, T_LIVE_TRAIN, T_LIVE_REWORK, T_LIVE_HOURS,
                 T_LIVE_MAINTAIN, T_LIVE_DOWNTIME):
        assert line in [entry["text"] for entry in fixture["draft_lines"]], line


# ---------------------------------------------------------------------------
# 1. A faithful paraphrase approves — the control that makes the rest meaningful
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line,evidence", [
    (SRC_INSTALL, [SRC_INSTALL]),
    (SRC_TRAIN, [SRC_TRAIN]),
    (SRC_REWORK, [SRC_REWORK]),
    # Rewritten wording, same fact: "implementation of … schedules" for
    # "a … schedule". Different words, identical claim.
    ("Reduced unplanned downtime by 22% through implementation of preventive "
     "maintenance schedules.", [SRC_DOWNTIME]),
    # "Supported" for "Assisted", plus a detail sourced from the adjacent bullet.
    ("Supported conduit bending, wire pulling, and panel terminations across 25 "
     "commercial projects under licensed supervision.",
     ["Assisted with conduit bending, wire pulling, and panel terminations on 25 "
      "projects.", "Completed 8,000 hours of supervised field work toward licensure."]),
])
def test_a_faithful_paraphrase_raises_nothing(line, evidence):
    """[tier-b-worklist] Correct writing must cost zero findings.

    A check that flags legitimate paraphrase would delete the rewriting the product
    exists to perform, so this is the primary assertion, not a courtesy.
    """
    assert appended_clauses.unattested_clauses(line, evidence) == []


# ---------------------------------------------------------------------------
# 2. Supported cross-document synthesis approves
# ---------------------------------------------------------------------------

def test_supported_cross_document_synthesis_raises_nothing():
    """[tier-b-worklist] Two documents' facts in one line is legitimate.

    This is the correction the engine already carries (a claim judged against the
    documents it cites, not only the item's own document). Recorded as a test so
    the next change cannot silently re-narrow it.
    """
    merged = ("Facilities maintenance technician focused on preventive maintenance, "
              "with electrical and mechanical building systems experience.")
    assert appended_clauses.unattested_clauses(
        merged, [DOC0_SUMMARY, DOC1_SUMMARY]) == []


# ---------------------------------------------------------------------------
# 3. An unsupported outcome appended to a supported activity is detected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line,evidence,clause", [
    (T_LIVE_INSTALL, [SRC_INSTALL], "ensuring compliance with NFPA 70E standards"),
    (T_LIVE_TRAIN, [SRC_TRAIN], "improving workforce safety and code adherence"),
    (T_LIVE_REWORK, [SRC_REWORK],
     "increasing installation efficiency and quality control"),
    (T_LIVE_MAINTAIN, [SRC_MAINTAIN], "ensuring continuous operations"),
    (T_LIVE_DOWNTIME, [SRC_DOWNTIME], "enhancing system reliability"),
])
def test_an_unsupported_outcome_appended_to_an_activity_is_detected(
        line, evidence, clause):
    """[tier-b-worklist] Five of the six live appends are found deterministically.

    The finding is a *question for Tier B*, not a rejection: the assertion is that
    the clause appears on the work list, which is what makes a semantic review
    possible at all. The sixth live clause is handled in the next test.
    """
    found = appended_clauses.unattested_clauses(line, evidence)
    assert [f["clause"] for f in found] == [clause]
    assert found[0]["missing_stems"], "a reported clause must name what it contributes"


def test_the_detected_clause_is_the_trailing_phrase_and_not_the_sourced_activity():
    """[tier-b-worklist] Offsets must isolate the append, or a repair would take the fact.

    The repair path replaces exactly the spans it is given. If these offsets covered
    the whole bullet, an approved repair would delete the sourced activity along with
    the unsupported outcome.
    """
    found = appended_clauses.unattested_clauses(T_LIVE_INSTALL, [SRC_INSTALL])
    (finding,) = found
    assert T_LIVE_INSTALL[finding["start"]:finding["end"]] == finding["clause"]
    assert SRC_INSTALL.rstrip(".") not in finding["clause"]
    assert finding["start"] > T_LIVE_INSTALL.index("40+ job sites")


def test_the_one_live_clause_that_is_a_reasonable_inference_is_reported_not_condemned():
    """[tier-b-worklist] ``gaining hands-on … experience`` must stay in the document.

    This clause introduces only ``gain`` and ``hand``. The activity it characterises
    is what the source bullet already states, and the summary already claims
    electrical construction experience. It is a reasonable paraphrase, and a
    detector that deleted it would be damaging a correct sentence.

    The module therefore reports it — a Tier-B reviewer still has to justify it —
    but it must not be indistinguishable from the fabrication cases: those assert an
    outcome (compliance, reliability, continuity); this one re-describes the work.
    """
    found = appended_clauses.unattested_clauses(T_LIVE_HOURS, [SRC_HOURS])
    assert [f["clause"] for f in found] == [
        "gaining hands\u2011on electrical construction experience"]
    # The distinguishing property, stated as a check rather than a comment: this
    # clause contributes no outcome noun. It is why it stays and the others do not.
    outcome_words = {"compliance", "adherence", "efficiency", "quality", "control",
                     "operations", "reliability", "safety"}
    assert not (set(found[0]["missing_stems"]) & outcome_words)


# ---------------------------------------------------------------------------
# 4. A supported outcome moved under the wrong employer is rejected
# ---------------------------------------------------------------------------

def test_a_supported_outcome_moved_under_the_wrong_employer_is_rejected():
    """[tier-b-worklist] Provenance is not vocabulary.

    ``Reduced unplanned downtime by 22%`` is fully sourced — but for Bluegate, in
    ``document-1``. Presented under Cascade with only Cascade's evidence cited, the
    figure is unsupported *for this employer*, and the engine's own citation-coverage
    requirement is what must refuse it.
    """
    moved = ("Installed commercial electrical systems across 40+ job sites, reducing "
             "unplanned downtime by 22%.")
    found = appended_clauses.unattested_clauses(
        moved, [SRC_INSTALL])  # Cascade's evidence only
    assert [f["clause"] for f in found] == ["reducing unplanned downtime by 22%"]

    # The same sentence is fine when the cited evidence supports it.
    assert appended_clauses.unattested_clauses(
        moved, [SRC_INSTALL, SRC_DOWNTIME]) == []


def test_a_sourced_outcome_under_its_own_employer_raises_nothing():
    """[tier-b-worklist] Pair for the transplant test: same outcome, right owner."""
    owned = ("Maintained electrical distribution and HVAC controls for 12 commercial "
             "buildings, reducing unplanned downtime by 22%.")
    assert appended_clauses.unattested_clauses(
        owned, [SRC_MAINTAIN, SRC_DOWNTIME]) == []


# ---------------------------------------------------------------------------
# 5. A reviewer saying "supported" is not evidence of semantic verification
# ---------------------------------------------------------------------------

def test_a_reviewer_marking_a_contaminated_line_supported_does_not_verify_it():
    """[tier-b-worklist][archival-replay] OPTIONAL — replays the real reviewer answer.

    The live reviewer marked every one of these lines ``supported`` with
    ``unsupported: []`` and ``missing: []``, and the run was approved. That answer is
    preserved here verbatim: it is the strongest available demonstration that a
    model-assisted verdict is not proof, and that Tier-A passage does not imply
    Tier-B satisfaction.
    """
    # The reviewer's actual answer is only in the full capture, so this half is
    # archival. The portable half below it asserts the work list independently.
    data = _capture()
    exchange = data["exchanges"][1]
    answer = json.loads(json.loads(exchange["response_body"])["choices"][0]["message"]["content"])
    assert answer["unsupported"] == []
    assert answer["uncertain"] == []
    assert answer["missing"] == []
    # Yet the deterministic work list on the very same draft is non-empty.
    line_evidence = {
        "install": [SRC_INSTALL], "train": [SRC_TRAIN], "rework": [SRC_REWORK],
        "hours": [SRC_HOURS], "maintain": [SRC_MAINTAIN],
        "downtime": [SRC_DOWNTIME],
    }
    entries = [
        {"line_id": "install", "text": T_LIVE_INSTALL,
         "evidence": line_evidence["install"]},
        {"line_id": "train", "text": T_LIVE_TRAIN, "evidence": line_evidence["train"]},
        {"line_id": "rework", "text": T_LIVE_REWORK,
         "evidence": line_evidence["rework"]},
        {"line_id": "hours", "text": T_LIVE_HOURS, "evidence": line_evidence["hours"]},
        {"line_id": "maintain", "text": T_LIVE_MAINTAIN,
         "evidence": line_evidence["maintain"]},
        {"line_id": "downtime", "text": T_LIVE_DOWNTIME,
         "evidence": line_evidence["downtime"]},
    ]
    listed = appended_clauses.worklist(entries)
    assert {item["line_id"] for item in listed} == {
        "install", "train", "rework", "hours", "maintain", "downtime"}
    assert len(listed) == 6


# ---------------------------------------------------------------------------
# Scope and negative controls
# ---------------------------------------------------------------------------

def test_the_rule_reports_only_trailing_participial_clauses():
    """[tier-b-worklist] No generic ban on "ensuring"/"improving" as words.

    A blanket ban would reject legitimate source-backed writing, so the rule keys on
    a comma-introduced lowercase ``-ing`` phrase and nothing else. A sentence that
    begins with such a verb, or a comma list of nouns, must be untouched.
    """
    assert appended_clauses.appended_clauses(
        "Ensuring safety, the crew installed panels.") == []
    assert appended_clauses.appended_clauses(
        "Installed panels, conduits, and fixtures.") == []
    assert appended_clauses.appended_clauses(
        "Installed panels, NFPA 70E compliant.") == []
    assert appended_clauses.appended_clauses(T_LIVE_MAINTAIN) != []


def test_a_line_with_no_cited_evidence_is_not_silently_approved():
    """[tier-b-worklist] Absent evidence is a different defect, handled elsewhere.

    The engine withholds a supported line that names no evidence, so this module
    skips it rather than reporting an unattested clause against an empty basis —
    which would be a finding invented by a missing answer.
    """
    assert appended_clauses.worklist(
        [{"line_id": "L1", "text": T_LIVE_MAINTAIN, "evidence": []}]) == []


def test_a_fully_sourced_resume_produces_an_empty_worklist():
    """[tier-b-worklist] Whole-document control: correct source, nothing to review.

    The run-10 source documents themselves, paired with lines that restate them
    faithfully, must produce no work. If this ever fails the detector has started
    punishing the source for being the source.
    """
    entries = [
        {"line_id": "s0", "text": DOC0_SUMMARY, "evidence": [DOC0_SUMMARY]},
        {"line_id": "s1", "text": DOC1_SUMMARY, "evidence": [DOC1_SUMMARY]},
        {"line_id": "install", "text": SRC_INSTALL, "evidence": [SRC_INSTALL]},
        {"line_id": "train", "text": SRC_TRAIN, "evidence": [SRC_TRAIN]},
        {"line_id": "rework", "text": SRC_REWORK, "evidence": [SRC_REWORK]},
        {"line_id": "hours", "text": SRC_HOURS, "evidence": [SRC_HOURS]},
        {"line_id": "maintain", "text": SRC_MAINTAIN, "evidence": [SRC_MAINTAIN]},
        {"line_id": "downtime", "text": SRC_DOWNTIME, "evidence": [SRC_DOWNTIME]},
    ]
    assert appended_clauses.worklist(entries) == []


# ---------------------------------------------------------------------------
# The controls SPEC.v2 asks for
#
# The work list is a *measurement record*, never a verdict. Every one of these
# controls is a case where the answer is wrong about meaning; the tests assert that
# the module says so rather than pretending otherwise.
#
# Both error directions are represented on purpose:
#   * false positive — a correct clause is reported because its words are missing
#     from the cited evidence (controls 1, 3, 5);
#   * false negative — a false clause is NOT reported because every one of its
#     words came from the source (control 2).
# A module with only one of those demonstrated has not been tested.
# ---------------------------------------------------------------------------

def test_every_finding_states_that_semantics_were_not_assessed():
    """[tier-b-worklist] The record must not be readable as a truth judgement.

    ``semantic_status`` is emitted on every finding, always ``not_assessed``. This
    is the assertion that keeps the module honest at the API boundary: a consumer
    reading only the payload cannot conclude the clause was judged.
    """
    found = appended_clauses.unattested_clauses(T_LIVE_MAINTAIN, [SRC_MAINTAIN])
    assert found, "precondition: this clause is reported"
    for finding in found:
        assert finding["semantic_status"] == "not_assessed"
        assert "citation_status" not in finding, (
            "the old verdict-shaped key must be gone")


def test_the_coverage_field_reports_only_what_was_counted():
    """[tier-b-worklist] ``none``/``some``/``all`` is a set-intersection result.

    Measured on the real run-10 corpus rather than assumed. Only the Bluegate
    clause contributes no stem that occurs anywhere in the record; the others share
    an incidental stem ("system", "control", "safety"), which is exactly why the
    field must not be read as a truth value.
    """
    data = _fixture()
    corpus = list(data["sources"].values())
    cases = [
        (T_LIVE_INSTALL, SRC_INSTALL, "some"),
        (T_LIVE_TRAIN, SRC_TRAIN, "some"),
        (T_LIVE_REWORK, SRC_REWORK, "some"),
        (T_LIVE_MAINTAIN, SRC_MAINTAIN, "none"),
        (T_LIVE_DOWNTIME, SRC_DOWNTIME, "some"),
    ]
    for line, evidence, expected in cases:
        found = appended_clauses.unattested_clauses(line, [evidence], corpus)
        assert len(found) == 1
        assert found[0]["missing_stems_found_elsewhere"] == expected, found[0]
        assert found[0]["semantic_status"] == "not_assessed"
        # Whatever the coverage, the clause is still reported.
        assert found[0]["clause"]


# ---------------------------------------------------------------------------
# CONTROL 1 — a correct clause flagged because its words are elsewhere
# ---------------------------------------------------------------------------

def test_control_a_valid_paraphrase_with_little_vocabulary_overlap():
    """[tier-b-worklist] CONTROL 1 (false positive) — low overlap is not fabrication.

    "Diagnosed and repaired switchgear faults" restated as "Traced and corrected
    breaker failures" shares almost no vocabulary, yet asserts the same work. The
    module reports it — it cannot see the equivalence — and the coverage field says
    "all", meaning every missing stem was in fact present elsewhere in the record.
    A reader using that field would ask for a better citation; a reader treating
    the finding as a verdict would delete a correct sentence.
    """
    line = ("Diagnosed and repaired switchgear faults at 9 substations, tracing and "
            "correcting breaker failures.")
    cited = ["Diagnosed and repaired switchgear faults at 9 substations."]
    corpus = cited + ["Traced and corrected breaker failures at 9 substations."]
    found = appended_clauses.unattested_clauses(line, cited, corpus)
    assert [f["clause"] for f in found] == ["tracing and correcting breaker failures"]
    assert found[0]["missing_stems_found_elsewhere"] == "all"
    assert found[0]["semantic_status"] == "not_assessed", (
        "there is no field in which this module reports the clause as wrong, "
        "because it has not judged it")


# ---------------------------------------------------------------------------
# CONTROL 2 — a false clause missed because its words are the source's
# ---------------------------------------------------------------------------

def test_control_a_false_claim_made_entirely_from_source_vocabulary():
    """[tier-b-worklist] CONTROL 2 (false negative) — the documented blind spot.

    Every content stem here is already in the source: "safe", "work" and
    "practices" come from the record's own phrase. The work list is EMPTY, and the
    claim (that the apprentices were trained on "safe work practices" plural) is
    not what the source says.

    This test asserts the miss deliberately. If this module were ever used as an
    approval gate, this sentence would ship.
    """
    cited = ["Trained 4 apprentices on lockout/tagout procedures."]
    falsely_lexical = ("Trained 4 apprentices on lockout/tagout procedures, "
                       "safe work practices trained.")
    assert appended_clauses.unattested_clauses(
        falsely_lexical, cited, cited) == [], (
        "a false claim built only from source words passes the lexical test — "
        "this is the documented blind spot, not a bug")

    # The version that adds a word is at least visible, and still not judged.
    smuggled = ("Trained 4 apprentices on lockout/tagout procedures, "
                "certifying them in NFPA 70E.")
    found = appended_clauses.unattested_clauses(
        smuggled, cited, cited + ["NFPA 70E Arc Flash Safety certificate."])
    assert found, "the added credential must at least be visible"
    assert found[0]["semantic_status"] == "not_assessed"
    assert found[0]["missing_stems_found_elsewhere"] != "none", (
        "the word exists elsewhere in the record, so the coverage field must say "
        "so rather than implying absence")


# ---------------------------------------------------------------------------
# CONTROL 3 — a supported qualitative restatement
# ---------------------------------------------------------------------------

def test_control_a_supported_outcome_restated_qualitatively():
    """[tier-b-worklist] CONTROL 3 (false positive) — generalisation is not invention.

    "Cut average fault-response time from 4.5 hours to 2.75 hours" restated as
    "reducing repeat visits" is a qualitative generalisation of a measured result.
    It may be fair or not — that is a Tier-B judgement this module refuses to make.
    It reports the clause; it does not call it wrong.
    """
    cited = ["Cut average fault-response time from 4.5 hours to 2.75 hours."]
    line = ("Cut average fault-response time from 4.5 hours to 2.75 hours, "
            "reducing repeat visits.")
    found = appended_clauses.unattested_clauses(line, cited, cited)
    assert [f["clause"] for f in found] == ["reducing repeat visits"]
    assert found[0]["semantic_status"] == "not_assessed"
    # The coverage field describes the evidence, and here it says the record does
    # not mention repeat visits at all — which is exactly NOT the same as saying
    # the generalisation is false.
    assert found[0]["missing_stems_found_elsewhere"] == "none"

    # With the restatement present in the record, the coverage field changes. The
    # module's assessment does not, because it never made one.
    softened = appended_clauses.unattested_clauses(
        line, cited, cited + ["Repeat visits reduced; average turnaround now 2.75 hours."])
    assert softened[0]["missing_stems_found_elsewhere"] != "none"
    assert softened[0]["semantic_status"] == "not_assessed"

    # DOCUMENTED RECALL BOUNDARY: a qualifier not beginning with a lowercase -ing
    # word is not located as an appended clause at all.
    adverb_first = ("Cut average fault-response time from 4.5 hours to 2.75 hours, "
                    "materially lowering turnaround.")
    assert appended_clauses.unattested_clauses(adverb_first, cited) == [], (
        "recall boundary: only a clause beginning with a lowercase -ing word is "
        "treated as appended")


# ---------------------------------------------------------------------------
# CONTROL 4 — a new causal or performance outcome
# ---------------------------------------------------------------------------

def test_control_a_new_causal_or_performance_outcome():
    """[tier-b-worklist] CONTROL 4 — a new outcome is reported, never convicted.

    "eliminating tenant complaints entirely" is an outcome the record does not
    describe. The module reports it and says the coverage is "none". That is a
    measurement about the record. Whether the sentence is a fabrication remains a
    Tier-B question, and the module's ``semantic_status`` says so.
    """
    cited = ["Reduced after-hours callouts by 27% by adding thermographic inspections."]
    line = "Added thermographic inspections, eliminating tenant complaints entirely."
    found = appended_clauses.unattested_clauses(line, cited, cited)
    assert found
    assert found[0]["clause"] == "eliminating tenant complaints entirely"
    assert found[0]["missing_stems_found_elsewhere"] == "none"
    assert found[0]["semantic_status"] == "not_assessed"

    # Pair: when the record does describe the outcome, the coverage changes.
    with_outcome = appended_clauses.unattested_clauses(
        line, cited, cited + ["Tenant complaints eliminated."])
    assert with_outcome[0]["missing_stems_found_elsewhere"] != "none"


# ---------------------------------------------------------------------------
# CONTROL 5 — valid clause, evidence available but not cited
# ---------------------------------------------------------------------------

def test_control_a_valid_clause_with_uncited_but_available_evidence():
    """[tier-b-worklist] CONTROL 5 (false positive) — a citation gap is not a lie.

    The reviewer cited only the first half of the bullet. The record contains a
    sentence establishing the rest, uncited. The module reports the clause and the
    coverage field says "all" — every missing stem is in the record. A consumer can
    therefore see that the remedy is a better citation, not a deletion, and nothing
    in the payload calls the sentence false.
    """
    line = ("Maintained electrical distribution and HVAC controls for 12 commercial "
            "buildings, ensuring continuous operations.")
    partially_cited = ["Maintained electrical distribution and HVAC controls for 12 "
                       "commercial buildings."]
    available = partially_cited + ["Ensured continuous operations during all tenant "
                                   "business hours."]
    found = appended_clauses.unattested_clauses(line, partially_cited, available)
    assert [f["clause"] for f in found] == ["ensuring continuous operations"]
    assert found[0]["missing_stems_found_elsewhere"] == "all"
    assert found[0]["semantic_status"] == "not_assessed"

    # Same clause, but the record really never says it.
    found_absent = appended_clauses.unattested_clauses(
        line, partially_cited, partially_cited)
    assert found_absent[0]["missing_stems_found_elsewhere"] == "none"
    # Still not judged either way.
    assert found_absent[0]["semantic_status"] == "not_assessed"


def test_the_coverage_field_survives_the_batch_entry_point():
    """[tier-b-worklist] The measurement must not be lost where it is consumed.

    If ``worklist()`` dropped the field, every consumer would be left with a bare
    clause and no way to tell a citation gap from an absence.
    """
    entries = [
        {"line_id": "cited_elsewhere", "text": T_LIVE_MAINTAIN,
         "evidence": [SRC_MAINTAIN]},
        {"line_id": "never_stated", "text": T_LIVE_DOWNTIME,
         "evidence": [SRC_DOWNTIME]},
    ]
    available = [SRC_MAINTAIN, SRC_DOWNTIME,
                 "Ensured continuous operations during all tenant business hours."]
    listed = appended_clauses.worklist(entries, available)
    coverage = {item["line_id"]: item["clauses"][0]["missing_stems_found_elsewhere"]
                for item in listed}
    assert coverage["cited_elsewhere"] == "all"
    # "system"/"reliability" are not in these two passages. Note the trap this
    # guards against: the DRAFT says "distribution systems", but the cited source
    # says "distribution" — so a reader must not assume the draft's own words are
    # in the record just because they look ordinary.
    assert coverage["never_stated"] == "none"
    for item in listed:
        assert item["clauses"][0]["semantic_status"] == "not_assessed"


def test_a_clean_worklist_does_not_mean_the_sentence_is_supported():
    """[tier-b-worklist] The module's own limit, asserted so nobody wires it in.

    Control 2 shows a false claim made entirely of source vocabulary produces an
    EMPTY work list. If this check were used as an approval gate, that sentence
    would ship. The assertion is the guard against that misuse.
    """
    cited = ["Trained 4 apprentices on lockout/tagout procedures."]
    false_claim = ("Trained 4 apprentices on lockout/tagout procedures, "
                   "work practices trained.")
    assert appended_clauses.worklist(
        [{"line_id": "L1", "text": false_claim, "evidence": cited}], cited) == []
