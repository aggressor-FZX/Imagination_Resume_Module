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

import json
from pathlib import Path

import pytest

import appended_clauses

# ---------------------------------------------------------------------------
# Verbatim run-10 evidence
# ---------------------------------------------------------------------------

# parents[0] is tests/, parents[1] is Imagination_Resume_Module/, parents[2] is the
# workspace root that holds the preserved live-run evidence.
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


def _capture() -> dict:
    if not CAPTURE.exists():
        pytest.skip("run-10 capture not present; this suite reads preserved evidence")
    return json.loads(CAPTURE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The preserved evidence is what these tests claim it is
# ---------------------------------------------------------------------------

def test_the_capture_still_holds_the_two_source_documents_and_the_approved_draft():
    """[tier-b-worklist] Guard against the fixtures drifting from the real run.

    Without this, the paired cases below could keep passing after the capture was
    replaced, and would be measuring a reconstruction rather than run 10.
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
    assert found[0]["unattested"], "a reported clause must name what it contributes"


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
    assert not (set(found[0]["unattested"]) & outcome_words)


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
    """[tier-b-worklist] Run 10's actual reviewer answer, replayed.

    The live reviewer marked every one of these lines ``supported`` with
    ``unsupported: []`` and ``missing: []``, and the run was approved. That answer is
    preserved here verbatim: it is the strongest available demonstration that a
    model-assisted verdict is not proof, and that Tier-A passage does not imply
    Tier-B satisfaction.
    """
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
# The five controls SPEC.v2 asks for
#
# The work list is a *report*, never a verdict. These controls exist because a
# lexical rule is exactly the kind of check that gets over-trusted: every one of
# them is a case where the deterministic answer is wrong about meaning, and the
# test asserts that the module says so rather than pretending otherwise.
# ---------------------------------------------------------------------------

def test_control_a_valid_paraphrase_with_little_vocabulary_overlap_is_not_called_false():
    """[tier-b-worklist] CONTROL 1 — low overlap is not evidence of fabrication.

    "Diagnosed and repaired switchgear faults" restated as "Traced and corrected
    breaker failures" shares almost no vocabulary, yet asserts the same work. The
    module reports it (it must — it cannot see the equivalence), so the assertion
    is about *how* it is reported: with ``available_evidence`` supplied the clause
    is ``insufficient_citation``, never ``unsupported``, because the meaning IS in
    the source. That label is the difference between "review this" and "this is
    invented".
    """
    line = ("Diagnosed and repaired switchgear faults at 9 substations, tracing and "
            "correcting breaker failures.")
    cited = ["Diagnosed and repaired switchgear faults at 9 substations."]
    # The record establishes the same work in other words, uncited.
    corpus = cited + ["Traced and corrected breaker failures at 9 substations."]
    found = appended_clauses.unattested_clauses(line, cited, corpus)
    assert [f["clause"] for f in found] == ["tracing and correcting breaker failures"]
    # Reported, but NOT as an unsupported claim: every unattested word is in the record.
    assert found[0]["citation_status"] == "insufficient_citation"
    # Critically, the low overlap did not make it "unsupported".
    assert found[0]["citation_status"] != "unsupported"


def test_control_a_false_claim_made_entirely_from_source_vocabulary_is_invisible():
    """[tier-b-worklist] CONTROL 2 — the failure mode a lexical rule cannot see.

    Every word here is in the sources: "trained", "apprentices", "lockout",
    "tagout", "NFPA", "70E", "safe", "work", "practices". The clause shares content
    stems with the cited evidence, so the work list is EMPTY — and the claim is
    false, because the source says 4 apprentices were trained on lockout/tagout,
    not that they were NFPA 70E certified.

    This test asserts the miss deliberately. It is the documented limit of the
    mechanism and the reason the work list must never be wired into the gate as a
    truth validator: a clean work list does not mean the sentence is supported.
    """
    cited = ["Trained 4 apprentices on lockout/tagout procedures."]
    # False claim, and every content stem is already in the source: "safe", "work",
    # "practices" all come from the record's own phrase. The work list is EMPTY.
    fully_lexical = ("Trained 4 apprentices on lockout/tagout procedures, "
                     "safe work practices trained.")
    assert appended_clauses.unattested_clauses(fully_lexical, cited, cited) == [], (
        "a false claim built only from source words passes the lexical test — "
        "this is the documented blind spot, not a bug")

    # The version that DOES add a word is caught, but only as a question. The added
    # word exists elsewhere in the record in an unrelated sense, so even with the
    # corpus supplied the module refuses to call it a fabrication.
    smuggled = ("Trained 4 apprentices on lockout/tagout procedures, "
                "certifying them in NFPA 70E.")
    found = appended_clauses.unattested_clauses(
        smuggled, cited, cited + ["NFPA 70E Arc Flash Safety certificate."])
    assert found, "the added credential must at least be visible"
    assert found[0]["citation_status"] != "unsupported", (
        "a false claim whose new word appears elsewhere in the record is NOT "
        "classifiable lexically; the module must not claim that it is")


def test_control_a_supported_outcome_restated_qualitatively_is_still_reported():
    """[tier-b-worklist] CONTROL 3 — qualitative restatement is a Tier-B question.

    "Cut average fault-response time from 4.5 hours to 2.75 hours" restated as
    "materially faster fault response" is supported by the source and cannot be
    dismissed on vocabulary grounds alone. The module reports it, because the
    qualitative claim is broader than the measured one — a human must decide
    whether the generalisation is fair. Reporting it is not an accusation.
    """
    cited = ["Cut average fault-response time from 4.5 hours to 2.75 hours."]
    line = ("Cut average fault-response time from 4.5 hours to 2.75 hours, "
            "reducing repeat visits.")
    found = appended_clauses.unattested_clauses(line, cited, cited)
    assert [f["clause"] for f in found] == ["reducing repeat visits"]
    # The module cannot know that 4.5h -> 2.75h justifies "fewer repeat visits". It
    # reports the clause and stops. Calling this a fabrication would be wrong: the
    # generalisation may be entirely fair. It is a Tier-B question.
    assert found[0]["citation_status"] == "unsupported", (
        "reported, but the label describes the EVIDENCE (nothing about repeat "
        "visits is in the record), not whether the generalisation is fair")
    # With the restatement present in the record, the label softens to a citation issue.
    softened = appended_clauses.unattested_clauses(
        line, cited, cited + ["Repeat visits reduced; average turnaround now 2.75 hours."])
    assert softened[0]["citation_status"] != "unsupported"

    # DOCUMENTED BOUNDARY: a qualifier that does not start with a gerund is not
    # located as an appended clause at all. "materially lowering turnaround" is
    # invisible, where "lowering turnaround" is caught. Asserted rather than left
    # to be discovered, because it bounds what the work list can claim to cover.
    adverb_first = ("Cut average fault-response time from 4.5 hours to 2.75 hours, "
                    "materially lowering turnaround.")
    assert appended_clauses.unattested_clauses(adverb_first, cited) == [], (
        "recall boundary: only a clause beginning with a lowercase -ing word is "
        "treated as appended")

    # And with no corpus supplied, the module refuses to emit a status at all.
    assert "citation_status" not in appended_clauses.unattested_clauses(line, cited)[0]


def test_control_an_unsupported_causal_outcome_is_reported_as_unsupported():
    """[tier-b-worklist] CONTROL 4 — an invented causal chain is what Tier B must stop.

    "by adding thermographic inspections" is real. "cutting after-hours callouts by
    27%" is real. Joining them as cause and effect, for an employer where that
    inspection programme is not described, invents a causal claim. No word is new
    in isolation; the *relation* is.
    """
    cited = ["Reduced after-hours callouts by 27% by adding thermographic inspections."]
    line = ("Added thermographic inspections, eliminating tenant complaints entirely.")
    found = appended_clauses.unattested_clauses(line, cited, cited)
    assert found, "the appended causal outcome must be visible"
    assert found[0]["clause"] == "eliminating tenant complaints entirely"
    # No part of this outcome is in the record: it is the fabrication question, and
    # it is exactly the shape of run 10's "ensuring continuous operations".
    assert found[0]["citation_status"] == "unsupported"
    # Pair: if the record DOES describe the outcome, the label is no longer that.
    with_outcome = appended_clauses.unattested_clauses(
        line, cited, cited + ["Tenant complaints eliminated."])
    assert with_outcome[0]["citation_status"] != "unsupported"


def test_control_a_valid_clause_with_uncited_but_available_evidence_is_a_citation_defect():
    """[tier-b-worklist] CONTROL 5 — never delete a correct statement for a lazy citation.

    This is the case SPEC.v2 singles out. A supported sentence whose evidence exists
    in the record but which the reviewer failed to cite must be labelled
    ``insufficient_citation``. Treating it as ``unsupported`` would delete a correct
    statement because of a reviewer's omission — the opposite of the intended
    behaviour.
    """
    line = ("Maintained electrical distribution and HVAC controls for 12 commercial "
            "buildings, ensuring continuous operations.")
    # The reviewer cited only the first half of the bullet.
    partially_cited = ["Maintained electrical distribution and HVAC controls for 12 "
                       "commercial buildings."]
    # The record also contains the sentence that establishes continuity, uncited.
    available = partially_cited + ["Ensured continuous operations during all tenant "
                                   "business hours."]
    found = appended_clauses.unattested_clauses(line, partially_cited, available)
    assert [f["clause"] for f in found] == ["ensuring continuous operations"]
    assert found[0]["citation_status"] == "insufficient_citation", (
        "a clause attested in the record but not cited is a citation defect; "
        "the statement may be perfectly correct")

    # Same clause, but the record really never says it: now it is the other case.
    found_absent = appended_clauses.unattested_clauses(
        line, partially_cited, partially_cited)
    assert found_absent[0]["citation_status"] == "unsupported"


def test_insufficient_citation_and_unsupported_are_distinguishable_in_a_worklist():
    """[tier-b-worklist] The distinction must survive the batch entry point.

    A consumer needs to act differently on the two: re-request evidence for one,
    review meaning for the other. If ``worklist()`` dropped the label, that
    information would be lost at the only place it is consumed.
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
    # The batch entry point must carry the label through; without it a consumer could
    # not tell a citation gap from a meaning gap at the only place it consumes them.
    assert all("citation_status" in item["clauses"][0] for item in listed)
    status = {item["line_id"]: item["clauses"][0]["citation_status"] for item in listed}
    # The continuity clause is a pure citation defect here: the record contains every
    # one of its words, in a sentence about continuous operations.
    assert status["cited_elsewhere"] == "insufficient_citation"
    # The reliability clause has no such sentence available; it stays unresolved.
    assert status["never_stated"] == "unsupported"
    assert status["cited_elsewhere"] != status["never_stated"]

    # Pair: supply the sentence for the second line too, and it becomes a citation
    # defect as well — proving the label tracks the record, not the line's identity.
    available_both = available + ["Enhanced system reliability through scheduled "
                                 "maintenance."]
    listed_both = appended_clauses.worklist(entries, available_both)
    status_both = {item["line_id"]: item["clauses"][0]["citation_status"]
                   for item in listed_both}
    assert status_both["never_stated"] != "unsupported"


# ---------------------------------------------------------------------------
# The run-10 appends, labelled
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line,evidence,expected", [
    # Measured on the real corpus, not assumed. Only the Bluegate bullet is
    # wholly absent from the record; the others share an incidental stem, which is
    # exactly why a binary label would have been misleading.
    (T_LIVE_INSTALL, [SRC_INSTALL], "partially_attested"),
    (T_LIVE_TRAIN, [SRC_TRAIN], "partially_attested"),
    (T_LIVE_REWORK, [SRC_REWORK], "partially_attested"),
    (T_LIVE_MAINTAIN, [SRC_MAINTAIN], "unsupported"),
    (T_LIVE_DOWNTIME, [SRC_DOWNTIME], "partially_attested"),
])
def test_the_live_appends_are_labelled_by_what_the_record_does_and_does_not_contain(
        line, evidence, expected):
    """[tier-b-worklist] Run 10's five findings, split by what the record contains.

    ``unsupported`` means no part of the appended clause occurs anywhere in either
    document — true of ``ensuring continuous operations``. ``partially_attested``
    means some fragment does occur, in an unrelated sense, so the label must NOT be
    upgraded to a clean citation defect.

    This is the measurement that killed the simpler two-way split I first wrote.
    ``enhancing system reliability`` shares only ``system`` with the corpus; a
    binary rule called that a mere citation problem, which would have told a
    reviewer to go hunting for a citation that does not exist.
    """
    data = _capture()
    corpus = [source["text"] for source in data["sources"]]
    found = appended_clauses.unattested_clauses(line, evidence, corpus)
    assert len(found) == 1
    assert found[0]["citation_status"] == expected, found[0]
    # Whatever the label, the clause is reported: the label never suppresses it.
    assert found[0]["clause"]


def test_a_clean_worklist_does_not_mean_the_sentence_is_supported():
    """[tier-b-worklist] The module's own limit, asserted so nobody wires it into the gate.

    Control 2 shows a false claim made entirely of source vocabulary produces an
    EMPTY work list. If this check were used as an approval gate, that sentence
    would ship. The assertion is the guard against that misuse.
    """
    cited = ["Trained 4 apprentices on lockout/tagout procedures."]
    false_claim = ("Trained 4 apprentices on lockout/tagout procedures, "
                   "work practices trained.")
    assert appended_clauses.worklist(
        [{"line_id": "L1", "text": false_claim, "evidence": cited}], cited) == []
