"""Controlled recovery: does a repaired draft actually get published?

The recordings stop after the initial review. They therefore prove the engine now
*asks* for a repair, but not what becomes of the answer. This module supplies
separately labelled test-authored answers for the phases the recordings never reached
and drives the real engine, real adapter, real validator and real publication gate
through them.

What a controlled answer can and cannot prove
---------------------------------------------
A controlled answer is written by the same person who wrote the answer key, so it
cannot independently corroborate that key. What these runs establish is the
*mechanism*:

* an unsupported addition removed by repair is genuinely gone from the published text;
* a supported fact that was already correct survives the repair untouched;
* a missing required field can be restored, and only onto the record it belongs to;
* a draft that still carries a defect is still rejected after the repair;
* the repair path cannot delete an accomplishment, move a fact between employers,
  invent a qualification, or exceed its call budget.

They do not show that a real model would answer this way, and are never reported as
live results.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

import correction_provider
import live_capture
from source_correction import SourceDocument, correct
from tests.support import controlled_answers as ca

RUN7 = Path("/home/skystarved/Render_Dockers/.live-run-2026-11-02/captures/"
            "run7/0a267e04-affd-4cee-b944-26c574703039.json")

requires_run7 = pytest.mark.skipif(not RUN7.exists(), reason="run-7 capture absent")


# ---------------------------------------------------------------------------
# One runner, used by every scenario
# ---------------------------------------------------------------------------

class Scenario:
    """Answers for the phases after the recording ends, in the adapter's wire form.

    The adapter translates the engine's request into an index-only question before it
    reaches the session, so an answer must be in that same index form: line ids,
    segment indices, and the verdict/status lists the strict translator reads. The
    answers are therefore built from the *wire* request, which keeps this test coupled
    to what is actually sent rather than to an assumption about it.
    """

    def __init__(self, review, *, repair=None, review_after=None,
                 review_first_from: str | None = None):
        self._review = review
        self._repair = repair
        self._review_after = review_after
        # Set only when the draft under test differs structurally from the recorded
        # one. The recorded initial review answers *that exact draft*, so it cannot
        # judge an inserted heading and returns an answer the adapter rejects. When a
        # test substitutes a draft, it must also answer the first review.
        self._review_first_from = review_first_from
        self.served: list[str] = []

    def answer(self, phase: str, wire: dict):
        # The wire form does not carry the engine's phase name: a re-review is
        # byte-shaped like any other review, so the detector reports "review" for
        # every one of them. Once the recording is exhausted, any review-shaped
        # request is therefore a re-review — the initial review is always served by
        # the recording itself unless a substituted draft forced it to be replaced.
        if phase in {"initial_review", "post_patch_review", "post_restore_review",
                     "review"}:
            builder = (self._review_after if self._review_after is not None
                       else self._review)
            return builder(wire)
        if phase == "repair" and self._repair is not None:
            return self._repair(wire)
        raise AssertionError(f"unexpected controlled phase {phase!r}")


def _served(body: str):
    """An aiohttp-shaped 200 carrying ``body``, for a controlled answer."""
    encoded = body.encode("utf-8")

    class Served:
        status = 200

        def __init__(self):
            self.content = self

        async def iter_chunked(self, size):
            for index in range(0, len(encoded), size):
                yield encoded[index:index + size]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    return Served()



def drive(scenario: Scenario, *, timeout: float = 60.0, draft: str | None = None):
    """Run the real correction over run-7, answering later phases from the scenario.

    ``draft`` overrides the captured draft. The default is the recorded run-7 writer
    output, which is the evidence. One scenario passes a variant that carries all six
    headings, because the recorded draft is missing a Projects heading — that is a
    *writer* defect at revision b855087, not something the correction engine can fix
    (repair may only replace a flagged span and cannot insert a newline). Testing the
    all-headings invariant against the recorded draft would therefore be testing the
    wrong layer and would fail for a reason that has nothing to do with repair.
    """
    data = live_capture.load(RUN7)
    substituted = draft is not None
    if draft is None:
        draft = data["draft"]
    extension = live_capture.ControlledExtension({})
    extension.prepare(data)

    class Session(live_capture.ExtendedReplaySession):
        def post(self, url, **kwargs):
            exchanges = self.recorded.get("exchanges") or []
            if self.position < len(exchanges):
                # A substituted draft is not the draft the recording reviewed, so its
                # recorded first review does not apply and the adapter would reject it.
                # Answer that phase from the scenario instead, and consume the recorded
                # exchange so ordering stays intact.
                phase_now = self._phase(kwargs)
                if substituted and phase_now == "review":
                    wire = json.loads(kwargs["json"]["messages"][1]["content"])
                    payload = scenario.answer("initial_review", wire)
                    scenario.served.append("initial_review")
                    self.position += 1
                    self.calls += 1
                    self.requests.append(kwargs)
                    self.controlled_requests.append(
                        {"phase": "initial_review",
                         "origin": extension.CONTROLLED})
                    return _served(ca.success(payload))
                return super().post(url, **kwargs)
            phase = self._phase(kwargs)
            wire = json.loads(kwargs["json"]["messages"][1]["content"])
            payload = scenario.answer(phase, wire)
            scenario.served.append(phase)
            self.calls += 1
            self.position += 1
            self.requests.append(kwargs)
            self.controlled_requests.append({"phase": phase,
                                             "origin": extension.CONTROLLED})
            encoded = ca.success(payload).encode("utf-8")

            class Served:
                status = 200

                def __init__(self):
                    self.content = self

                async def iter_chunked(self, size):
                    for index in range(0, len(encoded), size):
                        yield encoded[index:index + size]

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    return False

            return Served()

    session = Session(data, extension)
    previous = os.environ.get(correction_provider.DISCLOSURE_SETTING)
    os.environ[correction_provider.DISCLOSURE_SETTING] = "1"
    try:
        provider = correction_provider.CorrectionProvider("no-network", session)
        sources = tuple(SourceDocument(i["source_id"], i["text"])
                        for i in data["sources"])
        result = asyncio.run(correct(draft, sources, provider,
                                     timeout_seconds=timeout))
    finally:
        if previous is None:
            os.environ.pop(correction_provider.DISCLOSURE_SETTING, None)
        else:
            os.environ[correction_provider.DISCLOSURE_SETTING] = previous
    return result, session, data


# ---------------------------------------------------------------------------
# Wire-form answer builders
# ---------------------------------------------------------------------------

def seg_index(wire: dict, needle: str) -> int:
    found = next((s["index"] for s in wire["segments"] if needle in s["text"]), None)
    if found is None:
        raise AssertionError(f"no wire segment contains {needle!r}")
    return found


def review(supported: dict[str, object] | None = None,
           unsupported: tuple[str, ...] = (),
           *, missing: tuple[str, ...] = ()) -> callable:
    """Build a review answer in the adapter's index form.

    ``supported`` maps a draft-line substring to the source substring(s) that
    establish it — a str or a tuple of str. Multiple needles are allowed because one
    draft line routinely restates two source items (a summary sentence can carry both
    halves of a wrapped paragraph, and a heading line carries the heading segment).

    ``unsupported`` names draft lines the reviewer rejects. ``structural`` is used
    only for a heading the scenario did not map to a source heading, since a heading
    with no source counterpart asserts nothing. Anything else the scenario did not
    vouch for is ``unsupported`` — the conservative reading, which makes these tests
    fail closed.

    ``missing`` names source substrings to report missing regardless of citation, which
    is how "a restoration is still needed" states its requirement explicitly rather
    than relying on an accidental gap.
    """
    supported = supported or {}

    def needles_for(text: str) -> tuple[str, ...]:
        for marker, value in supported.items():
            if marker in text:
                return (value,) if isinstance(value, str) else tuple(value)
        return ()

    def build(wire: dict) -> dict:
        verdicts: dict[str, list[str]] = {"supported": [], "unsupported": [],
                                          "uncertain": [], "structural": []}
        evidence: dict[str, list[int]] = {}
        for line in wire["lines"]:
            text, lid = line["text"], line["line_id"]
            needles = needles_for(text)
            if needles:
                verdicts["supported"].append(lid)
                evidence[lid] = [seg_index(wire, needle) for needle in needles]
                continue
            if any(marker in text for marker in unsupported):
                verdicts["unsupported"].append(lid)
                continue
            if text.lstrip().startswith("#"):
                verdicts["structural"].append(lid)
                continue
            verdicts["unsupported"].append(lid)

        represented, reported_missing = [], []
        cited = {index for indices in evidence.values() for index in indices}
        for segment in wire["segments"]:
            if any(marker in segment["text"] for marker in missing):
                reported_missing.append(segment["index"])
            elif segment["index"] in cited:
                represented.append(segment["index"])
            else:
                reported_missing.append(segment["index"])

        return {**verdicts, "evidence": evidence,
                "represented": represented, "missing": reported_missing}

    return build


def repair(mapping: dict[str, str]) -> callable:
    """Answer a repair: replace the flagged span of each named line.

    ``mapping`` maps a substring of the flagged line to its replacement text. A
    flagged line with no mapping is deleted, which is what the "repair deletes an
    accomplishment" negative control requires.
    """

    def build(wire: dict) -> dict:
        by_line = {line["line_id"]: line["text"] for line in wire["lines"]}
        edits = []
        for claim_id in wire["flagged_claim_ids"]:
            # A claim id is ``{line_id}:{start}:{end}``, so the line id is the FIRST
            # field. Splitting from the right yields "L000002:0", which matches no
            # line and produced a KeyError the engine reported as provider_error.
            lid = claim_id.split(":", 1)[0]
            text = by_line[lid]
            replacement = next((r for k, r in mapping.items() if k in text), "")
            edits.append({"claim_id": claim_id, "replacement": replacement})
        return {"edits": edits}

    return build


# ---------------------------------------------------------------------------
# The run-7 draft, line by line, in reviewer terms
# ---------------------------------------------------------------------------

# Every line the source establishes. Keyed by a substring unique to the draft line.
# A value may be a tuple when one draft line restates more than one source item — the
# summary sentence carries both halves of the wrapped source paragraph, and a heading
# line carries its own heading segment. Under-claiming here is not harmless: a line the
# reviewer flags but the scenario does not route to a source item has no replacement,
# so the repair deletes it. That is exactly what the first version of this scenario did
# to the summary.
SOUND_LINES = {
    "## Professional Summary": "## Professional Summary",
    "## Professional Experience": "## Professional Experience",
    "## Education": "## Education",
    "## Certifications": "## Certifications",
    "## Technical Skills": "## Skills",
    "Journeyman electrician with 8+ years": (
        "Journeyman electrician with 8+ years",
        "experience across construction, facilities maintenance",
    ),
    "Cascade Power Solutions": "Cascade Power Solutions",
    "Journeyman Electrician |": "Journeyman Electrician",
    "Led installation and maintenance": "Installed and maintained",
    "Trained 6 apprentices": "Trained 6 apprentices",
    "Optimized panel labeling": "Cut rework by 18%",
    "Harborline Mechanical": "Harborline Mechanical",
    "Electrical Apprentice |": "Electrical Apprentice",
    "Supported conduit bending": "Assisted with conduit bending",
    "Example Technical College": (
        "Example Technical College",
        "Associate of Applied Science, Electrical Systems Technology",
    ),
    "Washington Journey Level Electrician": "Washington Journey Level Electrician",
    "NFPA 70E Arc Flash Safety": "NFPA 70E Arc Flash Safety",
}

# The demonstrated run-7 defects, plus the summary line the recorded reviewer also
# flagged (as uncertain, not unsupported — it carries an appended proficiency list past
# where the source summary ends).
DEFECT_MARKERS = ("Accumulated 8,000 hours", "**Electrical:**",
                  "Journeyman electrician with 8+ years")

# What a repair of those defects should write instead.
#
# Two things here are load-bearing and were learned by watching this scenario fail:
#
# * ``Journeyman electrician`` / ``Accumulated``: the recorded reviewer flags those
#   lines, and a repair with no replacement for a flagged line DELETES it. Restating the
#   source summary is the correct repair — the candidate wrote it — and omitting it
#   would quietly turn "repair the appended skills list" into "delete the summary".
# * ``**Electrical:**``: the repair must carry BOTH source skill lines. An earlier
#   version kept only the NFPA/lockout/tagout list, dropped "Commercial electrical
#   systems, conduit bending, panel labeling", and the gate withheld for the dropped
#   fact. That is a coverage loss caused by the repair, not by the writer, and it is the
#   correct outcome.
REPAIRED_LINES = {
    "Journeyman electrician with 8+ years":
        "Journeyman electrician with 8+ years of commercial and industrial electrical "
        "experience across construction, facilities maintenance, and power distribution.",
    "Accumulated 8,000 hours":
        "- Completed 8,000 hours of supervised field work toward licensure.",
    "**Electrical:**":
        "**Skills:** Commercial electrical systems, conduit bending, panel labeling; "
        "NFPA 70E, lockout/tagout, inspection checklists",
}

# The re-review of a repaired draft: everything sound, plus the repairs now acceptable.
REPAIRED_SUPPORTED = {
    **{k: v for k, v in SOUND_LINES.items() if k != "Journeyman electrician with 8+ years"},
    "Journeyman electrician with 8+ years": (
        "Journeyman electrician with 8+ years",
        "experience across construction, facilities maintenance",
    ),
    "Completed 8,000 hours of supervised": "Completed 8,000 hours",
    "**Skills:**": (
        "Commercial electrical systems, conduit bending, panel labeling",
        "NFPA 70E, lockout/tagout, inspection checklists",
    ),
}



def _draft_with_all_headings() -> str:
    """The recorded run-7 draft with an empty Projects heading inserted.

    The insertion is the ONLY difference. It exists so a test about structural
    completeness can be about structure: the recorded draft omits the heading entirely,
    which repair cannot fix, and conflating the two would make a repair test fail for a
    writer reason.
    """
    from source_correction import _lines
    draft = live_capture.load(RUN7)["draft"]
    lines = draft.splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("## Education"):
            lines.insert(index, "")
            lines.insert(index + 1, "## Projects")
            return "\n".join(lines)
    raise AssertionError("the recorded draft has no Education heading to anchor on")


def _repaired_review():
    return review(REPAIRED_SUPPORTED, missing=())


def _defective_review():
    return review(SOUND_LINES, unsupported=DEFECT_MARKERS, missing=())


# ---------------------------------------------------------------------------
# Scenario 1: the demonstrated defect is repaired and published
# ---------------------------------------------------------------------------

@requires_run7
def test_repaired_draft_is_published_with_the_defects_removed():
    """[controlled] A repairable draft is repaired, re-reviewed, validated, published.

    This is the recovery the ordering defect made impossible. Both demonstrated run-7
    defects are removed by an edit, the complete re-review re-authenticates every span,
    and the gate publishes.
    """
    scenario = Scenario(_defective_review(), repair=repair(REPAIRED_LINES),
                        review_after=_repaired_review())

    result, _session, _data = drive(scenario)

    # The initial review comes from the recording; only the phases after it are
    # controlled, which is the whole point of the extension.
    # The re-review is detected as a plain review: the wire form carries no engine
    # phase name, so the recorded detector names every review-shaped request "review".
    assert scenario.served[:2] == ["repair", "review"], scenario.served
    assert result.status == "approved", (result.reason, result.commentary)
    assert result.markdown is not None
    assert "mastering core electrical installations" not in result.markdown
    assert "**Electrical:**" not in result.markdown
    assert "supervised field work" in result.markdown
    assert (result.review_calls, result.repair_calls) == (2, 1)


@requires_run7
def test_supported_facts_survive_a_repair_untouched():
    """[controlled] Repair edits only the flagged span.

    A repair that rewrote a neighbouring fact would look correct if only the defect's
    absence were checked, so the surviving values are asserted explicitly.
    """
    scenario = Scenario(_defective_review(), repair=repair(REPAIRED_LINES),
                        review_after=_repaired_review())

    result, _session, _data = drive(scenario)
    assert result.status == "approved", (result.reason, result.commentary)
    md = result.markdown

    for surviving in ("40+ job sites", "6 apprentices", "8,000 hours",
                      "25 commercial projects", "03/2020", "06/2016",
                      "Example Technical College"):
        assert surviving in md, f"supported fact lost by repair: {surviving}"


@requires_run7
def test_unresolved_defect_still_withholds_after_a_repair():
    """[controlled] A repair that does not fix the defect does not publish.

    The negative control for recovery: publishing must depend on the defect being gone,
    not on the repair having been attempted. Two repair passes are allowed, so both are
    answered with the same ineffective edit, and every review keeps objecting.
    """
    scenario = Scenario(_defective_review(), repair=repair(REPAIRED_LINES),
                        review_after=_defective_review())

    result, _session, _data = drive(scenario)

    assert result.status == "withheld"
    assert result.markdown is None
    assert result.reason in {"unresolved_claims", "source_facts_missing"}, result.reason
    assert result.repair_calls >= 1, "the repair path must have been attempted"


# ---------------------------------------------------------------------------
# Negative controls for the repair path
#
# Each of these states a way a repair *could* be used to smuggle a defect past the gate
# and asserts that it cannot. They are the reason "repair now runs" is not on its own a
# good outcome: making a path reachable is only safe if the path stays bounded, so every
# bound gets its own test.
# ---------------------------------------------------------------------------

@requires_run7
def test_repair_cannot_delete_an_accomplishment():
    """[controlled] A deletion does not become an approval.

    Deleting a flagged line is a legal repair move — it is how unsupported content is
    removed. It must not be a way to *lose* a supported fact, so the re-review is
    written to notice the now-absent source item. The gate must withhold, not publish a
    resume with a bullet missing.
    """
    # Repair deletes the two defect lines and the summary; the re-review reports the
    # source items those lines carried as missing, which is the honest reading.
    scenario = Scenario(_defective_review(), repair=repair(REPAIRED_LINES),
                        review_after=review(REPAIRED_SUPPORTED,
                                            missing=("8+ years of commercial",
                                                     "Commercial electrical systems")))

    result, _session, _data = drive(scenario)

    assert result.status == "withheld", "deleting an accomplishment must not publish"
    assert result.reason == "source_facts_missing", result.reason
    assert result.markdown is None


@requires_run7
def test_repair_cannot_invent_a_qualification():
    """[controlled] A repair that writes a credential the source lacks is rejected.

    "Master Electrician licence" is a real-looking qualification and is not in the
    source. Two independent things must stop it: the re-review below does not claim it,
    and the engine's own lexical veto names "Master" as content the source does not
    contain. Either alone withholds.

    A note on how this test was first written: the repair mapping was authored as
    ``{"Accumulated 8,000 hours": invented, **REPAIRED_LINES}``, and the spread put the
    correct replacement back, so the test silently exercised the *good* repair and
    passed for the wrong reason. The mapping is now built so that the override is
    explicit.
    """
    invented = {**REPAIRED_LINES,
                "Accumulated 8,000 hours":
                    "- Holds a Master Electrician licence and completed 8,000 "
                    "supervised hours."}
    scenario = Scenario(_defective_review(), repair=repair(invented),
                        review_after=_repaired_review())

    result, _session, _data = drive(scenario)

    assert result.status == "withheld", "an invented qualification must not publish"
    assert result.markdown is None


@requires_run7
def test_engine_vetoes_an_invented_entity_without_reviewer_help():
    """[engine] The lexical veto fires on its own, naming the invented word.

    Stated separately from the end-to-end control because it is a different and
    stronger claim: the engine does not have to take the reviewer's word for an
    invented *capitalised entity*. ``_introduces_beyond`` compares the candidate span
    against the source document, and "Master" is reported as content the source does
    not state.

    Scope, stated honestly: this is a narrow lexical veto. It catches an invented named
    entity or an unsourced figure. It does not and cannot judge a semantic claim, so it
    is not a substitute for the reviewer.
    """
    from source_correction import _introduces_beyond

    data = live_capture.load(RUN7)
    source = data["sources"][0]["text"]

    invented = ("- Holds a Master Electrician licence and completed 8,000 supervised "
                "hours.")
    faithful = "- Completed 8,000 hours of supervised field work toward licensure."
    original = "- Completed 8,000 hours of supervised field work toward licensure."

    assert "Master" in _introduces_beyond(invented, [source], original)
    assert _introduces_beyond(faithful, [source], original) == []


@requires_run7
def test_moved_fact_is_held_back_by_the_reviewer_not_the_engine():
    """[controlled] Honest statement of the attribution boundary for a repaired line.

    This is a *limitation* test, not a guarantee test, and it is written the way it is
    because the first draft of it asserted the opposite and the assertion was wrong.

    A repair that attaches Cascade's accomplishment to Harborline uses only words the
    source contains, so the lexical veto sees nothing: ``_introduces_beyond`` answers
    ``[]`` here. The unique-attribution machinery (§ ``_correct_unique_attribution``)
    acts on explicit ``label: body`` records, which a resume bullet is not. What stops
    such a move is the reviewer withholding its support — an independent reviewer, not a
    deterministic check.

    So the two halves are asserted separately: the engine does not catch it lexically,
    and a reviewer that declines to substantiate it withholds the draft. Encoding this
    as a pass makes the boundary visible instead of leaving a false impression that a
    moved fact is mechanically impossible.
    """
    from source_correction import _introduces_beyond

    data = live_capture.load(RUN7)
    source = data["sources"][0]["text"]
    moved = ("- Installed and maintained commercial electrical systems across 40+ job "
             "sites for Cascade Power Solutions.")
    original = "- Assisted with conduit bending, wire pulling, and panel terminations."

    # 1. The engine has no lexical objection: every word is in the source.
    assert _introduces_beyond(moved, [source], original) == []

    # 2. A reviewer that refuses to substantiate the moved line withholds the draft.
    #
    # The move happens through a line the *recorded* review flagged, because repair may
    # only replace a flagged span and the initial review comes from the capture and
    # cannot be edited here. The summary line is used: it is flagged (as uncertain) in
    # the recording, and the repair rewrites it to credit Cascade's 40+ job sites to
    # Harborline — every word sourced, the attribution wrong.
    moved_line = ("Journeyman electrician with 8+ years of experience, including 40+ job "
                  "sites delivered for Harborline Mechanical Contractors.")
    moved_map = {**REPAIRED_LINES, "Journeyman electrician with 8+ years": moved_line}

    def refusing(wire: dict) -> dict:
        base = _repaired_review()(wire)
        for line in wire["lines"]:
            if "delivered for Harborline" in line["text"]:
                lid = line["line_id"]
                base["supported"] = [x for x in base["supported"] if x != lid]
                evidence = dict(base.get("evidence") or {})
                carried = evidence.pop(lid, [])
                base["evidence"] = evidence
                if lid not in base["unsupported"]:
                    base["unsupported"].append(lid)
                for index in carried:
                    if index in base["represented"]:
                        base["represented"].remove(index)
                    if index not in base["missing"]:
                        base["missing"].append(index)
        return base

    scenario = Scenario(_defective_review(), repair=repair(moved_map),
                        review_after=refusing)

    result, _session, _data = drive(scenario)

    assert result.status == "withheld", (
        "a reviewer that declines to substantiate a moved fact must withhold")
    assert result.markdown is None


@requires_run7
def test_repair_is_bounded_to_two_passes_and_the_call_budget():
    """[controlled] Repair cannot loop, and cannot exceed the documented call budget.

    The defect is written into every re-review, so a correct engine has to stop rather
    than keep repairing. Two repair passes are the documented allowance; the run must end
    with a withheld result, a bounded call count, and no more than the provider permits.
    """
    scenario = Scenario(_defective_review(), repair=repair(REPAIRED_LINES),
                        review_after=_defective_review())

    result, session, _data = drive(scenario)

    assert result.status == "withheld"
    assert result.repair_calls <= 2, f"repair looped: {result.repair_calls}"
    # The recording contributes two calls; everything after it is controlled.
    assert len(session.requests) <= 7, (
        f"provider calls exceeded the documented budget: {len(session.requests)}")
    assert result.reason in {"unresolved_claims", "source_facts_missing"}, result.reason


# ---------------------------------------------------------------------------
# The five promised headings survive without invented filler
# ---------------------------------------------------------------------------

@requires_run7
def test_all_six_headings_survive_a_repair_without_filler():
    """[controlled] Every promised heading is present, and none is padded.

    "Keep the headings" and "invent nothing to fill them" pull against each other, so
    both halves are asserted: the headings exist, and the sections whose source content
    is thin do not acquire entries the source never had.

    The draft used here carries all six headings and is otherwise identical to the
    recorded one. That is a deliberate, stated substitution: the recorded run-7 draft
    has no Projects heading at all, which is a *writer* defect (`b855087`) and not
    something the correction engine can resolve — repair replaces a flagged span and
    cannot insert a newline. Asserting the all-headings invariant against the recorded
    draft would test the writer's defect while claiming to test repair, so the heading is
    supplied here and the writer's own omission is recorded as a known defect in
    docs/live-runs/REMAINING_DEFECTS.md.
    """
    defect_free_layout = _draft_with_all_headings()
    scenario = Scenario(_defective_review(), repair=repair(REPAIRED_LINES),
                        review_after=_repaired_review())

    result, _session, _data = drive(scenario, draft=defect_free_layout)
    assert result.status == "approved", (result.reason, result.commentary)
    md = result.markdown

    present = ca.sections_present(md)
    for heading, found in present.items():
        assert found, f"promised section missing: {heading}"

    # Empty where the candidate has nothing, not filled with invention.
    projects_body = ca.section_body(md, "Projects")
    assert projects_body == "", (
        f"Projects must be empty for this candidate, got: {projects_body!r}")
    assert "Smart Electrical Inspection" not in md

    # No invented entries in the sections the source barely populates.
    certifications = ca.section_body(md, "Certifications")
    assert "State of Washington" not in certifications
    assert "2023" not in certifications
    assert "2022" not in certifications

    education = ca.section_body(md, "Education")
    for invented in ("GPA", "Relevant Courses", "Honors"):
        assert invented not in education

    # Projects is a required heading AND must be empty: the candidate has none, so an
    # absent heading fails the product's promise while a filled one invents a project.
    projects_body = ca.section_body(md, "Projects")
    assert projects_body == "", (
        f"Projects must be empty for this candidate, got: {projects_body!r}")
    assert "Smart Electrical Inspection" not in md
