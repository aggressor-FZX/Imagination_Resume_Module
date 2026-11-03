"""Source-fidelity regressions for the Imaginator correction path.

These are the *fidelity* acceptance tests, distinct from the engineering tests:

* expectations come from ``tests/fixtures/fidelity/expected_facts.py``, an answer key
  written from the source document alone and not from the implementation;
* every test states which layer it exercises, because "the route works" and "the model
  returns better wording" are different claims.

Layers used here
----------------
``[capture]``  the protected live capture, replayed through the real adapter, real
               engine and real validator, with no network.
``[engine]``   the real correction engine driven by a *labelled scripted* provider.
               A scripted provider proves orchestration and validation, never that a
               real model would answer this way.
``[writer]``   real source -> parsing/sanitizing code that actually runs in the path.

Nothing here mocks PDF/DOCX bytes. Renderer content checks live in
``tests/test_export_receipt_title_binding.py`` and the renderer suites; they are
labelled separately.
"""
import json
import os
import socket
from pathlib import Path

import pytest

import live_capture
import source_correction
from source_correction import (
    CorrectionResult,
    SourceDocument,
    correct,
)
from tests.fixtures.fidelity import expected_facts as facts


CAPTURE = Path("/home/skystarved/Render_Dockers/.live-run-2026-11-02/retained/"
               "delivery-run-4-withheld.json")


def _capture_available() -> bool:
    return CAPTURE.exists()


requires_capture = pytest.mark.skipif(
    not _capture_available(),
    reason="protected live capture is not present on this machine")


# ---------------------------------------------------------------------------
# [capture] The recording still reproduces the original rejection
# ---------------------------------------------------------------------------

@requires_capture
def test_captured_draft_rejection_is_still_reproducible(monkeypatch):
    """[capture] Run 4's exact draft is still withheld, for the same reason.

    This is the primary regression artifact. If a change makes the recorded exchange
    approve, either the reviewer's completeness answer is no longer being honoured or
    the fixture is no longer reaching the engine.

    What changed, and why the old numbers are no longer asserted here: the gate used to
    withhold on a missing fact *before* entering the repair path, so run 4 recorded
    ``repair_calls == 0``. That ordering was a defect — 19 lines were flagged and no
    attempt to fix them was made — and it is now repaired first. The recording holds only
    its original two exchanges, so the repair answer does not exist and the run still
    withholds; the *route* to the rejection differs, the outcome does not. The original
    diagnosis is pinned separately from the capture's own bytes by
    ``test_original_diagnosis_is_read_from_the_capture``.
    """
    monkeypatch.setattr(socket.socket, "connect", _forbid_network)
    result, provider, session = live_capture.replay(str(CAPTURE))

    assert result.status == "withheld"
    assert result.markdown is None
    # The engine now attempts the repair the old code never reached, and cannot finish
    # because the recording has no repair answer.
    assert result.repair_calls == 1
    assert session.position == len(session.recorded["exchanges"])


@requires_capture
def test_original_diagnosis_is_read_from_the_capture():
    """[capture] Run 4's original rejection is pinned to the bytes that produced it.

    The stopping point is evidence about a revision, not a claim about current code.
    Recomputing "what the old code did" from current code would silently rewrite the
    history that made the defect findable, so it is read out of the artifact's own
    ``review`` and ``timing`` fields.
    """
    stopped = live_capture.recorded_stopping_point(str(CAPTURE))

    assert stopped["status"] == "withheld"
    assert stopped["reason"] == "source_facts_missing"
    assert stopped["review_calls"] == 1
    assert stopped["repair_calls"] == 0, (
        "the recorded run must show that repair was never attempted")
    assert stopped["phases"] == ["source_inventory", "review"]
    assert stopped["phase_count"] == 2


@requires_capture
def test_capture_matches_the_answer_key_source():
    """The capture's source is the same text the answer key was written from."""
    data = live_capture.load(str(CAPTURE))
    assert data["sources"][0]["text"] == facts.SOURCE_TEXT
    assert data["synthetic"] == live_capture.SYNTHETIC_MARKER


@requires_capture
def test_captured_draft_contains_the_inventions_the_key_forbids():
    """The recorded draft is defective in exactly the ways the ledger predicts.

    Asserted against the *capture*, not against an implementation, so the ledger's
    claims about run-4 are themselves checkable evidence.
    """
    data = live_capture.load(str(CAPTURE))
    draft = data["draft"]

    # Fabricated placeholder standing in for both real job titles.
    assert "Professional | 03/2020" in draft
    assert "Professional | 06/2016" in draft
    for record in facts.EMPLOYMENT_RECORDS:
        assert record["role"] not in draft, (
            f"role {record['role']!r} unexpectedly present in the captured draft")

    # Content with no basis in the source.
    for project in facts.ABSENT_FROM_SOURCE["projects_section_content"]:
        assert project in draft
    for measure in facts.ABSENT_FROM_SOURCE["invented_measurements"]:
        assert measure in draft
    for technology in facts.ABSENT_FROM_SOURCE["invented_technologies"]:
        assert technology.lower() in draft.lower()
    assert "GPA" in draft

    # Credentials were given invented issuers and dates.
    assert "Department of Labor & Industries (2020)" in draft
    assert "National Fire Protection Association (2019)" in draft

    # The unfaithful "six apprentice crews" upgrade.
    assert "six apprentice crews" in draft
    assert "6 apprentices" not in draft
    assert "Six apprentices" not in draft


# ---------------------------------------------------------------------------
# [writer] Title preservation: the defect that destroyed both real roles
# ---------------------------------------------------------------------------

def test_writer_never_fabricates_a_placeholder_title():
    """[writer] A rejected title must leave the field empty, never invented.

    Before the fix, ``sanitize_experience_record`` wrote the literal word
    "Professional" into every title key, which then flowed into the writer draft as
    though it were the candidate's role.
    """
    from role_title_sanitizer import sanitize_experience_record

    for record in facts.EMPLOYMENT_RECORDS:
        header = f"{record['role']} | {record['start']} - {record['end']}"
        out = sanitize_experience_record({"title_line": header})
        assert out["title_line"] == header

    rejected = sanitize_experience_record({"title_line": "unparseable fragment . here"})
    for placeholder in facts.FABRICATED_TITLE_PLACEHOLDERS:
        assert placeholder not in (rejected.get("title_line") or "")
    assert rejected["title_line"] == ""


def test_writer_recognizes_trade_titles():
    """[writer] Real trade roles validate instead of being thrown away."""
    from role_title_sanitizer import is_valid_extracted_job_title

    for record in facts.EMPLOYMENT_RECORDS:
        assert is_valid_extracted_job_title(record["role"]), record["role"]


# ---------------------------------------------------------------------------
# [writer] Unicode round-trip at the provider boundary
# ---------------------------------------------------------------------------

def test_provider_stream_preserves_unicode_punctuation():
    """[writer] UTF-8 SSE bytes decode as UTF-8, not ISO-8859-1.

    ``requests`` falls back to ISO-8859-1 for ``text/*`` without a charset, which is
    exactly what ``text/event-stream`` is. Decoding through the response's inferred
    encoding turned an en dash into ``â€“`` before it ever reached the draft.
    """
    from openrouter_safe_client import _iter_sse_lines_utf8

    punctuation = {
        "en dash": "\u2013",
        "em dash": "\u2014",
        "right quote": "\u2019",
        "non-breaking hyphen": "\u2011",
        "bullet": "\u2022",
    }

    class _Response:
        encoding = "ISO-8859-1"  # what requests infers for text/event-stream

        def __init__(self, data: bytes):
            self._data = data

        def iter_content(self, chunk_size=8192):
            for i in range(0, len(self._data), chunk_size):
                yield self._data[i:i + chunk_size]

    for label, character in punctuation.items():
        body = f"03/2020 {character} Present".encode("utf-8")
        line = 'data: {"choices":[{"delta":{"content":"%s"}}]}' % body.decode("utf-8")
        decoded = _iter_sse_lines_utf8(_Response(line.encode("utf-8")))
        text = "".join(decoded)
        assert "\u00e2" not in text, f"{label} was mojibaked"
        assert character in json.loads(text[5:])["choices"][0]["delta"]["content"]


def test_provider_stream_preserves_accented_names():
    """[writer] Legitimate non-ASCII is not damaged by the encoding fix."""
    from openrouter_safe_client import _iter_sse_lines_utf8

    class _Response:
        def __init__(self, data: bytes):
            self._data = data

        def iter_content(self, chunk_size=8192):
            yield self._data

    payload = "data: {\"name\": \"Jos\u00e9 \u00c5berg \u2013 M\u00fcller\"}\n"
    text = "".join(_iter_sse_lines_utf8(_Response(payload.encode("utf-8"))))
    assert "Jos\u00e9" in text and "\u00c5berg" in text and "M\u00fcller" in text
    assert "\u00e2" not in text and "\u00c3" not in text


# ---------------------------------------------------------------------------
# [engine] Evidence substantiation: a citation must support the whole span
# ---------------------------------------------------------------------------

def test_partially_supported_skills_line_cannot_pass():
    """[engine] Shared words in a citation are not proof for the rest of the span.

    A line whose citation quotes "inspection checklists" must not be accepted as
    "representing" that source item while the same span also claims Agile, Scrum and
    Gantt charts.
    """
    from source_correction import _quote_shares_content

    assert _quote_shares_content("inspection checklists", "inspection checklists")
    assert _quote_shares_content("Standardized panel labeling", "panel labeling")
    # Stop words alone are not substantiation.
    assert not _quote_shares_content("the with and", "with the and")
    # Completely unrelated additions share nothing with the cited item.
    assert not _quote_shares_content(
        "Agile, Scrum and Gantt charts", "inspection checklists")


def test_review_rejects_a_representation_the_span_does_not_restate():
    """[engine] The validator refuses a "represented" claim citing unrelated text.

    Spans and ids are exact, the citation is valid on its own terms, and the line is
    fully partitioned — so the ONLY thing that can reject this answer is the
    substantiation rule. With that rule disabled the same answer validates, which is
    what makes this test a real control rather than a re-test of an earlier guard.
    """
    from source_correction import _Invalid, _validate_review

    source = "Skills: inspection checklists"
    item = {"id": f"doc:0:{len(source)}", "source_id": "doc", "start": 0,
            "end": len(source), "quote": source, "classification": "required"}
    line_text = "Agile, Scrum and Gantt charts"
    lines = [{"line_id": "L000001", "text": line_text, "absolute_start": 0}]
    response = {
        "claims": [{
            "id": f"L000001:0:{len(line_text)}", "line_id": "L000001", "start": 0,
            "end": len(line_text), "verdict": "supported",
            "citations": [{"source_id": "doc", "start": 0, "end": len(source),
                           "quote": source}],
        }],
        "representations": [{
            "source_claim_id": item["id"], "status": "represented",
            "spans": [{"line_id": "L000001", "start": 0, "end": len(line_text)}],
        }],
    }
    with pytest.raises(_Invalid):
        _validate_review(response, lines, {"doc": source}, [item])


# ---------------------------------------------------------------------------
# [engine] Bounded restoration of a genuinely missing fact
# ---------------------------------------------------------------------------

# Labelled scripted provider. It is NOT a model: it answers deterministically so the
# engine's orchestration can be asserted. Every such test says so in its name/docstring.

RESTORE_SOURCE = SourceDocument(
    "candidate",
    "Cascade Power Solutions, LLC\n"
    "Journeyman Electrician | 03/2020 - Present\n"
    "Installed and maintained commercial electrical systems across 40+ job sites.\n",
)

# Draft with the role title destroyed, exactly as live run 4 produced it: the
# sanitizer replaced the real title with the placeholder word "Professional".
# (A draft with NO line relating to the role at all must not be restored: the engine
# refuses to invent an anchor for a fact it cannot place. That behaviour is asserted
# separately in test_no_anchor_means_no_restoration.)
RESTORE_DRAFT = (
    "**Cascade Power Solutions, LLC**\n"
    "*Professional | 03/2020 - Present*\n"
    "- Installed and maintained commercial electrical systems across 40+ job sites.\n"
)


def _segmented_inventory(source_text: str, special: str):
    """Build a valid inventory: every character partitioned contiguously.

    The engine requires the required items to cover each source exactly, which is what
    makes "represented vs missing" well defined. Items follow source line boundaries
    (so one item never straddles two unrelated facts), and the ``special`` substring
    is guaranteed to be its own item so a test can drive it being absent.
    """
    spans: list[tuple[int, int]] = []
    offset = 0
    for segment in source_text.splitlines(keepends=True):
        begin, end = offset, offset + len(segment)
        offset = end
        if special in segment:
            local = segment.index(special)
            cut = begin + local
            if cut > begin:
                spans.append((begin, cut))
            spans.append((cut, cut + len(special)))
            tail = cut + len(special)
            if tail < end:
                spans.append((tail, end))
        else:
            spans.append((begin, end))
    items = []
    for left, right in spans:
        if right <= left:
            continue
        quote = source_text[left:right]
        items.append({
            "id": f"candidate:{left}:{right}",
            "source_id": "candidate",
            "start": left,
            "end": right,
            "quote": quote,
            "classification": "required",
        })
    return items


def _cite(source_text: str, needle: str):
    start = source_text.index(needle)
    return [{"source_id": "candidate", "start": start,
             "end": start + len(needle), "quote": needle}]


def _scripted_provider(monkeypatch, phases):
    """Install a labelled scripted provider; returns the list of phases it saw."""
    seen = []

    async def provider(phase, payload):
        seen.append(phase)
        handler = phases.get(phase)
        if handler is None:
            raise AssertionError(f"unexpected provider phase {phase!r}")
        return handler(payload)

    monkeypatch.setattr(source_correction, "_sleep", _no_sleep, raising=False)
    return provider, seen


async def _no_sleep(_seconds):
    return None


def _claim(line_id, start, end, verdict, citations):
    return {"id": f"{line_id}:{start}:{end}", "line_id": line_id, "start": start,
            "end": end, "verdict": verdict, "citations": citations}


def _run_async(coro):
    import asyncio

    return asyncio.run(coro)


def _review_script(source_text: str, title_state: str):
    """Build a review answer for ``RESTORE_DRAFT``.

    ``title_state`` is "missing" (the role title is absent) or "present". Every draft
    line is judged and every source item is accounted for.

    Each represented item's claim cites the item's exact source bounds, which is what
    the engine requires: a citation must cover the whole item it claims to represent.
    The reviewer here is deliberately generous — it proves orchestration, not judgement.
    """
    title_quote = "Journeyman Electrician"

    def handler(payload):
        items = payload["source_claims"]

        def item_for(quote: str):
            return next((i for i in items if quote in i["quote"]), None)

        def cite_for(item):
            return [{"source_id": item["source_id"], "start": item["start"],
                     "end": item["end"], "quote": item["quote"]}]

        claims = []
        for line in payload["lines"]:
            text, lid = line["text"], line["line_id"]
            if "03/2020" in text and title_state == "missing":
                # The empty title field asserts nothing; the dates are supported.
                dates = item_for("03/2020 - Present")
                claims.append(_claim(lid, 0, 2, "structural", []))
                claims.append(_claim(lid, 2, len(text), "supported",
                                     cite_for(dates) if dates else []))
                continue
            if "03/2020" in text:
                # One line, two source items: the role title and its dates. Both must
                # be cited, because a citation has to cover the item it represents.
                cites = [cite_for(i) for i in (item_for(title_quote),
                                               item_for("03/2020 - Present"))
                         if i is not None]
                merged = [c for group in cites for c in group]
                claims.append(_claim(lid, 0, len(text), "supported", merged))
                continue
            needle = ("Cascade Power Solutions" if "Cascade" in text
                      else "Installed and maintained")
            item = item_for(needle)
            claims.append(_claim(lid, 0, len(text), "supported",
                                 cite_for(item) if item else []))

        reps = []
        for item in items:
            if item["quote"].strip() == title_quote and title_state == "missing":
                reps.append({"source_claim_id": item["id"], "status": "missing",
                             "spans": []})
                continue
            # Attach the item to the line whose claim cites its exact bounds.
            span = None
            for claim in claims:
                if claim["verdict"] != "supported":
                    continue
                if any(c["source_id"] == item["source_id"]
                       and c["start"] == item["start"] and c["end"] == item["end"]
                       for c in claim["citations"]):
                    # The span must be the claim's own exact span: the engine
                    # matches representations against claims by (line, start, end).
                    span = {"line_id": claim["line_id"], "start": claim["start"],
                            "end": claim["end"]}
                    break
            if span is None:
                reps.append({"source_claim_id": item["id"], "status": "missing",
                             "spans": []})
            else:
                reps.append({"source_claim_id": item["id"], "status": "represented",
                             "spans": [span]})
        return {"claims": claims, "representations": reps}

    return handler


def _inventory_handler(source_text: str, special: str):
    items = _segmented_inventory(source_text, special)

    def handler(_payload):
        return {"source_claims": [dict(item) for item in items]}

    return handler, items


def _restore_handler(source_text: str, items, replacement: str):
    """Answer a restore phase.

    Targets the line whose text is the title-less header (``* | 03/2020 ...*``), which
    is the line a missing role title belongs on. ``before`` must echo that exact text.
    """
    title_id = next(item["id"] for item in items
                    if item["quote"].strip() == "Journeyman Electrician")

    def handler(payload):
        # The header line carrying the dates, whose role title is the missing fact.
        line = next(l for l in payload["lines"] if "03/2020" in l["text"])
        return {"restorations": [{
            "line_id": line["line_id"],
            "source_claim_id": title_id,
            "before": line["text"],
            "replacement": replacement,
        }]}

    return handler


def test_scripted_restoration_restores_a_missing_title(monkeypatch):
    """[engine, scripted provider] A missing title is restored, then fully re-reviewed.

    The answers below are test scaffolding, not model output. What is proven is the
    mechanism: exactly one bounded restore call, the restored document re-reviewed end
    to end, and approval reached through the unchanged validation path.
    """
    source_text = RESTORE_SOURCE.text
    inventory_handler, items = _inventory_handler(source_text, "Journeyman Electrician")
    provider, seen = _scripted_provider(monkeypatch, {
        "source_inventory": inventory_handler,
        "initial_review": _review_script(source_text, "missing"),
        "restore": _restore_handler(source_text, items,
                                    "*Journeyman Electrician | 03/2020 - Present*"),
        "post_restore_review": _review_script(source_text, "present"),
    })

    result = _run_async(correct(RESTORE_DRAFT, (RESTORE_SOURCE,), provider))

    assert seen == ["source_inventory", "initial_review", "restore",
                    "post_restore_review"], seen
    assert result.status == "approved", (result.reason, result.commentary)
    assert result.restore_calls == 1
    assert "Journeyman Electrician" in result.markdown
    assert result.markdown.count("03/2020 - Present") == 1
    assert "source fact(s) the draft had omitted" in result.commentary


def test_restoration_is_bounded_to_one_attempt(monkeypatch):
    """[engine, scripted provider] Still-missing after a restoration ends in withholding.

    The second review reports the title missing again. There is no second restoration:
    repeating it would be a rewriting loop, not a bounded repair.
    """
    source_text = RESTORE_SOURCE.text
    inventory_handler, items = _inventory_handler(source_text, "Journeyman Electrician")
    provider, seen = _scripted_provider(monkeypatch, {
        "source_inventory": inventory_handler,
        "initial_review": _review_script(source_text, "missing"),
        "restore": _restore_handler(source_text, items,
                                    "*Journeyman Electrician | 03/2020 - Present*"),
        "post_restore_review": _review_script(source_text, "missing"),
    })

    result = _run_async(correct(RESTORE_DRAFT, (RESTORE_SOURCE,), provider))

    assert seen.count("restore") == 1
    assert result.status == "withheld"
    assert result.reason == "source_facts_missing"
    assert result.markdown is None


def test_restoration_cannot_smuggle_in_a_new_fact(monkeypatch):
    """[engine, scripted provider] A restoration restating nothing from the item is refused."""
    source_text = RESTORE_SOURCE.text
    inventory_handler, items = _inventory_handler(source_text, "Journeyman Electrician")
    provider, seen = _scripted_provider(monkeypatch, {
        "source_inventory": inventory_handler,
        "initial_review": _review_script(source_text, "missing"),
        # Nothing in common with "Journeyman Electrician": pure invention.
        "restore": _restore_handler(
            source_text, items,
            "Managed Kubernetes platform and Python microservices"),
    })

    result = _run_async(correct(RESTORE_DRAFT, (RESTORE_SOURCE,), provider))

    assert "post_restore_review" not in seen
    assert result.status == "withheld"
    assert result.reason == "source_facts_missing"
    assert result.markdown is None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _forbid_network(*args, **kwargs):  # pragma: no cover - only fires on failure
    raise AssertionError("replay attempted a network connection")
