"""The three obstacles between the prose engine and the live service.

`semantic_review.engine` is correct on its own terms — all 23 integration tests
pass. What it CANNOT do is run through the live `CorrectionProvider` unchanged.

These tests pin the exact boundaries, because each one is a control that must not
be quietly bent to make the integration look finished:

1. The provider requires a JSON **object** reply. The prose contract answers with
   prose, so every live reply is rejected as `invalid provider response`.
2. The disclosure layer redacts by KEY NAME. The draft is only covered while it
   travels under a covered key; naming it anything else silently removes it from
   redaction, which is a security change, not a refactor.
3. The reviewer sees a REDACTED draft but the repair must edit the ORIGINAL. When
   those differ, a redacted passage does not locate, and the engine fails closed.

Nothing here is fixed. They are the reasons the wiring stops here.
"""

from __future__ import annotations

import json

import pytest

import decision_contract as dc
from disclosure import (
    _TEXT_FIELDS, contains_pii, payload_disclosure_errors, redact_payload,
    redact_text,
)
from semantic_review.engine import review_and_release

POLICY = "length_preserving_pii_redaction_v1"


@pytest.fixture(autouse=True)
def _approved_disclosure(monkeypatch):
    """Enable the approved disclosure policy for THIS module only.

    The rest of the suite deliberately runs without it, because several tests
    assert that nothing is transmitted when disclosure is NOT approved. Setting
    this globally would invert those assertions, so it is scoped here.
    """
    monkeypatch.setenv("CORRECTION_PROVIDER_DISCLOSURE", "1")


# ── 1. The provider cannot carry a prose reply ─────────────────────────────

def test_1_the_provider_rejects_a_prose_reply():
    """`_strict_json` must return a dict; prose is a hard failure, not a fallback.

    So a prose-answering reviewer routed through `CorrectionProvider` raises
    `ValueError("invalid provider response")` on EVERY call, the engine maps that
    to `provider_error`, and every document is withheld.
    """
    from correction_provider import _strict_json

    prose = "Decision: approve\n\nIssues:\n- none\n"
    with pytest.raises(ValueError):
        parsed = _strict_json(prose)
        if not isinstance(parsed, dict):
            raise ValueError("invalid provider response")


def test_1b_the_json_reply_path_is_the_only_accepted_shape():
    from correction_provider import _strict_json
    assert isinstance(_strict_json('{"decision": "approve"}'), dict)


# ── 2. Redaction is keyed by name, not by content ──────────────────────────

RESUME_WITH_PII = (
    "## Professional Summary\nOperations analyst.\n\n"
    "## Professional Experience\n"
    "**Harborline Transit Authority** -- Tacoma, WA\n"
    "- Reconciled 1,180 fuel invoices; escalated discrepancies to "
    "jeff.calderon@example.com weekly.\n"
)


def test_2_the_allowlist_covers_exactly_three_free_text_keys():
    """This is the contract the disclosure control enforces."""
    assert _TEXT_FIELDS == frozenset({"text", "quote", "contract"})


def test_2b_a_covered_key_redacts_the_draft_and_a_dict_literal_verifies_it():
    payload = {"contract": dc.contract_for("conservative"),
               "text": RESUME_WITH_PII}
    out, redactions = redact_payload(payload)
    assert redactions == 1
    assert not contains_pii(out["text"])
    assert len(out["text"]) == len(RESUME_WITH_PII), "length must be preserved"
    assert payload_disclosure_errors(out, approved_policy=POLICY) == []


def test_2c_an_uncovered_key_name_silently_skips_redaction():
    """THE security change. A neutral key name disables the control.

    Note the payload still VERIFIES CLEAN, because the verification walks the same
    allowlist. So the failure is silent: PII leaves the process and every gate
    reports success.
    """
    payload = {"contract": dc.contract_for("conservative"),
               "draft": RESUME_WITH_PII}
    out, redactions = redact_payload(payload)

    assert redactions == 0, "no redaction was attempted"
    assert contains_pii(out["draft"]), "PII is still present"
    assert payload_disclosure_errors(out, approved_policy=POLICY) == [], \
        "and the gate reports it as clean"

    # Therefore the prose payload must carry the draft under `text`.
    safe, n = redact_payload({"contract": dc.contract_for("conservative"),
                              "text": RESUME_WITH_PII})
    assert n == 1 and not contains_pii(safe["text"])


# ── 3. Redacted review input vs. original document to edit ─────────────────

def test_3_a_redacted_passage_does_not_locate_in_the_original_draft():
    """The reviewer can only quote what it was shown, i.e. the redacted text."""
    redacted, n = redact_text(RESUME_WITH_PII)
    assert n == 1

    redacted_line = redacted.splitlines()[-1]
    original_line = RESUME_WITH_PII.splitlines()[-1]
    assert redacted_line != original_line
    assert dc.passage_resolves_in_draft(redacted_line, redacted)
    assert not dc.passage_resolves_in_draft(redacted_line, RESUME_WITH_PII)


def test_3b_the_engine_fails_closed_rather_than_editing_the_wrong_text():
    """A repair aimed at redacted text, applied to the original, is a no-op.

    That is the correct failure direction — nothing is published — but it means a
    naive wiring withholds every résumé whose draft carries a covered PII
    category, i.e. any document with a contact line.
    """
    import asyncio

    revise = ("Decision: revise\n\nIssues:\n"
              "- passage: - Reconciled 1,180 fuel invoices; escalated "
              "discrepancies to [email removed]██████████ weekly.\n"
              "  problem: unsupported\n  sources: [resume]\n"
              "  why: the method is not sourced.\n"
              "  correction: - Reconciled 1,180 fuel invoices.\n")

    async def review(_document: str) -> str:
        return revise

    result = asyncio.run(review_and_release(
        RESUME_WITH_PII, {"resume": RESUME_WITH_PII}, review))

    assert result.status == "withheld"
    assert result.markdown is None
    # Stronger than expected: the parser rejects the redacted passage
    # STRUCTURALLY (`passage_not_in_draft`), because it can only resolve passages
    # against the document it is given. A redacted passage is not a valid passage
    # for the original document, so the review never even reaches the repair step.
    assert result.reason == "malformed_output"
    assert result.repair_calls == 0


# ── The shape of the required change ──────────────────────────────────────

def test_the_payload_the_live_provider_would_accept_is_expressible():
    """A prose payload CAN satisfy the existing disclosure controls verbatim.

    Draft under `text`, sources under `sources`, instructions under `contract`:
    every covered key, no new key, no weakened check. What it cannot do is satisfy
    `_strict_json`'s dict requirement, which is the one control that has to change.
    """
    payload = {"contract": dc.contract_for("conservative"),
               "sources": [{"source_id": "resume", "text": RESUME_WITH_PII}],
               "text": RESUME_WITH_PII}

    out, n = redact_payload(payload)
    assert n >= 1
    assert not contains_pii(out["text"])
    assert not contains_pii(out["sources"][0]["text"])
    assert payload_disclosure_errors(out, approved_policy=POLICY) == []
    # And it serializes as a JSON object, which is what the transport requires.
    assert isinstance(json.loads(json.dumps(out)), dict)
