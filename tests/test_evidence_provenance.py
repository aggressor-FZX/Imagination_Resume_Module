"""Evidence provenance — generated output must never become evidence.

Owner item 1: "Derive approved evidence only from source/confirmed candidate
data, never generated output. Add a regression proving an invented project
cannot approve itself."

The defect this pins: `_approved_entities_from_payload` read
`analysis_result["projects"]`, which is GENERATED output. The editor writes
"**Automated Reporting Dashboard**", that string becomes the allow-list, and the
entity check then finds the same string "allowed". Self-approval in one step.
"""
import sys
from pathlib import Path

import pytest

IMAG = Path("/home/skystarved/Render_Dockers/Imagination_Resume_Module")
BENCH = Path("/home/skystarved/Render_Dockers/resume_bench")
for p in (str(IMAG), str(BENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

import app as app_module  # noqa: E402
from rctb.ledger import Ledger, check_entities  # noqa: E402

_derive = app_module._approved_entities_from_payload


# ── the self-approval regression ─────────────────────────────────────────
def test_invented_project_cannot_approve_itself():
    """Generated projects must not appear in the allow-list."""
    analysis_result = {
        # GENERATED output — must be ignored as evidence
        "projects": [{"name": "Automated Reporting Dashboard",
                      "bullets": ["Built a platform"]}],
        "experiences": [{"company": "Cascade Power Solutions, LLC"}],
        # SOURCE echo — legitimate evidence
        "source_projects": [{"name": "Commercial Building Retrofit"}],
    }
    payload = {"projects": []}   # caller confirms: no projects

    approved = _derive(analysis_result, payload)

    assert approved is not None, "must still have evidence"
    lowered = " | ".join(approved).lower()
    assert "automated reporting dashboard" not in lowered, (
        "a GENERATED project entered the allow-list and could approve itself"
    )
    assert "commercial building retrofit" in lowered, (
        "source-provided project evidence was dropped"
    )


def test_generated_projects_alone_yield_no_evidence():
    """If nothing but generated output exists, evidence is MISSING (None)."""
    analysis_result = {
        "projects": [{"name": "Fabricated Platform"}],
        "experiences": [{"company": "Made Up Corp"}],
        "education": [{"institution": "Invented University"}],
    }
    approved = _derive(analysis_result, None)
    assert approved is None, (
        f"generated output was treated as evidence: {approved}"
    )


def test_callers_supplied_projects_are_evidence():
    approved = _derive({}, {"projects": [{"name": "Real Project Alpha"}]})
    assert approved and any("real project alpha" in e.lower() for e in approved)


def test_empty_caller_projects_is_empty_not_missing():
    """A candidate with no projects: source examined, nothing found."""
    approved = _derive({}, {"projects": [], "education": []})
    assert approved == [], (
        "explicitly-empty source data must be EMPTY ([]), not MISSING (None)"
    )


def test_no_source_at_all_is_missing():
    assert _derive({}, None) is None
    assert _derive(None, None) is None


# ── end-to-end: the attack path is closed ────────────────────────────────
def test_end_to_end_invented_project_is_rejected():
    """Full check: generated project present, source has none -> rejected."""
    analysis_result = {
        "projects": [{"name": "Automated Reporting Dashboard"}],
        "source_projects": [],
    }
    approved = _derive(analysis_result, {"projects": []})
    assert approved == [], "no source projects, so allow-list is empty"

    ledger = Ledger.from_dict({
        "profile_id": "p",
        "companies": ["Cascade Power Solutions, LLC"],
        "claims": [{"fact_id": "f1", "value": "30", "unit": "percent"}],
    })
    generated = (
        "## Projects\n"
        "**Automated Reporting Dashboard** -- Power BI, Excel\n"
        "- Built a self-service platform\n"
    )
    violations = check_entities(generated, ledger, allowed_entities=approved)
    assert violations, (
        "the invented project was accepted because it approved itself"
    )
    assert any("Automated Reporting Dashboard" in v["text"] for v in violations)


def test_buggy_allowlist_construction_would_self_approve():
    """BYPASS DEMONSTRATION: show what the old code did.

    Reconstructs the previous behaviour to prove the regression is meaningful —
    if the old approach also rejected, the test would prove nothing.
    """
    analysis_result = {
        "projects": [{"name": "Automated Reporting Dashboard"}],
    }

    def old_derive(result):
        """The removed implementation: reads generated output as evidence."""
        out = []
        for pr in (result.get("projects") or []):
            if isinstance(pr, dict) and pr.get("name"):
                out.append(pr["name"])
        return out

    old = old_derive(analysis_result)
    new = _derive(analysis_result, None)

    assert "Automated Reporting Dashboard" in old, "control: old code self-approved"
    assert new is None or not any(
        "automated reporting dashboard" in e.lower() for e in (new or [])
    ), "new code must not self-approve"

    # And the entity check would have passed under the old allow-list.
    ledger = Ledger.from_dict({"profile_id": "p", "companies": [], "claims": []})
    generated = ("## Projects\n**Automated Reporting Dashboard** -- Power BI\n")
    assert not check_entities(generated, ledger, allowed_entities=old), (
        "control: with the buggy allow-list, the invented project passes"
    )
    assert check_entities(generated, ledger, allowed_entities=["Real Project"]), (
        "with a correct allow-list it is rejected"
    )
