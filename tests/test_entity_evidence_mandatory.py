"""Entity evidence is mandatory and three-state (owner item 1 + 2).

The distinction that matters:

    MISSING  (None)  -> the route forgot to pass evidence. WIRING BUG. Blocked.
    EMPTY    ([])    -> the candidate genuinely has no projects. Legitimate.
    POPULATED([...]) -> entities are checked against the list.

Collapsing MISSING and EMPTY into one value is how a route silently skips entity
verification and still publishes.

Owner item 2 additionally requires proving:
  * a candidate with no projects cannot RECEIVE a generated project
  * an existing project cannot GAIN unsupported accomplishments by keeping its name
"""
import sys
from pathlib import Path

import pytest

IMAG = Path("/home/skystarved/Render_Dockers/Imagination_Resume_Module")
BENCH = Path("/home/skystarved/Render_Dockers/resume_bench")
for p in (str(IMAG), str(BENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from publication_gate import (  # noqa: E402
    EntityEvidence, Reason, Verdict, decide,
)
from rctb.ledger import Ledger, check_entities  # noqa: E402

GOOD = """## Professional Experience
**Cascade Power Solutions, LLC** — Tacoma, WA
- Troubleshoot and resolve faults in 480V three-phase distribution panels, motor
  control centers, and lighting control systems, reducing average diagnostic time
  by 30% across the service area
- Lead a crew of 2-3 apprentices on multi-building retail center buildouts,
  coordinating daily task assignments and conducting morning safety briefings
- Install, maintain, and repair commercial electrical wiring in buildings from
  5,000 to 80,000 square feet, ensuring NEC 2023 compliance throughout the site

## Education
**Bates Technical College** — Tacoma, WA | Graduated May 2015
- A.A.S. Electrical Technology, GPA 3.6 / 4.0
"""

VERIFIED = {"verdict": "PASS (tier A only)", "checked": True,
            "unsupported_metrics": []}


# ── three-state evidence ─────────────────────────────────────────────────
def test_missing_evidence_blocks_as_wiring_bug():
    """None means the route forgot the parameter — never treated as 'none exist'."""
    d = decide(resume_markdown=GOOD, metrics_support=VERIFIED,
               approved_entities=None)
    assert d.blocked()
    assert Reason.ENTITY_EVIDENCE_MISSING in d.reasons
    assert d.entity_evidence == EntityEvidence.MISSING.value


def test_empty_evidence_is_legitimate_and_does_not_block():
    """A candidate with no projects must still be able to publish."""
    d = decide(resume_markdown=GOOD, metrics_support=VERIFIED,
               approved_entities=[])
    assert d.publishable, d.reason_codes
    assert d.entity_evidence == EntityEvidence.EMPTY.value


def test_populated_evidence_is_reported():
    d = decide(resume_markdown=GOOD, metrics_support=VERIFIED,
               approved_entities=["Cascade Power Solutions, LLC"])
    assert d.publishable
    assert d.entity_evidence == EntityEvidence.POPULATED.value


def test_missing_and_empty_are_distinguishable():
    """The whole point of the distinction."""
    missing = decide(resume_markdown=GOOD, metrics_support=VERIFIED,
                     approved_entities=None)
    empty = decide(resume_markdown=GOOD, metrics_support=VERIFIED,
                   approved_entities=[])
    assert missing.entity_evidence != empty.entity_evidence
    assert missing.blocked() and not empty.blocked()


def test_non_list_evidence_blocks():
    for bad in ("Cascade Power", 42, {"a": 1}):
        d = decide(resume_markdown=GOOD, metrics_support=VERIFIED,
                   approved_entities=bad)
        assert d.blocked(), f"{bad!r} must be treated as missing evidence"


# ── typed decision, not string matching ──────────────────────────────────
def test_verdict_enum_rejects_near_misses():
    """String prefix matching was the fail-open. Typed comparison is exact."""
    for near in ("PASS", "PASS ", "pass (tier a only)", "PASS (tier A)",
                 "PASSTIER A", "PASS (tier A only) but"):
        d = decide(resume_markdown=GOOD, metrics_support={
            "verdict": near, "checked": True}, approved_entities=[])
        assert d.blocked(), f"{near!r} must not pass typed comparison"


def test_verdict_enum_accepts_exact_value_only():
    d = decide(resume_markdown=GOOD, metrics_support={
        "verdict": Verdict.PASS_TIER_A.value, "checked": True},
        approved_entities=[])
    assert d.publishable


def test_reason_codes_are_strings_for_presentation():
    d = decide(resume_markdown="", metrics_support=None, approved_entities=None)
    assert all(isinstance(c, str) for c in d.reason_codes)
    assert "empty_output" in d.reason_codes


# ── item 2a: no-project candidate cannot RECEIVE a project ───────────────
def test_candidate_with_no_projects_cannot_receive_one():
    """Editor invents a project for someone who has none."""
    empty_ledger = Ledger.from_dict({
        "profile_id": "no-projects",
        "companies": ["Cascade Power Solutions, LLC"],
        "claims": [{"fact_id": "f1", "value": "30", "unit": "percent"}],
    })
    # Candidate supplied NO projects -> approved_entities carries only employers.
    approved = ["Cascade Power Solutions, LLC"]

    generated = (
        "## Professional Experience\n"
        "**Cascade Power Solutions, LLC** — Tacoma, WA\n"
        "- Reduced diagnostic time by 30%\n\n"
        "## Projects\n"
        "**Automated Reporting Dashboard** — Power BI, Excel\n"
        "- Built a self-service reporting platform\n"
    )
    violations = check_entities(generated, empty_ledger,
                                allowed_entities=approved)
    assert violations, "an invented project was accepted for a candidate with none"
    assert any("Automated Reporting Dashboard" in v["text"] for v in violations)


# ── item 2b: existing project cannot GAIN unsupported accomplishments ────
def test_real_project_cannot_gain_unsupported_accomplishments():
    """Keeping a legitimate name must not authorise new claims beneath it.

    The name check alone would pass here, which is exactly why numeric/entity
    checks must both run. This test asserts the ACCOMPLISHMENT is caught.
    """
    ledger = Ledger.from_dict({
        "profile_id": "has-projects",
        "companies": ["Cascade Power Solutions, LLC"],
        "claims": [{"fact_id": "f1", "value": "30", "unit": "percent"}],
    })
    approved = ["Cascade Power Solutions, LLC", "Commercial Building Retrofit"]

    # Legitimate project name, invented accomplishment beneath it.
    boosted = (
        "## Projects\n"
        "**Commercial Building Retrofit** — AutoCAD, ETAP\n"
        "- Increased tenant revenue by 87% and cut costs by 42% annually\n"
    )
    # Entity check alone would accept the name.
    entity_v = check_entities(boosted, ledger, allowed_entities=approved)
    assert not entity_v, "control: the NAME itself is legitimate"

    # But the truth check must reject the invented accomplishments.
    from rctb.ledger import check_truth
    truth = check_truth(boosted, ledger, allowed_entities=approved)
    assert not truth.passed, (
        "legitimate project name authorised unsupported accomplishments"
    )
    surfaces = {v["text"] for v in truth.tier_a_violations}
    assert any("87" in s for s in surfaces)


def test_real_project_with_supported_accomplishment_passes():
    ledger = Ledger.from_dict({
        "profile_id": "has-projects",
        "companies": ["Cascade Power Solutions, LLC"],
        "claims": [{"fact_id": "f1", "value": "30", "unit": "percent"}],
    })
    ok = (
        "## Projects\n"
        "**Commercial Building Retrofit** — AutoCAD, ETAP\n"
        "- Reduced diagnostic time by 30% across the retrofit\n"
    )
    from rctb.ledger import check_truth
    truth = check_truth(ok, ledger,
                        allowed_entities=["Commercial Building Retrofit"])
    assert truth.passed, truth.tier_a_violations
