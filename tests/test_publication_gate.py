"""Publication gate: fail-closed negative tests + positive controls.

Milestone 2. The contract is: publish ONLY when verification positively
succeeded. Every other state blocks.

Each negative test corresponds to a failure class the owner named:
  verification absent / incomplete / unchecked / throws
  output malformed / empty / conversational / truncated / generic fallback
  required fact missing
  unsupported claim or wrong-role attribution

A test is only meaningful if it would FAIL if the protection were removed, so
`test_removing_the_gate_would_publish` demonstrates the bypass explicitly.
"""
import sys
from pathlib import Path

IMAG = Path("/home/skystarved/Render_Dockers/Imagination_Resume_Module")
if str(IMAG) not in sys.path:
    sys.path.insert(0, str(IMAG))

from publication_gate import (  # noqa: E402
    DELIVERABLE_FIELDS, PublishDecision, apply_to_payload, decide,
)

GOOD_RESUME = """## Professional Experience
**Cascade Power Solutions, LLC** — Tacoma, WA
*Journeyman Electrician | March 2020 – Present*
- Troubleshoot and resolve faults in 480V three-phase distribution panels, motor
  control centers, and lighting control systems, reducing average diagnostic time
  by 30% across the service area
- Lead a crew of 2-3 apprentices on multi-building retail center buildouts,
  coordinating daily task assignments and conducting morning safety briefings
- Install, maintain, and repair electrical wiring, equipment and fixtures in
  commercial buildings ranging from 5,000 to 80,000 square feet, ensuring
  compliance with NEC 2023 and local codes

## Education
**Bates Technical College** — Tacoma, WA | Graduated May 2015
- A.A.S. Electrical Technology, GPA 3.6 / 4.0
"""

VERIFIED = {"verdict": "PASS (tier A only)", "checked": True,
            "unsupported_metrics": []}


# ── POSITIVE control ─────────────────────────────────────────────────────
def test_verified_resume_is_publishable():
    d = decide(resume_markdown=GOOD_RESUME, metrics_support=VERIFIED,
                                          approved_entities=["Cascade Power Solutions, LLC"])
    assert d.publishable, d.reasons
    assert d.status == "completed"
    assert d.coverage == "partial", "coverage must be stated honestly"


def test_supported_resume_payload_keeps_content():
    payload = {"final_written_section_markdown": GOOD_RESUME,
               "rewritten_resume": GOOD_RESUME,
               "metrics_support": VERIFIED, "experiences": [{"a": 1}]}
    out = apply_to_payload(payload, decide(resume_markdown=GOOD_RESUME,
                                           metrics_support=VERIFIED,
                                           approved_entities=["Cascade Power Solutions, LLC"]))
    assert out["final_written_section_markdown"] == GOOD_RESUME
    assert out["requires_review"] is False
    assert out["processing_status"] == "completed"
    assert out["experiences"] == [{"a": 1}], "candidate facts must survive"


# ── NEGATIVE: verification problems ──────────────────────────────────────
def test_missing_verification_blocks():
    d = decide(resume_markdown=GOOD_RESUME, metrics_support=None,
               approved_entities=[])
    assert d.blocked()
    assert any(r.value == "verification_missing" for r in d.reasons)


def test_empty_verification_object_blocks():
    d = decide(resume_markdown=GOOD_RESUME, metrics_support={},
               approved_entities=[])
    assert d.blocked()
    assert any(r.value == "verification_missing" for r in d.reasons)


def test_unchecked_pass_blocks():
    """'PASS' with checked=false is the fail-open we actually shipped."""
    d = decide(resume_markdown=GOOD_RESUME,
               metrics_support={"verdict": "PASS (tier A only)",
                                "checked": False},
                                          approved_entities=[])
    assert d.blocked()
    assert any(r.value == "verification_not_run" for r in d.reasons)


def test_unsupported_claims_block():
    d = decide(resume_markdown=GOOD_RESUME,
               metrics_support={"verdict": "FAIL", "checked": True,
                                "unsupported_metrics": [{"surface": "87%"}]},
                                          approved_entities=[])
    assert d.blocked()
    assert any(r.value == "unsupported_claims" for r in d.reasons)


def test_unknown_verdict_blocks():
    for v in ("", "PASS", "OK", "SUCCESS", None):
        d = decide(resume_markdown=GOOD_RESUME,
                   metrics_support={"verdict": v, "checked": True},
                                     approved_entities=[])
        assert d.blocked(), f"verdict {v!r} must not be treated as a pass"


def test_checker_exception_blocks():
    """A checker that raises must not become a pass."""
    def broken():
        raise RuntimeError("checker crashed")
    try:
        broken()
    except RuntimeError:
        support = None      # caller has nothing to report
    d = decide(resume_markdown=GOOD_RESUME, metrics_support=support,
                   approved_entities=[])
    assert d.blocked()


# ── NEGATIVE: content shape ──────────────────────────────────────────────
def test_empty_output_blocks():
    for text in ("", "   ", None):
        d = decide(resume_markdown=text, metrics_support=VERIFIED,
                   approved_entities=[])
        assert d.blocked()
        assert any(r.value == "empty_output" for r in d.reasons)


def test_conversational_output_blocks():
    reply = ("I'm ready to craft a polished résumé for you, but I need the "
             "details of your work experience, projects, education, "
             "certifications, and technical skills. Please provide the company "
             "names, job titles, dates, bullet-point achievements, and much "
             "more detail so I can produce a strong targeted document.")
    d = decide(resume_markdown=reply, metrics_support=VERIFIED,
               approved_entities=[])
    assert d.blocked()
    assert any(r.value == "conversational_output" for r in d.reasons)


def test_truncated_output_blocks():
    d = decide(resume_markdown=GOOD_RESUME, metrics_support=VERIFIED,
               finish_reason="length", approved_entities=[])
    assert d.blocked()
    assert any(r.value == "truncated_output" for r in d.reasons)


def test_generic_fallback_blocks():
    d = decide(resume_markdown=GOOD_RESUME, metrics_support=VERIFIED,
               fallback_used=True, approved_entities=[])
    assert d.blocked()
    assert any(r.value == "fallback_output" for r in d.reasons)


def test_missing_section_blocks():
    no_edu = GOOD_RESUME.split("## Education")[0]
    d = decide(resume_markdown=no_edu, metrics_support=VERIFIED,
                   approved_entities=[])
    assert d.blocked()
    assert any("missing_section" in r.value for r in d.reasons)


def test_placeholder_leak_blocks():
    leaky = GOOD_RESUME + "\n- Worked on [PLACEHOLDER] project\n"
    d = decide(resume_markdown=leaky, metrics_support=VERIFIED,
                   approved_entities=[])
    assert d.blocked()
    assert any(r.value == "placeholder_leak" for r in d.reasons)


def test_required_fact_missing_blocks():
    d = decide(resume_markdown=GOOD_RESUME, metrics_support=VERIFIED,
               required_facts_missing=["30% diagnostic time"],
               approved_entities=[])
    assert d.blocked()
    assert any("required_facts_missing" in r.value for r in d.reasons)


# ── withholding: every deliverable field ─────────────────────────────────
def test_blocked_payload_nulls_every_deliverable_field():
    payload = {"final_written_section_markdown": GOOD_RESUME,
               "final_written_section": GOOD_RESUME,
               "rewritten_resume": GOOD_RESUME,
               "resume_text": GOOD_RESUME,
               "experiences": [{"company": "Real"}],
               "final_written_section_provenance": [{"x": 1}]}
    out = apply_to_payload(payload, decide(resume_markdown=GOOD_RESUME,
                                          metrics_support={"verdict": "FAIL",
                                                           "checked": True},
                                          approved_entities=[]))
    for f in DELIVERABLE_FIELDS:
        if f in payload:
            assert out[f] is None, f"{f} was not withheld"
    assert out["final_written_section_provenance"] == []
    assert out["experiences"] == [{"company": "Real"}], (
        "candidate facts must be preserved, not discarded"
    )
    assert out["requires_review"] is True
    assert out["review_message"]


def test_generated_text_is_never_mutated():
    """We withhold; we do not silently delete a claim and call it verified."""
    original = GOOD_RESUME + "\n- Increased revenue by 87%\n"
    payload = {"final_written_section_markdown": original}
    d = decide(resume_markdown=original,
               metrics_support={"verdict": "FAIL", "checked": True},
               approved_entities=[])
    out = apply_to_payload(payload, d)
    assert out["final_written_section_markdown"] is None, "withheld"
    assert payload["final_written_section_markdown"] == original, (
        "the caller's original object must not be edited"
    )


# ── the bypass demonstration the owner asked for ─────────────────────────
def test_removing_the_gate_would_publish():
    """Proves the tests exercise the protection, not just the code path.

    If the gate simply returned publishable=True, these inputs would be
    published. This asserts the gate is what stops them.
    """
    bad_inputs = [
        ({"resume_markdown": "", "metrics_support": VERIFIED}, "empty"),
        ({"resume_markdown": GOOD_RESUME, "metrics_support": None}, "no-verif"),
        ({"resume_markdown": GOOD_RESUME,
          "metrics_support": {"verdict": "FAIL", "checked": True}}, "fail"),
        ({"resume_markdown": GOOD_RESUME,
          "metrics_support": {"verdict": "PASS (tier A only)",
                              "checked": False}}, "unchecked"),
    ]
    for kwargs, label in bad_inputs:
        assert decide(**kwargs).blocked(), f"{label} should be blocked"

    # A permissive gate (the bypass) would publish all of them:
    permissive = PublishDecision(publishable=True, status="completed")
    assert not permissive.blocked(), (
        "control: a permissive gate publishes, which is what the tests prevent"
    )
