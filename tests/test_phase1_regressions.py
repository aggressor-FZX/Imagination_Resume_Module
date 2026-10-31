"""Phase 1 regression suite — RCTB §9 cases R-1 .. R-9.

Every test here MUST fail before its corresponding Phase 1 fix and pass after.
That ordering is the proof required by SPEC.v2 §9 / §9.4.

Run from the Imagination_Resume_Module repo root:
    python3 -m pytest tests/test_phase1_regressions.py -v
"""
import json
import os
import re
import sys

import pytest

# The module requires settings at import time; Phase 1 item 1.7 removes that.
# Until then, supply explicit dummy values (no real credentials).
os.environ.setdefault("OPENROUTER_API_KEY_1", "test-dummy-key-1")
os.environ.setdefault("OPENROUTER_API_KEY_2", "test-dummy-key-2")
os.environ.setdefault("IMAGINATOR_AUTH_TOKEN", "test-dummy-token")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# R-1: sanitizer must preserve line structure (evidence C4)
# ─────────────────────────────────────────────────────────────────────────────
RESUME_FOR_SANITIZE = (
    "MARCUS D. HOLLOWAY\n"
    "Licensed Journeyman Electrician\n"
    "Tacoma, WA 98405\n"
    "\n"
    "PROFESSIONAL EXPERIENCE\n"
    "\n"
    "Cascade Power Solutions, LLC - Tacoma, WA | March 2020 - Present\n"
    "- Install and repair electrical wiring in commercial buildings\n"
    "- Troubleshoot 480V panels, reducing diagnostic time by 30%\n"
)


def test_r1_sanitizer_preserves_line_count():
    """R-1: blank lines and section boundaries survive sanitization."""
    from imaginator_new_integration import sanitize_resume_pii

    # Trailing newline is consumed by .strip(); compare like-for-like.
    before = RESUME_FOR_SANITIZE.rstrip("\n")
    after = sanitize_resume_pii(RESUME_FOR_SANITIZE)

    assert len(after.split("\n")) == len(before.split("\n")), (
        f"line count changed {len(before.split(chr(10)))} -> {len(after.split(chr(10)))}"
    )
    assert after.count("\n\n") == before.count("\n\n"), "blank lines were destroyed"


def test_r1_section_header_stays_on_its_own_line():
    """R-1: 'PROFESSIONAL EXPERIENCE' must not be glued to the prior line."""
    from imaginator_new_integration import sanitize_resume_pii

    after = sanitize_resume_pii(RESUME_FOR_SANITIZE)
    lines = [ln.strip() for ln in after.split("\n")]
    assert "PROFESSIONAL EXPERIENCE" in lines, (
        f"section header was merged into another line: {lines[:6]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# R-9: sanitizer must not delete legitimate content (inverse of R-1)
# ─────────────────────────────────────────────────────────────────────────────
def test_r9_sanitizer_keeps_career_content():
    """R-9: careers facts survive; only PII is removed."""
    from imaginator_new_integration import sanitize_resume_pii

    after = sanitize_resume_pii(RESUME_FOR_SANITIZE)
    for must_keep in [
        "Cascade Power Solutions, LLC",
        "Tacoma, WA",
        "March 2020 - Present",
        "480V three-phase" if "480V three-phase" in RESUME_FOR_SANITIZE else "480V",
        "30%",
        "Journeyman Electrician",
    ]:
        assert must_keep in after, f"legitimate content deleted: {must_keep!r}"


def test_r9_sanitizer_removes_pii():
    """R-9: known PII patterns are removed."""
    from imaginator_new_integration import sanitize_resume_pii

    with_pii = (
        "JANE DOE\n"
        "jane.doe@example.com | (253) 555-0100\n"
        "4821 Pinecrest Drive, Tacoma, WA 98405\n"
        "\n"
        "PROFESSIONAL EXPERIENCE\n"
        "Acme Industries - Tacoma, WA | 2020 - Present\n"
        "- Reduced cycle time by 18%\n"
    )
    after = sanitize_resume_pii(with_pii)
    assert "jane.doe@example.com" not in after
    assert "(253) 555-0100" not in after
    assert "4821 Pinecrest Drive" not in after
    # career evidence retained
    assert "Acme Industries" in after
    assert "18%" in after


# ─────────────────────────────────────────────────────────────────────────────
# R-2: metric extraction must surface full normalized quantity (evidence C5)
# ─────────────────────────────────────────────────────────────────────────────
METRIC_SOURCE_TEXT = (
    "Secured $500K in federal R&D funding. Reduced load time by 40%. "
    "Managed 13 vessels. Saved 30 hours per week. Processed 10+ TB of data."
)

EXPECTED_METRICS = [
    # (value, unit) — the normalized quantity required by SPEC §5.3
    ("500", "K"),        # $500K
    ("40", "%"),         # 40%
    ("13", "vessels"),   # 13 vessels
    ("30", "hours"),     # 30 hours  <- currently truncated to bare "30"
    ("10", "TB"),        # 10+ TB
]


def test_r2_metric_extraction_returns_all_metrics():
    """R-2: all five documented metrics surface as full quantity + unit."""
    from stages.drafter import _extract_metrics  # noqa: WPS433 (may not exist yet)

    found = _extract_metrics(METRIC_SOURCE_TEXT)
    flat = json.dumps(found).lower()
    for value, unit in EXPECTED_METRICS:
        assert value in flat and unit.lower() in flat, (
            f"metric {value}{unit} missing from extraction: {found}"
        )


def test_r2_metric_extraction_keeps_time_unit():
    """R-2: '30 hours' must not degrade to a bare number."""
    from stages.drafter import _extract_metrics

    found = _extract_metrics("Saved 30 hours per week.")
    flat = json.dumps(found).lower()
    assert "hours" in flat, f"time unit lost, got: {found}"


# ─────────────────────────────────────────────────────────────────────────────
# R-3: metric validator must not delete real metrics (evidence C6)
# ─────────────────────────────────────────────────────────────────────────────
def test_r3_validator_does_not_mutate_text():
    """R-3: source-backed metrics survive; no silent prose surgery."""
    from stages.star_editor import StarEditor

    bullet = "- Reduced reporting overhead by 40% for the 13-vessel research fleet"
    markdown = f"## Professional Experience\n**Data Scientist** -- Cogito Metric\n{bullet}\n"

    editor = StarEditor.__new__(StarEditor)  # no __init__: no LLM client needed
    result = editor._validate_metrics_against_input(
        {"final_markdown": markdown}, original_resume_text=""
    )

    out = result["final_markdown"]
    assert "40%" in out, f"real metric deleted: {out!r}"
    assert out == markdown, "text was mutated instead of reported"


def test_r3_validator_reports_unsupported_without_deleting():
    """R-3: unsupported metrics are reported, not stripped."""
    from stages.star_editor import StarEditor

    markdown = "## Experience\n- Increased revenue by 87% overnight\n"
    editor = StarEditor.__new__(StarEditor)
    result = editor._validate_metrics_against_input(
        {"final_markdown": markdown}, original_resume_text="No metrics here."
    )
    # Reporting channel must exist and be non-empty.
    assert result.get("unsupported_metrics"), "no reporting channel for unsupported metrics"
    # And the caller's text is left intact for review.
    assert result["final_markdown"] == markdown


# ─────────────────────────────────────────────────────────────────────────────
# R-5: prompt must not authorise invention (evidence C8)
# ─────────────────────────────────────────────────────────────────────────────
def test_r5_prompt_has_no_invention_licence():
    """R-5: TIER 2 / TIER 3 / METRIC INJECTION / hardcoded metrics are gone."""
    from stages.drafter import _build_drafter_prompts

    system_prompt, user_prompt = _build_drafter_prompts(
        experiences=[{"company": "X", "role": "Y", "duration": "2020-2021",
                      "bullets": ["did work"]}],
        job_ad="Requires 5+ years SQL",
        research_data={},
        resume_text="Reduced by 40%.",
    )
    combined = system_prompt + "\n" + user_prompt

    for banned in ["TIER 2", "TIER 3", "METRIC INJECTION", "13-vessel",
                   "MUST weave", "securing $500K"]:
        assert banned not in combined, f"invention licence still present: {banned}"

    # Positive control: the evidence rule must now be stated.
    lowered = combined.lower()
    assert "evidence" in lowered and ("never estimate" in lowered
                                      or "do not estimate" in lowered), (
        "replacement evidence-only rule missing from prompt"
    )


# ─────────────────────────────────────────────────────────────────────────────
# R-6: date binding by experience_id, not list index (SPEC §9)
# ─────────────────────────────────────────────────────────────────────────────
def test_r6_dates_bind_by_id_not_position():
    """R-6: reordered model output must not swap employer dates."""
    from stages.drafter import Drafter

    originals = [
        {"company": "Alpha Corp", "role": "Senior", "duration": "01/2020 – 06/2022",
         "experience_id": "exp_01", "bullets": ["a"]},
        {"company": "Beta LLC", "role": "Junior", "duration": "07/2022 – Present",
         "experience_id": "exp_02", "bullets": ["b"]},
    ]
    # Model returns them in REVERSED order.
    model_output = {
        "rewritten_experiences": [
            {"company": "Beta LLC", "role": "Junior", "bullets": ["b2"],
             "experience_id": "exp_02"},
            {"company": "Alpha Corp", "role": "Senior", "bullets": ["a2"],
             "experience_id": "exp_01"},
        ],
        "seniority_applied": "mid",
    }

    drafter = Drafter.__new__(Drafter)
    parsed = drafter._parse_response(json.dumps(model_output), originals)

    by_company = {e["company"]: e for e in parsed["rewritten_experiences"]}
    assert by_company["Alpha Corp"]["duration"] == "01/2020 – 06/2022", (
        f"Alpha got wrong dates: {by_company['Alpha Corp']['duration']}"
    )
    assert by_company["Beta LLC"]["duration"] == "07/2022 – Present", (
        f"Beta got wrong dates: {by_company['Beta LLC']['duration']}"
    )
