"""Tier-A claim verification, usable outside the editor stage.

WHY THIS MODULE EXISTS (owner item 1)

Tier-A verification used to live only inside `StarEditor._validate_metrics_against_input`.
That meant a route could only report `metrics_support` if it happened to run the
editor. Measured: `POST /api/analysis/multi-file` assembles its final markdown
from `run_synthesis_async`, which never runs the editor, so its response carried
no verdict at all. The publication gate correctly blocked (`verification_missing`)
— but the route had no way to positively verify, so a legitimate multi-file run
could never be published. The only honest outcomes were "blocked" or "unverified".

The fix is NOT to relax the gate. It is to make verification available to every
route that produces a final document. Extracting the checker into its own module
lets the editor stage and the route share one implementation, so there is still
exactly one definition of "Tier A".

Coverage is deliberately narrow and the verdict string says so: only numeric
claims are compared against source text. Employer/date/entity checks live in the
gate and in the ledger; semantic claims (invented responsibility, invented
technology) are NOT checked here. Hence "PASS (tier A only)" — a bare PASS would
overstate what was verified.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Metric surfaces checked for source support. Value in group 1, unit in group 2
# where a unit exists. Applied to the WHOLE document, not just bullet lines —
# checking only lines starting with "-" missed invented Summary/Skills numbers.
OUTPUT_METRIC_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*'
    r'(%|percent|\$|k\b|thousand|million|billion|hours?|days?|weeks?|months?|'
    r'years?|man[\- ]?hours|vessels?|TB|GB|MB|users?|records?|projects?|teams?|'
    r'employees?|x\b)',
    re.IGNORECASE,
)

PASS_VERDICT = "PASS (tier A only)"
FAIL_VERDICT = "FAIL"


def unsupported_metrics(
    markdown: Optional[str],
    source_text: Optional[str],
) -> List[Dict[str, Any]]:
    """Return numeric claims in `markdown` with no support in `source_text`.

    Does not mutate the markdown. Deleting facts from finished prose is the
    wrong remedy (it destroys correct content and hides the problem); reporting
    them lets the gate block and a human review.
    """
    text = markdown or ""
    source_lower = (source_text or "").lower()
    orig_numeric_tokens = {
        m.group(0) for m in re.finditer(r'\d+(?:\.\d+)?', source_lower)
    }

    found: List[Dict[str, Any]] = []
    if not text:
        return found

    for match in OUTPUT_METRIC_RE.finditer(text):
        value = match.group(1)
        unit = (match.group(2) or "").lower().replace(" ", "").replace("-", "")
        surface = match.group(0).strip()

        supported = False
        if source_lower:
            variants = [
                f"{value}{unit}", f"{value} {unit}",
                f"{value}%", f"{value} percent",
                f"{value}k", f"{value} thousand",
                f"{value}m", f"{value} million",
                f"{value}b", f"{value} billion",
                f"{value} vessels", f"{value} hours", f"{value} days",
                f"{value} weeks", f"{value} months", f"{value} years",
                f"{value} tb", f"{value} gb",
                (f"{int(float(value)):,}" if '.' in value
                 else f"{int(value):,}"),
            ]
            supported = any(v in source_lower for v in variants) or (
                value in orig_numeric_tokens
            )

        if not supported:
            found.append({
                "surface": surface,
                "value": value,
                "unit": unit,
                "reason": (
                    "no_candidate_evidence" if source_lower
                    else "no_source_text_supplied"
                ),
            })
    return found


def verify_metrics(
    markdown: Optional[str],
    source_text: Optional[str],
) -> Dict[str, Any]:
    """Produce a `metrics_support` block for the publication gate.

    `checked` is True only when source text was actually supplied. A route that
    cannot supply the source must not claim it verified anything: the gate
    treats PASS+checked=False as VERIFICATION_NOT_RUN and blocks, which is the
    correct outcome for "we could not check".
    """
    unsupported = unsupported_metrics(markdown, source_text)
    checked = bool((source_text or "").strip())
    return {
        "verdict": FAIL_VERDICT if unsupported else PASS_VERDICT,
        "unsupported_metrics": unsupported,
        "checked": checked,
    }
