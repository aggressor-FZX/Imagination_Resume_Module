#!/usr/bin/env python3
"""Record one real-model synthetic correction case for offline replay.

Why this exists: the correction adapter's failures were not replayable, because the
model's answers were deliberately never logged. A ``source_completeness_uncertain``
in production therefore cost a paid run to reproduce. This tool records the
provider exchanges for one explicitly synthetic case so the decision can be
recomputed offline, through the real adapter and the real validator.

Safety properties:

* the source and draft must both contain ``SYNTHETIC-ONLY``, and the recorder
  refuses a request that carries no such marker;
* the capture is written ``0600`` and ``captures/`` is git-ignored;
* nothing is printed except counts, cost and the output path -- never text.

This spends provider money, so it is never run automatically and never by tests.

Usage::

    export PROBE_OR_KEY=sk-or-...
    PYTHONPATH=. python tools/capture_correction_case.py <case-id>

Then replay it with no network at all::

    PYTHONPATH=. python -m pytest tests/test_correction_capture_replay.py
"""
from __future__ import annotations

import asyncio
import os
import sys

import aiohttp

import correction_capture as cc
from correction_provider import CorrectionProvider
from source_correction import SourceDocument, correct


M = cc.SYNTHETIC_MARKER

SOURCE = f"""## Professional Experience
**{M} Cascade Power Solutions, LLC** -- Example City, WA
*Journeyman Electrician | 03/2020 - Present*
- Installed and maintained commercial electrical systems across 40+ job sites.
- Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices.
- Cut rework by 18% by standardizing panel labeling and inspection checklists.

## Education
**{M} Example Technical College** -- Example City, WA
*Associate of Applied Science, Electrical Systems Technology | 2019*
"""

# A realistic rewrite: paraphrased bullets, reordered narrative, added sections.
DRAFT = f"""## Professional Summary
{M} Journeyman electrician with 8+ years of commercial electrical experience across power, industrial, and facilities environments.

## Professional Experience
**{M} Cascade Power Solutions, LLC** -- Example City, WA
*Journeyman Electrician | 03/2020 - Present*
- Delivered installation and maintenance for commercial electrical systems on more than 40 sites.
- Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices.
- Reduced rework by 18% by standardizing panel labeling and inspection checklists.

## Education
**{M} Example Technical College** -- Example City, WA
*Associate of Applied Science, Electrical Systems Technology | 2019*

## Skills
- Commercial electrical systems, conduit bending, panel labeling
- NFPA 70E, lockout/tagout, inspection checklists
"""


async def main(case_id: str) -> int:
    key = os.getenv("PROBE_OR_KEY", "").strip()
    if not key:
        print("PROBE_OR_KEY is not set", file=sys.stderr)
        return 2
    # The production setting; recorded so the capture reflects live behaviour.
    os.environ.setdefault("CORRECTION_PROVIDER_DISCLOSURE", "1")

    capture = cc.Capture(case_id=case_id, source_text=SOURCE, draft_text=DRAFT,
                         notes="real-model synthetic capture for offline replay")
    async with aiohttp.ClientSession() as inner:
        provider = CorrectionProvider(key, cc.RecordingSession(inner, capture))
        result = await correct(DRAFT, (SourceDocument("resume", SOURCE),), provider)
        # Let any cancelled provider coroutine record its final usage state.
        await asyncio.sleep(0)

    capture.results.append({
        "status": result.status, "reason": result.reason,
        "reviews": result.review_calls, "repairs": result.repair_calls,
        "phases": [call["phase"] for call in provider.usage["calls"]],
        "tokens": provider.usage["total_tokens"],
        "cost_usd": provider.usage["estimated_cost_usd"],
    })
    path = cc.save_capture(capture)
    print(f"case={case_id} status={result.status} reason={result.reason}")
    print(f"requests={len(capture.calls)} tokens={provider.usage['total_tokens']} "
          f"cost_usd={provider.usage['estimated_cost_usd']:.6f}")
    print(f"capture={path}")
    return 0 if result.status == "approved" else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(sys.argv[1])))
