"""Extract a minimal portable fixture from the preserved run-10 capture.

The preserved capture in ``.live-run-2026-11-02/`` is immutable evidence and stays
where it is. This script reads it and writes a small, versioned, self-contained copy
containing ONLY the strings the required regression tests need. Nothing is modified in
the source.

Why extraction rather than referencing the capture directly
-----------------------------------------------------------

Tests that read the preserved capture can only run on one machine. The required
regressions — the ones that must run from a clean checkout — cannot depend on an
optional, uncommitted directory. The extracted fixture is committed, versioned, and
contains no more than the assertions require.

Each value is copied verbatim and its SHA-256 is recorded so drift is detectable.

Usage::

    python tests/fixtures/tier_b/extract_from_capture.py          # verify only
    python tests/fixtures/tier_b/extract_from_capture.py --write   # rewrite the fixture
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "run10_excerpt.json"
CAPTURE = (HERE.parents[2].parent
           / ".live-run-2026-11-02" / "captures" / "run10"
           / "850ee3ea-ca2e-4210-92b4-0e57ed5149eb.json")

# The minimum the required tests need. Every entry is a verbatim slice of the
# capture; the extraction asserts that, so a fixture cannot silently drift into
# being invented text.
WANTED_SOURCES = ("document-0", "document-1")

WANTED_DRAFT_LINES = [
    "Installed and maintained commercial electrical systems across 40+ job sites, "
    "ensuring compliance with NFPA 70E standards.",
    "Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices, "
    "improving workforce safety and code adherence.",
    "Cut rework by 18% by standardizing panel labeling and inspection checklists, "
    "increasing installation efficiency and quality control.",
    "Completed 8,000 hours of supervised field work toward journeyman licensure, "
    "gaining hands\u2011on electrical construction experience.",
    "Maintained electrical distribution systems and HVAC controls for 12 commercial "
    "buildings, ensuring continuous operations.",
    "Reduced unplanned downtime by 22% through implementation of preventive "
    "maintenance schedules, enhancing system reliability.",
]

WANTED_SOURCE_LINES = [
    "Installed and maintained commercial electrical systems across 40+ job sites.",
    "Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices.",
    "Cut rework by 18% by standardizing panel labeling and inspection checklists.",
    "Completed 8,000 hours of supervised field work toward licensure.",
    "Maintained electrical distribution and HVAC controls for 12 commercial buildings.",
    "Reduced unplanned downtime by 22% through a preventive maintenance schedule.",
    "SYNTHETIC-ONLY Facilities maintenance technician focused on preventive maintenance of",
    "SYNTHETIC-ONLY Journeyman electrician with 8+ years of commercial and industrial electrical",
]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build() -> dict:
    if not CAPTURE.exists():
        raise SystemExit(f"source capture not present: {CAPTURE}")
    data = json.loads(CAPTURE.read_text(encoding="utf-8"))

    sources = {item["source_id"]: item["text"] for item in data["sources"]}
    draft = data["draft"]

    # Verify every wanted value really occurs in the capture before copying it.
    for line in WANTED_SOURCE_LINES:
        if not any(line in text for text in sources.values()):
            raise SystemExit(f"source line not found in capture: {line!r}")
    for line in WANTED_DRAFT_LINES:
        if line not in draft:
            raise SystemExit(f"draft line not found in capture: {line!r}")

    return {
        "fixture_version": 1,
        "provenance": {
            "kind": "extracted_excerpt",
            "source_capture": "850ee3ea-ca2e-4210-92b4-0e57ed5149eb.json",
            "source_capture_sha256":
                "c141846a0d6ef29242e4cf24095978c52c1a88ae671bc84d639de8db4a98713f",
            "note": (
                "Verbatim excerpt. The full capture is immutable evidence and is NOT "
                "committed; this file contains only the strings the required "
                "regressions assert on."
            ),
        },
        "sources": {sid: sources[sid] for sid in WANTED_SOURCES},
        "draft_lines": [
            {"text": line, "sha256": _sha(line)} for line in WANTED_DRAFT_LINES
        ],
        "source_lines": [
            {"text": line, "sha256": _sha(line)} for line in WANTED_SOURCE_LINES
        ],
    }


def main() -> int:
    fixture = build()
    if "--write" in sys.argv:
        FIXTURE.write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
        print(f"wrote {FIXTURE}")
        return 0
    if not FIXTURE.exists():
        print("fixture missing; run with --write")
        return 1
    existing = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if existing == fixture:
        print("fixture matches the capture")
        return 0
    print("fixture DIFFERS from what the capture would produce")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
