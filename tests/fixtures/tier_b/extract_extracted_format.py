"""Build the portable extracted-format regression fixture from the preserved capture.

The preserved capture in ``.live-run-2026-11-02/`` stays immutable. This script reads it
and writes a small, versioned, committed fixture containing only what the regression tests
need. Nothing in the source is modified.

Why a fixture rather than reading the capture
---------------------------------------------

The capture lives outside git on one machine. The regression these tests protect — "a
source extracted from a real .docx must be read like any other source" — has to run from a
clean checkout. The extracted text also cannot be synthesised from the hand-written
Markdown fixtures, because the whole defect was that the extracted shape lacks the markers
those fixtures carry.

What is copied is the SOURCE text and the WRITER DRAFT, verbatim, with a SHA-256 per blob.
Nothing derived from the reviewer is copied: the original reviewer's line IDs and verdicts
were produced against this exact draft under the old rule order, so replaying them against
patched output would be a false claim of faithfulness. Downstream assertions use explicitly
scripted answers instead.

Usage::

    python tests/fixtures/tier_b/extract_extracted_format.py           # verify
    python tests/fixtures/tier_b/extract_extracted_format.py --write   # rewrite
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "extracted_format_excerpt.json"
CAPTURE = (HERE.parents[2].parent
           / ".live-run-2026-11-02" / "captures" / "run12"
           / "83e2eb3f-0fec-4e3d-adc9-c20b43981be0.json")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build() -> dict:
    if not CAPTURE.exists():
        raise SystemExit(f"source capture not present: {CAPTURE}")
    data = json.loads(CAPTURE.read_text(encoding="utf-8"))

    sources = {item["source_id"]: item["text"] for item in data["sources"]}
    draft = data["draft"]

    # The fixture is only meaningful if the source really is the extracted shape: no
    # emphasis markers, and a bare heading. Assert it rather than trusting the filename.
    for source_id, text in sources.items():
        if "**" in text or text.lstrip().startswith("#"):
            raise SystemExit(f"{source_id} carries Markdown markers; not the extracted shape")
    if not sources["document-0"].lstrip().startswith("Professional Summary"):
        raise SystemExit("document-0 does not begin with a bare heading")

    return {
        "fixture_version": 1,
        "provenance": {
            "kind": "extracted_excerpt",
            "source_capture": "83e2eb3f-0fec-4e3d-adc9-c20b43981be0.json",
            "source_capture_sha256":
                "b2d4b11e25222764b957af70cee6651b24e9b089cf24cd0764b3c439dbc9e67f",
            "note": (
                "Verbatim source text and writer draft from run 12. The extracted .docx "
                "shape carries NO Markdown emphasis or heading markers, which is the "
                "defect this fixture preserves. No reviewer line IDs or verdicts are "
                "included: they were produced against this draft under the old rule "
                "order and are not a faithful replay input for patched output."
            ),
        },
        "sources": {sid: {"text": text, "sha256": _sha(text)}
                    for sid, text in sources.items()},
        "draft": {"text": draft, "sha256": _sha(draft)},
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
    print("fixture matches the capture" if existing == fixture
          else "fixture DIFFERS from what the capture would produce")
    return 0 if existing == fixture else 1


if __name__ == "__main__":
    raise SystemExit(main())
