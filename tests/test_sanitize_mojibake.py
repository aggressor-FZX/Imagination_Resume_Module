"""DOCX-01/EXPORT-01 regression: sanitize_export_text must repair the
double-encoded UTF-8 (mojibake) sequences found in the stored live run-3
output (analysis_jobs id 3), where e.g. "04/2025 – Present" was stored as
"04/2025 â€“ Present" (U+00E2 U+0080 U+0093).

The exact byte sequences below were read from the DB (hex c3a2c280c293...).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resume_builder.build_resume_from_export import sanitize_export_text


def test_mojibake_en_dash_from_stored_run3():
    # hex from DB: 30342f3230323520c3a2c280c2932050726573656e74
    s = "04/2025 \u00e2\u0080\u0093 Present"
    assert sanitize_export_text(s) == "04/2025 - Present"


def test_mojibake_nonbreaking_hyphen():
    # "multi‑source" stored as "multiâ€'source"
    s = "multi\u00e2\u0080\u0091source law\u00e2\u0080\u0091enforcement"
    assert sanitize_export_text(s) == "multi-source law-enforcement"


def test_mojibake_narrow_nbsp():
    # "10 TB+" stored as "10â€¯TB+"
    s = "10\u00e2\u0080\u00afTB+"
    assert sanitize_export_text(s) == "10-TB+"


def test_mojibake_right_quote():
    # "NOAA’s" stored as "NOAAâ€™s"
    s = "NOAA\u00e2\u0080\u0099s"
    assert sanitize_export_text(s) == "NOAA's"


def test_clean_unicode_dash_still_works():
    # A genuine U+2013 en dash (not mojibake) maps through the normal path.
    assert sanitize_export_text("clean \u2013 dash") == "clean -- dash"


def test_mojibake_repair_runs_before_c1_remap():
    # If the C1 remap ran first, U+0080 would become "€" and U+0093 would
    # become a quote, breaking the sequence. Verify the full pipeline keeps
    # the intended character.
    s = "A\u00e2\u0080\u0093B"
    assert sanitize_export_text(s) == "A-B"