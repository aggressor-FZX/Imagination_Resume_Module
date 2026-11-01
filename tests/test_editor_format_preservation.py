"""Synthetic regression tests for final resume formatting and verification."""

import json
import asyncio

from stages.star_editor import StarEditor


MARKDOWN = (
    "## Professional Experience\n"
    "**Example Company**\n"
    "*Engineer*\n"
    "- Built reliable systems.\n\n"
    "## Education\n"
    "**Example College**\n"
    "- Bachelor of Science\n\n"
    "## Skills\nGo\n"
)


def test_placeholder_cleanup_preserves_line_boundaries():
    editor = StarEditor(None)
    actual = editor._remove_placeholders(MARKDOWN + "\n*Dates | Location*\n")
    assert "\n\n## Education\n" in actual
    assert "\n## Skills\nGo\n" in actual
    assert "Dates | Location" not in actual


def test_existing_headings_survive_normalization():
    editor = StarEditor(None)
    actual = editor._normalize_section_boundaries(editor._remove_placeholders(MARKDOWN))
    assert "## Professional Experience\n" in actual
    assert "## Education\n" in actual
    assert "## \n" not in actual


def test_short_sections_survive_when_other_sections_are_added():
    editor = StarEditor(None)
    actual = editor._ensure_all_sections(
        {"final_markdown": MARKDOWN},
        certifications=[{"name": "Synthetic Credential", "issuer": "Example"}],
    )["final_markdown"]
    assert "## Skills\nGo" in actual
    assert "## Certifications" in actual


def test_appended_metrics_are_checked_after_section_repair():
    class FakeClient:
        async def call_llm_async(self, **_kwargs):
            return json.dumps({"final_markdown": MARKDOWN, "final_plain_text": MARKDOWN})

    editor = StarEditor(FakeClient())
    result = asyncio.run(editor.polish(
        {"rewritten_experiences": []}, {}, original_resume_text="Example College Engineer",
        certifications=[{"name": "Synthetic Credential", "year": "99 hours"}],
    ))
    assert "99 hours" in result["final_markdown"]
    assert result["metrics_support_checked"] is True
    assert result["unsupported_metrics"]