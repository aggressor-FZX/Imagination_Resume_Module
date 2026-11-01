from approved_resume import export_payload
from stages.star_editor import StarEditor


def test_bold_employer_with_location_remains_a_renderable_block():
    markdown = (
        "## Professional Experience\n"
        "**Example Company** -- Example City, WA\n"
        "*Electrician | 2020 - Present*\n"
        "- Installed electrical systems.\n\n"
        "## Education\n**Example College**\n- Electrical Technology\n"
    )
    output = StarEditor(None)._normalize_section_boundaries(markdown)
    assert "\n**Example Company** -- Example City, WA\n" in output
    payload = export_payload(output)
    assert len(payload["experience"]) == 1
    assert "Example Company" in payload["experience"][0]["title"]
    assert any("Installed electrical systems" in bullet for bullet in payload["experience"][0]["bullets"])