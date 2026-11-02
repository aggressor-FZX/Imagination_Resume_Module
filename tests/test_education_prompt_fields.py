from stages.star_editor import StarEditor


def test_editor_prompt_retains_parser_field_and_end_date():
    editor = StarEditor(None)
    prompt = editor._create_user_prompt(
        [], "mid", [], education=[{
            "degree": "Associate of Applied Science",
            "institution": "Example Community College",
            "field": "Business Administration",
            "end_date": "05/2017",
        }],
    )
    assert "Associate of Applied Science" in prompt
    assert "Example Community College" in prompt
    assert "Business Administration" in prompt
    assert "05/2017" in prompt