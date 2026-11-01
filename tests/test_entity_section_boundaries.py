from approved_resume import unsupported_entities


def test_skill_categories_after_education_are_not_entities():
    markdown = (
        "## Education\n**Example College**\n- Degree\n"
        "## Technical Skills\n**Machine Learning & AI:** Python\n"
    )
    assert unsupported_entities(markdown, "Example College Degree Python") == []


def test_invented_employer_remains_blocked_after_skills_section():
    markdown = (
        "## Technical Skills\n**Machine Learning & AI:** Python\n"
        "## Professional Experience\n**Invented Corporation**\n- Built systems.\n"
    )
    assert unsupported_entities(markdown, "Example College Python") == ["Invented Corporation"]