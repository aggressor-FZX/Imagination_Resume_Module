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


def test_unanchored_degree_and_school_are_not_combined():
    markdown = "## Education\n**Associate of Applied Science, Example Community College**\n"
    source = "EDUCATION\nExample Community College | May 2017\nAssociate of Applied Science in Business Administration\n"
    # Parsed field co-occurrence is not independent proof of association.
    assert unsupported_entities(markdown, source)


def test_invented_or_cross_document_education_pair_remains_blocked():
    markdown = "## Education\n**Associate of Applied Science, Example Community College**\n"
    source = "EDUCATION\nAssociate of Applied Science\n\n---\n\nEDUCATION\nExample Community College\n"
    assert unsupported_entities(markdown, source)
    assert unsupported_entities(markdown, "EDUCATION\nAssociate of Applied Science\nOther College\n")


def test_cross_entry_education_pair_is_not_authorized():
    source = ("EDUCATION\nExample Technical College | May 2015\nA.A.S. Electrical Technology\n\n"
              "Example Community College | May 2017\nAssociate of Applied Science in Business Administration\n")
    markdown = "## Education\n**Associate of Applied Science, Example Technical College**\n"
    assert unsupported_entities(markdown, source)


def test_incomplete_degree_is_not_approved_as_completed():
    markdown = "## Education\n**Associate of Applied Science, Example Community College**\n"
    source = "EDUCATION\nExample Community College\nCoursework toward Associate of Applied Science\n"
    assert unsupported_entities(markdown, source)