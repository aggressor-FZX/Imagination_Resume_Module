"""Offline renderer regressions: only explicitly supplied resume content is used."""

from copy import deepcopy

import pytest

from resume_builder.build_resume_from_export import (
    DEFAULT_TEMPLATE,
    ResumePayload,
    ResumeSection,
    _build_section_title,
    build_docx,
    build_fallback_payload,
    generate_section_body,
    latex_escape,
    parse_resume_payload,
    render_template,
    split_location,
    split_name,
)


@pytest.mark.parametrize("parser", [parse_resume_payload, build_fallback_payload])
@pytest.mark.parametrize(
    "source",
    [
        {},
        {"data": {}},
        {"experience": [], "skills": []},
        {"name": "", "location": "", "summary": "", "headline": ""},
    ],
)
def test_empty_payload_has_no_sample_content(parser, source):
    assert parser(source) == ResumePayload(name="")


@pytest.mark.parametrize("parser", [parse_resume_payload, build_fallback_payload])
@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("name", "Supplied Candidate", {"name": "Supplied Candidate"}),
        ("email", "candidate@example.test", {"email": "candidate@example.test"}),
        ("professional_summary", "Supplied profile.", {"summary": "Supplied profile."}),
        ("headline", "Supplied title", {"headline": "Supplied title"}),
        ("user_skills", ["Supplied skill"], {"skills": [("Skills", ["Supplied skill"])]}),
        ("user_skills", "Supplied skill", {"skills": [("Skills", ["Supplied skill"])]}),
        (
            "experiences",
            [{"role": "Supplied role", "responsibilities": ["Supplied duty."]}],
            {"experience": [ResumeSection(title="Supplied role", body_lines=["Supplied duty."])]},
        ),
    ],
)
def test_sparse_alias_payload_never_requires_unrelated_career_fields(parser, field, value, expected):
    fields = {"name": "", **expected}
    assert parser({field: value}) == ResumePayload(**fields)


@pytest.mark.parametrize("text_key", ["resume_text", "rewrittenResume", "approved_markdown"])
def test_unstructured_text_does_not_populate_empty_structured_experience(text_key):
    source = {"experience": [], "skills": [], text_key: "Raw text is not structured career data."}
    assert parse_resume_payload(source) == ResumePayload(name="")


@pytest.mark.parametrize("parser", [parse_resume_payload, build_fallback_payload])
@pytest.mark.parametrize("wrapped", [False, True])
def test_canonical_payload_preserves_only_supplied_content(parser, wrapped):
    source = {
        "name": "Supplied Candidate",
        "location": "Profile City, ZZ",
        "phone": "555-0100",
        "email": "candidate@example.test",
        "linkedin": "linkedin.com/in/supplied-candidate",
        "github": "github.com/supplied-candidate",
        "headline": "Supplied headline",
        "summary": "Supplied summary.",
        "skills": [{"category": "Supplied group", "items": ["Supplied skill"]}],
        "experience": [{
            "role": "Supplied role",
            "company": "Supplied employer",
            "location": "Role City, YY",
            "duration": "2020 -- 2022",
            "bullets": ["Supplied responsibility."],
        }],
        "projects": [{"title": "Supplied project", "bullets": ["Supplied result."]}],
        "education": ["Supplied education"],
        "certifications": ["Supplied certification"],
        "publication": "Supplied publication & coauthors (2024)",
    }
    original = deepcopy(source)
    bundle = {"data": source} if wrapped else source
    assert parser(bundle) == ResumePayload(
        name="Supplied Candidate",
        location="Profile City, ZZ",
        phone="555-0100",
        email="candidate@example.test",
        linkedin="linkedin.com/in/supplied-candidate",
        github="github.com/supplied-candidate",
        headline="Supplied headline",
        summary="Supplied summary.",
        skills=[("Supplied group", ["Supplied skill"])],
        experience=[ResumeSection(
            title="Supplied role -- Supplied employer -- Role City, YY",
            body_lines=["Supplied responsibility."],
            date_string="2020 -- 2022",
        )],
        projects=[ResumeSection(title="Supplied project", body_lines=["Supplied result."])],
        education=["Supplied education"],
        certifications=["Supplied certification"],
        publication="Supplied publication & coauthors (2024)",
    )
    assert source == original


def test_explicit_identity_and_career_aliases_are_retained():
    parsed = parse_resume_payload({
        "full_name": "Alias Candidate",
        "preferred_location": "Alias City, ZZ",
        "phone_number": "555-0101",
        "email_address": "alias@example.test",
        "linkedin_url": "https://linkedin.com/in/alias-candidate",
        "github_url": "https://github.com/alias-candidate",
        "target_job_title": "Alias headline",
        "profile_summary": "Alias summary.",
        "aggregate_skills": ["Alias skill"],
        "professional_experience": [{
            "title": "Alias role",
            "organization": "Alias employer",
            "responsibilities": "First duty.;Second duty.",
            "start_date": "2020",
        }],
        "publication_line": "Alias publication",
    })
    assert parsed == ResumePayload(
        name="Alias Candidate",
        location="Alias City, ZZ",
        phone="555-0101",
        email="alias@example.test",
        linkedin="https://linkedin.com/in/alias-candidate",
        github="https://github.com/alias-candidate",
        headline="Alias headline",
        summary="Alias summary.",
        skills=[("Skills", ["Alias skill"])],
        experience=[ResumeSection(
            title="Alias role -- Alias employer",
            body_lines=["First duty.", "Second duty."],
            date_string="2020",
        )],
        publication="Alias publication",
    )


@pytest.mark.parametrize("splitter", [split_name, split_location])
@pytest.mark.parametrize("value", ["", " \t\n "])
def test_empty_identity_splitters_never_invent_profile_data(splitter, value):
    assert splitter(value) == ("", "")


def test_supplied_identity_splitters_are_unchanged():
    assert split_name("First Middle Last") == ("First", "Middle Last")
    assert split_name("Single") == ("Single", "")
    assert split_location("Profile City, ZZ") == ("Profile City", "ZZ")
    assert split_location("Remote") == ("Remote", "")


@pytest.mark.parametrize(
    "entry,expected",
    [
        ({"role": "Engineer", "company": "Employer"}, "Engineer -- Employer"),
        ({"title": "Engineer", "organization": "Employer"}, "Engineer -- Employer"),
        ({"role": "Engineer"}, "Engineer"),
        ({"company": "Employer"}, "Employer"),
        ({"title_line": "Engineer | Employer | Other City | 2020"}, "Engineer"),
        ({}, "Entry 1"),
    ],
)
def test_role_location_is_appended_without_changing_existing_title_selection(entry, expected):
    assert _build_section_title(entry, "Entry 1") == expected
    assert _build_section_title({**entry, "location": " \t "}, "Entry 1") == expected
    assert _build_section_title({**entry, "location": " Role City, ZZ "}, "Entry 1") == (
        expected + " -- Role City, ZZ"
    )


@pytest.mark.parametrize("section_key", ["experience", "projects"])
def test_role_location_survives_latex_and_docx_without_becoming_profile_location(section_key, tmp_path):
    resume = parse_resume_payload({section_key: [{
        "role": "Supplied role",
        "company": "Supplied employer",
        "location": "Role City, ZZ",
        "bullets": ["Supplied duty."],
    }]})
    assert resume.location == ""
    section = getattr(resume, section_key)[0]
    assert "Supplied role -- Supplied employer -- Role City, ZZ" in generate_section_body(section)

    from docx import Document

    path = build_docx(resume, tmp_path, "supplied")
    text = "\n".join(paragraph.text for paragraph in Document(path).paragraphs)
    assert section.title in text
    assert "Supplied duty." in text


def test_empty_payload_replaces_all_substantive_latex_template_samples():
    rendered = render_template(DEFAULT_TEMPLATE.read_text(encoding="utf-8"), parse_resume_payload({}))
    assert r"\name{}{}" in rendered
    # Empty contact commands cause a real ModernCV makecvtitle failure. Omit
    # them entirely rather than reintroducing sample contact values.
    for field in [r"\address", r"\phone", r"\email", r"\social"]:
        assert field not in rendered
    for sample in [
        "Jeff", "Calderon", "Marysville", "916", "jeffdcalderon", "aggressor-FZX",
        "Engineering Data Scientist", "Engineering data scientist", "10+ years",
        "NOAA", "Naval Research", "Boeing", "Python", "Monte Carlo", "Journal of Instrumentation",
    ]:
        assert sample not in rendered
    assert rendered.startswith(r"\documentclass")
    assert rendered.rstrip().endswith(r"\end{document}")


def test_supplied_publication_is_rendered_instead_of_sample_publication():
    publication = "Supplied publication & coauthors (2024)"
    resume = parse_resume_payload({"skills": [], "publication": publication})
    rendered = render_template(DEFAULT_TEMPLATE.read_text(encoding="utf-8"), resume)
    assert latex_escape(publication) in rendered
    assert "Journal of Instrumentation" not in rendered
    assert "Liquid Scintillator" not in rendered