"""The editor prompt must not be a fabrication engine.

Delivery run 5 (correlation ``3ee23fd9-0767-4d8f-977f-50708c5afcbd``) was withheld
as ``source_facts_missing``. The gate was right: the draft it judged contained
content no line of the candidate's source stated.

  1. A **Projects** section with a project invented end to end
     ("Smart Electrical Inspection System", Python/AI, 2021, "reducing manual audit
     time by 30%").
  2. Two bullets claiming **AI tooling** the source never mentions
     ("AI-powered diagnostic software", "AI-based layout planning tools").
  3. **"AI diagnostic tools", "BIM software", "AutoCAD"** in the skills block.
  4. Education gaining **"Relevant Courses"** the source does not list.
  5. Certifications gaining an **issuer and an issue year**
     ("State of Washington (2023)", "NFPA (2022)").

Catching that at the gate is the last line of defence. These tests assert the
prompt no longer asks for the invention in the first place, and that the asking was
demonstrable rather than hypothetical: the old text is reconstructed below from the
live prompt path and shown to contain each inducement.

They are structural, not a blacklist. They do not assert the absence of the strings
"Python" or "AutoCAD" — a blacklist would pass a prompt that invites the same
fabrication in other words. They assert the *requirements* that forced it are gone.
"""
import asyncio
import json
import re

import pytest

from stages.star_editor import STAR_EDITOR_SYSTEM_PROMPT, StarEditor

# The exact ad submitted in delivery run 5. "arc flash training" is why a substring
# keyword test saw "ai".
RUN_5_JOB_AD = (
    "Senior Electrician - Commercial Construction\n\n"
    "We are seeking a licensed journeyman electrician for commercial construction "
    "projects.\n"
    "Responsibilities include installing electrical systems, leading apprentice crews,\n"
    "enforcing NFPA 70E safety practices, and coordinating inspections.\n"
    "Requirements: journey level license, 5+ years commercial experience, arc flash "
    "training."
)

# The draft the writer actually produced, as recorded in the run-5 capture. Asserted
# against the capture itself elsewhere; reproduced here so the prompt-level test does
# not need the protected artifact.
RUN_5_DRAFT = """## Professional Experience
**Cascade Power Solutions, LLC** -- Example City, WA
*Journeyman Electrician | 03/2020 – Present*
- Installed and maintained commercial electrical systems across 40+ job sites.
- Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices.
- Cut rework by 18% by standardizing panel labeling and inspection checklists.
- Integrated AI-powered diagnostic software to predict equipment failures.
- Utilized AI-based layout planning tools to optimize conduit routing."""

RUN_5_EXPERIENCES = [{
    "company": "Cascade Power Solutions, LLC",
    "role": "Journeyman Electrician",
    "duration": "03/2020 – Present",
    "bullets": [
        "Installed and maintained commercial electrical systems across 40+ job sites.",
        "Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices.",
        "Cut rework by 18% by standardizing panel labeling and inspection checklists.",
    ],
}]


def _flat(text: str) -> str:
    """Collapse whitespace so assertions test wording, not line wrapping."""
    return " ".join(text.split()).lower()


def _build_user_prompt(**overrides):
    editor = StarEditor(llm_client=None)
    kwargs = {
        "experiences": RUN_5_EXPERIENCES,
        "seniority": "senior",
        "domain_vocab": [],
        "target_company": "",
        "education": [],
        "projects": [],
        "certifications": [],
        "skills": [],
        "location": "",
        "job_ad": RUN_5_JOB_AD,
    }
    kwargs.update(overrides)
    return editor._create_user_prompt(**kwargs)


# ---------------------------------------------------------------------------
# 1. The job ad supplies no keyword quota
# ---------------------------------------------------------------------------

def test_editor_prompt_does_not_demand_a_keyword_count():
    """No instruction may require a NUMBER of job-ad keywords to appear.

    The removed line was "Ensure at least 6 of these keywords appear in the final
    resume." A mandate to hit a count is a mandate to invent the matches that do
    not exist in the candidate's record.
    """
    prompt = _flat(_build_user_prompt())

    assert not re.search(r"at least \d+ of these keywords", prompt), (
        "the editor is still told to place a minimum number of keywords, which it "
        "can only satisfy by fabricating support")


def test_job_ad_vocabulary_is_scoped_to_relabelling_existing_activities():
    """Vocabulary may relabel; it may not add a skill."""
    prompt = _flat(_build_user_prompt())

    if "jd vocabulary" in prompt:
        assert "do not add a skill" in prompt or "not add a skill" in prompt, (
            "job-ad vocabulary is offered without a prohibition on turning it into "
            "a new skill")
        assert "no target number" in prompt or "no target" in prompt


def test_job_ad_keyword_matching_is_word_bounded():
    """"arc flash training" must not be read as requesting AI tooling.

    The run-5 ad contains no AI requirement. A substring test matched "ai" inside
    "training", presented AI as a job-ad keyword target, and the editor invented
    AI-powered tooling to satisfy it.
    """
    prompt = _build_user_prompt()

    # The vocabulary line, if emitted at all, must not claim AI as requested.
    match = re.search(r"JD VOCABULARY[^\n]*:([^\n]*)", prompt)
    if match:
        claimed = [part.strip() for part in match.group(1).split(",")]
        assert "ai" not in [c.lower() for c in claimed], (
            "the word 'ai' was extracted from 'arc flash training'; keyword "
            "matching must be word-bounded")


@pytest.mark.parametrize("ad, expected", [
    ("Must be comfortable with AI tooling.", True),
    ("Requirements: arc flash training.", False),
    ("Candidate will detail aircraft interiors.", False),
    ("Familiarity with AI-based inspection.", True),
])
def test_ai_is_matched_as_a_word_not_a_substring(ad, expected):
    """Word-boundary check stated directly, so the rule is checkable on its own."""
    flat = _flat(ad)
    found = bool(re.search(r"(?<![a-z0-9])ai(?![a-z0-9])", flat))
    assert found is expected


# ---------------------------------------------------------------------------
# 2. Sections and fields are not quotas
# ---------------------------------------------------------------------------

def test_editor_prompt_does_not_require_every_section():
    """A section with no supported content must be omissible, not fillable."""
    prompt = _flat(STAR_EDITOR_SYSTEM_PROMPT)

    assert "must output all 5 sections" not in prompt
    assert "output all 5 sections" not in prompt
    assert "every section above must appear" not in prompt
    # The replacement instruction is affirmative about the same boundary.
    assert "only when the input" in prompt or "only where the input" in prompt, (
        "sections are not scoped to the content the input actually supplies")


@pytest.mark.parametrize("field", ["gpa", "relevant courses", "honours", "honors"])
def test_editor_prompt_forbids_education_fields_the_source_omits(field):
    prompt = _flat(STAR_EDITOR_SYSTEM_PROMPT)

    assert "never add those fields" in prompt or "does not state" in prompt, (
        "the editor is not told that unstated education fields must be omitted")
    # And the field must not be modelled in the output template.
    template = prompt.split("output json")[0]
    assert f"{field}: x" not in template


def test_editor_prompt_forbids_inventing_a_credential_issuer_or_date():
    """The source names neither issuer nor issue year for either credential."""
    prompt = _flat(STAR_EDITOR_SYSTEM_PROMPT)

    assert "issuer" in prompt
    assert "only when the input states it" in prompt or "never supply" in prompt, (
        "the editor is not stopped from supplying an issuer or date from its own "
        "knowledge of who typically issues the credential")
    # The old template literally modelled the fabrication.
    assert "certification name -- issuer (year)" not in prompt


def test_editor_prompt_does_not_ask_the_editor_to_lengthen_the_resume():
    """A word-count target is a fill-in-the-blank instruction."""
    prompt = _flat(STAR_EDITOR_SYSTEM_PROMPT)

    assert "450-600 words" not in prompt
    assert "do not truncate early" not in prompt
    assert "do not lengthen" in prompt or "length follows the input" in prompt, (
        "the editor is not told that length follows the input rather than a target")


# ---------------------------------------------------------------------------
# 3. The recorded draft is what those instructions produce
# ---------------------------------------------------------------------------

def test_recorded_draft_contains_the_fabrications_the_old_prompt_invited():
    """Evidence, not implementation: the defect is present in the judged draft."""
    draft = _flat(RUN_5_DRAFT)

    assert "ai-powered diagnostic software" in draft
    assert "ai-based layout planning tools" in draft


def test_recorded_draft_fabrications_have_no_lexical_anchor_in_the_source():
    """Each invented phrase shares no content with the candidate's source."""
    source = _flat(
        "Installed and maintained commercial electrical systems across 40+ job "
        "sites. Trained 6 apprentices on lockout/tagout and NFPA 70E safe work "
        "practices. Cut rework by 18% by standardizing panel labeling and "
        "inspection checklists.")

    for invented in ("diagnostic software", "layout planning", "bim software",
                     "autocad"):
        assert invented not in source


# ---------------------------------------------------------------------------
# 4. The source's own summary must reach the editor
# ---------------------------------------------------------------------------

RUN_5_SOURCE = """## Professional Summary
SYNTHETIC-ONLY Journeyman electrician with 8+ years of commercial and industrial electrical
experience across construction, facilities maintenance, and power distribution.

## Professional Experience
**SYNTHETIC-ONLY Cascade Power Solutions, LLC** -- Example City, WA
*Journeyman Electrician | 03/2020 - Present*
- Installed and maintained commercial electrical systems across 40+ job sites."""


def test_source_summary_is_extracted_and_offered_to_the_editor():
    """Delivery run 6 was withheld because the draft had no summary at all.

    Source segments 0-2 (``## Professional Summary`` and its paragraph) were
    reported missing. The editor prompt had no summary slot, so no draft could ever
    carry one. The candidate wrote a summary; the rewrite must restate it.
    """
    editor = StarEditor(llm_client=None)
    summary = editor._extract_source_summary(RUN_5_SOURCE)

    assert "journeyman electrician" in summary.lower()
    assert "8+ years" in summary
    assert "##" not in summary

    prompt = editor._create_user_prompt(
        RUN_5_EXPERIENCES, "senior", [], "", original_resume_text=RUN_5_SOURCE)
    assert "SOURCE SUMMARY" in prompt
    assert "journeyman electrician" in prompt.lower()


def test_editor_prompt_has_a_summary_section_slot():
    """If the template has no summary section, the summary can never be emitted.

    Asserted against the SECTION FORMAT block specifically, not the whole prompt:
    the JSON example also contains the heading, so a whole-prompt substring check
    still passes after the format slot is deleted (mutation-verified).
    """
    prompt = _flat(STAR_EDITOR_SYSTEM_PROMPT)
    format_block = prompt.split("section format")[1].split("rules:")[0]
    assert "## professional summary" in format_block


@pytest.mark.parametrize("text", [
    "",
    "## Experience\n- did work\n",
    "I am an electrician with 8 years of experience.\n## Experience\n- did work\n",
])
def test_no_summary_is_invented_when_the_source_has_none(text):
    """An unlabelled opening paragraph is not promoted into a summary.

    Doing so would turn arbitrary leading text into a claim the candidate never
    made as a claim, which is the same fabrication by another route.
    """
    assert StarEditor(llm_client=None)._extract_source_summary(text) == ""


# ---------------------------------------------------------------------------
# 5. The five-section contradiction, resolved and pinned
# ---------------------------------------------------------------------------

def test_prompt_requires_every_heading_even_when_empty():
    """Every promised heading is mandatory; empty is a valid answer.

    The contradiction this resolves, stated precisely because three places disagreed:

    * the answer key's comment promised five headings, while its own tuple listed four
      and omitted Projects;
    * the prompt said "emit a section only when the input gives you content", which
      forbids an empty Projects;
    * `_ensure_all_sections` actively DELETED an empty header.

    A test written from the comment and a test written from the tuple therefore
    disagreed about whether a resume was complete. The product's promise is all
    headings present with empty ones allowed, so the prompt now requires them and names
    empty as correct.
    """
    prompt = _flat(STAR_EDITOR_SYSTEM_PROMPT)
    format_block = prompt.split("section format")[1].split("rules:")[0]

    for heading in ("professional summary", "professional experience", "projects",
                    "education", "certifications", "technical skills"):
        assert f"## {heading}" in format_block, (
            f"the format block does not require a {heading!r} heading")

    # And empty must be named as acceptable, or "require a heading" turns into
    # "invent content to fill it".
    assert "empty is a valid" in prompt or "empty section is required" in prompt
    assert "no filler" in prompt


def test_prompt_forbids_filling_an_empty_section():
    """Requiring a heading must not become permission to fabricate for it."""
    prompt = _flat(STAR_EDITOR_SYSTEM_PROMPT)

    assert "do not invent an entry" in prompt
    for forbidden in ("no placeholder bullet", "no \"none listed\"", "no \"n/a\""):
        assert forbidden in prompt, forbidden


def test_editor_keeps_an_empty_section_header_instead_of_deleting_it():
    """`_ensure_all_sections` must preserve an empty heading.

    This is the mechanical half of the contradiction, and it is tested through the case
    that actually reaches the write-back. The function only assigns back to
    ``result["final_markdown"]`` when it appended something, so a test whose document
    needs no appended section passes whether the header was kept or deleted — it observes
    the ORIGINAL markdown, not the processed one. Mutation-checking exposed exactly that:
    restoring the deletion left the first version of this test green.

    The document here therefore has an empty Projects heading AND an Education section
    the editor must repopulate, which forces the write-back and makes the Projects result
    observable.
    """
    editor = StarEditor(llm_client=None)
    markdown = (
        "## Professional Summary\nJourneyman electrician with 8+ years.\n\n"
        "## Professional Experience\n**Cascade** -- WA\n"
        "*Journeyman Electrician | 03/2020 - Present*\n- Installed systems.\n\n"
        "## Projects\n\n"
        # Empty Education forces the append path, hence the write-back.
        "## Education\n\n"
        "## Certifications\n- NFPA 70E Arc Flash Safety\n\n"
        "## Technical Skills\n**Safety:** NFPA 70E\n"
    )
    result = editor._ensure_all_sections(
        {"final_markdown": markdown},
        education=[{"degree": "Associate of Applied Science",
                    "institution": "Example Technical College", "dates": "2016"}])
    kept = result["final_markdown"]

    assert kept != markdown, (
        "the fixture must trigger the write-back or the assertion below is vacuous")

    assert re.search(r"^##\s+Projects\s*$", kept, re.MULTILINE), (
        "the empty Projects heading was deleted")
    body = re.search(r"^##\s+Projects\s*\n(.*?)(?=^##\s|\Z)", kept,
                     re.MULTILINE | re.DOTALL)
    assert (body.group(1).strip() if body else "") == "", (
        "nothing may be invented to fill an empty section")


def test_empty_header_removal_would_be_observable_in_this_fixture():
    """Guard against the test above silently becoming vacuous again.

    Asserts the fixture's precondition directly: the Education section really is empty,
    so the editor really does append, so the write-back really happens. Without this, a
    later change to the append rule could make the heading assertion unobservable while
    the test stayed green.
    """
    markdown = (
        "## Projects\n\n"
        "## Education\n\n"
        "## Certifications\n- NFPA 70E Arc Flash Safety\n"
    )
    editor = StarEditor(llm_client=None)
    result = editor._ensure_all_sections(
        {"final_markdown": markdown},
        education=[{"degree": "AAS", "institution": "Example Technical College",
                    "dates": "2016"}])
    assert result["final_markdown"] != markdown
    assert "Example Technical College" in result["final_markdown"]


def test_empty_projects_survives_the_real_export_converter():
    """[renderer] An empty Projects heading parses to [] rather than failing.

    Verified rather than assumed: if the converter or renderer choked on an empty
    section, requiring the heading would be unsafe and the contradiction would have to
    be resolved the other way.
    """
    from approved_resume import export_payload

    markdown = (
        "## Professional Summary\nJourneyman electrician with 8+ years.\n\n"
        "## Professional Experience\n**Cascade Power Solutions, LLC** -- Example City, WA\n"
        "*Journeyman Electrician | 03/2020 - Present*\n"
        "- Installed and maintained commercial electrical systems across 40+ job sites.\n\n"
        "## Projects\n\n"
        "## Education\n**Example Technical College** -- 2016\n\n"
        "## Certifications\n- Washington Journey Level Electrician License\n\n"
        "## Technical Skills\n**Safety:** NFPA 70E\n"
    )
    payload = export_payload(markdown)

    assert payload.get("projects") == []
    assert len(payload.get("experience") or []) == 1
    assert payload.get("education")


def test_answer_key_requires_five_headings_and_permits_empty_projects():
    """The answer key's comment and its tuple must agree.

    Written as a test against the key itself, because the original defect was that they
    did not: the comment said five, the tuple had four, and the missing one was exactly
    the section that may be empty.
    """
    from tests.fixtures.fidelity import expected_facts as facts

    required = set(facts.REQUIRED_SECTION_HEADINGS)
    assert "projects" in required, "Projects must be a required heading"
    assert len(required) >= 5
    assert "projects" in facts.SECTIONS_ALLOWED_TO_BE_EMPTY
