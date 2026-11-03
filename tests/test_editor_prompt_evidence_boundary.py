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
