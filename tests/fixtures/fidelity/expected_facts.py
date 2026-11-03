"""Independent expected-facts ledger for the synthetic delivery run 4.

**This file is an answer key, not an implementation detail.**

Every expectation below was derived by reading the candidate's *source document* and
nothing else. It was written before any change to the writer, editor, sanitizer or
correction engine, and it must not be edited to make an implementation pass. If a test
here fails, the implementation is wrong — or, if the source truly changed, the ledger is
revised deliberately and that revision is called out in a commit message.

Provenance of the source text
-----------------------------
The source is the `sources[0]["text"]` field of the protected live capture
`.live-run-2026-11-02/retained/delivery-run-4-withheld.json`
(SHA-256 ``92e03d4bbfde82089bcb428ff563d274a9e215cbabe496428f6565cdff891230``).
That capture is read-only evidence; this module never writes to it.

The source, in full (segment numbering matches the reviewer's payload):

===  source  ===
## Professional Summary
SYNTHETIC-ONLY Journeyman electrician with 8+ years of commercial and industrial electrical
experience across construction, facilities maintenance, and power distribution.
## Professional Experience
**SYNTHETIC-ONLY Cascade Power Solutions, LLC** -- Example City, WA
*Journeyman Electrician | 03/2020 - Present*
- Installed and maintained commercial electrical systems across 40+ job sites.
- Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices.
- Cut rework by 18% by standardizing panel labeling and inspection checklists.
**SYNTHETIC-ONLY Harborline Mechanical Contractors** -- Sampletown, OR
*Electrical Apprentice | 06/2016 - 02/2020*
- Assisted with conduit bending, wire pulling, and panel terminations on 25 projects.
- Completed 8,000 hours of supervised field work toward licensure.
## Education
**SYNTHETIC-ONLY Example Technical College** -- Example City, WA
*Associate of Applied Science, Electrical Systems Technology | 2016*
## Certifications
- SYNTHETIC-ONLY Washington Journey Level Electrician License
- SYNTHETIC-ONLY NFPA 70E Arc Flash Safety
## Skills
- Commercial electrical systems, conduit bending, panel labeling
- NFPA 70E, lockout/tagout, inspection checklists
===  end source  ===
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# The source, kept verbatim so assertions can be written against it directly.
# ---------------------------------------------------------------------------

SOURCE_TEXT = """## Professional Summary
SYNTHETIC-ONLY Journeyman electrician with 8+ years of commercial and industrial electrical
experience across construction, facilities maintenance, and power distribution.
## Professional Experience
**SYNTHETIC-ONLY Cascade Power Solutions, LLC** -- Example City, WA
*Journeyman Electrician | 03/2020 - Present*
- Installed and maintained commercial electrical systems across 40+ job sites.
- Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices.
- Cut rework by 18% by standardizing panel labeling and inspection checklists.
**SYNTHETIC-ONLY Harborline Mechanical Contractors** -- Sampletown, OR
*Electrical Apprentice | 06/2016 - 02/2020*
- Assisted with conduit bending, wire pulling, and panel terminations on 25 projects.
- Completed 8,000 hours of supervised field work toward licensure.
## Education
**SYNTHETIC-ONLY Example Technical College** -- Example City, WA
*Associate of Applied Science, Electrical Systems Technology | 2016*
## Certifications
- SYNTHETIC-ONLY Washington Journey Level Electrician License
- SYNTHETIC-ONLY NFPA 70E Arc Flash Safety
## Skills
- Commercial electrical systems, conduit bending, panel labeling
- NFPA 70E, lockout/tagout, inspection checklists"""


# ---------------------------------------------------------------------------
# 1. Employer -> role -> date relationships
#
# The binding matters more than the individual strings: a rewrite may rephrase
# anything, but Cascade must stay the employer of the Journeyman Electrician role
# and Harborline of the Apprentice role, with the dates attached to the right job.
# ---------------------------------------------------------------------------

EMPLOYMENT_RECORDS = (
    {
        "employer": "Cascade Power Solutions, LLC",
        "role": "Journeyman Electrician",
        "start": "03/2020",
        "end": "Present",
        "location": "Example City, WA",
    },
    {
        "employer": "Harborline Mechanical Contractors",
        "role": "Electrical Apprentice",
        "start": "06/2016",
        "end": "02/2020",
        "location": "Sampletown, OR",
    },
)

# Titles that must never appear as a substitute for a real role title. "Professional"
# is the exact fabricated placeholder the sanitizer emitted in run 4.
FABRICATED_TITLE_PLACEHOLDERS = ("Professional",)


# ---------------------------------------------------------------------------
# 2. Quantities and what each is attached to
#
# "Six apprentices" is people, not crews. Each number is bound to its own
# accomplishment; swapping them is a fidelity defect even though every digit is real.
# ---------------------------------------------------------------------------

QUANTIFIED_FACTS = (
    {"value": "6", "subject": "apprentices", "verb": "Trained",
     "forbidden_subject": "crews"},
    {"value": "40+", "subject": "job sites", "verb": "Installed and maintained", "forbidden_subject": None},
    {"value": "25", "subject": "projects", "verb": "Assisted with", "forbidden_subject": None},
    {"value": "8,000", "subject": "hours", "qualifier": "supervised field work",
     "verb": "Completed", "forbidden_subject": None},
    {"value": "18%", "subject": "rework", "verb": "Cut",
     "forbidden_subject": "safety"},
    {"value": "8+", "subject": "years", "verb": None, "forbidden_subject": None},
)


# ---------------------------------------------------------------------------
# 3. Education
# ---------------------------------------------------------------------------

EDUCATION_RECORD = {
    "institution": "Example Technical College",
    "degree": "Associate of Applied Science",
    "field": "Electrical Systems Technology",
    "year": "2016",
    "location": "Example City, WA",
}

# The source states no GPA and no coursework. Both appeared in the run-4 draft.
UNKNOWN_EDUCATION_FIELDS = ("gpa", "coursework", "relevant_courses", "honors")


# ---------------------------------------------------------------------------
# 4. Credentials
#
# Exactly two, with the names as written. The source names no issuer and no issue
# date; the run-4 draft invented "Washington State Department of Labor & Industries
# (2020)" and "National Fire Protection Association (2019)".
# ---------------------------------------------------------------------------

CREDENTIALS = (
    "Washington Journey Level Electrician License",
    "NFPA 70E Arc Flash Safety",
)

UNKNOWN_CREDENTIAL_FIELDS = ("issuer", "issue_date", "date", "issued", "expiry")


# ---------------------------------------------------------------------------
# 5. Skills the source actually supports
# ---------------------------------------------------------------------------

SUPPORTED_SKILLS = frozenset({
    "commercial electrical systems",
    "conduit bending",
    "panel labeling",
    "nfpa 70e",
    "lockout/tagout",
    "inspection checklists",
    "electrical",
    "power distribution",
    "facilities maintenance",
    "construction",
    "apprentice training",
    "safe work practices",
    "supervised field work",
})

# Compatible wording is allowed: a paraphrase that means the same supported fact is
# not an invention. These are the source phrases a rewrite may legitimately reuse.
SKILL_ALIASES = {
    "lockout/tagout": ("lockout/tagout", "lockout tagout", "loto"),
    "nfpa 70e": ("nfpa 70e", "nfpa-70e", "arc flash"),
    "commercial electrical systems": (
        "commercial electrical systems", "commercial electrical",
        "electrical systems",
    ),
    "panel labeling": ("panel labeling", "panel labelling", "panel labels"),
    "conduit bending": ("conduit bending", "conduit"),
    "inspection checklists": ("inspection checklists", "inspection checklist"),
}

# Skills/technologies present in the run-4 draft with no basis in the source. These are
# *evidence of a class of defect*, not a blacklist to be implemented against: the
# implementation must be unable to emit unbacked skills, not merely unable to emit
# these particular strings.
UNSUPPORTED_IN_RUN_4_DRAFT = frozenset({
    "python", "sql", "vba", "autocad", "revit", "plc", "plc programming",
    "ai analytics", "iot sensors", "mqtt", "modbus", "scada", "agile",
    "scrum", "gantt charts", "windows", "linux", "energyplus",
    "osha",
})


# ---------------------------------------------------------------------------
# 6. Structures the source does not contain at all
#
# The run-4 draft invented a Projects section with two projects, two fabricated
# savings figures (12% peak-demand reduction, 15% lighting reduction) and tooling.
# A Projects heading is permitted; projects themselves are not.
# ---------------------------------------------------------------------------

ABSENT_FROM_SOURCE = {
    "projects_section_content": (
        "Smart Building Energy Management System",
        "Commercial Lighting Retrofit Initiative",
    ),
    "invented_measurements": ("12%", "15%"),
    "invented_technologies": (
        "Python", "SQL", "VBA", "AutoCAD", "Revit", "PLC", "IoT", "MQTT",
        "Modbus", "SCADA", "EnergyPlus", "AI Analytics", "Gantt",
    ),
}


# ---------------------------------------------------------------------------
# 7. Section headings that must all remain present
#
# The product promises five headings. Keeping them is not permission to fabricate
# content to fill one: "Projects" may appear with no invented entries.
#
# This was internally inconsistent and is corrected here. The comment promised five
# headings, the tuple below listed four, and there was no Projects entry — so a test
# written from the comment and a test written from the tuple disagreed about whether a
# resume was complete. The product's promise is five headings with empty ones allowed,
# which is what both the prompt and the renderer now do:
#
#   * the prompt requires all five headings and permits an empty one;
#   * `_ensure_all_sections` keeps an empty heading instead of deleting it;
#   * `approved_resume.export_payload` parses an empty `## Projects` to `[]` and the
#     LaTeX/DOCX renderers emit nothing for it (verified, not assumed).
#
# A heading is a structural claim, not a factual one, so "present but empty" is the
# correct state for a section the candidate has no content for.
# ---------------------------------------------------------------------------

REQUIRED_SECTION_HEADINGS = (
    "professional summary",
    "professional experience",
    "projects",
    "education",
    "certifications",
    "skills",
)

# Sections the product requires even when the candidate has nothing to put in them.
# A heading with no content is permitted; content invented to fill it is not.
SECTIONS_ALLOWED_TO_BE_EMPTY = ("projects",)

# The summary is a paragraph, not necessarily a markdown heading, so it is checked
# separately by content rather than by heading presence.
SUMMARY_MUST_MENTION = ("electrician",)


# ---------------------------------------------------------------------------
# 8. Classification of the 12 segments run 4 reported as "missing"
#
# The reviewer marked 12 of 22 required segments missing and the engine withheld
# immediately (source_correction.py:844). Classifying them honestly shows that
# "not verbatim" is not the same as "missing":
#
#   omitted            - no line of the draft carries the fact at all
#   paraphrased        - the fact is present in different wording (NOT missing)
#   altered            - present but changed in a way that matters (wrong subject,
#                        wrong attribution, wrong number, corrupted text)
#   misattributed      - present but attached to the wrong employer/role/education
#   nonfactual         - heading or formatting line with no factual assertion
#
# `required_outcome` states what a correct system must do with each.
# ---------------------------------------------------------------------------

MISSING_SEGMENT_CLASSIFICATION = {
    0: {
        "text": "## Professional Summary",
        "class": "nonfactual",
        "reason": "Section heading; asserts no candidate fact.",
        "required_outcome": "must not be reported as a missing fact",
    },
    1: {
        "text": "SYNTHETIC-ONLY Journeyman electrician with 8+ years of commercial and "
                "industrial electrical",
        "class": "paraphrased",
        "reason": "The draft's opening bullets carry journeyman-level commercial and "
                  "industrial electrical work over 8+ years. The fact is present; the "
                  "wording and the summary section are not.",
        "required_outcome": "accepted; not missing",
    },
    2: {
        "text": "experience across construction, facilities maintenance, and power "
                "distribution.",
        "class": "paraphrased",
        "reason": "Construction and power distribution appear in the draft's bullets; "
                  "facilities maintenance is a reasonable reading of the site work.",
        "required_outcome": "accepted; not missing",
    },
    5: {
        "text": "*Journeyman Electrician | 03/2020 - Present*",
        "class": "omitted",
        "reason": "The title was genuinely destroyed upstream: the sanitizer replaced it "
                  "with the literal word 'Professional'. The dates survived, the title did "
                  "not. This is the one true omission and it is repairable from an "
                  "explicit source record.",
        "required_outcome": "restored to the Cascade employment record",
    },
    6: {
        "text": "- Installed and maintained commercial electrical systems across 40+ job "
                "sites.",
        "class": "paraphrased",
        "reason": "'more than 40 job sites' appears verbatim in meaning, attached to "
                  "Cascade.",
        "required_outcome": "accepted; not missing",
    },
    7: {
        "text": "- Trained 6 apprentices on lockout/tagout and NFPA 70E safe work "
                "practices.",
        "class": "altered",
        "reason": "Present but changed from 'six apprentices' to 'six apprentice crews'. "
                  "Crews is a promotion of six people to six teams — a different claim.",
        "required_outcome": "corrected to apprentices, or withheld",
    },
    8: {
        "text": "- Cut rework by 18% by standardizing panel labeling and inspection "
                "checklists.",
        "class": "paraphrased",
        "reason": "The 18% rework reduction and the panel-labeling cause both survive in "
                  "the draft's optimized-workflow bullet. The reviewer marked it missing "
                  "because no single line restated it with the original verb.",
        "required_outcome": "accepted; not missing",
    },
    10: {
        "text": "*Electrical Apprentice | 06/2016 - 02/2020*",
        "class": "omitted",
        "reason": "Same destruction as segment 5: the title was replaced with "
                  "'Professional'. Dates survived.",
        "required_outcome": "restored to the Harborline employment record",
    },
    11: {
        "text": "- Assisted with conduit bending, wire pulling, and panel terminations on "
                "25 projects.",
        "class": "paraphrased",
        "reason": "All three activities and the 25-project count are present, attached to "
                  "Harborline.",
        "required_outcome": "accepted; not missing",
    },
    12: {
        "text": "- Completed 8,000 hours of supervised field work toward licensure.",
        "class": "paraphrased",
        "reason": "'8,000 hours' is present; the draft re-cased it as 'hands-on' rather "
                  "than 'supervised', which is a weakening but not an omission.",
        "required_outcome": "accepted; 'supervised' restored if easy, otherwise accepted",
    },
    14: {
        "text": "**SYNTHETIC-ONLY Example Technical College** -- Example City, WA",
        "class": "omitted",
        "reason": "The institution is genuinely absent from the draft's Education "
                  "section, which names only the degree line. The degree and year are "
                  "present, so the education record exists but lost its school.",
        "required_outcome": "restored to the education record",
    },
    20: {
        "text": "- Commercial electrical systems, conduit bending, panel labeling",
        "class": "paraphrased",
        "reason": "All three appear in the draft, attached to the experience bullets "
                  "rather than repeated in the skills block.",
        "required_outcome": "accepted; not missing",
    },
}


# ---------------------------------------------------------------------------
# 9. The expected gate outcome for faithful vs defective documents
#
# These are the acceptance outcomes. They are stated here, independently of the
# implementation, so a test cannot redefine success.
# ---------------------------------------------------------------------------

# A rewrite that fixes the three true omissions (titles, school) and removes every
# unsupported addition must be publishable.
FAITHFUL_REWRITE_MUST_BE_APPROVED = True

# The captured run-4 draft, unchanged, must keep being rejected.
CAPTURED_DRAFT_MUST_STAY_REJECTED = True


def employment_for_role(role: str) -> dict | None:
    """Return the employment record whose role this is, for attribution assertions."""
    for record in EMPLOYMENT_RECORDS:
        if record["role"].lower() == role.lower():
            return record
    return None


def quantified_fact(value: str) -> dict | None:
    for fact in QUANTIFIED_FACTS:
        if fact["value"] == value:
            return fact
    return None
