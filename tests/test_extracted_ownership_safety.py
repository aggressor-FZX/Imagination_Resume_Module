"""Offline safety controls for extracted experience records."""
import pytest

import source_preservation as sp


SOURCE = """Professional Experience
Alpha Electric -- Tacoma, WA
Electrician | 2015 - 2019
Completed 8,000 hours of supervised field work on electrical installations.
Inspected electrical systems and maintained safety records.
Mystery Employer
Technician | 2020 - Present
Compared quarterly inspections against the prior four quarters to set maintenance priorities.
"""


def test_unrecognized_head_stops_ownership_for_plain_and_marked_bullets():
    marked = SOURCE.replace("Compared quarterly", "- Compared quarterly")
    for source in (SOURCE, marked):
        assert sp._employer_of(source, source.index("Compared quarterly")) == ""
        records = sp._source_experience_records(source)
        assert all("prior four quarters" not in b
                   for record in records for b in record["bullets"])
        draft = ("## Professional Experience\n**Alpha Electric** -- Tacoma, WA\n"
                 "*Electrician | 2015 - 2019*\n"
                 "- Compared quarterly inspections against historical records to set maintenance priorities.\n")
        out, changes = sp.preserve_experience_bullets(draft, source)
        assert (out, changes) == (draft, [])


def test_unrecognized_draft_head_does_not_inherit_previous_employer():
    draft = ("## Professional Experience\nAlpha Electric -- Tacoma, WA\n"
             "Electrician | 2015 - 2019\n- Maintained equipment.\n"
             "Mystery Employer\nTechnician | 2020 - Present\n"
             "- Completed 8,000 hours of hands-on field work on electrical installations.\n")
    assert sp._employer_of(draft, draft.index("- Completed")) == ""
    out, changes = sp.preserve_quantified_qualifiers(draft, SOURCE)
    assert (out, changes) == (draft, [])


def test_unrecognized_head_without_role_is_still_a_boundary():
    source = ("Professional Experience\nAlpha Electric -- Tacoma, WA\n"
              "Electrician | 2015 - 2019\nMaintained equipment.\n"
              "Mystery Employer\n"
              "Completed 8,000 hours of supervised field work.\n")
    assert sp._employer_of(source, source.index("Completed")) == ""
    assert sp._source_experience_records(source)[0]["bullets"] == ["Maintained equipment."]


def test_duplicate_quantified_source_lines_do_not_get_first_occurrence_owner():
    source = ("Professional Experience\nAlpha Electric -- Tacoma, WA\n"
              "Electrician | 2015 - 2019\n"
              "Completed 8,000 hours of supervised field work.\n"
              "Beta Power -- Kent, WA\nTechnician | 2020 - Present\n"
              "Completed 8,000 hours of supervised field work.\n")
    draft = ("## Professional Experience\n**Beta Power** -- Kent, WA\n"
             "*Technician | 2020 - Present*\n"
             "- Completed 8,000 hours of hands-on field work.\n")
    out, changes = sp.preserve_quantified_qualifiers(draft, source)
    assert "8,000 hours of supervised field work" in out
    assert changes == ["restored the source wording for a quantified bullet"]


def test_no_matching_by_order_and_shared_exact_bullets_are_ambiguous():
    source = ("Professional Experience\nAlpha Electric -- Tacoma, WA\n"
              "Electrician | 2015 - 2019\nInspected equipment and maintained safety records.\n"
              "Beta Power -- Kent, WA\nTechnician | 2020 - Present\n"
              "Inspected equipment and maintained safety records.\n")
    draft = ("## Professional Experience\n**Current Employer** --\n*Technician |*\n"
             "- Inspected equipment and maintained safety records.\n")
    assert sp.preserve_employer_records(draft, source) == (draft, [])


def test_unstructured_plain_paragraph_is_not_assumed_to_be_a_bullet():
    source = "Inspected electrical systems and maintained safety records.\n"
    draft = "- Inspected electrical systems and maintained safety records, ensuring flawless compliance.\n"
    assert sp.preserve_experience_bullets(draft, source) == (draft, [])


def test_source_separator_and_incomplete_record_do_not_inherit_an_owner():
    source = ("Professional Experience\nAlpha Electric -- Tacoma, WA\n"
              "Electrician | 2015 - 2019\nMaintained equipment.\n---\n"
              "Compared quarterly inspections against the prior four quarters to set maintenance priorities.\n"
              "Beta Power -- Kent, WA\nInspected equipment and maintained safety records.\n")
    records = sp._source_experience_records(source)
    assert len(records) == 1
    assert records[0]["bullets"] == ["Maintained equipment."]
    assert sp._employer_of(source, source.index("Compared quarterly")) == ""


def test_unowned_marked_bullet_cannot_be_assigned_to_known_employer():
    source = ("Professional Experience\nAlpha Electric -- Tacoma, WA\n"
              "Electrician | 2015 - 2019\nMaintained equipment.\n"
              "Mystery Employer\nTechnician | 2020 - Present\n"
              "- Compared quarterly inspections against the prior four quarters to set maintenance priorities.\n")
    draft = ("## Professional Experience\n**Alpha Electric** -- Tacoma, WA\n"
             "*Electrician | 2015 - 2019*\n"
             "- Compared quarterly inspections against historical records to set maintenance priorities.\n")
    assert sp.preserve_experience_bullets(draft, source) == (draft, [])


def test_faithful_plain_source_paraphrase_is_preserved():
    draft = ("## Professional Experience\n**Alpha Electric** -- Tacoma, WA\n"
             "*Electrician | 2015 - 2019*\n"
             "- Electrical systems were inspected and safety records maintained.\n")
    assert sp.preserve_experience_bullets(draft, SOURCE) == (draft, [])


def test_bridge_does_not_delete_supported_records_paraphrase():
    draft = ("## Professional Experience\n**Alpha Electric** -- Tacoma, WA\n"
             "*Electrician | 2015 - 2019*\n"
             "- Inspected electrical systems, applying trend analysis and maintaining safety logs.\n")
    assert sp.preserve_experience_bullets(draft, SOURCE, ["trend analysis"]) == (draft, [])


def test_bridge_keeps_supported_paraphrase_but_removes_distinct_outcome():
    draft = ("## Professional Experience\n**Alpha Electric** -- Tacoma, WA\n"
             "*Electrician | 2015 - 2019*\n"
             "- Inspected electrical systems, applying trend analysis and maintaining safety logs, "
             "ensuring flawless regulatory compliance.\n")
    out, changes = sp.preserve_experience_bullets(draft, SOURCE, ["trend analysis"])
    assert "applying trend analysis and maintaining safety logs" in out
    assert "ensuring flawless regulatory compliance" not in out
    assert changes == ["removed an unsupported appended clause"]


def test_bridge_leaves_source_related_ambiguous_outcome_for_review():
    draft = ("## Professional Experience\n**Alpha Electric** -- Tacoma, WA\n"
             "*Electrician | 2015 - 2019*\n"
             "- Inspected electrical systems, applying trend analysis and ensuring safety compliance.\n")
    assert sp.preserve_experience_bullets(draft, SOURCE, ["trend analysis"]) == (draft, [])


def test_outcome_shaped_paraphrase_without_shared_words_is_left_for_review():
    draft = ("## Professional Experience\n**Alpha Electric** -- Tacoma, WA\n"
             "*Electrician | 2015 - 2019*\n"
             "- Inspected electrical systems, applying trend analysis and ensuring proper documentation.\n")
    assert sp.preserve_experience_bullets(draft, SOURCE, ["trend analysis"]) == (draft, [])


@pytest.mark.parametrize("suffix", [
    "applying trend analysis and maintaining safety logs.",
    "maintaining safety logs and applying trend analysis.",
    "maintaining safety logs, applying trend analysis.",
    "applying trend analysis while maintaining safety logs.",
    "applying trend analysis and performing safety checks.",
])
def test_bridge_retains_activity_paraphrases_in_any_position(suffix):
    draft = ("## Professional Experience\n**Alpha Electric** -- Tacoma, WA\n"
             "*Electrician | 2015 - 2019*\n"
             "- Inspected electrical systems, " + suffix + "\n")
    assert sp.preserve_experience_bullets(draft, SOURCE, ["trend analysis"]) == (draft, [])


@pytest.mark.parametrize("suffix", [
    "maintaining safety logs, ensuring flawless regulatory compliance, applying trend analysis.",
    "ensuring flawless regulatory compliance and applying trend analysis and maintaining safety logs.",
])
def test_distinct_outcome_removed_between_preserved_paraphrase_and_bridge(suffix):
    draft = ("## Professional Experience\n**Alpha Electric** -- Tacoma, WA\n"
             "*Electrician | 2015 - 2019*\n"
             "- Inspected electrical systems, " + suffix + "\n")
    out, changes = sp.preserve_experience_bullets(draft, SOURCE, ["trend analysis"])
    assert "maintaining safety logs" in out
    assert "applying trend analysis" in out
    assert "ensuring flawless regulatory compliance" not in out
    assert changes == ["removed an unsupported appended clause"]


def test_inseparable_bridge_and_outcome_remain_for_semantic_review():
    draft = ("## Professional Experience\n**Alpha Electric** -- Tacoma, WA\n"
             "*Electrician | 2015 - 2019*\n"
             "- Inspected electrical systems, applying trend analysis to guarantee flawless compliance.\n")
    assert sp.preserve_experience_bullets(draft, SOURCE, ["trend analysis"]) == (draft, [])


def test_bridge_with_transplanted_figure_is_not_silently_trimmed():
    draft = ("## Professional Experience\n**Alpha Electric** -- Tacoma, WA\n"
             "*Electrician | 2015 - 2019*\n"
             "- Inspected electrical systems, applying trend analysis and ensuring 100% compliance.\n")
    assert sp.preserve_experience_bullets(draft, SOURCE, ["trend analysis"]) == (draft, [])


def test_full_preservation_keeps_paraphrase_and_removes_distinct_outcome():
    draft = ("## Professional Experience\n**Alpha Electric** -- Tacoma, WA\n"
             "*Electrician | 2015 - 2019*\n"
             "- Inspected electrical systems, applying trend analysis and maintaining safety logs, "
             "ensuring flawless regulatory compliance.\n")
    out, changes = sp.preserve_source_facts(draft, SOURCE, ["trend analysis"])
    assert "applying trend analysis and maintaining safety logs" in out
    assert "ensuring flawless regulatory compliance" not in out
    assert "removed an unsupported appended clause" in changes


def test_paraphrase_and_permitted_bridge_survive_whole_line_replacements():
    draft = ("## Professional Experience\n**Alpha Electric** -- Tacoma, WA\n"
             "*Electrician | 2015 - 2019*\n"
             "- Inspected electrical systems and maintained safety records, applying trend analysis.\n"
             "- Completed 8,000 hours of hands-on field work on electrical installations, applying trend analysis.\n")
    out, _ = sp.preserve_source_facts(draft, SOURCE, permitted_bridges=["trend analysis"])
    assert out.count("trend analysis") == 2
    assert "hands-on field work" in out


def test_repeated_employer_records_are_not_merged_by_name():
    source = ("Professional Experience\nAlpha Electric -- Tacoma, WA\n"
              "Electrician | 2015 - 2019\nCompleted 8,000 hours of supervised field work.\n"
              "Alpha Electric -- Tacoma, WA\nManager | 2020 - Present\n"
              "Compared quarterly inspections against the prior four quarters to set maintenance priorities.\n")
    draft = ("## Professional Experience\n**Alpha Electric** -- Tacoma, WA\n"
             "*Manager | 2020 - Present*\n"
             "- Compared quarterly inspections against historical records to set maintenance priorities.\n"
             "- Completed 8,000 hours of hands-on field work.\n")
    assert sp.preserve_experience_bullets(draft, source) == (draft, [])
    assert sp.preserve_quantified_qualifiers(draft, source) == (draft, [])