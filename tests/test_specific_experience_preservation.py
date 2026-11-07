"""Synthetic regressions for role-scoped comparison and purpose restoration."""
import pytest

import source_preservation as sp


def record(bullets, employer="North Works", role="Analyst | 2020 - 2024"):
    return (f"## Professional Experience\n**{employer}** -- City, WA\n"
            f"*{role}*\n" + "".join(f"- {bullet}\n" for bullet in bullets))


@pytest.mark.parametrize(("source", "draft"), [
    ("Tracked monthly energy usage across 12 locations and compared each month against the same month in the prior year.",
     "Tracked monthly energy usage across 12 locations using period-over-period analytics to highlight variances."),
    ("Built the operations scorecard that supervisors review in the weekly meeting.",
     "Built operations scorecards reviewed weekly by supervisors for decision-making insights."),
    ("Audited 3 years of service cost records to find which equipment failed repeatedly.",
     "Audited 3 years of service cost records to identify repeated equipment failures and support fleet reliability improvements."),
    ("Rebuilt the delivery delay log that the dispatch team relies on.",
     "Rebuilt the delivery delay log to enhance dispatch team visibility into recurring logistical disruptions."),
    ("Maintain parts availability records for 7 service branches.",
     "Maintained parts availability records across 7 service branches to support timely equipment repair and uptime."),
])
def test_observed_shapes_restore_only_matched_source_line(source, draft):
    out, changes = sp.preserve_specific_experience_details(record([draft]), record([source]))
    assert f"- {source}" in out
    assert draft not in out
    assert len(changes) == 1
    assert sp.preserve_specific_experience_details(out, record([source])) == (out, [])


def test_comparison_that_keeps_same_month_prior_year_is_left_intact():
    source = "Tracked monthly energy usage across 12 locations against the same month in the prior year."
    draft = "Tracked monthly energy usage across 12 locations against the corresponding month in the previous year."
    assert sp.preserve_specific_experience_details(record([draft]), record([source])) == (record([draft]), [])


def test_source_stated_outcome_and_ordinary_purpose_are_left_intact():
    source = "Maintained parts availability records for 7 service branches to support timely repair."
    draft = "Maintained parts availability records across 7 service branches to support timely repairs."
    assert sp.preserve_specific_experience_details(record([draft]), record([source])) == (record([draft]), [])
    source = "Built the operations scorecard that supervisors review in weekly meetings."
    draft = "Built operations scorecards reviewed weekly by supervisors for planning."
    assert sp.preserve_specific_experience_details(record([draft]), record([source])) == (record([draft]), [])


def test_uncertain_lexical_outcome_is_not_a_universal_verdict():
    source = "Rebuilt the delivery delay log that the dispatch team relies on."
    draft = "Rebuilt the delivery delay log to improve dispatch team visibility."
    assert sp.preserve_specific_experience_details(record([draft]), record([source])) == (record([draft]), [])


def test_bridge_and_adjacent_outcome_are_left_for_review_not_erased():
    source = "Flagged slow-moving stock and proposed a reduction plan adopted by 2 branches."
    draft = ("Flagged slow-moving stock and contributed to inventory optimization through a "
             "reduction plan adopted by 2 branches to support rapid growth.")
    assert sp.preserve_specific_experience_details(
        record([draft]), record([source]), ["inventory optimization"]
    ) == (record([draft]), [])


def test_wrong_role_or_employer_never_receives_source_fact():
    source = "Maintain parts availability records for 7 service branches."
    draft = "Maintained parts availability records across 7 service branches to support timely repair."
    for document in (record([draft], role="Manager | 2020 - 2024"),
                     record([draft], employer="South Works")):
        assert sp.preserve_specific_experience_details(document, record([source])) == (document, [])


def test_duplicate_employer_and_duplicate_draft_claim_refuse_repair():
    source = "Maintain parts availability records for 7 service branches."
    draft = "Maintained parts availability records across 7 service branches to support timely repair."
    doubled_source = record([source]) + record([source], role="Manager | 2010 - 2019")
    assert sp.preserve_specific_experience_details(record([draft]), doubled_source) == (record([draft]), [])
    doubled_draft = record([draft, draft])
    assert sp.preserve_specific_experience_details(doubled_draft, record([source])) == (doubled_draft, [])


def test_same_employer_different_role_or_unknown_boundary_is_not_rewritten():
    source = "Maintain parts availability records for 7 service branches."
    draft = "Maintained parts availability records across 7 service branches to support timely repair."
    first = record([draft])
    second = "**North Works** -- City, WA\n*Manager | 2010 - 2019*\n- " + draft + "\n"
    boundary = "Unrecognized Division\n- " + draft + "\n"
    document = first + second
    assert sp.preserve_specific_experience_details(document, record([source])) == (document, [])
    document = first + boundary
    out, changes = sp.preserve_specific_experience_details(document, record([source]))
    assert len(changes) == 1
    assert out.count(draft) == 1
    assert out.index(source) < out.index("Unrecognized Division") < out.index(draft)


def test_tied_source_candidates_and_transplanted_figures_refuse_repair():
    sources = ["Maintain parts availability records for 7 service branches.",
               "Maintain parts availability records for 7 service offices."]
    ambiguous = "Maintained parts availability records for 7 service teams to support timely repair."
    assert sp.preserve_specific_experience_details(record([ambiguous]), record(sources)) == (record([ambiguous]), [])
    transplanted = "Maintained parts availability records for 8 service branches to support timely repair."
    assert sp.preserve_specific_experience_details(record([transplanted]), record(sources[:1])) == (record([transplanted]), [])


def test_full_pipeline_retains_bridge_and_owners_while_restoring_period():
    first_source = "Tracked monthly energy usage across 12 locations against the same month in the prior year."
    first_draft = "Tracked monthly energy usage across 12 locations using period-over-period analytics."
    second_source = "Flagged slow-moving stock and proposed a reduction plan adopted by 2 branches."
    second_draft = "Flagged slow-moving stock and contributed to inventory optimization through a reduction plan adopted by 2 branches."
    source = record([first_source]) + "\n---\n" + record([second_source], "South Works", "Planner | 2018 - 2019")
    draft = record([first_draft]) + "\n" + record([second_draft], "South Works", "Planner | 2018 - 2019")
    out, changes = sp.preserve_source_facts(draft, source, ["inventory optimization"])
    assert first_source in out
    assert second_draft in out
    assert out.index(first_source) < out.index("**South Works**") < out.index(second_draft)
    assert "restored a same-month prior-year comparison" in changes