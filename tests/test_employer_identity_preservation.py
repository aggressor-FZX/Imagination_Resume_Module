"""Offline ownership regressions for employer-record preservation."""
import pytest

from source_preservation import preserve_employer_records


SOURCE = """## Professional Experience
**Alpha Corp** -- City, WA
*Engineer | 2019 - 2021*
- Built the alpha tracking system.

**Beta Corp** -- Town, OR
*Analyst | 2021 - 2023*
- Audited the beta inventory.
"""


@pytest.mark.parametrize("draft", [
    # Reordering with generic labels cannot prove ownership.
    """## Professional Experience
**Current Employer** --
*Analyst |*
- Audited beta inventory.
**Previous Employer** --
*Engineer |*
- Built an alpha tracker.
""",
    # Even one known employer does not justify assigning the other by position.
    """## Professional Experience
**Beta Corp** -- Town, OR
*Analyst | 2021 - 2023*
- Audited beta inventory.
**Current Employer** --
*Engineer |*
- Built an alpha tracker.
""",
])
def test_ambiguous_reordered_records_are_unchanged(draft):
    assert preserve_employer_records(draft, SOURCE) == (draft, [])


def test_reordered_placeholders_use_unique_exact_bullets_not_position():
    draft = """## Professional Experience
**Current Employer** --
*Analyst |*
- Audited the beta inventory.
**Previous Employer** --
*Engineer |*
- Built the alpha tracking system.
"""
    out, changes = preserve_employer_records(draft, SOURCE)
    assert changes == ["restored the source's employer and role records"]
    assert "**Beta Corp** -- Town, OR\n*Analyst | 2021 - 2023*\n- Audited the beta inventory." in out
    assert "**Alpha Corp** -- City, WA\n*Engineer | 2019 - 2021*\n- Built the alpha tracking system." in out
    assert preserve_employer_records(out, SOURCE) == (out, [])


def test_mixed_records_repair_only_the_uniquely_identified_placeholder():
    draft = """## Professional Experience
**Beta Corp** -- Town, OR
*Analyst | 2021 - 2023*
- Audited the beta inventory.
**Current Employer** --
*Engineer |*
- Built the alpha tracking system.
"""
    out, changes = preserve_employer_records(draft, SOURCE)
    assert changes
    assert "**Beta Corp** -- Town, OR\n*Analyst | 2021 - 2023*\n- Audited the beta inventory." in out
    assert "**Alpha Corp** -- City, WA\n*Engineer | 2019 - 2021*\n- Built the alpha tracking system." in out


def test_real_employer_can_restore_only_its_own_dateless_role():
    draft = """## Professional Experience
**Beta Corp** -- Town, OR
*Analyst |*
- Audited beta inventory.
**Alpha Corp** -- City, WA
*Engineer | 2019 - 2021*
- Built an alpha tracker.
"""
    out, changes = preserve_employer_records(draft, SOURCE)
    assert changes
    assert "**Beta Corp** -- Town, OR\n*Analyst | 2021 - 2023*" in out
    assert "**Alpha Corp** -- City, WA\n*Engineer | 2019 - 2021*" in out


def test_placeholder_can_be_identified_by_unique_exact_role_and_dates():
    draft = """## Professional Experience
**Current Employer** --
*Analyst | 2021 - 2023*
- Audited beta inventory.
**Previous Employer** --
*Engineer | 2019 - 2021*
- Built an alpha tracker.
"""
    out, changes = preserve_employer_records(draft, SOURCE)
    assert changes
    assert "**Beta Corp** -- Town, OR\n*Analyst | 2021 - 2023*" in out
    assert "**Alpha Corp** -- City, WA\n*Engineer | 2019 - 2021*" in out


def test_conflicting_evidence_never_reassigns_ownership():
    draft = """## Professional Experience
**Beta Corp** -- Town, OR
*Analyst |*
- Built the alpha tracking system.
**Current Employer** --
*Analyst | 2021 - 2023*
- Built the alpha tracking system.
"""
    assert preserve_employer_records(draft, SOURCE) == (draft, [])


def test_identical_bullets_and_duplicate_draft_claims_are_ambiguous():
    shared = SOURCE.replace("Audited the beta inventory.", "Built the alpha tracking system.")
    draft = """## Professional Experience
**Current Employer** --
*Analyst |*
- Built the alpha tracking system.
**Previous Employer** --
*Engineer |*
- Built the alpha tracking system.
"""
    assert preserve_employer_records(draft, shared) == (draft, [])
    # Even unique source evidence cannot identify which duplicate draft owns it.
    assert preserve_employer_records(draft, SOURCE) == (draft, [])


def test_unknown_real_employer_and_nonplaceholder_role_are_never_overwritten():
    draft = """## Professional Experience
**Unknown Corp** -- City, WA
*Engineer |*
- Built the alpha tracking system.
**Current Employer** --
*Director | 2010 - 2015*
- Audited the beta inventory.
"""
    assert preserve_employer_records(draft, SOURCE) == (draft, [])


def test_single_record_without_identity_evidence_is_not_assigned_by_elimination():
    source = "## Professional Experience\n**Alpha Corp**\n*Engineer | 2019 - 2021*\n"
    draft = "## Professional Experience\n**Current Employer**\n*Engineer |*\n"
    assert preserve_employer_records(draft, source) == (draft, [])


def test_repeated_employer_name_needs_distinct_record_evidence():
    source = """## Professional Experience
**Alpha Corp** -- City, WA
*Engineer | 2019 - 2021*
- Built the alpha tracking system.
**Alpha Corp** -- City, WA
*Manager | 2021 - 2023*
- Led the alpha operations group.
"""
    draft = """## Professional Experience
**Alpha Corp** -- City, WA
*Engineer |*
- Built an alpha tracker.
"""
    assert preserve_employer_records(draft, source) == (draft, [])


def test_matching_bullet_in_next_record_does_not_identify_first_record():
    draft = """## Professional Experience
**Current Employer** --
*Engineer |*
**Previous Employer** --
*Analyst |*
- Audited the beta inventory.
"""
    out, changes = preserve_employer_records(draft, SOURCE)
    assert changes
    assert "**Current Employer** --\n*Engineer |*" in out
    assert "**Beta Corp** -- Town, OR\n*Analyst | 2021 - 2023*\n- Audited the beta inventory." in out