"""A bare canonical heading must survive correction into the approved Markdown.

Measured 2026-11-07 on ``realistic_candidate`` / Conservative. The editor logged
``Keeping empty section header 'Projects'``, the correction then revised one span,
and the approved markdown carried five headings instead of six. Re-running
preservation on the delivered text added ``Projects`` back, which proves the
heading was present and was removed.

A heading asserts no candidate fact. It names a section the product promises, so
losing one removes structure without removing content — the worst of both.

The check runs at the approval point, after every edit, so it is the last word on
structure. It is a STRUCTURE repair: it inserts a heading line and changes no
existing character, which is why it does not require re-review.

These tests pin the fix AND the four narrowness guarantees that keep it from
becoming a content rewriter.
"""

import source_correction as sc

# The exact shape that lost Projects: five headings, no Projects.
FIVE_HEADINGS = """## Professional Summary

Journeyman electrician with 12 years of experience.

## Professional Experience
**Northline Fabrication, Inc.** -- Spokane, WA
- Fabricated structural steel assemblies for 14 industrial contracts.

## Education
**Spokane Falls Institute of Technology** -- Spokane, WA

## Certifications
- Washington 01 Journey Level Electrician License

## Technical Skills
- Conduit bending, switchgear troubleshooting
"""

# The draft as correction RECEIVED it: the same document with Projects present.
# The invariant is stated against this, not against the source — a heading that
# was there when correction started must still be there when it approves.
DRAFT_AT_ENTRY = FIVE_HEADINGS.replace(
    "## Education", "## Projects\n\n## Education", 1)


def _headings(markdown):
    return [line.strip() for line in markdown.splitlines()
            if line.strip().startswith("## ")]


# ── The fix ────────────────────────────────────────────────────────────────

def test_the_missing_projects_heading_is_restored():
    """The exact measured case: five headings in, six out."""
    out, changes = sc._restore_missing_headings(FIVE_HEADINGS, DRAFT_AT_ENTRY)
    assert "restored the Projects heading" in changes
    assert "## Projects" in out
    assert len(_headings(out)) == 6


def test_the_restored_heading_lands_in_the_promised_order():
    """Projects belongs between Professional Experience and Education."""
    out, _ = sc._restore_missing_headings(FIVE_HEADINGS, DRAFT_AT_ENTRY)
    assert _headings(out) == [
        "## Professional Summary",
        "## Professional Experience",
        "## Projects",
        "## Education",
        "## Certifications",
        "## Technical Skills",
    ]


def test_restoring_a_heading_changes_no_existing_character():
    """A structure repair must not edit content, which is why it needs no
    re-review."""
    out, _ = sc._restore_missing_headings(FIVE_HEADINGS, DRAFT_AT_ENTRY)
    original = [line for line in FIVE_HEADINGS.splitlines() if line.strip()]
    restored = [line for line in out.splitlines() if line.strip()]
    # Every original line survives, in order, with only the heading inserted.
    assert [line for line in restored if line != "## Projects"] == original


def test_a_complete_document_is_left_untouched():
    """Nothing is added when every promised heading is already present."""
    complete = FIVE_HEADINGS.replace(
        "## Education", "## Projects\n\n## Education")
    out, changes = sc._restore_missing_headings(complete, complete)
    assert changes == []
    assert out == complete


# ── Narrowness guarantee 1: factual text is never a heading ────────────────

def test_a_factual_line_beginning_with_a_canonical_name_is_not_a_heading():
    """A bullet that starts with "Projects" is content, not structure.

    The check must not exempt factual text merely because it looks like a
    heading. Only a BARE line — the whole line is the name — counts.
    """
    markdown = """## Professional Summary

Electrician.

## Professional Experience
- Projects delivered on schedule across 6 sites.
- Education and training of 4 apprentices.
"""
    draft_at_entry = markdown.replace(
        "## Professional Experience",
        "## Projects\n\n## Education\n\n## Professional Experience", 1)
    out, changes = sc._restore_missing_headings(markdown, draft_at_entry)
    # The factual lines are untouched.
    assert "- Projects delivered on schedule across 6 sites." in out
    assert "- Education and training of 4 apprentices." in out
    # And the real headings are still restored, because those bullets are not
    # headings.
    assert "restored the Projects heading" in changes
    assert "restored the Education heading" in changes


def test_a_sentence_containing_a_canonical_name_is_not_a_heading():
    """A prose line that mentions a section name is not that section."""
    markdown = """## Professional Summary

Electrician.

## Professional Experience
Projects and Education were both handled.
"""
    draft_at_entry = markdown.replace(
        "## Professional Experience",
        "## Projects\n\n## Education\n\n## Professional Experience", 1)
    out, changes = sc._restore_missing_headings(markdown, draft_at_entry)
    assert "Projects and Education were both handled." in out
    assert "restored the Projects heading" in changes


# ── Narrowness guarantee 2: an existing alias is not duplicated ────────────

def test_an_existing_alias_counts_as_the_canonical_heading():
    """``## Skills`` is ``Technical Skills``; it must not be duplicated."""
    markdown = """## Professional Summary

Electrician.

## Professional Experience
- Did work.

## Skills
- SQL
"""
    out, changes = sc._restore_missing_headings(markdown, markdown)
    assert "restored the Technical Skills heading" not in changes
    assert out.count("## Skills") == 1
    assert "## Technical Skills" not in out


# ── Narrowness guarantee 3: the summary stays conditional ──────────────────

def test_no_summary_heading_is_invented_when_the_source_has_none():
    """An empty summary heading would claim a summary the candidate never wrote.

    This mirrors ``ensure_required_headings``, which treats the summary as the
    one conditional heading.
    """
    markdown = "## Professional Experience\n- Did work.\n"
    # The draft never carried a summary heading, so none may be invented.
    out, changes = sc._restore_missing_headings(markdown, markdown)
    assert "## Professional Summary" not in out
    assert "restored the Professional Summary heading" not in changes


def test_the_summary_heading_is_restored_when_the_source_has_one():
    markdown = "## Professional Experience\n- Did work.\n"
    draft_at_entry = "## Professional Summary\n\nElectrician.\n\n" + markdown
    out, changes = sc._restore_missing_headings(markdown, draft_at_entry)
    assert "## Professional Summary" in out
    assert "restored the Professional Summary heading" in changes


# ── Narrowness guarantee 4: nothing is moved or duplicated ─────────────────

def test_an_existing_heading_is_never_moved():
    """A section already present keeps its position, even out of order."""
    markdown = """## Professional Experience
- Did work.

## Professional Summary

Electrician.
"""
    out, _ = sc._restore_missing_headings(markdown, markdown)
    # Professional Experience still comes first, as the author left it.
    assert _headings(out)[0] == "## Professional Experience"


def test_no_heading_is_duplicated():
    out, _ = sc._restore_missing_headings(FIVE_HEADINGS, DRAFT_AT_ENTRY)
    for name in ("Professional Summary", "Professional Experience", "Projects",
                 "Education", "Certifications", "Technical Skills"):
        assert out.count(f"## {name}") == 1, name
