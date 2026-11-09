"""[retained-approved-tails] Two unsupported clauses reached an APPROVED document.

The case
--------
Retained input: `tests/fixtures/realistic_candidate` (candidate-main.txt,
candidate-supplement.txt, intended-job-description.txt), verified byte-identical
to the recorded failing run by sha256:

    candidate-main.txt            d5b7a9d408e43dfeb48f2c35837878b2d23b1800de…
    candidate-supplement.txt      3c6f9fcfc6681ef70bf711c63f46d9fbded07004cb28…
    intended-job-description.txt  06577fab0ca934efd8872e83dc691d76ac918710c69…

Reproduced through the real provider adapter
(`scripts/repro_correction_case.py`), the run **approved** — and the approved
markdown carried two clauses the candidate never wrote:

    source   : "Cut average fault-response time from 4.5 hours to 2.75 hours."
    approved : "…to 2.75 hours through targeted troubleshooting protocols."

    source   : "Reduced after-hours callouts by 27% by adding thermographic
                inspections."
    approved : "…by implementing thermographic inspections and proactive
                maintenance strategies."

Neither clause appears anywhere in either source document.

Root cause — two independent blind spots, both at the trimming boundary
-----------------------------------------------------------------------
1. `_TRAILING_PURPOSE_BOUNDARY` recognised `to`/`for`/`so that`/`using`/`toward`/
   `in compliance with` but **not `through` or `by`**. So no boundary was offered
   for either tail, and `_trim_trailing_purpose` — which already implements every
   measured guard — was never called.
2. `_introduces_beyond` flags only CAPITALISED names and unbacked figures. Every
   appended word here is lowercase prose, so it saw nothing.

So the trimmer was correct and unreachable. The repair widens the boundary
vocabulary only; the trimmer's existing rules still decide whether anything is
removed.

This is NOT a phrase blacklist: the boundary only decides where to look. The tail
must still introduce content the matched bullet lacks, and a mixed tail
substitutes the candidate's own bullet rather than composing prose.
"""

import pathlib

import pytest

import source_preservation as sp

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "realistic_candidate"

# The two bullets as the approved run delivered them.
APPROVED_BULLET_1 = ("- Cut average fault-response time from 4.5 hours to 2.75 "
                     "hours through targeted troubleshooting protocols.")
APPROVED_BULLET_2 = ("- Reduced after-hours callouts by 27% by implementing "
                     "thermographic inspections and proactive maintenance "
                     "strategies.")


@pytest.fixture(scope="module")
def source_text():
    return "\n\n---\n\n".join(
        (FIXTURE / name).read_text()
        for name in ("candidate-main.txt", "candidate-supplement.txt"))


def draft_with(bullets):
    return ("## Professional Experience\n"
            "**Pend Oreille Power Systems** -- Spokane Valley, WA\n"
            "*Journeyman Electrician | 07/2019 - 02/2023*\n"
            f"{bullets[0]}\n\n"
            "**Riverbend Facilities Group** -- Coeur d'Alene, ID\n"
            "*Maintenance Electrician | 03/2023 - Present*\n"
            f"{bullets[1]}\n")


def test_the_through_tail_is_removed(source_text):
    """`through targeted troubleshooting protocols` is not in any source."""
    assert "through targeted troubleshooting protocols" not in source_text.casefold()
    out, changes = sp.preserve_source_facts(
        draft_with((APPROVED_BULLET_1, "- Maintain electrical distribution.")),
        source_text)
    assert "targeted troubleshooting protocols" not in out
    assert "removed an unsupported appended clause" in changes


def test_the_and_tail_is_corrected(source_text):
    """`proactive maintenance strategies` is not in any source."""
    assert "proactive maintenance strategies" not in source_text.casefold()
    out, changes = sp.preserve_source_facts(
        draft_with(("- Installed panels.", APPROVED_BULLET_2)), source_text)
    assert "proactive maintenance strategies" not in out


def test_the_repair_emits_a_prefix_or_the_source_bullet_never_new_prose(source_text):
    """Removal must not compose replacement wording.

    The trimmer emits a PREFIX of the input; the mixed-tail path substitutes the
    candidate's own source bullet. Both are checked here on the real sources.
    """
    out, _ = sp.preserve_source_facts(
        draft_with((APPROVED_BULLET_1, APPROVED_BULLET_2)), source_text)
    kept = [line.strip()[2:] for line in out.splitlines()
            if line.strip().startswith("- ")]
    original_bodies = [APPROVED_BULLET_1[2:], APPROVED_BULLET_2[2:]]
    for body in kept:
        assert any(orig.startswith(body) or body in source_text
                   for orig in original_bodies), (
            f"emitted text is neither a prefix of the draft bullet nor present in "
            f"the source: {body!r}")


def test_the_source_figures_survive_the_repair(source_text):
    """Removing a clause must never drop a sourced number."""
    out, _ = sp.preserve_source_facts(
        draft_with((APPROVED_BULLET_1, APPROVED_BULLET_2)), source_text)
    for figure in ("4.5", "2.75", "27%"):
        assert figure in out, f"{figure} was lost"


# ── Negative controls: the widened boundary must not become a rewrite engine ──


@pytest.mark.parametrize("draft_bullet,source_bullet,label", [
    # A faithful verb rewrite inside an otherwise identical tail.
    ("- Reduced after-hours callouts by 27% by implementing thermographic "
     "inspections.",
     "- Reduced after-hours callouts by 27% by adding thermographic inspections.",
     "verb rewrite"),
    # A bullet identical to source must be untouched.
    ("- Cut average fault-response time from 4.5 hours to 2.75 hours.",
     "- Cut average fault-response time from 4.5 hours to 2.75 hours.",
     "identical"),
    # A conjunction of sourced material must not be split at `and`.
    ("- Pulled and terminated branch circuits across 31 commercial buildouts.",
     "- Pulled and terminated branch circuits across 31 commercial buildouts.",
     "sourced conjunction"),
])
def test_faithful_text_is_left_unchanged(draft_bullet, source_bullet, label):
    src = ("## Professional Experience\n"
           "**X Corp** -- Spokane, WA\n"
           "*Electrician | 01/2020 - Present*\n"
           f"- {source_bullet}\n")
    draft = ("## Professional Experience\n"
             "**X Corp** -- Spokane, WA\n"
             "*Electrician | 01/2020 - Present*\n"
             f"{draft_bullet}\n")
    out, changes = sp.preserve_experience_bullets(draft, src)
    assert changes == [], f"{label} was wrongly corrected: {changes}"
    assert draft_bullet[2:] in out, f"{label} text was altered"


def test_a_same_document_but_wrong_record_clause_is_still_corrected(source_text):
    """A clause sourced from ANOTHER record does not authorise this bullet.

    `thermographic inspection` legitimately appears in the candidate's Skills and
    in the Riverbend bullet. It must not therefore be accepted as a method for the
    Pend Oreille fault-response bullet, which is a different employer record.
    """
    draft = ("## Professional Experience\n"
             "**Pend Oreille Power Systems** -- Spokane Valley, WA\n"
             "*Journeyman Electrician | 07/2019 - 02/2023*\n"
             "- Cut average fault-response time from 4.5 hours to 2.75 hours "
             "through thermographic inspection.\n")
    out, changes = sp.preserve_source_facts(draft, source_text)
    assert "through thermographic inspection" not in out, (
        "a clause sourced from a DIFFERENT record was accepted as evidence for "
        "this bullet")
    assert "removed an unsupported appended clause" in changes
