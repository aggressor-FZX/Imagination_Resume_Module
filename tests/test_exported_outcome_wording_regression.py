"""Desired-behavior regression for the exported unsupported wording and qualifier.

Measured on the retained Imaginative run ``de375f60`` (capture
``captures/restoration-mapping-validation/imaginative-535341063cd245f7be18ab5ac0132de1``).
The run was APPROVED (``requires_review: false``, receipt issued, export 200) and
its compiled PDF contains wording the source never states, while dropping a
source qualifier. That is a false approval, not an acceptable outcome.

Exact source -> output comparison (frozen fixture, verbatim):

  SOURCE  "- SYNTHETIC-ONLY Reconciled 1,180 fuel invoices against depot
           dispensing records each quarter."
  OUTPUT  "- Reconciled 1,180 fuel invoices quarterly against depot records,
           ensuring accurate fuel expense reporting."
  ADDED   "ensuring accurate fuel expense reporting" -- an outcome/purpose
          clause no source states.

  SOURCE  "- SYNTHETIC-ONLY Prepared the monthly cost summary reviewed by the
           regional manager."
  OUTPUT  "- Prepared monthly cost summaries for regional management review,
           highlighting key cost drivers."
  ADDED   "highlighting key cost drivers" -- an outcome clause no source states.

  SOURCE  "- SYNTHETIC-ONLY Compared quarterly parts usage against the prior
           four quarters to set reorder points."
  OUTPUT  "- Analyzed quarterly parts usage with historical comparison to set
           data-driven reorder points."
  DROPPED "against the prior four quarters" -- the specific comparison basis.

WHY APPROVAL ALLOWED THEM (verified directly against the real bullets):

* ``_introduces_beyond`` returns ``[]`` for all three. It detects only
  capitalised proper nouns and unbacked figures; these are lowercase outcome
  words with no new figure.
* ``_quote_shares_content`` returns ``True`` for each, because the rewritten
  bullet still shares content words with its source item. A shared word proves
  relatedness, not that the whole meaning is preserved.
* The reviewer marked these lines ``supported``, so flag-then-repair never
  engaged, and the completeness check asks only whether each required item is
  *represented*, not whether extra claims were added or a qualifier dropped.

THE REPAIR. ``source_preservation.preserve_experience_bullets`` is the
deterministic layer that corrects writer output BEFORE review. It now:

* replaces a bullet whose faithful prefix carries an ungrounded trailing clause
  with the candidate's own source line (never composing prose);
* restores a bullet that dropped a numeric qualifier, including a spelled-out
  number such as "four quarters";
* leaves a faithful paraphrase untouched -- verbatim copying is NOT required;
* refuses a figure transplant, a DOCX-style source, and a bullet whose prefix
  is itself ungrounded, so no gate is weakened.

These tests assert the DESIRED behavior on the exact retained failing output:
it must be corrected, not merely characterized as currently accepted. The
frozen fixture is unchanged and no factual check is weakened.
"""
import asyncio
import json
import re
from pathlib import Path

import pytest

import correction_protocol as protocol
import source_correction as sc
import source_preservation as sp
from skill_bridge import build_bridges
from source_correction import SourceDocument, correct

BASE = (Path(__file__).resolve().parent.parent / "captures"
        / "restoration-mapping-validation"
        / "imaginative-535341063cd245f7be18ab5ac0132de1")

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bridge_opportunity"

# Verbatim source bullets from the frozen fixture.
SOURCE_RECONCILED = ("- SYNTHETIC-ONLY Reconciled 1,180 fuel invoices against "
                     "depot dispensing records each quarter.")
SOURCE_COST_SUMMARY = ("- SYNTHETIC-ONLY Prepared the monthly cost summary "
                       "reviewed by the regional manager.")
SOURCE_REORDER = ("- SYNTHETIC-ONLY Compared quarterly parts usage against the "
                  "prior four quarters to set reorder points.")

# Verbatim output bullets from the approved final_written_section_markdown.
OUTPUT_RECONCILED = ("- Reconciled 1,180 fuel invoices quarterly against depot "
                     "records, ensuring accurate fuel expense reporting.")
OUTPUT_COST_SUMMARY = ("- Prepared monthly cost summaries for regional "
                       "management review, highlighting key cost drivers.")
OUTPUT_REORDER = ("- Analyzed quarterly parts usage with historical comparison "
                  "to set data-driven reorder points.")

CASES = [
    ("ensuring accurate fuel expense reporting", SOURCE_RECONCILED, OUTPUT_RECONCILED),
    ("highlighting key cost drivers", SOURCE_COST_SUMMARY, OUTPUT_COST_SUMMARY),
    ("omitted 'against the prior four quarters'", SOURCE_REORDER, OUTPUT_REORDER),
]


def _frozen_source() -> str:
    """The exact frozen two-document source, joined as the orchestrator does."""
    main = (FIXTURE / "candidate-main.txt").read_text()
    supp = (FIXTURE / "candidate-supplement.txt").read_text()
    return main + "\n\n---\n\n" + supp


def _retained_markdown() -> str:
    """The exact approved output from the retained capture, verbatim."""
    return json.loads((BASE / "analysis.raw").read_text())[
        "final_written_section_markdown"]


# --- 1. Why the false approval happened (unchanged, still measured) ---------

@pytest.mark.parametrize("label,source,output", CASES,
                         ids=[c[0] for c in CASES])
def test_the_addition_check_does_not_flag_the_observed_wording(label, source, output):
    """The engine's lexical veto sees no introduced name or figure."""
    assert sc._introduces_beyond(output, [source], source) == [], label


@pytest.mark.parametrize("label,source,output", CASES,
                         ids=[c[0] for c in CASES])
def test_the_rewritten_bullet_still_shares_content_with_its_source(label, source, output):
    """Relatedness is not fidelity."""
    assert sc._quote_shares_content(output, source) is True, label


def test_the_added_outcome_clauses_are_not_in_the_source():
    """The additions are genuinely unbacked, not paraphrases of source words."""
    all_source = SOURCE_RECONCILED + " " + SOURCE_COST_SUMMARY + " " + SOURCE_REORDER
    for phrase in ("ensuring accurate fuel expense reporting",
                   "highlighting key cost drivers"):
        assert phrase not in all_source
        assert all(word[0].islower() for word in phrase.split())


def test_the_dropped_qualifier_is_a_real_source_fact():
    """The omitted qualifier is a specific comparison basis, not a stop word."""
    assert "against the prior four quarters" in SOURCE_REORDER
    assert "against the prior four quarters" not in OUTPUT_REORDER
    assert "historical comparison" in OUTPUT_REORDER


# --- 2. DESIRED BEHAVIOR: the exact failing output is corrected -------------

def test_the_retained_failing_output_is_corrected_not_approved_as_is():
    """The whole retained markdown must not survive preservation unchanged."""
    corrected, changes = sp.preserve_source_facts(_retained_markdown(), _frozen_source())
    assert changes, "the false-approval output was left untouched"
    assert corrected != _retained_markdown()


def test_the_two_unsupported_outcome_clauses_are_removed():
    """Both appended outcome clauses must be gone from the corrected document."""
    corrected, _ = sp.preserve_source_facts(_retained_markdown(), _frozen_source())
    assert "ensuring accurate fuel expense reporting" not in corrected
    assert "highlighting key cost drivers" not in corrected


def test_the_four_quarter_comparison_period_is_restored():
    """The material qualifier must survive, not be replaced by a vague phrase."""
    corrected, _ = sp.preserve_source_facts(_retained_markdown(), _frozen_source())
    assert "against the prior four quarters" in corrected
    assert "historical comparison" not in corrected
    assert "data-driven reorder points" not in corrected


def test_the_supported_figures_and_records_survive_the_correction():
    """Correcting the additions must not drop the candidate's real facts."""
    corrected, _ = sp.preserve_source_facts(_retained_markdown(), _frozen_source())
    for fact in ("1,180 fuel invoices", "14 depots", "7 service branches",
                 "2 branches", "3 years", "9 years"):
        assert fact in corrected, f"supported fact lost: {fact}"


def test_every_changed_line_is_the_candidates_own_wording():
    """The repair substitutes source text; it never composes a new sentence."""
    corrected, _ = sp.preserve_source_facts(_retained_markdown(), _frozen_source())
    source = _frozen_source().replace("SYNTHETIC-ONLY", "")
    draft_lines = {line.strip().lstrip("- ").strip()
                   for line in _retained_markdown().splitlines() if line.strip()}
    for line in corrected.splitlines():
        candidate = line.strip().lstrip("- ").strip()
        if not candidate or candidate in draft_lines or line.strip().startswith("#"):
            continue
        assert candidate in source, f"preservation invented a line: {candidate!r}"


# --- 3. Faithful paraphrase control: verbatim copying is NOT required -------

FAITHFUL_PARAPHRASES = [
    # Different verbs and inflections, no new content word, every figure kept.
    ("- Reconciled 1,180 fuel invoices against depot dispensing records each quarter.",
     SOURCE_RECONCILED),
    ("- The regional manager reviewed the monthly cost summary that was prepared.",
     SOURCE_COST_SUMMARY),
    ("- Compared quarterly parts usage against the prior four quarters to set reorder points.",
     SOURCE_REORDER),
    ("- Prepared monthly cost summaries for regional management review.",
     SOURCE_COST_SUMMARY),
]


@pytest.mark.parametrize("paraphrase,source", FAITHFUL_PARAPHRASES)
def test_a_faithful_paraphrase_is_left_untouched(paraphrase, source):
    """The repair must not require verbatim copying of the source line."""
    draft = "## Professional Experience\n" + paraphrase + "\n"
    out, changes = sp.preserve_experience_bullets(draft, _frozen_source())
    assert paraphrase in out, f"faithful paraphrase was rewritten: {paraphrase!r}"
    assert "removed an unsupported appended clause" not in changes
    assert "restored a dropped material qualifier" not in changes


def test_a_faithful_paraphrase_with_a_source_figure_is_not_trimmed():
    """'cutting rework by 18%' is a paraphrase, not an unsupported addition."""
    source = "- SYNTHETIC-ONLY Cut rework by 18% by standardizing panel labeling.\n"
    draft = "- Optimized panel labeling, cutting rework by 18%.\n"
    out, changes = sp.preserve_experience_bullets(draft, source)
    assert "cutting rework by 18%" in out
    assert changes == []


# --- 4. The gate is not weakened --------------------------------------------

def test_a_figure_transplant_is_left_for_the_reviewer():
    """A different source figure is a transplant, not a dropped qualifier."""
    source = ("- SYNTHETIC-ONLY Installed systems across 40+ job sites.\n"
              "- SYNTHETIC-ONLY Maintained 12 commercial buildings.\n")
    draft = "- Installed systems across 12 commercial buildings.\n"
    out, changes = sp.preserve_experience_bullets(draft, source)
    assert "12 commercial buildings" in out, "the transplant was erased"
    assert changes == []


def test_a_docx_style_source_is_left_to_the_reviewer():
    """No Markdown bullet markers means this rule does not apply."""
    source = "Performed preventive maintenance on shop welding equipment.\n"
    draft = "- Performed preventive maintenance on shop welding equipment, ensuring uptime.\n"
    out, changes = sp.preserve_experience_bullets(draft, source)
    assert "ensuring uptime" in out
    assert changes == []


# --- 4b. The three named safeguards ----------------------------------------

def test_a_number_bearing_unsupported_outcome_is_not_exempt():
    """'Trained 4 apprentices' must not justify 'increased productivity by 4%'.

    The clause carries a number that also occurs in the source, but the 4%
    outcome is not a restatement of four apprentices. A rule that exempted any
    clause carrying a source figure would admit this invented claim.
    """
    source = "- SYNTHETIC-ONLY Trained 4 apprentices on lockout/tagout procedures.\n"
    draft = ("- Trained 4 apprentices on lockout/tagout procedures, "
             "increased productivity by 4%.\n")
    out, changes = sp.preserve_experience_bullets(draft, source)
    assert "increased productivity by 4%" not in out
    assert "removed an unsupported appended clause" in changes
    # The candidate's real fact survives.
    assert "Trained 4 apprentices" in out


def test_a_permitted_skill_bridge_survives_experience_bullet_correction():
    """An authorised inferred label must reach semantic review, not be erased.

    Imaginative passes the validated bridge labels; Conservative passes none.
    The same draft is therefore corrected in Conservative (the label is not
    permitted) and left alone in Imaginative (it is).
    """
    source = "- SYNTHETIC-ONLY Compared weekly sales figures across 12 branches.\n"
    draft = ("- Compared weekly sales figures across 12 branches, "
             "applying trend analysis.\n")
    imaginative, ch_i = sp.preserve_experience_bullets(
        draft, source, permitted_bridges=["trend analysis"])
    assert "trend analysis" in imaginative
    assert ch_i == []
    conservative, ch_c = sp.preserve_experience_bullets(draft, source)
    assert "removed an unsupported appended clause" in ch_c
    assert "trend analysis" not in conservative


def test_a_permitted_bridge_does_not_authorise_an_outcome_in_the_same_bullet():
    """The bridge exemption is phrase-scoped, not clause-scoped.

    A validated label may stand, but it must not shield an unsupported outcome
    sitting beside it. The whole bullet is corrected from the candidate's own
    source line, so neither the invented outcome nor the inferred label is
    silently approved by a lexical rule.
    """
    source = ("**SYNTHETIC-ONLY Acme Corp**\n"
              "- SYNTHETIC-ONLY Compared weekly sales figures across 12 branches.\n")
    draft = ("**Acme Corp**\n"
             "- Compared weekly sales figures across 12 branches, applying trend "
             "analysis and ensuring accurate reporting.\n")
    out, changes = sp.preserve_experience_bullets(
        draft, source, permitted_bridges=["trend analysis"])
    assert "ensuring accurate reporting" not in out
    assert "removed an unsupported appended clause" in changes
    # The candidate's real activity survives, verbatim from the source.
    assert "Compared weekly sales figures across 12 branches." in out


def test_a_permitted_bridge_alone_is_not_erased():
    """A bridge with no unsupported outcome beside it still reaches review."""
    source = ("**SYNTHETIC-ONLY Acme Corp**\n"
              "- SYNTHETIC-ONLY Compared weekly sales figures across 12 branches.\n")
    draft = ("**Acme Corp**\n"
             "- Compared weekly sales figures across 12 branches, applying trend analysis.\n")
    out, changes = sp.preserve_experience_bullets(
        draft, source, permitted_bridges=["trend analysis"])
    assert "trend analysis" in out
    assert changes == []


def test_employer_attribution_is_not_crossed_by_the_correction():
    """A correction under one employer must not touch another employer's bullet."""
    source = ("**SYNTHETIC-ONLY Alpha Corp**\n"
              "- SYNTHETIC-ONLY Reconciled 1,180 fuel invoices each quarter.\n"
              "**SYNTHETIC-ONLY Beta Corp**\n"
              "- SYNTHETIC-ONLY Prepared the monthly cost summary.\n")
    draft = ("**Alpha Corp**\n"
             "- Reconciled 1,180 fuel invoices each quarter, ensuring accurate reporting.\n"
             "**Beta Corp**\n"
             "- Prepared the monthly cost summary.\n")
    out, changes = sp.preserve_experience_bullets(draft, source)
    assert "ensuring accurate reporting" not in out
    assert "- Prepared the monthly cost summary." in out
    assert changes == ["removed an unsupported appended clause"]


def test_a_source_line_from_another_employer_is_not_substituted():
    """Automatic replacement must not cross employer ownership.

    Beta's draft bullet carries Alpha's fact (a transplant) plus an unsupported
    outcome. The only source bullet it lexically matches belongs to Alpha. The
    deterministic layer must NOT substitute Alpha's line under Beta's heading;
    it leaves the bullet for the reviewer instead.
    """
    source = ("**SYNTHETIC-ONLY Alpha Corp**\n"
              "- SYNTHETIC-ONLY Reconciled 1,180 fuel invoices each quarter.\n"
              "**SYNTHETIC-ONLY Beta Corp**\n"
              "- SYNTHETIC-ONLY Prepared the monthly cost summary reviewed by the "
              "regional manager.\n")
    draft = ("**Alpha Corp**\n"
             "- Reconciled 1,180 fuel invoices each quarter.\n"
             "**Beta Corp**\n"
             "- Reconciled 1,180 fuel invoices each quarter, ensuring accurate "
             "reporting.\n")
    out, changes = sp.preserve_experience_bullets(draft, source)
    assert changes == [], "a cross-employer substitution was made"
    assert "- Reconciled 1,180 fuel invoices each quarter, ensuring accurate reporting." in out


def test_ambiguous_ownership_is_left_for_review():
    """Two employers with identical source bullets: no automatic replacement."""
    source = ("**SYNTHETIC-ONLY Alpha Corp**\n"
              "- SYNTHETIC-ONLY Reconciled 1,180 fuel invoices each quarter.\n"
              "**SYNTHETIC-ONLY Beta Corp**\n"
              "- SYNTHETIC-ONLY Reconciled 1,180 fuel invoices each quarter.\n")
    draft = ("**Alpha Corp**\n"
             "- Reconciled 1,180 fuel invoices each quarter.\n"
             "**Beta Corp**\n"
             "- Reconciled 1,180 fuel invoices each quarter, ensuring accurate "
             "reporting.\n")
    out, changes = sp.preserve_experience_bullets(draft, source)
    assert changes == [], "an ambiguous match was resolved by guessing"
    assert "ensuring accurate reporting" in out


# --- 4c. Through the real correction path ----------------------------------

def test_bridge_plus_outcome_through_the_correction_path_is_not_approved():
    """A permitted bridge must not authorise an outcome in the same bullet.

    Drives the exact shape through preservation AND the real correction engine
    with a scripted reviewer that marks every line supported -- the observed
    ``de375f60`` false-approval mode. The unsupported outcome must be gone from
    the delivered text; a green approval carrying it would be the defect.
    """
    source = ("## Professional Experience\n"
              "**SYNTHETIC-ONLY Acme Corp**\n"
              "- SYNTHETIC-ONLY Compared weekly sales figures across 12 branches.\n")
    draft = ("## Professional Experience\n"
             "**Acme Corp**\n"
             "- Compared weekly sales figures across 12 branches, applying trend "
             "analysis and ensuring accurate reporting.\n")

    accepted, _ = build_bridges(
        [{"skill": "trend analysis",
          "evidence_quote": "Compared weekly sales figures across 12 branches",
          "rationale": "The activity identifies changes over time.",
          "source_id": "document-0"}],
        sources={"document-0": source}, mode="imaginative")
    bridges = [b.to_dict() for b in accepted]
    assert bridges, "expected a validated bridge"

    preserved, _ = sp.preserve_source_facts(
        draft, source, permitted_bridges=["trend analysis"])
    assert "ensuring accurate reporting" not in preserved, (
        "preservation let the unsupported outcome through")

    class ApprovingReviewer:
        """Marks every line supported -- the observed false-approval mode."""

        def __init__(self):
            self.phases = []

        async def __call__(self, phase, payload):
            self.phases.append(phase)
            if phase == "source_inventory":
                request = protocol.indexed_inventory_request(payload)
                return protocol.strict_inventory_from_indexed(
                    {"required": [s["index"] for s in request["_segments"]],
                     "non_resume": [], "uncertain": []}, request)
            if phase in ("initial_review", "post_patch_review",
                         "post_restore_review"):
                request = protocol.indexed_review_request(payload)
                lines = request["_lines"]
                structural = [l["line_id"] for l in lines
                              if l["text"].strip().startswith("#")]
                supported = [l["line_id"] for l in lines
                             if l["line_id"] not in structural]
                answer = {
                    "supported": supported, "structural": structural,
                    "evidence": {lid: list(range(len(request["_items"])))
                                 for lid in supported},
                    "represented": list(range(len(request["_items"]))),
                    "missing": [], "unsupported": [], "uncertain": [],
                }
                return protocol.strict_review_from_indexed(answer, request)
            return {"edits": []}

    reviewer = ApprovingReviewer()
    result = asyncio.run(correct(
        preserved, (SourceDocument("document-0", source),), reviewer,
        timeout_seconds=60, permitted_bridges=bridges))

    # Whatever the engine decides, the delivered text must not carry the
    # unsupported outcome. A withheld result is also an acceptable outcome.
    delivered = result.markdown or ""
    assert "ensuring accurate reporting" not in delivered, (
        "the unsupported outcome reached the delivered text")
    assert "Compared weekly sales figures across 12 branches" in delivered or (
        result.status == "withheld")


# --- 5. The observed case is real (retained evidence) ----------------------

@pytest.mark.skipif(not (BASE / "analysis.raw").exists(),
                    reason="protected capture not present")
def test_the_approved_response_really_contains_the_wording():
    """The false approval is real: the approved markdown carries these bullets."""
    raw = json.loads((BASE / "analysis.raw").read_text())
    assert raw.get("requires_review") is False
    assert raw.get("publication_receipt")
    markdown = raw["final_written_section_markdown"]
    assert "ensuring accurate fuel expense reporting" in markdown
    assert "highlighting key cost drivers" in markdown
    assert "against the prior four quarters" not in markdown


@pytest.mark.skipif(not (BASE / "export.raw").exists(),
                    reason="protected capture not present")
def test_the_wording_reached_the_compiled_pdf():
    """Export 200 is not factual acceptance: the PDF carries the same wording."""
    import base64
    import zlib

    data = json.loads((BASE / "export.raw").read_bytes())
    pdf = base64.b64decode(data["pdf_base64"])
    blob = ""
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", pdf, re.DOTALL):
        try:
            blob += zlib.decompress(match.group(1)).decode("latin-1") + "\n"
        except Exception:
            pass
    fragments = re.findall(r"\((?:[^()\\]|\\.)*\)", blob)
    compact = re.sub(r"\s+", "", "".join(f[1:-1] for f in fragments)).lower()

    def present(phrase):
        return re.sub(r"\s+", "", phrase).lower() in compact

    assert present("ensuring accurate fuel expense reporting")
    assert present("highlighting key cost drivers")
    assert present("historical comparison")
    assert not present("against the prior four quarters")
