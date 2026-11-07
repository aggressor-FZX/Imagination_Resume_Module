"""An approved result must never carry a provenance marker.

Measured on Conservative website run ``ebab2e2f`` (2026-11-07, revision
``555e26b``). The run was approved, but the delivered education line read:

  SYNTHETIC-ONLY Inland Records Institute -- Fife, WA

The marker is fixture metadata, never a resume fact. ``preserve_source_facts``
strips it at its own render point, but a later repair edit can copy a source line
verbatim and bring the marker back. The invariant is therefore enforced once at
the approval point, where every edit path converges.

These tests reuse the engine harness in ``test_source_correction`` so the review
envelope is the same shape the real route produces.
"""
from tests.test_source_correction import DRAFT as BASE_DRAFT, review, run

# The draft carries the marker, which is the shape a verbatim repair produces.
DRAFT = BASE_DRAFT.replace("Attended North College",
                           "SYNTHETIC-ONLY Attended North College")


def _run(draft):
    async def callback(phase, payload):
        return review(payload)

    return run(draft, callback)


def test_an_approved_result_never_carries_a_provenance_marker():
    result = _run(DRAFT)
    assert result.status == "approved", result.reason
    assert "SYNTHETIC-ONLY" not in result.markdown


def test_the_marker_is_removed_without_losing_the_fact():
    result = _run(DRAFT)
    assert "North College" in result.markdown
    assert "30%" in result.markdown


def test_a_clean_draft_is_unchanged_by_the_invariant():
    clean = DRAFT.replace("SYNTHETIC-ONLY ", "")
    result = _run(clean)
    assert result.status == "approved", result.reason
    assert result.markdown.count("North College") == 1
