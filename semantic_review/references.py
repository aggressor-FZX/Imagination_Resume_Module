"""Reference validity: does the reviewer's evidence point at something real?

This is the last runtime dependency that lived in `experiment/`. It is re-implemented
here so production has no edge into the measurement apparatus.

The distinction this preserves
------------------------------
A resolving reference proves the model pointed at real text. It does **not** prove
the explanation is true, and its result must never be reported as if it did. The
previous design's exhaustive bookkeeping bought no faithfulness precisely because
validity was mistaken for support.

So this function answers one narrow question: can every offered source ID or bare
quote be found? Whether the citation supports the claim is the model's judgement,
and it is left to the model.
"""

from __future__ import annotations

import decision_contract as dc


def _normalise(text: str) -> str:
    """Whitespace-flattened, case-folded text for quote comparison.

    Sources and drafts are hard-wrapped, so a quote that spans a line break must
    still match. Flattening on both sides is what makes that work; it cannot
    manufacture a match for text that was never written.
    """
    return " ".join((text or "").split()).casefold()


def references_resolve(decision: dc.Decision, sources: dict[str, str]) -> bool:
    """Check every offered ID or bare quote — not whether evidence supports a claim.

    Returns False when any single reference cannot be located. A partial pass is not
    a pass: an answer that cites one real source and one invented one is exactly the
    shape a fabricated justification takes.
    """
    normalised_sources = [_normalise(text) for text in sources.values()]

    for issue in decision.issues:
        for reference in issue.sources:
            ref_id = reference.split(":", 1)[0].strip()
            if ref_id in sources:
                continue
            quote = _normalise(reference)
            if len(quote) > 8 and any(quote in text for text in normalised_sources):
                continue
            return False
    return True
