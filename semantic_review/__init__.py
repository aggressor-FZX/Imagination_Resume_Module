"""Semantic review: the production home for LLM-led approval, repair and release.

Why this package exists
-----------------------
The decision/repair/re-review behaviour was developed in `experiment/` because it
was being measured. It is now being integrated into the live service, and production
must not import experiment runners, authorization ledgers or report generators.

Only two things were actually needed at runtime:

    decision_contract.py        the prose contract + parser (already at top level)
    experiment/repair.py        the scoped repair
    experiment/binding_candidate.references_resolve

So `repair` is moved here verbatim, and `references_resolve` is re-implemented here
rather than imported from the experiment, breaking the last edge. What stays in
`experiment/` is the measurement apparatus: run harnesses, ledgers, packets, report
renderers.

The division of responsibility this package preserves
----------------------------------------------------
    MODEL   decides faithfulness, permitted paraphrase, material omission,
            attribution, under the selected mode.
    CODE    enforces authorization, reference validity, document identity,
            required structure, receipt integrity, execution limits.

A `revise` is never permission to release. Semantic regex may observe but cannot
override a valid approval or remove supported meaning.
"""

from semantic_review.repair import apply_repair
from semantic_review.references import references_resolve

__all__ = ["apply_repair", "references_resolve"]
