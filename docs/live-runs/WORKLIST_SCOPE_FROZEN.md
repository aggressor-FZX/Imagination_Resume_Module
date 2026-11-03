# Lexical work list — SCOPE FROZEN

**Status as of 2026-11-03: frozen.** No further naming changes, no coverage machinery, no
integration into the publication gate.

## What is frozen

`Imagination_Resume_Module/appended_clauses.py`, its fixture
(`tests/fixtures/tier_b/`), and its tests. The module:

1. locates a trailing participial clause after a comma, and
2. reports, per clause, which content stems are absent from the evidence cited **for that
   line**, plus how many of those stems occur elsewhere in the record.

It emits `semantic_status: "not_assessed"` on every finding and has **no importer in the
pipeline**. `git grep appended_clauses` outside its own file and tests returns nothing, so
merging the branch cannot change runtime behaviour.

## What was rejected, and why

| Proposal | Rejected because |
|---|---|
| Wire the work list into the publication gate | Control 2 demonstrates a **false** causal claim built entirely from source vocabulary producing an **empty** work list. A gate on this field would ship that sentence. Also control 3 shows a false positive would delete a correct restatement. |
| Add more coverage values / thresholds | The set-intersection already distinguishes what it can. `none`/`some`/`all` is the measurement; adding granularity would imply the field carries semantic weight it does not. |
| Rename the fields again | They now state exactly what is measured. Further renaming is churn. |
| Add a semantic classifier (embeddings, entailment model) | Out of scope for this pass, and it would need its own bounded real-model evidence before being trusted, per SPEC.v2 §2.2. |
| Suppress findings where coverage is `"all"` | Would hide control 1 and control 5 — the exact false positives the module must keep visible so a reviewer can see why a clause was flagged. |

## Why frozen rather than extended

The measurements that motivated freezing:

| Case | Work list | Is the clause sound? |
|---|---|---|
| Supported recombination (coordination) | empty | **Yes** |
| False recombination (same words, causal link) | **empty** | **No** |
| Entailed restatement ("responding more quickly") | reported, coverage `none` | **Yes** |
| Unentailed restatement ("reducing repeat visits") | reported, coverage `none` | **No** |
| Paraphrase, evidence present elsewhere | reported, coverage `all` | **Yes** |
| False claim, all words from source | empty | **No** |

Four of six rows are wrong about meaning in one direction or the other. No amount of
additional lexical machinery fixes that: the module measures word overlap, and the
distinctions that matter here are semantic. Extending it would produce more confident
answers to questions it cannot answer.

## What it is for, and the only sanctioned use

A **work list for a human or Tier-B reviewer**: these clauses were not attested by the
evidence cited for their line, so look at them. That is all.

Accepting a clean work list as evidence of support is the failure mode SPEC.v2 exists to
prevent, and control 2 is the test that keeps that visible.
