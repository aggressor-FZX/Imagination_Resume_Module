# Offline replay of comparison-20261109-0719 — corrected findings

No paid runs were made for this document. Everything below comes from re-reading the
**retained 24 responses** with `experiment/replay.py`, which performs no network
calls.

## What the original table actually measured

The first comparison reported 21 of 24 attempts as `operational_failure`, dominated
by `envelope_not_json`. Replaying the retained bytes shows that label was wrong
about nearly all of them.

| Category | Count | What it means |
|---|---|---|
| Output-budget truncation | **11** | `max_tokens: 1200` cut the answer mid-string |
| Parser / contract defect | 10 | Correct format, rejected — see below |
| Expected revision (`revise`) | 2 | A **workflow**, not a false refusal |
| Semantic refuse | 1 | A reference failed validity |

**`envelope_not_json` was never a semantic result and, for 11 of 12 cases, was not
even a JSON syntax problem.** The `current` cells were given the legacy span
envelope contract, the model answered that contract correctly, and the response was
severed at the token limit:

```
current+advisory / faithful_paraphrase
  4,904 chars, ends: {"source_claim_id": "candidate-main.txt:45:45:51
```

Unterminated. The parser was the right one (`requested_format()` derives it from
the cell, and the payload begins `{"claims": [`) — the answer simply never finished.

One `current` answer did complete (3,525 chars, closing brace present) and parsed
correctly into a reference refusal. So "all current cells truncated" would have been
the same over-generalisation in the other direction; the replay reports 11 of 12.

## The 10 "parser defects" are mostly correct reviews

These were not parser faults. They were a **contract ambiguity** plus **presentation
differences**, both now fixed:

1. **The omission marker was written six ways.** `(omitted)`, `*(omitted)*`,
   `**omitted**`, `omitted`, `(omit)` were all rejected as `passage_not_in_draft` —
   a nonsensical complaint, since an omission is *defined* by the text being absent.
   Now canonicalised.

2. **An omission quoting SOURCE text was rejected.** The reviewer correctly
   identified a dropped source fact and quoted the source line (there is no draft
   text to quote). Accepted now **only** when the problem is `omitted` and the text
   really is in a source; an invented passage still fails both checks.

3. **Markdown emphasis on the decision and problem words** (`Decision: **revise**`,
   `problem: **unsupported**`) produced `problem_word_unknown:**unsupported**`. Now
   stripped when it wraps the whole value; interior emphasis is preserved so the
   passage still matches the draft.

4. **`revise` with `correction: (omit)`** was rejected as `revise_without_correction`.
   The fix for an omission is to ADD a sourced fact, not to rewrite a passage, so
   requiring a replacement made the commonest correct revision unrepresentable.

All four are presentation or contract-shape issues. **None weakens a guard**: an
invented passage, an unknown source, an unknown problem word, and an omission
quoting text in no source all still fail, and each has a test.

## The finding that invalidates the comparison

**Every held-out case supplies both source documents, but a draft that omits the
`Ridgeline Equipment Services` record present in `candidate-supplement.txt`.**

```python
for case in CASES:
    employers = re.findall(r"\*\*([A-Z][A-Za-z ]+)\*\* --", " ".join(case.sources.values()))
    missing  = [e for e in employers if e not in case.draft_before_trim]
```

| Case | Expectation | Draft omits |
|---|---|---|
| `faithful_paraphrase` | approve | Ridgeline Equipment Services |
| `supported_method` | approve | Ridgeline Equipment Services |
| `transferable_skill` | approve | Ridgeline Equipment Services |
| `unsupported_method` | revise | Ridgeline Equipment Services |
| `missing_fact` | revise | Ridgeline Equipment Services |
| `wrong_record_attribution` | revise | Ridgeline Equipment Services |

A faithful reviewer **must** flag a dropped employer record. The three cases
expecting `approve` were therefore unsatisfiable, and the model's refusal was
**correct** each time. The retained `faithful_paraphrase` answer says so explicitly:

> "The main source file does not mention Ridgeline Equipment Services or the
> Inventory Analyst role. While the supplement includes this employer and role, the
> draft omits the job title and dates from the supplement…"

So the comparison did not measure approval quality. It measured the harness's
missing omission handling against cases whose expectations were themselves wrong.
`tests/test_replay_regressions.py::test_every_approve_case_draft_covers_its_sources`
is the guard so this cannot recur silently.

## Metrics corrected

- **Operational failure ≠ semantic false refusal.** Truncation, transport failure
  and empty answers are operational.
- **`revise` is the contract working**, not a false refusal. Counting it as one
  penalises the reviewer for finding a real defect.
- **A reference refusal is a refusal**, not an operational failure — the previous
  `reference_invalid` label sat under `operational_failure`.
- **Parser acceptance is not semantic correctness.** Nothing in this replay asserts
  that an answer is *right*; only that a correct answer is no longer thrown away.
  Author expectations remain **provisional** until the blind audit.

## What must change before any rerun

1. **Raise `max_tokens`.** 1,200 cannot hold a full span envelope. This alone
   invalidated half the matrix.
2. **Fix or re-scope the held-out cases.** Either include the Ridgeline record in
   the approve-case drafts, or supply only the source document the draft was built
   from. Do not silently drop the record and keep the `approve` expectation.
3. **Freeze adapters, prompts, cases and settings**, as instructed, and retain
   pre/post-rule drafts. The rule is not yet demonstrated to change only its own
   behaviour, because the cells never reached a valid comparison.
4. Then run a small end-to-end contract check before repeating the four-way
   comparison within an agreed budget.

## Test status

`tests/test_replay_regressions.py`: 21 tests from the **actual** retained answers.
20 pass. The single failure is
`test_every_approve_case_draft_covers_its_sources`, which is the case-validity
defect above — deliberately left failing rather than weakened, so the invalid cases
cannot be quietly reused.

Full suite: run `python -m pytest tests/ -q -o addopts='' -p no:cacheprovider`.

## Scope

Experimental only. Production approval is untouched — this worktree
(`experiment/llm-led-approval`) imports `decision_contract` and never modifies the
production path. No winning configuration is declared.
