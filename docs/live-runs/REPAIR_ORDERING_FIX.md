# Repair ordering: the demonstrated recovery defect, fixed offline

No paid analysis was performed for this change. Every claim below is reproducible from
the protected captures on this machine.

## The defect

`source_correction.py` withheld on a missing source fact **before** entering the repair
path:

```python
if "missing" in incomplete:
    if flagged:
        pass          # <-- fell through to `return blocked("source_facts_missing")`
    ...
    return blocked("source_facts_missing")
```

So a review reporting both unsupported lines and omitted facts withheld immediately.
Repair — the one path that can remove unsupported content — was unreachable for exactly
the drafts that needed it.

Evidence, from the captures themselves (not recomputed):

| run | revision | flagged lines | missing items | `repair_calls` |
|---|---|---|---|---|
| 4 | `5f3c1f0` | 19 | 12 | **0** |
| 5 | `9057d966` | 13 | 6 | **0** |
| 6 | `fb75917` | 3 | 4 | **0** |
| 7 | `b855087` | 3 | 6 | **0** |

## The fix

When a review flags unsupported lines *and* reports missing facts, the flagged spans are
repaired first and the coverage verdict is taken on the repaired draft by the full
re-review that follows. Nothing is approved on the strength of the deferral:

* repair may only replace a **flagged** span (`_apply_edits`), so the deferral cannot add
  content;
* the re-review re-authenticates every span, every citation and every coverage verdict;
* a fact still missing after repair is withheld by the same branch it always was.

When there are **no** flagged lines the behaviour is unchanged: the bounded restoration
runs, then `source_facts_missing`.

## What this does not claim

* **The captures cannot validate the recovery.** All four contain exactly two exchanges
  (`source_inventory`, `review`) and stop where the old code stopped. Replaying new code
  against them proves the engine now *asks* for a repair; it cannot prove the answer is
  applied, because the answer was never recorded. That is what the controlled answers are
  for, and they are labelled as test-authored scaffolding throughout.
* **A moved fact is not caught lexically.** `_introduces_beyond` compares against the
  source *document*, so re-attributing Cascade's accomplishment to Harborline uses only
  sourced words and passes. What stops it is the reviewer declining to substantiate the
  line. Encoded as a limitation test, not a guarantee.

## Verification

| claim | how it is proved |
|---|---|
| original rejection pinned to its revision | `test_repair_ordering.py`: SHA-256 per capture + `recorded_stopping_point` read from the artifact, `repair_calls == 0` asserted |
| repair reachable on all four | same file: `repair_calls == 1`, full recording consumed, `session.unanswered` names the phase |
| recovery publishes | `test_repair_recovery_controlled.py`: approved, defects gone, facts intact |
| defect survives a repair ⇒ withheld | same file |
| five headings, no filler | same file + `test_route_repair_publish.py` |
| real route + real renderers | `test_route_repair_publish.py`: `/analyze` → `/resume/export`, `pdflatex` + `python-docx`, `pdftotext` read back |
| residual objections classified | `test_run7_objection_classification.py` |

Mutation-checked: restoring the old `return blocked(...)` before repair fails 17 tests.

Suite: **902 passed, 3 skipped**.

## Residual run-7 objections, classified

See `REMAINING_DEFECTS.md` for the full table. Summary:

A checker rejection is not itself proof of fabrication. Where the reviewer or a
deterministic check rejects a line, the class records *what kind* of failure it is —
which decides whether the remedy is removal, restoration or coverage.

| objection | primary class | secondary | remedy |
|---|---|---|---|
| `8,000 hours of supervised` → `hands-on … mastering core` | **lost_qualifier** | unsupported_addition (the trailing clause) | restore the qualifier AND remove the clause |
| skills block invents a category, admits `installation`/`maintenance` | **unsupported_addition** | — | remove |
| summary appends a proficiency list past the source summary's end | **unsupported_addition** | reviewer_error (the first clause was objected to wrongly) | remove the appended list |
| bold name concatenates degree + institution | **formatting** | — | emit degree and institution separately (caught by `unsupported_entities`) |
| degree field "Electrical Systems Technology" dropped | **omission** | — | carry the field |
| summary's first clause restates the source | **reviewer_error** | — | none |
| heading segments reported missing | **formatting** | — | emit the heading (empty is correct) |

Two corrections to the earlier key, both made because the earlier version called things
fabrication that asserted nothing false:

* **"Electrical Systems Technology dropped" is an omission, not a fabrication.** The
  draft drops a fact; it states nothing untrue. Calling it fabrication overstates the
  defect and invites a "stop asserting this" fix for a line whose real fault is missing
  coverage.
* **The education line's concatenated bold name is formatting, not fabrication.** Every
  word is sourced; only their combination into a bold *name* is wrong, and the
  consequence is that `unsupported_entities` cannot match it. The fix is layout.

The classification is required to disagree with the reviewer somewhere
(`test_the_classification_does_not_assume_the_reviewer_is_always_right`), because a
classification that never does is not a classification.

## Not done, deliberately

The writer defects above are **not** fixed here. They were demonstrated by run 7 and
belong to the writer prompt, whose last three changes each required a paid submission to
validate. Fixing them offline is possible and is the next unit of work; the ordering fix
is what makes one submission able to exercise repair, re-review and publication instead of
stopping at the gate.
