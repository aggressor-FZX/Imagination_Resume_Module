# Can a material-issues-only instruction eliminate invented objections?

**Proposal only. No calls made.** This sets out the candidate instruction, the
offline evidence for and against it, a frozen test, and its budget. Nothing here
changes production or the frozen fixtures.

## The problem being targeted

From the manual inspection, the binding failure is **not** missed defects. It is that
clean drafts are refused on non-material objections:

| | clean-draft issues | of which non-material |
|---|---|---|
| qwen | 7 | 7 |
| gpt-4o-mini | 10 | 10 |

17 of 17 objections on clean drafts are non-material. There are three recognisable
kinds:

1. **Equivalent-word objections** — "monitored"/"tracked", "produced"/"built",
   "with"/"against", "sent"/"reported", "entered"/"logged", "during 2024"/"in 2024".
   Both models request verbatim restoration. 15 of the 17.
2. **Invented timeline exclusivity** — qwen only, 2 instances.
3. **Self-voiding notes** — issues whose own `why` says "acceptable under synonym
   rules. No need to revise." Recorded as issues anyway, which forces `revise`.

## Offline test of the mechanism, before spending anything

The prerequisite question is whether *materiality is mechanically separable* on the
answers we already hold. If code cannot separate material from non-material, asking a
model to do it is a hope rather than a design.

Two candidate filters were tested against all 29 issues from the seven matched cases:

| Filter | Kept | Dropped | Verdict |
|---|---|---|---|
| Structural (resolves, or `omitted`, and has a correction) | **27 / 29** | 2 | Fails. Nearly everything passes. |
| Fact-token (correction changes a number, date, or content word beyond a synonym list) | **22 / 29** | 7 | Fails. Still keeps most equivalent-word objections, and it needs a hand-written synonym list, which is itself the judgement under test. |

**Neither filter is adequate.** In particular the fact-token filter keeps the
"tracked"/"monitored" objection on `unsupported_method` (gpt #2) and the
"completed"/"ran" objection on `cal_lab_faithful` (gpt #1), because the surrounding
clause differs. It is the *equivalence of the words*, not the presence of a changed
token, that decides materiality — and equivalence is exactly what an LLM is being
asked to judge.

This is the honest state: **the mechanism is plausible but unproven, and a simple
mechanical proxy does not work.** That is why a small frozen test is warranted rather
than a confident change.

## The candidate instruction

Short, additive, and expressible as a decision rule rather than a preference. Appended
to the shared contract, one block:

```
Only report MATERIAL issues. An issue is material if addressing it would change a
FACT the document asserts:
  - a figure, percentage, quantity, or date;
  - which employer, project or role a fact belongs to;
  - whether a source-backed achievement is present at all;
  - the method by which a sourced achievement was accomplished.

Do NOT report:
  - word choice that keeps the same meaning ("tracked"/"monitored",
    "built"/"produced", "with"/"against", "sent"/"reported", "during 2024"/"in 2024");
  - an expectation of verbatim source wording;
  - a timeline overlap the sources themselves state — overlapping roles are a fact,
    not a contradiction, and asserting otherwise is not a material issue.

If a passage reads faithfully, do not mention it. A document with no material issue
is `approve`. If you can only describe a difference without stating what fact would
change, it is not an issue.
```

Three deliberate design choices:

- **It defines materiality by consequence** ("what fact would change"), not by a word
  list, so it generalises past the seven cases.
- **It names the observed synonym pairs explicitly** as non-issues. This is fixture
  leakage of a kind — but the pairs are generic English paraphrase, not
  case-specific facts, and the instruction says they are *examples* of one rule.
- **It forbids the mechanism that produces self-voiding notes** ("if a passage reads
  faithfully, do not mention it"), which is the cheapest single fix available.

## Why this might still fail, stated in advance

1. **Both models already ignore an equivalent instruction.** The frozen text says
   "Do not require verbatim wording" and "Preserve faithful paraphrases". Both models
   violated it in 15 of 17 clean-draft objections. A more specific restatement may
   fare no better — this is the strongest reason to expect a null result.
2. **gpt-4o-mini's failures are inside its correction field, not its issue list.** The
   prompt-leak contamination and the verbatim restorations would survive a
   materiality rule, because the *finding* is often material while the *correction*
   is unusable.
3. **Suppressing issues is the exact direction that risks suppressing real
   omissions.** Two of the six defective cases hinge on an omission, and one
   (`cal_ops_missing`) was already mis-stated. A rule that discourages reporting is
   the wrong prior for that failure mode.

## Proposed frozen test

### Cases — 12, all already frozen, no new authoring

| Group | Cases | Why |
|---|---|---|
| Clean drafts (must **approve**) | `faithful_paraphrase`, `cal_ops_faithful`, `cal_lab_faithful` | The three that both models currently refuse. This is the primary endpoint. |
| Genuine method defects (must **revise**, correct clause) | `unsupported_method`, `cal_ops_unsupported`, `cal_lab_unsupported` | Guards against over-suppression on the most common real defect. |
| Omission / wrong-record (must **revise**, omission named) | `omitted_record`, `cal_lab_omitted`, `cal_ops_omitted`, `cal_lab_missing`, `wrong_record_attribution` | Guards the failure mode the instruction most plausibly harms. |

Twelve cases, drawn only from the two existing frozen fixtures. No new drafts, no new
sources, no fixture edits.

### Models and arms

- **Two**: `qwen/qwen3-coder` (incumbent) and `openai/gpt-4o-mini` (the comparison
  model). Two is the minimum that can distinguish "this instruction works" from "this
  instruction works on this model".
- **One arm only**: current frozen instructions + the materiality block appended.
  The existing run is the control; re-running the unmodified prompt would only
  re-measure what is already recorded.

### Predictions, fixed before the run

Recorded now so the result cannot be reinterpreted afterwards:

| Outcome | Reading |
|---|---|
| clean drafts approve ≥ 2/3 **and** defect recall stays 8/9 | the instruction works; proceed to a delivery test |
| clean drafts approve ≥ 2/3 **and** defect recall drops | over-suppression; the rule trades one failure for another |
| clean drafts still 0–1/3 | materiality is not instruction-addressable with these models; stop iterating the prompt |

### Budget

| | |
|---|---|
| Cases | 12 |
| Models | 2 |
| Arms | 1 |
| Logical review calls | **24** (12 × 2) |
| Attempts per call | 2 |
| **Hard request ceiling** | **48** |
| Expected spend | ≈ **$0.020** (13 of 24 cases at the observed $0.0005–$0.0012/call) |
| **Observed-cost ceiling** | **$0.10** |
| Re-reviews | **none** — this tests judgement only; adding the repair path adds a second variable |

All accounting via `RequestMeter`; the request ceiling is enforced before every
attempt including retries, and any response without usage is recorded as unknown
spend, not zero.

### What would not be done

- No production change of any kind.
- No automatic semantic veto, and no auto-dismissal of objections into approval — the
  instruction tells the reviewer, it does not overrule it in code.
- No third broad sweep; if this nulls, the answer is "materiality is not
  prompt-addressable here", and the next step is the contract work already identified
  (contextualise a timeline objection; require an omission to carry its owning
  record), not another instruction.
- No rerunning of individual failures.

### The success criterion this is measured against

> faithful documents approve, defective documents receive accurate correction
> requests, and corrected documents can reach release.

This test can only inform the first two. The third — corrected documents reaching
release — requires a re-review, and both valid re-reviews so far refused. That
remains untested and is not in scope here.
