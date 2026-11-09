# Is `qwen/qwen3-coder` the bottleneck? Matched cross-model comparison

One additional reviewer, same frozen inputs. **Only the model differs** — identical
drafts, identical sources, identical instruction text (`4261e9cb…`), identical mode
(`imaginative`) and policy. No instruction, fixture or prompt change was made.

Data: `COMPARISON.md`, `cross-model-openai_gpt-4o-mini.json`, `plan.json`.

## Budget

| | qwen (from calibration) | gpt-4o-mini (this run) |
|---|---|---|
| Requests | 7 | **7 / 14** |
| Cost | (already reported) | **$0.002249** |
| Unmetered responses | 0 | **0** |
| Operational failures | 0 | **0** |

Combined added spend for this comparison: **$0.002249**.

## The methodological correction, applied

The exercise had been using "did the objection mention the needle?" as an accuracy
measure. That is a locator, not a grader. `experiment/detection_rubric.py` now grades
each defective case on the five things a correct detection must establish:

1. the relevant **fact**,
2. whether it is **in the source**, and the answer stated,
3. whether it is **in the draft**, and the answer stated,
4. which **employer owns** it,
5. what actually needs **correcting**.

Applied to the calibration run, this changes the headline from "12 of 12 named the
defect" to **10 of 12 automated-rubric agreements** — an automated count,
not independently verified accuracy — (`results/calibration-v1/DETECTION_RUBRIC.md`):

| case | result |
|---|---|
| `cal_lab_missing` | names the 24-hour fact and says it is missing, but **never says whose record owns it** |
| `cal_ops_missing` | states the 94% figure is "not present in the source file" — **false**; it is in `ops-supplement.txt`. Proposes no correction. |

Both were previously scored as detections because the needle was present. Neither
meets the five-part bar.

## Building the rubric exposed two parser defects

Both turned a CORRECT answer into a rejected one — the error class that most flatters
a reviewer's apparent accuracy:

| Defect | Effect | Fix |
|---|---|---|
| `**problem:** value` unparsed | one complete, correct detection (`missing_fact`) recorded as **zero issues**, scored `revise_without_issues` | field regex accepts emphasis inside or outside the colon |
| Passage with trailing emphasis (`… each quarter.*`) | a legitimate `omitted` issue lost its source-quote acceptance and was reported as a **miscitation** | trailing marker run stripped from the passage value |

Verified non-regressive: the retained matrix tallies are byte-identical afterwards
(`invented_objection` 6, `detected` 5, `not_detected` 1,
`detected_with_a_miscitation` 2).

## What the two models actually did

| | qwen/qwen3-coder | openai/gpt-4o-mini |
|---|---|---|
| usable answers | 5 / 7 | 5 / 7 |
| **claims source-stated concurrency is impossible** | **2** | **0** |
| objects to permitted synonymy | 2 | 2 |
| clean drafts approved | 1 / 3 | **0 / 3** |
| defective drafts fully detected (5-part) | 4 / 4 | **3 / 4** |

### The concurrency objection is model-specific

This is the answer to the question the comparison was run to settle.

qwen, on a draft whose two sources state `03/2021 - Present` and `09/2023 - Present`:

> "the source documents show concurrent employment (overlapping dates). This
> misrepresents the nature of the roles as sequential rather than concurrent."

and on the clean baseline:

> "indicating an overlap or inconsistency in timeline that is not clarified or
> supported … the concurrent 'Present' roles without clarification, which violates
> …"

**gpt-4o-mini raised this objection zero times across the same seven drafts.** It
objected to other things — synonymy, wording precision — but it did not invent an
exclusivity rule the sources do not contain.

Since the instructions already say, in the frozen text, that "overlapping dates are
a fact, not a contradiction" and "object only when the DRAFT asserts a timeline the
source denies", this behaviour surviving two instruction revisions is evidence about
the model, not about the wording.

### But the second model has its own, different failure

gpt-4o-mini **missed the omission on `omitted_record`** — the case where an entire
employer record is missing from the draft. Precisely: its raw decision word was
`approve`, it attached two issues to that approval, so the answer is **structurally
invalid** (`approve_carries_issues`), and its objection set never mentions `Ridgeline`
or `Inventory Analyst`. **No release is demonstrated** — this was a review-only run
with no release path, and the gate would reject an `approve_carries_issues` envelope.

The accurate description is: *raw approval word plus issues, invalid response,
omission missed*. It is a missed defect plus a malformed answer, not a demonstrated
release. Its synonymy objections are just as pedantic as qwen's:

> "The term 'monitored' does not accurately reflect the original wording 'tracked',
> which is a more precise description of the activity performed."
> "The term 'produced' does not accurately reflect the original wording 'built'."

## Conclusion

**The incumbent model is a substantial part of the bottleneck, and it is not the
whole of it.**

- The concurrency invention that drove half of qwen's false refusals **disappears
  with a different model** under identical instructions. That is strong evidence the
  behaviour is a property of that model, not of the prompt.
- Synonymy objections survive the change of model, so that one is **not** solved by
  swapping the reviewer — it is either an instruction problem that two revisions have
  not fixed, or a property of LLM review at this contract.
- A second model brings its own failure, but a different one: it **misses the
  omission** and returns an invalid answer (`approve` + issues). No release results.

So "swap the model" is not a fix on its own, and "iterate the prompt" is not a fix on
its own. The two failures are different and need different responses.

## Recommended next step

The evidence now supports two narrow, testable changes rather than another broad
prompt iteration:

1. **Make "no overlap objection without a draft-stated conflict" a checkable rule,
   not an instruction sentence.** Two instruction revisions did not stop qwen, so the
   contract should treat a concurrency claim on a source-backed overlap as an
   *invalid objection* and require the reviewer to cite the conflicting draft text.
   That converts a prompt hope into a mechanical test.
2. **Make an omission claim carry its owning record.** Both five-criterion rubric
   failures, and gpt-4o-mini's missed omission, are ownership failures. Requiring the
   owning employer in an omission issue is checkable offline and would have caught
   all three.

Neither requires a provider call to validate.

## Not established

- **Two models, one draft set, temperature 0, seven cases.** Two models is a
  comparison, not a survey; this does not establish where "LLM-led review generally"
  sits.
- **No independent human assessment** of either model's objections. The side-by-side
  is quoted material, not an adjudication.
- gpt-4o-mini was **cheaper** on these seven cases ($0.002249 vs $0.003369 for the
  same drafts), so cost is not an argument either way here; the comparison is about
  behaviour, not value. Per-case cost at production volume was not measured.
- No integrated delivery test. Production untouched.
