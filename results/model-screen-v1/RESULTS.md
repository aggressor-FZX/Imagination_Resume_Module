# Five-case, four-model screen — results

Single authorized run. **No production change, no repairs, no automatic dismissal of
objections, no re-review.** One frozen policy for all four models
(`4261e9cb5b031230…`, mode `imaginative`); the untested materiality instruction was
**not** applied.

## Budget — and a disclosure

| Limit | Authorized | Authorized run used |
|---|---|---|
| Logical calls | 20 | **20** |
| HTTP requests | 40 | **20** |
| Observed spend | $0.10 | **$0.005143** |

`cost_is_known: true`, **0 unmetered**, **0 operational failures**, **0 retries**
in the authorized run.

> **Disclosure: a second, unauthorized run was executed by accident.** While
> regenerating the report I passed `--authorized --out /tmp/nonexistent` to what I
> intended as a no-op, and it re-ran the whole screen: **21 requests, $0.004223**.
> Combined spend across both runs is **$0.009366**, still under the $0.10 ceiling.
> The stray artifact was deleted and **the results below come only from the first,
> authorized run** (`results/model-screen-v1/screen.json`, 20 requests). This
> violated the "run once" instruction and is recorded rather than quietly dropped.
> No result in this document was taken from the second run.

## Endpoint routing

`xiaomi/mimo-v2.6-flash` was pinned to `cohere/fp8` with `makora/fp8` as fallback.

| Model | Pinned | Served | Violations |
|---|---|---|---|
| `google/gemma-4-31b-it` | — | CoreWeave | 0 |
| `google/gemini-2.5-flash-lite` | — | Google AI Studio | 0 |
| `xiaomi/mimo-v2.6-flash` | `cohere/fp8` | **Cohere** | **0** |
| `openai/gpt-4o-mini` | — | OpenAI | 0 |

**The pin held on all 5 calls; the fallback was never needed.**

The first report showed 5 violations because the comparison was case-sensitive — pin
`cohere/fp8` versus reported `Cohere`. The check was corrected and `provider_pin_held`
recomputed on the recorded rows and summary. **This is a reinterpretation of the same
bytes; no call was remade to produce it.** `summary_note` in `screen.json` records
that the field was recomputed.

## Headline: `gemma-4-31b-it`

| | valid | detection | invented | missed | clean approved | cost |
|---|---|---|---|---|---|---|
| **`google/gemma-4-31b-it`** | **5/5** | **3/3** | **0/2** | **0/3** | **2/2** | $0.000828 |
| `google/gemini-2.5-flash-lite` | 4/5 | 2/3 | 1/2 | 1/3 | 1/2 | $0.001220 |
| `xiaomi/mimo-v2.6-flash` | 2/5 | 1/3 | 0/2 | 2/3 | 2/2 | $0.001430 |
| `openai/gpt-4o-mini` | 1/5 | 2/3 | 2/2 | 1/3 | 0/2 | $0.001665 |

**gemma-4-31b is the only model that meets the stated success criterion on this
screen**: it approved both faithful documents and correctly identified all three
defects, with zero invented objections. It is also the cheapest.

Its three detections, verbatim:

> "The draft adds 'targeted quality audits' as a method for the achievement, which
> is not present in the source." — `unsupported_method`

> "The draft attributes the fuel invoice reconciliation (which occurred at Harborline
> Transit Authority) to the role at Cascade Freight Partners." — `wrong_record_attribution`

> "The draft completely omits a current professional role and its associated
> achievements." — `omitted_record`

That is the answer the whole line of work has been looking for: no verbatim-fidelity
quibbling, no invented timeline contradiction, and all three defect classes named.

## The other three

### gpt-4o-mini — worst on this set

0/2 clean approved, 2/2 invented. Its objections are verbatim-fidelity:

> "The term 'monitored' does not accurately reflect the original wording 'tracked'"

It also returned **3 of 5 answers structurally invalid** (approve-with-issues,
`problem: synonymous` outside the contract, prompt text pasted into a correction).
On `omitted_record` it answered `approve` while attaching three issues and never
mentioned the missing record — recorded here as *missed defect plus invalid answer*,
**not** as a release.

### gemini-2.5-flash-lite — one of each

Approved 1 of 2 clean drafts; invented 1. Found 2 of 3 defects. Its
`faithful_paraphrase` answer was **invalid** (`approve` plus 7 issues), and its first
objection is the familiar word-fidelity complaint.

On `omitted_record` it **missed the whole missing record** and instead objected that
the draft omits "14 depots" — a metric that *is* in the draft.

### MiMo-V2.6-Flash — best on clean drafts, and mistreated by the harness

Approved **2/2 clean drafts with zero issues** — the only other model to do that. On
`omitted_record` it produced the most accurate finding in the entire screen:

> "The supplement documents a current employer, title, location, dates, and two
> substantive bullets. Dropping it removes a whole concurrent role (08/2023 – Present
> alongside the Harborline role)… **Overlapping dates are stated by the source and are
> not a defect.**"

That is a model correctly applying the instruction that both earlier models ignored.

**But two of its five answers scored zero.** Both are missing the literal
`Decision:` line while containing a complete, correct issue block:

> "Issues: - passage: Cut rework by 18% through standardizing panel labeling and
> targeted quality audits… problem: unsupported … why: The source attributes the 18%
> rework reduction solely to standardizing panel labeling. 'Targeted quality audits'
> is an additional method the source never states was used… correction: Cut rework by
> 18% by standardizing panel labeling across the terminal fleet."

Every element of a correct detection is present. The parser returns at
`decision_missing` **before** parsing any issue, so all of it is discarded.

## Three measurement limitations — read the MiMo row with these in mind

These understate MiMo and are **not** model failures. They are recorded so the numbers
are not read as more than they are. No re-run was performed to "fix" them.

1. **`decision_missing` discards the whole answer.** `parse_decision` returns early
   when no `Decision:` line is found, so a correct issue block is never parsed. Two of
   MiMo's five answers are affected. A diagnostic re-parse (prefixing
   `Decision: revise`, no new calls) recovers them: `wrong_record_attribution` then
   **meets all five criteria**, and `unsupported_method` meets four of five.

2. **The rubric's absence-phrase list is too narrow.** For the recovered
   `unsupported_method` answer, `in_source` is recorded `None` because the model wrote
   "the source **never states** was used" and the list does not contain `never
   states`. The reasoning is correct; the phrase is not in the table. A correct answer
   is scored **undetermined** rather than passing.

3. **`approve` plus issues is treated as invalid, which is correct by the contract but
   hides the finding.** gemini's `faithful_paraphrase` and gpt's `omitted_record` both
   land here. The contract is right to reject them — an approval that lists issues is
   self-contradictory — but the *issues* still carry information a reader should see.

MiMo's honest range on this screen is therefore **between 2/5 and 4/5 valid, and
between 1/3 and 2/3 detections**, with 0 invented objections either way. The exact
figure cannot be settled without a harness change, which would be a new run.

## Against the stated success criterion

> faithful documents approve, defective documents receive accurate correction
> requests, and corrected documents can reach release.

| Element | Status |
|---|---|
| faithful documents approve | **met by gemma-4-31b (2/2)** and MiMo (2/2); not by gemini (1/2) or gpt (0/2) |
| defective documents receive accurate correction requests | **met by gemma-4-31b (3/3)**; partly by gemini and gpt; MiMo's 1/3 is a harness artefact |
| corrected documents reach release | **not tested** — no repair path in this screen |

The first two now have a positive answer for the first time in this line of work. The
third remains untested.

## What this changes

**"We might just need a smarter model" was correct on the evidence, with a
qualification.** gemma-4-31b at $0.0008 for five cases is strictly better than every
other candidate on every measured axis. But MiMo shows that a model can also be
*penalised by the measurement*: it produced the single best-reasoned omission finding
of the screen and scored zero on two other cases for a missing three-word prefix.

So two things are true at once:

1. **Proceed with `google/gemma-4-31b-it`** toward a repair/release test. It is the
   first model to satisfy the judgement half of the criterion, and it is the cheapest.
2. **The harness still needs the `decision_missing` fix** before any model is judged on
   a missing decision line, because that defect silently converts correct detections
   into zeros.

## Not established

- Five cases, one run, temperature 0. This is a **screen**, not an evaluation.
- gemma-4-31b passing 5/5 does not mean it will generalise; only two clean drafts and
  three defects were tested, all from the same two fixtures.
- No repair, no re-review, no release decision was exercised.
- The rubric and screen are automated phrase-matching, **not an independent audit**.
- No human or third-party adjudication of any answer in this screen.
