# Findings — release-override repair accepted as a checkpoint

Experimental checkpoint, superseding `FINDINGS-v3-preliminary.md`. **No provider
calls in this document.** Remote base: `ad4132e358edbc34fe1be6f3ad0fa0b0edb6791d`
on `experiment/llm-led-approval`. Production untouched (empty diff for
`source_correction.py`, `correction_provider.py`, `app.py`, `publication_gate.py`,
`stages/`, `models.py`, `orchestrator.py`, `imaginator_flow.py` vs `master`).

The gate was **not** redesigned and the matrix was **not** rerun. Everything below is
either an offline replay of the 28 retained attempts or new scaffolding around them.

---

## 1. The denominator (item 2)

The candidate's success/failure denominator is **14 prose cells**, not 28.

| Kind | Count | Treatment |
|---|---|---|
| contract-compatible prose, response retained | **14** | in the denominator |
| legacy JSON span envelope | 13 | excluded — answers a *different* contract |
| truncated, no response at all | 1 | excluded — operational failure |

Both excluded groups remain in the packet with their own reason, so nothing is
dropped. A legacy JSON answer is a correct answer to the envelope contract; calling
it a model failure because the prose gate cannot read it would repeat the original
error in a new place.

## 2. Initial judgment (axis 1)

All 14 prose cells returned `revise`. **Agreement now requires a valid exact match**,
which changes the numbers materially from the preliminary packet:

| Agreement class | Count |
|---|---|
| `agree` (valid answer, exact decision match) | **4** |
| `disagree` (valid answer, wrong decision) | **6** |
| `invalid_response` (decision word present, envelope rejected) | **4** |
| `cannot_assess` | 0 |
| `operational_failure` | 0 |

**Success/failure rate denominator: 10** — not 14.

### Correction to the preliminary figure

The preliminary packet reported **8 agree / 6 disagree** over 14 cells. That was
wrong, for two reasons the current code fixes:

1. It compared booleans — `(decision == "approve") == (expected == "approve")` — so
   a `cannot_assess` answer on a `revise` case scored as agreement, because both
   sides evaluate to `False`. A withhold is not a revision request.
2. It counted an answer as a verdict even when the parser had **rejected the
   envelope**. Four cells return the word `revise` while carrying
   `passage_not_in_draft` or `revise_without_actionable_issue`; those are not usable
   verdicts and are now `invalid_response`.

Both errors inflated the reviewer's apparent accuracy. The honest reading is
**4 of the 10 usable answers agree**, and 4 answers were not usable at all.

Agreement remains one mechanical fact, not evidence an objection was correct — §5.

## 3. Edit applicability — a property of the scaffold (axis 2)

| Outcome | Count |
|---|---|
| proposal drawn | 2 |
| `ambiguous_or_multiple_issues` | 8 |
| `not_attempted` (review unverifiable) | 4 |

**This axis describes the narrow repair scaffold, not the reviewer.**

### Scaffold limitations, kept separate from reviewer errors (item 4)

1. **One issue only.** Any cell raising two or more issues is refused. Eight cells
   fall here. The scaffold cannot represent "fix these two bullets", so a reviewer
   that correctly identified a defect *and* a second candidate defect is refused for
   being thorough.
2. **Single-line, single-bullet only.** Multi-line corrections, commentary appended to
   a correction, and any edit touching a heading or a date line are refused.
3. **Replacement must be an exact bullet of the same employer** in the source. A
   correct fix phrased differently is refused
   (`replacement_not_exact_owned_source_bullet`).
4. **Omissions are never auto-applied.** `problem: omitted` is refused as
   `omission_requires_explicit_destination`. Deliberate: the harness does not compose
   résumé content.
5. **`(omit)` with commentary is refused.** `"(omit)" (Note: …)` is not an omit
   marker; applying it as replacement text would insert the note into the document.

Consequence: **a withheld document does not implicate the reviewer's judgement.**
Twelve of fourteen cells never reached a proposal, and in eight of those the reason
was scaffold narrowness.

## 4. Approval of the exact corrected document (axis 3)

| Re-review identity | Count | Reading |
|---|---|---|
| `same_document` | 2 | the retained re-review saw exactly the gate's proposal |
| `different_document` | **5** | the retained re-review answered about a **different** text |
| no re-review | 7 | the gate never drew a proposal |

**Item 7 is enforced in code.** A retained re-review of a different document is
reported as **no observation** — `None`, never an approval and never a refusal. Five
cells would otherwise have been scored on an answer about the wrong text.

**Approval now counts only when the re-review is valid AND says `approve`** (item 2).
Reading it off gate release would have attributed a scaffold property to the
reviewer, since release also requires the proposal to have been applicable.

| Valid re-review outcome | Count |
|---|---|
| `approved` | **0** |
| `not_approved` (valid, said `revise`) | 1 |
| `invalid_re_review` (right document, unusable answer) | 1 |

The corrected-document approval axis has **no positive observation** in the retained
data. The one cell that refused (`unsupported_method`, enabled arm) did so with a
usable verdict; the other (`unsupported_method`, advisory arm) answered the right
document but its envelope was rejected, which is neither an approval nor a refusal.

## 5. Are the objections correct? (source-linked, `JUDGMENTS.md`)

Classified mechanically against each declared defect, using needles whose location is
verified in the fixture (`assert_needles` fails loudly if a fixture moves under them):

| Verdict | Count |
|---|---|
| `detected` | 5 |
| `detected_with_a_miscitation` | 2 |
| `not_detected` | 1 |
| `invented_objection` | 6 |

**This corrects the earlier "2 of 4" claim**, which was true only of the
`revised+advisory` arm:

| Case | advisory | enabled |
|---|---|---|
| `unsupported_method` | detected | detected |
| `wrong_record_attribution` | detected | detected |
| `missing_fact` | **not_detected** | detected |
| `omitted_record` | detected + miscitation | detected + miscitation |

So the reviewer is **not consistently blind to omissions** — it found the missing
figure in one arm and not the other.

What it does consistently:
- **refuses every faithful document** (3 of 3 approve cases, both arms) on objections
  corresponding to no defect;
- **miscites absent text as present** in both `omitted_record` cells: quotes
  `**Ridgeline Equipment Services** …` as though it were in the draft while a separate
  objection correctly reports it missing. Verified mechanically: neither "Ridgeline"
  nor "Inventory Analyst" occurs anywhere in that draft.

The six `invented_objection` verdicts on three faithful documents across two arms are
the strongest finding here, and the inventions are nameable:
- a `Present`/`Present` overlap declared "impossible", when the two sources state it
  themselves (`Fuel Reporting Analyst | 04/2019 - Present` in the main document;
  `Inventory Analyst | 08/2023 - Present` in the supplement);
- "through" vs "by" declared an added causal mechanism (the source says "by");
- verb tense `Maintained`/`Maintain` and preposition `with`/`against` raised on a
  document with no declared defect at all.

## 6. Release (axis 4)

**0 of 14 released.** Reasons: 8 `revision_unresolved`, 2 `re_review_not_approved`,
4 `initial_review_unverifiable`. Zero undetermined.

The gate honours all three decisions:
- **approve** releases immediately — proven by positive offline control (§7), not by
  the retained data, which contains no approval;
- **revise** never releases on the first answer; release requires an applicable
  repair *and* a new valid `approve` of the corrected text;
- **cannot_assess** withholds and spends no repair call.

## 7. Positive control added (item 3)

`tests/test_gate_controls.py` — 12 tests. The previously missing one:

`test_a_valid_approve_releases`, parametrised over all three approve cases: asserts
release, reason `initial_llm_approve`, no repair attempted, and **exactly one provider
call**.

Paired with `revise` (no release without a new approval; release only after approval of
the corrected text) and `cannot_assess` (withholds, no repair call), plus a test that
the three produce **three distinct release reasons**. Without the positive control,
every "fail-closed" statement rested on the absence of releases — which a gate that
never releases at all would also satisfy.

## 8. Reviewer instructions refined (item 5)

Four general evidence principles added to the shared contract (mode-independent):

1. **Overlapping dates are a fact, not a contradiction** — object only when the draft
   asserts a timeline the source denies.
2. **Synonymous wording is permitted** — by/through, cut/reduced, tracked/monitored,
   built/produced; object only when the change alters what, how, or how much.
3. **A materially missing source fact must be identified**, with its location cited.
4. **A fact must stay under the record that stated it.**

These are principles, not fixture knowledge: they name no employer, figure or case. A
test asserts they do **not** create a licence to ignore planted dates
(`test_forging_a_different_timeline_is_still_rejected_by_the_contract`).

`contract_fingerprint()` hashes the exact prompt text so a frozen run cannot drift
from the instructions recorded with it.

## 9. Bounded calibration prepared and frozen (item 6)

`experiment/calibration_cases.py` + `experiment/run_calibration.py`.

- **17 cases**: the 7 existing + **10 unfamiliar paired controls** in two new domains
  (water-quality laboratory, cold-chain logistics), sharing no content line with the
  original fixture.
- **Faithful and defective in every pair.** 2 faithful, 8 defective across 4 defect
  kinds; each defective draft is one declared transformation from its baseline —
  verified as exactly **1 changed region**, which is the right test (a changed-line
  count would misread a wrapped bullet as several changes).
- **Frozen before the calls.** `results/calibration-v1/frozen-instructions.json`:
  mode `imaginative` (matching all 28 retained cells), instructions sha256
  `4261e9cb5b031230aa2a3adf3f818346eab02318dd229b430eca31f166d17e7b`, fixture
  fingerprint `b4b73e68b4d69b394216ccf5b68be6c5239719d31809b4fa9f5c394eff25a224`.
- **Bounded, and the ceiling is now provable.** 27 **logical** calls (17 first + at
  most 10 re-reviews) × 3 attempts = a **hard ceiling of 81 HTTP requests**, checked
  before *every* attempt including retries. The previous "$1.00 / 27 calls" statement
  counted one unit per logical call and was therefore not a limit at all — a retried
  call could send three requests while the budget advanced once.

### Accounting that does not overclaim (item 3)

| Ceiling | Is it hard? |
|---|---|
| 81 HTTP requests | **yes** — checked before each attempt, retries included |
| $1.00 | **no** — enforced against *observed* cost only |

`RequestMeter` records every attempt, including 429/5xx and transport errors, and
sets `cost_is_known = False` if any response omitted usage. It then reports
`unmetered_caveat` rather than asserting a spend limit it cannot prove. A missing
usage object is treated as **unknown cost, not zero** — the original `$0.00` bug.

- **Re-review guard**: the re-review prompt is built from the proposal and the
  recovered document is asserted equal to it; a mismatch raises.

**No calls have been made.** `--dry-run` validated the plan only.

The prompts, cases and budget plan were re-frozen **after** the three behavioural
fixes, and their identity is unchanged by that work: instructions sha256
`4261e9cb…` and fixture `b4b73e68…` are the same values as before. Only the
accounting and classification code changed, which is what "freeze the runner" has to
mean if the freeze is to be checkable.

### Offline regressions added before the freeze (item 4)

`tests/test_calibration_accounting.py` — 23 tests, no provider calls:

| Behaviour | Regression | The bug it would catch |
|---|---|---|
| exact agreement | `test_semantic_agreement_requires_a_valid_exact_match` (9 parametrised cases) | boolean comparison scoring `cannot_assess` as agreement |
| | `test_the_rate_denominator_excludes_non_verdicts` | counting invalid/operational cells in the rate |
| corrected-doc approval | `test_approval_requires_valid_and_approve_not_merely_release` | reading approval off gate release |
| | `test_an_invalid_re_review_is_reported_separately_from_a_refusal` | collapsing unreadable into refusal |
| request accounting | `test_a_retried_call_reports_every_request_it_sent` | one unit per logical call |
| | `test_a_transport_error_is_metered_as_an_attempt` | timeouts vanishing from the count |
| | `test_missing_usage_is_not_zero_cost` | treating unknown spend as free |
| | `test_the_meter_refuses_to_send_beyond_the_ceiling` | a ceiling that does not bind |

One of these tests found a real bug while being written: a transport error left
`cost_is_known = True`, because only the `ok_no_usage` status incremented the
unmetered counter. Fixed — any attempt without reported usage now makes spend
unknown.

## 10. What this checkpoint does NOT establish

- **No positive release on a corrected document.** 0 of 14, and the two valid
  re-reviews both said `revise`. The approve→release path is proven only by unit
  control.
- **No independent audit.** `results/binding-audit-v3/blind/` holds 28 verified
  entries; no auditor returned a judgement. A packet is not an audit.
- **No calibration result.** The fixture is frozen and bounded; running it is the
  next step and needs a spend decision.
- **One model, temperature 0, seven familiar cases.** Nothing here estimates field
  reliability.
- **Detection is not correction.** A `detected` verdict means the objection *named*
  the defect; it says nothing about whether the repair is right. That is axis 3, which
  has no positive observation.

## 11. Recommended next step

1. Decide the calibration budget (27 calls / $1.00 ceiling already frozen) and run
   `python -m experiment.run_calibration` once, unchanged.
2. Get a real blind audit of `results/binding-audit-v3/blind/`; nothing here
   substitutes for it.
3. Only then consider widening the repair scaffold (multi-issue, exact-bullet
   relaxation), because §3 shows scaffold narrowness rather than reviewer error causes
   most withholds.
