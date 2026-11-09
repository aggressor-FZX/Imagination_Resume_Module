# Review of the seven cases — are the objections correct, and does the gate behave?

Offline analysis of the 28 retained attempts. **No provider calls.** Packet per case
is in `cases/`; expected labels are in `EXPECTED_LABELS.md`, kept separate on
purpose. Remote commit reviewed: `0378cb6`.

---

## 1. The gate honours all three decisions

| Decision | Behaviour | Evidence |
|---|---|---|
| `approve` | releases immediately on a valid initial approval | no retained answer contains one, so the path is tested only by unit tests |
| `revise` | **never** releases on the first `revise` | 14 prose first answers all said `revise`; 0 released |
| `cannot_assess` | withholds | not exercised by any retained answer |

`revised+advisory` released 5 in the original run; under the binding gate **all 5
withhold**. That is the intended effect: the previous releases were a harness
policy override, not model approvals.

**Released: 0 of 28.** Two cells reached an applicable repair
(`unsupported_method`, both arms) and both re-reviews still said `revise`, so both
withheld. This is fail-closed and correct — but it means the gate has never been
observed releasing on a repaired document, only in unit tests.

## 2. Format mismatch is not a defect

13 of 14 legacy cells answer the JSON span envelope. The gate is a prose contract,
so it cannot read them. That is reported as
`format_mismatch_json_answer_to_prose_gate` — **not** a model failure and not a
refusal. The 14th legacy cell is the truncated one with no retained response, and
is reported as unverifiable.

## 3. Are the objections correct? Mostly not.

Ground truth verified mechanically against both sources:

| Case | Real mutation | Reviewer named it? |
|---|---|---|
| `unsupported_method` | `targeted quality audits` in neither source | **YES** — precise |
| `wrong_record_attribution` | 1,180 invoice moved under Cascade | **YES** — precise |
| `missing_fact` | 1,180 invoice bullet deleted | **NO** — never mentions the invoice or the deletion |
| `omitted_record` | whole Ridgeline record missing from draft | **NO** — asserted the record was *in* the draft and should be removed |

### The two failures are serious

**`missing_fact`** — the reviewer produced three objections, none about the deleted
bullet: it argued verb tense ("Maintained" vs "Maintain") and the preposition
"with" vs "against". The `revise` verdict is right by accident; the reasoning is
wrong and the correction would rewrite wording rather than restore the lost fact.

**`omitted_record`** — the reviewer quoted `**Ridgeline Equipment Services** …` as
though it were **present** in the draft and demanded `correction: (omit)`. Verified:
neither "Ridgeline Equipment Services" nor "Inventory Analyst" appears anywhere in
that draft. It objected to the *opposite* of the real defect and proposed deleting
a record that was already absent — a no-op. Two earlier rows in the same matrix did
correctly identify the omission, so the capability exists but is unreliable.

### The three approve cases are judged on spurious grounds

All three `approve` cases drew `revise`, and the objections are artefacts:

- `faithful_paraphrase` — objected that Ridgeline's `08/2023 - Present` **overlaps**
  Harborline's `04/2019 - Present`, calling concurrent employment "impossible".
  Both sources state both roles as Present. This is an invented contradiction, not
  a defect.
- `supported_method` — objected that "Reduced rework by 18% **through** standardizing
  panel labeling" adds a "causal mechanism" the source lacks. The source says
  "Cut rework by 18% **by** standardizing panel labeling". Same mechanism.
- `transferable_skill` — objected on the same invented overlap, though it did also
  correctly name `demonstrating reconciliation`, which is the real clause.

So on faithful text the reviewer is not conservative — it is **inventive**, and the
common source of invention is a timeline conflict it assumes rather than checks.

## 4. What this means

- **The gate is sound; the reviewer is not.** Fail-closed release plus a reviewer
  that objects to real defects correctly in 2 of 4 cases gives safe-but-unusable
  behaviour: it withholds faithful documents and withholds defective ones, so a
  withhold no longer distinguishes them.
- **A withhold must not be read as correct detection.** Before this review, 7
  withholds on `revised+enabled` could easily have been read as 7 correct catches.
  The truth is: 2 precise detections, 2 missed targets, 3 spurious objections.
- **The shared timeline assumption dominates.** Both sources mark two roles
  `Present`. That single artefact drives most of the false objections, and it is a
  property of the fixture as much as the reviewer — a human reader would ask the
  candidate rather than refuse.
- **Two of the four defect cases pass for the wrong reason**, so a rise in
  "`revise` on defect cases" is not evidence the reviewer improved.

## 5. Not established

- No independent human or model audit has judged these documents; the blinded
  packet exists (`results/binding-audit-v3/blind/`, 28 entries) but no auditor
  returned a review in this session.
- One model, temperature 0, seven cases. Nothing here estimates field reliability.
- The gate has never released a repaired document outside unit tests.
- No claim that any final document is faithful — that needs the audit.

## 6. Recommended next step

The binding gate does not need redesign. The reviewer needs one targeted change:
**treat overlapping `Present` roles as a fact to report, not a contradiction to
refuse**, and require an omission claim to be grounded in text absent from the
draft before proposing `(omit)`. Both are testable offline against these retained
answers without new calls.
