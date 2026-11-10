# Gemma repair-and-release test — proposal for authorization

**No calls made.** Inference is stopped. Neither screen will be repeated.

Status of Gemma: **screening winner, not a production-qualified replacement.** It is
the only model that approved both faithful drafts and named all three defects on the
5-case screen, at the lowest cost. That establishes the *judgement* half of the
success criterion on five cases. It establishes nothing about repair, re-review, or
release.

## What this test adds

The one element of the success criterion never exercised: **corrected documents can
reach release.** No run in this series has ever released a corrected document.

The repair path already exists (`experiment/binding_candidate.py::safe_repair` →
re-review of the exact proposal). What has never happened is a model that (a) finds a
defect, (b) produces a repair the scaffold can apply, and (c) approves the corrected
text.

## Cases — 5 drafts, all already frozen, no new authoring

| # | Draft | Role | Required outcome |
|---|---|---|---|
| 1 | `faithful_paraphrase` | **untouched paired control** | approve on first review; **no edit may be applied** |
| 2 | `unsupported_method` | unsupported-method correction | revise → remove the invented clause → approve corrected |
| 3 | `wrong_record_attribution` | wrong-record correction | revise → remove the merged clause → approve corrected |
| 4 | `omitted_record` | source-backed record restoration | revise → name the missing record → **scaffold refuses auto-insert** |
| 5 | `cal_ops_faithful` | **second untouched control (unfamiliar)** | approve; no edit |

Two controls, because one is not a control: `faithful_paraphrase` is the familiar
clean draft and `cal_ops_faithful` the unfamiliar one. If an edit is applied to either,
the release result is void regardless of the defect cases.

### Why case 4 is expected to fail the repair, and why that is the point

`safe_repair` refuses `problem: omitted` (`omission_requires_explicit_destination`).
The harness must not compose résumé content. So the predicted outcome for the
restoration case is:

- the model must **name the missing record and its source** — testable;
- the scaffold must **refuse to auto-insert** — also testable, and a pass;
- **no release** for that document — correct, not a failure.

Recording that as a successful refusal is different from recording it as a repair.

## Protocol per case

1. **First review** of the frozen draft. If `approve` and valid → release, no edit.
2. If `revise` and valid → `safe_repair` proposes a corrected document, or refuses.
3. If a proposal exists, **a NEW review is sent of that exact document.** The re-review
   prompt is built from the proposal and the recovered document is asserted equal to
   it, or the run raises. No re-review of a different text is ever counted.
4. Release only if that re-review is valid and says `approve`.

**Every changed document gets a new review of that exact text.** No approval is
carried over from the original draft.

## What is recorded

Per case: the first decision and its validity; every issue with passage, problem,
source references and correction; the repair status and unified diff of the applied
edit (or the refusal reason); the **exact corrected text**; the re-review's decision,
validity and problems; the release decision and the rule that produced it; **the
actual provider reported by each response**; every attempt with its status; and cost.

Untouched controls additionally record **an edit count that must be zero**.

## Model, provider and policy

- **Model:** `google/gemma-4-31b-it` only. This is a repair test, not a model sweep.
- **Provider:** **no pin.** Gemma is served by multiple endpoints; the screen saw
  `CoreWeave`. A pin is not requested here because the test is about the repair path,
  not endpoint identity — but **the reported provider is recorded per call**, so a
  mid-run provider change is visible rather than assumed away.
- **Policy:** the same frozen contract as every run in this series
  (`4261e9cb5b031230…`, mode `imaginative`). No instruction change, no materiality
  block — those remain untested proposals.
- **Temperature 0**, `max_tokens` 8192.

## Limits — precise, for authorization

| Limit | Value | Basis |
|---|---|---|
| Cases | **5** | 4 defect classes + 2 controls, one control shared |
| First reviews | **5** | one per case |
| Re-reviews | **at most 5** | only where a proposal is drawn; the restoration case is expected to draw none |
| **Logical calls** | **10 maximum** | 5 first + 5 re-review |
| Attempts per call | **2** | one retry on transport failure only |
| **Hard request ceiling** | **20** | 10 × 2, checked before every attempt including retries |
| **Observed-cost stop** | **$0.05** | Gemma screen cost was $0.000828 for 5 calls → ~$0.0017 for 10 |
| Projected spend | **≈ $0.0017** | 3% of the stopping point |
| Re-runs | **0** | one-shot; the authorization guard enforces it |

The `$0.05` stopping point is enforced against **observed** cost and is **not** a proof
of maximum spend. Any response without usage is recorded as unknown, and the report
will say so rather than quoting a ceiling it cannot substantiate.

The request ceiling is **hard and cumulative**: it is checked before each attempt, and
the authorization token records cumulative requests across any invocation, so a second
invocation cannot quietly reach 40 while appearing compliant.

## Authorization tokens

Execution requires a token, not a flag:

```
python -m experiment.model_screen_repair --authorized gemma-repair-2026-11-09
```

A repeated invocation with the same token is refused **before a provider client is
constructed**. Regenerating the report afterwards uses a separate, network-free path:

```
python -m experiment.model_screen_repair --render results/gemma-repair-v1/repair.json
```

`--render` takes no token, imports no provider code, and cannot dispatch.

## Predictions, fixed before the run

| Case | Predicted | If wrong |
|---|---|---|
| `faithful_paraphrase` | approve, 0 edits | any edit invalidates the run |
| `cal_ops_faithful` | approve, 0 edits | any edit invalidates the run |
| `unsupported_method` | repair drawn and approved → **release** | if refused, the clause fix is reachable but not approvable |
| `wrong_record_attribution` | repair drawn and approved → **release** | as above |
| `omitted_record` | repair **refused by scaffold**, record named | if a repair is drawn, the scaffold has widened silently |

**Success for this test:** ≥1 corrected document reaches release, both controls stay
untouched, and every release is backed by a valid re-review of that exact text.

**Failure that is still informative:** repairs are drawn but never approved → the
scaffold and the reviewer disagree about what a corrected document looks like, and
that is a contract problem, not a model problem.

## Not in scope

- No production merge, deployment, account change or credit adjustment.
- No instruction change, no new prompt, no second model.
- No integrated website test yet — that is proposed **only after** this succeeds
  (outline below, not authorised).

## Afterwards: the smallest integrated website test

Only if the repair path succeeds. Outline, for planning only:

1. Take **one** real PDF/DOCX through the existing document-reader endpoint.
2. Run orchestrator → Imaginator → Gemma review with the binding gate.
3. Assert: the download renders, the corrected document is the one that was reviewed,
   and the release decision is the gate's rather than a UI default.
4. **Smallest** means one document, one pass, no new infrastructure, no model change —
   and it exercises an actual PDF/DOCX download, which is the only thing that proves
   the pipeline end to end.

That test would need its own authorization and its own limits; nothing in it is
approved by this proposal.
