# Gemma repair-and-release test — RESULTS

Authorization token `gemma-repair-2026-11-09-01`, used **once**. Model
`google/gemma-4-31b-it`. No production change, no merge, no deploy.

## Budget

| Limit | Authorized | Used |
|---|---|---|
| Logical calls | 10 | **8** |
| HTTP requests | 20 | **8** |
| Observed spend | $0.05 | **$0.001262** |

`cost_is_known: true`, 0 unmetered, 0 retries, 0 operational failures.
Five cases needed 8 calls, not 10: three hit `revise` then released on re-review;
two were approved on the first call and correctly needed no second.

## Outcome — all five documents released

| Case | Role | First | Repair | Re-review | Edits | Released |
|---|---|---|---|---|---|---|
| `faithful_paraphrase` | **control (familiar)** | approve | not attempted | — | **0** | **yes** |
| `cal_ops_faithful` | **control (unfamiliar)** | approve | not attempted | — | **0** | **yes** |
| `unsupported_method` | clause removal | revise | **applied** | approve | 1 | **yes** |
| `wrong_record_attribution` | wrong-owner correction | revise | **applied** | approve | 1 | **yes** |
| `omitted_record` | record restoration | revise | **applied** | approve | 1 | **yes** |

**This is the first time in this line of work that a corrected document has reached
release.** Every release is backed by a valid re-review of exactly that text, and
every receipt verified (`receipt=ok`). Both controls stayed byte-identical, so no
edit was applied to a document that was already faithful.

## The three corrections, as applied

**Clause removal** — the invented method is gone, the sourced achievement survives:

```diff
-- Cut rework by 18% through standardizing panel labeling and targeted quality audits across the terminal fleet.
+- Cut rework by 18% through standardizing panel labeling across the terminal fleet.
```

18%, "standardizing panel labeling", "Cascade Freight Partners" and "terminal fleet"
all survive. On live output Gemma wrote "through" where the source says "by" — exactly
the paraphrase the old byte-identity check refused.

**Wrong-owner correction** — the merged claim leaves Cascade, and the fact stays under
Harborline, whose bullet was never touched:

```diff
-- Prepared the monthly cost summary that the regional manager reviewed and reconciled 1,180 fuel invoices each quarter.
+- Prepared the monthly cost summary reviewed by the regional manager.
```

**Record restoration** — every line taken from `candidate-supplement.txt`:

```diff
+**Ridgeline Equipment Services** -- Fife, WA
+*Inventory Analyst | 08/2023 - Present*
+- Maintain parts availability records for 7 service branches.
+- Compared quarterly parts usage against the prior four quarters to set reorder points.
```

Verified: no invented dates (`08/2022`/`03/2023` absent), no duplicate record, both
source bullets complete to "reorder points", placed before `## Certifications`.

## What was proven, in order

1. **The repair gaps are closed on LIVE output**, not only on replayed text. All
   three corrections were accepted; the old `safe_repair` would have refused all
   three (`replacement_not_exact_owned_source_bullet` twice,
   `omission_requires_explicit_destination` once).
2. **The release path completes.** original → revise → repaired → exact re-review →
   valid approve → structure check → receipt → verified → released.
3. **Hard controls bind.** Sections checked before any receipt was minted; the
   released hash equals the proposed hash for all three corrected documents, so the
   exported version is the approved version.
4. **Controls are inert.** Zero edits on both faithful drafts.

## Two integration defects found and fixed while wiring this

Both would have silently broken release, and neither was visible from the experiment
alone:

| Defect | Symptom | Fix |
|---|---|---|
| `verify_receipt` needs the renderer payload | every receipt verified as `receipt_not_approved` | the payload is derived from the **approved** markdown, never an earlier draft or a source alias |
| verdict must be the typed value | passing the string `"approved"` minted a receipt that failed verification | uses `Verdict.PASS_TIER_A.value`, imported rather than copied |

## Remaining blocker for website validation

**The live site does not run this path, and I cannot deploy it.**

Established facts:

- `imaginator-resume-cowriter` and `orchestrator-backend` are **healthy** (both 200).
- Production reviewer is `OR_SLUG_DRAFTER = "qwen/qwen3-coder"` with **no separate
  reviewer slug** — the live site has never used Gemma.
- Production review uses the older **span-envelope contract**
  (`source_correction.py`: `claims`/`representations` with code-point offsets), not
  `decision_contract`'s prose contract. They are different protocols.
- `origin/master` **lacks** `experiment/repair.py` and `experiment/release_trace.py`;
  the experiment branch is 18 commits ahead of master and master is 0 ahead of it, so
  a merge is a real code change to the service that generates every résumé.
- **No `RENDER_API_TOKEN` is present in this environment**, and no deployment can be
  triggered from here.

So the honest sequence is: the repair path is proven, but wiring it into
`imaginator-resume-cowriter` is a production change that requires (a) an explicit
decision to replace the span-envelope review path with the prose contract, and
(b) Render credentials to deploy. Neither is available under this authorization,
which states *Production changes: none*.

**What is needed to finish item 6** (a real PDF/DOCX through the website):

1. Approval to merge the experiment into a release branch of
   `Imagination_Resume_Module` and point the imaginator at it — this replaces the
   live review contract, so it needs to be an explicit decision, not an inference.
2. `RENDER_API_TOKEN` (or a human-triggered deploy from the dashboard).
3. A signed-in test account for the browser run. Guest auth is **not** reopened as a
   prerequisite, per the brief.
4. A new authorization for the website run's inference.

Until then, no PDF or DOCX has been produced by the website with this reviewer, and I
will not report one.

## Not established

- Five documents, one run, temperature 0. The repair path is proven on these cases,
  not on arbitrary output.
- No PDF/DOCX was generated or inspected — that requires the deployment above.
- The receipt was verified with a **test** signing secret
  (`repair-test-receipt-secret-2026`), not the production `PUBLICATION_RECEIPT_SECRET`.
  The production key is not in this environment, so production receipt issuance is
  untested here.
