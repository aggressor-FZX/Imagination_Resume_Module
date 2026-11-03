# Two proposals — not executed, not authorised

Neither proposal has been run. No submission, deployment, or gate change is authorised
by this document. Each names its own stopping rule and spend limit.

---

# Proposal A — realistic acceptance on the unchanged live build

## Question

Against the **current deployed build, unmodified**, does the normal website flow:

1. accept two uploaded documents through the ordinary UI,
2. preserve every source fact from the reviewed candidate pack with correct employer and
   institution attribution,
3. rewrite only where the source supports it,
4. produce an export whose filename matches the analysis,
5. produce PDF and DOCX content matching the approved markdown, and
6. download through an **ordinary, unforced** click?

Proposal A measures **current behaviour**. It is an acceptance observation, not a fix
validation. It **does not imply** that run 10's appended-outcome defect is resolved — the
lexical work list is not in the pipeline, so the same semantic risk is expected to be
present and will very likely be observed again.

## Tested revisions

None. The deployed revisions are used as-is:

| Service | Deployment | SHA |
|---|---|---|
| `imaginator-resume-cowriter` | current live | verify at run time from deploy metadata, expect `7cc415b` |
| orchestrator-backend | current live | verify at run time |
| orchestrator-frontend | current live | verify at run time |

If any deployed SHA differs from the expected value, **stop** — that is a different build
than this proposal describes.

## Inputs

The frozen, reviewed candidate pack (after the owner approves and freezes it):

- `candidate-main.txt` → rendered to `.docx`
- `candidate-supplement.txt` → rendered to `.docx`
- `intended-job-description.txt` → pasted verbatim into the job-description field
- Job title field: **"Maintenance Electrician"** (the ad's own title)

Staged to a **Windows-accessible directory** before upload; both filenames verified in the
page UI before submitting.

## Procedure

1. Record preflight: deployed SHAs, credit balance, capture flags (`CORRECTION_LIVE_CAPTURE=0`
   unless separately authorised).
2. Upload both files with the supported file input; verify both names in the UI; capture
   the ready-to-submit screenshot.
3. **Attempt an ordinary click** on "Continue to description", then on "Analyze". Record
   any interaction failure before considering a workaround.
4. Record the correlation ID and the credits delta.
5. On the results page, record the rendered markdown and the export request metadata:
   request URL, payload `headline`, response `filename`, and the client-chosen download
   name.
6. **Attempt an ordinary click** on each Download button. If it fails, record the failure
   and the diagnostic (`document.elementFromPoint` at the button centre) **before** any
   forced click.
7. If forced or JavaScript clicks are used, label the subsequent pipeline success
   separately from ordinary-interaction acceptance.
8. Open the saved PDF and DOCX; compare content against the approved markdown and against
   the F01–F22 ledger.

## Expected evidence

| Check | Pass condition |
|---|---|
| Upload, ordinary | Both filenames visible in UI; no `disabled` attribute removed |
| Attribution | All four employers, both institutions, each with own role/date/degree |
| Metrics | All eight, each under its own employer |
| Prohibited additions | No motor controls, no invented figure, no invented entity |
| Locations | Spokane, Spokane Valley, Coeur d'Alene preserved, ID/WA correct |
| Filename | `Machine_Learning_EngineerTest` must **not** appear; expect a name derived from "Maintenance Electrician" or the neutral `resume` |
| Export metadata | Request `headline` and response `filename` recorded, so a mismatch is **observed** not inferred |
| PDF content | Every approved content unit present |
| DOCX content | Same content, matching structure |
| Layout | No truncation, no orphaned heading, page break reasonable |
| Ordinary download | Recorded pass/fail **separately** from any forced-click success |
| Semantic | Appended outcome clauses recorded if present — expected, not a regression |

## Spend limit

- **Account credits: 33** (one analysis).
- **Provider dollars: no ceiling set by this document.** The correction stage measured
  $0.0029 in run 10; whole-pipeline cost is unmeasured. Request a separate provider-dollar
  allowance at authorisation time rather than inferring one from credits.

## Stopping rule

Stop at the first of:

1. one analysis submitted (no re-submission for any reason);
2. credits consumed exceeds 33;
3. any deployed SHA differs from the expected value;
4. the analysed document cannot be exported after **one** ordinary click plus **one**
   diagnosed forced click.

## Cleanup

- Preserve the capture, the approved markdown, both source documents and the exported
  PDF/DOCX locally with SHA-256.
- Reset `CORRECTION_LIVE_CAPTURE=0`, redeploy, confirm `capture=0`, purge remote captures.
- Leave the historical captures and the single-file freeze tag untouched.

---

# Proposal B — bounded semantic-review improvement

## Question

Can **one** specific change to the reviewer contract or writer prompt reduce unsupported
outcome claims — the run-10 defect class — **without** rejecting faithful paraphrases?

## Candidate change (exactly one)

**Require a per-clause citation from the reviewer, instead of a per-line verdict.**

Today the reviewer answers `supported: [line_ids]` with `evidence: {line_id: [segment
indices]}`. One verdict covers a whole line, so a line whose opening is sourced and whose
tail is invented receives a single `supported` marking, and the engine's only defence —
flag then repair — is never engaged. That is precisely what happened in run 10: 28 lines
marked supported, `unsupported: []`, `repair_calls = 0`.

The change: require `supported` to name, for each clause it approves, the source segment
that establishes **that clause**. A clause with no named segment is not a supported clause.

This is deliberately **not** a vocabulary rule, and the lexical work list is **not**
involved. It is advisory only and stays out of the pipeline.

## Tested revisions

| | |
|---|---|
| Revision under test | imaginator at the freeze tag's parent with only the reviewer contract changed |
| Control | the same revision with the contract unchanged |
| Deployment | **staging/local only.** No production deployment is proposed |

## Inputs — frozen

| Input | Source |
|---|---|
| Two-document source | `candidate-main.txt` + `candidate-supplement.txt` (frozen pack) |
| Job ad | `intended-job-description.txt` |
| Approved writer draft set | fixed drafts from the fixture, including: a faithful two-document draft, a draft with one appended outcome, a draft with a transplanted metric, a draft with an invented qualification |
| Reviewer model | held constant across both arms — the same model, same temperature, same seed where supported |
| Run 10 replay | the verbatim run-10 draft and sources from the committed `run10_excerpt.json` |

## Expected evidence

| Observation | Required result |
|---|---|
| Faithful paraphrase still approves | yes — the contract change must not reject a correct draft |
| Supported cross-document synthesis still approves | yes |
| Appended outcome clause is flagged or withheld | yes in at least 3 of the 4 run-10 addition cases |
| Transplanted metric still rejected | yes, unchanged |
| Invented qualification still rejected | yes, unchanged |
| Reviewer output parses under the strict validator | yes, on every call |
| Repair recovery engaged | `repair_calls > 0` in at least one case — currently 0 in every run |

**Offline first.** Every row above is asserted by an offline test with a scripted reviewer
before any real-model call. The offline suite already covers the mechanics; what it cannot
show is whether a real model can answer a clause-level contract without producing
malformed envelopes.

## Spend limit

- Real-model comparison: **2 analyses' worth of correction calls**, both arms, no more.
- Provider dollars: correction-stage cost was $0.0029 per run in run 10. Request a ceiling
  of **$0.05** for this comparison, which is ~17× the measured correction cost and covers
  retries, model variance, and the larger prompt.
- Credits: **0.** This uses no account credits — it is a local/staging correction call, not
  a website analysis.

## Stopping rule

Stop at the first of:

1. both arms have produced comparable evidence (2 runs each);
2. cumulative provider spend reaches $0.05;
3. the offline suite fails a paired control — a change that breaks faithful paraphrase is
   rejected regardless of its effect on the defect;
4. the reviewer produces malformed envelopes on more than 1 of 4 calls at the new contract
   — that is a format problem, to be fixed offline, not by spending more.

## Cleanup

- No capture flags changed. If capture is needed, use the existing guard, preserve locally,
  then purge remote — the run-10 procedure.
- Record provider usage from the response bodies, not from an activity export.
- No production deployment, no gate change, no branch push.
