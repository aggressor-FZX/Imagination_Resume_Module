# Live acceptance report — 2026-11-02

Synthetic documents only. No customer data. No secrets, résumés, prompts or model
responses written to ordinary logs. All six authorized synthetic analyses were
used; the $2 spending ceiling was not approached (≈ $0.004 recorded).

## Live revisions

| Service | ID | Live commit | Verified |
|---|---|---|---|
| imaginator-resume-cowriter | `srv-d3nf73ur433s73bh9j00` | `4059d12fa322` | deploy `dep-dbk7c8lg1s2s738n2750`, live 11:38:58Z |
| orchestrator-backend | `srv-d4fb5o4hg0os738sb3a0` | `dbbdf9077141` | unchanged |
| orchestrator-frontend | `srv-d4fb5f4hg0os738sasgg` | `dbbdf9077141` | unchanged |

Rollback: Imaginator `10b20d8` (last pre-correction build) or `429a6dd`.

## Acceptance table

| Criterion | Result | Evidence |
|---|---|---|
| `python-docx` restored from declared dependency; affected tests rerun | **PASS** | `requirements.txt` pins `1.1.2`; installed; the three previously blocked modules now run: **76 passed**, nothing skipped |
| Timeout budget honours remaining end-to-end time | **PASS** | six live budgets (32.8, 120, 84, 113, 105, 53.5 s) each equalled `180 − 20 − pipeline`; offline tests cover 0 s, `nan`, `True`, ceiling |
| Each correction call respects the remaining budget | **PASS** | run 6: review received the remaining 39.1 s and was cancelled exactly there (`budget_left_ms: 0`, `outcome: deadline`) |
| Events distinguish provider timeout / deadline / insufficient budget / validation rejection | **PASS** | 26 offline tests across all four phases; live runs produced `outcome: ok` with a validation reason, `deadline`, and `cancelled` as separate records |
| Input-size / output-capacity limitation stated accurately | **PASS** | see Limitations — measured, not assumed |
| Regression suite | **PASS (partial)** | 590 passed, 3 skipped; 22 scripted route doubles still encode the retired wire shape |
| Pushed through the production path, auto-deployed, revisions verified | **PASS** | 6 pushes; every deploy verified live by deploy ID **and** commit |
| Live single-file analysis approved and delivered | **FAIL** | 6 runs, all withheld, each with a distinct instrumented reason |
| Live multi-file analysis | **NOT RUN** | no approved single-file run to extend; allowance spent on single-file |
| Source facts and education relationships preserved in an approved output | **NOT RUN** | no approved live output |
| Correction commentary displayed live | **NOT RUN** | requires approval |
| Valid publication receipt live | **NOT RUN** | requires approval |
| Downloaded PDF/DOCX containing approved content | **NOT RUN** | controls correctly disabled while withheld |
| Account credits | 1463 → 1265 (−198, 6 × 33) | UI |
| Correction provider spend | ≈ **$0.0041** | 11 calls with recorded usage; one cancelled call unbilled |

## What was fixed and deployed

1. **`python-docx`** — declared but absent locally, which is why 13 export tests
   failed and one module would not collect. Restored from the pin; those tests now
   run and pass.
2. **Budget arithmetic** — `min(120 s, 180 − 20 − pipeline elapsed)`, withheld
   with `pipeline_budget_exhausted` and **no provider call** if under 30 s remains.
   Six live runs confirm it on every path.
3. **Per-call budget** — a phase receives only the remaining deadline, and the
   engine reports `timeout_<phase>` versus `provider_call_timeout_<phase>`
   separately. Run 6 demonstrates the first live.
4. **Privacy-safe phase events** — budget, phase start/end, elapsed time, request
   count, timeout category, tokens and cost, keyed by the orchestrator correlation
   ID. Sanitized at the logging boundary to numbers and fixed vocabularies; a
   sentinel test proves document text cannot reach a log line.
5. **The original blocker** — asking a model for exact code-point offsets, a
   gapless partition, and free-form category names. Live evidence: `end=544` for a
   492-character source, a dropped newline between spans, and invented categories
   (`header`, `experience_entry`, `job_title`, `responsibility`). Every real
   response was withheld as malformed. The provider now requests **aggregated index
   lists** and derives all offsets and quotes itself; the engine and its strict
   validator are unchanged, so no factual check was relaxed.
6. **Coverage semantics** — three live reasons traced to the mapping, not the model:
   a paraphrase left a rewritten line with **no candidate segment**; a supported
   line had no citation; section **headings** were `required` although no draft
   line can "represent" a heading. Each was reproduced locally with a
   rewrite-shaped draft before being fixed.

## Live runs

| # | Submit (UTC) | Correlation ID | Pipeline | Budget | Calls | Credits | Reason |
|---|---|---|---|---|---|---|---|
| 1 | 10:52:14 | `55a1988b-1789-422a-b784-89ddf6cc0aef` | 127.2 s | 32.8 s | 8.0 + 11.3 s | −33 | `source_completeness_uncertain` |
| 2 | 11:01:03 | `0978262d-7303-403c-9fcf-dedc89f3f096` | 29.1 s | 120.0 s | 15.5 + 3.6 s | −33 | `malformed_output` |
| 3 | 11:17:53 | `1ab8e6eb-3507-41c6-9b2e-f23b97cf3827` | 76.0 s | 84.0 s | 30.3 + 40.6 s | −33 | `malformed_output` |
| 4 | 11:24:17 | `706787f6-9bae-4d01-9716-bf3ca8950c74` | 46.9 s | 113.1 s | 10.8 + 14.7 s | −33 | `source_completeness_uncertain` |
| 5 | 11:33:58 | `c77c8116-4487-4621-b16f-a3dc28db7ea0` | 54.9 s | 105.1 s | 10.9 + 26.1 s | −33 | `source_completeness_uncertain` |
| 6 | 11:40:36 | `f22a014e-b29c-43b0-87e9-6c41b6d075cf` | 106.5 s | 53.5 s | 14.4 s, then **deadline at 39.1 s** | −33 | `timeout_initial_review` |

Every provider call that was not cut off returned `finish_reason: stop` with
`outcome: ok` in 3.6–40.6 s. The original timeout theory is superseded: time was
never the first constraint.

## Where the work stands

The correction pipeline now behaves correctly against the real model in every
offline check, at zero cost:

| Draft | Result |
|---|---|
| faithful (byte-identical) | **approved**, source returned unchanged |
| realistic rewrite (paraphrased + added Summary/Skills sections) | **approved** |
| one altered number | **withheld** |

After all the loosening of the *wire format*, a single changed figure is still
rejected. The probes are `/tmp/probe_pair.py` and `/tmp/probe_extra.py` and are
reproducible at zero cost.

The remaining live gap is **wall-clock headroom**, not correctness. The Imaginator
pipeline itself took 29–127 s (median ≈ 65 s), and correction needs roughly 25–55 s
for two calls at this document size. Run 6 is the clear case: a 106.5 s pipeline
left 53.5 s and the review was cut off. Nothing inside the correction path can
create time the pipeline has already spent.

## Limitations, stated accurately

- **Output allowance.** Under the original contract the reviewer copied every span
  with offsets and quotes, so output grew with input; the accepted 48 KB ceiling
  measured ≈95.5 k characters (≈24–32 k tokens) and could not fit 12,000. The
  index-list protocol answers with integers, so input size no longer drives output
  size. The test measures the real wire answer and asserts the maximum accepted
  source fits with a wide margin. This is **measured, not proven by a live 48 KB
  upload**, and accepted input sizes were **not** reduced.
- **Headroom** is the live constraint. Attempts during this work, all reverted
  because they traded away something more important:
  raising the deadline ceiling (cannot exceed the `180 − 20` bound, so it changes
  nothing); trimming the review prompt to cited segments only (lost the precise
  `source_facts_missing` reason, downgrading it to `uncertain`).
  The right fix is to make the slow phase shorter or usable when partial:
  stream the review, or split review by section so no single call is long.
  Increasing timeouts further would not help — one failing run had 53.5 s
  available, while an earlier run had 113 s and used 25 s.
- **22 scripted route doubles** still encode the retired strict wire shape. This is
  test-harness debt; the production exchange is covered by 26 phase/timing tests
  plus six live runs, and should be converted in a dedicated change.
- **Privacy scope unchanged**: the disclosure gate still scans source documents
  only, not draft lines or claims.
- **Spend is only partly measured.** Correction calls recorded usage
  (≈ $0.0041 over 6 runs; one cancelled call unbilled). Pipeline, researcher and
  job-search spend is not visible in these events and is **not** included.
  Account credits are not dollars.

## Next step

Section-split or streamed review, so a slow phase cannot exhaust the remaining
budget. That is a code change on the same deploy path, verifiable offline against
the existing probes before a live run. Then one single-file run re-validates, and
the multi-file run follows.
