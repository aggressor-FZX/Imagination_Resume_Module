# Session 18 — source-word escape closed; approval vs faithful rate separated

## Verification commands

```
cd /home/skystarved/Render_Dockers/Imagination_Resume_Module/.worktrees/resumeDraft_fix
OPENROUTER_API_KEY_1=x OPENROUTER_API_KEY_2=y IMAGINATOR_AUTH_TOKEN=z STAGE_COOLDOWN_SECONDS=0 \
  /home/skystarved/Render_Dockers/.venv/bin/python -m pytest tests/ \
  --ignore=tests/test_loader_source_preservation_offline.py -q -o addopts='' -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/skystarved/Render_Dockers/Loader/.venv/bin/python -m pytest -p no:cacheprovider \
  -o addopts='' -q tests/test_loader_source_preservation_offline.py
git diff --check
```

Result: **1661 passed, 13 skipped** (5 new regressions); export controls **85 passed**.

## Deployed

- Imaginator **`3e63eac`** live (`dep-dbnf46o473hc739ipo30`), health 200.
- Branch `resumeDraft_fix` merged to `master` (service deploys from `master`).

## The source-word escape — fixed

Three approved Conservative bullets (runs `dac8b3f5`, `b2c9686f`, `0365614d`)
carried an appended outcome whose tail ALSO echoed a word the source states:

```
"...cost summaries enabling regional manager oversight of key operational expenses"
   "regional manager"                    -> in source
   "oversight of key operational expenses" -> NOT in source
```

The old rule refused the whole tail because it contained a source word, so the
unsupported outcome reached an approved export. Trimming instead would have
dropped the sourced fact. Neither is faithful.

**Repair:** substitute the candidate's own source bullet — the mechanism this
module already uses. It keeps every fact the bullet states, removes the addition,
and composes nothing. It fires only when the tail is genuinely MIXED (carries
source content the prefix lacks AND ungrounded content) and never when the tail
restates a figure the bullet states (that is a faithful paraphrase of the
source's own outcome, e.g. "cutting rework by 18%").

A bare outcome participle with no comma is a weaker signal, so it is judged last
and **bullet-scoped**: the segment must be entirely ungrounded against the
MATCHED BULLET and carry no figure. Bullet-scoping is the point — on `dac8b3f5`
the tail's words all occur somewhere in the source, so a document-wide check
found nothing ungrounded.

No phrase blacklist, no global override of `missing`, no automatic removal of all
purpose language. The six existing omission controls are unchanged.

## Approval rate vs verified faithful-delivery rate

These are different numbers and must be reported separately.

| Batch | Approved | Faithful delivery |
|---|---|---|
| bridge_opportunity / Conservative | 1/1 | 1/1 |
| bridge_opportunity / Imaginative | 1/1 | 1/1 |
| realistic_candidate / Conservative | 0/1 (withheld) | n/a |
| realistic_candidate / Imaginative | 1/1 | 1/1 |
| **Total** | **3/4 (75%)** | **3/3 approved were faithful (100%)** |

Session 17 reported 8/10 approved. That number was an **approval** rate; three of
those approvals carried appended clauses, so the faithful-delivery rate was
lower. On the final build the two rates coincide for the approved runs because
the escape is closed — but they are still reported separately, because an
approval is a gate decision and faithful delivery is a content property.

**False approvals on the final build: 0.** Every bullet in every approved run was
checked against the source; the three bullets that match a purpose/outcome
pattern are verbatim source text ("...to find which equipment failed repeatedly",
"...to set reorder points", "...for review by regional manager").

## The one withhold is upstream of this change

`realistic_candidate` / Conservative withholds deterministically (2/2 runs) with
`source_completeness_uncertain`. It fires at `source_correction/source_inventory`
— before any draft text is judged — so it is structurally upstream of the
preservation layer and is not a regression from this session. The same candidate
approves in Imaginative mode with 4 grounded bridges.

## Inference cost vs credit cost

Measured from `llm_usage_log` (service `imaginator`), 2026-11-07:

| Stage | Model | Calls | USD |
|---|---|---|---|
| source_correction (judge) | `qwen/qwen3-coder` | 111 | 0.2958 |
| pipeline (writer) | (unlabelled) | 112 | 0.2235 |
| researcher | `inception/mercury-2` | 24 | 0.0359 |
| **Total** | | **247** | **0.5552** |

34 runs, 784,895 tokens, **$0.5552** total = **$0.0163/run**.

Credit value at the $5/450 tier ($0.01111/credit): 33 credits = **$0.3667**.
So inference is **4.4% of credit value** — a **22.6×** gross margin on inference
alone. This is API spend, not Stripe revenue.

## Resume completeness — FIXED (was: two export defects)

**Status: fixed in `061954a`, tests in `627d159`.** The section below is kept as
the record of what was measured and why the fix looks the way it does.

### The fix

Both defects were in `resume_builder/build_resume_from_export.py`:

1. `build_summary_block` now emits `\section{Professional Summary}` before the
   summary text, and the DOCX path calls a new `add_section_heading` helper
   instead of writing a bare paragraph. The heading stays **conditional** on
   there being summary text, matching `ensure_required_headings`: an empty
   summary heading would claim a summary the candidate never wrote.
2. `generate_projects_block` and the DOCX Projects block no longer guard the
   heading with `if resume.projects:`. The heading is unconditional; only the
   body is conditional. Nothing is invented to fill an empty section.

The DOCX renderer's five copy-pasted heading blocks (each with its own border
`pPr`/`pBdr` construction) are now one `add_section_heading` helper, so a section
cannot drift from the others again.

### Verification

- `tests/test_export_section_headings.py` (6 tests) renders real files and reads
  them back, including a PDF/DOCX section-set comparison.
- `tests/test_export_route_section_headings.py` (4 tests) does the same through
  `POST /resume/export`, the endpoint the website calls, so a route-level payload
  defect cannot hide behind a renderer-only test.
- Full suite: **1671 passed, 13 skipped**.

### What was measured before the fix

## Resume completeness — what the sections actually show

Expected headings (`REQUIRED_HEADINGS`): Professional Summary, Professional
Experience, Projects, Education, Certifications, Technical Skills.

The **delivered markdown** carries all six. The **exports do not**:

| Section | Markdown | PDF | DOCX |
|---|---|---|---|
| Professional Summary | yes | **heading dropped** | **heading dropped** |
| Professional Experience | yes | yes | yes |
| Projects | yes (empty) | **dropped** | **dropped** |
| Education | yes | yes | yes |
| Certifications | yes | yes | yes |
| Technical Skills | yes | yes | yes |

Two distinct defects, both in `resume_builder/build_resume_from_export.py`:

1. **Summary heading is never emitted.** `build_summary_block` writes the summary
   text with no `\section{...}`, and the DOCX path writes `r.text =
   resume.summary` with no heading. The text survives; the label does not.
2. **Empty sections are omitted entirely.** `if resume.projects:` skips the
   heading when the list is empty, so `Projects` disappears rather than appearing
   as an empty section. `REQUIRED_HEADINGS` expects it present.

Content completeness is otherwise good: 3 employers, 9 bullets, 2 education
records, 1 certification, 4 skill lines. `Projects` is empty in the source, so
its emptiness is faithful — only the heading is missing.

## RESOLVED: OpenRouter balance, and a misdiagnosis it exposed

The account was topped up (balance now ~$9.82). Re-running the live verification
surfaced a second, unrelated defect that the outage had been hiding.

### A malformed answer was reported as a provider error

A live Imaginative run withheld as `provider_error`, and the log looked like an
outage:

```
provider.call          http_status 200  outcome ok  phase initial_review
correction.phase_end   outcome callback_error
correction.end         reason provider_error
```

Both provider calls returned 200 with `outcome: ok`. The reviewer had returned
JSON that could not be mapped onto the engine's strict review shape, and the
adapter's generic `except Exception` swallowed that as `provider_error`.

`UnjudgedLines` was already mapped to `incomplete_review`. Its sibling
`MalformedIndexedResponse` — the same base class — was not, so a duplicate line
id or an out-of-range segment index was reported as a network fault. That sends
an operator to check credentials when the real answer is "the reviewer returned
an unusable answer".

**Fixed in `3e8953f`:** the adapter translates it into a new
`MalformedReviewAnswer` and re-raises; the engine maps it to `malformed_output`
and names the phase event `malformed_answer`. The new exception is a
`ValueError`, deliberately not a subclass of any transport exception, so an
adapter catching `ProviderUnavailable` cannot mistake a validation failure for
availability and retry it. Five regression tests pin the mapping, including that
the handler precedes the generic catch-all.

### Batch on the final build (`3e8953f`)

| Batch | Status | Reason |
|---|---|---|
| bridge_opportunity / Conservative | approved | — |
| bridge_opportunity / Imaginative | approved | 3 grounded bridges |
| realistic_candidate / Conservative | withheld | `source_completeness_uncertain` |
| realistic_candidate / Imaginative | withheld | `no_op_repair` |

**Approval rate: 2/4 (50%). Faithful delivery: 2/2 approved (100%). False
approvals: 0.** Both approved runs carry all six headings and zero escape
shapes. Both withholds now report real content verdicts instead of
`provider_error` — which is the point of the fix: the reason code now describes
what actually happened.

Note the second candidate's verdicts are not stable across runs
(`source_completeness_uncertain` / `no_op_repair` / previously `provider_error`).
That is model variance in the reviewer, not a code path; the withhold itself is
consistent.

### End-to-end export verification (real downloaded files)

A live website run on the final build produced a PDF and DOCX that were
downloaded and read back with PyMuPDF and python-docx:

- **All six headings present in both files** (previously four).
- **Zero escape shapes** in either file.
- Section order correct; `Projects` present and empty with no invented content.

## Historical: OpenRouter account was overdrawn

A live website verification run on 2026-11-07 12:16 returned
`correction_status: withheld`, `reason: provider_error`. The log shows the real
cause:

```
"event": "provider.call", "http_status": 402, "model": "qwen/qwen3-coder",
"outcome": "provider_error", "phase": "source_inventory"
```

`GET https://openrouter.ai/api/v1/credits` for both keys reports
`total_credits: 1240`, `total_usage: 1240.158620078` — the account is **overdrawn
by $0.16**. Per-key limits still show headroom (`limit_remaining` 92.51 and
99.48), which is why the keys look healthy; the 402 is account-level.

**No fresh live analysis can run until the account is topped up.** This is a
billing state, not a code defect, and it is why the export fix was verified
through the real `/resume/export` route in-process rather than through a new
website run.

## Remaining work

1. **Top up OpenRouter**, then re-run the live website verification to confirm
   the six headings reach a downloaded PDF and DOCX end to end.
2. `realistic_candidate` / Conservative withhold: decide whether
   `source_completeness_uncertain` at inventory is the intended outcome for that
   document, or whether the inventory contract is over-strict.
3. Consider surfacing a 402 distinctly from a generic `provider_error`. A
   billing failure currently looks identical to a transient provider fault, so
   the operator is told "could not be verified" when the real answer is "the
   account has no credit".

---

# Session 17 — Imaginative delivered with grounded bridges; batches recorded

## Verification commands

```
cd /home/skystarved/Render_Dockers/Imagination_Resume_Module/.worktrees/resumeDraft_fix
OPENROUTER_API_KEY_1=x OPENROUTER_API_KEY_2=y IMAGINATOR_AUTH_TOKEN=z STAGE_COOLDOWN_SECONDS=0 \
  /home/skystarved/Render_Dockers/.venv/bin/python -m pytest tests/ \
  --ignore=tests/test_loader_source_preservation_offline.py -q -o addopts='' -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/skystarved/Render_Dockers/Loader/.venv/bin/python -m pytest -p no:cacheprovider \
  -o addopts='' -q tests/test_loader_source_preservation_offline.py
git diff --check
```

Result: **1656 passed, 13 skipped**; extractor suite **17 passed**.

## Deployed

- Imaginator **`2d842a7`** live, health 200. Frontend/backend **`4ea9ae5`** unchanged.
- `CORRECTION_LIVE_CAPTURE=0` (applied by real deploy in session 16, read back).

## Credit reconciliation — the 33-credit gap is explained

`836 → 143` is 693 credits. 20 recorded submissions × 33 = 660. The difference is
**one additional submission**: `analysis_jobs` shows **21** jobs on 2026-11-07
(ids 79–99), not 20. My session-16 table omitted job **94** (06:47:33), which is
the Conservative run whose correlation I recorded as `104b7f7a`. So 21 × 33 = 693
exactly, and the balance arithmetic is correct.

**Real money vs credits.** These are separate ledgers and do not convert:

| Ledger | Amount |
|---|---|
| Stripe `payment_records`, status `completed` | **$50.00** (6 payments: 5×$5 + 1×$25) |
| Stripe `payment_records`, status `pending` | $55.00 (11 abandoned checkouts, never charged) |
| Stripe `payment_records`, status `refunded` | $10.00 (2 refunds) |
| `credit_transactions` purchases | 5,888 credits |
| `credit_transactions` analysis debits | −120 credits (2 rows only) |
| `credit_transactions` refund | +60 credits |

The ledger records only **2** analysis debits, but 56 analysis jobs exist. The
per-analysis debit is therefore **not** written to `credit_transactions`; the
balance is decremented elsewhere. That is a pre-existing bookkeeping gap, not
something this session introduced, and it means credits cannot be reconciled to
dollars through the ledger alone.

**No real money was spent this session.** All 13 submissions today were paid from
the existing credit balance. Provider dollars (OpenRouter) are a third, separate
balance and are estimated at **$0.0826** for the six direct diagnostics; website
runs report `estimated_cost_usd` per response but are not billed to the account.

**Top-up applied:** +2,000 credits as a `grant` row (`grant-session16-acceptance`)
with a matching `users.credits` update in one transaction. Balance **143 → 2,143**.
Ledger total now 7,828, matching the user row.

## Milestone status

| Milestone | Status |
|---|---|
| 1 — Honest provider-output handling | **Done** |
| 2 — Preserve candidate records | **Done** |
| 3 — Customer journey, Conservative | **Done** (session 16 baseline) |
| 3 — Customer journey, Imaginative | **Done** — approved, PDF+DOCX verified, 3 grounded bridges |
| 4 — Two-mode distinction | **Done** — see below |
| 5 — Consistency batches | **Recorded** — 5 per mode, results below |
| 6 — Files and operational finish | **Done** — files verified, capture off, services healthy |

## The six-row comparison (run `61bce5ef`)

| # | Source identity | Pass-1 verdict | Post-repair text / ownership | Pass-2 request mapping | Raw answer | Normalized verdict |
|---|---|---|---|---|---|---|
| 1 | `document-0:181:239` Harborline Transit Authority | `supported`, 1 citation, span L000007:0:46 | L000007 unchanged, byte-identical | L000007 present, same id | `structural`, 0 citations | `missing` |
| 2 | `document-0:605:657` Cascade Freight Partners | `supported`, 1 citation, span L000013:0:40 | L000013 unchanged | L000013 present | `structural`, 0 citations | `missing` |
| 3 | `document-1:170:226` Ridgeline Equipment Services | `supported`, 1 citation, span L000019:0:44 | L000019 unchanged | L000019 present | `structural`, 0 citations | `missing` |
| 4 | `document-0:979:1038` Puget Sound Technical College | `supported`, 1 citation, span L000029:0:47 | L000029 unchanged | L000029 present | `structural`, 0 citations | `missing` |
| 5 | `document-1:547:599` Inland Records Institute | `supported`, 1 citation, span L000032:0:40 | L000032 unchanged | L000032 present | `structural`, 0 citations | `missing` |
| 6 | `document-0:413:512` depot scorecard bullet | `unsupported` (L000009) | L000009 changed: marker stripped, bridge text removed | L000009 present, new text | `supported` | `represented` |

**Diagnosis.** Rows 1–5 are a genuine model self-contradiction: the same line, the
same id, unchanged text, confirmed then denied. Row 6 is the opposite direction —
the repair changed the line and the reviewer then accepted it, which is correct.

There is **no request-mapping defect**: the line ids and text are identical
between passes, so the reviewer received the same evidence and answered
differently.

## The fix: bounded coverage reconciliation

`_validate_review` now reconciles a `missing` verdict only when ALL hold:

1. the prior pass recorded the item `represented` with a span;
2. that span's line still exists and its text is **byte-identical**;
3. the item's quote is verbatim in that line (markup-insensitive);
4. the item's quote is verbatim in the source document it came from;
5. the current pass offers no competing evidence.

The restored claim cites the item's **real source offsets**, so the evidence is
the current draft plus the original source — not the prior verdict. Any changed
line, paraphrase, transplant or genuinely absent item keeps the refusal.

`tests/test_coverage_reconciliation.py` pins the captured case plus **nine**
refusals. The six pre-existing omission controls still pass.

## Second fix: invented figure in a trailing clause

Run `9b291b32` was approved with "enabling supervisors to assess performance
across 14 sites" appended to a bullet that states no figure. The transplant guard
refused any tail figure, so the invented "14 sites" reached the export. The guard
now refuses only figures the matched bullet **does** state and the prefix lacks.
A dangling-participle walk-back prevents leaving "…, enabling supervisors".

## Imaginative delivery (faithful)

Correlation `5c395bcf-5e30-4320-a59b-722bb7532914`, revision `2d842a7`, 6.4 s,
0 repairs, 1 review, **approved**.

Three grounded bridges survived with provenance in the export policy state:

| Skill | Origin | Source |
|---|---|---|
| trend analysis | researcher_inference | document-0 |
| equipment failure analysis | researcher_inference | document-0 |
| stock movement analysis | researcher_inference | document-1 |

PDF 167,196 B and DOCX 50,304 B, both HTTP 200. Text extracted from both files
contains all three bridge labels, all three employers, `1,180`, `prior four
quarters` and CPIM; none contains `SYNTHETIC-ONLY`, `14 sites`, `enabling
supervisors` or `improve dispatch team visibility`.

**Two-mode distinction is demonstrated:** Conservative delivered the same
candidate with source-only skills and no inferred label; Imaginative delivered
three inferred labels, each with a source document and origin, and no invented
tool, metric, credential or employment history.

## Acceptance batches (revision `2d842a7`, frozen input)

Conservative — 5 submissions:

| # | Correlation | Result | Notes |
|---|---|---|---|
| 1 | `a766d7dc` | withheld `source_completeness_uncertain` | reviewer variance |
| 2 | `dac8b3f5` | approved | 3 appended clauses survived |
| 3 | `ba7dfbd5` | approved | 2 appended clauses survived |
| 4 | `b2c9686f` | approved | 2 appended clauses survived |
| 5 | `0365614d` | approved | 1 appended clause survived |

Imaginative — 5 submissions:

| # | Correlation | Result | Notes |
|---|---|---|---|
| 1 | `12a59e70` | approved | no bridge label in output |
| 2 | `0de80639` | approved | 1 bridge label |
| 3 | `83b88200` | approved | 3 bridge labels |
| 4 | `2164e5e0` | approved | 2 bridge labels |
| 5 | `39857755` | withheld `source_completeness_uncertain` | 1 repair, 2 reviews |

**Observed success rate: 8/10 approved (80%).** Conservative 4/5, Imaginative
4/5. Latency 4.0–18.6 s. This is a small sample and is **not** proof of universal
reliability.

**Honest limitation:** the appended-clause trimmer does not catch every case. In
the Conservative batch, clauses whose tail echoes a source word ("impacting fleet
reliability", "enabling regional manager oversight") are refused by the
`new_source_content` guard and reach the export. That is a real fidelity gap, not
a passing negative control.

## Credits and cost

- 13 website submissions this session, **429 credits** (2,143 → 1,747 after the
  top-up; 1,780 → 1,747 across the batches).
- 33 analysis jobs today in total (ids 79–111).
- Provider dollars: **$0.0826 estimated** for six direct diagnostics. Website
  provider cost is reported per response but not billed to the account.
- HasData credits were not measured; the integrated job-search check ran on every
  submission and returned real matches.

## Next steps

1. Close the appended-clause gap: a tail that echoes one source word but adds an
   unsupported outcome still reaches export. The captured examples are in the
   Conservative batch above.
2. Run the second representative candidate.
3. Exercise the export tamper controls (content, policy state, owner).

## Uncommitted / private

- Pre-existing untracked `scripts/` excluded.
- `captures/recovery-validation/` git-ignored, retained.
- Main module checkout keeps its pre-existing dirty `HANDOFF.md` and untracked
  `docs/Plans/`.

---

# Session 16 — Conservative delivered faithfully; Imaginative blocked by reviewer

## Verification commands

```
cd /home/skystarved/Render_Dockers/Imagination_Resume_Module/.worktrees/resumeDraft_fix
OPENROUTER_API_KEY_1=x OPENROUTER_API_KEY_2=y IMAGINATOR_AUTH_TOKEN=z STAGE_COOLDOWN_SECONDS=0 \
  /home/skystarved/Render_Dockers/.venv/bin/python -m pytest tests/ \
  --ignore=tests/test_loader_source_preservation_offline.py -q -o addopts='' -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/skystarved/Render_Dockers/Loader/.venv/bin/python -m pytest -p no:cacheprovider \
  -o addopts='' -q tests/test_loader_source_preservation_offline.py
git diff --check
```

Result: **1645 passed, 13 skipped**; real-extractor suite **17 passed** (two pytest config warnings). The two suites need separate environments because the main venv lacks `fitz`.

## Deployed revisions

- Imaginator `srv-d3nf73ur433s73bh9j00`: **`bc9e6df`** live (`dep-dbnb…`), health 200.
- Frontend/backend: **`4ea9ae5`** (unchanged this session), health 200.
- `CORRECTION_LIVE_CAPTURE` is **0** and was applied by a real deploy; verified by reading the service env var back.
- Non-deploying branch `wip/two-mode-preservation-2026-11-07` tracks the same commits.

## Milestone status

| Milestone | Status |
|---|---|
| 1 — Honest provider-output handling | **Done.** Reasoning is never promoted to an answer; `finish_reason` other than `stop` is a failure; malformed/typed-wrong research arrays are explicit failures; a valid zero-findings response stays distinct. |
| 2 — Preserve candidate records | **Done for the frozen DOCX/PDF inputs.** Real Loader extraction is exercised; ownership, qualifiers, skills, education shape and marker invariants are enforced. |
| 3 — Customer journey, Conservative | **Done.** Approved through the website, PDF and DOCX downloaded and inspected, policy state transported, no unsupported clauses, no marker leak. |
| 3 — Customer journey, Imaginative | **Not done.** Withheld on reviewer reliability (see below). |
| 4 — Two-mode distinction | **Not demonstrated end to end.** The bridge exemption is implemented and unit-tested; no Imaginative delivery reached the files. |
| 5 — Consistency batches | **Not started.** Requires a stable Imaginative path. |
| 6 — Files and operational finish | **Partially done.** Conservative files verified; capture off; services healthy. |

## Conservative delivery (faithful)

Correlation `67c1af07-df99-44b8-ab60-b7da8089c532`, revision `8606610`, 30.2 s, 1 repair, 3 reviews, **approved**.

Delivered text compared line by line against both frozen sources: all three employers, roles, dates, locations, `1,180` invoices, `14` depots, `3` years, `7` service branches, `2` branches, the four-quarter qualifier, all four skills, both education records and CPIM are present and correctly owned. No appended purpose/outcome clause, no `SYNTHETIC-ONLY` marker, no invented tool, metric or credential.

Downloads: PDF 167,096 B (1 page, pdfTeX/moderncv) and DOCX 50,296 B (python-docx), both HTTP 200 with `skill_inference_policy_state` present in the request body. Text extracted from both files matches the displayed résumé.

## Imaginative blocker

Correlation `61bce5ef-59e7-41d4-bfbf-05fa7353555a`, revision `bc9e6df`, 15.8 s, 1 repair, 2 reviews, **withheld `source_facts_missing`**.

The reviewer's second pass reported five employer/institution head lines and one bullet as `missing` although every one is present in the draft. The first pass reported only the depot-scorecard bullet. This is reviewer inconsistency, not a deterministic defect: the same heads were `represented` in the first pass of the same run.

The bridge exemption itself works — the first pass no longer downgrades a representation for a permitted label. What remains is that the reviewer's coverage verdict is unreliable on this input, and the engine correctly refuses to publish on it.

**Do not** override a `missing` verdict to force approval. That was measured this session: it broke six withhold controls, including the negative control proving an omitted fact is never silently accepted.

## Defects fixed this session (all deployed)

1. `7622305` reasoning-as-answer → explicit failure; `finish_reason` honoured; typed research arrays validated.
2. Employer placeholder repair requires unique identity, not record order.
3. Missing source skills are inserted in a Skills section, never over experience bullets.
4. Streaming responses validate `finish_reason` and structured JSON; no retry after visible tokens.
5. Appended purpose/outcome tails are trimmed against the **matched bullet**, iteratively, preserving the candidate's own prefix.
6. Corporate suffixes (`Inc.`, `Ltd.`) are recognised as employer heads.
7. Each document's summary is written on its own line, not merged.
8. Bare extracted section headings are structural, including when the inventory says `required`.
9. A skills block that omits source skills is completed from the source.
10. Education requires the institution to be the bold head.
11. A repair may not delete a verbatim source span the reviewer only called `uncertain`.
12. Provenance markers are stripped at the approval point.
13. An uncited whole source line marked `uncertain` is upgraded to `supported`.
14. Permitted bridge labels are exempt from the introduced-content check, phrase-scoped.

## Live attempts this session

Direct `/analyze` diagnostics (frozen text source; **not** website tests, not credit-consuming):

| Revision | Mode | Correlation | Result | s | est. USD |
|---|---|---|---|---:|---:|
| 7622305 | Conservative | diagcons-20f402920358 | approved | 21.6 | .0114649 |
| 7622305 | Imaginative | diagcons-c4498f3a7831 | approved | 27.9 | .0170910 |
| b99377e | Conservative | diagcons-ace33487af7b | withheld `source_completeness_uncertain` | 25.1 | .0135656 |
| b99377e | Imaginative | diagcons-4156e59aa7c8 | approved but **false approval** (unsupported outcomes, lost qualifier) | 42.4 | .0145075 |
| d2b9578 | Conservative | diagcons-a64b81c72ab2 | withheld `timeout_initial_review` | 160.1 | .0085499 |
| d2b9578 | Imaginative | diagcons-61b1360e311a | withheld `source_completeness_uncertain` | 104.7 | .0173819 |

Session subtotal **$0.082561 estimated**; all 13 retained direct attempts total **$0.167040 estimated**. These are adapter estimates and can omit failed or cancelled provider calls; they are not billed totals.

Website submissions (signed-in authorized account, frozen DOCX uploads, intended job ad, Tacoma WA):

| # | Revision | Mode | Correlation | Result | Credits |
|---|---|---|---|---|---:|
| 1 | b99377e | Conservative | — | approved, but unsupported outcomes; export 409 `receipt_policy_state_missing` | 33 |
| 2 | d2b9578 | Imaginative | 22909c04 | withheld `source_facts_missing` | 33 |
| 3 | d2b9578 | Conservative | 845e8e80 | approved, 4 appended purpose clauses | 33 |
| 4 | 95db56e | Conservative | 104b7f7a | approved, dropped same-month qualifier | 33 |
| 5 | ae354e7 | Conservative | 64145e6c | withheld `no_op_repair` | 33 |
| 6 | ae354e7 | Conservative | a09a004a | withheld `source_completeness_uncertain` | 33 |
| 7 | 2b13fa8 | Conservative | eae427ea | withheld `source_completeness_uncertain` | 33 |
| 8 | 823c1b3 | Conservative | 3403b994 | withheld `source_completeness_uncertain` | 33 |
| 9 | 65ba882 | Conservative | d5143586 | approved, 3 appended clauses | 33 |
| 10 | 3dec9ad | Conservative | 56bc8c88 | withheld `unresolved_claims` | 33 |
| 11 | 9a9e574 | Conservative | b2198062 | withheld `source_facts_missing` | 33 |
| 12 | 4a90e44 | Conservative | 94c209ae | withheld `source_completeness_uncertain` | 33 |
| 13 | 3778d0c | Conservative | d0b19921 | withheld `source_completeness_uncertain` | 33 |
| 14 | b7fbecf | Conservative | c0b26422 | withheld `no_op_repair` | 33 |
| 15 | 555e26b | Conservative | ebab2e2f | approved, marker leak | 33 |
| 16 | 8606610 | Conservative | 40957281 | withheld `source_facts_missing` | 33 |
| 17 | 8606610 | Conservative | bff386a9 | withheld `source_facts_missing` | 33 |
| 18 | 8606610 | Conservative | 67c1af07 | **approved, faithful** | 33 |
| 19 | 8606610 | Imaginative | 12ecfaa4 | withheld `source_completeness_uncertain` | 33 |
| 20 | bc9e6df | Imaginative | 61bce5ef | withheld `source_facts_missing` | 33 |

**20 website submissions, 660 credits.** Balance moved 836 → 143. Every attempt is recorded; none was cherry-picked or rewritten.

## Next steps

1. Diagnose the reviewer's coverage inconsistency on the Imaginative input. The evidence is in the retained captures: the same head lines are `represented` in pass 1 and `missing` in pass 2 of one run. Do not weaken the gate; find why the reviewer's second pass loses them.
2. Once Imaginative delivers, run the five-per-mode stable batches and the second candidate, then the export tamper controls.
3. HasData usage was not measured this session; the integrated job-search check ran on the Conservative submissions and returned real matches.

## Uncommitted / private

- Pre-existing untracked `scripts/` is intentionally excluded.
- `captures/recovery-validation/` is git-ignored private evidence; retained, not committed.
- Main module checkout still has its pre-existing dirty `HANDOFF.md` and untracked `docs/Plans/`.

---

# Local preservation checkpoint — uncommitted, 2026-11-06

## Active branch and last commit

- Worktree `resumeDraft_fix`, branch `resumeDraft_fix`, HEAD `05e9c04`, nine commits ahead of `origin/resumeDraft_fix`. No commit or deployment was made for this checkpoint, at user request.

## Completed

- [x] Reproduced the three adjacent failures: two mixed bridge/outcome cases and an obsolete extracted-format assertion.
- [x] For an identified same-owner source bullet with matching figures, remove separable lexically ungrounded outcome clauses while retaining the candidate activity and permitted bridge. Added paired bridge-only/bridge-plus-outcome, grounded-outcome, ordering, correction-path, and PDF-loader controls.
- [x] Updated the extracted-format expectation only for the observed repairs; two other outcome clauses remain outside the lexical rule's scope.
- [x] Offline broad suite: 1508 passed, 13 skipped; loader suite: 17 passed, with two pytest configuration warnings. `git diff --check` passed.

## Remaining / next steps

1. Review the uncommitted diff, including the prior extracted-record and ownership changes. Do not commit or deploy without user authorization.
2. Mixed claims that are inseparable, share source vocabulary, lack reliable ownership, or involve transplanted/dropped figures are not deterministically resolved by this narrow rule. They may still be left to semantic review; the scripted false-approval test covers the separable case only. Investigate explicit fail-closed handling before claiming general bridge/outcome safety.
3. Live customer workflow, actual downloads, and representative additional documents remain unverified. The PDF fixture is digitally generated, not an OCR or complex-layout test; the DOCX fixtures remain frozen.

## Uncommitted / dirty files

- Modified: `source_preservation.py`, `tests/test_exported_outcome_wording_regression.py`, `tests/test_extracted_source_format.py`, `tests/test_loader_source_preservation_offline.py`, and this `HANDOFF.md`.
- Untracked: `tests/test_extracted_ownership_safety.py` (current offline safety regressions) and pre-existing `scripts/` (not examined or staged). All are intentionally unstaged per user request.

## Verification commands

Run from `/home/skystarved/Render_Dockers/Imagination_Resume_Module/.worktrees/resumeDraft_fix`:

```sh
OPENROUTER_API_KEY_1=x OPENROUTER_API_KEY_2=y IMAGINATOR_AUTH_TOKEN=z STAGE_COOLDOWN_SECONDS=0 /home/skystarved/Render_Dockers/Loader/.venv/bin/python -m pytest tests/test_loader_source_preservation_offline.py -q -o addopts='' -p no:cacheprovider
OPENROUTER_API_KEY_1=x OPENROUTER_API_KEY_2=y IMAGINATOR_AUTH_TOKEN=z STAGE_COOLDOWN_SECONDS=0 /home/skystarved/Render_Dockers/.venv/bin/python -m pytest tests/ -q -o addopts='' -p no:cacheprovider --ignore=tests/test_loader_source_preservation_offline.py
git diff --check
```

The main environment lacks PyMuPDF; the existing loader environment provides it but lacks pytest-asyncio, producing two configuration warnings for its focused run.

---

# Session 15 — deployed repairs; live acceptance remains open

## Verification commands

```
cd /home/skystarved/Render_Dockers/Imagination_Resume_Module/.worktrees/resumeDraft_fix
OPENROUTER_API_KEY_1=x OPENROUTER_API_KEY_2=y IMAGINATOR_AUTH_TOKEN=z STAGE_COOLDOWN_SECONDS=0 /home/skystarved/Render_Dockers/.venv/bin/python -m pytest tests/ --ignore=tests/test_loader_source_preservation_offline.py -q -o addopts='' -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/skystarved/Render_Dockers/Loader/.venv/bin/python -m pytest -p no:cacheprovider -o addopts='' -q tests/test_loader_source_preservation_offline.py
git diff --check
```

Results: 1565 passed, 13 skipped; real extractor suite 17 passed (two pytest config warnings). Full tests need separate environments because the main venv lacks fitz. Frontend isolated worktree `/tmp/cogito-export-policy`: 42 publication tests passed, one older real-FastAPI test excluded after failure (`fallback_output`); integration 6 passed. Full project TypeScript check had existing errors outside edited services.

## Branch and deployed revisions

- Imaginator: `resumeDraft_fix` deployed to `master` **d2b957881c6f11a5230a7643428e45f289b52aef**, Render `dep-dbna5lvavr4c73a947og` live. Published nondeploying branch `wip/two-mode-preservation-2026-11-07` also points at this revision before this handoff-only commit.
- Frontend and backend: isolated branch `fix/export-policy-state`, production `master` **4ea9ae5a02b018c4443879fbf3348295f7fc3f62**. Render frontend `dep-dbna91942hec73c8j3qg` and backend `dep-dbna91942hec73c8j35g` both live.
- [x] Actual frozen DOCX and generated synthetic PDF extract via Loader; role ownership, qualifier repair, missing skills, bridge and narrow outcome controls tested offline. Ambiguous semantic claims still require review; no universal fidelity claim.
- [x] Fixed missing v5 policy state in frontend export body, preserving receipt verification. Live pre-fix 409 was explicitly `receipt_policy_state_missing`. New release has not yet had an approved browser result with which to confirm downloads.
- [x] Imaginator/orchestrator `/health` 200. Render configured `CORRECTION_LIVE_CAPTURE` disabled; runtime value not independently checked. No capture was enabled this session.

## Direct attempts — all retained under `captures/recovery-validation/`

Frozen text source, direct `/analyze`; these are NOT website tests, platform credit consumption, or batch acceptance. Costs below are adapter estimates, can omit failed/cancelled provider calls.

| Revision | Mode | Correlation | Result | seconds | estimated USD |
|---|---|---|---|---:|---:|
| 7622305 | Conservative | diagcons-20f402920358 | approved | 21.6 | .0114649 |
| 7622305 | Imaginative | diagcons-c4498f3a7831 | approved | 27.9 | .01709095 |
| b99377e | Conservative | diagcons-ace33487af7b | withheld `source_completeness_uncertain` | 25.1 | .0135656 |
| b99377e | Imaginative | diagcons-4156e59aa7c8 | approved but false approval: unsupported outcome clauses and lost same-month/prior-year qualifier | 42.4 | .01450745 |
| d2b9578 | Conservative | diagcons-a64b81c72ab2 | withheld `timeout_initial_review`, cancelled provider usage unknown | 160.1 | .00854985 |
| d2b9578 | Imaginative | diagcons-61b1360e311a | withheld `source_completeness_uncertain` | 104.7 | .0173819 |

Six direct estimates total $0.08256065, not actual billed cost. Offline tests replay the b99377e false approval against d2b9578; this does not establish semantic completeness in new live output.

## Website submissions — signed-in authorized account

1. Conservative, b99377e + frontend 58a63d9: approved and displayed; fidelity inspection revealed unsupported purpose/outcomes. Export PDF and DOCX both 409 `receipt_policy_state_missing` because request omitted bound policy state. 33 credits, balance 836 → 803. Real integrated job matches were shown; HasData spend not determined.
2. Imaginative, d2b9578 + frontend 4ea9ae5: withheld `source_facts_missing` (Render correlation `22909c04-c799-47db-84fe-189ff0cdbb13`), no rewrite delivered and downloads correctly disabled. Receipt dialog says 33 credits; expected balance 770, confirm account refresh. No paid retries hidden. Both selected original frozen DOCX files, intended job ad, balanced style and Tacoma, WA through website controls. Different direct and website submissions are not interchangeable.

## Next steps

1. Preserve the existing authorized browser session. Diagnose `source_completeness_uncertain`, `source_facts_missing`, and the 101-second initial-review provider timeout using protected synthetic capture if needed (user authorized it), preflight and shut off via real redeploy afterward. Do not loosen the publication gate. Check health and account/provider billing before further traffic.
2. Run new Conservative and Imaginative website analyses on a stable revision and assess each source fact and bridge; compare actual browser PDF/DOCX downloads and visually inspect pages. Verify repaired v5 export with actual untampered and tampered content, policy and owner controls. The pre-fix old result used outdated frontend code and cannot validate the new release.
3. Run five recorded analyses per mode after stability, accounting for every retry/withhold; repeat failed batches after fixes and test a second candidate. Track HasData usage separately. Do not claim the finish line reached.
4. Main module checkout retains pre-existing dirty handoff and untracked plans. Imaginator worktree pre-existing `scripts/` remains untracked; private ignored captures are retained. `/tmp/cogito-export-policy/tsconfig.policy-check.json` is an untracked generated test config; do not commit it.

---

# Session 14 — partial repair; customer credentials unavailable

## Verification commands (offline; no provider traffic)

```
cd /home/skystarved/Render_Dockers/Imagination_Resume_Module/.worktrees/resumeDraft_fix
OPENROUTER_API_KEY_1=x OPENROUTER_API_KEY_2=y IMAGINATOR_AUTH_TOKEN=z STAGE_COOLDOWN_SECONDS=0 /home/skystarved/Render_Dockers/.venv/bin/python -m pytest tests/ -q -o addopts='' -p no:cacheprovider
git diff --check
```

Result: **1493 passed, 13 skipped**. The frozen DOCX/TXT/job hashes match `tests/fixtures/bridge_opportunity/FROZEN.md`.

## Active branch and last commit

- Worktree `resumeDraft_fix`, parent `7622305`; see `git log -1` for the WIP commit containing this checkpoint.
- Production Imaginator remains `76223056eb9cee9b9969db0a2f244b9f8f018db7` (`dep-dbn6rtui0phs7386udvg`, live). No production deploy or new capture activation occurred in session 14.
- Main module checkout is on `wip/correction-timeout-attribution` and has pre-existing dirty `HANDOFF.md` and untracked `docs/Plans/`; leave these alone except the session-14 header.

## Completed

- [x] User explicitly renewed authorization for live testing, replacing the old six-run/$1 cap; retain all attempt outcomes and observe billing boundaries.
- [x] Provider handling no longer turns reasoning into final output or `{}`; invalid or incomplete final content falls through the existing bounded model chain. Researcher retains explicit failure markers and rejects wrongly typed research arrays. Successful empty arrays remain distinct from failure.
- [x] Employer placeholder repair now requires unique, nonconflicting source identity, not record order. Missing source skill lines are inserted in a skills section, never over experience bullets, and require full re-review; review call-budget exhaustion withholds honestly.
- [x] Two live provider calls using the frozen source: direct OpenRouter Researcher request returned HTTP 200, `finish_reason=stop`, valid JSON, cost **$0.0016001865** reported by provider (3271 total tokens, 1029 reasoning tokens). The repaired local Researcher/adapter invocation returned six proposals and no fallback on first model; 3638 total tokens, estimated **$0.005544** by the adapter, actual billed cost not captured. These are provider-stage diagnostics, **not** two `/analyze` submissions. No HasData calls or credits consumed in this session.
- [x] Imaginator and orchestrator `/health` returned HTTP 200; browser reached `orchestrator-frontend-k95q.onrender.com` but was signed out. Custom `www.cogitometric.org` failed DNS from the browser. No live customer analyses, exports or downloads in this session.

## Remaining / next steps

1. Restore access to the authorized website account through an existing signed-in browser session or provide `COGITO_TEST_EMAIL` and `COGITO_TEST_PASSWORD` in the terminal environment; `COGITO_STORAGE_STATE` is also supported by the documented runner. Do not reset passwords or create accounts. `IMAGINATOR_AUTH_TOKEN` was also unavailable for direct `/analyze` diagnostics. Secrets must not be placed in the handoff.
2. Complete representative actual PDF/DOCX input preservation: paragraph-level experience bullets, unambiguous role ownership, qualifiers and per-claim bridge safety remain unproven. Current tests use plain-text extraction-shaped examples, not actual document reader output. Investigate skill restoration and duplicate source bullets across ownership before claiming fidelity.
3. Review the WIP diff and existing model fallback usage/cost telemetry, then deploy ordinary forward commits only after validation is possible. Fallback provider attempts currently may not be separately accounted by adapter stats; record actual provider activity separately.
4. Execute website journey both modes with frozen sources, then five recorded analyses per mode on the stable deployed revision plus second-candidate, tampered-content/policy/owner export controls, actual browser downloads, text extraction and visual PDF/DOCX inspection. Run one integrated job-search/enrichment check and account HasData usage separately.
5. Confirm final running revision, health, and capture disabled from the runtime (last known off from session 13; not independently rechecked in session 14). Do not claim success from health 200 alone.

## Uncommitted / private files

- Pre-existing untracked `scripts/` is intentionally excluded from the WIP commit; private ignored captures are not staged or removed.
- Main checkout's pre-existing dirty handoff and untracked plans remain outside this worktree commit.

---

# Two-Mode Skill Bridging — feature handoff (resumeDraft_fix)

## Current checkpoint — 2026-11-06 (combined repair, NOT yet deployed)

**Status: prepared, awaiting one bounded live validation. No deploy was triggered
for this checkpoint** (documentation-only pushes were explicitly stopped). The
combined repair is committed locally on `resumeDraft_fix`; it is NOT on `master`
and NOT live until the authorized validation is approved and a real deploy runs.

### A. The two captured `uncertain` inventory spans, classified

From protected capture `2dc8b2b7-d3c7-4b24-bf66-fff4018833e9` (35 claims:
25 required, 8 non_resume, 2 uncertain). Both uncertain spans are in the
supplement document:

| Span | Verdict | Why |
|---|---|---|
| `## Professional Summary\n` | **Formatting, no candidate claim** | A bare Markdown section heading. The SAME heading in the main document was `non_resume` on this run, and all 8 other headings in both documents were `non_resume`. It names a section, not a fact. |
| `SYNTHETIC-ONLY Operations analyst supporting stock planning and parts availability for a regional equipment service network.\n\n` | **Substantive candidate fact, genuinely ambiguous** | A role/summary statement about the candidate. Not formatting, not mixed. Its ownership/relevance was declined, so it must KEEP `uncertain` and still withhold. |

Neither is "mixed formatting and content". The narrow fix: in
`_validate_inventory`, when the model calls a BARE section heading `uncertain`,
reclassify only that word to `non_resume`. Detection uses a CLOSED vocabulary of
generic section-heading words (`_SECTION_HEADING_WORDS`); any candidate-specific
word (name, number, metric, verb) leaves the span exactly as classified. The
span is still partitioned and still required in `representations` — a review
that drops it is `malformed_output`. Replaying the exact captured inventory
changes exactly one item: `candidate-supplement.txt:0:24` uncertain→non_resume;
the substantive sentence stays uncertain. **This does not waive the substantive
span and does not turn all uncertain spans into structural content.**

### B. Exported unsupported wording and omitted qualifier — analysis finished

Imaginative run `de375f60` was APPROVED (`requires_review: false`, receipt
issued, export 200). Exact source → output (frozen fixture, verbatim):

| | Source | Approved output |
|---|---|---|
| Added outcome clause | `- SYNTHETIC-ONLY Reconciled 1,180 fuel invoices against depot dispensing records each quarter.` | `- Reconciled 1,180 fuel invoices quarterly against depot records, ensuring accurate fuel expense reporting.` |
| Added outcome clause | `- SYNTHETIC-ONLY Prepared the monthly cost summary reviewed by the regional manager.` | `- Prepared monthly cost summaries for regional management review, highlighting key cost drivers.` |
| Dropped qualifier | `- SYNTHETIC-ONLY Compared quarterly parts usage against the prior four quarters to set reorder points.` | `- Analyzed quarterly parts usage with historical comparison to set data-driven reorder points.` |

Both added clauses and the vague replacement reached the compiled PDF
(verified by extracting the PDF text streams from the retained `export.raw`).
**Why approval allowed them** (verified directly against the real bullets):
`_introduces_beyond` returns `[]` for all three — it detects only capitalised
proper nouns and unbacked figures, and these are lowercase outcome words with
no new figure. `_quote_shares_content` returns `True` for each because the
rewrite still shares content words with its source item; relatedness is not
fidelity. The reviewer marked the lines `supported`, so flag-then-repair never
engaged, and the completeness check asks whether each required item is
*represented*, not whether extra outcome claims were introduced or a qualifier
dropped. `tests/test_exported_outcome_wording_regression.py` pins this observed
case (10 tests). **No factual check was weakened and no remediation was
implemented** — extending the addition check to lowercase outcome clauses is a
separate, authorized change.

### C. Combined repair contents

- `source_correction.py`: `_SECTION_HEADING_WORDS`, `_is_section_heading()`, and
  the narrow reclassification in `_validate_inventory`.
- `tests/test_inventory_heading_classification.py` (23 tests, mutation-checked:
  disabling the correction fails 2).
- `tests/test_exported_outcome_wording_regression.py` (10 tests).
- Full suite: **1420 passed, 13 skipped** (1388 + 32 new).

### D. Protected live capture — turn-off arrangement (NOT yet applied)

`CORRECTION_LIVE_CAPTURE=1` and
`CORRECTION_LIVE_CAPTURE_DIR=/app/data/live-capture` are still set on the
imaginator (verified read-only from the running container's environ). Collection
is complete: both captures are retained privately at 0600 under
`~/.local/state/imaginator-capture/` and `captures/` (git-ignored). **After the
one bounded validation, turn capture off through the authorized config-change
process**: set `CORRECTION_LIVE_CAPTURE=0` via
`update_environment_variables` (or the Render dashboard), then trigger a REAL
deploy — a restart reuses the old container spec and will NOT apply the change.
Do not deploy documentation-only updates. Keep all diagnostic evidence private.

## Previous checkpoint — 2026-11-06 (recovery-after-repair fix)

- **Commit:** `450ff7d` on `resumeDraft_fix` and `master`; Imaginator deploy
  `dep-dbn1461srm7s73c5o8pg` (autoDeploy on `master`).
- **Root cause of the live withhold, established from a real capture, not from
  reading code.** Run `diagcons-28fc7c2a88e3` (Conservative, direct `/analyze`,
  frozen fixture) withheld as `incomplete_review`. The protected capture from
  that run records `phases: [source_inventory, initial_review, repair,
  post_patch_review]`, and the `post_patch_review` entry shows all **19** lines
  were asked about with no `must_judge` hint, raising `UnjudgedReview`. So the
  reviewer omitted exactly `L000008`–`L000013` (all six experience bullets) and
  the recovery branch returned `incomplete_review` **before any completion call
  was made**, solely because `state["repaired"]` was `True`.
- **The fix** removes `repaired` as a disqualifier. It was a proxy for "the
  draft may have changed", which is not the invariant: by the time a
  `post_patch_review` omits lines, the repaired markdown *is* the current
  document and the completion is rebuilt from it. The honest test is document
  identity, so the guard now compares a sha256 digest of the markdown the
  omission was recorded against (`_document_digest`) and refuses only when that
  document actually changed. Recovery is **still** at most once, still refuses a
  repeated identical omission, still honours the deadline, and still cannot
  approve anything on its own — it only adds judgements, which the unchanged
  completeness and attribution checks then decide.
- **Verification:** `1388 passed, 13 skipped`. Three recovery tests were rewritten
  or added and all are mutation-checked — restoring the old `repaired` guard
  fails `test_recovery_is_allowed_after_a_repair_of_the_same_document` and
  `test_a_post_repair_omission_is_completed_end_to_end`.
- **One authorized post-deploy diagnostic was run and did NOT exercise the fix.**
  Run `2dc8b2b7-d3c7-4b24-bf66-fff4018833e9` (record:
  `captures/recovery-validation/post-recovery-fix/`, capture copied in at 0600)
  withheld as `source_completeness_uncertain` with `phases: [source_inventory]`,
  `review_calls: 0`, `repair_calls: 0`. The inventory model classified 2 of 35
  supplement claims `uncertain`, and the early block at `source_correction.py`
  (the one that runs *before* the review loop) withheld immediately. Independent
  confirmation from the service log for the same correlation id:
  `required_source_claims: 25, source_claim_total: 35, reviews: 0, repairs: 0,
  elapsed_ms: 1016`. **This attempt neither confirms nor refutes the recovery
  fix** — the fixed code path was never reached.
- **There are two independent, intermittent gates on this fixture, not one.**
  (a) `source_inventory` returns an `uncertain` classification → withhold before
  any review; (b) a `post_patch_review` omits lines → the recovery path. Run
  `diagcons-28fc7c2a88e3` cleared (a) and hit (b); the post-deploy run hit (a) and
  never reached (b). Both are model-output variance on the same frozen input,
  which is also why the earlier checkpoint recorded delivery on "roughly half" of
  attempts. Neither gate is a wiring defect.
- **Not yet done (next bounded step):** one authorized Conservative diagnostic
  with capture still enabled, to observe whether a *second* `post_patch_review`
  now carries `must_judge`. The harness exists:
  `scripts/run_conservative_diagnostic.py` (one POST, no retry, records the
  response verbatim, prints only structural facts). Delivery is **not** claimed
  until that run is observed. Turn `CORRECTION_LIVE_CAPTURE` back off afterwards
  (requires a deploy, not a restart).
- **Content fidelity remains open and unchanged.** Imaginative `de375f60` added
  "ensuring accurate fuel expense reporting" and "highlighting key cost
  drivers" and dropped "against the prior four quarters"; `_introduces_beyond`
  returns `[]` for these because it only detects capitalised names and figures.
  Stage attribution is unknown (no drafter/editor snapshots). No per-request
  enrichment bypass exists on the multi-file route, and
  `JOB_DETAIL_ENRICHMENT_ENABLED` was **not** touched — that is a
  billing/provider config change needing explicit approval.

## Previous checkpoint — 2026-11-06

- **Restoration mapping fix deployed:** `4865c3b` on Imaginator master, Render
   deploy `dep-dbmr68cs728c739h33g0`. The engine now carries each selected
   source claim alongside its target line; the provider uses the corresponding
   indexed segment and fails closed on ambiguous mappings. Full post-restore
   review and existing factual checks remain unchanged. The previous failure's
   exact restore targets and answer remain unavailable, so its cause is unknown.
- **Verification:** focused restoration suite 130 passed; full offline suite
   1381 passed, 13 skipped. Command:
   `OPENROUTER_API_KEY_1=x OPENROUTER_API_KEY_2=y IMAGINATOR_AUTH_TOKEN=z /home/skystarved/Render_Dockers/.venv/bin/python -m pytest tests/ -q -o addopts='' -p no:cacheprovider --tb=short`.
- **Two authorized post-deploy attempts**, fixed fixture and no retry loop,
   saved separately under ignored `captures/restoration-mapping-validation/`:
   Imaginative `de375f60-a62c-48b7-abd0-be8701eb1fe8` exported after initial
   review with no restoration; its files contain unsupported outcome wording and
   omit a source qualifier, so export 200 is not factual acceptance. Conservative
   `3e1cbdb1-eeac-4e13-a162-eb0e367c58a9` withheld `source_facts_missing` after
   restore, with no receipt. Exact restore request/answer are not in the response;
   no claim can be made that the mapping fix resolved this run. Repeatability and
   fidelity remain open.
- Reported Imaginator costs for these two attempts were $0.0175941 and
   $0.01341595, respectively; these are not whole-pipeline costs. The
   Imaginative response reported 2 paid enrichment calls and 20 enrichment credits;
   Conservative reported zero. Account debit and total provider spend are unknown.
- The prior successful pair remains unchanged in ignored
   `captures/two-mode-2026-11-06/final_run/`; the frozen fixture hash tests still
   pass. Additional testing must preserve each attempt and requires its own
   bounded authorization.

- **Active branch:** `resumeDraft_fix`; investigation started from commit
   `119ab72`. The accompanying freeze/handoff commit is local only; do not push
   it to a deploy-linked branch without authorization.
- The user accepted the functional two-mode checkpoint but requires repeatable
   delivery on the fixed input. No further fixture changes or blind live retries.
- The exact five comparison inputs match `897d01b` and are frozen in
   `tests/fixtures/bridge_opportunity/FROZEN.md`; verify them with
   `python -m pytest tests/test_bridge_opportunity_frozen.py -q`.
- The final paired PDFs, DOCX files and saved responses are preserved privately
   under ignored `captures/two-mode-2026-11-06/`; its `INDEX.md` and historical
   event inventory distinguish the matching pairs from stale and missing data.
- **Corrections to earlier claims:** the 10-item check did not prove all facts
   and qualifiers survived. The saved Imaginative file changes `1,180 fuel
   invoices` to `1,180 fuel transactions` and `prior four quarters` to `trailing
   four-quarter averages`, and adds an unsupported purpose clause. Also, the
   `ed556783` completeness refusal followed initial review, not source inventory.
- The retained failure `896e4cc4` reached restore and withheld as
   `source_facts_missing`, but its target mapping and restore answer were not
   retained. An independently reproduced adapter defect assigns the first source
   item to every restore target; its role in that particular refusal is unknown.
   Do not attribute the failure to that defect or weaken checks based on it.
- No production fix or paid validation was performed in this investigation.
   Further production changes and bounded paid validation require explicit
   authorization. Earlier attempt outputs were overwritten; retain the gaps.
- **Intentionally untracked/ignored:** `captures/two-mode-2026-11-06/` contains
   the preserved responses and synthetic files, including signed receipts; it
   must remain private and must not be committed. No source fixture changes were
   made.
- **Next steps:** (1) inspect preserved files and request any available earlier
   captures; (2) obtain authorization before changing production or incurring
   model costs; (3) if authorized, investigate the restoration mapping with
   controlled source/target pairs and re-review, without claiming it caused
   run `896e4cc4`; (4) run a bounded fixed-input comparison only with explicit
   cost approval and persist every request outcome separately.

**Branch:** `resumeDraft_fix`, rebased onto `origin/master` (`2bcbb4e`) so the
correction/publication/receipt system that Phases C and D extend is present.
Previously this branch was cut from a stale local `master` (`b23bba4`) that
predated that system by 103 commits; that was a baseline mistake, not a change
of scope.

**Policy:** `docs/Plans/CogitoMetric_Official_Skill_Bridging_Plan_v1.md` (v1.1)
**Spec:** `docs/Plans/Two_Mode_Skill_Bridging_Implementation_Spec.md`

## Status

- [x] **Baseline integration** — rebased onto published `origin/master`
      (`2bcbb4e`). All correction/publication/receipt modules are now present.
      `wip/correction-timeout-attribution` was **not** modified or reset; its
      contents reached this branch via the published `origin/master` instead.
- [x] Phase A — two-mode permission contract (`skill_inference.py`,
      `models.py`, `orchestrator.py`, pipeline entry, `/analyze`,
      32 offline tests in `tests/test_skill_inference_mode.py`)
- [x] Phase C partial — **prompt-level enforcement** for both writers,
      10 offline tests in `tests/test_skill_inference_prompts.py`
- [x] Phase B — **typed, source-anchored bridge records** (`skill_bridge.py`),
      Researcher proposal schema, correction phrase exemption, both leak paths,
      41 record tests + 10 paired end-to-end tests
- [x] Phase D.4 — **policy state bound into the approval receipt** (v5),
      `_surviving_bridges()` filter, frontend transport, 25 binding tests
- [x] Phase C.7 — **accept / narrow / remove lifecycle**
      (`skill_bridge_decisions.py`), 31 tests
- [x] Phase C.8 — **redaction-safe bridge anchors** (real defect fixed, see below)
- [x] **Reviewer decisions wired into the real engine** — `bridge_decisions`
      through the adapter, narrow re-reviewed, remove not revivable, 18
      engine-level tests
- [x] **Two disclosure leaks found and fixed** (see below)
- [x] Phase E — **analysis cache identity includes mode + policy version**,
      14 tests; live cache can no longer serve an Imaginative result to a
      Conservative request
- [x] **Deployed to production and verified live** (see "Live verification")
- [x] **Capture-guard false refusal fixed** — `git check-ignore` returns 1 for a
      non-existent path under a directory-only rule, so the guard refused the
      FIRST capture in a clean checkout
- [x] **Phrase-scoped spans for a label embedded in prose** (spec C.6 rule 3) —
      14 tests in `tests/test_bridge_phrase_spans.py` plus two tightened
      end-to-end tests; see "Phase C.6 rule 3" below
- [x] **Preview/export parity and real-file content assertions** (Phase D
      remainder) — 9 real-render tests in `tests/test_bridge_render_parity.py`,
      7 export-semantics tests in `tests/test_export_generation_semantics.py`,
      13 frontend transport tests in
      `Login_Landing_page/src/services/__tests__/bridgeExportPayload.test.ts`;
      see "Phase D remainder" below

## Live verification (2026-11-05)

Deployed `imaginator-resume-cowriter` to master and probed the running service.

| Probe | Result |
|---|---|
| health endpoint | **PASS** 200 healthy |
| conservative accepted, no bridges | **PASS** |
| invalid mode refused | **PASS** 422 (`balanced` never silently defaults) |
| omitted mode → imaginative | **PASS** |
| export without receipt refused | **PASS** 409 |
| mode visible in orchestrator logs | **PASS** `Skill inference mode: imaginative (policy imaginative-v1)` |

**12/12 probes pass** against `627ba6c` (final). The tenth probe — a receipted
approval — reports SKIP; see "Live receipted approval — blocked, and why".

### A live defect this probing found, and fixed

The first full-fixture run returned a withheld result with
`skill_inference_mode: None`. Tracing it: `_mode_from_payload` short-circuited on
`None` instead of resolving it, but an **omitted** mode is not an unknown policy —
`normalize_mode(None)` resolves to `imaginative` / `imaginative-v1` and the
pipeline runs exactly that. So the difference reflected how the request was
*spelled*, not what happened, and the loss fell on the common path: the website
never sends the field, so **every website refusal carried no policy attribution**,
while a caller who sent it explicitly was told. Fixed; verified live after
redeploy — the same website-shaped request now reports
`imaginative` / `imaginative-v1`. 14 tests in
`tests/test_withheld_policy_attribution.py`, mutation-checked.


Not yet observed live: a **receipted** approval (`publication_receipt` +
`skill_inference_policy_state`). See "Live receipted approval — blocked, and
why" below; the offline suite proves receipt v5 issuance and tamper refusal.

### Live receipted approval — blocked, and why

Attempted six fixture shapes against `627ba6c`. None produced a receipt. The
blocker is not the bridging feature:

| Fixture shape | Withhold reason | Words |
|---|---|---|
| small (60 words) | `below_min_words` | 0 |
| 162-word, two employers | `source_facts_missing` | 0 |
| verbatim source mirror | `source_facts_missing` | 0 |
| structured line-per-fact | `below_min_words` | 0 |
| matched job ad, 3+3 bullets | `below_min_words` / `source_facts_missing` / `provider_error` (retry) | 0 |

`words: 0` on every withheld result is the withholding rule working: an
unapproved draft is not returned. Two structural gates pull in opposite
directions on a *synthetic* fixture:

* the publication gate requires **80 words**, so a short fixture cannot approve;
* the reviewer's source inventory requires every required source fact to be
  **represented**, so a longer draft has more chances to omit one.

A real candidate resume clears both because it is long *and* every fact in it is
its own activity. Neither gate was loosened to make this probe go green —
weakening a publication gate to obtain a passing acceptance test is exactly the
trade the spec forbids. The receipt half of the probe therefore still reports
SKIP, and the probe now checks **both** halves (approval exports real bytes;
tamper/dropped-state/appended-content are refused) so it will exercise fully the
first time a real approval is presented.

**What remains unproven live:** that a v5 receipt is issued and accepted end to
end on the deployed service. Offline, that path is covered by 25 receipt-binding
tests and 9 real-render parity tests.

### Test totals

```
python -m pytest tests/ -q -o addopts="" -p no:cacheprovider
# 1332 passed, 13 skipped, 0 failed
```

Two environment notes for reproducing that number outside the main checkout:
the two `test_correction_route_budget` tests read `../Login_Landing_page`, so a
sibling symlink is required; and the capture guard is now correct about
non-existent directories, so `test_live_capture_controls` passes without one.

The real-render suites (`test_bridge_render_parity`,
`test_export_isolation_and_parity`, `test_renderer_no_defaults`) need
`pdflatex` and `pdftotext`. They SKIP with a reason when the toolchain is
missing rather than passing on mocked bytes.

## Phase D remainder — parity, real files, export semantics

The spec's exit criterion for D is that *"preview and files reflect the same
reviewed content; only permitted skill labels are delivered"*. Three suites were
extended rather than replaced, because each was already authoritative for one
part of that claim and silent about another.

| Suite | Proves | Previously silent about |
|---|---|---|
| `test_bridge_render_parity.py` (9) | a reviewed label reaches real PDF **and** DOCX bytes; conservative excludes it; a narrowed bridge renders its replacement | the existing export suite never mentioned inference at all |
| `test_export_generation_semantics.py` (7) | export reaches no pipeline stage; builder failure raises instead of a success-shaped 200 | nothing pinned either |
| `bridgeExportPayload.test.ts` (13) | the browser transports policy state verbatim and never authors it; post-analysis edits cannot rewrite an approval | no frontend coverage of the new fields |

**Parity is asserted over the preview string, not the payload.** Every skill
label in the approved markdown must appear in both files; neither file may carry
a label the preview omitted; and the approved bullet must still render *under*
the approved employer (string parity is not attribution parity).

Per the spec, this is "preserved approved substantive content and ownership, not
identical heading counts or pagination". No byte-identical-across-runs claim is
made — layout engines and timestamps differ.

### A vacuous assertion that was caught and fixed

The no-generation tripwire initially patched only *module-level* functions. Stage
entry points are **class methods** — `PipelineOrchestrator.run_pipeline`,
`Researcher.analyze`, `Drafter.draft`, `StarEditor.polish` — so the test would
have asserted that nothing was called among a set of things the pipeline never
exposes that way. It now patches class methods too, records what it armed, and
**fails if nothing was armed**, so a future refactor cannot quietly restore the
vacuous version. 10 real entry points are armed today.

### Not asserted, deliberately

- **Any credit-debit property.** `export_resume` takes no credit dependency; the
  signature is asserted to stay `(payload, request, api_key)` so an added
  dependency requires a deliberate review, but claiming the debit path is
  exercised would be a claim without evidence.
- **Real-model parity.** Everything here is offline. A live run needs an
  allowance and a fixture the reviewer fully supports.
- **Browser save.** HTTP 200 is not a saved file; run 10 already showed the
  download never fired on the site. That remains a separate criterion.

## Phase C.6 rule 3 — a permitted label embedded in prose

Phase B admitted a permitted label only when the **whole** supported span was that
label (`- trend analysis`). A label used inside prose — `- Delivered trend analysis
for stakeholders` — was still reported unsupported: one whole-line verdict cannot
say "this phrase is permitted, judge the rest".

The adapter now splits such a line into up to three gapless spans: the label
(carrying its anchor's citation) and the text either side (kept at the reviewer's
verdict). The strict validator already demands a non-overlapping partition, which
this satisfies by construction.

```
- Delivered trend analysis for stakeholders
  [(0,12,'uncertain'), (12,26,'supported'), (26,43,'uncertain')]
```

The exemption stays narrow, and each guard has a paired test:

| Guard | Why |
|---|---|
| only the exact label phrase carries support | a label must not approve a sentence |
| the reviewer must still name a segment | the exemption is a floor, never an approval |
| the label must share **no** word with the anchor | otherwise a relied-upon overlap already explains it, and a split would invent support |
| residual spans keep `uncertain` | an unsupported tool or outcome in the same sentence still fails |

So `- Used Python for trend analysis and increased revenue 20%` now approves
**only** `trend analysis`; `used Python` and `increased revenue 20%` remain
`uncertain` and still block publication. Two older tests asserted the coarser
line-scoped rule ("nothing on that line may be supported") and are tightened to
assert the precise one.

**Deliberate limit.** A line with *partial* lexical overlap is not split. The
reviewer's verdict already applies there, and inventing a partition where an
overlap explains the label would be the adapter granting support.


## Two disclosure defects found and fixed

Both were in the *outgoing provider payload*, not the response.

| # | Defect | Fix |
|---|---|---|
| 1 | `correction_protocol` echoed the bridge record's **raw** `evidence_quote` into the visible `bridges` block. With PII inside the quoted activity, the raw text went to the provider while the redacted source sat right beside it. | `_redacted_quote_at()` derives the offer from the **redacted** source at the recorded coordinates. A bridge whose coordinates cannot be honoured is **not offered** (fails closed rather than leaking or fabricating). |
| 2 | `source_correction` passed `permitted_bridges` as a **visible** key, so the raw record — including its model-authored `rationale`, which the redactor never saw — travelled in the request body. | Moved to `_permitted_bridges`; the provider's existing `not key.startswith("_")` filter strips it. |

Verified on the serialized wire payload (underscore keys excluded, exactly as the
provider does), for PII **inside the quote**, **inside the rationale**, and
**outside the quote**.

A pre-existing test also caught a regression: `_validate_review` touched a
malformed `response` before type-checking it, turning `malformed_output` into a
generic `provider_error`. Fixed by checking `type(response) is not dict` first.

## Decision lifecycle through the engine

| Decision | Engine behaviour |
|---|---|
| **accept** | label stands; approval path unchanged |
| **narrow** | label rewritten, then `requires_review` forces a **complete re-review**. The updated record (same id, same anchor, new wording) is offered to that review, so the replacement is judged against the **original** candidate evidence. Structural validation alone never authorises it. |
| **remove** | label dropped from the offered set; per-bridge history prevents a later review from reviving it |
| **omitted / unknown** | no decision = no change; an unknown decision word or unknown bridge id is **malformed**, not approval |

## Structural bounds relaxed per review

Single words (`reporting`), acronyms (`SQL`, `ARIMA`) and longer phrases are **no
longer** refused on shape alone — that heuristic discarded legitimate skills. Only
claim shapes are rejected: sentence punctuation, category colons, metric + unit,
proficiency vocabulary. Whether a *specific* label is justified is the reviewer's
judgement against the candidate evidence, not a regex.

### Verification

```
python -m pytest tests/ -q -o addopts="" -p no:cacheprovider
# 1285 passed, 13 skipped, 3 failed  (all 3 pre-existing worktree .git-ignore artifacts)

python -m pytest tests/test_bridge_engine_lifecycle.py -q      # 18
python -m pytest tests/test_source_correction.py \
                 tests/test_bridge_engine_lifecycle.py \
                 tests/test_bridge_decisions_and_redaction.py -q   # 135
```


## Phase C.8 — a real defect that was found and fixed

Correction receives **length-preservingly redacted** sources; a bridge keeps the
**raw** quote it was validated against. When PII sat inside the quoted activity
the two differed character-for-character, and the exact-equality anchor check
**silently dropped a legitimate bridge** — a failure indistinguishable from
forgery:

```
recorded:  "Compared sales for jordan@example.com and identified trends"
slice:     "Compared sales for [email removed]███ and identified trends"
```

`_matches_redacted_quote()` now compares the recorded quote's own redaction at the
same width. A genuine anchor survives; a fabricated one still fails, because its
redaction will not match either. The redactor is length-preserving by contract, so
offsets need no adjustment — and if that contract ever breaks, the helper returns
`False` rather than guessing.

## Phase C.7 — decision lifecycle

| Decision | Applied when | Refused when |
|---|---|---|
| **accept** | the label is a standalone item | the label only appears inside prose (`label_not_standalone`) |
| **narrow** | standalone; replacement is a valid general label | replacement is ARIMA/Python/AWS/a metric/a proficiency, is identical, or is missing |
| **remove** | every occurrence is standalone | any occurrence is embedded in a sentence (`embedded_label_requires_repair`) |

Two guards are deliberate:

1. **Narrowing cannot escalate** — the replacement goes through the *same*
   validator as an original proposal. There is one definition of "general label".
2. **Partial removal is refused** — deleting the standalone item while the label
   survives in prose reports success without removing anything.

A refused decision leaves the text byte-identical. Reason codes are a fixed set;
caller free text is dropped rather than logged.

Also tightened `_is_general_label()`: sentence punctuation anywhere (a trailing
period slipped through the previous "punctuation + whitespace" pattern) and a
2–5 word bound. Verified balanced — 10/10 legitimate labels accepted, 8/8
prohibited shapes rejected.

### Verification

```
python -m pytest tests/ -q -o addopts="" -p no:cacheprovider
# 1256 passed, 13 skipped, 3 failed  (all 3 pre-existing worktree .git-ignore artifacts)

python -m pytest tests/test_bridge_decisions_and_redaction.py -q   # 31
python -m pytest tests/test_skill_bridge_records.py \
                 tests/test_skill_bridge_end_to_end.py \
                 tests/test_bridge_decisions_and_redaction.py -q   # 82
```


## Phase D.4 — what the receipt now binds

The gap closed: the receipt bound the *rendered document* but said nothing about
the inference mode or the surviving bridges. Mode/bridges are not part of the
rendered object, so a source-only approval could be presented as an approval of
inferred skills.

| Situation | Result |
|---|---|
| Source-only approval (no policy state) | **v4**, unchanged; existing receipts still verify |
| Policy state supplied | **v5** + `pol_fp` fingerprint |
| Export declares policy state, receipt is v4 | `receipt_policy_unbound` — a v4 token grants no bridge privilege, ever |
| Receipt is v5, export omits the state | `receipt_policy_state_missing` |
| Both present, any bound field altered | `receipt_policy_mismatch` |

Bound fields: mode, policy version, and per surviving bridge the `id`, `skill`,
`origin`, `source_id` and `policy_version`. Canonicalisation sorts by
`(id, skill)`, so reordering is not a change; `rationale`/`evidence_quote` are
**not** bound (they are not semantic identity and binding them would make two
identical documents differ).

`_surviving_bridges()` keeps only labels that actually reached the approved
markdown, using word-bounded matching — an unused or removed proposal is never
signed, and the filter also works in Conservative mode (where it yields nothing).

### Verification

```
python -m pytest tests/ -q -o addopts="" -p no:cacheprovider
# 1225 passed, 13 skipped, 3 failed  (all 3 pre-existing worktree .git-ignore artifacts)

python -m pytest tests/test_bridge_receipt_binding.py -q      # 25
python -m pytest tests/test_skill_bridge_records.py \
                 tests/test_skill_bridge_end_to_end.py -q     # 51
cd ../Login_Landing_page && npx tsc --noEmit                  # clean
```


## Spec-agent integration SHA (recorded per task item 5)

`wip/correction-timeout-attribution` is **3 commits ahead** of
`origin/master` (`2bcbb4e`) and those commits are **not** in this branch:

| SHA | Subject | In this branch |
|---|---|---|
| `ae2ed7b` | fix(imaginator_flow): resolve undefined analysis_result in final editor | **absent** |
| `483074d` | fix(multipart): refuse to publish when required generation fell back | **absent** |
| `30d4086` | fix(multipart): request-local generation outcomes; require explicit success | **absent** |

That branch is **not pushed to origin** and was **not modified** or reset by this
work. Its newer fallback/request-state/cache fixes are therefore **not** part of
the SHAs tested below. A future integration should merge those three commits and
re-run this suite; the locally measured baseline is `origin/master` = `2bcbb4e`.

## Phase B — what a bridge is, and what it is not

A bridge is a **proposal with an anchor**, not a fact:

* the server assigns the ID and the `origin` (`researcher_inference`) *after*
  validation, so a client-supplied origin tag grants nothing;
* `evidence_quote` must appear **verbatim** in one immutable candidate document
  (the multi-file join marker may not be spanned);
* Conservative produces **no records at all**, even from perfect proposals;
* taxonomy refs are optional and may only name refs the caller supplied.

The correction engine admits a permitted label with **zero lexical overlap**, but
only for the label phrase, and only when the entire supported span is that label
and its anchor matches a reviewed source segment (containment plus an exact-text
check, so forged offsets confer nothing). A mixed line still fails:

| Draft line | Outcome |
|---|---|
| `- trend analysis` (imaginative, anchored) | supported |
| `- trend analysis` (conservative) | uncertain — no authorisation exists |
| `- Used Python for trend analysis and increased revenue 20%` | not supported — the tool and outcome are unsupported |

Verified before/after on identical input:

```
imaginative   bridges=1 verdict=supported
conservative  bridges=0 verdict=uncertain
```

## Leak paths

| Path | Status |
|---|---|
| `implied_skills`/`domain_vocab` merged into `aggregate_skills` | **closed** — advisory only; new `resolved_skills` is candidate skills + permitted bridges (imaginative) or candidate skills alone (conservative) |
| `imaginator_flow.run_final_editor_async` raw research injection | **already closed** by the published evidence boundary; verified it never interpolates `research_data` |
| `preserve_skills_block()` deleting an inferred label | **closed** — permitted labels are retained; keyword passed only when bridges exist |

**Resolved by Phase D.4:** mode, policy version and the surviving bridge set are
now bound into a v5 receipt, and a v4 receipt cannot authorise an inferred skill.

## Current behaviour

`skill_inference_mode` is `imaginative` (default) or `conservative`; unknown
values fail validation. Conservative performs **zero** Researcher calls and
carries no default qualifications.

**The permission boundary is enforced in the prompts, not the temperature.**
`creativity_mode` remains an independent prose-style dial, and that separation is
only safe because both writers now state the policy in their prompt text:

| | Conservative | Imaginative |
|---|---|---|
| Drafter system prompt | `SKILL-INFERENCE POLICY: CONSERVATIVE` — no skill inference at all | `IMAGINATIVE (MANDATORY LIMITS)` — general transferable-skill label only, with explicit limits and a worked example |
| Sparse-input note | source-bounded; a shorter bullet is correct | may add a general label; no tool/algorithm/product/metric/proficiency |
| StarEditor | receives **no** `domain_vocab`; forbids inference and proficiency upgrades | receives vocabulary scoped to relabelling; forbids expert/advanced/proven/years/leadership on an inferred label |

The retired instruction — "EXPAND them into full bullets by inferring standard
industry tasks … focus on HOW the work was likely done" — is **gone**. It applied
in both modes, so a mode-only test could pass while a run still fabricated
duties.

### Verification (this branch)

```
python -m pytest tests/ -q -o addopts="" -p no:cacheprovider
# 1149 passed, 13 skipped, 3 failed

python -m pytest tests/test_skill_inference_mode.py tests/test_skill_inference_prompts.py -q
# 42 passed
```

The 3 failures are **pre-existing** and unrelated: `test_correction_route_budget`
(2) and `test_live_capture_controls` (1) — verified by stashing this branch's
changes and reproducing them on the clean rebased base. They concern a
`.git`-ignore check for a capture directory inside a git *worktree*, not skill
inference.

Mutation-verified: reintroducing the retired sparse-input instruction fails
`test_sparse_input_no_longer_authorizes_invention_in_either_mode`.

No deployment, no paid run, and no production mutation was made.

## Remaining risk in the current state

Prompt text is necessary but not sufficient. A model can still ignore it, and
nothing yet reviews an inferred label semantically:

1. **No bridge records exist yet** (Phase B). The writers are told what they may
do, but no typed, source-anchored proposal is supplied or validated, so
   Imaginative has permission without evidence discipline.
2. **The correction reviewer judges whole lines** and requires lexical overlap
   with cited source segments. A legitimate zero-overlap label such as
   "trend analysis" would therefore be reported unsupported and repaired or
   withheld (spec C.6).
3. **`source_preservation.preserve_skills_block()`** replaces a skills block it
   cannot match to source phrases — it can delete an inferred label before review
   sees it (spec C.5).

So the prompt work closes the "temperature as policy" gap; it does not yet make
an inferred skill survive to delivery. Phases B and C-remainder do that.

---

# Handoff — run-10 evidence reconciled; semantic work list added; fixtures awaiting review

**2026-11-03, session 11** (supersedes the session-10 header below).

- **Tests:** **1037 passed, 3 skipped** (was 996 — 41 new, all offline).
- **Tag:** `known-working-single-file-2026-11-02` → `73cf570` (kept; **unchanged and not
  moved**).
- **Production code:** **none modified this session.** `git status` shows new files only;
  `app.py`, `source_correction.py`, `source_preservation.py`, `stages/`,
  `correction_*` are untouched.
- **Deployed:** `47c4254` — **no new deployment made, none needed, none authorised.**
- **Capture:** still **OFF** (`capture=0`); remote capture directory **empty**.
- **Credits:** 935. **No analysis submitted this session.** Allowance remains exhausted.
- **Run-10 evidence:** unchanged — capture, both source documents, approved response and
  original hashes all preserved byte-for-byte.

## Two earlier conclusions were wrong, and are corrected

### 1. "No `/resume/export` request left the browser" — false

An instrumentation miss reported as a product failure. The browser listener watched
`/resume/export` while the real request goes to the orchestrator proxy
`/api/ml/imagination/export`. Server logs settle it:

| Stage | Evidence |
|---|---|
| Button | `Download PDF (LaTeX)` / `Download DOCX` clicked 16:26:12Z–16:27:26Z |
| Request | orchestrator `POST /api/ml/imagination/export` at **16:26:44.397Z** and **16:27:26.854Z** |
| Proxy → Imaginator | `[RESUME_EXPORT] Calling: …/resume/export` at 16:26:44.405Z / 16:27:26.861Z |
| Response | Imaginator `POST /resume/export **200 OK**` at 16:26:46.414Z / 16:27:28.402Z; `resume.export.complete` ×2 |
| Bytes | `pdf_base64` + `docx_base64` returned (`pdfTeX` ran) |
| Save | **Yes** — `resume.pdf` 128,226 B `pdfTeX-1.40.26`/`moderncv` (CreationDate 16:27:27Z); `Machine_Learning_EngineerTest.docx` 37,816 B `python-docx` |

**All 23 of 23** approved content units appear in the PDF. So the **receipt was accepted by
export** — for the *embellished* content, which is correct behaviour: the receipt binds
approved content, not truthful content. Export integrity is therefore **not** the
remaining problem.

Open question, not resolved by assumption: the exported filename
`Machine_Learning_EngineerTest` does not match the "Senior Electrician" title typed into
that form. Content provenance is not in doubt; job-title provenance is unexplained.

### 2. "Run 10's provider cost is unrecoverable" — wrong for the correction stage

The preserved local capture retains each response body with the provider's `usage.cost`:

| Phase | Tokens | `usage.cost` |
|---|---|---|
| `source_inventory` | 2,121 | $0.000704088 |
| `review` | 5,076 | $0.002208888 |
| **total** | 7,197 | **$0.002912976** |

**Correction stage only.** Researcher/drafter/star-editor usage is not in this capture, so
whole-pipeline provider cost is higher and still **unmeasured**. Not $0.00, and not
inferable from credits.

## The six appended clauses are not one category

`docs/live-runs/RUN_10_CLAIM_TABLE.md` judges each clause against the complete source and
its attribution. Result: **five unsupported additions, one legitimate paraphrase.**

| Clause | Verdict |
|---|---|
| `ensuring compliance with NFPA 70E standards` | unsupported addition — source ties NFPA 70E to training and the certificate, not to installation |
| `improving workforce safety and code adherence` | unsupported addition — no measured outcome |
| `increasing installation efficiency and quality control` | unsupported addition — different outcome from the sourced 18% |
| `ensuring continuous operations` | unsupported addition — asserts an uptime guarantee no source describes |
| `enhancing system reliability` | unsupported addition — reliability ≠ the sourced downtime figure |
| `gaining hands‑on electrical construction experience` | **supported inference — keep it.** Adds no qualification; characterises sourced field work |

Deleting all six would have removed a correct sentence. That is why they are judged
individually.

## What was added (offline only)

| Path | Purpose |
|---|---|
| `appended_clauses.py` | **[tier-b-worklist]** reports, per supported line, trailing participial clauses that share no content word with the evidence **cited for that line**. Produces a work list for semantic review. **Does not gate publication, does not decide fabrication, not wired into the pipeline.** |
| `tests/test_appended_clause_worklist.py` | 20 paired tests using verbatim run-10 text: faithful paraphrase approves, cross-document synthesis approves, 5 of 6 appends detected, wrong-employer transplant rejected, reviewer-said-supported shown not to be verification |
| `tests/fixtures/realistic_candidate/` | **[realistic-fixture]** candidate pack — 2 source docs, 22-fact ownership ledger, 6 named traps. **Awaiting review; not frozen, not hashed for approval, not submitted** |
| `tests/test_realistic_candidate_fixture.py` | 21 structural self-checks on that pack |

**Why the measured basis matters.** Against the whole cited *document*, only **1 of 6**
appends is detectable (`compliance`/`standard` occur elsewhere in `document-0`). Against
the evidence cited *for that line*, **5 of 5** are. The document-scoped comparison is
precisely what hid the defect. Mutation-tested: restoring document scope fails 8 of the
new tests.

**Vocabulary is not the test.** A "words absent from the source" rule would fire on 1 of 6
and would delete the one clause that is correct. The work list reports questions; Tier B
answers them.

## Remaining work after this branch

Phases A–E and both remainders are complete. What is left cannot be closed by
more offline work:

1. **A live receipted approval** (blocked, see above). Needs a real candidate
   document — long enough for the 80-word gate, with every required source fact
   represented. Do **not** loosen a gate to obtain it. When it arrives, the probe
   already exercises both halves and no code change is needed.
2. **A real browser save.** Run 10 recorded three attempts with no `/resume/export`
   request leaving the browser. HTTP 200 is proven; a saved file is not.
3. **Semantic inference quality.** Every test here is a scripted reviewer. Whether
   an inferred label is *apt* for a given candidate is a human judgement;
   `tests/fixtures/realistic_candidate/` is the vehicle and needs review before it
   is frozen and hashed.
4. **The three spec-agent commits** (`ae2ed7b`, `483074d`, `30d4086` on
   `wip/correction-timeout-attribution`) are still unmerged. The tested baseline is
   `origin/master`; merging those commits requires re-running this suite.
5. **`Login_Landing_page` is on branch `employment-type-search`, not `main`.** The
   frontend tests are committed there and that branch is pushed. It is unrelated
   work in the same checkout; merging it to `main` is a separate decision, so it
   was left alone.
6. **A pre-existing frontend failure** unrelated to this work:
   `src/services/__tests__/resumeMarkdownParser.run3live.test.ts` reads
   `/tmp/export_check/run3_markdown.txt`, a local scratch file that does not exist.
   40 of 41 tests in the services folder pass; this one fails identically with this
   branch's files removed.

## Verification commands

```
cd Imagination_Resume_Module   # on resumeDraft_fix
OPENROUTER_API_KEY_1=x OPENROUTER_API_KEY_2=y IMAGINATOR_AUTH_TOKEN=z \
  /home/skystarved/Render_Dockers/.venv/bin/python -m pytest tests/ -q \
  -o addopts="" -p no:cacheprovider      # 1346 passed, 13 skipped

cd ../Login_Landing_page
NODE_OPTIONS=--max-old-space-size=1024 node node_modules/jest/bin/jest.js \
  -c jest.unit.config.cjs src/services/__tests__/bridgeExportPayload.test.ts  # 13

# Live acceptance (needs IMAGINATOR_AUTH_TOKEN from final_env_vars.json)
python3 tools/live_bridge_probe.py       # 12/12, receipt half SKIP (see above)
```

---

# Session log


   should withhold, trigger a targeted re-review, or only annotate. Any of these changes
   prompt or reviewer behaviour, which per SPEC.v2 requires **bounded real-model evidence
   after** the offline tests — **not** an unapproved production retry.
2. **Review the realistic candidate pack** (`tests/fixtures/realistic_candidate/README.md`).
   Freeze and hash only after approval; record hashes there.
3. **Explain the export filename** mismatch (`Machine_Learning_EngineerTest` vs "Senior
   Electrician"). Cosmetic provenance question; no spend required.
4. **Whole-pipeline provider cost** for run 10 remains unmeasured — the correction measure
   is not the total.
5. Real-model repair recovery still unexercised (`repair_calls = 0` in every run).

**No submission is authorised.** The 1-analysis allowance is exhausted (credits 935). No
production mutation and no deployment was made or is authorised by these changes.

## Verification commands

```
source .venv/bin/activate && cd Imagination_Resume_Module
python -m pytest tests/ -q -o addopts="" -p no:cacheprovider            # 1037 passed, 3 skipped
python -m pytest tests/test_appended_clause_worklist.py -q -o addopts=""    # 20
python -m pytest tests/test_realistic_candidate_fixture.py -q -o addopts="" # 21
python -m pytest tests/test_route_two_documents.py -q -o addopts=""         # 22, real /analyze
cd ../Login_Landing_page
node node_modules/jest/bin/jest.js -c jest.integration.config.js \
  backend/tests/integration/multiFileRoute.test.js                      # 6 passed
```

---

# Session 10 — two-file website run executed and approved; one defect class remains

**2026-11-03, session 10** (supersedes the session-9 header below).

- **Tests:** 996 passed, 3 skipped (`python -m pytest tests/ -q -o addopts="" -p no:cacheprovider`).
- **Tag:** `known-working-single-file-2026-11-02` → `73cf570` (kept; unchanged, not moved).
- **Deployed:** `47c4254` on `master` (verified by deploy SHA and by reading the fix back
  out of the deployed revision).
- **Capture:** **OFF** (`CORRECTION_LIVE_CAPTURE=0`, redeployed, confirmed `capture=0`);
  remote capture directory **empty**.
- **Credits:** 968 → **935** (33, one analysis). Provider dollar cost for run 10 is
  **unmeasured**, not zero — see below.

## The authorised two-file run was executed: run 10

| | |
|---|---|
| Correlation ID | `850ee3ea-ca2e-4210-92b4-0e57ed5149eb` |
| Route | website → orchestrator `/api/analysis/multi-file?stream=1` → Imaginator `/analyze` |
| Input | `synthetic-single.docx` + `synthetic-second.docx`, 370-char job ad |
| Result | **approved**, initial review, `repair_calls = 0` |
| Detail | `docs/live-runs/RUN_10_TWO_FILE.md` |

**What it settles.** Two `source_documents` arrived under distinct ids. Both documents'
employment and education records survived, correctly attributed, with no employer, role,
date, degree or metric mixing, no duplicate records, and no invented figures or names. The
multi-document preservation fix is confirmed on the live route.

**What it does not settle — this is not a clean delivery.** Six bullets carry an appended
outcome clause neither source states (`ensuring compliance with NFPA 70E standards`,
`enhancing system reliability`, `gaining hands-on electrical construction experience`, …).
The reviewer marked all 28 lines `supported`, so the engine's only defence for that shape —
flag then repair — was never engaged; and `_introduces_beyond` flags capitalised names and
unbacked figures, not lowercase outcome words. Verified directly against the real bullet:
it returns `[]`.

**Also not verified.** The PDF and DOCX were **not** downloaded: no `/resume/export`
request left the browser on any of three attempts (patched `fetch`, coordinate click,
forced click). The content verified above is the approved markdown **as rendered by the
site**, not bytes read back from a compiled file. The offline suite proves the exporters
render this payload, but that is not this run. Browser download behaviour remains a
separate follow-up criterion.

## Next work, in priority order

1. **Appended outcome clauses in experience bullets** — the open fabrication risk. This is
   a reviewer-quality problem: a live reviewer that approves every line gives the repair
   path nothing to do. Two candidate directions, both offline-verifiable before any spend:
   extend the addition check to lowercase outcome verbs and their objects, or require the
   reviewer to cite an exact source span per bullet rather than a set of segment indices.
   Neither is implemented. **Do not spend to test.**
2. **Reconcile run 10's provider cost** from the OpenRouter activity export for 2026-11-03.
   The ledger records it as unknown by design, not as zero.
3. Real-model repair recovery is still unexercised (`repair_calls = 0` in every run).

**No second submission without a new allowance.** The 1-analysis allowance is exhausted.

## Verification commands (this session, unchanged from 9)

```
source .venv/bin/activate && cd Imagination_Resume_Module
python -m pytest tests/ -q -o addopts="" -p no:cacheprovider            # 996 passed, 3 skipped
python -m pytest tests/test_route_two_documents.py -q -o addopts=""     # 22 tests, real /analyze
cd ../Login_Landing_page
node node_modules/jest/bin/jest.js -c jest.integration.config.js \
  backend/tests/integration/multiFileRoute.test.js                      # 6 passed
```

---

# Prior handoff (session 9) — kept for history

**2026-11-03, session 9.** Two-document path verified offline; no credits spent that
session.

## Done this session (9)

- [x] **A cross-document claim is no longer treated as fabrication.** `_validate_review`
      passed ONE document to `_introduces_beyond` while the claim could cite several, so
      merging two faithful summaries reported document-1's own word "Facilities" as
      *introduced* and withheld a fully source-backed resume. The span is now judged against
      every document the claim cites. `tests/test_route_two_documents.py` (22 tests).
- [x] **Paired approval test.** The same supported summary approves as two lines AND as one
      line — asserting only one of the pair would prove either "the check got looser" or
      "nothing changed".
- [x] **Claim-level attribution still enforced**, by 7 controls each one line from an
      approving draft: appended unbacked tools, and transplanted employer / role / dates /
      achievement / qualification / figure. A multiset guard proves each really is a
      single-line edit (a set difference reported zero for the duplicate-line transplant).
- [x] **Paired explanation for a summary append.** `preserve_source_summary` repairs it
      before the gate, so that run publishes and the appended claim is absent from the
      published document. The bullet append — where no repair exists — is withheld by the
      gate. Asserting the first without the second would look like a miss.
- [x] **Export round trip both ways.** Approved payload → real `/resume/export` → real LaTeX
      + python-docx → both records read back from the produced files. Four alterations
      (appended claim, changed employer, moved date, dropped record) refused 409 under the
      existing receipt contract, plus the same content under another owner.
- [x] **Mutation-checked three ways.** Removing the addition check fails three controls;
      reverting the provenance fix fails the paired approval; disabling `verify_receipt`
      fails all five export tests.
- [x] No pipeline added. Publication gate unchanged. Single-file freeze untouched.

## Waiting on: one bounded captured run (needs an allowance)

**Approval was requested and is pending.** The session owner was unavailable, and silence
is not an allowance, so **no credits were spent**. This is the only remaining item.

| | |
|---|---|
| Input | `synthetic-single.docx` + `synthetic-second.docx` |
| Route | website → orchestrator `/api/analysis/multi-file` → Imaginator `/analyze` |
| Capture | `CORRECTION_LIVE_CAPTURE=1` for the run, then `=0`; purge container captures |
| Cost | **33 credits** (one analysis) |

### Procedure when the allowance exists

1. `CORRECTION_LIVE_CAPTURE=1` via `mcp_render-mcp-se_update_environment_variables`
   (merge; leave `CORRECTION_LIVE_CAPTURE_DIR` and `CORRECTION_PROVIDER_DISCLOSURE`).
2. Poll SSH until `capture=1`. A health 200 is **not** proof the deploy landed, and an
   env-var change needs a real deploy, not a restart.
3. `cd /app && PYTHONPATH=. python tools/live_capture_preflight.py` — expect exit 0.
4. Submit the two documents at `https://cogitometric.ai/app/resume-builder`.
5. Record `x-correlation-id`. The capture must show **two** `source_documents`.
6. Check the delivered PDF **and** DOCX: both employers, both institutions, both degrees
   with their own dates; no mixing, no duplicate, no invented qualification.
7. Export must accept the receipt. `CORRECTION_LIVE_CAPTURE=0`; purge container captures;
   keep the artifact locally.

### Pass/fail

**Passes** when every check in step 6 holds and the receipt is accepted by export.
**If withheld**, `scp` the capture and classify the objection against the source *before*
changing code — a checker rejection is not proof of fabrication.

## Verification commands (this session)

```
source .venv/bin/activate && cd Imagination_Resume_Module
python -m pytest tests/ -q -o addopts="" -p no:cacheprovider            # 996 passed, 3 skipped
python -m pytest tests/test_route_two_documents.py -q -o addopts=""     # 22 tests, real /analyze
python -m pytest tests/test_route_two_documents.py -q -o addopts="" -k export
```

## Verification commands (session 8, still valid)

```
python -m pytest tests/test_source_preservation.py -q -o addopts=""     # incl. multi-document §6
cd ../Login_Landing_page
node node_modules/jest/bin/jest.js -c jest.integration.config.js \
  backend/tests/integration/multiFileRoute.test.js                      # 6 passed
node node_modules/jest/bin/jest.js -c jest.unit.config.cjs \
  src/services/__tests__/multiFileRequest.test.ts                       # 3 passed
```

Note: `jest.integration.config.js` also collects `.worktrees/**`, where several DB-backed
suites fail for pre-existing environment reasons. Those failures reproduce with this
session's files removed.

---

# Prior handoff (session 8) — kept for history

**2026-11-02, session 8.**

- **Tests:** 981 passed, 3 skipped.
- **Tag:** `known-working-single-file-2026-11-02` → `73cf570` (kept; unchanged).
- **Branch:** `wip/correction-timeout-attribution`.
- **Capture was OFF** (`CORRECTION_LIVE_CAPTURE=0`), disclosure on
  (`CORRECTION_PROVIDER_DISCLOSURE=1`).
- **Credits:** unchanged. No paid submission that session.

## Done this session (8)

- [x] **The "website cannot capture multi-file at all" blocker is withdrawn.** It came from
      reading `app.py`'s `analyze_multi_file`, an Imaginator-local multipart endpoint that no
      orchestrator route targets. The website's two-file traffic goes to `/analyze`.
      `docs/ROUTE_MAP.md` already said so; it is now backed by execution, not line numbers.
- [x] **The two-file chain is recorded end to end**, offline, with a recording transport:
      frontend URL → orchestrator handler → downstream host/path → payload.
      `Login_Landing_page/src/services/__tests__/multiFileRequest.test.ts` (3 tests) and
      `Login_Landing_page/backend/tests/integration/multiFileRoute.test.js` (6 tests).
- [x] **Routing has not changed.** `git show 56517b8:backend/server.cjs` already calls
      `${imagBase}/analyze`; `git log -S` finds no retargeting commit.
- [x] **A real two-document defect was found and fixed** — the consequence the routing
      question exposed. Preservation was single-document-only: it deleted the second
      document's degree, truncated the second summary, and emitted `---` as a skill. Fixed in
      `source_preservation.py`; 10 unit tests + 7 route tests; mutation-checked.
- [x] **Ten tests relabelled.** `tests/test_multifile_measurement.py` →
      `tests/test_local_multipart_handler.py`. They exercise the local multipart handler and no
      longer claim to be multi-file acceptance.
- [x] Freeze tag and manifest kept. `docs/live-runs/FREEZE_2026-11-02.md` carries a dated
      correction; item 4 struck through rather than deleted.

## Still open (two documents)

1. A writer that merges both summaries into ONE line is withheld
   (`source_completeness_uncertain`), not truncated. Safe, but still a rejection. Pinned by
   `test_a_merged_two_document_summary_is_withheld_not_silently_truncated`.
2. Cross-document transplant of a quantified bullet between two DIFFERENT employers sharing
   a figure and qualifier: `_employer_of` scoping covers only within one employer's record.
3. No captured two-file run with a real model. The acceptance target is reachable but un-met.

## Verification commands (this session)

```
source .venv/bin/activate && cd Imagination_Resume_Module
python -m pytest tests/ -q -o addopts="" -p no:cacheprovider           # 981 passed, 3 skipped
python -m pytest tests/test_source_preservation.py -q -o addopts=""    # incl. multi-document §6
python -m pytest tests/test_route_two_documents.py -q -o addopts=""    # real /analyze, 2 docs

cd ../Login_Landing_page
node node_modules/jest/bin/jest.js -c jest.integration.config.js \
  backend/tests/integration/multiFileRoute.test.js                     # 6 passed
node node_modules/jest/bin/jest.js -c jest.unit.config.cjs \
  src/services/__tests__/multiFileRequest.test.ts                      # 3 passed
```

Note: `jest.integration.config.js` also collects `.worktrees/**`, where several DB-backed
suites fail for pre-existing environment reasons. Those failures reproduce with this
session's files removed.

---

# Prior handoff (session 7) — kept for history

**2026-11-02, session 7.**

- **Tests:** 913 passed, 3 skipped.
- **Latest commit:** `a2f584f`, pushed to `master`, deployed, verified in-container.
- **Branch:** `wip/correction-timeout-attribution` == `origin/master`.
- **Capture is OFF** (`CORRECTION_LIVE_CAPTURE=0`), disclosure on
  (`CORRECTION_PROVIDER_DISCLOSURE=1`), container capture dir empty.
- **Credits:** 1034. No analysis was purchased this session.

## Done this session

- [x] **Repair ordering fixed.** `source_correction.py` withheld on a missing source fact
      *before* entering the repair path, so a review reporting both unsupported lines and
      omitted facts withheld immediately. That is why every live run recorded
      `repair_calls == 0` (run 4: 19 flagged; run 5: 13; runs 6–7: 3). Flagged spans are
      now repaired first and the coverage verdict is taken on the repaired draft.
- [x] **Recovery proven offline**, not just reachability. A separately labelled layer
      answers the phases the recordings never performed
      (`live_capture.replay_with_extensions`, `tests/support/controlled_answers.py`);
      `ControlledExtension` refuses to overwrite a recorded phase, so the artifact stays
      authoritative for everything it contains. A repairable draft is repaired,
      re-reviewed, validated, approved, and exported to real PDF and DOCX whose content
      is read back.
- [x] **Negative controls** on that path: repair cannot delete an accomplishment, invent a
      qualification, loop, or exceed `MAX_CALLS`.
- [x] **Five/six-section contradiction resolved.** The answer key's comment promised five
      headings while its own tuple listed four without Projects; the prompt forbade an
      empty section; `_ensure_all_sections` deleted an empty header. Now: all headings
      required, empty is correct, filler forbidden. Verified end to end that an empty
      `## Projects` parses to `[]` and renders nothing.
- [x] **Objection classes corrected.** Formatting / omission / unsupported addition / lost
      qualifier / reviewer error, each mapped to its remedy. "Electrical Systems
      Technology dropped" was wrongly called fabrication and is now **omission** (the
      draft drops a fact, it asserts nothing false). The concatenated bold education name
      is **formatting**, not fabrication.
- [x] **Phase order and upper bounds restored** where exact historical call counts had
      been relaxed to `> 0`.

## Known defects NOT fixed (writer layer)

These are demonstrated and classified; fixing them needs prompt changes, and each prompt
change has so far cost a paid submission to validate.

1. **`8,000 hours of supervised` → `hands-on … mastering core`** — lost qualifier plus an
   invented trailing clause. Two defects, two repairs.
2. **Skills block** invents a category structure and admits activity verbs
   (`installation`, `maintenance`) plus the rank `apprentice supervision`.
3. **Summary** appends a proficiency list past where the source summary ends.
4. **Degree field dropped** ("Electrical Systems Technology") — omission.
5. **Education bold name** concatenates degree + institution, so `unsupported_entities`
   rejects it. Formatting.
6. **Writer omits the Projects heading entirely.** Repair cannot fix this: it replaces a
   flagged span and cannot insert a newline. The structural tests therefore run against a
   draft with the heading inserted, and say so.

Full table with reasons: `docs/live-runs/REMAINING_DEFECTS.md` and
`docs/live-runs/REPAIR_ORDERING_FIX.md`.

## Remaining: one captured website run (needs allowance)

Nothing else is outstanding. Authorization was requested; the session owner was
unavailable, so **no submission was made**. Balance 1034 credits; 33 per analysis.

### Procedure when an allowance exists

1. `CORRECTION_LIVE_CAPTURE=1` via `mcp_render-mcp-se_update_environment_variables`
   (merge, leave `CORRECTION_LIVE_CAPTURE_DIR` and `CORRECTION_PROVIDER_DISCLOSURE`).
2. Poll SSH until `capture=1`; a health 200 is NOT proof a deploy landed.
3. `cd /app && PYTHONPATH=. python tools/live_capture_preflight.py` — expect exit 0.
4. Submit `synthetic-single.docx`, title "Senior Electrician", the 370-char job ad at
   `https://cogitometric.ai/app/resume-builder`.
5. Record `x-correlation-id`; expect `repair` and a re-review in the phase log, which no
   live run has ever done.
6. Inspect résumé, commentary, receipt, and the actual PDF/DOCX.
7. `CORRECTION_LIVE_CAPTURE=0`; purge container captures; keep the artifact locally.

### If it is withheld

`scp` the capture, then
`python -c "from live_capture import replay; print(replay('<f>')[0])"`.
Classify the objection against the source **before** changing code — a checker rejection is
not proof of fabrication. Then fix offline and retest with a remaining attempt.

## Verification commands

```
source .venv/bin/activate && cd Imagination_Resume_Module
python -m pytest tests/ -q -o addopts="" -p no:cacheprovider      # 913 passed, 3 skipped
python -m pytest tests/test_repair_recovery_controlled.py -q -o addopts=""
python -m pytest tests/test_run7_objection_classification.py -q -o addopts=""
```

## Evidence and integrity

Protected captures, all byte-identical to when they were recorded:

| run | revision | sha256 (12) |
|---|---|---|
| 4 | `5f3c1f0` | `92e03d4bbfde` |
| 5 | `9057d966` | `8972d3711c6b` |
| 6 | `fb75917` | `269caf5e06b3` |
| 7 | `b855087` | `bb9643ffbaea` |

A capture is evidence about a revision. Its original diagnosis is read from its own
`review`/`timing` fields (`live_capture.recorded_stopping_point`), never recomputed from
current code — recomputing would rewrite the history that made the defect findable.

---

# Live acceptance — 2026-11-05 (authorised run)

**Live revisions:** imaginator `1adfa5b`, orchestrator-backend + frontend `e21fda2`.
**Input:** frozen `tests/fixtures/realistic_candidate/` (all 5 SHA-256 verified before the run).
**Path:** real orchestrator `POST /api/analysis/multi-file` → Imaginator `/analyze` → `/api/ml/imagination/export`.

## Result

| Step | Outcome |
|---|---|
| Imaginative via orchestrator | **APPROVED**, 249-word markdown, v5 receipt |
| PDF + DOCX download | **200**, real 127,809-byte PDF 1.7 and 37,758-byte DOCX |
| Facts preserved in **both files** | **22/22** (F01–F22) |
| Prohibited tools/achievements | **none** in either file |
| Conservative | mode `conservative`, policy `conservative-v1`, **no bridges, no policy state** |
| Conservative bridging call | **skipped** — no `[RESEARCHER] Analyzing job ad` in logs |
| Tamper: appended achievement | **409** `receipt_content_mismatch` |
| Tamper: altered employer | **409** `receipt_render_content_mismatch` |
| Tamper: policy state mode | **409** `receipt_policy_mismatch` |
| Tamper: dropped policy state | **409** `receipt_policy_state_missing` |
| Tamper: forged signature | **409** `receipt_signature_invalid` |
| Tamper: injected skill | **409** `receipt_render_content_mismatch` |
| Untampered control | **200** |

## Two defects found by running it (both fixed and deployed)

**1. No mode could be selected through the website, and the policy state never
returned.** `backend/server.cjs` had *zero* references to `skill_inference_mode`:
the multi-file payload is an explicit object literal, and the browser sent only
`creativity_mode`. Every website analysis silently took the default, so
Conservative was unreachable. Separately, both publication-field allow-lists
omitted `skill_inference_policy_state`, so a v5 receipt arrived at export
unaccompanied and was refused with `receipt_policy_state_missing` — an approval
that could be issued and never downloaded. Fixed in `e21fda2`; 6 regression
tests, mutation-checked.

**2. A commit that looked deployed but was not.** The first attempt at that fix
was amended onto master while the working tree still held the mutated
(mutation-tested) copy, so `344edee` shipped *without* the fix — and the
Conservative run duly came back `imaginative`. Caught by asserting the *response*
mode rather than trusting the deploy status. Re-committed as `e21fda2` after
verifying `git show HEAD:backend/server.cjs | grep -c skill_inference_mode` = 5.

## What this run does NOT establish

- **No inferred skill survived, because none was proposed.** Every approved
  skill in this candidate is a verbatim source skill, so there is no bridging
  opportunity to exercise: the Researcher correctly proposed zero bridges and
  `surviving_bridges` is `[]` in both modes. A bridging positive still needs a
  fixture whose activity implies a skill the source does not already name. This
  is a **fixture limitation, not a passing bridging test** — do not read it as one.
- **Conservative withheld** (`source_completeness_uncertain` / `no_op_repair`) on
  live model output. That is the correction gate working, not a mode defect: the
  mode, policy version and empty bridge set were all correct.
- **File name is `resume`** for this payload, because the approved headline is
  empty. Cosmetic, but the filename does not derive from the job title here.

## Cost

Synthetic account with 3 resume-analysis credits. Three Imaginator runs
(2 via orchestrator, 1 direct). Job-search enrichment debited 90 *job-search*
credits (9 paid calls) — separate pool, not resume analysis.

---

# Two-mode feature — final live comparison (2026-11-05)

**Live:** imaginator `897d01b`, orchestrator-backend + frontend `58a63d9` (+ `ff41695` UI).
**Fixture:** `tests/fixtures/bridge_opportunity/` — activity stated, skill label absent.

## Outcome

| | Imaginative | Conservative |
|---|---|---|
| Mode / policy reported | `imaginative` / `imaginative-v1` | `conservative` / `conservative-v1` |
| Bridging call | runs | **skipped by policy** (logged) |
| Surviving bridges | **5** | **0** |
| Delivered | approved, export 200 | approved, export 200 |
| PDF / DOCX | 125,594 B / 37,765 B | 125,226 B / 37,724 B |
| Source facts in **both files** | 10/10 | 10/10 |
| Prohibited content | none | none |

**Bridged labels, and whether they reached the bytes:**

`trend analysis`, `scorecard development`, `invoice reconciliation`,
`equipment failure analysis`, `stock level planning` — all five verified present
in both the PDF and the DOCX of the Imaginative delivery.

**The Conservative split is the proof the policy works.** Two of those five
(`invoice reconciliation`, `stock level planning`) are *source-declared* skills,
so they correctly appear in both modes. The genuinely inferred ones
(`trend analysis`, `scorecard development`, `equipment failure analysis`) appear
**only** in Imaginative. Conservative did not lose source facts to gain that
strictness: 10/10 in both.

## Three defects found by running it

1. **No way to choose a mode in the website.** The only control was the
   creativity preset, a prose-style dial; `skill_inference_mode` was never sent.
   Added a separate Skill Inference selector and wired it through the request.
2. **The job-ad fork pre-empted candidate-aware bridging.** The orchestrator's
   `runJobAdResearcher` posts only `{job_ad}`, so it cannot produce a
   candidate-anchored bridge — but forwarding its result made the Imaginator log
   `Stage 1: using precomputed research_data (fork-join skip)` and skip the
   Researcher that could. Every Imaginative run delivered zero bridges.
3. **The candidate documents never reached Stage 1.** After fixing (2), the
   Researcher still proposed nothing: `source_documents` was dropped at three
   layers (`app.py` → `run_new_pipeline_async` → `run_pipeline` → `kwargs.get`),
   so `quotable` was empty and the model correctly refused to quote anything.

All three were invisible in the offline suite, which passed throughout.

## Why the earlier "gate working" claim was wrong

`source_completeness_uncertain` is emitted from two places — the source
inventory and the post-review path. Instrumenting the wrong one produced no log
line, and a bare `except: pass` hid a `NameError` that made the instrument
unreachable. The evidence, once it fired, was `required_source_claims: 35,
uncertain_source_claims: 4`. **Do not accept "the gate is working" without that
number.**

## Still not established

- **Reviewer stability.** The live reviewer withheld this fixture on roughly
  half of attempts (`source_completeness_uncertain`, `incomplete_review`,
  `source_facts_missing`, `fallback_output`), against 1 attempt for
  `realistic_candidate`. The successful runs are genuine, but delivery is not
  reliable on this fixture and that variance is unexplained.
- **Visual inspection was not performed** — the model has no vision here.
  Layout was verified programmatically instead: both PDFs are 1-page US Letter,
  produced by pdfTeX 1.40.26, with all sections present and no truncation.
- A Conservative receipt rejects a genuine mode flip, a policy-version change and
  an injected bridge — all `receipt_policy_mismatch`. An earlier "ACCEPTED" in
  this session was a test bug (it flipped conservative→conservative).
