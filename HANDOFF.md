# Two-Mode Skill Bridging — feature handoff (resumeDraft_fix)

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
- [ ] Phase C remainder — phrase-scoped spans for non-Skills lines inside prose
- [ ] Phase D remainder — preview/export parity harness, real-file skills assertions
- [ ] Phase E — caches and isolation

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

## Next work, in priority order

1. **Wire the work list into the review** (not done). Decide whether a non-empty work list
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
