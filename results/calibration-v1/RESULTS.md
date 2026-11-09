# Calibration run 1 — results

Single authorized bounded run, executed once on `experiment/llm-led-approval`.
**No instruction, fixture or prompt changes were made before, during or after it.**
Machine-readable output: `calibration.json`. Generated report: `CALIBRATION.md`.

## Conditions actually met

| Limit | Authorized | Used | |
|---|---|---|---|
| Logical review/re-review calls | 27 | **18** | under |
| Actual HTTP requests incl. retries | 81 | **18** | under |
| Observed spend | stop at $1.00 | **$0.008488** | under |

- `cost_is_known: true` — **0 unmetered responses**, so the $0.008488 figure is a
  complete record of this run, not a partial one. Each of the 18 requests returned
  usage; every request produced a billable response, so requests == logical calls.
- Frozen identity, unchanged from before the run: instructions `4261e9cb5b031230…`,
  fixture `b4b73e68b4d69b39…`, mode `imaginative`.
- Split: $0.003195 first calls + $0.000562 re-review = $0.003757 familiar;
  $0.004731 unfamiliar (no re-reviews reached).

## 1. Accurate judgments

"Accurate" here means the reviewer's own words named the defect the fixture
actually contains. Verified against source/draft ground truth per case.

### Unfamiliar — 8 of 8 defective cases identified

| Case | Decision | Valid | Named the real defect |
|---|---|---|---|
| `cal_lab_unsupported` | `revise` | no (envelope) | **yes** — "does not mention the use of EPA Method 1604 membrane filtration" |
| `cal_lab_missing` | `revise` | no (envelope) | **yes** — "omits a key detail about timely reporting … 24-hour regulatory window" |
| `cal_lab_wrong_record` | `revise` | yes | **yes** — "640 drinking-water assays … performed at Tidewater … attributed to Cascade" |
| `cal_lab_omitted` | `revise` | yes | **yes** — "omits an entire role listed in the supplement file" |
| `cal_ops_unsupported` | `revise` | yes | **yes** — "'by consolidating LTL lanes onto two regional carriers' is not stated in the source" |
| `cal_ops_missing` | `revise` | no (envelope) | **yes** — "recovering 94% of missed delivery windows … not present in the source file" |
| `cal_ops_wrong_record` | `revise` | yes | **yes** — "audit of 260 inbound shipments … performed at Pioneer, not at Gorge" |
| `cal_ops_omitted` | `revise` | no (envelope) | **yes** — "draft omits a confirmed position … from `ops-supplement.txt`" |

Every unfamiliar defect was caught, and caught **for the right reason**. This is the
strongest result in the run: the reviewer generalised to two new domains
(water-quality laboratory, cold-chain logistics) with figures and employers it had
never seen.

### Familiar — 4 of 4 defective cases identified

| Case | Decision | Valid | Named the real defect |
|---|---|---|---|
| `unsupported_method` | `revise` | yes | **yes** — "targeted quality audits … not mentioned in the source" |
| `missing_fact` | `revise` | no (envelope) | **yes** — names `1,180` |
| `wrong_record_attribution` | `revise` | yes | **yes** — 1,180 assigned to Cascade, belongs to Harborline |
| `omitted_record` | `revise` | no (envelope) | **yes** — names `Ridgeline` / `Inventory Analyst` |

`missing_fact` was **missed in the retained advisory arm** last time. With the
refined instructions it is now named. That is the one instruction-sensitive fix that
is demonstrably working.

## 2. Invented objections

**Familiar: 3 of 3 faithful cases refused on invented grounds.**

| Case | Objection | Why it is invented |
|---|---|---|
| `faithful_paraphrase` | issue set with no actionable content (envelope rejected) | no declared defect; objection could not be validated |
| `supported_method` | "'through standardizing panel labeling' adds a causal mechanism" | the source says "**by** standardizing panel labeling" — same mechanism |
| `transferable_skill` | Ridgeline dates/role conflict | the sources state `04/2019 - Present` and `08/2023 - Present` themselves |

**Unfamiliar: 1 of 2 faithful cases refused on invented grounds.**

| Case | Objection | Why it is invented |
|---|---|---|
| `cal_lab_faithful` | (a) concurrent employment with Cascade; (b) "'during 2025' inaccurately suggests future action since 2025 has not yet occurred"; (c) "'sent' changes the specificity" | (a) the sources state both roles as `Present`; (b) the source says "in 2025" — the calendar year has passed; (c) "sent" vs "reported" is the synonymy the instructions explicitly permit |

`cal_ops_faithful` was **correctly approved** (`initial_llm_approve`) — the only
release in the whole run, and it is on the correct document.

So the invented-objection rate is **4 of 5 faithful drafts**, down from 6 of 6 in
the retained matrix but not eliminated, and the remaining inventions are of the same
two kinds: a `Present`/`Present` overlap read as a contradiction, and pedantic
wording objections the instructions name as permitted.

## 3. Operational failures and invalid responses

**0 operational failures.** No timeouts, no provider errors, no budget refusals.
The metering fix means this is a measured fact: 18 requests dispatched, 18 metered,
18 billable.

**7 of 17 responses were structurally invalid** — the decision word was present but
the envelope was rejected:

| Cause | Count | Cells |
|---|---|---|
| `problem_word_unknown` (model invented `problem: synonymous`) | 3 | `cal_lab_unsupported` |
| `passage_not_in_draft` | 4 | `omitted_record`, `cal_lab_missing`, `cal_ops_missing`, `cal_ops_omitted` |
| `revise_without_actionable_issue` | 4 | `faithful_paraphrase`, `cal_ops_missing`, `cal_ops_omitted` |
| `revise_without_issues` / `revise_without_correction` | 2 | `missing_fact`, `cal_ops_missing` |

**This is the dominant defect, and it is a contract problem, not a judgement
problem.** Two distinct behaviours produce it:

- **A new problem word.** The model wrote `problem: synonymous` to record *permitted*
  paraphrase. The contract's `problem` field is a closed set of defect kinds; there
  is no way to say "this is fine". The model tried to be informative and the envelope
  was rejected. Its actual defect detection in that cell was correct.
- **Omission claims quoted as source text.** For `omitted` issues the model quotes
  the *missing source text* in `passage`, which by definition is not in the draft, so
  `passage_not_in_draft` fires. `decision_contract` has an `omitted` path that
  accepts `(omitted)` as the passage, but the model often writes the source line
  instead. The proposed correction is still `(omit)`, which is what the contract asks
  for, so the substance is right and the envelope is not.

Neither was fixed here: the brief froze instructions and fixtures, and both are
contract/parser questions rather than instruction questions.

## 4. Repair limitations (scaffold, not reviewer)

**1 proposal drawn out of 17.** 16 cells never reached a repair:

| Scaffold refusal | Count |
|---|---|
| `not_attempted` (review unverifiable) | 8 |
| `ambiguous_or_multiple_issues` | 4 |
| `passage_not_unique_literal_bullet` | 3 |
| `replacement_not_exact_owned_source_bullet` | 1 |

The single drawn proposal (`unsupported_method`, familiar) was re-reviewed and
**refused** (`re_review_not_approved`). So the scaffold was exercised exactly once
and produced no release.

`ambiguous_or_multiple_issues` appears 4 times even though the reviewer named the
right defect: it raised a correct objection **plus** a synonymy note, and the
one-issue-only scaffold refused both. That is a scaffold cost imposed by a reviewer
that is being more informative than the scaffold can use.

## 5. Approval of the exact corrected document

| | Familiar | Unfamiliar | Total |
|---|---|---|---|
| Re-reviews attempted | 1 | 0 | 1 |
| Valid re-reviews | 1 | 0 | **1** |
| **Approved the exact proposal** | **0** | **0** | **0** |
| Refused (valid, said `revise`) | 1 | 0 | **1** |
| Invalid re-review | 0 | 0 | **0** |

The one re-review was **valid and returned `revise`** — a clean refusal, not an
envelope failure. Its reasoning is the important part: asked to approve a document
whose only change was removing the invented "targeted quality audits" clause, it
refused on the `Present`/`Present` overlap instead:

> "the candidate's main resume … indicates the individual was employed at Harborline
> Transit Authority from 04/2019 to Present. This creates a direct contradiction in
> timelines since 'Present' cannot refer to two overlapping roles"

So the correction of the actual defect was performed and the reviewer **still**
refused, for the same invented reason that drives half the other invented objections.
This is the single most informative cell in the run: fixing the real defect does not
change the outcome while the timeline misreading stands.

**No corrected document was approved.** The corrected-document approval axis still
has no positive observation, now across two independent runs.

## 6. Pair separation (the unfamiliar control)

The pairing design requires the faithful member to be **approved** and the defective
member refused **with a valid answer**. An invalid variant does not count: two
answers that the parser could not read are not a demonstrated separation.

| Pair | baseline | variant | variant valid | separated |
|---|---|---|---|---|
| lab faithful vs unsupported | revise | revise | **no** | **NO** |
| lab faithful vs missing | revise | revise | **no** | **NO** |
| lab faithful vs wrong_record | revise | revise | yes | **NO** |
| lab faithful vs omitted | revise | revise | yes | **NO** |
| ops faithful vs unsupported | **approve** | revise | yes | **yes** |
| ops faithful vs missing | **approve** | revise | **no** | **NO** |
| ops faithful vs wrong_record | **approve** | revise | yes | **yes** |
| ops faithful vs omitted | **approve** | revise | **no** | **NO** |

**2 of 8 pairs separated.** All four refusals above *named the right defect* (see
§1), so 8 of 8 variants are substantively correct; only 2 reach the bar of a valid,
demonstrated separation. The gap is entirely in validity, not in detection:

- where the faithful draft was refused (all four lab pairs), no separation is
  observable regardless of the variant;
- where it was approved (ops), two of four variants were still structurally invalid.

So the honest summary is: **defect detection is complete in both domains;
separation is blocked first by the faithful-draft refusal and second by envelope
rejection.**

## 7. Answer to the decision point

The instruction was: if faithful drafts approve while defective drafts receive
accurate revision requests, move toward a bounded integrated delivery test; if not,
diagnose the specific objections before adding another mechanism.

**Not yet.** The condition is met for detection and not met for approval:

- defective drafts: **12 of 12** named the real defect (8 unfamiliar, 4 familiar);
- faithful drafts: **1 of 5 approved** (`cal_ops_faithful`), 4 refused on invented
  grounds.

Adding a mechanism would not help — the reviewer already finds every defect. The
specific objections to diagnose are narrow and now well-evidenced:

1. **`Present`/`Present` concurrency read as a contradiction.** Responsible for
   `transferable_skill` and `cal_lab_faithful`. Both sources state the overlap; the
   reviewer asserts impossibility. Half of the invented objections.
2. **Permitted synonymy still objected to** ("through"/"by", "sent"/"reported",
   "during 2025"/"in 2025"). The instructions name exactly these pairs and the model
   quotes them back while still objecting.
3. **Envelope rejection on otherwise-correct answers** (7 of 17): `problem:
   synonymous` is unrepresentable, and omission passages are quoted from the source.

Items 1 and 2 are instructions; item 3 is the contract. The evidence says fix the
contract's ability to express "permitted paraphrase" and accept source-quoted
omission passages before touching any threshold.

## 8. Not established

- One run, one model (`qwen/qwen3-coder`), temperature 0, 17 cases. No reliability
  estimate.
- The familiar and unfamiliar sets are not independent samples; both was authored
  with the same defect taxonomy.
- No independent human audit of these 17 documents.
- No integrated delivery test has been run; production is untouched.
