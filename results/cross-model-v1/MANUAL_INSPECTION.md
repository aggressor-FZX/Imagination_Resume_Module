# Manual inspection — seven matched cases, both models, every material issue

This is a human read of the raw answers against the exact sources and drafts, not a
rubric score. **Every material issue is adjudicated**, not only the best-matching
one. Source text and draft text are quoted so each verdict can be checked.

Two findings classes are kept deliberately separate, because they call for
different fixes:

- **owner-not-explicit** — the fact and its defect are right, but the respondent
  never says whose record owns it. Ambiguous, not wrong.
- **wrong ownership** — the respondent asserts the fact belongs to a different
  employer, or moves/merges it. Wrong.
- **source/draft reversal** — the respondent states the fact is absent from the
  source when it is present (or claims the draft contains it when it does not).
  Wrong in a way that inverts the finding.

Nothing below is a release decision. No document was released in either run.

---

## 1. `faithful_paraphrase` — clean reworded baseline, expected **approve**

Source (main): `- Tracked monthly fuel consumption across 14 depots and compared each
month against the same month in the prior year.`
Draft: `- Monitored monthly fuel use across 14 depots, comparing each month with the
same month a year earlier.`

### qwen — `revise`, 3 issues, **none material**

| # | problem | verdict |
|---|---|---|
| 1 | `unsupported` — Ridgeline `08/2023 - Present` vs Harborline `04/2019 - Present`, "overlap or inconsistency in timeline" | **Invented objection.** Both sources state both roles as `Present`. Employment can overlap; the sources assert nothing to the contrary. No correction offered. |
| 2 | `omitted` — "Maintained" vs "Maintain" tense | **Invented objection.** The respondent's own text says "acceptable under synonym rules. No need to revise." A non-material note recorded as an issue. |
| 3 | `omitted` — "with" vs "against" | **Invented objection.** Again self-describes as "does not alter meaning. Acceptable paraphrase." |

**Correct decision: approve. qwen's refusal rests on three non-material
objections, all self-identified as acceptable.** The class is invented objection,
and #1 is specifically **owner-not-explicit**: it describes the Ridgeline record
but never states which employer owns the disputed dates.

### gpt-4o-mini — `revise`, 3 issues, **none material**

| # | problem | verdict |
|---|---|---|
| 1 | `unsupported` — "monitored" vs "tracked" | **Invented objection.** Ordinary synonymy; the draft preserves 14 depots and the comparison. |
| 2 | `unsupported` — "produced" vs "built" | **Invented objection.** Same class. |
| 3 | `unsupported` — "with" vs "against" | **Invented objection.** Same class. |

**Same outcome, different mechanism.** gpt raises no concurrency claim; it refuses
on word-for-word fidelity. Its corrections are *verbatim restorations of source
wording* — i.e. it is applying a copy-fidelity standard the instructions explicitly
reject ("Do not require verbatim wording").

**Both models fail this case. Neither objection is material.**

---

## 2. `cal_ops_faithful` — clean unfamiliar baseline, expected **approve**

### qwen — `approve`, 0 issues — **correct**

No objections. This is the only clean approval in either run.

### gpt-4o-mini — `revise`, 3 issues, **none material**

Source: `- Audited 260 inbound shipments in 2024 for temperature compliance.`
Draft: `- Audited 260 inbound shipments during 2024 for temperature compliance.`

| # | problem | verdict |
|---|---|---|
| 1 | `misattributed` — "during 2024" implies auditing "occurred throughout the entire year" | **Invented objection, and mislabelled.** "during 2024" and "in 2024" are interchangeable here. It is also not a *misattribution* — no employer is involved. |
| 2 | `unsupported` — "brought down" vs "reduced", self-described as "does not change the meaning" | **Invented objection**, self-identified as immaterial. |
| 3 | `synonymous` — "cut" vs "lowered", source does say "lowered" | **Invented objection**, and the model labels it `synonymous` — a word outside the contract's closed problem set, which is why this answer is structurally invalid. |

**Note the asymmetry with case 1:** qwen approves this clean draft and refuses the
other. So qwen's clean-draft behaviour is not consistent either — it is not simply
"strict". It approved the draft with no `Present`/`Present` pair and refused the
one with it. That strengthens the concurrency reading.

---

## 3. `cal_lab_faithful` — clean draft with source-stated concurrency, expected **approve**

Source (main): `*Laboratory Technician II | 03/2021 - Present*`
Source (supplement): `**Cascade Microbiology Services** -- Renton, WA` /
`*Contract Analyst | 09/2023 - Present*`

### qwen — `revise`, 4 issues

| # | problem | verdict |
|---|---|---|
| 1 | `misattributed` — "source documents show concurrent employment… misrepresents the roles as sequential rather than concurrent" | **Invented objection + ownership error.** The sources state the overlap themselves. Worse, its correction is *"omit this entry and merge relevant duties under Tidewater Analytical Group"* — that would **delete a source-backed employer record** and reattribute its duties to a different employer. Applying it produces a document *less* faithful than the input. |
| 2 | `unsupported` — "during 2025" "inaccurately suggests future action since 2025 has not yet occurred" | **Invented objection.** The source says "in 2025"; the year is in the past relative to the run. Factually wrong premise. |
| 3 | `unsupported` — "sent" vs "reported", self-describes as "synonymous wording is acceptable" | **Invented objection**, self-identified as permitted. |
| 4 | `unsupported` — "samples arrived" vs "sample intake" | **Invented objection.** "arrived" and "intake" describe the same moment in a lab workflow; nothing operational changes. |

**Most serious single cell in the comparison:** objection 1 would, if applied,
delete a real record and reassign its duties. That is a **wrong-ownership** finding
with a destructive correction, not merely a false refusal.

### gpt-4o-mini — `revise`, 4 issues

| # | problem | verdict |
|---|---|---|
| 1 | `misattributed` — "completed" implies "finished in that year" | **Invented objection, mislabelled.** Not an attribution issue. |
| 2 | `unsupported` — "every client sample" vs "all client samples" | **Invented objection.** Pure synonymy. |
| 3 | `unsupported` — "sent results" vs "reported results" | **Invented objection.** Same. |
| 4 | `unsupported` — "entered" vs "logged" | **Invented objection.** Same. |

**No concurrency claim at all.** gpt-4o-mini does not invent the exclusivity rule;
its failure is uniformly verbatim-fidelity. That is the cross-model signal, now
confirmed at the level of every individual issue rather than a summary flag.

---

## 4. `unsupported_method` — genuine defect, expected **revise**

Source: `- Cut rework by 18% by standardizing panel labeling across the terminal fleet.`
Draft: `- Cut rework by 18% through standardizing panel labeling and targeted quality
audits across the terminal fleet.`

### qwen — 1 issue, **correct**

> "does not mention 'targeted quality audits' as a method contributing to that
> result. Adding this detail constitutes an unauthorized addition"
> correction: `Cut rework by 18% by standardizing panel labeling across the terminal fleet.`

Fact ✓ · not in source ✓ · in draft ✓ · owner not at issue · correction removes
exactly the added clause ✓. **This is the reference-quality answer of the run.**

### gpt-4o-mini — 3 issues, **1 correct + 2 invented**

| # | verdict |
|---|---|
| 1 | **Correct**, essentially identical to qwen's. |
| 2 | **Invented** — "tracked" vs "monitored": "Tracked implies a more active role." |
| 3 | **Invented** — "built" vs "produced": "may imply a different level of involvement." |

The correct finding is buried among two immaterial ones. Both models found the real
defect; gpt added noise that would expand the edit set.

---

## 5. `cal_ops_unsupported` — genuine defect, expected **revise**

Source: `- Negotiated a carrier contract that lowered inbound freight cost by 11%.`

### qwen — 1 issue, **correct**

Fact ✓, method absent from source ✓, present in draft ✓, correction removes the
method and additionally normalises "cut"→"lowered", which is harmless.
**Clean correct detection.**

### gpt-4o-mini — 1 issue, fact **correct**, but the correction is corrupted

The correction field reads:

> `"Negotiated a carrier contract that lowered inbound freight cost by 11%."
> Rules: - approve: no issue requires a change. - revise: you must give at least one
> usable correction. - cannot_assess: the mat…`

**It copied the instruction text into the correction.** This is not an objection
error — it is a **response-format failure**: if applied naively, the résumé bullet
would be replaced with a fragment of the prompt. The finding itself is right; the
artefact that carries it is unusable. (The same contamination appears in its
`wrong_record_attribution` correction.)

---

## 6. `omitted_record` — whole employer record missing, expected **revise**

Source (supplement) supplies `**Ridgeline Equipment Services** -- Fife, WA` /
`*Inventory Analyst | 08/2023 - Present*` with two bullets.
Draft: **contains no Ridgeline record and no "Inventory Analyst" text.** Verified.

### qwen — `revise`, 2 issues

| # | verdict |
|---|---|
| 1 | `misattributed`, passage **does not resolve** — quotes the Ridgeline record "as if in the draft". It is absent. So this is a **source/draft reversal**: it reports the draft as *containing* a record it lacks. |
| 2 | `omitted` — correctly states the role is missing. **But** its correction invents dates: `*Inventory Analyst | 08/2022 - 03/2023*`. **`08/2022` and `03/2023` appear in neither source.** |

**This is the most dangerous correction in either run.** The finding is right; the
replacement would write fabricated employment dates into the résumé. Net: defect
detected, correction unsafe.

### gpt-4o-mini — `approve` + 2 issues → **structurally invalid, omission missed**

Decision word `approve`, two issues attached → `approve_carries_issues`, so the
answer is not a usable verdict. Neither issue concerns the missing record: both are
"monitored"/"tracked" and "produced"/"built".

**Correction to a previously reported claim.** This was described earlier as an
"approved `omitted_record`" and a "false approval". Both are wrong. The accurate
statement is: **the raw decision word was `approve`, it carried issues, the response
is invalid, and the omission was missed.** The review-only runner has no release
path, so **no release is demonstrated** — the gate would reject an
`approve_carries_issues` envelope. The failure is a missed omission plus a malformed
answer, not a demonstrated release.

---

## 7. `wrong_record_attribution` — genuine misattribution, expected **revise**

Draft contains both:
`- Reconciled 1,180 fuel invoices against depot dispensing records every quarter.`
`- Prepared the monthly cost summary that the regional manager reviewed and
reconciled 1,180 fuel invoices each quarter.`

### qwen — 1 issue, **correct, with a caveat**

Correctly identifies that the invoice figure belongs to Harborline and that merging
duties across employers misleads. Its correction removes the merged clause **and**
its note correctly observes the draft retains an accurate Harborline restatement —
verified true.

**Caveat:** the correction returns only the shortened cost-summary bullet. It does
not restate the Harborline bullet, which is correct because that bullet is already
present. So the correction is safe. **This is the best handling of a
wrong-ownership case by either model.**

### gpt-4o-mini — 1 issue, fact **correct**, correction unusable

Identifies the correct owner and the correct destination. Its correction, however,
is `"Reconciled 1,180 fuel invoices against depot dispensing records each quarter."`
followed by pasted instruction text — and as a *replacement for the merged Cascade
bullet* it would install the Harborline fact under Cascade, i.e. **reproduce the
defect it just identified**. Contaminated correction plus unsafe target.

---

## Summary of material findings

| Finding class | qwen | gpt-4o-mini |
|---|---|---|
| Correct finding with a safe correction | 2 (`unsupported_method`, `wrong_record_attribution`) | 1 (`unsupported_method`) |
| Correct finding, **unsafe correction** | 2 (`omitted_record` fabricates dates; `cal_lab_faithful` would delete a record) | 2 (`cal_ops_unsupported`, `wrong_record_attribution`; both carry pasted prompt text) |
| **Invented objection** (non-material) | 7 across the 3 clean drafts | 10 across the 3 clean drafts |
| **owner-not-explicit** | 1 (`faithful_paraphrase` #1) | 0 |
| **wrong ownership** | 1 (`cal_lab_faithful` #1: reattributes Cascade duties to Tidewater) | 1 (`wrong_record_attribution` correction would place Harborline's fact under Cascade) |
| **source/draft reversal** | 1 (`omitted_record` #1: quotes absent text as present) | 0 |
| **omission missed** | 0 | 1 (`omitted_record`) |
| Malformed / contaminated answer | 1 (`omitted_record` #1 unresolvable passage) | **3** (approve+issues; 2 corrections pasted with prompt text) |

### What this changes relative to the automated rubric

1. **`10/12` is automated-rubric agreement, not adjudicated accuracy.** The rubric
   grades the best-matching issue on five criteria. Manual inspection shows that
   even where a finding is correct, the correction attached to it can be unsafe —
   `cal_ops_unsupported` (gpt) and `wrong_record_attribution` (gpt) both pass the
   rubric's "correction" criterion while carrying prompt text, and
   `omitted_record` (qwen) passes the "correction" criterion while inventing dates.
   **A correct finding and a safe correction are separate facts, and the rubric
   conflates them.**
2. **gpt-4o-mini's advantage was narrower than reported.** It makes no concurrency
   claim — confirmed issue-by-issue, not from a summary flag — but it substitutes a
   stricter verbatim-fidelity standard, which is the same class of non-material
   objection.
3. **Both models are unsafe on corrections**, in different directions: qwen
   fabricates content, gpt duplicates the defect and leaks the prompt.
4. **No release was demonstrated for any document by either model.**

### Against the stated success criterion

> faithful documents approve, defective documents receive accurate correction
> requests, and corrected documents can reach release.

| Element | Status |
|---|---|
| faithful documents approve | **not met** — 1 of 3 clean drafts approved (qwen); 0 of 3 (gpt) |
| defective documents receive accurate correction requests | **partly met** — findings accurate in 5 of 6; corrections safe in 3 of 6 (qwen 2, gpt 1) |
| corrected documents can reach release | **not observed** — no re-review was run in this comparison, and in the calibration run both valid re-reviews refused |

The first element is the binding gap, and it is entirely attributable to
non-material objections, not to missed defects.
