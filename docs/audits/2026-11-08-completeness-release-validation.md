# Section-completeness: release and website/export validation

Recorded 2026-11-08. Closes item 4 of the brief.

## What was validated

| Check | Result |
|---|---|
| Deployed commit | `99996e4` on `imaginator-resume-cowriter` |
| Website run | approved, receipt issued |
| All five core sections in Markdown | yes |
| All five core sections in PDF | yes, correct order, 1 page, 0 blocks outside, 2 embedded fonts |
| All five core sections in DOCX | yes, correct order |
| Receipt verifies | `ok` (checked on the deployed service) |
| Tampered markdown refused | `receipt_content_mismatch` |

The run also exercised the US-only location change: the location field accepted
`Austin, TX` and the analysis completed normally, so the previous task's change
does not break this path.

## Attempt log — failures retained, not hidden

| # | Outcome | Reason |
|---|---|---|
| 1 | **withheld** | `source_completeness_uncertain`, `uncertain_source_claims: 1` |
| 2 | approved | retained as `conservative-a4.json` |
| 3 | script error | selector used the old placeholder; the field now reads "(US)" after the location change — fixed and re-run |
| 4 | approved | retained as `conservative-a4b.json`, exports verified |

3 submissions, 2 approved, 1 withheld, 0 false approvals.

The withhold was **not** caused by the section contract: `source_section_missing`
never appeared in the logs for that run. It was the pre-existing lexical
downgrade path (`uncertain_source_claims: 1`), which is the same intermittent
behaviour recorded earlier.

## Scope reminder

The contract is a WHOLE-SECTION check. It proves the five core sections carry
content. It does **not** prove every qualification survived — a section with one
remaining record passes while another is lost. That limit is pinned by
`test_a_section_with_one_surviving_record_satisfies_the_contract`.

## Artifacts

**CORRECTION, 2026-11-08.** An earlier version of this file called
`conservative-a4.json` a "withheld attempt, retained". That was wrong and
contradicted the table above it. Verified from the files themselves:

| Artifact | runId | Status | Evidence |
|---|---|---|---|
| `conservative-a4.json` | `88e76998-a5a9-4337-9130-da4462278ae1` | **approved** | `status: completed`, `requires_review: false`, 1891-byte markdown, receipt present |
| `conservative-a4b.json` | `052aa72d-120d-4034-b11f-28fb2d34c9b6` | **approved** | `status: completed`, `requires_review: false`, 1893-byte markdown, receipt present |

**No retained artifact was withheld.** The one withheld submission in the log
(`b6851d64-4d82-4153-baf9-1576e6a61a0b`, `source_completeness_uncertain`) left no
payload, because the sink was not running when it happened. That is why the
attempt log above has a withheld row with no artifact beside it.

What the service log shows for the retained runs, by outcome shape:

- one approval with `repairs: 1, reviews: 2` (a repair happened)
- one approval with `repairs: 0, reviews: 1` (clean first pass)

Which artifact is which is **not** established here. The two payloads differ by
2 bytes (1891 vs 1893) and that alone is not evidence. Recording the ambiguity
rather than inventing a correlation.

- `conservative.pdf`, `conservative.docx` — the files the site produced for the
  final approved run.
