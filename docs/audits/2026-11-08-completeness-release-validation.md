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

- `captures/final/acceptance4/conservative-a4.json` — withheld attempt, retained
- `captures/final/acceptance4/conservative-a4b.json` — approved, verbatim server response
- `captures/final/acceptance4/conservative.pdf`, `conservative.docx` — the files the site produced
