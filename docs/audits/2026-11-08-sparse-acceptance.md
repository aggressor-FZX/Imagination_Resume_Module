# Sparse-candidate acceptance and consolidated result

Recorded 2026-11-08. Exact build: **`99996e40`** (live on
`imaginator-resume-cowriter`), frontend `6833d811`.

## 1. Corrected records

**`conservative-a4.json` status.** Previously recorded as "withheld attempt,
retained". That was wrong. The file itself says `status: completed`,
`requires_review: false`, 1891-byte markdown, receipt present — it is an
**approved** run. No retained artifact from that batch was withheld. Corrected in
`2026-11-08-completeness-release-validation.md`.

**Contract description.** `section_contract.py` described itself as "a
completeness contract" and carried two contradictory comment blocks about empty
section appearance. Both corrected:

- the docstring now states it is a **whole-section presence check** and names
  explicitly what it does not establish (record-level completeness, faithful
  wording, every qualification surviving);
- the duplicate/contradictory appearance comment is replaced with the single
  resolved rule;
- two tests now guard this, including one that fails if the docstring ever asserts
  a completeness claim without denying it.

## 2. Sparse candidate — three attempts, all withheld

| # | Submission | Outcome | Reason |
|---|---|---|---|
| 1 | `sparse-conservative` | withheld | Imaginator **skipped**: no job_ad provided (my setup error) |
| 2 | `sparse2-conservative` | withheld | same — no job_ad |
| 3 | `sparse3-conservative` | withheld | `source_facts_missing` (reviewer reported a source fact missing) |

Attempts 1 and 2 are **pre-submission automation failures**, not product
behaviour: the pipeline correctly skipped the Imaginator because the form had no
job ad. They are counted separately from submission 3, which is a genuine
review-level refusal.

**The section contract was never implicated.** Verified:
`source_section_missing` does not appear in the service logs for any of these
runs. The contract runs at the approval point, and none of these runs reached it.

**It also had nothing to object to.** The sparse source backs only
Professional Summary, Professional Experience and Technical Skills. It contains
no Education, Projects or Certifications content, so the contract correctly
demands nothing for them. Confirmed locally:

```
source_backed: {'Professional Summary': 1, 'Professional Experience': 3,
                'Technical Skills': 1}
source has Education/Projects/Certs: []
```

**What this means for requirement 2.** The sparse acceptance could **not be
demonstrated on the website**, because every sparse submission was withheld
before export. There are therefore no sparse PDF/DOCX files to inspect for the
five-core-heading behaviour. The behaviour is covered by unit tests (which render
real PDF/DOCX from a sparse payload and assert all five headings present, empty,
with no invented filler), but that is not the same claim as a live export and is
not reported as one.

## 3. Imaginative acceptance — not attempted

Stopped deliberately. It requires an approved run on the same revision, and the
sparse path it would have shared is blocked at review. Running it on the full
fixture would restate the earlier acceptance rather than close the sparse gap.

## Consolidated status

| Requirement | Status |
|---|---|
| 1. Correct contradictory status + overstated description | **CLOSED** |
| 2. Sparse candidate website + PDF/DOCX verification | **NOT CLOSED** — blocked by `source_facts_missing` at review; 3 attempts withheld, 0 exports produced |
| 3. Imaginative on same build | **NOT CLOSED** — not attempted |
| 4. Record every attempt | **CLOSED** for the work performed above |

## Remaining failure, named

Sparse candidates are refused at the reviewer stage with `source_facts_missing`
before export. That is a **review-level** refusal, not a section-structure or
exporter defect. Whether it is correct behaviour (a sparse résumé genuinely
lacking facts the reviewer required) or an over-strict reviewer on short inputs
is **not established here** and would need its own investigation. It is outside
the scope I was given, and I am not claiming it as a false withhold.

## Submissions this session

| # | Type | Outcome |
|---|---|---|
| 1 | sparse (no job ad) | withheld — Imaginator skipped, pre-submission setup error |
| 2 | sparse2 (no job ad) | withheld — same |
| 3 | sparse3 (with job ad) | withheld — `source_facts_missing` |

3 submissions, 0 approved, 3 withheld, 0 retries, 0 timeouts, 0 false approvals.
2 of the 3 are automation-setup failures, not product behaviour.
