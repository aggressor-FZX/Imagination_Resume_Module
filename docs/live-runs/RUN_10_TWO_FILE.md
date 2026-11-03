# Run 10 — first two-file website delivery: approved, records correct, bullets embellished

**Date:** 2026-11-03
**Revision:** `47c4254` (live, verified by deploy SHA *and* by the presence of the
`cited_documents` provenance fix in the deployed source)
**Correlation ID:** `850ee3ea-ca2e-4210-92b4-0e57ed5149eb`
**Route:** website résumé builder → orchestrator `POST /api/analysis/multi-file?stream=1`
→ Imaginator `POST /analyze`
**Credits:** 968 → 935 (**33**, one analysis, no second submission)
**Outcome:** `correction_status = approved`, published with a receipt

This is the first two-document run through the website. It answers the routing and
preservation questions conclusively and exposes one remaining defect class.

## What was done, in the order the stopping rule requires

1. **Deployed revision verified before submitting.** Not by health 200 — the deployed
   commit's `source_correction.py` was read back from `origin/master` and the
   `cited_documents` call is present at line 710.
2. **Capture preflight passed** in the container: `READY`, `admitted_by: separate volume`,
   `EXIT=0`.
3. **One analysis submitted**, two synthetic DOCX files, 370-character job ad.
4. **Capture preserved** to `.live-run-2026-11-02/captures/run10/` (30,062 bytes,
   SHA-256 `c141846a0d6e…`) **before** anything else was touched.
5. **Delivered content inspected** against both sources (below).
6. **Capture disabled** (`CORRECTION_LIVE_CAPTURE=0`, redeployed, confirmed `capture=0`)
   and the remote capture **purged** (`files=0`).

## What the route and preservation got right

Confirmed from the capture, not inferred:

- the request reached `/analyze` (phases `source_inventory`, `review`);
- **two** `source_documents` with distinct ids — `document-0` (1,232 chars),
  `document-1` (676 chars);
- the reviewer's inventory covered all 35 source segments; `missing: []`;
- **approved on the initial review, `repair_calls = 0`**;
- pipeline 14.7 s, total 23.4 s, well inside the 120 s correction budget.

Confirmed from the published document (31 mechanical checks, all passing):

| Requirement | Result |
|---|---|
| Both employers' records | Cascade, Bluegate **and** Harborline all present |
| Both education records | Technical College (AAS, Electrical Systems Technology, 2016) and Community College (Certificate, Building Systems Maintenance, 2013) |
| No employer/role/date mixing | Each block carries only its own role and dates |
| No degree/institution mixing | Each institution keeps its own degree, field and year |
| Metrics correctly attributed | 40+ sites and 18% under Cascade; 12 buildings and 22% under Bluegate; neither crosses over |
| No duplicates | Every employer and institution appears exactly once |
| No invented figures | Every figure in the document traces to a source |
| No invented names | No capitalized entity absent from both sources |

**The multi-document preservation fix works on the live route.** Before it, this run would
have deleted the second degree and truncated the second summary. It did neither.

## What is still wrong: appended outcome clauses

Six bullets state an outcome neither source contains, appended to a faithful opening:

| Approved bullet (tail in bold) | Source line |
|---|---|
| Installed and maintained … across 40+ job sites, **ensuring compliance with NFPA 70E standards** | Installed and maintained commercial electrical systems across 40+ job sites. |
| Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices, **improving workforce safety and code adherence** | Trained 6 apprentices on lockout/tagout and NFPA 70E safe work practices. |
| Cut rework by 18% by standardizing panel labeling and inspection checklists, **increasing installation efficiency and quality control** | Cut rework by 18% by standardizing panel labeling and inspection checklists. |
| Maintained electrical distribution systems and HVAC controls for 12 commercial buildings, **ensuring continuous operations** | Maintained electrical distribution and HVAC controls for 12 commercial buildings. |
| Reduced unplanned downtime by 22% through implementation of preventive maintenance schedules, **enhancing system reliability** | Reduced unplanned downtime by 22% through a preventive maintenance schedule. |
| Completed 8,000 hours of supervised field work toward journeyman licensure, **gaining hands-on electrical construction experience** | Completed 8,000 hours of supervised field work toward licensure. |

Also rewritten though not strictly an addition: Harborline's "Assisted with …" became
"Supported … under licensed supervision", asserting supervision the source places on the
hours, not on the role.

### Why the gate did not stop it

Not a regression, and not the defect that was fixed this session. Two facts combine:

1. `_introduces_beyond` flags **capitalized names** and **unbacked figures**. The appended
   clauses are lowercase ordinary words — `ensuring`, `compliance`, `standards`,
   `improving`, `workforce`, `efficiency`, `quality` — so the lexical addition check
   reports nothing. Verified directly: run against this exact bullet it returns `[]`.
2. The only defence for that shape is the **reviewer** flagging the span, which routes it
   to repair. The live reviewer marked all 28 lines `supported` with `unsupported: []`.

So the check that would have caught it was never asked. This is the run-4 defect class —
an unsupported outcome appended to a supported clause — arriving through a reviewer that
approved everything, rather than through one that flagged it.

**Honest grading:** the four requirements as written are met — both documents' supported
facts retained, no mixing, no duplicates, no invented *qualifications*. The embellishment
is an extra defect class beyond those four, and it is a fabrication risk: "improving
workforce safety" is a claim the candidate never made. This is **not** a clean delivery.

## Not delivered / not verified

- **The PDF and DOCX were not downloaded.** The result page rendered the approved markdown
  and exposed Download buttons, but no `/resume/export` request left the browser on any
  attempt (patched `fetch`, mouse click at coordinates, forced click). Per the standing
  note, browser download behaviour is a separate follow-up criterion. The content verified
  above is the approved markdown as rendered by the site, **not** bytes read back from a
  compiled file. The offline suite does prove the exporters render this payload
  (`test_the_approved_two_document_payload_renders_both_records`), but that is not this
  run.
- Real-model repair recovery remains unexercised: `repair_calls = 0`.

## Cost

| | |
|---|---|
| Account credits | 33 (968 → 935) |
| Provider cost, this run | **not established.** The capture records the review and inventory exchanges; per-run `usage.cost` was not read out before the remote file was purged, and the local copy retains the response bodies but not the aggregated cost field. Tracked as **unknown**, not as zero. |

## Files

Under `.live-run-2026-11-02/captures/run10/` (outside the repo, deliberately uncommitted):

| File | Bytes | SHA-256 (12) |
|---|---|---|
| `850ee3ea-ca2e-4210-92b4-0e57ed5149eb.json` | 30062 | `c141846a0d6e` |
| `run10_approved.md` | 2079 | approved markdown as rendered |
| `run10_summary.json` | 866 | phases, sources, gate, timing |

Remote capture directory: empty. Capture flag: `0`.
