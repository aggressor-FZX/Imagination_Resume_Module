# Acceptance run after the trailing-clause fix — semantic judgments

Recorded 2026-11-08. Live website, actual retained server response.

## Imaginative

```
runId     (see captures/final/acceptance3/imaginative-a3.json)
status    completed, requires_review=false  -> approved
receipt   v5, signed
bridges   3 surviving: preventive maintenance, process improvement,
                        system maintenance
audit     0 mechanical defects, 6 disputed sentences
```

### Did the run exercise the fix?

**No — and that has to be said plainly.** The writer did not emit
`using blueprint specifications` or `state standards` on this run
(`writerHadBlueprint: false`, `writerHadStateStd: false`), so the new boundary
never had a clause to correct. This run is evidence the build delivers cleanly;
it is **not** independent evidence that the repair works.

The repair is evidenced by:

- the retained draft from run `43917767`, replayed through the real
  `preserve_source_facts`, which now removes both additions;
- four regression tests that fail on the pre-fix boundaries and pass with them;
- the two-document control `appended_unbacked_tools`, which now trims
  `using AutoCAD and SCADA` and publishes neither tool.

### The six disputed sentences, judged

| Sentence | Novel words | Judgment |
|---|---|---|
| Pulled and terminated branch circuits **during** 31 commercial buildouts | during | FAITHFUL — connective |
| Completed 8,000 apprenticeship hours **under licensed journeyman supervision** | supervision | FAITHFUL — restates "under a licensed journeyman" |
| Diagnosed and repaired switchgear faults **across** 9 substations | across | FAITHFUL — connective |
| Cut … 4.5 hours to 2.75 hours **through process improvement initiatives** | improvement, initiatives, process, through | **AUTHORISED** — `process improvement` is a signed surviving bridge whose evidence is this exact bullet ("Cut average fault-response time from 4.5 hours to 2.75 hours.") |
| Applied NFPA 70E standards and lockout/tagout procedures **during routine system maintenance** | during, routine | **AUTHORISED** — `system maintenance` is a signed surviving bridge (evidence: "Maintain electrical distribution and BAS controls for 6 commercial properties."). "routine" is an unbridged qualifier, noted below. |
| Reduced after-hours callouts by 27% **by implementing** thermographic inspections | implementing | FAITHFUL — "implementing" for "adding" |

Five are faithful paraphrases. Two are Imaginative's permitted inference, bound
into the signed receipt as surviving bridges — which is exactly the behaviour the
mode exists for.

**One residual observation, not a defect claim:** "routine" in the NFPA bullet is
an unbridged qualifier of the same lowercase shape as the retained additions. It
is far weaker (a cadence word, not a claimed input or credential), but it is the
same class and I am recording rather than dismissing it.

## Contrast with the retained false approval

| | retained `43917767` | this run |
|---|---|---|
| `blueprint specifications` | published | absent |
| `state standards` | published | absent |
| bridges | 2 | 3, all with matching evidence |
| disputed | 7 (2 unbridged additions) | 6 (0 unbridged additions) |

## Conservative

```
status  approved, receipt v5 verified ok
bridges none (Conservative passes no bridges, by design)
audit   0 mechanical defects, 3 disputed sentences
```

All three are connective paraphrases of the candidate's own facts and are
FAITHFUL:

| Sentence | Novel | Judgment |
|---|---|---|
| Pulled and terminated branch circuits **during** 31 commercial buildouts | during | FAITHFUL |
| Diagnosed and repaired switchgear faults **across** 9 substations | across | FAITHFUL |
| Reduced after-hours callouts by 27% **through** thermographic inspections | through | FAITHFUL |

Notably the Conservative writer produced "through thermographic inspections" and
the new boundary did NOT damage it — the trimmer judged "thermographic
inspections" grounded and left the clause alone.

## Exports

| Mode | PDF | DOCX |
|---|---|---|
| Conservative | 1 page, 0 blocks outside, 2 fonts, 6/6 headings ordered | 6/6 ordered |
| Imaginative | 1 page, 0 blocks outside, 2 fonts, 6/6 headings ordered | 6/6 ordered |

Neither PDF contains `blueprint` or `state standards`.

## Receipt controls

| Mode | Untouched | Tampered markdown |
|---|---|---|
| Conservative | valid `ok` | refused `receipt_content_mismatch` |
| Imaginative | valid `ok` | refused `receipt_content_mismatch` |

Verified on the deployed service, where the signing secret lives.

## Run log for this batch

| # | Mode | Outcome |
|---|---|---|
| 1 | Imaginative | approved |
| 2 | Conservative | approved |

2 submissions, 2 approved, 0 withheld, 0 retries, 0 timeouts, **0 false
approvals**.
