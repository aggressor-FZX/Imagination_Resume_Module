# Fixed-build acceptance batch (post compound-citation fix)

Recorded 2026-11-08. Live website, actual server responses retained verbatim.

## Result

| Mode | Outcome | Receipt | Deficiencies | Disputed | Unbridged additions |
|---|---|---|---|---|---|
| Conservative | approved | v5, verified `ok` | 0 | 0 | none |
| Imaginative | approved | v5, verified `ok` | 0 | 0 | **none** |

The two sentences that made run `43917767` a false approval are **absent** from
the delivered document:

```
'blueprint'       in final markdown: False
'state standards' in final markdown: False
```

Imaginative kept its one legitimate surviving bridge (`preventive maintenance`)
and dropped `process optimization` — so this is not "bridges off"; the mode still
infers, it just no longer ships unbacked elaboration.

## Every attempt, counted

| # | Mode | Outcome | Cause |
|---|---|---|---|
| 1 | Imaginative | **withheld** `malformed_output` | `constraint: duplicate_line_verdict` — the reviewer put one line in two verdict lists. Model formatting, classified precisely. |
| 2 | Imaginative | **withheld** `no_op_repair` | repair returned the flagged line unchanged (61 completion tokens). Model behaviour. |
| 3 | Imaginative | **approved** | captured, exported |
| 4 | Conservative | **withheld** `malformed_output` | second review rejected; distinct correlation id from #1 |
| 5 | Conservative | **approved** | captured, exported |

Totals over this batch: **5 submissions, 2 approved, 3 withheld, 0 false
approvals.** Two of the three withholds were malformed reviewer output and one was
a no-op repair — none was a coverage or faithfulness refusal, and none was caused
by the code change (which only alters which documents are passed as evidence).

Earlier this session, before this batch: 2 `timeout_initial_review` (provider
latency on the Venice backend, ~23s and ~39s inventory calls exhausting the
120 s correction budget). Recorded, not counted as product failures.

**Cumulative across every live attempt today: 7 submissions, 2 approved, 5
withheld, 0 false approvals.**

## Controls re-run on the new build

- Receipts verify `ok` for both modes (checked on the deployed service, where the
  signing secret lives — locally `verify_receipt` returns
  `receipt_signing_unavailable` and proves nothing).
- Tampered markdown refused: `receipt_content_mismatch` for both modes.
- Exports: both modes 1 page, 0 blocks outside the page, 2 embedded fonts, 6/6
  headings in required order in PDF and DOCX.

## What changed to produce this

`b0a1b02` — a skills-only document is no longer accepted as evidence of tool use
inside a role (the compound-citation defect). Demonstrated, with a regression
that fails on the pre-fix code.

## What did NOT change

The two additions were **not** caught by the deterministic check, and are still
not. `tests/test_retained_imaginative_false_approval.py` marks them
`xfail(strict=True)` with the reason, so the unmet requirement stays visible
rather than being quietly satisfied by a lucky model answer.

This accepted run is evidence that the build can deliver cleanly on this fixture.
It is **not** evidence that the added-lowercase-qualifier gap is closed: the same
input could still produce `using blueprint specifications` and the check would
still pass it.
