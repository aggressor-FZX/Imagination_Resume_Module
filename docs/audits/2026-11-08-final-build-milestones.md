# Final-build milestones and remaining failures

Recorded 2026-11-07. Live website runs against the deployed build
(`imaginator-resume-cowriter`, `orchestrator-backend-isj7`). All checks below
were performed on the live service, not on local test doubles.

## Milestones reached

| # | Milestone | Evidence |
|---|---|---|
| 1 | Live export-tampering controls | 4/4 tampered replays refused, each with a distinct reason; unmodified control 200. `docs/audits/2026-11-08-live-tamper-controls.json` |
| 2 | PDF section-order fix | Live PDF and DOCX both emit the six sections in `REQUIRED_HEADINGS` order and agree with each other. `docs/audits/2026-11-08-live-visual-check.json` |
| 3 | Coverage write-back across the next review | Proven by `test_a_downgraded_claim_is_not_reconciled_on_the_next_review`, and confirmed live: the withholding run now logs `uncertain_source_claims: 1` where the pre-fix code logged `0`. |
| 4 | Conservative live run, semantic audit | Approved, receipt issued, 0 mechanical defects, 0 disputed sentences. `captures/final/batch/audit-finalC.json` |

## Remaining failures (exact)

### 1. Imaginative mode still withholds — `source_completeness_uncertain`

Live run `b48e9c13-da39-47cb-bcb1-93954b2b3d56`:
```
required_source_claims: 30, source_claim_total: 39, uncertain_source_claims: 1
repairs: 2, reviews: 3, status: withheld
```
Two repairs ran and a third review could not clear the single remaining
uncertain claim. The withhold is now **diagnosable**: exactly one required source
claim was declined, after two repair attempts. It is not the "no reason given"
state the pre-fix code produced.

This is the only mode-specific delivery failure outstanding. Conservative mode
delivers; Imaginative does not.

### 2. DOCX section headings carry the `Normal` style

No outline/navigation structure, no heading semantics for screen readers or
Word's navigation pane. Cosmetic-to-accessibility impact; content and order are
correct. Not fixed.

### 3. Writer emits inconsistent bullet markers

Some bullets carry `- `, others do not, where the source has no markers at all.
Cosmetic. Not a preservation defect (the facts are present).

## What changed in this build

- `3dcf35b` PDF sections emitted in the promised order (template slot order).
- `d53d6be` over-claim downgrade written back to the mapping, not just a local.
- `5bc5f4a` cross-pass proof that a downgraded claim cannot be reconciled back.
- `745531c` audit tool checks employer heads against the source instead of
  assuming them (found by negative control: an injected "Northrop Grumman" head
  produced zero findings before this).

## Method note

The audit tool is an audit, not a gate. It does not change production behaviour,
and a novel word is not treated as an invented fact. Every one of its checks was
validated with a negative control before its clean result was trusted — the
employer-head hole above existed precisely because that had not been done.
