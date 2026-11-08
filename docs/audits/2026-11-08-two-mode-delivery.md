# Two-mode delivery on the final build

Recorded 2026-11-07. Both rows are live website runs against the deployed
service, each captured from the real path and audited semantically.

| Run | Mode | Outcome | Receipt | Mechanical defects | Disputed sentences |
|---|---|---|---|---|---|
| `finalC-conservative-1` | Conservative | approved | yes | 0 | 0 |
| `finalC-imaginative-1` | Imaginative | approved | yes | 0 | 0 |

## Imaginative detail

```
capture dd68fc90-bf1e-4f9d-a51a-6c20c76e196a
1 review, 0 repairs, 0 restores
supported 33 / structural 1 / represented 39 of 39
uncertain 0 / missing 0 / unsupported 0
6 bridges offered, 6 accepted
  preventive maintenance, fault diagnosis, process improvement,
  safety training, system maintenance, performance optimization
```

Imaginative therefore delivers under the fixed build, with skill bridging active
and every bridged label accepted by the reviewer against the candidate's own
evidence.

## What this does NOT establish

A single approved Imaginative run does not prove the `source_completeness_uncertain`
withhold is eliminated. The withholding shape seen on `b48e9c13` involved
narrowed or removed bridges (which force an extra repair/review cycle); the
approved run above had all six bridges accepted and needed no repair. The
withhold path is therefore *not* exercised here, and is still open — see
`2026-11-08-b48e9c13-forensics.md`.

## Controls

The audit tool is only trusted because it has been shown to fail:

- Injected omission, ownership move and fabricated employer into a clean output:
  the audit reported 2 mechanical defects and 1 disputed finding naming
  `Northrop Grumman`.
- That negative control lives in `captures/negative-controls/` and is NOT part of
  the audited batch.
