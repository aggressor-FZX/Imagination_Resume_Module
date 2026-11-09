# Investigation: why sparse candidates are refused

Recorded 2026-11-08. Build `99996e40`. Scope: the `source_facts_missing` and
`receipt_unavailable` refusals seen on sparse submissions. No production change
was made by this investigation.

## Summary

Two **different** refusals were being conflated under "sparse candidates are
withheld". They have different causes, and one of them is not about content at
all.

| Run | Correction verdict | Refused by | Cause |
|---|---|---|---|
| `sparse3` | **withheld** `source_facts_missing` | Imaginator, reviewer stage | Reviewer declined to confirm source items |
| `sparse-noident` | **APPROVED** (`correction.end` status: approved) | Orchestrator, receipt gate | `receipt_unavailable` — no trusted owner identity on the request |

Evidence for the second row, from the service log of `52d81b60`:

```
04:13:54 correction.end {..., "reviews": 1, "status": "approved"}
```

The Imaginator **approved** that sparse résumé. The user-visible refund came from
`app.py:935-940`:

```python
owner = trusted_owner_identity(request)
output = _issue_publication_receipt(output, owner=owner)
if not owner or not output.get("publication_receipt") or not output.get(
        "approved_export_payload"):
    return _withhold_correction(reason="receipt_unavailable", ...)
```

So the content passed review, and publication was blocked because the approval
could not be attributed to an account.

## The receipt gate, stated plainly

`trusted_owner_identity` reads ONLY the `x-cogito-owner-id` header — deliberately
not body fields, because a caller must not choose whose approval it presents.

The orchestrator sends that header from the authenticated session at both
Imaginator call sites in the multi-file handler:

```js
...(req.user && req.user.id
    ? { 'X-Cogito-Owner-Id': String(req.user.id) }
    : {}),
```

`/api/analysis/multi-file` does **not** require authentication — `req.user` is
optional. When a request arrives without an authenticated session, the header is
omitted, the Imaginator still runs and approves, and the orchestrator then
discards the approval with `receipt_unavailable`.

## What this means

1. **`receipt_unavailable` is correct behaviour, not a defect.** An approval that
   cannot be attributed to an account must not be published — that is the control
   that stops any account spending another's approval. The gate is doing its job.

2. **It is a confusing user experience, and that is the real finding.** A user
   whose session expired sees "the generated resume could not be verified and
   needs review" after a full multi-minute pipeline that actually *succeeded*.
   The message names verification, not authentication. Nothing tells them to sign
   in again. The credits are spent and the work is discarded.

3. **The sparse3 `source_facts_missing` refusal is a separate question and is NOT
   established as wrong.** The reviewer declined to confirm source items. Whether
   that is correct (a sparse résumé genuinely lacking required facts) or
   over-strict was not determined, and no change is proposed.

## Retracted earlier claim

An earlier note said sparse candidates are "refused at the reviewer stage with
`source_facts_missing`". That is true of `sparse3` but **false** of
`sparse-noident`, which the reviewer approved. The two must not be described with
one sentence.

## Not investigated here

- Why the reviewer marked items missing on `sparse3` (needs its own capture).
- Whether an expired session during a long pipeline should be detected earlier.
- The `sparse-conservative` / `sparse2-conservative` runs, which failed before
  submission because no job ad was provided (a setup error, not product
  behaviour).

## Recommendation

No code change is warranted on this evidence. If the UX point is worth acting on,
the smallest honest fix is to distinguish `receipt_unavailable` in the user-facing
message — "sign in to publish this result" — rather than reporting it as a
verification failure. That is a product decision, not something to slip in here.
