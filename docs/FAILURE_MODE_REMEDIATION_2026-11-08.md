# Failure-mode remediation — implementation status

Measured window: 20 correction runs, 2026-11-08T04:29 → 2026-11-09T04:13.
11 approved, 9 withheld. Every withheld run showed the user the same sentence:
*"The generated resume could not be verified and needs review."*

**Six of the nine withholds were our own infrastructure.** Only three concerned the
candidate's content. This document records what shipped, what was reverted and
why, and what remains.

---

## Evidence base

Every run received **the full 120s correction budget** — none were starved by
pipeline time:

```
22:37:54 pipe 20.1 budget 120.0 upstream 300
22:42:06 pipe 82.5 budget 120.0 upstream 300
...
04:12:21 pipe 46.1 budget 120.0 upstream 300
```

So the timeouts were **the 120s ceiling itself**, not upstream starvation.

Phase timing on the clearest failure:

| Phase | Run 04:08:55 | Run 04:12:21 |
|---|---|---|
| `source_inventory` | 45.7s | 33.6s |
| `initial_review` | **74.3s → deadline** | 59.2s → ok |
| Available | 120s | 120s |
| Needed | ≥ 120s | 92.8s |

`source_inventory` is a fixed 30–46s overhead before any review work begins.
Review then needs 59–74s. The sum sits exactly on the 120s line, so provider
jitter decided the outcome.

The two live `malformed_output` withholds were both `phase: initial_review`,
`constraint: duplicate_line_verdict`, after **4.9s of a 120s budget** — a review
envelope failing a mechanical check with 114s unspent.

---

## Shipped

### 1. `review_message` split by whose failure it was — **all 9**

`_withhold_correction` hardcoded one string for every reason, so a 74-second
provider timeout and a genuinely unplaceable source fact produced identical text
that reads as a judgement on the candidate's document.

Now `_review_message_for(reason)` selects between two classes:

- **infrastructure** (`_INFRASTRUCTURE_REASONS`): *"Our review service could not
  finish checking your resume in time. Your resume was not rejected — please try
  again."*
- **content** (`_CONTENT_REASONS`): *"We could not confirm every detail of your
  resume against the document you uploaded…"*

- **actionable** (`_ACTIONABLE_REASONS`, currently `receipt_unavailable`):
  *"Your resume passed review, but it could not be released to a signed-in
  account. Please sign in to continue."*

  This class is neither of the other two. Retrying cannot fix it, and it is not a
  judgement on the document: a receipt that could not be bound to an owner fails
  identically every time the visitor stays logged out. "Please try again" would be
  as wrong here as "your resume could not be verified" — both send the user
  somewhere useless.

An unknown reason defaults to **infrastructure** (retryable, non-accusatory):
inviting a retry costs one request, while telling someone to re-check a resume
that was actually fine costs them work.

New `retryable` field on `AnalysisResponse` is the machine-readable twin, so a
client offers a retry only when it would help. It is declared in `models.py` —
`AnalysisResponse` silently drops undeclared keys, the defect that once hid
`metrics_support`.

### 2. Correction deadline 120s → 220s — **3 (all timeouts)**

- `app.MAX_CORRECTION_DEADLINE_SECONDS`: 120 → **220**
- `source_correction.MAX_TIMEOUT_SECONDS`: 140 → **220** (must stay ≥ the above,
  or the granted budget is rejected as `invalid_input`)

`PROVIDER_TIMEOUT_CEILING_SECONDS` stays at 120 so one slow phase still cannot
consume the whole budget. `correction_budget_seconds()` still clamps to
`min(cap, upstream − 20 − elapsed)`, so this cannot overrun the caller: the JSON
route (180s) automatically keeps the smaller remainder. Only the SSE route (300s)
can now use the full 220s, which is exactly where the measurement said 173.5s was
available and the old cap discarded 53.5s of it.

Run 04:08:55 would have had inventory 45.7s plus up to 180s remaining for a review
needing 59–74s.

### 3. Bounded shape re-ask for a malformed REVIEW envelope — **2**

There was no retry at all: one bad field ended the run. A malformed review
envelope now gets **one** corrective re-ask naming the failed check
(`answer_shape_hint`), bounded identically to the existing unjudged-review
recovery:

- at most once per correction (`shape_retried`);
- never across a changed document (`shape_digest`);
- never when the deadline cannot fund a real call;
- never when the adapter's ordinary-call budget is spent (otherwise the refusal
  surfaces as a misleading `provider_error`);
- never a blind replay — the same constraint failing twice ends it.

**Reviews are re-asked; edits are not.** A malformed repair envelope still fails
closed. The model has already been told which spans to fix; replaying a
content-bearing edit is the very thing the `_Invalid` guarantee prevents. This
asymmetry is deliberate, and is why the retry lives in the review path only.

`_Invalid` gained an optional `constraint`, so the failing check is identifiable
(the same gap `MalformedReviewAnswer` already closed).

---

## Shipped, larger than planned

### 4. Free-trial visitors can now receive a resume — **1**

This was the worst finding, and the fix is more than a message tweak.

`app.py` withheld as `receipt_unavailable` whenever no **account** owner was
present. The orchestrator sends `x-cogito-owner-id` only when `req.user` exists,
so a free-trial visitor's resume was **corrected, reviewed, approved, and then
discarded** for being logged out. That reason is also outside the engine's
`allowed_reasons` set and is produced *after* `correction.status == "approved"` —
the approval genuinely happened.

The fix keeps the security invariant rather than weakening it:

- New `x-cogito-visitor-id` header, separate from the account header. A visitor
  has no account, so the owner header is legitimately empty; sending a device id
  there would mint an approval that *looks* account-owned.
- `resolve_publication_owner(request)` prefers a signed-in account, else falls
  back to the visitor, and reports which kind it used.
- The visitor owner is **namespaced** `visitor:<id>`. An account id is a verified
  login; a device id is a clearable cookie. If they could produce the same
  receipt owner, a visitor approval would be indistinguishable from an account
  approval at export.
- Malformed, absent, or browser-`fallback_` ids yield **no owner**, so the
  approval is withheld rather than bound to something untrustworthy.
- `server.cjs` forwards the visitor id on both analyze branches (SSE and JSON),
  reading it from the `X-Visitor-ID` header the client already sends (the
  multipart body is parsed later than that call).
- The export route now accepts a visitor owner, presenting the same
  `visitor:<id>` value the receipt was bound to.
- New `publication_owner_kind` field (`account` | `visitor`) tells the client
  which kind of approval it holds.

Proven offline end-to-end:

```
receipt minted: True
visitor export: True ok
account presenting visitor receipt: False receipt_owner_mismatch
no owner presented: False receipt_owner_mismatch
```

A visitor gets their resume and **cannot** spend someone else's approval.

---

## Implemented, then REVERTED — do not re-attempt without a new approach

Both were caught by `tests/test_route_two_documents.py`. They are recorded because
the fixes look obviously correct and will be proposed again.

### No-op repair: "resolve against the source" — REVERTED

Plan: when a repair returns a flagged span unchanged, break the tie with "does the
span appear verbatim in a source document?".

**This approves fabrication.** Caught by `[transplanted_qualification]`: appending
`HVAC controls` to the electrical-systems line was approved, because the phrase
*does* occur in the sources — under a **different record**. Attribution is about
which record a fact belongs to, not whether the fact exists somewhere. A global
"appears in a source" test is exactly the transplant the engine exists to reject.

A correct tie-break needs the citation the reviewer failed to give — which is why
the repair had nothing to work with — so nothing can be resolved locally.
Withholding stands. The shipped change is that the reason is no longer reported as
a content failure (item 1).

### Uncertain claim: deterministic restore — REVERTED

Plan: `source_completeness_uncertain` fires when nothing is flagged, so repair is
unreachable; run the same deterministic restorations the `missing` branch uses.

**Also approves a transplant**, for the same reason: inserting the source's
complete skill line makes a fact the draft *moved* (into another record) look
sourced. Also reverted.

Those two withholds (**2 of 9**) therefore remain. They are genuine: the reviewer
would not confirm a required fact and supplied no citation to check. What changed
is that the user is now told which class of failure it is and what to do, instead
of reading that their resume "could not be verified".

---

## Not changed

- `source_facts_missing` (**1**). Failing is correct: if the draft has nowhere
  faithful to attach the fact, synthesising a place is fabrication. The engine
  already tries skill restoration, then model restoration, then withholds.
  Improved only in messaging.
- The other engine reasons keep their behaviour; only the user-facing message and
  `retryable` changed.

---

## Where the numbers land

| Class | Was | Now |
|---|---|---|
| Timeouts | 3 withheld | recoverable (220s) |
| Malformed review envelope | 2 withheld | 1 re-ask, then withhold |
| No-op repair | 1 withheld | withheld (reverted; message fixed) |
| Missing receipt (free trial) | 1 withheld | **delivered** |
| `source_completeness_uncertain` | 2 withheld | withheld (reverted; message fixed) |
| `source_facts_missing` | 1 withheld | withheld (correct) |

Of 9 observed withholds: **4 removed**, 3 remain genuinely unrecoverable, and all
9 now report the correct cause to the user. The failure rate moves from ~45% to
~25% — and the ~25% that remain no longer accuse a document that was fine.

---

## Verification

```
cd Imagination_Resume_Module/.worktrees/resumeDraft_fix
OPENROUTER_API_KEY_1=x OPENROUTER_API_KEY_2=y IMAGINATOR_AUTH_TOKEN=z \
  STAGE_COOLDOWN_SECONDS=0 /home/skystarved/Render_Dockers/.venv/bin/python \
  -m pytest tests/ -o addopts='' -p no:cacheprovider -q \
  --ignore=tests/test_loader_source_preservation_offline.py
```

**1800 passed, 13 skipped** (was 1780; +20 new tests in
`tests/test_free_trial_publication_owner.py`).

Tests updated because their expectation legitimately changed:

- `test_source_correction.py::test_post_patch_review_is_required_and_schema_checked`
  — review_calls 2 → 3, the one bounded re-ask.
- `test_correction_timeout_attribution.py`, `test_correction_route_budget.py` —
  pinned the 120s cap; now assert the remainder arithmetic and the cap separately,
  so either can change without hiding a real defect.
- `test_completeness_diagnostic_fires.py` — the diagnostic moved into
  `_uncertain_required_items`, so the static guard follows it. Same defect guarded.

---

## Remaining / next

1. **Frontend**: surface `retryable` as a retry affordance, and
   `publication_owner_kind` as "sign in to save this to your account" for account
   approvals vs. direct download for visitor approvals.
2. **Deploy together**: imaginator (deadline, re-ask, owner) **and**
   orchestrator-backend (visitor forwarding, export gate). The visitor fix spans
   both; shipping one alone either keeps the old behaviour or refuses visitor
   exports.
3. **Re-measure** the withhold mix after deploy. The 20-event sample here is a
   lower bound — the log API caps at 20 entries and will not page older despite
   `hasMore`.
4. The 2 `source_completeness_uncertain` and 1 `no_op_repair` cases need a
   citation-aware tie-break, not a global source-containment test. That is a
   design change, not a patch.
