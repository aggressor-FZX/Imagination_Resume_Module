# Correction boundary — acceptance report (2026-11-02, session 2)

Commit `51affd2`. Deploy `dep-dbkdd6vb9u1s73fdrdtg`.

This session did **not** restart the audit. It closed the specific defects named in
the request, made real-provider probes replayable, and corrected the timing model from
measurements. Everything below is either a measurement or a test result; nothing is
inferred from a passing faithful-source probe.

**Scope of this document:** it records *implementation* acceptance only. Live delivery
acceptance — the website producing an approved, commented, receipted resume with
working PDF/DOCX downloads — is reported separately and was **not** achieved here.

## 1. Correction-boundary regressions closed

### 1.1 The retired wire format is gone from the suite

**Before:** 590 passed, 3 skipped, **22 failed** — every failure a scripted double
still emitting the pre-index-list shape.

**After:** **648 passed, 3 skipped, 0 failed.**

The 22 doubles were updated in place. Each kept its original acceptance assertion;
only the wire shape changed. Where a double had been asserting a field name that no
longer exists, the assertion was rewritten to assert the same *property* against the
new shape rather than deleted. Example: `assert set(payload) == {"contract", "sources"}`
became `assert set(payload) == {"contract", "segments"}` **plus** a new assertion that
no draft line is present in the inventory call, which is what the original test was
really protecting.

### 1.2 Candidate retrieval is separate from evidence approval

This was the substantive gap. `strict_review_from_indexed` used to attach citations to
a `supported` line straight from `line["source_segment_indices"]`, which the
application computes by lexical overlap. Overlap therefore *established* support. A
second path was worse: when a supported line had no candidate, the adapter
manufactured one from the best-overlapping segment, so the model never had to name
anything.

Now:

* retrieval still proposes candidates, because recall matters and a paraphrase can
  share no exact substring with its source;
* the reviewer must name the segments it relies on in an `evidence` map;
* a supported line that names none is `uncertain`, which withholds;
* a named segment that shares no content word with the line is rejected as
  unrelated text;
* each claim can only *remove* support — none of these paths can create approval.

### 1.3 Independent negative controls

`tests/test_correction_negative_controls.py` uses a **maximally permissive reviewer**
that approves every line, names every segment as evidence, and declares every fact
represented. Anything that still withholds, withheld for an application-side reason.
A faithful draft is approved by the same reviewer, so the controls are not vacuous.

| Defect class | Enforced by | Result |
|---|---|---|
| Altered metric (`30%`→`45%`) | `tier_a_verifier.verify_metrics` | FAIL, `unsupported_metrics: [45%]` |
| Wrong employer (bold heading) | `approved_resume.unsupported_entities` | `['Beta Works']` |
| Wrong education pairing (bold heading) | same | `['South College']` |
| Omitted fact | `source_correction` (`source_facts_missing`) | withheld, **0 repairs** |
| Irrelevant citation | `correction_protocol` evidence floor | `uncertain`, no citation |

#### Where a class is *not* caught — stated, not hidden

These limits are asserted as tests, so they cannot silently regress into
assumptions:

* `unsupported_entities` inspects **bold headings under known sections only**. It does
  not see `label: body` records, plain text, or omissions.
* `_attribution_ambiguity` fires only when the draft label is a **known** source label
  *and* the record body has more than one owner.
* Therefore an **invented `label: body` employer is not caught by any deterministic
  gate.** `test_invented_record_label_is_not_detected_by_the_engine` records this.
* With a permissive reviewer the correction engine alone approves an altered metric or
  an invented school. Enforcement is the **composed** route gate — correction, then
  `verify_metrics`, then `unsupported_entities`, then the signed receipt. The controls
  assert that composition, not a stronger claim about the engine.

A heuristic that would have closed the invented-label case was written and then
**reverted**: requiring every draft record label to appear in the source broke
legitimate generic section renaming (`Skills:` → `Tools:`), which
`test_unseen_alias_does_not_trigger_literal_attribution_guard` proved. A rule that
rejects correct rewrites is worse than a documented gap.

### 1.4 Disclosure covers the whole outgoing payload

The gate previously verified only `sources`. Draft lines, segment text and claim
quotes travelled in the same request and were never checked.

Now `disclosure.redact_payload` applies the same length-preserving substitution to
every free-text field actually transmitted, and `payload_disclosure_errors` re-walks
the exact outgoing structure to confirm no covered category remains. Width is
preserved, so the engine's code-point offsets stay valid, and the reply carries only
line ids and segment indices — never offsets or document text.

Two deliberate orderings, both pinned by tests:

* the **size bound runs before** the disclosure gate, so an oversized payload is
  rejected for being oversized regardless of content;
* the gate is evaluated against the **transmitted** structure, not the caller's
  claim about `sources`.

## 2. Real-provider probes are replayable

> **Evidence labelling (corrected).** The two recordings below are **real-provider
> probes driven by constructed synthetic drafts**, produced by
> `tools/capture_correction_case.py` against the correction engine directly. They are
> **not** captures of the original failed website runs: no `/analyze` request, no
> `/resume-builder` UI, and no writer-generated draft was involved. The original live
> failures were never captured, and their exact exchanges remain unrecovered. Nothing
> in this section is evidence about what the website did on 2026-11-02.

What the probes do establish is narrower and still useful: the recorder round-trips a
real provider exchange through the **real** `CorrectionProvider` and the **real**
`source_correction.correct`, so the decision is recomputed from provider bytes rather
than compared against a stored verdict.

Safety properties, each tested:

* refuses any text without the `SYNTHETIC-ONLY` marker (source *and* draft);
* refuses a request with no marker, so a recording cannot start mid-flight;
* writes `0600` in a git-ignored `captures/` directory;
* refuses a capture whose text does not match its own digest, or an unknown schema;
* replay blocks sockets entirely and asserts no connection is opened.

Two probes were recorded (spend: **$0.00226** total, 11,890 tokens):

| Probe | Result | Mechanism observed |
|---|---|---|
| `live-rewrite-01` | **approved**, 2 reviews + 1 repair, 4 calls | — |
| `live-rewrite-02` | withheld, 3 calls, reported `malformed_output` | **2 lines left unjudged** |

`live-rewrite-01` shows a paraphrased rewrite with added sections surviving a repair
and being approved by the validator. It says nothing about the website's own drafts.

### A mechanism that produces the `malformed_output` reason

`malformed_output` was the reported reason on website runs 2 and 3. Replaying probe
`live-rewrite-02` shows **one** way that reason arises: the model answered with
`supported`, `unsupported`, `structural`, `evidence`, `represented`, `missing` — and
**omitted `L000016` and `L000017`**, the two added Skills bullets, from every verdict
list.

The old code treated an omitted line as `uncertain`. Those lines then went to the
repair phase, where the model returned them **verbatim**, and `_apply_edits` correctly
rejected the no-op. The failure surfaced as `malformed_output` and had burned a paid
call.

This is a *candidate* mechanism for runs 2 and 3, not a demonstrated one: those runs'
provider responses were never captured, so the same reason code could have had a
different cause. Treating it as the cause would repeat exactly the mistake this
correction fixes.

Fixed without weakening anything:

* an omitted line now raises `UnjudgedLines` → `UnjudgedReview` → **`incomplete_review`**;
* a repair that reproduces the flagged span unchanged is **`no_op_repair`**, its own
  reason;
* both are added to the route's forwarded-reason allowlist (a test caught that they
  were being silently rewritten to `correction_failed`);
* the withheld draft is unchanged — it is still withheld, only attributed correctly.

Replay now consumes **2** calls instead of 3.

Rejected and reverted: issuing one bounded completion re-review that names the
unjudged lines. It would raise the documented six-call ceiling, and the probe ends at
the omission, so its effectiveness against a real model **cannot be shown**. An
unproven change to a safety bound is not a fix.

## 3. Execution time — measured, then bounded

### Which stage consumed the pipeline (run 6, from `stages.*` log lines)

| Stage | Duration | Share |
|---|---|---|
| Researcher | 0 ms | fork-join skip (`using precomputed research_data`) |
| **Drafter** | **78,858 ms** | **74%** |
| StarEditor | 23,402 ms | 22% |
| ATS score | ~1,583 ms | 2% |
| **Pipeline total** | **106,483 ms** | |

### The timeout of the route actually used

| Route | Timeout | Retry | Mechanism |
|---|---|---|---|
| JSON | **180 s** | up to 3 | `mlAuthMiddleware.makeAuthenticatedRequest` **hardcodes** it |
| SSE (multi-file) | **300 s** | none | `server.cjs` calls axios **directly**; middleware bypassed |

Both live-failing runs took the **SSE** branch. `app.UPSTREAM_CLIENT_TIMEOUT_SECONDS`
was 180 s for both, so the budget was computed against a ceiling **120 s lower than the
caller's actual deadline**. In run 6 that produced a 53,516 ms budget out of an
actually-available 173,516 ms, and correction then hit its own 120 s cap — not the
caller's.

### The bounded solution chosen

`_upstream_timeout_seconds(request)` detects the route from the `Accept` header the
orchestrator sets on the direct branch, and the budget is computed from the measured
ceiling. Guarantee preserved on both routes:

```
pipeline_elapsed + budget + 20s reserve <= that route's client timeout
```

The engine's own 120 s cap is **unchanged**, so this cannot extend total work; it
stops the budget being computed against the wrong ceiling. Twelve tests in
`tests/test_correction_route_budget.py` pin both measured constants against the
orchestrator source and recompute run 6 arithmetically.

**What this does and does not fix.** It removes a real 66.5 s under-allocation on the
SSE route. It does **not** make a slow Drafter faster, and it does not guarantee that
the review phase always fits: run 6 needed more than 39,074 ms of review after 14,442 ms
of inventory, and the measured Drafter alone can consume 79 s of the caller's budget.
Streaming was *not* chosen — no measurement supports it, and the request explicitly
forbids assuming it.

## 4. Spend (provider probes)

| Item | Cost |
|---|---|
| Probe `live-rewrite-01` | $0.001268 |
| Probe `live-rewrite-02` | $0.000995 |
| Debug attempt that reached no provider | $0.00 |
| **Total provider spend this session** | **$0.00226** |

These are direct OpenRouter charges from the probes, recorded here because they are
provider spend and must appear in the spending record. They are **separate** from
website credit consumption: no `/analyze` run was made this session, so no user credits
were spent. 1463→1265 credits were consumed by the six earlier website runs
(6 × 33 = 198), all before this session.

## 5. Status — two separate verdicts

### Implementation acceptance: PASS

Verified locally and deployed (`51affd2`, docs `166cc75`): the 22-doubles update, the
retrieval/evidence separation, the negative controls and their documented gaps, the
disclosure scope change, the probe capture/replay harness, the `incomplete_review` /
`no_op_repair` attribution, and the route-accurate budget. 648 passed, 3 skipped.

### Live delivery acceptance: NOT MET

No live `/analyze` run was made this session. The website has **not** been shown to
produce an approved resume, correction commentary, a valid publication receipt, or a
working PDF/DOCX download. The last six website runs (2026-11-02, earlier) all
withheld.

A withheld draft is a safety outcome, not delivery. The mission is therefore open.

Known limitations carried forward:

* the invented `label: body` employer gap (§1.3);
* review latency may exceed the remaining budget at large document sizes (§3);
* the original website-run exchanges were never captured, so their root causes remain
  unproven — the next live run must capture its own exchange.
