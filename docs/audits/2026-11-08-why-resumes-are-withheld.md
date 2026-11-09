# Why a résumé is withheld, and what the user is told

Recorded 2026-11-08. Build `99996e40`. Question asked: *why does the user see
"the generated resume could not be verified and needs review", and is the cause
"the candidate has nothing to offer the job"?*

## Short answer

**No, that is almost never the cause — and the system cannot even detect it.**

The correction engine **never sees the job ad**. `correct()` takes only the
draft and the candidate's source documents:

```
async def correct(markdown, sources, callback, *, timeout_seconds, observer,
                  permitted_bridges) -> CorrectionResult
```

`source_preservation.py` does not reference the job ad either. Neither module
can judge relevance, because neither is given the job description.

Proven: a pastry chef's résumé and a faithful rewrite of it, submitted against an
electrician role, is withheld with `source_facts_missing` — the same code a
perfectly relevant résumé gets. The engine refused it for a *reviewer* reason,
not for irrelevance.

## Every cause, categorised

There are **28 distinct reason codes** behind that one sentence. Grouped by what
they actually mean:

### A. Candidate content problems — 5 codes
| Code | Meaning |
|---|---|
| `source_facts_missing` | the reviewer said a fact from the source is not in the draft |
| `source_completeness_uncertain` | the reviewer would not confirm a required fact |
| `source_section_missing` | a section the source populates arrived empty |
| `attribution_conflict` | a fact appears under the wrong employer |
| `unresolved_claims` | still unsupported after two repair attempts |

### B. Nothing usable in the document — 2 codes
| Code | Meaning |
|---|---|
| `no_required_source_claims` | the source inventory found zero required claims |
| `no_supported_claims` | the reviewer confirmed no line at all |

**This is the closest thing to your hypothesis, and it is the rarest.**
`no_supported_claims` fires when the reviewer marks *every* line unsupported or
structural. It does not mean "irrelevant to the job" — it means the reviewer
would not stand behind any sentence. A short or poorly written résumé can hit it
without being irrelevant.

### C. Our own infrastructure — 11 codes
`provider_unavailable`, `provider_rate_limited`, `provider_rejected_request`,
`provider_error`, `provider_call_timeout_source_inventory`,
`timeout_source_inventory`, `timeout_repair`, `timeout_restore`,
`analysis_failed`, `pipeline_invalid`, `pipeline_budget_exhausted`

**This is the largest group and the one users are most likely to hit in
practice.** `timeout_initial_review` alone was observed repeatedly: the provider
backend took 23–59 seconds on a single inventory call, exhausting the 120-second
correction budget before the review could run. Retrying usually succeeds.

### D. The reviewer answered in a shape we reject — 5 codes
`malformed_output`, `malformed_source_inventory`, `no_op_repair`,
`privacy_redaction_failed`, `privacy_review_required`

Observed live: `duplicate_line_verdict` (the model put one line in two verdict
lists) and `no_op_repair` (the repair returned the line unchanged). Both are
model behaviour, not résumé quality.

### E. Not about the résumé at all — 5 codes
| Code | Meaning |
|---|---|
| `receipt_unavailable` | **no authenticated owner on the request** |
| `render_contract_ambiguous` | the approved markdown could not be parsed into a render payload |
| `section_contract_unavailable` | the section check itself could not run |
| `invalid_input` | malformed request |
| `source_documents_mismatch` | the submitted sources did not match the snapshot |

`receipt_unavailable` is worth calling out: investigated earlier today, it means
the Imaginator **approved** the résumé and the orchestrator discarded the approval
because the request carried no session. The user is told verification failed when
they actually just need to sign in.

## What the user is shown

Every one of the 28 codes produces the **same** banner:

> This draft needs review before you can use it
> The generated resume could not be verified and needs review.
> Reasons: `source_facts_missing`

The raw code is printed with no explanation. The banner says "could not be
verified", which describes none of groups C, D or E — and those are 21 of the 28.

**There is no `nothing_relevant_to_the_job` code.** It does not exist, and it
could not be produced without giving the correction engine the job ad.

## Why we are failing, honestly

| Cause | Frequency observed today | Is it the résumé's fault? |
|---|---|---|
| Provider latency/timeouts | most common | no |
| Reviewer malformed answer | 2 of 5 submissions | no |
| Missing owner identity | 1 submission | no |
| Real content uncertainty | occasional | partly |
| Nothing usable in the document | not observed | yes |

So the failure is **mostly ours, or the model's — not the candidate's**.

## What would actually help, in order

1. **Show the real reason, in words.** Map each code to one sentence: "our
   provider timed out — retry", "sign in to publish this", "the reviewer could not
   confirm one fact". This is the smallest change with the largest effect and it
   needs no new detection logic.

2. **Retry transient failures automatically.** Groups C and D are retryable and
   usually succeed on a second attempt. The user is currently asked to retry
   manually with no indication that retrying would work.

3. **Separate "we failed" from "your résumé failed".** A provider timeout should
   not look like a judgement on the candidate.

4. **Only then, if wanted:** a relevance signal. That is a *product* feature —
   it requires passing the job ad into a stage that currently never sees it, and
   deciding what "nothing relevant" means. It is not a bug fix, and building it
   because one cause *sounds* plausible would be the wrong order.

## Not established

Whether `source_facts_missing` on the sparse and pastry-chef runs is a *correct*
refusal. Both used a test double as the reviewer, not the live model, so neither
is evidence about production behaviour. Established only: the code is generic, and
it does not mean irrelevance.
