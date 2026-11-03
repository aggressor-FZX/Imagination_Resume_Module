# Route map: which frontend path reaches which Imaginator handler

Verified by reading the code, not inferred from handler names. This document exists
because an earlier claim in this repository — that the multi-file path bypasses source
correction — was **wrong**, and the error came from reasoning about similar names rather
than following the actual calls.

## Status: re-verified 2026-11-02, now by execution rather than by reading

The table below was previously supported by line-number citations. It is now also
supported by a test that **runs the real orchestrator handler offline** with two synthetic
uploads and a recording HTTP transport, and asserts the outgoing request:

```
Login_Landing_page/backend/tests/integration/multiFileRoute.test.js
```

Recorded outbound calls for a two-file upload, in order:

```
POST https://docreader.example.test/process          <- one per file, in upload order
POST https://docreader.example.test/process
POST https://imaginator.example.test/research        <- job-ad researcher fork (optional)
POST https://imaginator.example.test/analyze         <- THE ROUTE. exactly once.
GET  https://job-search-api-h9rl.onrender.com/api/v1/jobs/search?...
```

The payload of that single `/analyze` call, for the two fixtures:

| Field | Observed |
|---|---|
| `resume_text` | the two texts joined with `\n\n---\n\n` |
| `source_documents` | 2 entries, `source_id` `document-0` / `document-1`, each `{source_id, text, kind: "candidate_document"}` |
| `experience` | 2 records, one per employer, each keeping its own title and dates |
| `education` | 2 records, one per institution, each carrying `source_document_index` |

**File count does not change the route.** One file and two files both produce exactly one
`POST {imagBase}/analyze`; only `source_documents` and `resume_text` grow.

**No orchestrator request targets an Imaginator `/api/analysis/multi-file`.** The test
asserts this explicitly.

## Verified chain

| Frontend | Orchestrator (`Login_Landing_page/backend/server.cjs`) | Imaginator (`Imagination_Resume_Module/app.py`) |
|---|---|---|
| Résumé Builder (`ResumeBuilder.tsx`) → `processMultiFileRealAnalysis` from `src/services/analysis.ts:602` | `POST /api/analysis/multi-file?stream=1` | `POST /analyze` → `analyze_resume` → `_analyze_resume_impl` |
| `imaginationService.analyzeResume` from `src/services/imaginationService.ts:98` | `POST /api/ml/imagination/analyze` | `POST /analyze` → `analyze_resume` → `_analyze_resume_impl` |

### Evidence

**Multi-file.** `server.cjs` line 3847 registers `/api/analysis/multi-file`. Inside it,
the Imaginator call is at line 4478 (SSE) and 4511 (JSON), both `url: ${imagBase}/analyze`.
The payload it builds at line 4452 includes:

```js
const imagPayload = {
  resume_text: combinedText,
  source_documents: sourceDocuments,   // line 4456 - correction's evidence
  job_ad: jobAd,
  confidence_threshold: 0.7,
  ...
};
```

So the multi-file route **does** send `source_documents`, which only
`_snapshot_original_documents` consumes — that is the gate into source correction.

**Single-file / imagination proxy.** `server.cjs` line 1641 registers
`/api/ml/imagination/analyze`; its call is `const url = `${base}/analyze`` at line 1671.

**The Imaginator handler.** `app.py` line 1499 registers `POST /analyze`
(`async def analyze_resume` at line 1500), whose body delegates to `_analyze_resume_impl`
(line 1514, and line 1527 for the SSE branch). Source correction is invoked at line 698
via `_run_source_correction`.

### Conclusion

Both frontend paths reach **the same** Imaginator handler, and that handler runs source
correction. So **source correction is not bypassed on the multi-file path.**

## The handler that is *not* reached

`Imagination_Resume_Module/app.py` also defines `POST /api/analysis/multi-file`
(registered at line 1606, `async def analyze_multi_file` at line 1610). That is an
Imaginator-local multipart endpoint. No
orchestrator route targets it: every `imagBase` call in `server.cjs` is `${imagBase}/analyze`,
and a search of `Login_Landing_page` finds no request to an Imaginator
`/api/analysis/multi-file`.

It still matters, because it shares the publication gate and is exercised by
`tests/test_composite_publication.py`. But it is **not** on the website's path.

Its tests are labelled for what they exercise:
`tests/test_local_multipart_handler.py`. They were previously named
`test_multifile_measurement.py` and called "multi-file acceptance", which claimed more
than they demonstrated — they never traverse the website's downstream route.

## Did the routing change? No.

Established by history, not by assumption:

```
git show 56517b8:backend/server.cjs   # the commit that added the endpoint
```

That commit introduced `POST /api/analysis/multi-file` with
`url: \`${imagBase}/analyze\`` (line 962 at that revision) — the same target the test
records today. `git log -S` for the `/analyze` call in this region returns only
`56517b8` (introduction) and `5ce76c3` (SSE token forwarding). No commit retargeted the
route.

The line numbers moved (the payload grew `source_documents` in `675e4a2`), the branch
did not.

## Why the earlier claim was wrong

The orchestrator route `/api/analysis/multi-file` and the Imaginator route
`/api/analysis/multi-file` have the same path. Reading only the Imaginator handler named
`analyze_multi_file` suggests the website's multi-file traffic lands there, where
correction is indeed absent. It does not: the website's traffic goes to `/analyze`.

That is exactly the inference the instruction warned against, and it was made here.

## Consequences

- A captured **single-file** run and a captured **multi-file** run go through the same
  correction code, so the retry resilience added in `dad6666` applies to both.
- Earlier statements that "multi-file does not run source correction, so retry resilience
  does not apply to it" are withdrawn. The capability was present; what failed on the
  website was availability and timing, on the single-file runs that were attempted.
- Multi-file remains **untested end to end on the website**. Nothing here claims
  otherwise: the route is shared, not the evidence.

## What this means for multi-file correctness

Because both paths end at `/analyze`, a two-file upload runs the SAME preservation and
correction code as a single-file upload — tested by delivering two documents as one
combined source. That code was written for one document and **silently damaged two**.

Measured on two synthetic resumes (`tests/test_source_preservation.py`, section 6),
before the fix:

| Rule | Single-document assumption | Two-document result |
|---|---|---|
| `source_education_records` | first Education section | rebuilt Education from one record, **deleting the other degree** |
| `source_summary` | first Summary section | read the second document's summary as an "appended proficiency list" and **truncated it** |
| `source_skills_lines` | first Skills section | replaced the block with **one document's** list, and emitted `---` as a skill |

None of these raised. Each is a replacement, so the output looked like a normal
preservation change — which is why this needed an assertion rather than a review.

The fix (2026-11-02) reads EVERY matching section for the source readers, and adds
"a source record the draft never mentions" to the education trigger. Tests:
`tests/test_source_preservation.py` section 6.

Still **genuinely open** for two documents after both fixes:

- cross-document transplant of a quantified bullet between two DIFFERENT employers whose
  records share a figure and a qualifier (employer scoping covers the same-employer case
  only);
- end-to-end behaviour with a real model on two documents. The multi-document tests are
  scripted; no paid two-file run has been captured.

## Correction, 2026-11-03: supporting a cross-document claim is not invalid

The merged-summary behaviour recorded above was a **bug**, and the framing was wrong:
synthesising supported facts from several cited documents is legitimate output, not a
defect to tolerate.

What went wrong, measured. `_validate_review` checks a draft span for content the sources
never state. Its own comment said a span "may legitimately restate several items at once",
but the call passed **one** document — `[sources[item["source_id"]]]` — while the claim
could cite several. Merging two faithful summaries therefore reported document-1's own word
"Facilities" as *introduced*, downgraded the representation to `uncertain`, and withheld an
entirely source-backed resume with `source_completeness_uncertain`. Two bullet points
approved; the same sentence joined by a space did not.

Fixed (2026-11-03) by judging a span against **every document the claim cites**:

```python
cited_documents = [sources[citation["source_id"]]
                   for citation in claim["citations"]
                   if citation["source_id"] in sources]
introduced = _introduces_beyond(span, cited_documents or [sources[item["source_id"]]],
                                item["quote"])
```

This is not a loosening of the gate:

- a word in **neither** cited document is still introduced, which is what catches an
  appended unsupported qualification;
- a **figure** the cited documents do not contain is still introduced, which is what
  catches a re-attributed number;
- a representation must still be carried by a citation whose offsets **cover** the item,
  so a fact cannot be claimed from a document that does not contain it;
- `_correct_unique_attribution` still refuses a source record's label under another
  record's name.

Evidence — `tests/test_route_two_documents.py`:

| Test | Pins |
|---|---|
| `test_supported_summary_approves_whether_the_clauses_are_split_by_a_newline_or_a_space` | PAIRED: the same supported summary approves as two lines AND as one |
| `test_an_unsupported_addition_or_a_moved_fact_is_still_withheld` (7 cases) | appended unbacked tools, transplanted employer / role / dates / achievement / qualification / figure — all still withheld |
| `test_preservation_repairs_a_summary_append_before_the_gate_sees_it` | the paired explanation for why a *summary* append publishes: preservation removes it, and the published document omits it |
| `test_the_approved_two_document_payload_renders_both_records` | approved payload → real `/resume/export` → real LaTeX + python-docx → both records read back from the files |
| `test_export_refuses_altered_content_under_the_existing_receipt_contract` (4 alterations) | the same receipt does NOT authorise appended claims, a changed employer, a moved date, or a dropped record |
| `test_export_refuses_the_approved_content_presented_under_another_owner` | the receipt does not travel to another account |

Mutation-checked in both directions: removing the addition check makes three controls
fail; removing the provenance fix makes the paired approval fail; disabling
`verify_receipt` makes all five export tests fail. No pipeline was added and the
publication gate was not weakened.
