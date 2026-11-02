# Route map: which frontend path reaches which Imaginator handler

Verified by reading the code, not inferred from handler names. This document exists
because an earlier claim in this repository — that the multi-file path bypasses source
correction — was **wrong**, and the error came from reasoning about similar names rather
than following the actual calls.

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
