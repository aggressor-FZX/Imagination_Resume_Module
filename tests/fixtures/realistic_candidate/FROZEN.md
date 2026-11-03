# FROZEN — realistic candidate pack, 2026-11-03

**Status: FROZEN.** These are the exact files approved in principle for Proposal A. Any
edit to a source `.txt` invalidates the DOCX hashes below and requires a new freeze.

Rendered with `render_docx.py` (python-docx 1.1.2). Output is **byte-reproducible**:
`docProps/core.xml` carries a fixed `2013-12-23T23:15:00Z` timestamp, so re-running the
renderer produces identical bytes. Verified by rendering twice and comparing SHA-256.

## Frozen artifacts

| File | Bytes | SHA-256 |
|---|---|---|
| `candidate-main.docx` | 37359 | `1e7a69331913f649b001ae74bf852495404a965e116cb823479a48dccff0eb86` |
| `candidate-supplement.docx` | 37079 | `199b62d6febe6d5d87e7571afa82e72730b8829235fc90cf49df364f9f0cf261` |
| `intended-job-description.txt` | 920 | `06577fab0ca934efd8872e83dc691d76ac918710c6933e8245d0fce0e2163409` |
| `candidate-main.txt` (source) | — | `d5b7a9d408e43dfeb48f2c35837878b2d23b1800de1182bf6cb9aa2c9b4e4476` |
| `candidate-supplement.txt` (source) | — | `3c6f9fcfc6681ef70bf711c63f46d9fbded07004cb28fef9159221af08f031aa` |

## Verification performed at freeze time

| Check | Result |
|---|---|
| DOCX byte-reproducible across two renders | **yes** — identical SHA-256 |
| Every factual line of the source appears in the DOCX text | **yes** — 0 missing probes (15-char windows) |
| `SYNTHETIC-ONLY` markers preserved | main **15**, supplement **9** |
| Markers on every block-level factual line | yes (per `test_realistic_candidate_fixture.py`) |
| Capture guard accepts both documents | yes — `live_capture.assert_synthetic()` passes |
| Job description unchanged from review | yes — hash matches the reviewed version |

## The three live revisions Proposal A would run against

Read from Render deploy metadata, not from memory:

| Service | Service ID | Live deploy commit | Deploy time |
|---|---|---|---|
| **orchestrator-frontend** | `srv-d4fb5f4hg0os738sasgg` | `9e040e3043beb2163a2214bfcef0abdc099a6e99` | 2026-11-03T08:08:12Z |
| **orchestrator-backend** | `srv-d4fb5o4hg0os738sb3a0` | `9e040e3043beb2163a2214bfcef0abdc099a6e99` | 2026-11-03T08:08:13Z |
| **imaginator-resume-cowriter** | `srv-d3nf73ur433s73bh9j00` | `7cc415ba13057de35a10d2870a7ab361a424a82b` | 2026-11-03T16:35:32Z |

Note: frontend and orchestrator both report `9e040e3`, because that commit exists in
`Landing_Login_page` and deploys both services. The Imaginator is on `7cc415b`, which is
also `origin/master` for `Imagination_Resume_Module`.

## What "frozen" commits us to

1. These exact DOCX bytes are the uploaded inputs.
2. These exact job-description bytes are pasted into the job-description field.
3. No source edit before the run. If one is needed, the freeze is re-taken first.
4. The run is measured against F01–F22 in `REVIEW_PACKAGE.md`, not against a new ledger.
