# Interaction and upload honesty — run 10, audited 2026-11-03

What this file is: an accurate label on **how** run 10 was driven, so that nothing which
depends on forced interaction is later cited as ordinary-user acceptance. It records the
method used, not a verdict about the product.

## Upload: supported mechanism, and it worked

| Attempt | Paths passed | Result |
|---|---|---|
| 1 | `/home/skystarved/Render_Dockers/.live-run-2026-11-02/synthetic-single.docx` (Linux) | **Failed** — the browser runtime resolves paths on Windows, so a WSL path is not a path it can open |
| 2 | `C:\Users\jeffd\AppData\Local\Temp\cogito-multifile\synthetic-single.docx` and `…synthetic-second.docx` (Windows) | **Succeeded** via the supported `setInputFiles()` |

Attempt 1 was a filesystem mismatch, not a product defect, and it was fixed by staging the
files to a Windows-accessible directory rather than by changing the page. That method is
retained: **stage to a Windows-accessible path, then use `setInputFiles()`**.

Both filenames **were** verified in the page UI before submitting — the run recorded
`names: ["synthetic-single.docx", "synthetic-second.docx"]` read back from the file input
after attachment. No `disabled` attribute was removed and no application state was
changed to bypass validation.

## Clicks: all forced — this is the honest limitation

Every interaction that moved the flow was a **forced** click (`click({ force: true })`)
or a JavaScript `element.click()`:

| Step | Method | Ordinary user interaction? |
|---|---|---|
| `Continue to description` | `btn.click({ force: true })` | **No** |
| `Analyze Resume` | `getByRole(...).click({ force: true })` | **No** |
| `Download PDF (LaTeX)` | `click({ force: true })` | **No** |
| `Download DOCX` | `click({ force: true })`, then a JS `b.click()` | **No** |

A forced click bypasses the browser's actionability checks (visibility, stability, hit-target
interception). It therefore proves the handler works; it proves nothing about whether a
person clicking normally would reach it.

**What this does and does not establish**

- **Established:** the upload path accepts two files; the submit handler issues the
  multi-file request; the export buttons are wired to a working export route; the
  pipeline, the gate and the exporter all behave as recorded.
- **NOT established:** that an ordinary user can complete this flow. The download
  section is the specific place where normal clicks earlier appeared not to reach the
  request, and forced clicks were then used to get past it.

## The original "didn't work" observation, re-characterised

Two separate things were conflated in the session-10 record:

1. **A likely overlay or hit-target problem.** The DOCX button was measured as visible and
   in-viewport with a non-zero box, yet a normal click produced no network activity, while a
   forced click did. That is the signature of an element covering the button or of the
   button's hit area being displaced — a genuine UI defect candidate.
2. **An instrumentation miss.** The request listener watched `/resume/export`, but the real
   request path is the orchestrator proxy `/api/ml/imagination/export`. Requests *were*
   firing (server-side: two 200 OK responses) while the browser listener reported none.

Point 2 is settled and corrected. **Point 1 is not settled** — forced clicks were used
afterwards, so successful saves do not explain why the normal click stalled. Diagnosing it
requires an overlay/hit-target check against a live results page.

## Why it was not re-tested now

Reproducing it needs a results page, which needs an analysis, which is **not authorised**.
This section deliberately stops at "unresolved" rather than spending to resolve it.

When it is next testable, the check is: with a results page open, use
`document.elementFromPoint(cx, cy)` at the button's centre and compare the returned node to
the button. If they differ, the blocker is an overlay; if they match, the handler
attachment or an `isTrusted` check is the suspect. Neither requires a paid analysis if an
existing result can be reopened.

## Rule going forward

Any future run must state, per interaction, whether it was a normal click or a forced/
programmatic one. A forced click may establish pipeline behaviour; it must never be
recorded as ordinary-user acceptance.
