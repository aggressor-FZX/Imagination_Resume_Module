# Resume PDF / DOCX export

**Endpoint:** `POST /resume/export` (proxied by orch as `/api/ml/imagination/export`).  
**Returns:** JSON `{ pdf_base64, docx_base64, filename }` — not a raw binary pipe.

## How it works (simple)

1. Orch forwards the analysis payload to Imaginator `/resume/export`.
2. `build_resume_from_payload_base64` builds LaTeX → `pdflatex`, plus a matching DOCX.
3. Both files are base64-encoded in the JSON response; the browser rebuilds the download.

## What broke (and the fix)

| Failure | Cause | Fix |
|---------|--------|-----|
| `'utf-8' codec can't decode byte 0xa2…` | `subprocess.run(..., text=True)` decoded pdflatex stdout as UTF-8 | Capture **bytes**, decode with `errors="replace"` |
| `Unicode character U+0091 not set up for use with LaTeX` | Smart quotes mis-decoded as Windows-1252 C1 controls | `sanitize_export_text()` remaps C1 → ASCII and strips remaining controls |

Sanitization also normalizes em-dashes / smart quotes before LaTeX escape and DOCX runs.

**Code:** `resume_builder/build_resume_from_export.py` (`sanitize_export_text`, `compile_latex`, `build_docx`).

## Related pipeline

Job-ad Researcher can be precomputed via `POST /research` and passed as `research_data` on `/analyze` so Stage 1 is skipped (orch fork-join). That path does not change export shape.

SSE token streaming for Drafter/StarEditor is documented in [STREAMING.md](./STREAMING.md); export still uses the final analysis JSON only (never draft preview text).
