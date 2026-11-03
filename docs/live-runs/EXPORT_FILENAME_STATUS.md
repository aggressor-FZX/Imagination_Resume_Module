# Export filename investigation — status

**Status: renderer mapping verified; historical request metadata unresolved.**

This file replaces the earlier framing, which overstated what had been established.

## Verified

The renderer's filename rule is deterministic, established by running it three times:

```
job_title = None                            -> "resume"
job_title = "Senior Electrician"             -> "Senior_Electrician"
job_title = "Machine Learning EngineerTest"  -> "Machine_Learning_EngineerTest"
```

(`resume_builder/build_resume_from_export.py`, `safe_title = re.sub(r"[^A-Za-z0-9_-]+",
"_", job_title or resume.headline or "resume")`.)

Both run-10 artifact names are reachable from that rule, so **the renderer is not the
suspect**. Tests pinning this: `tests/test_export_filename_binding.py` (10 tests,
portable).

Also verified: the download name is taken from the response's `filename` field
(`triggerDownload`), and that field is derived server-side from the same title the
request carried. So the wire title and the saved name cannot diverge for one export.

## Unresolved

**What title the run-10 export request actually carried.** That request body was not
captured. The two saved artifacts are `resume.pdf` (empty title) and
`Machine_Learning_EngineerTest.docx` (a title matching no form value), which means the
payload's title differed between the two exports and neither matched the form.

The specific string that leaked is **inferred, not observed**. It must not be reported
as a finding.

## Explicitly not done

- Not inferred from saved filenames. Doing that was the error in the earlier report: a
  filename tells you what the payload contained only if you already know the mapping,
  and using it to *prove* the mapping is circular.
- No analysis purchased to reproduce a title/state problem. Reproducing it needs a
  results page, which needs an analysis, which is not authorised.

## Forward-looking coverage added

Rather than reconstructing history, the mechanism is now constrained from a known
starting state forward:

| Test file | What it pins |
|---|---|
| `src/pages/__tests__/exportFilenameTitleBinding.test.ts` (5) | Derivation order in `adaptAnalysisToData`: the form title outranks a stale match-me context; the analysis's own role is the fallback; a stale title can never be inherited. Mutation-tested — letting `matchMeContext` outrank the form title fails it. |
| `src/services/__tests__/exportRequestIdentity.test.ts` (5) | Starts with stale analysis context, submits a new electrician result, captures the **actual export request payload** and the **chosen download filename**; asserts the new result's title wins throughout and that the stale string appears nowhere in the request. Also asserts the documented neutral fallback (`resume` / empty headline) when no title exists. |

These do not prove what happened in run 10. They prove that a stale title cannot reach
the payload or the filename through the paths a new analysis takes.

## To close it, when a results page is next available

1. Open the results page for a known job title.
2. Read the export request payload's `headline` directly (network panel, before the
   response).
3. Compare to the typed title and to `triggerDownload`'s filename argument.

That is an observation, not an inference, and requires no paid analysis if an existing
result can be reopened.
