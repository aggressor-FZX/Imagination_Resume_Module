# Synthetic correction acceptance evidence

This self-contained bundle exercises the repository's actual asynchronous
`source_correction.correct` orchestration with local scripted callbacks. Every
person, organization, credential, metric and date is invented. It does not
load candidate data, use a provider, or make network calls; socket connections
and DNS are blocked while the cases run. Python standard library only.

Run from `Imagination_Resume_Module`:

```sh
PYTHONDONTWRITEBYTECODE=1 /home/skystarved/Render_Dockers/.venv/bin/python tests/fixtures/correction_acceptance/generate.py
```

The generator overwrites **only** this directory's `results.json`, prints each
case and exits 1 if any expected-versus-actual assertion fails. `results.json`
contains the shared synthetic sources and answer key, each draft, explicit
expectations, the result including commentary and coverage, checks, and the
complete callback payload/response sequence. `source_correction_sha256` binds
the observations to the exact engine file read on that run; it does not bind
other files or prove the whole environment. Regenerate after the attribution
guard is integrated into `main` and inspect any unmet assertions or changed
callback contract. An unexpected callback phase or scripted line is recorded
as a script error and is not accepted as a passing safety outcome.

## Cases

* `supported_duration_correct_employer`: a supported six-month duration,
  metric, employer, two independent education records and unrelated skills
  must be retained exactly without repair.
* `unsupported_duration_repair`: a separately flagged unsupported two-week
  suffix is removed; the supported six-month duration and every other byte
  match the answer key. The entire patched draft is reviewed again.
* `repeated_invalid_repairs`: the repair callback twice substitutes another
  unsupported duration. Both responses are syntactically valid but fail the
  scripted subsequent reviews. The final output must be withheld after at most
  two repairs and three reviews.
* `revalidation_missing_fact`: despite an otherwise successful removal, the
  second review explicitly reports an original education fact missing. The
  engine must withhold. The trace shows the fact actually remains in the
  patched draft: the callback intentionally reports a conservative finding.
* `wrong_employer_misleading_review`: a reviewer wrongly approves Beta as the
  owner of an Alpha achievement and provides a real Alpha citation. Desired
  outcome: withhold without callback, schema or timeout failure.
* `education_cross_mixing_misleading_review`: the two institutions' degrees
  and years are exchanged while the reviewer claims the original source records
  are represented. Desired outcome: withhold without callback, schema or timeout
  failure. The intact education control appears in the first case.

## Observed local run and matrix

On 2026-11-01, both commands in this README plus the Python and Node suites in
`CORRECTION_EVIDENCE.md` exited **0**. Engine: **6/6 cases passed** with zero
failed assertions. Route: **16/16 JSON+SSE requests passed, 411 assertions,
zero failures**, including real `/resume/export` output from `pdflatex` and
`python-docx` (`pdf_bytes=111461`, `docx_bytes=37278` for each approved case).
The base commit is Imaginator `65198c95cd524455daf1dc086ae4d1d25f5efbbc`;
Login `675e4a21746c98975cb1f72ea7d8ea0ccbf65a00`; Job Search
`ac5c2e3b20951b73bcb9d387082ade2535be5c3b`. These commit IDs are bases, not
hashes of the uncommitted implementation. `results.json` records
`source_correction_sha256`; `route_results.json` records SHA-256 of the route,
correction, provider, gate, verifier and export modules plus renderer
dependency paths. No deployment or model call occurred.

## Acceptance matrix

| Case | Engine | Route JSON/SSE | Real export | Live provider |
| --- | --- | --- | --- | --- |
| Supported duration, correct employer, two education records | PASS (unchanged) | PASS (approved, receipt, export payload) | PASS (PDF/DOCX contain both records) | NOT RUN |
| Unsupported duration corrected | PASS (`within 2 weeks` removed, 6 months retained) | PASS (approved; reported correction commentary; re-review) | PASS (exact answer key) | NOT RUN |
| Wrong employer corrected to its single source owner | PASS (`restored 1 source-owned record label`) | PASS (approved; accomplishment and dates retained) | PASS (Alpha/aligned content; wrong pairing absent) | NOT RUN |
| Unresolved multi-owner/cyclic attribution negative control | PASS (withheld `attribution_conflict`) | PASS (withheld, no receipt/export) | PASS (export not called; no receipt) | NOT RUN |
| Bounded repeated invalid repairs | PASS (withheld `unresolved_claims`, 2 repairs/3 reviews) | PASS (withheld) | NOT RUN (correctly withheld) | NOT RUN |
| Post-repair missing fact / numeric gate | PASS (withheld) | PASS (withheld) | NOT RUN (correctly withheld) | NOT RUN |
| Default disclosure boundary | NOT APPLICABLE | PASS (`privacy_review_required`, no provider request) | NOT RUN (correctly withheld) | NOT RUN |

Real export confirms the approved Markdown and `approved_export_payload`
reach both formats; it does not establish that every possible layout survives
rendering. `export` results are in `route_results.json` under `runs[].export`;
the binaries themselves are not stored.

## Model test doubles

`generate.py` uses `inventory()` and `reviewed()` scripted callbacks—not a model.
`route_evidence.py` uses `ScriptedTransport` behind the real `CorrectionProvider`
request/adapter/validation path. Both are labelled test doubles; the reviewer
reads record bodies and never calls a provider. The pipeline and disclosure
boundary are also doubled for positive cases. No result here is a semantic or
live-provider evaluation.

The comparison checks specific results, exact output where applicable, callback
sequences, bounded calls, unrelated-fact retention at each post-patch review,
and commentary's explicit need for independent review. Commentary reports the
number of spans flagged by the reviewer, not their text or a semantic reason for
each edit. For the two
adversarial cases, withholding is not considered valid when caused by a
callback exception, malformed review, or timeout. Inspect the recorded reason:
withholding for some unrelated reason still does not prove an attribution guard
is responsible.

These cases are **scripted controls**, not evidence that a model recognizes
unsupported facts, nor proof of semantic entailment, complete attribution
protection, real-provider behavior, production privacy, or PDF/DOCX parity.
Passing examples cannot establish that all wrong associations are blocked. The
source inventory and reviewer are test doubles and may deliberately lie. The
generator tests this small set of exact examples against the current local
engine, not a live deployment or a production publication workflow.

## Actual `/analyze` route evidence

`route_evidence.py` is a separate generator: it never imports or runs
`generate.py` and overwrites only `route_results.json`. From this repository:

```sh
env -i PATH=/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 /home/skystarved/Render_Dockers/.venv/bin/python tests/fixtures/correction_acceptance/route_evidence.py
```

It uses the project virtual environment, FastAPI's in-process test client, and
the actual `/analyze` JSON and SSE route, correction engine, provider adapter,
publication gate, numeric verifier, export parser and receipt issuer. It
records actual response bodies, parsed SSE events, request and draft, correction
observations, every synthetic provider payload/response and independent
expected-versus-actual checks. The fixed synthetic signing key is not a real
credential; resulting receipts are synthetic and expire. Timing, receipts and
other response fields can change between runs. Selected working-tree file
hashes are recorded, not a complete reproducible environment or a commit.

The source is a complete synthetic resume with explicit experience and education
sections, more than the route's minimum word count, and two education records.
The independently authored answer key is fixed before any provider decisions;
the output intentionally has an extra blank line because the route rejects a
verbatim source copy as fallback. The pipeline output, authentication, provider
transport, inventory and review decisions are controlled doubles. To exercise
correction, the synthetic cases explicitly bypass the **normally closed**
disclosure boundary; a separate control leaves it closed. No model is called.
The script clears inherited environment variables, disables both dotenv loading
paths before importing the app, blocks IP socket connections and sends plus DNS,
and uses in-memory transport. Local Unix sockets remain available to the test
client. This is not an OS-level network sandbox.

Each case runs once as JSON and once as SSE:

* `supported_duration_two_education`: unchanged six-month achievement, both
  education records, approved export payload and receipt.
* `corrected_duration`: remove the unsupported two-week suffix; require exact
  answer-key output and the complete post-patch review of retained facts.
* `wrong_employer_lying_reviewer`: a lying reviewer assigns Alpha's record to
  Betaa; the deterministic guard restores the single source owner and the route
  publishes the corrected document and commentary.
* `unresolved_attribution_negative_control`: a lying reviewer approves two
  exchanged education records; because the labels form a rewrite cycle with no
  unique owner, the route withholds `attribution_conflict` rather than guessing.
* `bounded_unresolved`: two valid repairs introduce further unsupported spans;
  the route withholds after six provider calls, including both re-reviews.
* `post_patch_missing_fact`: the reviewer conservatively marks a retained
  education record missing after correction; the route withholds.
* `post_correction_numeric_gate`: a lying re-review approves a repair that
  introduces 91 weeks; the real post-correction numeric check withholds it.
* `default_privacy_boundary`: no disclosure bypass; correction is withheld
  before the fake provider receives a request.

The local 2026-11-01 run matched all 16 JSON/SSE requests; inspect
`route_results.json` for the exact checks and file hashes. Approved cases also
call the real `/resume/export` route and extract text from the produced PDF
(`pdftotext -layout`) and DOCX (`python-docx`). This is a limited synthetic
observation against an **uncommitted working tree**, not suite completion, a
live-service test, real-provider acceptance, or a general semantic guarantee.
Missing coverage includes other document layouts and associations, multiple
source documents, provider/network failures and timeouts, and full pipeline
generation. Re-run after integration; failed checks remain visible in the
artifact and cause a nonzero exit.

## Privacy coverage and operational limits

Synthetic sentinel tests cover Imaginator's provider HTTP/envelope errors,
adapter failures, stage parser and orchestrator exceptions, normal integration,
and `/analyze` JSON/SSE; Node tests cover authenticated microservice/proxy,
parser/resilience, persistence/cost/session error paths and single/multi-file
route logging with doubles. Job Search's selected privacy suite (15 passed) now
runs with the secure-storage singleton redirected to a disposable temp path via
`JOB_SEARCH_SECURE_STORAGE_PATH`, asserted by an autouse fixture; the tracked
encrypted file's hash is unchanged by that run. Tests do not cover all
third-party, infrastructure, warning-level HTTP, broker, persistence or upstream
logs. Existing normal generation can send structured candidate data before
correction's disclosure gate; the gate only controls the new correction request.
Do not deploy this branch as-is.

The attribution guard is deterministic but deliberately narrow: it only
rewrites a literal `label: body` line when that body has exactly one explicit
source owner, and it withholds when the body has multiple owners or when the
rewrites form a label cycle (which could silently swap two accomplishments).
Paraphrased bodies, multiline records and unknown aliases are not resolved. It
can still produce a false conflict for two unrelated records that share a body;
it is a bounded counterexample fix, not an ownership proof.

For full reproducibility, `CORRECTION_EVIDENCE.md` records exact verification
commands, exit codes, an initial failing run and excluded baseline failures.