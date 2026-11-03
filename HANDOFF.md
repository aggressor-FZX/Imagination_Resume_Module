# Handoff — single-file frozen; the multi-file routing blocker was wrong and is withdrawn

**2026-11-02, session 8** (supersedes the session-7 header below).

- **Tests:** 981 passed, 3 skipped (`python -m pytest tests/ -q -o addopts="" -p no:cacheprovider`).
- **Tag:** `known-working-single-file-2026-11-02` → `73cf570` (kept; unchanged).
- **Branch:** `wip/correction-timeout-attribution`.
- **Capture is OFF** (`CORRECTION_LIVE_CAPTURE=0`), disclosure on
  (`CORRECTION_PROVIDER_DISCLOSURE=1`).
- **Credits:** unchanged. **No paid submission this session** — provider cost `$0.00`,
  every provider call in the tests is a local stub.

## Done this session (8)

- [x] **The "website cannot capture multi-file at all" blocker is withdrawn.** It came from
      reading `app.py`'s `analyze_multi_file`, an Imaginator-local multipart endpoint that no
      orchestrator route targets. The website's two-file traffic goes to `/analyze`.
      `docs/ROUTE_MAP.md` already said so; it is now backed by execution, not line numbers.
- [x] **The two-file chain is recorded end to end**, offline, with a recording transport:
      frontend URL → orchestrator handler → downstream host/path → payload.
      `Login_Landing_page/src/services/__tests__/multiFileRequest.test.ts` (3 tests) and
      `Login_Landing_page/backend/tests/integration/multiFileRoute.test.js` (6 tests).
- [x] **Routing has not changed.** `git show 56517b8:backend/server.cjs` already calls
      `${imagBase}/analyze`; `git log -S` finds no retargeting commit.
- [x] **A real two-document defect was found and fixed** — the consequence the routing
      question exposed. Preservation was single-document-only: it deleted the second
      document's degree, truncated the second summary, and emitted `---` as a skill. Fixed in
      `source_preservation.py`; 10 unit tests + 7 route tests; mutation-checked.
- [x] **Ten tests relabelled.** `tests/test_multifile_measurement.py` →
      `tests/test_local_multipart_handler.py`. They exercise the local multipart handler and no
      longer claim to be multi-file acceptance.
- [x] Freeze tag and manifest kept. `docs/live-runs/FREEZE_2026-11-02.md` carries a dated
      correction; item 4 struck through rather than deleted.

## Still open (two documents)

1. A writer that merges both summaries into ONE line is withheld
   (`source_completeness_uncertain`), not truncated. Safe, but still a rejection. Pinned by
   `test_a_merged_two_document_summary_is_withheld_not_silently_truncated`.
2. Cross-document transplant of a quantified bullet between two DIFFERENT employers sharing
   a figure and qualifier: `_employer_of` scoping covers only within one employer's record.
3. No captured two-file run with a real model. The acceptance target is reachable but un-met.

## Verification commands (this session)

```
source .venv/bin/activate && cd Imagination_Resume_Module
python -m pytest tests/ -q -o addopts="" -p no:cacheprovider           # 981 passed, 3 skipped
python -m pytest tests/test_source_preservation.py -q -o addopts=""    # incl. multi-document §6
python -m pytest tests/test_route_two_documents.py -q -o addopts=""    # real /analyze, 2 docs

cd ../Login_Landing_page
node node_modules/jest/bin/jest.js -c jest.integration.config.js \
  backend/tests/integration/multiFileRoute.test.js                     # 6 passed
node node_modules/jest/bin/jest.js -c jest.unit.config.cjs \
  src/services/__tests__/multiFileRequest.test.ts                      # 3 passed
```

Note: `jest.integration.config.js` also collects `.worktrees/**`, where several DB-backed
suites fail for pre-existing environment reasons. Those failures reproduce with this
session's files removed.

---

# Prior handoff (session 7) — kept for history

**2026-11-02, session 7.**

- **Tests:** 913 passed, 3 skipped.
- **Latest commit:** `a2f584f`, pushed to `master`, deployed, verified in-container.
- **Branch:** `wip/correction-timeout-attribution` == `origin/master`.
- **Capture is OFF** (`CORRECTION_LIVE_CAPTURE=0`), disclosure on
  (`CORRECTION_PROVIDER_DISCLOSURE=1`), container capture dir empty.
- **Credits:** 1034. No analysis was purchased this session.

## Done this session

- [x] **Repair ordering fixed.** `source_correction.py` withheld on a missing source fact
      *before* entering the repair path, so a review reporting both unsupported lines and
      omitted facts withheld immediately. That is why every live run recorded
      `repair_calls == 0` (run 4: 19 flagged; run 5: 13; runs 6–7: 3). Flagged spans are
      now repaired first and the coverage verdict is taken on the repaired draft.
- [x] **Recovery proven offline**, not just reachability. A separately labelled layer
      answers the phases the recordings never performed
      (`live_capture.replay_with_extensions`, `tests/support/controlled_answers.py`);
      `ControlledExtension` refuses to overwrite a recorded phase, so the artifact stays
      authoritative for everything it contains. A repairable draft is repaired,
      re-reviewed, validated, approved, and exported to real PDF and DOCX whose content
      is read back.
- [x] **Negative controls** on that path: repair cannot delete an accomplishment, invent a
      qualification, loop, or exceed `MAX_CALLS`.
- [x] **Five/six-section contradiction resolved.** The answer key's comment promised five
      headings while its own tuple listed four without Projects; the prompt forbade an
      empty section; `_ensure_all_sections` deleted an empty header. Now: all headings
      required, empty is correct, filler forbidden. Verified end to end that an empty
      `## Projects` parses to `[]` and renders nothing.
- [x] **Objection classes corrected.** Formatting / omission / unsupported addition / lost
      qualifier / reviewer error, each mapped to its remedy. "Electrical Systems
      Technology dropped" was wrongly called fabrication and is now **omission** (the
      draft drops a fact, it asserts nothing false). The concatenated bold education name
      is **formatting**, not fabrication.
- [x] **Phase order and upper bounds restored** where exact historical call counts had
      been relaxed to `> 0`.

## Known defects NOT fixed (writer layer)

These are demonstrated and classified; fixing them needs prompt changes, and each prompt
change has so far cost a paid submission to validate.

1. **`8,000 hours of supervised` → `hands-on … mastering core`** — lost qualifier plus an
   invented trailing clause. Two defects, two repairs.
2. **Skills block** invents a category structure and admits activity verbs
   (`installation`, `maintenance`) plus the rank `apprentice supervision`.
3. **Summary** appends a proficiency list past where the source summary ends.
4. **Degree field dropped** ("Electrical Systems Technology") — omission.
5. **Education bold name** concatenates degree + institution, so `unsupported_entities`
   rejects it. Formatting.
6. **Writer omits the Projects heading entirely.** Repair cannot fix this: it replaces a
   flagged span and cannot insert a newline. The structural tests therefore run against a
   draft with the heading inserted, and say so.

Full table with reasons: `docs/live-runs/REMAINING_DEFECTS.md` and
`docs/live-runs/REPAIR_ORDERING_FIX.md`.

## Remaining: one captured website run (needs allowance)

Nothing else is outstanding. Authorization was requested; the session owner was
unavailable, so **no submission was made**. Balance 1034 credits; 33 per analysis.

### Procedure when an allowance exists

1. `CORRECTION_LIVE_CAPTURE=1` via `mcp_render-mcp-se_update_environment_variables`
   (merge, leave `CORRECTION_LIVE_CAPTURE_DIR` and `CORRECTION_PROVIDER_DISCLOSURE`).
2. Poll SSH until `capture=1`; a health 200 is NOT proof a deploy landed.
3. `cd /app && PYTHONPATH=. python tools/live_capture_preflight.py` — expect exit 0.
4. Submit `synthetic-single.docx`, title "Senior Electrician", the 370-char job ad at
   `https://cogitometric.ai/app/resume-builder`.
5. Record `x-correlation-id`; expect `repair` and a re-review in the phase log, which no
   live run has ever done.
6. Inspect résumé, commentary, receipt, and the actual PDF/DOCX.
7. `CORRECTION_LIVE_CAPTURE=0`; purge container captures; keep the artifact locally.

### If it is withheld

`scp` the capture, then
`python -c "from live_capture import replay; print(replay('<f>')[0])"`.
Classify the objection against the source **before** changing code — a checker rejection is
not proof of fabrication. Then fix offline and retest with a remaining attempt.

## Verification commands

```
source .venv/bin/activate && cd Imagination_Resume_Module
python -m pytest tests/ -q -o addopts="" -p no:cacheprovider      # 913 passed, 3 skipped
python -m pytest tests/test_repair_recovery_controlled.py -q -o addopts=""
python -m pytest tests/test_run7_objection_classification.py -q -o addopts=""
```

## Evidence and integrity

Protected captures, all byte-identical to when they were recorded:

| run | revision | sha256 (12) |
|---|---|---|
| 4 | `5f3c1f0` | `92e03d4bbfde` |
| 5 | `9057d966` | `8972d3711c6b` |
| 6 | `fb75917` | `269caf5e06b3` |
| 7 | `b855087` | `bb9643ffbaea` |

A capture is evidence about a revision. Its original diagnosis is read from its own
`review`/`timing` fields (`live_capture.recorded_stopping_point`), never recomputed from
current code — recomputing would rewrite the history that made the defect findable.
