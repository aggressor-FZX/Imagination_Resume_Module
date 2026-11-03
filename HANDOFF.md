# Handoff — repair path proven offline; live delivery still needs an allowance

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
