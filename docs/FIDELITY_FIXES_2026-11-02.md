# Fidelity fixes for delivery run 4 — milestone report

Correction to the commit message: the classification is **7 paraphrased**, not 8
(segments 1, 2, 6, 8, 11, 12, 20). The commit body is otherwise accurate.

Revision: `ef36ba4` on `master`.
Regression artifact: `.live-run-2026-11-02/retained/delivery-run-4-withheld.json`,
SHA-256 `92e03d4bbfde82089bcb428ff563d274a9e215cbabe496428f6565cdff891230`, unchanged.
Answer key: `tests/fixtures/fidelity/expected_facts.py` (written before any code change).

---

## 1. Defect: the writer invented a job title

**File / function.** `role_title_sanitizer.py` → `sanitize_experience_record`.

**What was wrong.** Two faults compounded. The occupation vocabulary had no entry for
`electrician`, `journeyman` or `apprentice`, so both real titles failed validation; and
the fallback branch then *wrote* the literal word `"Professional"` into every title key.
A real fact was not merely lost, it was replaced by a fabrication that every downstream
stage treated as the candidate's role.

**Before / after.**

```
IN : {"title_line": "Journeyman Electrician | 03/2020 - Present"}
BEFORE: {"title_line": "Professional", "body": "Journeyman Electrician | 03/2020 - Present"}
AFTER : {"title_line": "Journeyman Electrician | 03/2020 - Present"}
```

A rejected title now leaves the field empty; the original text is still preserved in
`description`/`snippet`/`body`, so nothing is discarded — it is simply not promoted into
a field it failed to satisfy.

**Regression.** `test_writer_never_fabricates_a_placeholder_title`,
`test_writer_recognizes_trade_titles`. Mutation: restoring the `"Professional"` write
fails the first test.

**Caught during the fix.** Widening the vocabulary made
`Example City, WA | (202) 555-0101 | alex.electrician@example.com` validate, because the
local part of the e-mail address contains an occupation word. A pre-existing
parsing-fidelity test failed immediately; role validation now rejects any title
containing `@`, a phone number, or a URL.

---

## 2. Defect: UTF-8 decoded as ISO-8859-1

**File / function.** `openrouter_safe_client.py` → `OpenRouterSafeClient.call_model`,
stream branch; new `_iter_sse_lines_utf8`.

**What was wrong.** `requests` infers its encoding from the Content-Type charset and
falls back to **ISO-8859-1** for `text/*` with no charset — and OpenRouter returns
`text/event-stream`. `iter_lines(decode_unicode=True)` therefore decoded UTF-8 bytes as
Latin-1, and the corruption entered before the writer saw anything:

```
wire bytes (UTF-8):  03/2020 \xe2\x80\x93 Present   (en dash)
correct decode    :  03/2020 – Present
inferred decode   :  03/2020 â\x80\x93 Present      <-- what reached the draft
```

**Boundary identified.** Source text is pure ASCII; the captured draft contains
`\u00e2\u0080\u0093`. The first boundary where valid punctuation became mojibake is the
SSE decode, not the writer and not the renderer.

**Fix.** Decode the stream as UTF-8, which is what SSE is defined to be. The fallback is
`errors="replace"` **only** when the bytes are genuinely not valid UTF-8, so legitimate
accented names are never damaged. This is a boundary repair, not a global substitution of
`â`.

**Regressions.** `test_provider_stream_preserves_unicode_punctuation` (en dash, em dash,
right quote, non-breaking hyphen, bullet) and `test_provider_stream_preserves_accented_names`
(`José Åberg – Müller`). Mutation: reverting to the inferred-encoding decode fails both.

**Not changed.** The non-streamed JSON path was verified unaffected (`r.json()` honours
UTF-8 when no charset is declared), so exactly one boundary was altered.

---

## 3. Defect: a defective draft was automatically unrecoverable

**File / function.** `source_correction.py` → `_correct_body`, `_validate_review`,
new `_apply_restorations`, `_pick_target_line`, `_enclosing_source_line`,
`_quote_shares_content`.

**What was wrong.** `if "missing" in incomplete: return blocked("source_facts_missing")`
fired *before* any repair. Run 4's reviewer reported 12 of 22 items missing on its first
and only call, so the run was unrecoverable at that instant — even though 7 of those 12
were present in different wording and 2 of the 3 true omissions were repairable from an
explicit source record.

**The bounded mechanism.** At most **one** restoration attempt, and it can only do this:

- speak about the named missing items, one line each, chosen deterministically;
- echo the line it replaces, character for character;
- stay on one line (no newlines);
- state only the requested source item — a replacement sharing no content word with that
  item is refused as a new fact in disguise.

Then the **whole document is re-reviewed from scratch**: coverage, attribution and every
span are re-authenticated by the unchanged validator. Restoration cannot approve
anything on its own. If the answer is unusable, or the recording has no such phase, the
attempt fails closed and the diagnosis stays `source_facts_missing` rather than degrading
into `provider_error`.

**Before / after (scripted provider, labelled as such).**

```
BEFORE: ["source_inventory", "initial_review"]     -> withheld / source_facts_missing
AFTER : ["source_inventory", "initial_review", "restore", "post_restore_review"]
                                                   -> approved, restore_calls=1
```

**Regressions.** `test_scripted_restoration_restores_a_missing_title` (integrated
positive: reaches `approved` through the real engine, no override),
`test_restoration_is_bounded_to_one_attempt` (no second restoration),
`test_restoration_cannot_smuggle_in_a_new_fact`. Mutation: disabling restoration fails
the first two.

**Safety behaviour worth recording.** If the draft contains no line sharing any content
with the missing fact *and* none with its source line, no restoration is attempted. The
engine refuses to invent an anchor for a fact it cannot place — which is the correct
outcome, not a gap.

---

## 4. Defect: a citation was accepted as proof for text it did not restate

**File / function.** `source_correction.py` → `_validate_review`, new
`_quote_shares_content` / `_content_words`.

**What was wrong.** A `supported` claim with valid offsets could be reported as
`represented` for a source item whose text the span never mentions. A line citing
`inspection checklists` could therefore carry Agile, Scrum and Gantt charts through as
"represented". Offsets and shared stop words were doing the work of semantic proof.

**Fix.** The claim's own span must share at least one content word with the item it is
said to represent. Stop words (`the`, `with`, `and`, …) never count.

**Regression.** `test_review_rejects_a_representation_the_span_does_not_restate` —
constructed so the **only** failing guard is this rule. Verified: with the rule disabled
the identical answer validates, so the test is a real control. The first version of this
test was wrong (a length mismatch made an earlier guard fire) and mutation testing
exposed it.

---

## Answer key: the 12 segments run 4 called "missing"

| Class | Count | Segments |
|---|---|---|
| paraphrased | 7 | 1, 2, 6, 8, 11, 12, 20 |
| omitted | 3 | 5, 10 (both destroyed titles), 14 (the school) |
| altered | 1 | 7 (`six apprentices` → `six apprentice crews`) |
| nonfactual | 1 | 0 (a section heading) |

The reviewer's verdict was internally defensible but operationally wrong: 7 of 12 were
present in different wording. "Not verbatim" is not "missing", and treating it as such
is what made every paraphrasing draft unrecoverable.

---

## 5. Defect: the writer was *invited* to invent

**File / function.** `imaginator_flow.py` → `run_creative_draft_async`,
`run_final_editor_async`, `run_star_editor_async`.

**What was wrong.** The gate caught run 4's inventions, but the prompts were asking for
them. Three prompts offered non-evidence as candidate fact:

- the creative drafter was given `WEB RESEARCH INSIGHTS: Implied Skills / Industry
  Metrics` and told to "incorporate research insights naturally" and to "surface these
  implied skills" for things the candidate never listed — the direct source of
  technologies appearing in a resume whose owner never used them;
- the final editor received the same block on top of two model drafts with no statement
  of their status;
- the STAR editor was given research `result_metrics` as though they were the
  candidate's.

**Fix.** Research is removed from the drafting prompt entirely and the "surface implied
skills" instruction is reversed. The drafts are labelled *editing material, not
evidence*. Each prompt states the boundary explicitly: the job description says what the
employer wants and is never evidence of what the candidate has done; with no source
metric the claim stays qualitative rather than estimated; a Projects heading may appear
empty but filling it with inventions is forbidden; no GPA, issuer or issue date the
source does not state.

**Regression.** `tests/test_writer_evidence_boundary.py` (6 tests). Deliberately
structural — it pins the boundary, not a blacklist of run-4's strings. Mutation:
restoring the research block into the drafting prompt fails it.

---

## Acceptance status, reported separately

| Item | Status |
|---|---|
| Source-fidelity regressions | **PASS** — 28 fidelity tests across 4 files; 5 fixes mutation-verified |
| Real correction-and-publication route | **PASS (offline)** — integrated positive reaches `approved` through the real engine with a labelled scripted provider, no override. Not proven against a real model. |
| Real local PDF/DOCX content checks | **PASS** — 6 tests render with the real LaTeX compiler and real DOCX writer, then read the text back. No mocked bytes. |
| Deployed revision verified | **PASS** — container matches local: `_iter_sse_lines_utf8` present, `_STANDALONE_OCCUPATIONS` present, `_quote_shares_content` present, `EVIDENCE BOUNDARY` count 2, `Implied Skills` count 0, `CORRECTION_LIVE_CAPTURE=0`, `CORRECTION_PROVIDER_DISCLOSURE=1` |
| Captured website delivery | **FAIL** — run 4 withheld; no successful delivery yet |
| Paid website run for the fixes | **NOT RUN (blocked)** — see below |

Full suite: **805 passed, 3 skipped**. The passing count is supporting evidence only;
the completion claim rests on the mutation-verified fixes above.

### Spend allowance

The authorization for **one** synthetic single-file website analysis (33 credits) was
consumed by delivery run 4: 1166 → 1133, confirmed in the DB. Balance is 1133 credits,
but the *authorization* is spent. No further paid run has been made, and none will be
without explicit approval.

---

## What this does not yet prove

1. **No real-model verification.** Every restoration positive case uses a scripted
   provider. It proves the mechanism, not that a live model answers a restore request
   usefully. Only a live run can show that.
2. **The writer still produces invented projects, GPA, technologies and issuers.** Those
   are *caught* by the gate but not *prevented* at the writer. The researcher stage's
   implied-skill and industry-metric output is still offered to the final editor as
   candidate material, which is the likely source. That is the next defect to fix.
3. **`six apprentice crews` is only caught, not corrected.** The engine withholds it; it
   does not rewrite `crews` back to `apprentices`.
4. **No delivery yet.** The milestone — a customer receiving a usable approved résumé —
   is still unmet.

## Rollback

Revert `ef36ba4` to restore the previous behaviour. The capture is untracked and
unaffected by any revert. No other work was modified.
