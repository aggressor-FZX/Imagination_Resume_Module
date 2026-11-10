# Live-contract probe — Gemma CANNOT replace the reviewer as-is

**Outcome: the merge would have broken every résumé. It was not merged.**

You approved the deploy. Before pushing, I checked the one thing that would decide
whether it was safe: can the selected reviewer actually speak the contract the LIVE
service uses? It cannot. This document records that, because the discovery is more
valuable than the deploy would have been.

## The two contracts are different protocols

| | Live service (`app.py` → `source_correction`) | Where Gemma was selected |
|---|---|---|
| Format | JSON span envelope: `claims` + `representations` | Prose: `Decision:` + `Issues:` |
| Offsets | exact Unicode code-point offsets per line | none |
| Citations | `{source_id, start, end, quote}`, quote must equal the source slice | source ids, free text |
| Validator | `_validate_review` — `_exact_dict`, offset bounds, quote equality | `decision_contract.parse_decision` |

`app.py` contains **zero** references to `decision_contract`. The prose contract and
its repair path are not wired into production at all.

## Probe result — 4 tokens issued, 4 billed calls, $0.004868

| Phase | Result |
|---|---|
| `source_inventory` | **PASSES** after a fence fix (below) |
| `initial_review` | **FAILS** — `_Invalid: invalid envelope: unspecified` |

### Why the review fails

Gemma produced a well-formed envelope: 22 claims, 19 representations, correct top-level
keys, no extra keys, all verdicts valid. The failure is in the **values**:

```
claims with problems: 21 / 22
  offsets past the end of the line (0:24 on a 23-char line, 0:82 on 81)
  citation quote does not equal the source slice
```

The live contract demands exact code-point bookkeeping — the model must partition every
line into spans whose offsets and quotes reconcile to the character. **Gemma does not do
this reliably, and the live engine rejects the whole answer when it does not.** Every
correction would have withheld.

This is not a prompt problem. It is the structural exam the experiment deliberately
replaced, and Gemma was never selected for it — it was selected on the prose contract.

## The one change that was kept

`correction_provider._strict_json` now tolerates a Markdown code fence:

```diff
+_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
```

Gemma wraps its JSON in ```json fences, which previously failed at character 0. This is
a real robustness fix that helps **any** model, so it was kept. Strictness is
unchanged and tested: duplicate keys, non-finite numbers and surrounding prose all
still fail.

Verified: the full suite passes (2153 → 2169 with the new fence tests), and the current
production model's behaviour is untouched because a bare object parses exactly as before.

## What was reverted

The `OR_SLUG_REVIEWER` config change was reverted. Pointing the imaginator at Gemma
would have converted a working service into one that withholds every résumé.

## What this means for the goal

The repair and release path is proven — it released five documents including three
corrections, with receipts bound and controls untouched. But it lives on the **prose
contract**, and the live service uses the **span contract**. Two options remain, and
only one is honest:

1. **Port the prose contract into the live service.** Replace `source_correction`'s
   review phases with `decision_contract` + `experiment/repair` +
   `experiment/release_trace`. This is a large change: 37 references to
   coverage/claim-span machinery inside `source_correction`, plus the restoration and
   repair paths that depend on span offsets. It needs its own regressions and its own
   authorization.
2. **Find a reviewer that speaks the span contract.** Gemma does not. `qwen/qwen3-coder`,
   the incumbent, evidently does well enough that the service works today — but it is
   the model whose objections the experiment found unreliable.

Option 1 is the honest path to the goal you set: faithfulness judged by a model that is
good at judgement, with code doing the bookkeeping. Option 2 keeps the bookkeeping and
keeps the reviewer problem.

## Not done

- **No merge. No push to master. No deploy.** The imaginator still runs its current
  reviewer on its current contract.
- **No PDF or DOCX was produced through the website.** The path that would produce them
  is not the path that was validated.
- **Four probe tokens were issued for what is one question.** Two
  (`-01`, `-02`) failed on my own import bug before any call succeeded and were billed
  0 requests each; `-03` and `-04` each sent the two phases. Total probe spend
  **$0.004868**, all retained in `results/.authorizations.json`. Issuing a fresh token
  after each bug was wasteful and should have been one re-run under a single token —
  recorded rather than hidden.
