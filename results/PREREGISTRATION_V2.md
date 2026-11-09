# Preregistration — corrected four-way comparison (v2)

Committed **before** any paid call. Every number in the results must be traceable to
a decision recorded here, not to a choice made after seeing an answer.

Fixture fingerprint (`cases_v2.cases_fingerprint()`): **1aa800a3b14f22fd…**
Model: `qwen/qwen3-coder` · Temperature: 0 · Cases: 7 · Cells: 4 · Attempts: 28

---

## 1. What is being compared

| Cell | Instructions | Semantic rules |
|---|---|---|
| `current+advisory` | legacy span envelope (JSON) | explain only, cannot block |
| `current+enabled` | legacy span envelope (JSON) | can withhold |
| `revised+advisory` | decision contract (prose) | explain only, cannot block |
| `revised+enabled` | decision contract (prose) | can withhold |

Four controls stay ACTIVE in every cell: authorization, receipt integrity, reference
validity, execution limits. A comparison that switched them off would be measuring a
different system.

## 2. What was wrong with v1 — and why v2 is not a re-run

Two independent defects made the v1 table uninterpretable. Neither is a property of
the approaches under test.

1. **The fixture was invalid.** All six cases supplied both source documents; every
   draft contained only the main one, dropping `Ridgeline Equipment Services`. A
   faithful reviewer had to flag that, so the three `approve` expectations were
   unsatisfiable. Measured independently: `experiment/case_audit.py` rejects the v1
   fixture with 3 `covers_sources` failures.
2. **The legacy arm was truncated.** `max_tokens=1200` severed 11 of 12 responses
   mid-string at ~3,700–4,900 characters. Re-measured offline: **zero** legacy
   attempts produced a usable judgment.

Corrected: 7 cases on a complete two-document baseline, `max_tokens=8192`, and
truncation recorded as `operational` rather than scored as a refusal.

## 3. Fixture: seven cases, one mutation each

Baseline = the complete two-document source, faithfully written (all three employers,
all seven bullets, the certification).

| Case | Expectation | The single change from baseline |
|---|---|---|
| `faithful_paraphrase` | approve | none — the baseline itself |
| `supported_method` | approve | `by` → `through`; method is the sourced one |
| `unsupported_method` | revise | appends `and targeted quality audits` (in no source) |
| `missing_fact` | revise | deletes the 1,180-invoice bullet |
| `wrong_record_attribution` | revise | moves the 1,180 figure to Cascade Freight |
| `transferable_skill` | approve *(Imaginative only)* | adds a capability clause |
| `omitted_record` | revise | deletes the whole Ridgeline record |

Isolation is enforced, not asserted: a case differing from the baseline by more than
6 lines fails the audit, because a verdict could then not be attributed to its defect.

**Mode.** All cells run `imaginative`. `transferable_skill` is therefore approvable.
A Conservative reviewer refusing it would be **correct**, and is excluded from
false-refusal scoring by design — the expectation is mode-conditional and this is
recorded rather than hidden.

## 4. Preregistered predictions

Stated so a wrong prediction is visible rather than rationalised.

| # | Prediction | Falsified if |
|---|---|---|
| P1 | `revised` cells complete on ≥6/7 cases; `current` cells on ≥5/7 at 8,192 tokens | truncation persists at a sufficient budget |
| P2 | `revised+enabled` withholds `unsupported_method` and `wrong_record_attribution` | it approves either |
| P3 | `current+enabled` withholds ≥1 of the same two defects | it approves both |
| P4 | `revised` produces ≥1 actionable correction the harness can apply unambiguously | no case yields an applicable correction |
| P5 | `omitted_record` yields an `OmissionReport` naming source **and** destination, applied nowhere | the record is auto-inserted, or the omission is dropped |
| P6 | No cell fabricates a citation on this fixture | any `reference:fabricated` appears |
| P7 | Mode distinction holds: Conservative refuses `transferable_skill`, Imaginative approves | both agree (checked in the contract probe, §6) |

**No prediction is made about which approach is "better".** The measurable claims are
above; a winner would need the blind audit.

## 5. Metrics — kept on separate axes

Only the `semantic:` axis is a judgement about the document.

- `semantic:approve` / `semantic:revise` / `semantic:cannot_assess`
- `operational:truncation` / `transport` / `empty` — **never** scored as a refusal
- `format:mismatch_requested` — answered in a different shape than asked
- `parse:defect` / `parse:contract_defect` — code could not read, or a rule rejected
- `reference:convention` — quote real, offset differs (**not** a refusal, **not**
  fabrication)
- `reference:fabricated` / `reference:unknown_source` — evidence does not exist

Reporting rules, fixed now:

- An incomplete attempt is `scored=false`; it contributes to no numerator or
  denominator of approval quality.
- **Parser acceptance is never reported as semantic correctness.** A well-formed
  approval of a defective document is a false approval.
- Correct approvals, false approvals and false refusals are reported separately, per
  cell. There is no single "score".

## 6. Sequence

1. **Contract probe (non-comparative), ~4 calls.** One known-good approval and one
   unsupported-method revision, in both adapters, plus a Conservative/Imaginative
   pair on `transferable_skill` (P7). Purpose: prove each adapter can reach a verdict
   before spending on the matrix. A probe failure stops the run and is reported as a
   defect, not as a result.
2. **The 28-attempt matrix**, then stop.
3. **Blind packet** (`experiment/blind_packet.py`, seeded), audited before any cell is
   named as better.

`revision` rounds re-ask once — so the ceiling is 28 attempts *plus* at most one
re-check for `revised` cells that proposed an applicable correction.

## 7. Budget

The ceiling is enforced **before** each call by `RunBudget`, not reported afterwards:

| Bound | Value |
|---|---|
| Max provider calls | **70** |
| Max spend | **USD 2.00** |
| Max attempts | 28 + ≤14 re-checks |
| Wall clock per call | 240 s |

70 calls = 28 matrix + up to 14 re-checks + 4 probe + retry headroom. A call that
would exceed either bound is refused and recorded as
`operational_failure:budget_refused`, which is **not** a semantic result.

Observed v1 spend was ≈0 with typical latency 6–16 s per call, so USD 2.00 is a
comfortable ceiling rather than a target.

## 8. Stopping rules

The run stops and reports a **harness problem**, not a result, if:

- the contract probe cannot reach a verdict in either adapter;
- `reference:fabricated` appears on this fixture (P6) — the fixture would be suspect;
- >2 attempts truncate at 8,192 tokens — budget still insufficient, run invalid for
  that arm;
- either budget bound is hit;
- any attempt records an `exception` outcome.

A stopped run is reported with the counts that caused the stop. It is never trimmed
to the attempts that happened to succeed.

## 9. What this experiment cannot show

- **Nothing about production behaviour.** Both adapters are experimental; production
  approval is unchanged on this branch.
- **Nothing about a production failure rate.** 7 cases x 4 cells is enough to compare
  configurations, not to estimate how often an approach fails in service.
- **Nothing conclusive about semantics.** Expectations are the author's judgement.
  §3's mechanical part is audited by `case_audit.py`; the reading part is
  `undetermined` (5 findings) and requires the blind audit. Author expectations stay
  **provisional** until then.
- **Nothing about other models.** One model at temperature 0.

## 10. Committed before the run

- `experiment/cases_v2.py` — the fixture (fingerprint above)
- `experiment/case_audit.py` — the independent fixture auditor
- `experiment/replay_v2.py` — the corrected taxonomy
- `experiment/provider_v2.py` — budget and telemetry
- `experiment/runner_v2.py` — the matrix, omission handling, budget enforcement
- `tests/test_cases_v2.py` — 42 tests pinning every claim above
