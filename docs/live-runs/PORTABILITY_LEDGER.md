# Portability ledger — generated from test-result data

Every number below is computed from JUnit XML produced by two full suite runs, not
counted by hand. Regenerate with the commands in §5.

| Run | Total | Passed | Skipped | Failed |
|---|---|---|---|---|
| Clean checkout (`.live-run-2026-11-02/` absent) | 1066 | **986** | **80** | **0** |
| With historical captures present | 1066 | **1063** | **3** | **0** |

**Same 1066 tests both ways.** The 77-test difference is exactly the set that skips
only when captures are absent:

```
80 (clean) − 3 (history)      = 77 capture-replay skips
77 capture-replay + 3 live-API = 80 total clean skips   ✓ reconciles
986 (clean passed) + 77        = 1063 (history passed)  ✓ reconciles
```

## 1. Skip categories (clean checkout, from XML)

| Category | Skips | Triggers only when history is absent? |
|---|---|---|
| Capture replay (run 4/5/6/7/8/10 absent) | **77** | **Yes** |
| Live-API opt-in (`--run-live`) | **3** | No — always skips |
| **Total** | **80** | |

The 3 live-API skips are in `test_phase2_onet` (2) and `test_logic_mocks` (1). They
appear identically in both runs and are **not** history-related.

## 2. The 77 history-dependent skips, by file (from XML)

| File | Skips | Reason strings seen |
|---|---|---|
| `test_repair_ordering` | **21** | `run4/5/6/7/8 capture is not present` (5 each) |
| `test_source_preservation` | **15** | `run-7 capture absent` |
| `test_route_source_preservation` | **13** | `run-7 capture is not present` |
| `test_repair_recovery_controlled` | **9** | `run-7 capture absent` |
| `test_run7_objection_classification` | **7** | `run-7 capture absent` |
| `test_route_repair_publish` | **6** | `run-7 capture absent` (1), `run-7 capture absent; archival replay only` (5) |
| `test_fidelity_delivery_run_4` | **4** | `protected live capture is not present on this machine` |
| `test_appended_clause_worklist` | **2** | `full run-10 capture not present; archival replay only` |
| **Sum** | **77** | |

**Correction:** an earlier hand count gave `test_appended_clause_worklist` as 1 and the
total as 77 only by coincidence. The XML says 2, and the reason strings differ by test
collection order. This is precisely why the ledger is now generated instead of tallied.

## 3. Portable counterparts

Every engine behaviour the archival suites replay is covered by a test that needs no
historical file. Counts are from the clean-checkout run:

| Behaviour | Portable coverage | Ran clean |
|---|---|---|
| Correction engine, claim level | `test_source_correction.py` | 81 |
| Preservation + publish + export, two documents, real `/analyze` | `test_route_two_documents.py` | 22 |
| Restoration | `test_route_positive_restore.py` | 5 |
| Semantic route cases | `test_route_semantic_cases.py` | 9 |
| Export filename / title binding | `test_export_filename_binding.py` | 10 |
| Tier-B work list, from committed fixture | `test_appended_clause_worklist.py` | 32 |
| Realistic fixture structure | `test_realistic_candidate_fixture.py` | 23 |
| Preservation, in-file portable subset | `test_source_preservation.py` | 15 (of 30) |

Combined preservation/repair/export group on a clean checkout: **58 passed, 30 skipped**.

## 4. The one gap, named rather than faked

`test_repair_ordering.py`'s five-run comparison (21 skips) asserts that repair *ordering*
is consistent across five different recorded drafts. That is a cross-run property: no
single-draft test can express it, and synthesising five drafts to satisfy it would test
the synthesis. Left archival.

No test was added to raise any count.

## 5. Regenerate

```bash
cd Imagination_Resume_Module

# with history
python -m pytest tests/ -q -o addopts="" -p no:cacheprovider --junitxml=/tmp/hist.xml

# clean checkout
mv ../.live-run-2026-11-02 /tmp/lb
python -m pytest tests/ -q -o addopts="" -p no:cacheprovider --junitxml=/tmp/clean.xml
mv /tmp/lb ../.live-run-2026-11-02
```

Classification: `<skipped message="…">` in the XML is a skip, `<failure>`/`<error>` a
failure. A skip present in `clean` but absent in `hist` is history-dependent by
construction — no manual tagging involved.
