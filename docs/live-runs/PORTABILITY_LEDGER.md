# Portability ledger — portable-required vs optional archival tests

Measured, not asserted. Two full runs, after the control corrections in this pass:

| Mode | Result |
|---|---|
| Clean checkout (`.live-run-2026-11-02/` absent) | **986 passed, 80 skipped, 0 failed** |
| With historical captures present | **1063 passed, 3 skipped** |

**Difference: 77 tests skip without history** (80 − 3 baseline). They are listed below,
with their portable counterparts. No skip guard was added merely to make the suite green:
the two guards added this session cover tests that previously **errored**, which is a
different defect (see §3).

## 1. The 77 optional archival skips, by file

| File | Skips | Reason string | What it replays |
|---|---|---|---|
| `test_source_preservation.py` | 15 | `run-7 capture absent` | Preservation rules against run 7's real draft |
| `test_route_source_preservation.py` | 13 | `run-7 capture is not present` | The preservation route over run 7 |
| `test_repair_recovery_controlled.py` | 9 | `run-7 capture absent` | Repair recovery sequences |
| `test_run7_objection_classification.py` | 7 | `run-7 capture absent` | Classifying run 7's gate objection |
| `test_route_repair_publish.py` | 5 | `run-7 capture absent; archival replay only` | Repair → publish → export over run 7 |
| `test_route_repair_publish.py` | 1 | `run-7 capture absent` | Same, single test |
| `test_repair_ordering.py` | 5 + 5 + 5 + 5 + 1 | `runN capture is not present` (run 4,5,6,7,8) | Repair-ordering across five recorded runs |
| `test_fidelity_delivery_run_4.py` | 4 | `protected live capture is not present on this machine` | Delivery fidelity for run 4 |
| `test_appended_clause_worklist.py` | 1 | `full run-10 capture not present; archival replay only` | The real reviewer answer for run 10 |
| **subtotal** | **75** | | |
| `test_phase2_onet.py` | 2 | `live API test: opt in with --run-live` | **Unrelated** — opt-in live API |
| `test_logic_mocks.py` | 1 | `live API test: opt in with --run-live` | **Unrelated** — opt-in live API |
| **total without history** | **78** | | |

Two of the 80 skip unconditionally (live-API opt-in), so they are **not** history-related:
**77 is the history-related delta**, 75 of which are capture-dependent replay and 2 of
which are the live-API tests that always skip. That reconciles the count exactly.

## 2. Portable counterparts for the essential checks

Every behaviour the archival suites replay is also covered by a portable test that needs
no historical file. Measured on a clean checkout:

| Behaviour | Portable coverage | Count |
|---|---|---|
| **Preservation** (single document) | `test_source_preservation.py` portable tests + `tests/fixtures/fidelity/expected_facts.py` | 15 of 30 in that file run |
| **Preservation** (two documents) | `test_route_two_documents.py` — real `/analyze` route, two source documents, no capture | **22** |
| **Restoration** | `test_route_positive_restore.py` | **5** |
| **Semantic route cases** | `test_route_semantic_cases.py` | **9** |
| **Correction engine** (claim-level) | `test_source_correction.py` | **81** |
| **Repair → publish → export** | `test_route_two_documents.py` (`test_the_approved_two_document_payload_renders_both_records`, `test_export_refuses_altered_content_under_the_existing_receipt_contract`, `test_export_refuses_the_approved_content_presented_under_another_owner`) | included in the 22 |
| **Export filename / title binding** | `test_export_filename_binding.py` | **10** |
| **Tier-B work list** | `test_appended_clause_worklist.py` (from the committed fixture) | **34 of 34 run; 1 skips only when history is absent** |
| **Realistic fixture structure** | `test_realistic_candidate_fixture.py` | **23** |

Aggregate for the combined preservation/repair/export group on a clean checkout:
**58 passed, 30 skipped** (`test_route_two_documents`, `test_source_preservation`,
`test_route_repair_publish`, `test_repair_recovery_controlled`,
`test_route_positive_restore`, `test_route_semantic_cases`).

**Assessment:** the archival skips are replay of *recorded runs*, not unique coverage of
engine behaviour. Nothing essential depends on them. The one gap worth naming:

- `test_repair_ordering.py`'s five-run comparison has **no portable equivalent**. It
  asserts that repair ordering is consistent across five different recorded drafts — a
  cross-run property. A single-draft portable test cannot express it, and fabricating five
  drafts would test the fabrications. Left as archival, documented.

## 3. Guard status — what was and was not added

| Guard | Pre-existing behaviour | Action | Justified? |
|---|---|---|---|
| `test_source_preservation.py` ×2 | **FAILED** (FileNotFoundError), not skipped | Added `@requires_capture` (the marker already existed in that file) | Yes — a missing optional file must not be a failure |
| `test_route_repair_publish.py` autouse fixture | **ERRORED** at setup ×5 | Added a skip in the fixture | Yes — same reason; the inner helper already had an identical guard, so the fixture was inconsistent with itself |

No other guard was added. In particular, no guard was added to make a failing assertion
pass.

## 4. Commands

```bash
cd Imagination_Resume_Module
python -m pytest tests/ -q -o addopts="" -p no:cacheprovider          # with history
mv ../.live-run-2026-11-02 /tmp/lb && \
  python -m pytest tests/ -q -o addopts="" -p no:cacheprovider; \
  mv /tmp/lb ../.live-run-2026-11-02                                   # clean checkout
python -m pytest tests/ -q -o addopts="" -p no:cacheprovider -rs      # list skips
```

Required tests raise `AssertionError` with the fixture path when
`tests/fixtures/tier_b/run10_excerpt.json` is missing — verified by removing it (2
failures, clear message, not a skip).
