# Baseline manifests — two repositories

Generated 2026-11-03 after `git fetch origin`. **Every SHA below was read from the
repository, not carried forward from an earlier report.**

> **Correction to my previous session.** I previously wrote "substantial pre-existing
> branch differences" based on `git diff master`, which compared against a **stale local
> `master` ref**: 84 commits behind `origin/master` in the backend, 21 in the frontend.
> That diff described history already merged, not branch content. The manifests below use
> `origin/master`, which is the only correct baseline.

---

## Repository 1 — `Imagination_Resume_Module`

| Field | Value |
|---|---|
| **Current remote master SHA** | `7cc415ba13057de35a10d2870a7ab361a424a82b` |
| **Tested local HEAD SHA** | `0d5caeb9bb0ff7ec570be56d2d76719b83a72a75` |
| **Branch** | `wip/correction-timeout-attribution` |
| **Merge-base** | `7cc415ba13057de35a10d2870a7ab361a424a82b` |
| **Commits ahead / behind** | **4 ahead / 0 behind** (fast-forwardable) |

Note: the local `master` ref is `b23bba40`, **84 commits behind** `origin/master`. That
ref is stale and must not be used as a baseline.

### Commits ahead of remote master

```
0d5caeb review(tier-b): measured labels, portable fixture, corrected claim table
cd49052 docs(tier-b): correct stale docstring on worklist() to name all three ...
ca324e4 review(tier-b): trichotomy labelling, five SPEC.v2 controls, filename trace
d6838ed docs(spec-v2): run-10 claim table, export reconciliation, offline semantic ...
```

### Complete diff from remote master (`origin/master..HEAD`)

```
 HANDOFF.md                                         | 126 +++-
 appended_clauses.py                                | 209 +++++++
 docs/live-runs/EXPORT_FILENAME_STATUS.md           |  67 +++
 docs/live-runs/RUN_10_CLAIM_TABLE.md               | 129 +++++
 docs/live-runs/RUN_10_INTERACTION_AUDIT.md         |  79 +++
 docs/live-runs/RUN_10_TWO_FILE.md                  |  47 +-
 tests/fixtures/realistic_candidate/README.md       | 184 ++++++
 .../fixtures/realistic_candidate/REVIEW_PACKAGE.md | 217 +++++++++++
 .../realistic_candidate/candidate-main.txt         |  29 +
 .../realistic_candidate/candidate-supplement.txt   |  21 +
 .../intended-job-description.txt                   |  20 +
 tests/fixtures/tier_b/extract_from_capture.py      | 130 +++++
 tests/fixtures/tier_b/run10_excerpt.json           |  73 +++
 tests/test_appended_clause_worklist.py             | 637 +++++++++++++++++++++
 tests/test_export_filename_binding.py              | 135 +++++
 tests/test_realistic_candidate_fixture.py          | 291 ++++++++++
 tests/test_route_repair_publish.py                 |  10 +-
 tests/test_source_preservation.py                  |   2 +
 18 files changed, 2403 insertions(+), 3 deletions(-)
```

**Production source: none.** Verified explicitly:

```bash
git diff --stat origin/master..HEAD -- app.py source_correction.py \
  source_preservation.py stages/ imaginator_flow.py correction_protocol.py \
  publication_gate.py resume_builder/
# (no output)
```

The whole branch is documentation, fixtures, and tests. `test_route_repair_publish.py`
and `test_source_preservation.py` gain skip guards only (+10 / +2 lines).

### Session-only diff (this session's 3 commits: `d6838ed~1..HEAD`)

Identical to the branch diff above, because `d6838ed` is the branch's first commit. The
two ranges are listed separately only because they coincide here — they would not on a
longer branch, and conflating them is how a session's scope gets overstated.

### Merge assessment

Fast-forwardable, +0 behind, no production source. Safe to merge as *all four* commits or
as none; there is no intermediate state that alters behaviour.

---

## Repository 2 — `Login_Landing_page`

| Field | Value |
|---|---|
| **Current remote master SHA** | `9e040e3043beb2163a2214bfcef0abdc099a6e99` |
| **Tested local HEAD SHA** | `622cded3ab83a8f6de84b2b0adf50a91a09b118f` |
| **Branch** | `feat/hybrid-resume-ingest-phase1` |
| **Merge-base** | `9e040e3043beb2163a2214bfcef0abdc099a6e99` |
| **Commits ahead / behind** | **1 ahead / 0 behind** (fast-forwardable) |

Note: the local `master` ref is **21 commits behind** `origin/master`. Stale; not a valid
baseline.

### Complete diff from remote master (`origin/master..HEAD`)

```
 .../__tests__/exportFilenameTitleBinding.test.ts   | 145 +++++++++++++++++
 .../__tests__/exportRequestIdentity.test.ts        | 176 +++++++++++++++++++++
 2 files changed, 321 insertions(+)
```

Both files are under `__tests__/`. **No production source, no `src/` behaviour change, no
`backend/` change.** Verified: filtering `src/` and `backend/` for non-`__tests__` paths
returns nothing.

### Session-only diff

Identical to the branch diff — one commit, made this session.

### Merge assessment

Fast-forwardable, +0 behind, tests only. Safe to merge or discard with no runtime effect.

---

## What these manifests do and do not establish

**Established:** neither repository's branch carries unmerged production code against the
current remote master; both are fast-forwardable; the only non-test files added are
documentation, fixtures, and one standalone module that nothing imports.

**Not established by the manifests alone:** that the *deployed* revision equals remote
master. The backend is deployed at `7cc415b` (verified earlier from the Render deploy SHA
and by reading the fix out of the deployed source), which equals `origin/master` here. The
frontend static site's deployed SHA was not re-verified in this pass.

**A note on `appended_clauses.py`.** It is a new top-level module with **no importer**
anywhere in the pipeline. `git grep` for `appended_clauses` outside its own file and its
tests returns nothing, so adding it cannot change runtime behaviour. That is what makes
merging the branch a no-op for production, not merely an assertion that I did not wire it.
