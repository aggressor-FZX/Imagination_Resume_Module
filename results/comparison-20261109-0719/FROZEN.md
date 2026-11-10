# FROZEN — comparison-20261109-0719

**Do not edit any file in this directory.** These are the 24 retained attempts from
the first four-way comparison. They are kept exactly as produced because the
corrections that followed are only auditable if the original bytes survive.

    sha256sum -c SHA256SUMS

## Why frozen

The offline replay (`experiment/replay.py`, commit `a10cb76`) showed this run is
**invalid as a judgment-quality comparison**, for two independent reasons:

1. **Every case was unsatisfiable as designed.** All six cases supply both source
   documents, but every draft omits the `Ridgeline Equipment Services` record that
   `candidate-supplement.txt` contains. A faithful reviewer must flag that; the
   three `approve` expectations could not be met correctly.
2. **The `current` arm was cut off.** `max_tokens=1200` severed 11 of 12 legacy
   responses mid-string, so the legacy contract was never actually evaluated.

Neither defect is a property of the approaches under test. Corrected fixtures and
classifications live in separate, versioned modules; the originals are untouched.

## What may never be done with this directory

- Do not rewrite `attempts.json` to "fix" the classifications. Corrected readings
  belong in `experiment/replay_v2.py`, which reads these bytes and writes elsewhere.
- Do not delete a failing attempt. No retry-until-lucky.
- Do not cite these numbers as evidence for or against either contract.
