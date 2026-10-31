# Synthetic parsing-fidelity fixtures

Small source copy of the repository-owned synthetic profiles and independent
answer keys in `resume_bench/rctb/fixtures.py` (2026-10-31). App tests read only
the local files in this directory; that source is provenance, not a runtime
dependency. No private external resumes or answer keys were read or copied.

The legacy Holloway/Whitfield IDs are labels, not real people. All contact
details are synthetic (example.com, reserved 555 numbers, Example City/Town).
Each profile has two employers with independently specified date ranges, an
open-ended Present role, a closed historical role and separate education.

Answer keys are literal `ANSWER_KEY = {...}` assignments, never derived from
parser output. Missing files or invalid assignments must fail the tests rather
than skip or fall back to external data. R-7 employer preservation and phantom
block checks remain in place; R-8 also compares exact company/date pairings.
