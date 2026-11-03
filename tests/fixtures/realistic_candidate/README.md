# Realistic candidate fixture — awaiting review, NOT frozen, NOT submitted

Status: **candidate pack. Not hashed, not frozen, not approved for any paid run.**

The existing synthetic fixtures (`tests/fixtures/fidelity/`, `tests/fixtures/
correction_acceptance/`, and `.live-run-2026-11-02/synthetic-*.docx`) are **unchanged**.
This pack is additive and lives beside them. Nothing here overwrites a regression
fixture, a capture, or an answer key.

## Why a second pack

The run-10 sources used `Example City` / `Sampletown` and institution names built from
the word "Example". That is deliberate for capture safety, but it has a side effect that
matters for a preservation test: the locations look like placeholders, so it is not
possible to demonstrate whether the writer **preserves** a real-looking location instead
of inventing a plausible one. A candidate whose cities are real but whose employers and
institutions are fictional exercises preservation without teaching the writer to
substitute invented detail.

## The candidate (one person, both documents)

**Name:** Dana R. Whitfield — fictional.
**Location:** Spokane, WA.
**Test-only contact:** `dana.whitfield@example.invalid`, `(509) 555-0147`.
`.invalid` is reserved by RFC 2606 and can never resolve, so this cannot reach a real
mailbox.

**Chronology, oldest to newest, consistent across both documents:**

| Period | Employer | Role | Source document |
|---|---|---|---|
| 03/2012 – 09/2015 | Northline Fabrication, Inc. (Spokane, WA) | Welder / Maintenance Helper | `candidate-main.docx` |
| 10/2015 – 06/2019 | Kettle Falls Electrical Contractors (Spokane, WA) | Electrical Apprentice | `candidate-main.docx` |
| 07/2019 – 02/2023 | Pend Oreille Power Systems (Spokane Valley, WA) | Journeyman Electrician | `candidate-main.docx` |
| 03/2023 – Present | Riverbend Facilities Group (Coeur d'Alene, ID) | Maintenance Electrician | `candidate-supplement.docx` |

Education:
- Spokane Falls Institute of Technology (fictional) — AAS, Electrical Systems
  Technology, 2015 — `candidate-main.docx`
- Inland Northwest Safety Academy (fictional) — Certificate, NFPA 70E Arc Flash
  Standards, 2021 — `candidate-supplement.docx`

**Why the supplement carries the current employer and the certificate:** it is the shape
that exposes a single-document reader. A reader that takes only the first document will
lose the current job and the certificate entirely — the same defect run 10 fixed.

## Fact-ownership ledger

Every fact below has exactly one owner. Employer ownership of every metric is explicit,
because metric transplant is the failure this fixture must be able to detect.

| Fact | Value | Owner | Document |
|---|---|---|---|
| F01 | "Journeyman electrician with 12 years of electrical experience" | candidate (summary) | main |
| F02 | "spans commercial construction, industrial maintenance, and facility operations" | candidate (summary) | main |
| F03 | Northline Fabrication, Inc., Spokane, WA | employer | main |
| F04 | Welder / Maintenance Helper, 03/2012 – 09/2015 | employment record | main |
| F05 | "Fabricated and repaired structural steel assemblies for 14 industrial contracts" | Northline | main |
| F06 | Kettle Falls Electrical Contractors, Spokane, WA | employer | main |
| F07 | Electrical Apprentice, 10/2015 – 06/2019 | employment record | main |
| F08 | "Pulled and terminated branch circuits on 31 commercial buildouts" | Kettle Falls | main |
| F09 | "Completed 8,000 apprenticeship hours under a licensed journeyman" | Kettle Falls | main |
| F10 | Pend Oreille Power Systems, Spokane Valley, WA | employer | main |
| F11 | Journeyman Electrician, 07/2019 – 02/2023 | employment record | main |
| F12 | "Diagnosed and repaired switchgear faults at 9 substations" | Pend Oreille | main |
| F13 | "Cut average fault-response time from 4.5 hours to 2.75 hours" | Pend Oreille | main |
| F14 | "Trained 4 apprentices on lockout/tagout procedures" | Pend Oreille | main |
| F15 | Spokane Falls Institute of Technology, AAS Electrical Systems Technology, 2015 | education | main |
| F16 | Riverbend Facilities Group, Coeur d'Alene, ID | employer | supplement |
| F17 | Maintenance Electrician, 03/2023 – Present | employment record | supplement |
| F18 | "Maintain electrical distribution and BAS controls for 6 commercial properties" | Riverbend | supplement |
| F19 | "Reduced after-hours callouts by 27% by adding thermographic inspections" | Riverbend | supplement |
| F20 | Inland Northwest Safety Academy, Certificate NFPA 70E Arc Flash Standards, 2021 | education | supplement |
| F21 | "Washington 01 Journey Level Electrician License" | certification | supplement |
| F22 | Skills: conduit bending, switchgear troubleshooting, thermographic inspection, BAS controls, lockout/tagout, NFPA 70E | skills | both |

### Deliberate traps built into the ledger

These exist so a passing run means something. Each is a fact a draft could plausibly
mis-state:

1. **Two employers in one state, two in another.** Spokane/Spokane Valley are WA; Coeur
   d'Alene is ID. Moving the current role to WA is an attribution error a location-based
   guess would make.
2. **Two "electrical" education records, different institutions and levels.** AAS 2015 vs
   Certificate 2021. Swapping year or institution must fail.
3. **Three distinct metrics, three distinct owners.** 14 contracts (Northline), 31
   buildouts (Kettle Falls), 9 substations / 4.5→2.75 hours / 4 apprentices (Pend
   Oreille), 6 properties / 27% (Riverbend). No metric may move.
4. **The current employer is in the SECOND document.** A single-document reader drops it.
5. **8,000 hours appears once, for Kettle Falls only.** It must not attach to Pend
   Oreille.
6. **"4 apprentices" (Pend Oreille) is not "6 apprentices".** A near-miss number.

## Capture-safety compatibility — checked, not assumed

`live_capture.assert_synthetic()` refuses to write a capture unless **every** source
document contains the literal marker `SYNTHETIC-ONLY`. This pack therefore keeps the
marker on every factual line, exactly as the run-10 sources do, and this fixture is
**not** an exception to that rule.

The marker is compatible with realistic cities because it is a per-line prefix that the
writer is expected to drop from output — `source_preservation._PROVENANCE_MARKERS`
already handles that. Realism and the safeguard are not in tension here, so the safeguard
does not need to be weakened to accommodate this fixture, and it has not been.

Constraint accepted as a consequence: the rendered output for this fixture must **not**
contain `SYNTHETIC-ONLY`, and any test that runs it must assert that.

## Intended job description

`intended-job-description.txt` — a "Maintenance Electrician – Commercial Properties"
posting for a fictional Riverbend-area facility services employer. It is the target the
rewrite would be judged against, and it deliberately asks for things the candidate
partially has:

| Posting requirement | Candidate's actual position |
|---|---|
| WA 01 Journey Level License | Has it (supplement) |
| 8+ years electrical experience | 12 years overall; 8,000 apprenticeship hours + 3.5 years journeyman |
| Switchgear, motor controls, BAS | Switchgear yes; motor controls **not** in the record; BAS yes (supplement) |
| NFPA 70E arc flash training | Certificate 2021 (supplement) |
| Multi-site portfolio | 6 commercial properties (supplement) |

**Motor controls is the deliberate gap.** It appears in the posting and nowhere in the
candidate's record.

## Corrected expectation for the job-ad gap

An earlier version of this document said importing motor controls would be "withheld".
That was the wrong remedy and is corrected here.

| Situation | Correct behaviour |
|---|---|
| Candidate genuinely lacks motor controls | The skill stays **absent from the résumé**, and **may appear in gap analysis** as a missing requirement. This is the normal, correct outcome — the résumé is truthful and the gap is reported to the user. |
| The writer invents motor controls in the résumé | The contaminated clause must be **removed or repaired**, or the result **withheld** if repair fails. The defect is the invention, not the absence. |

**A truthful résumé is never withheld merely because the job ad asks for a skill the
candidate does not have.** Withholding in that case would punish correct output: every
candidate lacks something the posting asks for, so that rule would reject nearly every
résumé. What must be detected is the *invention*, which is the run-4 defect class
("Agile, Scrum, Gantt" appearing against an item that only said "inspection
checklists").

So the B-case is:

| Case | Expectation |
|---|---|
| G1-a | Draft omits motor controls entirely → **approves**; gap analysis may list it. |
| G1-b | Draft claims motor controls experience → **repaired or withheld** (invented qualification). |

Test G1-a is the one an earlier draft would have got wrong.

## The six named traps and their expected outcomes

| # | Trap | Expected outcome |
|---|---|---|
| T1 | Two states: Spokane/Spokane Valley are WA, Coeur d'Alene is ID | A draft that moves the current role to WA is **withheld** (attribution error). Test B-4. |
| T2 | Two electrical education records: AAS 2015 vs Certificate 2021, different institutions | Swapping institution or year is **withheld**. Test B-5. |
| T3 | Three metrics, three owners: 14 contracts (Northline), 31 buildouts (Kettle Falls), 9 substations / 4.5→2.75 h / 4 apprentices (Pend Oreille), 6 properties / 27% (Riverbend) | Any metric moved to another employer is **withheld**. Test B-3. |
| T4 | Current employer, certificate and licence exist **only** in the supplement | A single-document reader drops them; the two-document route must retain them. Test B-2. |
| T5 | `8,000 hours` belongs to Kettle Falls only | Attaching it to Pend Oreille is **withheld**. |
| T6 | `4 apprentices` (Pend Oreille) is not `6` — run 10's number | A near-miss figure is **withheld**. Test B-6. |
| G1 | Job ad asks for motor controls; record never mentions it | **Omitting it approves** (and gap analysis may flag it). **Inventing it is repaired or withheld.** See the corrected expectation above. |

## What is required before this pack can be used

1. **Review of the two source documents and the ledger above** by the owner.
2. Freeze and hash **after** approval; record the hashes in this file.
3. Confirm the marker requirement still holds for both `.docx` files as rendered.
4. Only then may a separately authorised paid run reference it.

No paid submission is requested here, and none is authorised by this document.

## Current hashes — informational, NOT a freeze

Recorded so a later reviewer can detect accidental edits. These are **not** an approval
and must not be treated as one; they will be re-taken after review.

| File | SHA-256 |
|---|---|
| `candidate-main.txt` | `d5b7a9d408e43dfeb48f2c35837878b2d23b1800de1182bf6cb9aa2c9b4e4476` |
| `candidate-supplement.txt` | `3c6f9fcfc6681ef70bf711c63f46d9fbded07004cb28fef9159221af08f031aa` |
| `intended-job-description.txt` | `06577fab0ca934efd8872e83dc691d76ac918710c6933e8245d0fce0e2163409` |
