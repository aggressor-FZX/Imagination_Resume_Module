# Realistic candidate — review package (UNAPPROVED, UNFROZEN)

**Status: awaiting review.** Nothing here is frozen, hashed for approval, or submitted.
The existing regression fixtures and captures are unchanged.

Real cities. Fictional employers, institutions, and person. Test-only, unroutable
contact details.

---

## 1. `candidate-main.txt` — full contents

```text
## Professional Summary
SYNTHETIC-ONLY Journeyman electrician with 12 years of electrical experience spanning
SYNTHETIC-ONLY commercial construction, industrial maintenance, and facility operations.

## Professional Experience

**SYNTHETIC-ONLY Northline Fabrication, Inc.** -- Spokane, WA
*Welder / Maintenance Helper | 03/2012 - 09/2015*
- SYNTHETIC-ONLY Fabricated and repaired structural steel assemblies for 14 industrial contracts.
- SYNTHETIC-ONLY Performed preventive maintenance on shop welding and cutting equipment.

**SYNTHETIC-ONLY Kettle Falls Electrical Contractors** -- Spokane, WA
*Electrical Apprentice | 10/2015 - 06/2019*
- SYNTHETIC-ONLY Pulled and terminated branch circuits on 31 commercial buildouts.
- SYNTHETIC-ONLY Completed 8,000 apprenticeship hours under a licensed journeyman.

**SYNTHETIC-ONLY Pend Oreille Power Systems** -- Spokane Valley, WA
*Journeyman Electrician | 07/2019 - 02/2023*
- SYNTHETIC-ONLY Diagnosed and repaired switchgear faults at 9 substations.
- SYNTHETIC-ONLY Cut average fault-response time from 4.5 hours to 2.75 hours.
- SYNTHETIC-ONLY Trained 4 apprentices on lockout/tagout procedures.

## Education
**SYNTHETIC-ONLY Spokane Falls Institute of Technology** -- Spokane, WA
*Associate of Applied Science, Electrical Systems Technology | 2015*

## Skills
- SYNTHETIC-ONLY Conduit bending, switchgear troubleshooting, thermographic inspection
- SYNTHETIC-ONLY Lockout/tagout, NFPA 70E
```

## 2. `candidate-supplement.txt` — full contents

```text
## Professional Summary
SYNTHETIC-ONLY Maintenance electrician focused on electrical distribution and
SYNTHETIC-ONLY building automation systems in occupied commercial properties.

## Professional Experience

**SYNTHETIC-ONLY Riverbend Facilities Group** -- Coeur d'Alene, ID
*Maintenance Electrician | 03/2023 - Present*
- SYNTHETIC-ONLY Maintain electrical distribution and BAS controls for 6 commercial properties.
- SYNTHETIC-ONLY Reduced after-hours callouts by 27% by adding thermographic inspections.

## Education
**SYNTHETIC-ONLY Inland Northwest Safety Academy** -- Coeur d'Alene, ID
*Certificate, NFPA 70E Arc Flash Standards | 2021*

## Certifications
- SYNTHETIC-ONLY Washington 01 Journey Level Electrician License

## Skills
- SYNTHETIC-ONLY Electrical distribution, BAS controls, thermographic inspection
- SYNTHETIC-ONLY NFPA 70E, lockout/tagout
```

## 3. Identity and chronology

**Candidate:** Dana R. Whitfield (fictional) · Spokane, WA
**Test contact:** `dana.whitfield@example.invalid` · `(509) 555-0147`
(`.invalid` is reserved by RFC 2606 and can never resolve.)

| Period | Employer | Role | Location | Doc |
|---|---|---|---|---|
| 03/2012 – 09/2015 | Northline Fabrication, Inc. | Welder / Maintenance Helper | Spokane, **WA** | main |
| 10/2015 – 06/2019 | Kettle Falls Electrical Contractors | Electrical Apprentice | Spokane, **WA** | main |
| 07/2019 – 02/2023 | Pend Oreille Power Systems | Journeyman Electrician | Spokane Valley, **WA** | main |
| 03/2023 – Present | Riverbend Facilities Group | Maintenance Electrician | Coeur d'Alene, **ID** | supplement |

Education: Spokane Falls Institute of Technology — AAS Electrical Systems Technology,
2015 (main) · Inland Northwest Safety Academy — Certificate NFPA 70E Arc Flash
Standards, 2021 (supplement).

No gaps, no overlaps, oldest to newest.

## 4. Complete F01–F22 ownership ledger

| # | Fact | Value | Owner | Doc |
|---|---|---|---|---|
| F01 | Summary claim | "Journeyman electrician with 12 years of electrical experience" | candidate | main |
| F02 | Summary claim | "spans commercial construction, industrial maintenance, and facility operations" | candidate | main |
| F03 | Employer | Northline Fabrication, Inc., Spokane, WA | employer | main |
| F04 | Employment record | Welder / Maintenance Helper, 03/2012 – 09/2015 | employment | main |
| F05 | Achievement | "Fabricated and repaired structural steel assemblies for 14 industrial contracts" | **Northline** | main |
| F06 | Achievement | "Performed preventive maintenance on shop welding and cutting equipment" | **Northline** | main |
| F07 | Employer | Kettle Falls Electrical Contractors, Spokane, WA | employer | main |
| F08 | Employment record | Electrical Apprentice, 10/2015 – 06/2019 | employment | main |
| F09 | Achievement | "Pulled and terminated branch circuits on 31 commercial buildouts" | **Kettle Falls** | main |
| F10 | Achievement | "Completed 8,000 apprenticeship hours under a licensed journeyman" | **Kettle Falls** | main |
| F11 | Employer | Pend Oreille Power Systems, Spokane Valley, WA | employer | main |
| F12 | Employment record | Journeyman Electrician, 07/2019 – 02/2023 | employment | main |
| F13 | Achievement | "Diagnosed and repaired switchgear faults at 9 substations" | **Pend Oreille** | main |
| F14 | Achievement | "Cut average fault-response time from 4.5 hours to 2.75 hours" | **Pend Oreille** | main |
| F15 | Achievement | "Trained 4 apprentices on lockout/tagout procedures" | **Pend Oreille** | main |
| F16 | Education | Spokane Falls Institute of Technology — AAS Electrical Systems Technology, 2015 | institution | main |
| F17 | Employer | Riverbend Facilities Group, Coeur d'Alene, ID | employer | supplement |
| F18 | Employment record | Maintenance Electrician, 03/2023 – Present | employment | supplement |
| F19 | Achievement | "Maintain electrical distribution and BAS controls for 6 commercial properties" | **Riverbend** | supplement |
| F20 | Achievement | "Reduced after-hours callouts by 27% by adding thermographic inspections" | **Riverbend** | supplement |
| F21 | Education | Inland Northwest Safety Academy — Certificate NFPA 70E Arc Flash Standards, 2021 | institution | supplement |
| F22 | Certification | Washington 01 Journey Level Electrician License | candidate | supplement |

Skills (both documents): conduit bending, switchgear troubleshooting, thermographic
inspection, BAS controls, lockout/tagout, NFPA 70E, electrical distribution.

### Metric ownership, isolated

| Metric | Sole owner |
|---|---|
| 14 industrial contracts | Northline |
| 31 commercial buildouts | Kettle Falls |
| 8,000 apprenticeship hours | Kettle Falls |
| 9 substations | Pend Oreille |
| 4.5 → 2.75 hours | Pend Oreille |
| 4 apprentices | Pend Oreille |
| 6 commercial properties | Riverbend |
| 27% callout reduction | Riverbend |

## 5. `intended-job-description.txt` — full contents

```text
Maintenance Electrician - Commercial Properties

Riverbend region facility services provider seeks a licensed maintenance electrician
for occupied commercial properties.

Responsibilities:
- Maintain electrical distribution, panels, and branch circuits across multiple sites
- Troubleshoot and repair building automation and HVAC control wiring
- Perform thermographic inspections and preventive maintenance scheduling
- Respond to after-hours electrical service calls
- Work under NFPA 70E and lockout/tagout requirements

Requirements:
- Washington 01 Journey Level Electrician License or able to obtain
- 8+ years electrical experience, including commercial and industrial work
- Experience with switchgear, motor controls, and building automation systems
- NFPA 70E arc flash training
- Ability to work independently across a multi-site portfolio

This posting is synthetic test material. The employer is fictional.
```

## 6. Expected output requirements

A run against this pack passes only if **all** of the following hold.

### Required in the output

| # | Requirement |
|---|---|
| R1 | All four employers present, each once: Northline, Kettle Falls, Pend Oreille, Riverbend. |
| R2 | Each employer keeps its own role and dates, with no exchange. |
| R3 | Both education records present with their own institution, award, field and year. |
| R4 | All eight metrics present, each under its own employer (table above). |
| R5 | The Washington 01 Journey Level Electrician License appears once, attributed to the candidate. |
| R6 | `SYNTHETIC-ONLY` appears **nowhere** in the rendered output. |
| R7 | Real locations preserved as written: Spokane, Spokane Valley, Coeur d'Alene — including the ID/WA split. |
| R8 | No duplicate employer, institution, credential, or metric. |

### Prohibited additions

| # | Prohibited |
|---|---|
| P1 | Any employer, institution, credential, certification body or metric absent from F01–F22. |
| P2 | **Motor controls** — in the posting, absent from the record. Must not appear in the résumé. |
| P3 | Any invented figure, percentage, duration, dollar amount or headcount. |
| P4 | Any outcome or causal claim the source does not state (the run-10 defect class). |
| P5 | Moving an employer to another state, or a role/degree to another employer. |
| P6 | An invented location substituted for a real one. |
| P7 | A duplicated bullet restated to appear like a second achievement. |

### Where the gaps may legitimately appear

| Content | Résumé | Gap analysis |
|---|---|---|
| Motor controls (required by ad, absent from record) | **Must be absent** | **May be listed** as a missing requirement |
| Anything else the ad asks for and the record lacks | Must be absent | May be listed |

**A truthful résumé is never withheld because the ad requests an absent skill.** Only an
invented claim is a defect: if a draft claims motor controls, the contaminated clause must
be **removed or repaired** — and the result withheld if repair fails. A candidate who
lacks motor controls and says nothing about it has produced correct output.

## 7. Capture-safety handling

`live_capture.assert_synthetic()` refuses to record unless **every** source document
contains `SYNTHETIC-ONLY`. This pack keeps that marker on every block-level factual
line — the same convention as the run-10 sources — so the guard is satisfied **without
being weakened**. It is a per-line prefix that `source_preservation._PROVENANCE_MARKERS`
strips from output, which is why a real city and a synthetic marker can coexist without
making the fixture unlike a normal upload: the marker is metadata, not content, and it
never reaches the rendered résumé (requirement R6).

The `*Role | dates*` and `*Degree | year*` sub-lines carry no marker, matching the
established convention; each is adjacent to a marked labelled line, so no section can
lose its provenance proof by being dropped.

## 8. Not done, pending review

- Not frozen. Not hashed for approval.
- No analysis submitted.
- No `.docx` rendering performed; the two `.txt` sources are the review artifact.
