# Semantic judgments — fixed-build batch (`finalB-`)

Every disputed sentence from `audit.json`, judged by reading it against the
frozen source. A novel word is **not** an invented fact; the judgment is whether
the sentence states something the source does not.

Verdicts:

- **FAITHFUL** — same fact, different wording. No new claim.
- **FAITHFUL (bridge)** — the added wording is an authorized, server-validated,
  reviewer-approved skill label.
- **ADDITION** — the sentence states a claim the source does not make. This is
  the defect class.

---

## FAITHFUL — wording only

| Sentence | Source | Why faithful |
|---|---|---|
| Tracked monthly fuel consumption across 14 depots, comparing each month against the same month in the prior year. | …and compared each month against the same month in the prior year. | Participle for finite verb. Same comparison. |
| Rebuilt the shipment delay log used by dispatch teams | …that the dispatch team relies on. | "relies on" → "used by"; singular → plural. Same fact. |
| Maintained parts availability records across 7 service branches | Maintain parts availability records for 7 service branches. | Tense and preposition only. |
| Prepared monthly cost summaries reviewed by regional management | Prepared the monthly cost summary reviewed by the regional manager. | Plural and "manager" → "management". Same fact. |
| Reduced after-hours callouts by 27% by implementing thermographic inspections. | …by adding thermographic inspections. | "adding" → "implementing". Same action. |
| Trained 4 apprentices on lockout/tagout safety procedures | Trained 4 apprentices on lockout/tagout procedures. | Lockout/tagout **is** a safety procedure; the adjective adds no claim. |
| Flagged slow-moving stock and proposed reduction plans adopted across 2 branches | …proposed the reduction plan adopted for 2 branches. | Singular → plural. Same fact. |
| Built the depot scorecard reviewed by supervisors during weekly operations meetings. | …that supervisors review in the Monday operations meeting. | A recurring Monday meeting is weekly. Fair reading, no new claim. |
| Executed stock planning via quarterly usage comparisons, informing reorder point adjustments for branch supplies. | Compared quarterly parts usage against the prior four quarters to set reorder points. | "informing reorder point adjustments" restates "to set reorder points". "branch supplies" is the same parts inventory. |

## FAITHFUL (bridge) — authorized skill label

Each label below appears in that run's `skill_bridges`, was validated against a
source segment, and was approved by the reviewer.

| Sentence | Bridge label |
|---|---|
| Audited 3 years … to identify repeat equipment failures through failure pattern analysis. | failure pattern analysis |
| Audited 3 years … through failure analysis techniques. | failure analysis |
| Reconciled 1,180 fuel invoices … via invoice reconciliation processes. | invoice reconciliation |
| Tracked monthly fuel consumption across 14 depots using trend analysis, comparing each month … | trend analysis |
| Tracked monthly fuel consumption across 14 depots using period-over-period analysis to highlight variances. | period-over-period analysis |
| Tracked monthly fuel consumption … using trend analysis; enabled year-over-year comparisons for operations review. | trend analysis |
| Built the depot scorecard through scorecard development, enabling supervisors to review performance … | scorecard development |
| Maintained parts availability records for 7 service branches through systematic inventory record maintenance. | inventory record maintenance |
| Identified slow-moving stock and proposed a reduction plan adopted by 2 branches using slow-moving stock identification techniques. | slow-moving stock identification |
| Flagged slow-moving stock and proposed inventory optimization plans adopted by 2 branches. | inventory optimization |
| Diagnosed and repaired switchgear faults across 9 substations using fault diagnosis techniques. | fault diagnosis |

**Note on "year-over-year".** The source says "compared each month against the
same month in the prior year". That *is* a year-over-year comparison. The phrase
is a faithful description, not an addition — this is exactly the case a lexical
check gets wrong.

## ADDITION — a claim the source does not make

| Sentence | Source | What is added |
|---|---|---|
| Reduced average fault-response time from 4.5 hours to 2.75 hours **through systematic troubleshooting**. | Cut average fault-response time from 4.5 hours to 2.75 hours. | A **causal mechanism**. The source states the result, never how it was achieved. |
| Built the depot scorecard through scorecard development, **enabling supervisors to review performance** during weekly operations meetings. | …that supervisors review in the Monday operations meeting. | A **purpose clause**. The review is stated; "enabling … to review performance" recasts it as an outcome the scorecard produced. |
| Built depot performance scorecards **leveraged in weekly leadership reviews** | …that supervisors review in the Monday operations meeting. | **Audience elevation.** "supervisors" → "leadership". The source names supervisors, not leadership. |
| Completed 8,000 apprenticeship hours **fulfilling journeyman certification requirements**. | Completed 8,000 apprenticeship hours under a licensed journeyman. | A **purpose/qualification claim**. The source does not say the hours fulfilled certification requirements. |
| Diagnosed and repaired switchgear faults across 9 substations **supporting grid reliability**. | Diagnosed and repaired switchgear faults at 9 substations. | An **outcome claim**. Grid reliability is never mentioned. |
| Performed preventive maintenance on welding and cutting equipment **to ensure operational readiness**. | Performed preventive maintenance on shop welding and cutting equipment. | A **purpose claim**. Operational readiness is never mentioned. |
| Fabricated and repaired structural steel assemblies for 14 industrial contracts **using shop welding equipment**. | Fabricated and repaired structural steel assemblies for 14 industrial contracts. | **Cross-bullet transplant.** "shop welding and cutting equipment" is the *next* bullet's subject, not this one's method. |
| Fabricated and repaired structural steel assemblies for 14 industrial contracts **using shop tools**. | (as above) | **Cross-bullet transplant**, and "tools" is not the source's word. |
| Pulled and terminated branch circuits during 31 commercial buildouts **under licensed supervision**. | Pulled and terminated branch circuits on 31 commercial buildouts. | **Cross-bullet transplant.** "under a licensed journeyman" belongs to the 8,000-hours bullet. |
| Maintained inventory accuracy across 7 service branches **through centralized** parts availability tracking. | Maintain parts availability records for 7 service branches. | **"centralized"** is a claim about how the records were kept. |
| Reduced after-hours callouts by 27% by implementing thermographic inspections **across facilities**. | …by adding thermographic inspections. | **Scope addition.** The source gives no coverage claim. |
| Flagged slow-moving stock and proposed a reduction plan **later** adopted for 2 branches. | …the reduction plan adopted for 2 branches. | **Temporal claim.** "later" is not stated. |

---

## Summary

| Verdict | Distinct sentences |
|---|---|
| FAITHFUL | 9 |
| FAITHFUL (bridge) | 11 |
| ADDITION | 12 |

The additions cluster into four shapes, all of which the engine's own controls
target: **purpose clause**, **outcome/causal claim**, **cross-bullet transplant**,
and **audience/scope elevation**. None is a novel-word artifact.

The one mechanical defect found by the audit is separate and unambiguous:
`finalB-bridge_opportunity-imaginative-4` omits the metric **"prior four
quarters"** from the Ridgeline record, so the comparison basis is dropped.
