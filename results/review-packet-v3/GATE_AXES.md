## Four axes, reported separately

**Candidate denominator: 14 prose cells.** 13 legacy JSON-envelope cells and 1 truncated cell are excluded and reported in their own columns below.

> Only contract-compatible prose answers with a retained response are in the success/failure denominator. Legacy JSON envelopes answer a different contract; a truncated cell has no answer at all.

### Axis 1 — initial judgment

- decisions: `{'revise': 14}`
- agrees with preregistered expectation: `{False: 6, True: 8}`
- *Agreement with the preregistered expectation is one mechanical fact, not evidence the objection was correct.*

### Axis 2 — edit applicability (a property of the scaffold)

- `{'no_proposal': 12, 'proposal_drawn': 2}`
- scaffold refusal reasons: `{'ambiguous_or_multiple_issues': 8, 'proposed_exact_owned_source_bullet': 2, 'not_attempted': 4}`
- *A refusal here is a limitation of the narrow repair scaffold, not a finding about the reviewer's judgement.*

### Axis 3 — approval of the exact proposed document

- `{None: 12, 'not_approved': 2}`
- re-review document identity: `{'different_document': 5, 'same_document': 2, None: 7}`
- *A retained re-review of a different document is reported as no observation, never as an approval or a refusal.*

### Axis 4 — release

- released: **0 of 14**
- withheld: 14 · undetermined: 0
- reasons: `{'revision_unresolved': 8, 're_review_not_approved': 2, 'initial_review_unverifiable': 4}`
- *Undetermined means the gate could not complete its own rule with the evidence retained. It is neither a release nor a refusal.*

| Case | Cell | Decision | Edit | Corrected-doc approval | Release |
|---|---|---|---|---|---|
| `faithful_paraphrase` | `revised+advisory` | `revise` | `ambiguous_or_multiple_issues` | `None` | withhold (`revision_unresolved`) |
| `supported_method` | `revised+advisory` | `revise` | `ambiguous_or_multiple_issues` | `None` | withhold (`revision_unresolved`) |
| `unsupported_method` | `revised+advisory` | `revise` | `proposed_exact_owned_source_bullet` | `not_approved` | withhold (`re_review_not_approved`) |
| `missing_fact` | `revised+advisory` | `revise` | `ambiguous_or_multiple_issues` | `None` | withhold (`revision_unresolved`) |
| `wrong_record_attribution` | `revised+advisory` | `revise` | `ambiguous_or_multiple_issues` | `None` | withhold (`revision_unresolved`) |
| `transferable_skill` | `revised+advisory` | `revise` | `ambiguous_or_multiple_issues` | `None` | withhold (`revision_unresolved`) |
| `omitted_record` | `revised+advisory` | `revise` | `not_attempted` | `None` | withhold (`initial_review_unverifiable`) |
| `faithful_paraphrase` | `revised+enabled` | `revise` | `ambiguous_or_multiple_issues` | `None` | withhold (`revision_unresolved`) |
| `supported_method` | `revised+enabled` | `revise` | `ambiguous_or_multiple_issues` | `None` | withhold (`revision_unresolved`) |
| `unsupported_method` | `revised+enabled` | `revise` | `proposed_exact_owned_source_bullet` | `not_approved` | withhold (`re_review_not_approved`) |
| `missing_fact` | `revised+enabled` | `revise` | `not_attempted` | `None` | withhold (`initial_review_unverifiable`) |
| `wrong_record_attribution` | `revised+enabled` | `revise` | `not_attempted` | `None` | withhold (`initial_review_unverifiable`) |
| `transferable_skill` | `revised+enabled` | `revise` | `ambiguous_or_multiple_issues` | `None` | withhold (`revision_unresolved`) |
| `omitted_record` | `revised+enabled` | `revise` | `not_attempted` | `None` | withhold (`initial_review_unverifiable`) |

| Excluded cell | Kind | Why |
|---|---|---|
| `current+advisory`/faithful_paraphrase | `legacy_json_envelope` | legacy answer to the JSON span contract; the prose gate cannot read it |
| `current+advisory`/supported_method | `legacy_json_envelope` | legacy answer to the JSON span contract; the prose gate cannot read it |
| `current+advisory`/unsupported_method | `legacy_json_envelope` | legacy answer to the JSON span contract; the prose gate cannot read it |
| `current+advisory`/missing_fact | `legacy_json_envelope` | legacy answer to the JSON span contract; the prose gate cannot read it |
| `current+advisory`/wrong_record_attribution | `no_response` | no response retained (truncated) |
| `current+advisory`/transferable_skill | `legacy_json_envelope` | legacy answer to the JSON span contract; the prose gate cannot read it |
| `current+advisory`/omitted_record | `legacy_json_envelope` | legacy answer to the JSON span contract; the prose gate cannot read it |
| `current+enabled`/faithful_paraphrase | `legacy_json_envelope` | legacy answer to the JSON span contract; the prose gate cannot read it |
| `current+enabled`/supported_method | `legacy_json_envelope` | legacy answer to the JSON span contract; the prose gate cannot read it |
| `current+enabled`/unsupported_method | `legacy_json_envelope` | legacy answer to the JSON span contract; the prose gate cannot read it |
| `current+enabled`/missing_fact | `legacy_json_envelope` | legacy answer to the JSON span contract; the prose gate cannot read it |
| `current+enabled`/wrong_record_attribution | `legacy_json_envelope` | legacy answer to the JSON span contract; the prose gate cannot read it |
| `current+enabled`/transferable_skill | `legacy_json_envelope` | legacy answer to the JSON span contract; the prose gate cannot read it |
| `current+enabled`/omitted_record | `legacy_json_envelope` | legacy answer to the JSON span contract; the prose gate cannot read it |
