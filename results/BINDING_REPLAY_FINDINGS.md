# Binding-decision replay — 2026-11-09

This corrects `RESULTS_V2.md`; no winner or production recommendation follows.
Source: unchanged `comparison-v2b-20261109/attempts.json`, SHA-256
`b0e4200c3ecca127bfebb46afde822da0a8e06ca693ca5681785477dadceed59`.
`binding-replay-v3.json` gives one stage-separated record per attempt. A harness
`approved` outcome is an experimental release, not a website delivery.

## Model decision versus harness release

All 14 prose reviewer first responses say `revise`. Legacy JSON responses do not
contain a prose decision; their decisions are mechanically inferred by the adapter.
The following five were nevertheless released by the harness, not approved by the
LLM:

| Cell | Case | First | Re-review | Release classification |
|---|---|---|---|---|
| revised+advisory | faithful_paraphrase | revise | revise | release-policy override |
| revised+advisory | supported_method | revise | revise | release-policy override |
| revised+advisory | wrong_record_attribution | revise | revise | release-policy override |
| revised+advisory | transferable_skill | revise | none | release-policy override |
| revised+advisory | omitted_record | revise | none | release-policy override |

All seven retained matrix re-reviews say `revise`; none concludes with a valid
re-review `approve`. Three of the five overridden releases followed a re-review.
Zero of 28 attempts record a rule change; therefore this matrix did not compare
enabled versus advisory rule actions. The switch instead changed release behavior.
Read the diff and revision log together: a logged `replaced` can be a no-op, and
re-review did not necessarily see an appropriate correction.

## Safety, detection, correction and delivery are separate

- A withhold can be safe without demonstrating the model detected the actual
  defect. A refusal of all seven cases is not four correct defect detections.
- A proposed edit is not successful just because it changed text or the harness
  logged it. After the wrong-employer edit, the wrong invoice clause was removed,
  **but reviewer commentary was introduced**. The omitted-record release still
  lacked Ridgeline; its reviewer discussed the record as though it were present.
- In the retained `revised+enabled / faithful_paraphrase` final draft, `*(omit)*`
  replaced the source-backed role/date text; the document was withheld, not safely
  corrected. The truncation has no retained reviewer response or final document.
- Fidelity and issue correctness require the blind audit, not the initial fixture
  expectation, the output label, or parser acceptance. No successful correction or
  faithful delivery rate is inferred from the available harness statuses.

## Reference checks are not comparable

Legacy evaluates source-ID, quote and offset; among 13 non-truncated responses the
recorded counts include 159 `offsets_off` citations. Six attempts recorded a
`reference_fabricated` failure, but that means the quote failed that adapter's
named-source check, not necessarily that the fact is invented: markdown variants
and evidence located in a different source are possible. Prose normally extracts
source IDs and discards attached quote text before verifying that an ID exists.
Zero recorded prose reference failures does **not** establish 14/14 grounded or
semantically supported answers. Quote existence, correct ownership and semantic
support are progressively stronger and must each be checked separately.

## Accounting and methodological correction

The previous `$0.00` claim is false: 40 retained telemetry events report
`usage.cost` totaling **$0.040509414** for the matrix plus its probe; this is a
provider-reported subtotal, not a reconciled bill. At least one additional
8,192-completion-token truncated call is missing its response and usage, so total
cost and exact HTTP request count cannot be established from these artifacts. The
runner read `usage.total_cost`, omitted the truncated call from its budget, and
did not reserve the next call's cost; its claimed hard dollar ceiling was not
established. The preregistered stopping rule on invalid citations was not enforced.
The retained run is exploratory, not a valid preregistered comparison.

## Candidate and remaining work

`experiment/binding_candidate.py` is experimental only. Regex observations do not
alter the review input or decide release. An initial valid `approve` can release;
`revise` withholds unless an unambiguous, owner-verified repair is proposed and a
new valid review approves that exact revised text. `cannot_assess`, malformed
reviews, unknown references and unresolved repairs withhold. No omission is
automatically inserted or deleted. This tests release control, not a factual
guarantee: a model approval and an existing source ID do not prove support.

`binding-audit-v3/blind/` contains all 28 retained entries and the two source
documents; its mapping is outside `blind/`. A packet is not an audit. Independent
auditor invocations failed to return a response in this session; no independent
judgments or final semantic labels have been fabricated. No additional paid calls
were made: the required independent audit is incomplete and a user-approved
remaining budget is not evidenced by the retained records.