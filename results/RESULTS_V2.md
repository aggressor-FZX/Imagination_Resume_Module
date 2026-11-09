# v2 four-way comparison — results

> **Superseded 2026-11-09:** This is the historical report, not a valid
> comparison of LLM-led final approval. See `BINDING_REPLAY_FINDINGS.md` and
> `binding-replay-v3.json`. In particular, five releases overrode `revise`,
> reference checks were not equivalent, and the reported $0.00 cost and
> claimed verdict counts were incorrect. The 28 retained attempts remain intact.

Run: `results/comparison-v2b-20261109` · **40 provider calls, $0.00 of the $2.00
ceiling** · 28 matrix attempts + 4 probe + retries · model `qwen/qwen3-coder`,
temperature 0 · fixture fingerprint `1aa800a3b14f22fd`.

Preregistered in `results/PREREGISTRATION_V2.md` **before** the first paid call.

---

## The table

Only the `semantic:` column is a judgement about the document. Every other column is
recorded so a reader can see *why* a cell produced the number it did.

| Cell | Verdicts reached | Incomplete | Reference failure | Correct approvals | False approvals | False refusals | Operational | Semantically correct |
|---|---|---|---|---|---|---|---|---|
| `current+advisory` | 3/7 | 1 | 3 | 2 | 1 | 0 | 1 | **2/3** |
| `current+enabled` | 4/7 | 0 | 3 | 1 | 1 | 1 | 0 | **2/4** |
| `revised+advisory` | 7/7 | 0 | 0 | 3 | 2 | 0 | 2 | **3/7** |
| `revised+enabled` | 7/7 | 0 | 0 | 0 | 0 | 3 | 0 | **4/7** |

Column integrity verified: every attempt lands in exactly one column
(`experiment/rescore.py` asserts this).

## The finding that actually separates the arms

**The legacy adapter cannot produce verifiable evidence on this fixture.**

| | `current` (both) | `revised` (both) |
|---|---|---|
| Reference failures | **6 of 14** | **0 of 14** |
| Verdicts reached | 7 of 14 | 14 of 14 |

Six legacy attempts cited something that did not verify. Re-measured offline, those
citations split three ways:

- **`reference_fabricated`** — the quote appears in no source document;
- **`reference_cites_draft`** — the quote is text from the **document under review**.
  Two of them quoted the literal string `DRAFT TO REVIEW:`, which is harness framing,
  not résumé content.

So on this evidence the legacy span-envelope contract, as prompted, does not
reliably distinguish *the source* from *the draft*. That is a property of the
contract, not of any individual answer, and it is why six legacy attempts never
reached a semantic verdict at all.

**Caveat, stated plainly:** this is one model at temperature 0 on seven cases. It is
enough to show the legacy adapter's evidence discipline is fragile. It is **not**
enough to estimate how often that happens in service.

## Where the two contracts actually disagree

| Case | `revised+advisory` | `revised+enabled` | Reading |
|---|---|---|---|
| `faithful_paraphrase` (approve) | approved ✓ | **refused** ✗ | enabled withholds a faithful document |
| `supported_method` (approve) | approved ✓ | **refused** ✗ | enabled withholds a paraphrase |
| `transferable_skill` (approve) | approved ✓ | **refused** ✗ | enabled withholds a permitted capability |
| `unsupported_method` (revise) | recorded, delivered | withheld ✓ | the intended difference |
| `missing_fact` (revise) | recorded, delivered | withheld ✓ | the intended difference |
| `wrong_record_attribution` (revise) | approved ✗ | withheld ✓ | advisory delivered a misattribution |
| `omitted_record` (revise) | approved ✗ | withheld ✓ | advisory delivered a dropped record |

`revised+enabled` withheld **all seven** cases, including all three faithful ones. It
therefore scores 4/7 semantically correct while being unusable as a delivery path: a
reviewer that refuses every document is correct on the revise cases by construction.

`revised+advisory` reached a verdict on all seven and got the two "must catch"
defects right in *detection* terms (both were reported), but **delivered** both
defective documents, because advisory does not block.

**Neither configuration is usable as it stands**, and they fail in opposite
directions:

- **advisory delivers defective documents** (2 false approvals: a misattributed
  figure and a dropped employer record);
- **enabled delivers nothing** (3 false refusals of faithful documents).

An acceptable policy has to withhold on `revise` *without* withholding on faithful
documents. Neither cell in this matrix does that, so this run does **not** identify a
configuration that can replace production approval.

## Reference failures are not semantic refusals

v1 filed its one completed legacy answer as a semantic refusal. Re-measured, all 10
of its quotes were **real source text at wrong offsets** — a harness convention
defect. The v2 taxonomy separates four evidence outcomes:

| Outcome | Meaning | Counts as |
|---|---|---|
| `matches` / `matches_length_convention` | quote at the stated offset | evidence verified |
| `offsets_off` | quote exists, offset differs | **convention**, not a refusal |
| `cites_draft` | quote is the document under review | refusal ground, own category |
| `fabricated` | quote in no source | refusal ground |
| `unknown_source` | source id names nothing supplied | refusal ground |

In this run, zero `reference:convention` results occurred — the legacy adapter's
failures were `fabricated` and `cites_draft`, i.e. genuinely bad pointers rather
than arithmetic.

## Truncation

`max_tokens=8192` (was 1200). **One** truncation in 28 attempts, against 11 of 12 in
v1. Every attempt recorded `finish_reason`, token usage, latency and cost; the
highest completion was 3,004 tokens, well inside the budget.

The single truncation (`current+advisory` / `wrong_record_attribution`) is excluded
from every semantic denominator and is **not** counted as a refusal.

## Omission handling

Four omissions were reported across the run. **`applied` was `no` in every case** —
the harness names the cited source and a destination, and inserts nothing.

Destinations were derived three ways, each recorded so the inference is auditable:
`anchor_in_draft` (the reviewer pointed at real text), `source_order` (inferred from
neighbouring records), `undeterminable` (the answer does not say). An ambiguous
anchor is reported ambiguous rather than guessed.

Groundedness is verified, not trusted: one report cited
`candidate-supplement.txt` for text that source does not contain, and was recorded
`Grounded: NO` rather than being accepted.

## Mode

The Conservative/Imaginative pair reached the **same** outcome on
`transferable_skill` in the probe, so the probe cannot separate "policy inert" from
"this case does not discriminate". Recorded as a finding, not a pass. **P7 is
therefore unsupported, not confirmed.**

## Scoring defects found and fixed while reporting

Two reporting-layer bugs, both of which made an arm look better or worse than it was.
Neither is a property of either contract:

1. **Double counting.** An operational failure on a `revise` case appeared in both
   `false_refusals` and `operational_failures`, inflating the failure count for
   whichever arm hit more parser problems.
2. **A failed parse scored as a correct judgement.** `unsupported_method` expects
   `revise`; one attempt was `operational_failure` and scored `semantic_ok = True`
   because "not approved" was compared with "expects revise". The harness had not
   read the answer. An operational failure now reaches **no** verdict.

Attempts were not modified. `experiment/rescore.py` rebuilds the table offline, and
it asserts column integrity — one attempt, one column.

## What this does not show

- **Nothing about production.** Both adapters are experimental; production approval
  is unchanged on this branch.
- **Nothing about a production failure rate.** 7 cases × 4 cells compares
  configurations; it does not estimate field reliability.
- **Nothing conclusive about semantics.** Author expectations remain **provisional**.
  Five are `undetermined` by mechanical audit and need the blind audit.
- **Nothing about other models.** One model, temperature 0.

## Standing conclusion

**No winner is declared.** The run supports one defensible statement:

> On this fixture, the prose decision contract produced verifiable evidence on 14 of
> 14 attempts while the legacy span envelope produced unverifiable evidence on 6 of
> 14 — and neither semantic-policy setting is deliverable, because advisory approves
> known defects and enabled refuses faithful documents.

Choosing between the approaches needs a policy that separates those two behaviours,
and the blind audit to confirm the expectations are right. That is the next
experiment, not this one.
