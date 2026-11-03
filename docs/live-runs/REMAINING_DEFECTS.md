# Remaining defects after run 7 (allowance exhausted)

Run 7 (`0a267e04-affd-4cee-b944-26c574703039`, revision `b855087`) was withheld as
`source_facts_missing`. The reviewer marked **16 of 22** source segments represented,
**2 lines unsupported** and **6 segments missing**. Those numbers are the whole
diagnosis; nothing below is inferred from a hypothesis.

## 1. `- Accumulated 8,000 hours of hands-on field experience toward journeyman licensure, mastering core electrical installations.` (L000014, unsupported; segment 12 missing)

The source says:

> Completed 8,000 hours of supervised field work toward licensure.

Three separate corruptions in one line:

* **"supervised" → "hands-on"**. The source's qualifier is `supervised`, and it is
  load-bearing: 8,000 supervised hours is the licensure pathway. "Hands-on" drops it.
* **"toward licensure" → "toward journeyman licensure"**. The source says licensure;
  naming journeyman is a claim about which licence the hours counted toward.
* **"mastering core electrical installations"** — invented outcome, no basis in the
  source.

This is the same class as the `six apprentices → six crews` defect: a real number with
its subject changed. It survives the drafter's prompt because the rewrite is allowed to
tighten prose, and nothing in the prompt says a **qualifier attached to a number** is
part of the fact.

**Suggested fix:** state in the drafter prompt that a number's qualifier travels with
it — "8,000 hours of supervised field work" may be restated as "8,000 supervised field
hours" but not as "8,000 hours of hands-on experience". Pair it with a lexical check in
`_introduces_beyond` for a figure whose immediate qualifier is replaced, not merely a
figure whose subject is renamed.

## 2. `**Electrical:** installation, maintenance, commercial electrical systems, … apprentice supervision, inspections, panel labeling, …` (L000024, unsupported; segments 20–21 missing)

The skills block did not claim rank this time, but it did two things:

* **Flattened the source's skills into a category the source never used.** The source
  has `- Commercial electrical systems, conduit bending, panel labeling` and
  `- NFPA 70E, lockout/tagout, inspection checklists`. The draft relabelled the union as
  "Electrical:" and added `installation` and `maintenance`, which are activities, not
  the listed skills.
* **Still lost the two source skill lines as discrete rows.** Segments 20 and 21 were
  reported missing even though their words appear, because the reviewer requires the
  skill *line* to be represented as a skill listing. Merging everything into one
  category line reads as omission.

**Suggested fix:** two independent changes. (a) Require the skills section to preserve
the source's own grouping when the source groups skills, rather than inventing a
category structure. (b) Treat `install*`/`maint*` verbs as activities, not skills, so
they cannot enter a skills list — a narrow lexical rule, since these are the observed
offenders.

## 3. Segments 1–2 (the summary paragraph) still reported missing despite being restated

The draft's summary is:

> Journeyman electrician with 8+ years of commercial and industrial electrical
> experience across construction, facilities maintenance, and power distribution,
> proficient in NFPA 70E, Arc Flash safety, electrical systems, apprentice supervision,
> and inspections.

The first clause restates segment 1 and 2 almost verbatim. The reviewer still marked
them missing. The likely cause is that the summary line was judged **as one unit** whose
appended clause ("proficient in … apprentice supervision") is not established, so the
whole line became unsupported and therefore represented nothing.

This is the `_introduces_beyond` shape again — but note it landed on segment 1–2
(coverage) rather than on the appended clause (claim). That is arguably correct
behaviour: the reviewer decided a line containing an unsupported clause represents
nothing. The real defect is upstream — the editor appended a proficiency list to a
summary the source ends at "power distribution."

**Suggested fix:** tell the editor the summary must end where the source summary ends.
Do not append a proficiency list. This is one sentence of prompt and is testable offline
by asserting the summary section contains no skills enumeration.

## 4. Segment 15 (`*Associate of Applied Science, Electrical Systems Technology | 2016*`) missing

The draft's education line is
`**Associate of Applied Science, Example Technical College** -- 2016`.

The field **"Electrical Systems Technology" was dropped**. The institution, degree and
year are all correct; the major is gone. This is a pure coverage loss from the education
template, which models `**Degree, Institution** -- Date` and has no field slot.

**Suggested fix:** add the field to the education template
(`**Degree, Field, Institution** -- Date`) and require it when the source states it.
Alternatively add a `field` slot to `_ensure_all_sections`' education block, since
`EDUCATION_RECORD["field"]` already exists in the answer key and is currently never
emitted by any path.

## Cross-cutting note

`repair_calls` has been **0 in all three runs**, including the two that had unsupported
lines. The gate withholds on `missing` before attempting repair when both are present
(`source_correction.py`, the `if "missing" in incomplete:` branch). So the repair path
that could have fixed defect 1 without a new submission has never executed in
production. That ordering is deliberately untouched — see the comment at that call
site — because the protected run-4 recording contains only two exchanges and a replay
must not request a phase the recorder never saw. **Fixing it needs a capture that
records the repair phase**, which is the single highest-value change for reducing
submissions per defect.

## Verification available without spending

All three captures replay offline through the real engine:

```
python -c "from live_capture import replay; print(replay('<capture>.json')[0])"
```

Each returns `withheld / source_facts_missing`, 1 review, 0 repairs, 0 restores, all
exchanges consumed. Any prompt change can therefore be checked against the recorded
drafts for *what the writer would now produce*, but the captures record the **reviewer's**
answer, not the writer's — so a writer fix cannot be validated by replay alone. It needs
either a new submission or a writer-level test that asserts on generated prompts.
