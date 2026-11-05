# Two-Mode Skill Bridging — Implementation Specification

**Status:** Proposed implementation specification; no implementation or release approval implied.  
**Date:** 2026-11-04  
**Policy:** `CogitoMetric_Official_Skill_Bridging_Plan_v1.md`, including the v1.1 clarifications on `resumeDraft_fix` (plan commit `c1e46bb`). The copy in this directory is v1.0 at time of writing; reconcile the copies before implementation.  
**Target:** `Imagination_Resume_Module`, with coordinated changes in `Login_Landing_page` where required. Verify the actual branches and integration baseline before editing. The feature worktree was created from `master` at `b23bba4`; do not assume it contains the newer correction and receipt work described below.

## 1. Purpose and policy

Allow useful, candidate-anchored transferable skill interpretations to reach the finished résumé in **Imaginative** mode, without admitting invented employment history. **Conservative** mode uses supplied facts and faithful normalization and skips bridging inference. Imaginative is the default; there is no Balanced inference mode. This policy is separate from any existing prose-style temperature control: inspect and document the latter before changing its behavior.

An inferred general label such as “trend analysis” may describe a candidate who compared sales trends; “time-series analysis” may be appropriate for analysis of measurements over time. Neither example establishes a particular algorithm, software product, implementation, certification, proficiency, or new event. Target-job requirements and O*NET data provide context, not evidence that a candidate performed an activity.

Do not add an approval questionnaire, a new admission-model call, mandatory taxonomy infrastructure, or a numeric confidence threshold treated as proof. Keep existing ownership, privacy, attribution, factual, section, completion and export controls.

## 2. Integration map and baseline checks

The existing UI sends `creativity_mode` through the multi-file analysis path. The application has a Researcher → Drafter → StarEditor pipeline, source-aware correction/publication gates and a PDF/DOCX renderer; actual route behavior and the branch versions must be verified before relying on those observations. Researcher output is currently excluded from the inspected Drafter prompt; the inspected editor takes domain vocabulary. A possible mixed `aggregate_skills` → export fallback exists, but reachability and exploitability must be established on the integrated route rather than assumed. A reported legacy editor path likewise requires verification: prior evidence describes a fix on a newer revision, so do not assume the old finding still applies.

**Before edits:** record `git status`, branch, HEAD, base, and relevant application/frontend/backend/bench SHAs; compare the worktree with the newer correction implementation. Identify active JSON, multi-file/SSE, precomputed research and export routes; record their request, cache and approval boundaries. Coordinate ownership of overlapping files with the spec-completion agent. Never reset another checkout or overwrite unrelated changes.

**Tests:** trace actual route calls with local fakes and unique sentinels; assert the relevant path was executed, not just that a helper exists. Record which documented commands and dependencies exist on the chosen baseline. No live paid tests are required for this phase.

## 3. Phase A — settings and request propagation

1. Introduce `skill_inference_mode` with exactly `imaginative` and `conservative`; omission means `imaginative`, unknown values are rejected at the external boundary and Imaginator boundary. Record the effective value and `imaginative-v1` policy version in request-local analysis state and approved export state.
2. Trace and propagate the field through UI, multipart/SSE and JSON routes, orchestration, precomputed research, writer, correction, cache and export. Define a single normalization point per trust boundary; do not silently reinterpret unknown values.
3. Present two inference options with a clear review notice for Imaginative. Inspect old `creativity_mode` consumers and decide separately whether to retain a prose-style control or migrate it; do not silently equate style and inference permission.

**Suggested tests:** default, both valid values, invalid values, multipart and JSON propagation, concurrent requests with different modes, UI default and selected state, and old-client omission. Capture the actual outbound Imaginator payload and assert the effective mode; test missing and invalid separately. Confirm style behavior independently.

## 4. Phase B — provenance and Researcher

### B.1 Concrete changes and ownership

| Existing integration point | Phase-B change | Boundary |
|---|---|---|
| `stages/researcher.py`: `Researcher.analyze`, prompt and response parser | Replace untyped implied-skill strings with anchored proposal records; remove generic fallback qualifications | Produces proposals, never final publication eligibility |
| `orchestrator.py`: Stage 1 / precomputed-research branch | Branch on effective inference mode **before** any Researcher call or reuse; pass immutable candidate context; preserve stage outcome | Must not convert an empty/failure result into successful generation |
| `app.py`: `/research` implementation and `ResearchRequest` | Make the fork's capability explicit: job-only input can produce role context, not candidate-authorized bridges | A precomputed result needs validated trust and candidate binding before proposal reuse |
| `imaginator_new_integration.py`: `run_new_pipeline_async` and orchestrator call | Carry the effective mode and source context into Stage 1; keep candidate skills separate from research | Do not merge proposals or JD vocabulary into candidate evidence |
| `models.py` and a proposed small `skill_bridging.py` module | Typed bundle/record validation, source-reference checking and bounded normalization | Reuse existing evidence models; no new service or database required |

These are implementation targets based on inspected code, not a statement that every branch has the same functions. Integrate with the chosen baseline rather than copying older implementations. Writer/reviewer changes belong to Phase C; this phase defines their input and proves it reaches that boundary.

### B.2 Input contract: what the Researcher may reason from

Construct one request-local context from the existing immutable candidate sources. Keep it separate from model-generated drafts:

- Effective `skill_inference_mode`, server-selected policy version, analysis identity and candidate-input version/fingerprint.
- Candidate source documents with their real IDs and immutable text. Reuse the correction system's source identity where possible; do not assume parser experience IDs exist.
- Structured experience/project references mapped back to those documents. Employer/project ownership comes from the source map, not a Researcher assertion.
- Candidate-supplied skills as candidate evidence, not blended with role requirements.
- Target job/title in an explicitly labeled **requirements/context** block.
- Optional bounded taxonomy context with its actual provenance; absence is valid.

The inspected Researcher currently slices `job_ad[:1500]` and serialized experiences to 1,000 characters. Blindly slicing JSON can cut a record in half and excludes later evidence without saying so. Replace that prompt construction with complete, individually bounded source excerpts and a coverage manifest. Preserve the original text for validation outside the prompt. A model can cite only an excerpt it was actually shown.

Use stable document IDs from input; attach server excerpt IDs and original character offsets when producing excerpts. If an experience cannot be mapped reliably, use a valid document/span anchor rather than inventing an experience ID. An occupation title alone may inspire a proposal, but a usable bridge must also cite candidate activity/background that supports its interpretation.

**Tests B-01–B-04:**

1. Two documents with similarly named employers retain different document/excerpt IDs; quoted text is resolved only in its referenced document.
2. An excerpt drawn from the second document remains valid even when earlier documents exhaust their individual quota; coverage/truncation is explicit.
3. Excerpt construction emits complete structured records, never truncated JSON. A removed excerpt cannot be cited successfully.
4. Empty/unmappable candidate context yields no bridges even when the JD contains a rich skill list; existing candidate facts remain unchanged.

### B.3 Output contract: separate proposal from permission

Use a typed research bundle with these distinct fields. Names below are proposed new contract fields, not claims about existing schema:

| Field | Contents / authority |
|---|---|
| `skill_inference_mode`, `policy_version` | Set by server context, not accepted from model output as authority |
| `analysis_id`, `candidate_input_version` | Binding supplied by the server; persisted in the existing analysis state |
| `skill_bridges` | Structurally validated **proposals**; not yet reviewed/eligible |
| `target_vocabulary` | JD/occupation terminology; never candidate skills by default |
| `job_requirements` | Requirements, including numerical requirements; never candidate achievements |
| `work_archetypes`, `framing_advice` | Advisory context; no authority to invent activities |
| `grounding` | Available/unavailable/not-requested status and real reference/version metadata |
| `research_status`, `reason_codes`, `coverage` | Stage outcome, safe diagnostic reasons and represented/omitted source IDs |

Each bridge carries the policy's minimal fields: server-generated `id`, neutral `skill`, `origin=researcher_inference`, valid source/excerpt reference, optional real `experience_id`, exact `evidence_quote`, nonempty `rationale`, `taxonomy_refs`, and server-selected `policy_version`. The server resolves source offsets and ownership; the model does not choose authoritative analysis IDs, approval flags or eligibility.

The rationale must explain the transfer, not merely repeat “required by the job.” Example: comparing weekly sales values → trend analysis, because the described activity identifies changes over time. Keep this a **proposal** until Phase C's existing reviewer evaluates semantic plausibility. Structural validation cannot prove that a rationale is true.

Preserve origin after review; do not relabel an inferred skill as `candidate_evidence`. Record narrowing/removal reasons compactly. Do not create a numerical confidence score that grants permission.

**Tests B-05–B-08:**

5. Accept a well-formed trend-analysis proposal with an exact candidate quote; assert it remains a proposal, not eligible or source evidence.
6. Strip/reject model-supplied `approved`, `eligible`, owner, analysis identity and policy overrides; assert none changes server context.
7. Reject empty labels/rationales, broken references and unsupported schema shapes per record; valid sibling proposals survive.
8. Assert vocabulary and job metrics are stored separately and never appear in the returned candidate-skill evidence collection.

### B.4 Execution sequence and Conservative cost guarantee

1. Validate and resolve mode at request entry; Stage 1 uses that effective value, not a model response.
2. **Conservative:** return a deterministic bundle with empty bridges and vocabulary, grounding `not_requested`, and explicit `skipped_by_policy` research status. Do this before Researcher invocation, taxonomy fetches or precomputed research reuse. Do not call the Researcher and discard its output afterward.
3. **Imaginative:** assemble bounded source/context input; attach usable existing taxonomy context; run the existing Researcher inference once through its provider adapter. Preserve existing bounded provider retry policy; do not add another admission or review call.
4. Parse response, validate each proposal against exact candidate references, assign server IDs and retain rejected-record reasons without exposing private text in logs.
5. Return structurally valid proposals plus advisory context to the writer/reviewer integration. Mark semantic review pending; Phase C decides final eligibility.
6. Carry stage outcome separately from proposal count: successful analysis can propose zero bridges; provider failure is not a successful empty response.

The inspected `research_used_fallback()` recognizes specific old default strings. Replace reliance on those strings with the new explicit outcomes, while retaining existing completion safeguards. A policy skip is intentional, not a provider failure; genuine Researcher failure must not be concealed by `completed` or produce generic “Problem Solving” qualifications. If source-only continuation is allowed by existing completion policy, record it explicitly; do not redefine that policy in Phase B.

**Tests B-09–B-12:**

9. Conservative: Researcher/provider/taxonomy spies assert **zero calls**, including when Imaginative precomputed output is supplied.
10. Imaginative with no precomputed result: one logical Researcher invocation, no new admission/reviewer invocation; record provider attempts separately if existing retry behavior occurs.
11. Successful zero-proposal response remains distinguishable from timeout, malformed response and provider failure; no generic default skill enters any output.
12. Required-generation failure remains a failure under existing completion checks; `skipped_by_policy` follows the intentionally skipped branch and cannot hide an unrelated writer failure.

### B.5 Precomputed research and the job-only fork

The inspected `/research` request is job-ad-only. It cannot supply a candidate-activity quote, so its output may be reused as role context only. Do not treat its old `implied_skills` strings as eligible bridges or claim that it performed candidate-aware bridging.

For this release, reuse that job-only context where appropriate and perform candidate-aware proposal generation once the candidate sources exist. A future/updated candidate-aware precomputed result may skip duplicate generation only if it carries verifiable server-controlled binding to the same source version, mode, policy and relevant job context. A matching JSON `origin` tag is not verification. Reuse an existing trusted internal artifact/receipt mechanism if available; otherwise regenerate rather than add an unbounded new trust system.

Do not reuse candidate quotes across analyses merely because skill labels match. Any transport changes needed to carry the bundle must be coordinated with the orchestrator contract; a frontend body cannot assert it is trusted precomputed research.

**Tests B-13–B-16:**

13. Old job-only result supplies role vocabulary but no candidate bridges; the candidate-aware call is not accidentally skipped.
14. Verified candidate-aware precomputation with matching bindings avoids the duplicate Researcher call.
15. Changed candidate text, job context, policy, mode or analysis binding invalidates privileged reuse; changed owner cannot inherit another analysis's proposals.
16. Client-created records copied into a precomputed field cannot bypass validation; old untyped cache entries cannot acquire bridging authority.

### B.6 Grounding: reuse actual data, not a new mandatory dependency

The inspected O*NET resolver is in `imaginator_new_integration.py`; resolution/enrichment occurs after writing and its enrichment branch is location-gated. Stage 1 must not assume that context already exists. `_resolve_onet_code()` resolves an occupation, not candidate competence. Its upstream passthrough checks format, whereas its LLM resolution validates existence through an API call; neither operation proves a proposed skill.

First accept relevant O*NET context already supplied through trusted server paths. If existing retrieval is reused earlier, keep it bounded and Imaginative-only for bridging purposes; do not couple bridging to location. Pass occupation code/title and only actually retrieved skills/activities/relationships. The current keyword-search and summary-validation helpers do not by themselves provide a complete skill/activity graph. Do not invent data from their names or add new endpoint claims without verifying the implementation.

Use taxonomy references only when present in the supplied context, retaining source/version when known. Model-generated citations to unseen records are removed/rejected. Distinguish `provided`, `unavailable` and `not_requested`; a known occupation code with no skill data is not “fully grounded.” Missing taxonomy permits labeled model-reasoned proposals, still subject to candidate-reference validation and later review. ESCO/local assets are optional and require verified version/licensing/packaging before use; do not restart suspended services.

**Tests B-17–B-20:**

17. Supplied taxonomy record reaches the prompt with its provenance; only its real reference can survive in a bridge.
18. Taxonomy says Python is relevant but candidate only compared sales totals: it cannot authorize Python as candidate evidence; proposal remains subject to prohibited-fact review.
19. No location or unavailable taxonomy does not prevent candidate-anchored general proposals; status is accurate and no citation is fabricated.
20. Conservative triggers no fetch solely for bridging. O*NET failures use a fake transport in tests and cannot cause external network access.

### B.7 Proposed size limits and failure behavior

Initial implementation limits below are design defaults to test, not measured safe capacity claims. Keep the existing Researcher model configuration and 2,048-token output budget initially; if contract tests show it is insufficient, change the budget explicitly with memory/cost rationale rather than silently truncating JSON.

- At most **6 bridge proposals** per response; 120 characters per skill label, 400 per quote and 500 per rationale. Oversized/malformed records are rejected rather than cut into a different meaning.
- Candidate prompt excerpts: up to **12,000 characters total**, selected round-robin across documents before adding further excerpts. Split at source line/paragraph boundaries; preserve original offsets and report omitted coverage. Never use an excerpt from one employer to authorize another employer's claim.
- Target-job context: up to **3,000 characters** with explicit truncation metadata. Taxonomy context: up to **3,000 characters** and complete records only. These are character bounds, not guarantees about provider token limits; verify actual prompt token fit for the configured model.
- Deduplicate labels by whitespace/case normalization **within the same source anchor**. Do not merge evidence or erase employer attribution when the same skill has different anchors.
- If the response envelope is malformed, return explicit failure with no bridges; if one record is malformed, retain independently valid siblings and report partial rejection. Preserve existing stage-requiredness rules for publication.
- Log counts, statuses, durations and safe reason codes only; do not log evidence quotes, personal information or provider error bodies. Usage telemetry remains distinct from publication authority.

**Tests B-21–B-24:**

21. Oversized arrays/fields and token-truncated JSON cannot slip through; record-level rejection does not silently modify quotes or labels.
22. Round-robin excerpt selection preserves document diversity, exact offsets and deterministic coverage; identical inputs produce the same context ordering.
23. Duplicate labels with different source owners retain distinct anchors; no cross-employer evidence merging.
24. Capture logs for normal, malformed and provider-failure paths with private sentinel values; assert sentinels and raw exceptions are absent while counts/outcomes remain observable.

### B.8 Phase-B deliverables and exit criteria

Deliver the typed bundle/validator, bounded prompt builder, mode-aware Stage-1 branch, explicit fallback outcomes, trusted precompute adapter and focused offline tests. Suggested test modules: `test_skill_bridge_records.py`, `test_researcher_mode_execution.py`, `test_researcher_grounding_context.py`, and `test_researcher_precomputed_binding.py`; naming is proposed and must not overwrite existing suites.

Run these with fake providers/transports and no network. Record actual call counts, limits exercised and test exit codes. Phase B is complete when Conservative makes no bridging call; Imaginative produces source-bound, typed proposals; malformed/untrusted output cannot become permission; grounding status and coverage are accurate; and the writer/reviewer boundary receives the mode and proposal context without mixing them with source evidence. Semantic acceptance, removal/narrowing in finished text and PDF/DOCX delivery remain Phase C/D acceptance, not claims made by these unit tests.

## 5. Phase C — writer and reviewer boundary

### C.1 Required outcome and scope

Implement **Researcher proposals → writer uses anchored proposals → existing reviewer accepts, narrows or removes them → only the resulting permitted text is published**. Do not require an additional pre-drafting admission call. Phase B's reference checks make a proposal usable drafting material, not final authorization. Only the existing semantic review, applied under the selected policy, makes its actual use eligible.

The exemption covers an identified general skill phrase, never the whole sentence, all words in a source document, or every Researcher output. Keep candidate facts and inferred interpretations distinguishable through correction. A rejected bridge should be removed/narrowed while retaining the résumé; unrelated unresolved facts, missing required source content, failed generation, malformed review or privacy/owner failure can still require withholding.

This phase changes the active writer and correction contracts. Canonical receipt/export binding is delivered in Phase D, but Phase C must hand it the exact corrected markdown and final bridge decisions. It must not produce a customer-visible intermediate approval lacking that binding.

### C.2 Traced implementation and concrete conflicts

Trace basis: local application checkout used for this specification, not a verification of the deployed service or the older feature worktree. Reconfirm function presence when integrating; source comments sometimes describe older behavior than the executable code.

| Actual code boundary | Current behavior found | Required Phase-C treatment |
|---|---|---|
| `orchestrator.py` → `Drafter.draft()` → `create_drafter_prompt()` | Research data is passed but not interpolated into the inspected standard Drafter prompt; experiences are capped at six | Add a distinct proposal block keyed to retained source anchors, not unrestricted research. Do not attach a seventh role's bridge to one of the six retained roles |
| `stages/drafter.py`: sparse-input branch | Under 50 counted input words, an appended instruction explicitly requests “standard industry tasks” and how work was “likely done” | Replace that real branch, not only an offline prompt helper; both modes forbid invented tasks |
| `StarEditor.polish()` / `_create_user_prompt()` | Editor takes rewritten experiences, source summary and skills; injects `domain_vocab[:10]`, sorts bullets by vocabulary and limits bullets/entries | Separate proposals from vocabulary and candidate facts; preserve source ownership despite ordering/truncation; prevent the unconditional vocabulary block becoming skill authority |
| `StarEditor.polish()` → `_ensure_all_sections()` → `source_preservation.preserve_source_facts()` | Section assembly is followed by deterministic preservation before numeric verification | Audit this exact post-processing path: it can erase a permitted bridge before semantic review sees it |
| `source_preservation.preserve_skills_block()` | When source skill lines exist, a nonliteral term can cause replacement of the entire Skills block with source-only lines | Keep source skills, but do not blanket-delete structurally valid proposal uses awaiting review; no early promotion to source evidence |
| `preserve_source_summary()` / `ensure_required_headings()` | Unsourced summary content is removed, and changed summary wording can be replaced verbatim with source summary; summary heading is source-conditional | Allow neutral capability framing under Imaginative while retaining all source summary facts; Conservative keeps source-only behavior. Do not fill empty sections with inferred history |
| `app.py`: `_analyze_resume_impl()` | Snapshots source documents before generation; preflight checks completion/structure; calls `_run_source_correction()` only for valid completed generation | Carry immutable server mode/proposal context beside the sources. Do not use pipeline-generated aliases as evidence or bypass preflight |
| `_run_source_correction()` → `correct()` → `CorrectionProvider` | Sources are length-preservingly redacted; correction receives only markdown, sources, provider and budgets | Add explicit bounded policy/proposal context to this real boundary, preserving disclosure and coordinate correspondence |
| `correction_protocol.indexed_review_request()` / `strict_review_from_indexed()` | Production wire protocol judges whole lines; supported lines need named source indices and lexical overlap; maps each line to a single full-line claim | Add phrase-scoped bridge decisions inside existing review calls; do not mark an entire mixed line supported merely because a bridge is allowed |
| `source_correction._validate_review()` | Strict claim partitions, exact citations, source-item coverage, lexical sharing and `_introduces_beyond()` checks | Support scoped interpretations without relaxing source citations, entity/numeric checks or coverage globally |
| `correct()` repair/restoration loop | Repairs only flagged spans, then full re-review; missing source facts may trigger bounded restoration; at most two repairs, existing deadlines/call budget | Rebind bridge occurrences on every changed draft; preserve repair-before-completeness ordering and budget limits |
| `app.py`: after approved correction | Rechecks numeric support and explicit entities, builds fresh response, issues receipt; intentionally discards earlier pipeline provenance | Carry only validated final bridge decisions/mode into the fresh approved state; raw proposal metadata must not be forwarded wholesale |

The normal `/analyze` path accepts a dictionary result and returns through the new pipeline/correction branch; older generation/editor code appearing later in that function is not evidence it executes. A separate local multipart handler exists. Establish route reachability with tests before modifying legacy prompts; do not rewrite inactive functions merely because their names resemble active stages.

### C.3 Request-local policy and proposal context

Add an explicit, immutable context argument through the real active call chain. Proposed conceptual contents:

- Effective inference mode and policy version from validated request/server state.
- Analysis and immutable candidate-input binding from Phase B.
- Structurally validated proposal records, including exact source anchors and rationale; never a client assertion of eligibility.
- A current-draft identity and an application-owned map of exact occurrences to proposal IDs, section and applicable employer/project scope.
- Final review decisions and resulting permitted wording, distinct from original proposals.

Do not append bridge definitions to `SourceDocument.text`, inventory segments or original resume strings. That would make the model's proposal appear to be a candidate fact, contaminate metrics checking and invalidate source completeness. Do not use `RUN_METRICS`, mutable globals, draft notes or a caller-supplied `origin` flag to authorize an exception. Existing callers with no context retain the old source-only correction contract; the new inference-enabled route must explicitly supply its validated context.

Treat all candidate text, rationale and taxonomy text as data. A rationale containing instructions cannot override policy. A matching source quote is necessary, not semantic proof: the reviewer must still judge whether “compared weekly figures” supports “trend analysis” and whether the actual use overstates it.

### C.4 Writer and editor prompt changes

1. Retain candidate facts as the source of ordinary factual claims. Add a separate **TRANSFERABLE-SKILL PROPOSALS — PENDING REVIEW** block containing only Phase B's validated labels, anchor references and rationale; do not reintroduce raw `implied_metrics`, occupational duties or full research blobs.
2. State the scoped permission: Imaginative may use the proposed general label in Skills or neutral capability language, or use it to characterize an existing anchored activity. It may not add events, specific implementation/algorithm/software, credentials, seniority, proficiency or outcomes. Conservative receives an empty proposal collection and a no-bridging instruction.
3. Replace contradictory blanket “same words only” rules for skill labels with the narrowly scoped exception. Leave those restrictions intact for credentials, tools, numerical claims and other concrete facts. Do not merely append a permissive paragraph after an unchanged prohibition and hope the model resolves the conflict.
4. Replace the actual sparse-input expansion instruction. Fewer notes may yield fewer/shorter bullets; no mandated bullet, keyword, metric or word-count quota. An accepted skill does not authorize “developed a forecasting model” from comparing figures.
5. Keep gold-example content syntax-only. Never source bridge evidence from golden bullets, JD, another candidate, or another role's draft. Proposals anchored to omitted roles may appear as general candidate capabilities only if the underlying source stays available to the reviewer; they must not be placed under the wrong retained employer.
6. Allow capability-only summaries in Imaginative when useful, even if no source summary heading exists; retain supplied summary facts when they do exist. No years, proficiency or role-specific specialization is implied by a bridge. Conservative must not acquire a generated capability claim by sharing a prompt/template with Imaginative.

**Tests C-01–C-05:** capture prompts from actual `Drafter.draft()` and `StarEditor.polish()` calls, both modes and both sparse/normal input. Assert the bridge block, immutable anchor and explicit limits reach the provider; raw research metrics/JD-only tools do not. Exercise 49/50-word sparse boundary using the implementation's counting rule. Mutation: restore the actual “standard industry tasks” suffix and prove the R-5 prompt test fails. Test a bridge anchored beyond the six-experience cap, and reordered/capped editor bullets, to ensure no employer reassignment.

### C.5 Preservation must protect facts without deleting the new policy

Pass policy/proposal context to the deterministic preservation step. Do **not** disable preservation wholesale in Imaginative or treat every absent-source word as permitted.

- Preserve explicit skill lines from every source document. Permit identified proposal labels to survive pending semantic review, without replacing the entire Skills block solely because such a label is nonliteral. Unknown labels do not inherit that allowance. Keep the later semantic reviewer responsible for accept/narrow/remove, including when the source has no Skills section (a current early-return case).
- Preserve source summary meaning, dates, education fields and numeric qualifiers. Make summary restoration coexist with neutral proposal-based capability language; avoid the current full-summary replacement erasing every bridge. A missing summary in the source is not itself a ban on capability framing under Imaginative.
- Keep neutral category formatting distinct from qualifications: a category heading cannot manufacture “Leadership,” “Expert” or certification. Case/format normalization must not turn “training” into “crew leadership.”
- A restoration or section rebuild that changes the text invalidates previous occurrence offsets. Recompute occurrences from the resulting draft; never copy a permission span onto new text at the old offset.
- Do not restore a bridge the reviewer removed simply because it still appears in the proposal collection. After semantic decisions, preservation may retain only surviving permitted labels; original source facts still have their own independent retention requirements.

**Tests C-06–C-10:** source Skills plus a valid nonliteral proposal both reach review; no source Skills still allows the proposal path; two-document Skills are not reduced to document one; a source summary plus capability phrase retains its original facts; no source summary allows neutral Imaginative framing but no Conservative inference. Include supervised-hours qualifiers, degree field, empty Projects/Certifications and rejected bridge non-resurrection. Run preservation twice to test idempotence and ensure source-only callers retain existing behavior.

### C.6 Extend the existing review protocol, not the number of reviewers

This is the central integration task. Changing only `_REVIEW_CONTRACT` in `source_correction.py` is insufficient: `CorrectionProvider` translates the payload to `correction_protocol.REVIEW_CONTRACT`, and `strict_review_from_indexed()` rejects unexpected keys. The provider-visible request, response parser, strict validation and engine must agree.

**Proposed wire extension inside initial/post-patch/post-restore review calls:** supply mode/policy and bounded bridge records separately from numbered source segments. The application discovers exact whole-label occurrences in the current draft (word boundaries, Markdown-aware visible text, Unicode-safe mapping to original line offsets), assigns occurrence IDs and sends their bridge IDs and scope. Do not use a substring match: `R`, `AI`, or `analysis` inside unrelated words cannot grant permission.

Ask the same reviewer for its ordinary line verdict/evidence/represented/missing result **plus one decision per supplied bridge occurrence**: `accept`, `narrow`, `remove`, or `uncertain`, a fixed reason code, and the candidate source segment indices establishing the activity. Narrowing also names the proposed broader replacement label. The reviewer must judge the complete line's other facts independently. The server owns IDs, exact original ranges and citations; the model must not invent offsets or add bridge IDs.

**Mapping and validation rules:**

1. Every offered occurrence receives exactly one decision; duplicate, unknown or omitted IDs are incomplete/malformed review, not approval. Accept only references valid for the same source-bound proposal and analysis. A reviewer-invented bridge is not admitted.
2. An accepted occurrence becomes a distinct interpretation-bearing claim span, retaining candidate-activity citations and bridge identity; surrounding text remains ordinary factual/structural spans. Keep gapless, nonoverlapping line coverage. A bridge decision cannot override an unsupported/uncertain verdict for residual facts on that line.
3. The old indexed adapter creates one claim per entire line. Extend that conversion explicitly: accepted label spans are split out using application-owned ranges. Ordinary source-only lines retain the established path. Do not pretend an unchanged full-line adapter provides phrase-level authorization.
4. Pure inferred Skills labels can have no lexical overlap with their activity quote. Allow the lexical-overlap floor to differ **only for the exact reviewed bridge span**, requiring its validated activity anchor and explicit acceptance. Do not remove the floor for unrelated lines or force fake overlap by inserting source words into the label.
5. Likewise, `_introduces_beyond()` may reject capitalized multiword labels. Exclude only the exact authorized phrase from the added-name check after validating its source association; still evaluate surrounding numbers, tools and names. Never add bridge labels to the vocabulary of whole source documents or remove numeric checks.
6. A bridge-only Skills phrase does not prove that all details of its source activity are represented. Source inventory coverage still requires the activity/qualifiers to survive elsewhere or in a genuinely complete rendition. Do not mark “compared sales in 12 branches” represented merely because “trend analysis” is present.
7. Existing line-level unsupported judgments may still require whole-line repair. Keep this limitation explicit: do not automatically upgrade a mixed line based on a bridge accept decision. The existing repair model can rewrite that flagged line, followed by full review; no extra admission model is introduced.

**Tests C-11–C-17:** exercise provider translation and strict engine together, not a mocked-away parser. Accept a zero-word-overlap skill with a valid anchor; reject the same text without context; reject unknown/duplicate/unjudged occurrence IDs and wrong source indices. Test title case, Markdown emphasis, Unicode characters, word boundaries, repeated labels in different sections and overlapping proposed labels. Confirm accepted bridge plus unsupported Python/20%/leadership remains flagged; ensure a bridge label alone does not satisfy source activity completeness. Mutation: allow whole-line exemption and prove mixed-claim negative controls fail.

### C.7 Narrowing, removal, repair and occurrence lifecycle

Do not solve overreach by publishing a reviewer verdict alongside unchanged text. Only the resulting permitted content may be approved.

- **Accept:** authorize the exact reviewed occurrence in this draft and this scope, without promoting its origin to source evidence.
- **Narrow:** validate a nonempty bounded neutral replacement and apply it through the existing flagged-span/line repair path, then review the resulting text again. Keep the original proposal ID plus the narrowed wording/decision history in server state; the original stronger label is not authorized. Narrowing cannot introduce Python, an algorithm or a credential absent candidate support.
- **Remove:** flag the occurrence/use for existing repair. A standalone skill item can be removed; an embedded phrase must be repaired without dropping supplied facts or leaving broken factual meaning. Do not delete an entire mixed line automatically simply to remove one label.
- **Uncertain or unsupported surrounding fact:** no permission is granted. Use existing bounded repair/re-review; withhold if unresolved. Reviewer transport/schema failure is not equivalent to a deliberate bridge rejection that can safely be removed.

After every repair, restoration or deterministic edit, regenerate line/occurrence IDs against the current text and re-evaluate all bridge uses in the complete existing review call. An old acceptance cannot authorize the same offset in a new string or a moved phrase under another employer. A narrowed phrase must be reviewable against its original candidate anchor, not a newly invented source entry.

Preserve the current engine's order: repair flagged claims before final missing-source coverage decisions. A rejected bridge must not make repair unreachable by being prematurely classified as missing candidate evidence. Missing source facts remain independently recoverable under existing restoration limits. Restoration currently restricts replacements to source wording; it must not manufacture inferred labels or erase an accepted bridge unnoticed—any changed line is re-reviewed. If the existing budget cannot complete required repair plus re-review, withhold accurately; do not increase call limits or add a reviewer to make tests pass.

**Tests C-18–C-23:** accept unchanged bridge with baseline inventory/review call counts; narrow/remove through existing repair plus re-review without an extra admission call; reject no-op repair and unresolved narrowing; repair a mixed bridge/unsupported-achievement sentence while retaining source facts; test simultaneous rejected bridge and missing source item reaches repair first; test restoration changes line length/order so stale acceptance is unusable. Include two repair attempts exhausted, cancellation, deadline exhaustion and incomplete-review recovery; never issue final approval for unreviewed replacement text.

### C.8 Disclosure, output and handoff to publication

`_run_source_correction()` redacts sources length-preservingly before constructing engine `SourceDocument`s. Original Phase-B quotes may therefore differ from the exact slices available to the correction engine. Carry validated source coordinates through this mapping; derive provider-visible quotes from the redacted sources. Do not compare a raw quote to redacted text and then bypass quote checking when it fails. Protect rationale and proposal text with the same approved disclosure checks as other outgoing payload fields; no raw private quote may leak through the new context.

Ensure indexed translation preserves public policy/bridge fields: `CorrectionProvider` removes underscore-prefixed fields from outgoing JSON, and adapters reconstruct requests. A context placed only in private adapter keys will never reach the reviewer. Preserve a private application-owned copy for verification, but send the bounded, redacted context the model actually needs.

Return only final surviving decisions and occurrences from the approved correction result, bound to the final markdown. `app.py` currently constructs a fresh response and drops earlier provenance; explicitly carry the validated state rather than accidentally losing it or copying raw pipeline proposals. Keep post-correction numeric verification and explicit employer/education/entity checks against candidate sources; inferred skills do not broaden those checks. Phase D must bind this state and resolved text before issuing the final export receipt. Streaming draft tokens remain previews, not approved content, and must not be usable as export state.

**Tests C-24–C-27:** PII inside quote/rationale is absent from provider JSON and logs; redacted source coordinates remain valid; the actual provider-visible review request contains mode/bridge context; approved correction result includes only surviving labels and correct final occurrences, not removed proposals or stale draft aliases. Assert the real `/analyze` result cannot issue a usable receipt after incomplete correction, failed required generation or dropped binding, and cross-request contexts cannot authorize another analysis.

### C.9 Paired behavioral fixtures and policy regression table

Use small synthetic fixtures with immutable source ownership and exact expected outcomes. Each positive has a one-change negative; do not rely on a blacklist of one run's invented words.

| Source / proposed use | Imaginative expectation | Paired prohibited change |
|---|---|---|
| Compared weekly sales figures → “trend analysis” | General skill may survive review and final markdown | “ARIMA forecasting”, Python or a revenue increase without support |
| Analyzed measurements over time → “time-series analysis” | Neutral general capability may survive | “Developed/deployed a forecasting model” invents an event |
| Sonar work with real signal-analysis activity → broader signal/time-series terminology | Reviewer judges actual anchor and degree of specificity | ML model development or expertise from job title alone |
| Trained apprentices → a proposed general communication skill | Assess neutral skill, retain training facts | Crew leadership, supervision, team size or rank not established by that activity alone |
| Employer A activity → skill used in A's existing bullet | Permitted if that use relabels the same activity | Move skill-bearing activity/achievement to Employer B |
| Source Skills + one inferred label + empty Projects | Retain source skills and accepted label; Projects stays empty | Invent project to demonstrate the skill |
| No source summary → neutral capability-only framing | Allowed in Imaginative after review | Add years, “advanced”, “proven”, leadership or certification |

For every row, Conservative has no Researcher proposal and no nonliteral bridging entitlement; source activities remain. An explicit source-backed skill stays allowed in either mode even when its wording resembles a bridge label. Match by provenance/use, not by globally banning the label.

### C.10 Work sequence and exit criteria

Implement in this order within the agreed application baseline:

1. Add request-local policy/proposal context plumbing and backward-compatible source-only defaults; prove it reaches the actual provider-visible review.
2. Update actual sparse/normal writer/editor prompts and proposal-use attribution; prove intent with prompt-capture and R-5 mutation tests.
3. Adapt preservation so proposals survive to review while original facts remain; add idempotence and multi-document regressions.
4. Extend indexed review, strict validation and interpretation-span mapping together; test positive nonliteral labels and mixed-claim negatives before permitting publication.
5. Integrate narrowing/removal with existing repair/re-review and restoration; prove budgets and coverage ordering are unchanged.
6. Carry only reviewed final state to canonical publication/export integration; run a real local route test with fake generation/review providers and real protocol/engine, not a standalone helper positive.

Suggested additional test modules: `test_bridge_writer_prompts.py`, `test_bridge_preservation.py`, `test_bridge_review_protocol.py`, `test_bridge_correction_lifecycle.py`, and `test_bridge_analyze_route.py`; reuse existing fixtures where possible. Keep the spec-completion agent's R-5 and generation-completion tests as independent acceptance, not weaker rewritten substitutes.

**Exit criteria:** an actual source-bound nonliteral skill reaches final approved markdown in Imaginative; the same proposal has no bridging entitlement in Conservative; rejected bridges narrow/remove without destroying source facts; unsupported concrete facts and wrong attribution still fail/repair; review context survives translation and redaction; no extra admission/reviewer stage exists; every edited draft is re-reviewed within existing budgets; and final approved bridge state is available for Phase D's canonical binding. Offline scripted responses prove integration, not model judgment. Holdout/human evaluation and separately authorized real-model validation remain separate evidence.

## 6. Phase D — skills, preview and export

### D.1 Customer route: endpoints and data flow

The following flow was traced in the local checkout. Paths are application contracts; configured hosts are resolved from environment, not hardcoded for implementation. It does not establish which revision is deployed.

| Hop | Endpoint / in-process boundary | Data entering | Data leaving / authority |
|---|---|---|---|
| 1. Resume Builder browser | `POST /api/analysis/multi-file?stream=1` on orchestrator backend | Multipart files, target role/ad/location, selected inference mode | SSE progress and final analysis response; HTTP/SSE transport success alone is not publication approval |
| 2. Orchestrator extraction | `POST /process` on Document Reader, for uploaded documents | Multipart document plus service API authentication | Extracted candidate text; orchestrator assembles distinct `source_documents` and combined original evidence |
| 3. Optional existing research fork | Imaginator `POST /research` through the existing server research caller | Job-only context in the inspected contract | Advisory precomputed research; not candidate bridge authorization (Phase B rules apply) |
| 4. Orchestrator writing request | Imaginator `POST /analyze` | `resume_text`, `source_documents`, target context, structured sections, server mode/proposal context and trusted owner header | Completed/reviewed markdown, canonical export object, signed approval and final policy state, or an explicit withheld result |
| 5. Analyze transport variants | Same `/analyze`, JSON or SSE via request headers | Same semantic request; SSE uses `Accept: text/event-stream` | Orchestrator privately consumes upstream writing output and forwards the authoritative final result; no separate approval endpoint |
| 6. Preview | In-browser adapters: `readPublicationFields()` → `readResumeRewrite()` / `ResumeBuilder` result adapter | Final analysis response from hop 1 | Approved markdown shown to user; veto/null is authoritative and must not fall back to an older draft |
| 7. Download request | Browser `POST /api/ml/imagination/export` | Approved markdown, exact canonical render fields, signed policy state and `publication_receipt`; session credentials | JSON file response or error. Both PDF and DOCX buttons currently call this same endpoint |
| 8. Export proxy | Imaginator `POST /resume/export` | Body forwarded by orchestrator, `X-API-Key`, `X-Cogito-Owner-Id` from authenticated session | Receipt/policy validation followed by rendering; no new Researcher, reviewer, taxonomy or job-search call |
| 9. Build files | `build_resume_from_payload_base64()` (in-process, not another HTTP endpoint) | The same verified render object used by receipt verification | LaTeX → `pdflatex` PDF and `python-docx` DOCX, returned as `{pdf_base64, docx_base64, filename}` |
| 10. Save | `triggerDownload()` in browser (no additional service endpoint) | Base64 bytes and server-chosen filename | PDF/DOCX Blob, object URL and download; bytes must exist before reporting success |

Also test the existing direct JSON analyze proxy `POST /api/ml/imagination/analyze` → Imaginator `POST /analyze` if it remains supported. Do not confuse the browser's `/api/analysis/multi-file` with an Imaginator-local multipart handler: endpoint names do not prove route reachability. The inspected browser uses the orchestrator route even for a single selected file. Job-search calls run elsewhere in analysis but are not export authorizers; no suspended extractor service is required by this feature.

The export path's existing 300-second timeout comments mention O*NET/LLM calls, but the traced handler verifies and renders rather than rerunning writing. Correct documentation/comments when touching that path; do not add those calls to make the comment true.

### D.2 Files and current behavior to change deliberately

| File / function | Finding | Required action |
|---|---|---|
| `imaginator_new_integration.py`: skill aggregation | Code can merge Researcher `implied_skills` and `domain_vocab` into aggregate skills | Separate candidate evidence, proposals, reviewed skills and occupational context; verify actual values reaching that branch |
| `approved_resume.export_payload()` | Parses approved markdown into explicit career/skills sections; refuses ambiguous career layout | Keep this as the sole career-content projection after correction, not a second independent generation step |
| `app.py`: `_issue_publication_receipt()` | Rebuilds canonical payload from final markdown, adds only supplied approved identity fields, then signs | Add validated policy/analysis binding explicitly before returning approval; do not append skills after signing |
| `publication_gate.canonical_render_content()` | Hashes `asdict(ResumePayload)` from the renderer parser; ignores metadata not represented in that dataclass | Bind inference metadata separately; adding arbitrary keys to canonical payload does not currently sign them |
| `publication_gate.issue_receipt()` / `verify_receipt()` | Current v4 binds exact markdown, renderer-normalized object, owner and expiry; only v4 is supported | Introduce an explicit versioned binding for the new policy state, with no downgrade path for inference-enabled output |
| `src/services/publication.ts` | Enumerates allowed publication fields and known response envelopes; null/veto takes priority | Add the policy state fields to deliberate transport, retaining null/veto semantics |
| `resumeExportService.buildExportPayload()` | Canonical object wins and is sent unchanged; older fallbacks can consume aggregate skills or reparse markdown | Require canonical approval on the new-policy export route; keep legacy compatibility from gaining inference authority |
| `backend/server.cjs`: analysis result publication-field copies | Copies a fixed field list through response envelopes | Preserve new validated fields in JSON and SSE branches; do not drop mode/policy at final transport |
| `backend/server.cjs`: export proxy | Requires session owner and receipt, forwards session-derived owner header, passes upstream status | Preserve this trust boundary; body owner/user IDs cannot select approval identity |
| `resume_builder/build_resume_from_export.py` | `parse_resume_payload()` follows nested `data` precedence; skills can fall back to aggregate aliases; builders share parsed payload | Canonical route must not let missing/empty skills trigger mixed raw aliases; verifier and builder must consume the identical effective object |

The inspected canonical export path already bypasses several old fallbacks. Therefore the aggregate leak is a path to test and contain, not proof that every current approved download leaks. Demonstrate reachability with unique sentinel values and actual receipt verification before reporting a delivered vulnerability as fixed.

### D.3 Resolve skills once, before approval

Define one request-local resolved skill state from source-backed skills plus Phase C's surviving decisions:

- **Conservative:** supported supplied candidate skills/faithful normalizations only. No bridge permission from cache, vocabulary, profile aliases or a stale Imaginative response.
- **Imaginative:** supported supplied skills plus accepted/narrowed neutral skill labels actually retained in final reviewed text. A proposal that was never used or was removed does not get appended at export.
- Retain each inferred label's origin and binding in metadata; never relabel it candidate evidence. Dedup presentation labels conservatively without merging different evidence owners or deleting original source facts.
- Keep target vocabulary, job requirements, research metrics, rejected labels and raw `aggregate_skills` outside delivered-skill authority. A generic aggregate collection may remain for advisory consumers only if its meaning is explicit; do not reuse it as proof of hard qualifications.
- Reconcile resolved state with the final reviewed markdown. A narrowed label must be the one in preview, canonical `skills` and final bridge decisions. A bridge used only in capability language is bound there; do not force an additional Skills entry merely to make arrays match.
- Validate that supported explicit skill facts remain represented under existing source-retention rules. If correction removed a required fact, finish repair/review before signing; do not silently append the fact to a signed object later.

Any preapproval deterministic skill-section assembly must occur before the final review that authorizes the delivered document. After review, parsing may project approved text to structured fields but must not create another factual claim. Treat explicit empty arrays as authoritative: empty canonical Skills is not a request to use stale aggregate/profile skills.

**Tests D-01–D-05:** both-mode paired fixtures; source skill retained when also proposed as a bridge; narrowed label replaces stronger label everywhere; removed/unused proposal never appears at export; empty canonical skills plus poisoned aggregate alias cannot repopulate Skills. Include source-backed term absent from the original Skills heading but validly described in a source activity, and two sources with duplicate display labels but distinct provenance.

### D.4 Canonical approval and policy binding

Use the existing HMAC receipt machinery, dedicated signing secret and final canonical career object; no new database table or separate signature service is required. The signed approval must commit to:

1. Exact final markdown fingerprint (current behavior).
2. Complete effective renderer object fingerprint, including any approved header/contact fields (current behavior).
3. A deterministic fingerprint of new policy state: effective mode, policy version, server analysis identity, candidate-input version, and surviving bridge IDs with permitted wording and source/use bindings.
4. Existing authenticated owner, issue time, expiry and approved verdict.

**Proposed compatibility decision:** add a new receipt wire version for approvals carrying inference policy. Keep v4 verification only for unmodified legacy source-only approvals if compatibility is required; a v4 token must not grant bridge privileges. Never accept an unsigned mode/default as a substitute for the new binding. Implementers must update issuance and verification together and specify the exact supported-version set; this spec does not claim that v5 already exists.

Canonicalize policy state deterministically on the server: fixed keys/types, explicit empty surviving-bridge list, stable record ordering and exact approved label text. Do not hash private source documents into a response as plaintext; bind their version/fingerprint and references. Transport only minimal metadata required for validation. A response-side policy envelope is untrusted until it matches the signed fingerprint; the browser is a courier, not an authority.

At export, validate the signed policy state and renderer content **before** invoking either renderer. Reject changed/missing mode, policy, input/analysis identity, bridge IDs, labels or source/use bindings, even if they render the same visual text. The renderer does not need to semantically review bridges again: it renders already approved content. Expired/unsupported approval requires reanalysis through existing flow, not a silent receipt refresh over arbitrary client content.

Keep receipt version (wire format) distinct from `imaginative-v1` (permission policy). A signed receipt attests approval under the recorded policy, not objective proof of skill. No change to numerical verdict semantics is required to authorize nonnumeric skill interpretation.

**Tests D-06–D-10:** real issue/verify round trip for both modes with explicit policy state; one-field mutation matrix covering every bound field; reorder policy records according to specified canonicalization; remove metadata while retaining receipt; submit inference-enabled state with legacy v4; unsupported version, expired token, wrong owner, unavailable signing key. Instrument renderers and assert zero render calls for every refused case.

### D.5 Response envelopes, preview and download payload

Extend the explicit publication-field transport in `models.py`/analysis response, orchestrator publication copies, frontend `PublicationFields` and export builder. Choose one policy-state envelope name consistently across these boundaries; do not sprinkle different aliases into each layer. Both JSON and SSE final results carry the same authoritative approval set.

The browser previews `final_written_section_markdown` from the final accepted response. Preserve its exact string for receipt use even if Markdown rendering produces a different DOM. Display formatting, category rendering and UI skill chips must not regenerate submitted career content from `aggregate_skills`, `processed_skills` or a stale draft. An explicit `null` markdown/canonical approval or veto in a supported envelope must suppress display/export fallback.

For the new policy path, `buildExportPayload()` must send the exact canonical object plus matching markdown, receipt and bound policy state. Preserve original canonical values; do not add default name “Resume”, current form job title, profile contacts, a location, new skill categories or client-reparsed bullets. A user editing the form after analysis must not silently alter the approved headline, mode or content. Such changes require the normal review/approval flow, not mutation of the old result.

The proxy owns authentication headers. Continue deriving `X-Cogito-Owner-Id` from the session, never body hints. Never echo signing secrets, service credentials, source quotes or full private evidence in client errors. Preserve upstream refusal status/reason so the UI can show a review/reanalysis message. A 200 analysis response with `needs_review` is not an exportable success.

**Tests D-11–D-15:** snapshot outbound browser payload and compare with approved object; nested response envelopes and top-level null/veto precedence; same final fields for SSE/JSON; editing job title or inference toggle after analysis does not change old approval; injected body owner cannot alter trusted header; anonymous/wrong-session export fails. Blocked/no-receipt UI sends zero download requests and does not show stale draft content.

### D.6 Renderer invariants and file-content parity

Keep the existing shared `ResumePayload` parsing for PDF and DOCX. Verification and rendering must follow identical nested-`data`/alias precedence; preferably resolve one effective canonical object at the export boundary and pass that same object through verification and build. Unknown or conflicting alternate envelopes must not let the verifier inspect one object while the renderer selects another.

`build_resume_from_payload_base64()` currently uses a private temporary directory, creates both PDF and DOCX, reads them, and removes artifacts on exit. PDF generation renders `main_template.tex` and runs `pdflatex` twice; DOCX is built independently with `python-docx` from the same parsed content. Do not convert one format into the other or rerun an LLM for “better formatting.”

Preserve supplied Experience, Education, Skills, Projects, Certifications and approved summary/publications. Empty promised sections may remain in approved markdown while file templates omit empty headings; define parity as preserved approved substantive content and ownership, not identical heading counts or pagination. No filler entries may be created. Both formats must contain the reviewed inferred label in Imaginative and exclude it in Conservative unless separately source-backed.

Text sanitization repairs punctuation/control characters and escapes LaTeX; do not change skill meaning, strengthen proficiency or drop meaningful technical characters. Compare extracted text using a documented allowance for layout/punctuation normalization, not an aggressive alphanumeric squash that can hide omitted qualifiers or merge distinct skills. Include `C++`, `C#`, slash compounds, accents, ampersands, percentages and Unicode dashes. Avoid claiming byte-identical PDF/DOCX output across runs; timestamps/layout engines can differ.

Filename/headline authority stays with the approved payload; do not supply an outer job-title override unbound by the receipt. Identical safe filenames across requests must remain isolated. A failure in either builder must not return a success-shaped response with missing file bytes; keep temporary-file cleanup on compilation errors, DOCX save errors and cancellation where supported.

**Tests D-16–D-20:** render real files with local LaTeX and `python-docx`, then extract/read back substantive skills, source facts and employer ownership; check both modes and a narrowed bridge. Test punctuation/Unicode and no template default facts; nested-data/outer-field mismatch; same-title concurrent exports with distinguishable private content; LaTeX failure, DOCX failure and empty bytes return an error with cleanup. Clearly mark missing local tools as blocked verification, not a renderer pass using mocked bytes.

### D.7 Download endpoint semantics and edge cases

Preserve the current response `{pdf_base64, docx_base64, filename}` and correct MIME types. A PDF-only button currently triggers generation of both files but saves only PDF; DOCX-only does the same for DOCX. Do not claim an optimization or introduce a new format-specific endpoint unless separately scoped. Export performs zero model/taxonomy calls and must not debit another resume-analysis charge merely for downloading approved bytes; verify the actual credit middleware behavior before asserting this property.

Validate absent/invalid base64, missing requested bytes and malformed response; show failure rather than a phantom completed download. Object URLs and temporary DOM anchors must be released. Browser download blocking is a delivery failure distinct from server approval; record whether bytes were returned and whether the browser actually saved files.

Repeated downloads with a valid unexpired receipt may re-render the same approved content under existing behavior; do not add one-time-token semantics or a new storage cache in this phase. Repeated clicks, session expiry and network failures must not start a new analysis automatically or switch modes to make export succeed. UI retries may resubmit the same approval, never modify its facts.

**Tests D-21–D-24:** same endpoint used by both buttons; requested bytes validated and MIME/filename correct; no Researcher/reviewer/O*NET/job-search invocation during export; verify download does not invoke the analysis debit path. Test repeated click/network error/session expiry without auto-analysis, Blob URL cleanup, and zero-byte/malformed base64 handling. Separately capture successful browser save in approved end-to-end validation; HTTP 200 alone is insufficient.

### D.8 Implementation order and acceptance evidence

1. Add paired canonical-skills fixtures and sentinel tests proving the actual fallback paths. Preserve explicit-source compatibility while removing raw aggregate authority from inference-enabled exports.
2. Build final mode-aware content before review/approval; connect Phase C's surviving decisions to server canonical projection without postapproval skill injection.
3. Add versioned policy-state binding to receipt issuance/verification, then tamper tests with renderer-not-called assertions.
4. Update orchestrator response copies and frontend publication/export transport together; cover SSE and JSON with the same fixtures.
5. Enforce identical effective payload selection at verifier/renderer, test real PDF/DOCX content and private artifact isolation.
6. Run route-level integration: browser request builder → orchestrator proxy with fake upstream transport → real Imaginator analyze/correction/export contracts using scripted providers and actual receipt functions. A separate real-render test must read file contents; no single mocked test establishes the whole delivery.

Suggested modules: `test_bridge_export_binding.py`, `test_bridge_render_content.py`, frontend `bridgeExportPayload.test.ts`, and backend `bridgeExportProxy.test.js`; extend existing receipt/export suites rather than replacing negative controls. Use the repository's safe Jest runner for frontend/backend suites; serialize resource-heavy real-render tests. No paid provider run or deployment is authorized by this section.

**Exit criteria:** both modes survive the exact listed route; preview and files reflect the same reviewed content; only permitted skill labels are delivered; policy/bridge state cannot be changed after approval; mixed aggregate and legacy research cannot bypass canonical approval; owner/privacy/expiry/generation safeguards remain; export uses no generation/review model call; and both file contents are verified independently. Record integrated application/frontend/backend SHAs, commands and real exit codes. Distinguish offline protocol/render evidence from separately authorized live customer delivery.

## 7. Phase E — caches and isolation

Inspect each actual analysis/research cache rather than assuming one exists. Include candidate/input identity, job context, generation identity, effective mode and policy version where applicable. Bind cached bridge records to their source version; reject or recompute incompatible records. Legacy untyped entries cannot acquire bridge privileges: recompute or use source-only handling. Publication decisions and completion status remain request-local, not authorized by global telemetry.

**Suggested tests:** same input in opposite modes produces separate cache identities; stale policy and changed source invalidate; legacy cache cannot grant inference privilege; precomputed research cannot cross mode or candidate boundaries. Run parallel simulated requests with distinct owners, modes and failure outcomes and assert no result, receipt or authorization crosses requests.

## 8. Integration acceptance and verification

Run focused offline tests after each phase, then integrate against the agreed application and frontend/backend SHAs. Preserve the spec-completion agent's required-generation, fallback and receipt safeguards. Acceptance requires both mode paths, verified reference and bridge-level review, prohibited-fact rejection, source retention, cache isolation, preview/PDF/DOCX parity, tamper refusal, privacy and ownership. Record call counts and any remaining non-bridging calls. Use existing isolated harness/render gates only after verifying their paths and prerequisites in the chosen checkout; preserve real exit codes and avoid concurrent broad test suites.

Contract and scripted-review tests do not establish semantic accuracy. Separately evaluate inference quality and overreach with held-out examples and human review; live model or paid customer-path validation requires separate authorization. Record failures and limitations rather than claiming completion from a partial suite.

## 9. Delivery, risk and rollout

- **Cost/latency:** Conservative must avoid the bridging-only call. Measure actual calls and token usage; do not invent per-request savings. Imaginative may use bounded existing context; no new mandatory external dependency or unrestricted browsing.
- **Safety:** Model output and taxonomy are not proof. Reference checks and semantic review have limits; preserve existing independent publication gates and do not infer a general exemption from a skill label.
- **Deployment:** No deployment or paid run is authorized by this specification. Before an external write, verify the Render Cogito Alpha workspace ID, service, linked branch, auto-deploy settings and rollback revision; obtain approval for the specific action. The prior docs push was not accompanied by auto-deploy verification, so deployment status was not established by that push.
- **Rollback:** retain the verified previous release and configuration; revert or restore without erasing test evidence. Report integrated SHAs, commands, exit codes, unmet acceptance criteria and dirty files in the handoff.

## 10. Open decisions for implementers

1. Which active route and baseline owns source-aware correction and signed canonical receipts? Integrate rather than reimplement it on the older feature baseline.
2. How are immutable multi-document source spans and analysis identity represented in the existing system? Define IDs and trust boundary from actual code.
3. Can the current reviewer accept, narrow and remove phrase-scoped bridges while preserving source facts, without a new call? Demonstrate with an integration test; if not, identify the gap before claiming eligibility.
4. Which existing taxonomy context is available early enough and correctly licensed/versioned? Missing context is an explicitly labeled fallback, not a blocker.
5. Which old style controls must remain visible or compatible? Resolve separately from the two-mode inference contract.