# CogitoMetric — Official Implementation Plan: Two-Mode Skill Bridging

Version: 1.0  
Decision date: 2026-11-04  
Status: Product policy adopted for implementation; implementation and release acceptance pending  
Target application repository: aggressor-FZX/Imagination_Resume_Module  
Target branch: resumeDraft_fix (user-designated new branch from master)  
Deployment scope: Render — Cogito Alpha workspace (user-designated; workspace ID must be verified before changes)

## 1. Decision and purpose

Ship a working build with two settings only:

1. **Imaginative (default):** automatically infer plausible transferable skills from candidate background, supported where available by O*NET or other occupational resources and model reasoning.
2. **Conservative:** skip skill-bridging inference and its model cost; use supplied facts, faithful rewriting, and straightforward normalization.

No Balanced mode. No mandatory questions, skill-approval interface, or separate admission-model call in this release. Optional future review UI is explicitly deferred and must not block delivery.

The goal is to retain the Researcher's useful reasoning rather than discard all implied skills. A skill bridge receives a narrowly scoped exception to direct-source requirements; it does not receive a general factual-gate exemption.

## 2. Policy boundary

### What Imaginative permits

- Skill names absent from the original wording when plausibly connected to candidate activities or background.
- Broader transferable-skill framing, supported by taxonomy relationships and contextual reasoning.
- Automatically including eligible bridges in Skills and capability-oriented summary language.
- Relabelling an existing experience activity without adding another activity, outcome, or responsibility.

The connection need not be proven by literal source wording. It must have a real, accurately referenced candidate-background anchor and a defensible semantic rationale. Occupation-title context can inform proposals, but target-job requirements alone cannot authorize them. When an inference becomes too specialized, use the broader plausible skill instead.

### What neither mode permits without separate support

- New software, named tools, methods, credentials, licenses, security clearances, degrees, or employers.
- Invented dates, years of experience, numerical outcomes, proficiency levels, leadership, or team size.
- New duties, projects, achievements, or deliverables presented as things the candidate performed.
- Reassigning facts between employers or projects.
- Treating a job requirement as a candidate achievement.

Example: comparing weekly sales figures may support trend analysis and analytical problem-solving. It does not authorize Python, development of a forecasting model, or a 20% revenue improvement.

Example: sonar work may support signal-analysis terminology and exploration of related time-series skills. It does not automatically establish machine-learning model development or deployment experience.

### Gate exception

For eligible skill phrases in Imaginative mode, replace the direct-source test with: "This is a plausible transferable-skill interpretation connected to the supplied candidate background and permitted by the selected policy."

All other checks remain: reference validity, attribution, unsupported concrete facts, privacy, owner authorization, stage completion, schema, required sections, canonical payload integrity, and receipt/export binding.

The exception attaches to an identified skill phrase, not an entire sentence. A sentence mixing a permitted bridge with an unsupported achievement still fails for the achievement.

## 3. Settings contract

Canonical API values:

```json
{"skill_inference_mode": "imaginative"}
```

```json
{"skill_inference_mode": "conservative"}
```

- Missing value defaults to imaginative for this new policy.
- Unknown values return a validation error; never silently choose a mode.
- Store the effective mode and policy version with the analysis and canonical export state.
- Frontend, orchestrator, precomputed research, writer, editor, correction, and export must agree on the effective mode.
- Conservative skips the lateral-bridge model call. Do not claim elimination of all pipeline inference cost.
- Conservative may retain non-bridging analysis needed elsewhere, but must not reuse previously generated bridges.
- Mode is a permission policy, not merely a temperature setting.
- If old controls affect prose style, preserve that concern separately and document migration of old values. Do not guess a mapping for existing settings without inspecting their consumers.

Suggested concise UI: "Imaginative — adds inferred transferable skills. Review your resume before use." Conservative: "Uses only supplied skills and faithful rewrites."

## 4. Minimal provenance contract

Use the smallest typed record that supports routing and auditing:

```json
{
  "id": "bridge-004",
  "skill": "trend analysis",
  "origin": "researcher_inference",
  "experience_id": "exp-02",
  "evidence_quote": "Compared weekly sales figures to identify trends...",
  "rationale": "Transferable analytical skill suggested by the described activity.",
  "taxonomy_refs": [],
  "policy_version": "imaginative-v1"
}
```

Add document/fact references when the current evidence model provides them. Do not fabricate IDs or assume fields already exist.

- Validate source references and quotations against immutable candidate input.
- Only the controlled server Researcher path, or validated trusted precomputed output, may create eligible bridge records.
- A client-supplied origin tag never grants a gate exemption.
- Researcher origin remains researcher_inference after eligibility; do not relabel it candidate_evidence.
- Keep supplied skills, eligible bridges, target vocabulary, and job requirements distinct.
- Preserve a compact decision/reason when a proposal is narrowed or excluded; do not introduce a new review framework.
- Existing reviewer machinery evaluates semantic plausibility under this policy; no additional model call is required by this plan.
- Research failure returns no invented default qualifications. Resume generation may continue from source facts subject to existing completion rules.

## 5. Grounding strategy

Reuse available O*NET context before introducing infrastructure. Where appropriate, ESCO can supplement terminology and transferable-skill relationships. Existing local taxonomy assets may be reused only after verifying schema, licensing, version, and packaging cost.

Inject bounded occupation/activity/skill context into the Researcher. Do not add unrestricted web browsing or scraping.

Taxonomies support skill meaning and relevance; they do not prove individual competence. Prefer candidate-activity connections over simply copying the target occupation's skill list.

- Keep references/version when available.
- Grounding unavailable: Imaginative may continue using model reasoning, retaining empty taxonomy_refs and an accurate grounding status. Never fabricate citations.
- Conservative: no new taxonomy fetch solely for bridging.
- Avoid new large dependencies or unbounded payloads; respect current instance memory limits and existing resume caps.
- This plan does not require a new live taxonomy dependency or a paid experiment to implement locally.

## 6. Required fix #1 — export provenance leak

Reported path to confirm: implied_skills/domain_vocab merge into aggregate_skills, which an export fallback may treat as candidate Skills.

Replace accidental export behavior with explicit mode-aware construction:

```
Conservative: supplied candidate skills
Imaginative: supplied candidate skills + eligible typed bridges
```

- Raw mixed aggregate_skills must not be the fallback authority for delivered skills.
- JD-only domain_vocab and implied_metrics never enter candidate skills or achievements by default.
- Preserve legitimate explicit skills through a deliberate compatibility path.
- Use the same resolved skill collection in preview and the server-approved canonical export payload.
- Build and bind the final content before issuing a receipt; never append skills after approval.
- Preserve all five output sections: Experience, Education, Skills, Projects, Certifications.

## 7. Required fix #2 — legacy editor research leak

Reported path to confirm: imaginator_flow.run_final_editor_async receives unrestricted research content.

Replace the raw research block with separated inputs:

1. Candidate facts: ordinary factual policy.
2. Eligible skill bridges: scoped Imaginative exception.
3. Target-role vocabulary: relevance only, not candidate facts.

Unapproved raw suggestions, JD metrics, and occupational duties must not become claims simply by reaching the editor. Apply consistent handling to every route that can deliver a resume, including active stages and still-used legacy multipart paths. Do not modify inactive code merely for symmetry.

## 8. Writer, editor, and correction behavior

Researcher proposes typed bridges. Existing validation/review checks references and applies the policy. Drafter and editors may use eligible bridges without requiring literal source wording. Correction must retain valid bridges instead of stripping them as absent-from-source terms.

Replace sparse-input instructions authorizing "standard industry tasks" or guessing "how work was likely done" with:

> In Imaginative mode, infer plausible transferable skills from the supplied candidate background. Use eligible skill labels to describe existing activities. Do not invent activities, tools, credentials, metrics, achievements, responsibilities, or proficiency. In Conservative mode, do not generate skill bridges.

Do not weaken the R-5 acceptance test to hide invented-task instructions. Update its expectation to distinguish permitted skill inference from fabricated duties.

A skill bridge does not establish seniority, years of experience, exact-tool mastery, or certification. Matching may consider it as inferred transferability, not proof of satisfying a hard qualification.

## 9. Cache and request-state requirements

- Cache identity includes effective mode and policy version, plus existing candidate/job identity and relevant generation identity.
- Do not serve Imaginative output to Conservative requests.
- Cached bridge records retain their source references and content binding.
- Old untyped cache entries cannot gain bridge privileges automatically; recompute or apply the conservative source-only path.
- Keep publication-relevant state request-local. Global telemetry must not authorize another request's publication.
- Preserve the main spec agent's completion/fallback safeguards. Skill inference is not an exception to failed generation.

## 10. Implementation order and worktree ownership

### Researcher/provenance agent

Work in its separate worktree on resumeDraft_fix. Before edits, record branch, HEAD, application baseline, and integration dependencies. Never switch another agent's checkout or force-move shared branches.

1. Inspect current mode consumers and reported leak paths.
2. Add two-mode contract and typed bridge records.
3. Route eligible bridges into drafting/correction consistently.
4. Fix export collection and legacy editor injection using the same contract.
5. Replace sparse-input invented-task wording.
6. Add focused offline tests and provide local commits/diff.

### Spec-completion agent

Own acceptance tests and integration validation, not competing prompt edits. Continue request-local admission/cache work independently. Supply the real R-5 prompt-boundary test. Integrate the agreed Researcher commits in its own worktree and rerun isolated gates.

### Integration boundary

Review overlapping files before cherry-picking. Record tested application, frontend/backend, and bench SHAs. Preserve unrelated work and historical result files. No new model leaderboard or diagnostic-only planning loop.

## 11. Acceptance matrix

1. Default selects Imaginative; invalid values fail validation.
2. Conservative produces no bridges and invokes no bridging model call.
3. A useful nonliteral skill bridge survives research, drafting, correction, preview, and export in Imaginative.
4. The same bridge is absent in Conservative while source activities remain.
5. A bridge may add a skill label but not an unsupported tool, metric, duty, credential, or proficiency.
6. Wrong-employer/project attribution remains rejected.
7. JD-only vocabulary and metrics do not leak through aggregate_skills.
8. Raw legacy research cannot bypass skill routing.
9. Client-created provenance labels cannot bypass checks.
10. Empty/error research introduces no generic default skills.
11. Mode/policy caches do not cross-contaminate, including precomputed and legacy entries.
12. Actual sparse and normal prompts allow the intended policy and prohibit invented tasks.
13. Registered generation completes; failed required generation cannot receive a receipt.
14. PDF/DOCX contain the intended mode-aware skills and match the canonical approved payload.
15. Existing ownership, privacy, tampering, required-section, and source-retention tests remain intact.

Record inference quality and overreach separately. Passing contract tests does not prove every semantic inference true. Holdout/human review and separately authorized behavioral validation remain distinct from implementation acceptance.

Existing isolated gates, adjusted only if the verified environment requires it:

```bash
unshare -Urn bash -c 'ip link set lo up && ../venv/bin/python verify_offline.py harness'
unshare -Urn bash -c 'ip link set lo up && ../venv/bin/python verify_offline.py render'
```

Retain real exit codes; when piping through tail/tee, enable pipefail or capture the Python process exit explicitly. Report results rather than inferring success from filtered output.

## 12. Render / GitHub release controls

Target Render workspace is Cogito Alpha, not another workspace. Resolve and record its actual workspace ID and the exact affected service IDs before any modification.

Target branch is resumeDraft_fix from master. Verify the actual local/remote branch, base SHA, and existing commits; do not recreate, overwrite, or reset an existing branch.

Connector verification during plan preparation:

- GitHub lookup for refs/heads/resumeDraft_fix in aggressor-FZX/Imagination_Resume_Module returned 404. This does not rule out an unpushed local branch. No remote branch was created.
- Render workspace lookup was blocked by the connector approval gate. Cogito Alpha is the user-designated target, not independently verified in this session.
- No GitHub or Render state was modified to produce this document.

Before pushing, inspect Render auto-deploy configuration: a push may itself deploy. Resolve the exact target and obtain confirmation for external writes/deployments through the connector workflow. Do not assume this document authorizes a paid run, service change, branch rewrite, or production deployment.

Deployment sequence after local acceptance:

1. Review code/diff and integration SHAs.
2. Confirm workspace, service, source branch, auto-deploy behavior, and rollback revision.
3. Obtain approval for the resolved external action.
4. Push/deploy only approved content.
5. Validate both mode paths, preview/export parity, and fallback refusal in the approved environment.
6. Keep fixed-model smoke validation separate from website-to-orchestrator-to-Imaginator customer-path acceptance; obtain any required paid allowance separately.

Rollback: retain the previous known-good application revision and configuration. Revert the feature change or restore the verified release if admission/export behavior regresses. Do not delete evidence or rewrite historical test results.

## 13. Out of scope

- Approval/questionnaire UI.
- Balanced mode or temperature-based mode emulation.
- New admission-model call or broad semantic-verification platform.
- Unrestricted web browsing, new large taxonomy infrastructure, model ranking.
- Fabricated metrics or duties in any mode.
- Automatic production deployment or paid testing.

## 14. Definition of done

The two settings operate consistently; useful inferred skills survive intentionally; both reported leaks are closed; raw suggestions cannot masquerade as explicit evidence; concrete factual safeguards remain; isolated gates pass against integrated commits; and deployment evidence is recorded only after separately approved release actions.

This document is the governing implementation policy for this feature. It supersedes earlier three-mode and mandatory approval-UI proposals. It changes only skill-bridging policy, not the remaining SPEC.v2 requirements.
