---
name: triage-sweep-run-issues
description: Triage and drive all issues from a CFA/glrmask sweep run, including paired build asymmetry, valid build timeouts, TBM threshold failures, and discrepancies, when given a sweep output directory, merged result, chunks, or run log.
---

# Triage Sweep Run Issues

Use this when the user points to a CFA/glrmask sweep run and asks to "deal with
issues", "fix the run", "triage failures", or similar. Own the issue queue and
priority order; do not make the user keep naming categories.

## Priority Order

1. Paired build asymmetry and paired build-result parity.
2. Build timeouts only for problems both frameworks are intended to build.
3. TBM threshold failures, usually `glrmask_native` TBM `> 12us`.
4. Discrepancies and unhandled adjudication cases.
5. Plot/report regeneration after higher-priority issues are resolved or
   intentionally classified as out of scope.

Do not optimize an issue in a lower priority class while an unclassified higher
priority issue remains.

## Initial Audit

Given a run directory or result file:

- Locate chunks, merged results, readable report, slow-step reports, mismatch
  reports, and logs.
- If the run is chunked and no merged result exists, merge chunks once with
  `scripts.merge_sweep_chunks` and keep the merged artifact path.
- Summarize build statuses per framework: built/built, failed/failed,
  timeout/failed, built/failed, failed/built, timeout/built.
- Extract timeout cases and peer framework outcomes.
- Extract TBM threshold cases using `scripts.extract_slow_tbm_problems` with the
  relevant framework and threshold.
- Extract or adjudicate discrepancies only after build parity and timing
  blockers are classified.

Save audit artifacts under `/tmp/<run-name>_issue_audit_*` and record commands.

## Paired Build Asymmetry

Paired-framework asymmetry outranks timing work.

- If `llguidance_native` fails fast and `glrmask_native` times out, builds
  slowly, or produces TBMs, do not optimize glrmask timing first. Treat it as a
  build-result parity issue.
- If one native framework builds and the other fails, the case is not a valid
  paired timing/discrepancy comparison until the asymmetry is understood.
- Do not hide asymmetry in CFA by skipping a peer build or suppressing a
  successful backend after the other backend fails. Preserve independent backend
  build outcomes in reports; fix parity in backend behavior.
- Before aligning to `llguidance_native`, inspect the local llguidance source
  clone for the actual rejection rule. The repo is typically at
  `/Users/isaacbreen/Projects2/downloads/repos/llguidance-guidance-ai`; for
  JSON-schema `schema too large`, start at `parser/src/json/shared_context.rs`.
- Prefer a glrmask-side fast fail for known unsupported features or comparable
  schema-size/complexity preflight only after confirming the corresponding
  llguidance rule from source. Do not make `llguidance_native` more permissive
  to hide asymmetry.
- Record whether the issue is a true capability gap, an unsupported-feature
  parity bug, a schema-size parity bug, or an intentional glrmask-only
  capability measurement.

For timeout-vs-fail cases such as llguidance `schema too large`, the timeout is
not the primary bug. The priority is glrmask backend behavior that either
rejects quickly with comparable semantics or genuinely builds within budget; CFA
must not erase the mismatch by paired-mode suppression.

## Build Timeouts

Only optimize build time after paired build-result parity is classified.

- If both frameworks are intended to build the problem, profile the glrmask build
  and identify the dominant phase before editing.
- Use capped probes first. Do not burn long runs on a case that the paired peer
  already rejects.
- Prefer importer/schema-lowering fixes for JSON-schema build explosions before
  generic compiler changes, unless evidence shows the generated grammar is not
  the cause.
- Revert patches that improve a local phase but do not improve the end-to-end
  build or worsen another dominant phase.

## TBM Threshold Failures

Use stabilized results for decisions.

- Default hard target: `TBM <= 12us`, `commit <= 10us`, `mask <= 20us`.
- Use `scripts.extract_slow_tbm_problems --threshold-us 12 --framework
  glrmask_native` to get the residual set.
- Rerun threshold cases with `make example-specific` or the established
  override/stabilization target. Do not lower timing knobs unless the user
  explicitly asks for that exact override.
- Use `scripts.report_slow_steps` on the stabilized/override artifact to find
  exact problem/example/step/token sources.
- Profile exact slow steps only for diagnosis. Do not use raw `profile_step`
  totals as final timing evidence.
- For JSON-schema workloads, inspect importer-emitted grammar and table/action
  shape before runtime edits. Runtime fast paths that reduce counters but worsen
  stabilized timings must be reverted.
- Assume the glrmask runtime has already been heavily optimized. For remaining
  JSON-schema TBM issues, prefer identifying ambiguity in the generated grammar,
  tying it to the exact schema/importer construct, and reducing that ambiguity in
  the importer. Treat runtime edits as a last resort after emitted-grammar and
  table/action evidence rule out importer-shape fixes.

## Discrepancies

Discrepancy work starts with local oracle classification.

- Stop broad adjudication once the requested number of unhandled cases is found.
- Always inspect the actual discrepancy tuple before assigning blame:
  schema/problem, example prefix bytes, disputed token bytes, and framework
  votes. Decide from the JSON Schema and generated grammar whether that token
  should be able to come next. Treat framework outputs and any ground-truth
  checker as evidence, not authority; the checker can be wrong.
- `make example-specific PROBLEM=<problem> ...` should produce enough detail to
  see the disputed token bytes and per-framework votes. If it does not, fix the
  CFA sweep/extract/report path before relying on broad classifications.
- For JSON Schema cases, reason directly from the schema location implied by the
  prefix: object key vs property value, required/optional property, string
  constraints (`enum`, `const`, `pattern`, `format`, length), array/object start,
  and allowed whitespace. A classification rule is acceptable only when this
  direct schema+prefix+token reasoning supports it.
- For each representative case, compare `mask`, `commit_token`, and
  `commit_bytes` for the exact prefix/token.
- If all glrmask layers agree with each other, suspect importer/schema semantics
  before mask construction.
- If `commit_bytes` accepts but `mask` or `commit_token` rejects, suspect
  possible-matches, terminal DWA/id-map, parser DWA, or runtime mask expansion.
- Use `make example-specific` per problem to reproduce. Avoid broad reruns unless
  needed to prove a fix.

## Reporting And Closure

Maintain a live issue queue with status:

- `classified`: root class known, no edit yet.
- `fixed`: patch validated and committed.
- `rejected`: experiment reverted with evidence.
- `deferred`: intentionally out of scope with reason.

Do not regenerate final plots until paired build asymmetry, relevant build
timeouts, and TBM threshold cases are fixed or explicitly deferred. When plots
are regenerated, use the merged/override result that includes accepted fixes.

Commit coherent changes promptly:

- Skill/process updates separately.
- CFA workflow fixes separately.
- glrmask code fixes separately.
- Notes updates with the corresponding performance work when they are required
  evidence.
