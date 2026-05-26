---
name: optimize-tbm
description: Optimize glrmask/CFA time-between-mask performance; use when investigating slow mask, commit, or TBM timings, especially CFA report-slow-steps results or profile_step output.
---

# Optimize TBM

Temporary precedence note: for JSON-schema importer regressions after the modular
importer rewrite, use `$json-schema-importer-regression` first. This generic TBM
workflow is temporarily secondary for those cases; still apply its measurement
integrity rules when the temporary skill asks for TBM evidence.

Use only stabilized timings for decisions: compare per-step values after cross-run per-step minimum stabilization, not raw single-pass sweep spikes.

Full-report invariant:
- Never run CFA full-report, chunked full-report, report-slow-steps intended for
  report plots, or broad sweep artifacts in `glrmask_native`-only mode.
- Do not set `FORCE_DISABLE_LLGUIDANCE_NATIVE` for report or sweep work. That
  mode is only allowed for a narrow one-off diagnostic when the human explicitly
  asks for glrmask-only data in that turn.
- Standard CFA report plots require at least two frameworks. If a run has only
  `glrmask_native`, stop and rerun with the normal framework set instead of
  inventing replacement plots or reporting the artifact as a full report.
- If llguidance is slow, reduce sample size, chunk size, timing runs, or run a
  focused two-framework subset. Do not remove llguidance from the report.
- For report/sweep commands, use the Makefile defaults. Only set `SAMPLE_SIZE`
  and `OUT_DIR` when the human asks for a particular size or destination. Do not
  override frameworks, timing runs, min-run thresholds, build runs, build
  timeouts, discrepancy budgets, chunk size, seeds, or other knobs unless there
  is a specific documented reason in the current task.

Do not optimize by caching across timing runs or repeated invocations of the same
step/example. Cross-run memoization, warmed materialized-mask caches, and
constraint-level caches keyed by a completed mask/dense set are invalid for TBM
work because they measure benchmark reuse instead of single-call mask generation
cost. Caches are only acceptable when they represent normal precomputed
constraint artifacts built before timing, or per-state/generation caches that
serve the real API semantics without relying on repeated benchmark runs.

Do not improve one TBM/build sample by disabling, skipping, threshold-gating, or
making opt-in an existing correctness, compression, table, parser-DWA,
terminal-DWA, or runtime optimization. Those optimizations often exist for
important cases outside the immediate sample. When such an optimization is hot,
preserve its semantic effect and optimize its implementation, data structures,
sharing, memoization, or downstream representation. Disabling or narrowing it
requires explicit human approval and broader tradeoff evidence.

Recognizer-only principle: glrmask's parser/table/runtime optimizations need to
preserve mask/commit recognition behavior, not parse-tree or parse-structure
shape. When optimizing TBM or build/runtime interactions, aggressively look for
unused parse-structural distinctions that can be represented symbolically,
quotiented, shared, or discarded without changing which token sequences are
accepted or rejected. Do not retain extra parser states, delayed continuations,
goto distinctions, or action structure merely because they preserve an unused
parse shape.

Ambiguity-classification requirement:
- Before naming an ambiguity or choosing a fix, inspect both the emitted GLRM
  grammar and the final parser action/profile shape. Do not stop at a
  grammar-level story such as "shift/reduce" if unit-reduction forwarding,
  stack-effect lowering, parser-DWA construction, or runtime profiling has
  compiled the ambiguity into another representation.
- For any ambiguous step, identify the exact emitted GLRM fragment, the concrete
  pre/post parser stacks, and the final table/runtime action kind. In
  particular, check whether the ambiguity is encoded as `Action::Split`,
  `Action::StackShifts(Vec<StackShift>)`, or
  `Action::GuardedStackShifts(Vec<GuardedStackShift>)`.
- Treat multi-entry `StackShifts` and `GuardedStackShifts` as possible compiled
  ambiguity carriers, not merely deterministic fast paths. A single action can
  fan out one concrete stack into multiple recognized continuations after unit
  reductions or other table rewrites have removed the visible reduce operation.
- When the profile shows `n_nondet_reduce_ops == 0`, do not infer there was no
  grammar ambiguity. First check for stack-effect fanout, frontier-state fanout,
  and prior advances that materialized the ambiguity into multiple GSS paths.
- The master/lead must do this classification directly for hard TBM cases
  before delegating implementation. Workers can gather logs, but the lead owns
  the exact diagnosis, action representation, and proof target.

MRE construction for ambiguity/TBM cases:
- Prefer a schema MRE obtained from the exact CFA schema that produced the slow
  step. Reduce by deleting/subsetting original schema material first: remove
  unrelated root properties, anyOf branches, object properties, required entries,
  enum entries, descriptions, bounds, and nested keys only when the same live
  prefix/token oracle still reproduces. Do not invent a fresh ambiguous schema
  or splice properties between unrelated original branches.
- Keep the oracle anchored to the original CFA problem/example/step: the same
  prefix tail, token bytes/id class, parser stack split, action kind, and
  nondeterministic wave counters. A smaller schema that merely has some
  ambiguity is not a valid TBM MRE if it changes the causal ambiguity class.
- In the Rust MRE comment, state which CFA problem/example/step it came from and
  what was deleted. If the minimized schema contains surprising survivors or
  misspellings from the source schema, keep them unchanged and call out that
  they came from the original rather than being invented.

Requirements:
- `commit` max: below `10us`; `10us` is the hard ceiling.
- `mask` max: below `20us`.
- `TBM` max: below `12us`; `12us` is the hard ceiling.

Workflow:
Notes requirement:
- For TBM optimization work, keep a running dated note in `/Users/isaacbreen/Projects2/gcg-paper/notes/` as work proceeds.
- Record slow-step sources, profile commands, env vars, before/after stabilized timings, failed experiments, and keep/revert decisions.
- Check the current note before retrying an approach.
- Do not report success, pause the investigation, or commit TBM/runtime work until the dated note has been updated for the completed chunk. The note must include artifact paths and the keep/revert decision, not just a final summary reconstructed later.

## Pre-Commit Evidence Checklist
Before committing TBM or runtime-latency work, verify the commit message and body name the exact problem/schema ID(s) and examples improved, the metric (`mask`, `commit`, `TBM`, or build time), the threshold used for accept/reject, and before/after stabilized values for those named cases when known. Include profile/sweep/log artifact paths when available. If before/after is unavailable, state exactly what was run, what was not run, why the remaining measurement was skipped, and the motivating evidence without implying a measured win. Mention any known tradeoff, especially when a runtime optimization changes compile/build behavior.

1. Identify slow steps with CFA `report-slow-steps` on a stabilized artifact.
2. Profile the exact step with `scripts.profile_step`, matching active `GLRMASK_*` env vars from the report.
3. Account for all measured time before optimizing; if profile buckets do not explain the cost, add narrow profiled-entrypoint instrumentation rather than broad debug paths.
4. Separate assertion/debug overhead from production cost, especially `GLRMASK_ASSERT_COMMIT_TOKEN_MASK_EQUIVALENCE`.
5. Validate improvements with the same stabilized report path and state the before/after max timings.
