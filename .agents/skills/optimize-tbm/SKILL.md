---
name: optimize-tbm
description: Optimize glrmask/CFA time-between-mask performance; use when investigating slow mask, commit, or TBM timings, especially CFA report-slow-steps results or profile_step output.
---

# Optimize TBM

Use only stabilized timings for decisions: compare per-step values after cross-run per-step minimum stabilization, not raw single-pass sweep spikes.

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

Requirements:
- `commit` max: below `10us`; `10us` is the hard ceiling.
- `mask` max: below `20us`.
- `TBM` max: below `25us`.

Workflow:
Notes requirement:
- For TBM optimization work, keep a running dated note in `/Users/isaacbreen/Projects2/gcg-paper/notes/` as work proceeds.
- Record slow-step sources, profile commands, env vars, before/after stabilized timings, failed experiments, and keep/revert decisions.
- Check the current note before retrying an approach.

## Pre-Commit Evidence Checklist
Before committing TBM or runtime-latency work, verify the commit message names the exact problem/schema ID(s), the metric (`mask`, `commit`, `TBM`, or build time), and before/after stabilized values when known. If before/after is unavailable, name the profile or sweep artifact and state the motivating evidence without implying a measured win. Mention any known tradeoff, especially when a runtime optimization changes compile/build behavior.

1. Identify slow steps with CFA `report-slow-steps` on a stabilized artifact.
2. Profile the exact step with `scripts.profile_step`, matching active `GLRMASK_*` env vars from the report.
3. Account for all measured time before optimizing; if profile buckets do not explain the cost, add narrow profiled-entrypoint instrumentation rather than broad debug paths.
4. Separate assertion/debug overhead from production cost, especially `GLRMASK_ASSERT_COMMIT_TOKEN_MASK_EQUIVALENCE`.
5. Validate improvements with the same stabilized report path and state the before/after max timings.
