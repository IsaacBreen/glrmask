---
name: optimize-tbm
description: Optimize glrmask/CFA time-between-mask performance; use when investigating slow mask, commit, or TBM timings, especially CFA report-slow-steps results or profile_step output.
---

# Optimize TBM

Use only stabilized timings for decisions: compare per-step values after cross-run per-step minimum stabilization, not raw single-pass sweep spikes.

Requirements:
- `commit` max: below `10us`; `10us` is the hard ceiling.
- `mask` max: below `20us`.
- `TBM` max: below `25us`.

Workflow:
1. Identify slow steps with CFA `report-slow-steps` on a stabilized artifact.
2. Profile the exact step with `scripts.profile_step`, matching active `GLRMASK_*` env vars from the report.
3. Account for all measured time before optimizing; if profile buckets do not explain the cost, add narrow profiled-entrypoint instrumentation rather than broad debug paths.
4. Separate assertion/debug overhead from production cost, especially `GLRMASK_ASSERT_COMMIT_TOKEN_MASK_EQUIVALENCE`.
5. Validate improvements with the same stabilized report path and state the before/after max timings.
