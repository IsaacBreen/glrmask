---
name: optimize-glrmask-build
description: 'Optimize glrmask compile and build time. Use for GLRMASK_PROFILE_COMPILE, profile_build, build regressions, L2P terminal-DWA construction, id-map and equivalence analysis, partitioning, example-specific investigations, and compile-time profiling in glrmask or constraint-framework-analysis.'
user-invocable: true
---

# Optimize Glrmask Build

## When to Use
- Investigating glrmask compile-time or build-time regressions
- Profiling `GLRMASK_PROFILE_COMPILE` output or `profile_build` runs
- Diagnosing `id_map`, state equivalence, vocab equivalence, L2P terminal-DWA, partitioning, determinize, postprocess, or `possible_matches` costs
- Running `make example-specific` on a single target case to isolate build behavior

## Hard Invariants
- For L2P terminal-DWA construction, state equivalence and vocab equivalence analysis must always run fully.
- Max-length may be skipped in controlled cases, but the full exact state/vocab equivalence pass must not be bypassed.
- Generated masks must be exact.
- No over-approximation, no under-approximation, no approximate acceptance paths, and no sampled, hash-only, or fingerprint-only proof of semantic equivalence.
- Do not restore shortcut id-map paths that bypass exact L2P equivalence.

## Regression Warning
- On 2026-05-09, the `fast_sound_id_map` bypass made `Github_easy---o76439` compile in roughly `50s`.
- Disabling that bypass and restoring full L2P equivalence brought the same case back to roughly `0.7s`.
- Do not restore fast-sound, identity, lex-dedup, or similar shortcut L2P id-map paths.

## Workflow
1. Profile first. Do not guess.
2. Identify the dominant phase from raw profile output before proposing changes.
3. Separate time into these buckets:
   - `id_map` / equivalence analysis
   - `terminal_nwa_build`
   - `determinize`, postprocess, minimize, or compact phases
   - `possible_matches`
4. Determine whether the regression is global, partition-specific, or example-specific.
5. Prefer targeted reproduction on one problem before any broader sweep.
6. Validate both correctness and timing on the target case before reporting broader conclusions.

## Targeted Commands

```bash
cd /Users/isaacbreen/Projects2/constraint-framework-analysis
PYTHONPATH=. python -m scripts.profile_build --problem <problem> --framework glrmask_native --vocab llama3 --raw-log /tmp/<name>.log --json-output /tmp/<name>.json
```

```bash
cd /Users/isaacbreen/Projects2/constraint-framework-analysis
GLRMASK_PROFILE_COMPILE=1 GLRMASK_PROFILE_COMPILE_SUMMARY=1 make example-specific PROBLEM=jsb/data/<problem>.json FRAMEWORKS='glrmask_native' MAX_EXAMPLES=1
```

## Interpretation Checklist
- If `id_map` dominates, inspect state equivalence, vocab equivalence, tokenizer simplification, and partitioning decisions.
- If `terminal_nwa_build` dominates, inspect terminal grouping, partition sizes, and whether the id-map stayed too close to identity.
- If `determinize` or postprocess dominates, inspect terminal-NWA size, collapse and prune effectiveness, and whether upstream reduction failed.
- If `possible_matches` dominates, isolate tokenizer-state and vocab-token growth separately from terminal-DWA work.

## Validation Requirements
- Re-run the exact target profile after each material change.
- Confirm the profile moved in the intended phase, not just in total time.
- Preserve exact mask behavior on the target case.
- Do not report an optimization as valid until correctness and timing both hold on the target case.