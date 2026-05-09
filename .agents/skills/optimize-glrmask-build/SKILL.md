---
name: optimize-glrmask-build
description: 'Optimize glrmask compile and build time.'
user-invocable: true
---

# Optimize Glrmask Build

## When to Use
- Investigating glrmask compile-time or build-time behaviour.

## Hard Invariants
- For L2P terminal-DWA construction, state equivalence and vocab equivalence analysis must always run fully.
- Max-length may be skipped in controlled cases, but the full exact state/vocab equivalence pass must not be bypassed.
- Generated masks must be exact.
- No over-approximation, no under-approximation in generated masks or in the artifacts used to generate masks, including in equivalence analysis (not over-merging, although under-merging itself isn't incorrect per se), terminal DWA computation, or parser DWA computation.

## Example Commands

```bash
cd /Users/isaacbreen/Projects2/constraint-framework-analysis
GLRMASK_PROFILE_COMPILE=1 GLRMASK_PROFILE_COMPILE_SUMMARY=1 make example-specific PROBLEM=jsb/data/<problem>.json FRAMEWORKS='glrmask_native'
```