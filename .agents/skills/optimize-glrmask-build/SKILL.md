---
name: optimize-glrmask-build
description: 'Optimize glrmask compile and build time.'
user-invocable: true
---

# Optimize Glrmask Build

Temporary precedence note: for JSON-schema importer regressions after the modular
importer rewrite, use `$json-schema-importer-regression` first. This generic
build-time workflow is temporarily secondary for those cases; still apply its
measurement and build-profile rules when the temporary skill asks for build-time
evidence.

## When to Use
- Investigating glrmask compile-time or build-time behaviour.

## Hard Invariants
- For L2P terminal-DWA construction, state equivalence and vocab equivalence analysis must always run fully.
- Max-length may be skipped in controlled cases, but the full exact state/vocab equivalence pass must not be bypassed.
- Generated masks must be exact.
- For parser/table/runtime optimizations, remember the primary semantic contract is **recognition**, not parse-tree or parse-structure preservation. When optimizing GLR tables, parser DWA, stack effects, reductions, or synthetic states, prefer representations that preserve recognizer-visible behavior while quotienting or discarding parse-structural distinctions that are not used by mask/commit correctness. Do not keep extra states, delayed continuations, goto distinctions, or action structure solely to preserve an unused parse shape.
- No over-approximation, no under-approximation in generated masks or in the artifacts used to generate masks, including in equivalence analysis (not over-merging, although under-merging itself isn't incorrect per se), terminal DWA computation, or parser DWA computation.
- Cross-schema or cross-grammar build benchmarks must not reuse completed schema/grammar compile results. Do not cache or return whole `Constraint`s, parser DWAs, possible-matches artifacts, terminal DWAs, lowered grammars, or other schema/grammar-dependent compile outputs across different compile calls. The only allowed cross-problem caching is vocabulary-only data: artifacts determined solely by the fixed provider vocabulary/tokenizer and independent of the user-provided schema or grammar.
- JSON Schema import shape is not a presentation detail. Do not change a pattern, terminal, or lexer expression into helper nonterminals just to make GLRM dumps shorter or prettier. Such changes alter terminal structure and can materially change tokenizer/DWA/id-map build behavior; make them only when the user explicitly asks for an importer/grammar-structure change and the compatibility implications have been discussed. Presentation/local literal compaction inside the same terminal is a different class of change, but keep it narrowly scoped and explicit.
- Do not fix build-time hotspots by adding schema-, benchmark-, or pattern-specific importer fast paths unless the user explicitly asks for that class of change. When a generated regex, NFA, DFA, parser automaton, possible-matches pass, or equivalence pass is slow, first identify the general compiler mechanism that is slow (for example subset-state explosion, epsilon-closure cost, alphabet partitioning, transition materialization, memoization misses, or repeated equivalent work) and fix that mechanism directly. A narrow recognizer for one observed schema shape is a last resort requiring explicit approval and a documented reason the general compiler cannot be made fast enough.
- Do not solve build regressions by disabling, skipping, threshold-gating, or making opt-in an existing correctness, compression, table, parser-DWA, terminal-DWA, or runtime optimization. Those optimizations usually protect cases outside the immediate benchmark set. If an optimization is the hot path, preserve its semantic effect and optimize its implementation, data structures, sharing, memoization, or downstream representation. Disabling or narrowing such an optimization requires explicit human approval after documenting the protected behavior and measuring the broader tradeoff.

## Example Commands

## Measurement Workflow
- `constraint-framework-analysis` imports the installed `_glrmask` extension, not the repo source tree directly.
- After any Rust or Python extension change in `/Users/isaacbreen/Projects2/glrmask2`, run `make ffi-release` there, then restart the CFA `GlrmaskWorkerPool`, or measurements may use stale installed code.
- For timeout schemas, use short capped probes first (10s, then at most 30s unless the user asks otherwise). Treat a small schema that exceeds those caps as a compiler bug to localize, not as a reason to add an importer shortcut for that schema's pattern text.

## Optimization Notes
- When optimizing glrmask build/compile time, keep a running dated note in `/Users/isaacbreen/Projects2/gcg-paper/notes/` as work proceeds.
- Record committed wins, failed experiments, exact env vars, representative before/after timings, and keep/revert decisions.
- Before retrying an idea, check the current note for rejected experiments so work is not repeated.

## Pre-Commit Evidence Checklist
Before committing build-performance work, verify the commit message and body name the exact problem/schema ID(s) improved, the build metric, before/after build values for those named cases when known, and any measured broad distribution/count changes such as timeout counts or `>10s`/`>5s` buckets. Include profile/sweep/log artifact paths when available. If the change is motivated by a timeout or tail-latency artifact but exact before/after is unavailable, state exactly what was run, what was not run, why the remaining measurement was skipped, and the motivating evidence without implying a measured win. Mention any known tradeoff, especially when a build-time shortcut can affect runtime TBM or mask latency.

```bash
cd /Users/isaacbreen/Projects2/constraint-framework-analysis
GLRMASK_PROFILE_COMPILE=1 GLRMASK_PROFILE_COMPILE_SUMMARY=1 make example-specific PROBLEM=jsb/data/<problem>.json FRAMEWORKS='glrmask_native'
```
