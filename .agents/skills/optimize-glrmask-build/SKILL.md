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
- Cross-schema or cross-grammar build benchmarks must not reuse completed schema/grammar compile results. Do not cache or return whole `Constraint`s, parser DWAs, possible-matches artifacts, terminal DWAs, lowered grammars, or other schema/grammar-dependent compile outputs across different compile calls. The only allowed cross-problem caching is vocabulary-only data: artifacts determined solely by the fixed provider vocabulary/tokenizer and independent of the user-provided schema or grammar.
- JSON Schema import shape is not a presentation detail. Do not change a pattern, terminal, or lexer expression into helper nonterminals just to make GLRM dumps shorter or prettier. Such changes alter terminal structure and can materially change tokenizer/DWA/id-map build behavior; make them only when the user explicitly asks for an importer/grammar-structure change and the compatibility implications have been discussed. Presentation/local literal compaction inside the same terminal is a different class of change, but keep it narrowly scoped and explicit.

## Example Commands

```bash
cd /Users/isaacbreen/Projects2/constraint-framework-analysis
GLRMASK_PROFILE_COMPILE=1 GLRMASK_PROFILE_COMPILE_SUMMARY=1 make example-specific PROBLEM=jsb/data/<problem>.json FRAMEWORKS='glrmask_native'
```
