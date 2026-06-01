# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## `analysis` split

The old `analysis.rs` mixed several mathematically distinct operations:

1. emitting compile profiling;
2. defining `AnalyzedGrammar`;
3. resolving direct and indirect recursion;
4. expanding nullable productions;
5. compressing nullable runs;
6. eliminating hidden left recursion;
7. removing unreachable productions;
8. flattening single-production expansions;
9. deduplicating productions;
10. computing nullable/FIRST/FOLLOW sets;
11. tests.

Chunk 09 splits those concerns into physical files under `src/parser/glr/analysis/`.

## Important distinction

Normalization is not table construction.  It should be possible to reason about normalization as a language-preserving transform on flat productions before thinking about GLR item sets.  That is why the analysis split sits before the table split.

## Fragment responsibilities

- `profile.rs`: profiling emission for analysis/normalization.
- `model.rs`: `AnalyzedGrammar` data model and constructor.
- `right_recursion.rs`: right-recursion rewriting and helpers.
- `null_production_inline.rs`: exhaustive null production expansion.
- `nullable_run_compress.rs`: compression of nullable runs using optional trees.
- `left_recursion.rs`: hidden left-recursion elimination.
- `reachability_unit_dedup.rs`: reachability filtering, expandable unit production handling, and rule deduplication.
- `normalize.rs`: orchestration of the normalization pass.
- `fixed_point_sets.rs`: nullable, FIRST, FOLLOW computations.
- `tests.rs`: existing tests from the analysis file.

## Deferred work

The next cleanup should make these true modules.  The hardest boundary will be around helper visibility between normalization and set computation.  Do not turn this into public API; use `pub(super)` narrowly.
