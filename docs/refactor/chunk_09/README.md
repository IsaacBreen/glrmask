# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## What this chunk is

Chunk 09 is the parser-domain cleanup.  Earlier chunks promoted the paper-level compiled objects: Terminal DWA, Scan/CanMatch, Parser DWA, grammar IR, and JSON Schema lowering.  Those objects all rely on a lower-level parser model: a normalized grammar, a GLR transition table, and a one-terminal stack-advance relation.  Before this chunk, that model lived under `compiler::glr`, which suggested it was only compile-time infrastructure.  That was mathematically wrong: `commit`, `mask`, template DFA characterization, and Parser-DWA construction all depend on the same parser-stack transition semantics.

## What changed

1. `src/compiler/glr/**` was moved to `src/parser/glr/**`.
2. `src/compiler/glr/mod.rs` became a hidden compatibility shim.
3. `src/lib.rs` now declares `pub(crate) mod parser;`.
4. Runtime and compile imports now refer to `crate::parser::glr`.
5. The historical `parser` submodule was renamed to `advance`, because it does not define a parser; it defines terminal stack advancement over a parser GSS.
6. Exact predicates were renamed from `stack_may_advance_on*` to `stack_can_advance_on*`.
7. Direct environment-variable reads in parser advance and table optimization were collected into typed option objects.
8. `analysis.rs`, `advance/mod.rs`, and `table/optimize.rs` were split into reading fragments.

## What did not change

This chunk intentionally does not compile, test, benchmark, or rustfmt.  It also does not rewrite the GLR algorithm itself.  The point is to expose the mathematical boundaries first, so later compile-repair and algorithm-improvement work has the right source map.

## Reading order

1. `src/parser/glr/README.md`
2. `src/parser/glr/mod.rs`
3. `src/parser/glr/analysis.rs`
4. `src/parser/glr/table/mod.rs`
5. `src/parser/glr/table/optimize.rs`
6. `src/parser/glr/advance/mod.rs`
7. This directory's `01_mathematical_contract.md` and `16_review_checklist.md`
