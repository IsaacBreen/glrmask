# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Remaining GLR backlog

Chunk 09 is intentionally structural.  The next GLR-specific improvements are:

1. Convert textual include fragments into true Rust modules.
2. Split `table/build.rs` into item-set construction, conflict resolution, action/goto assembly, and profiling.
3. Split `table/mod.rs` so the `GLRTable` struct, ambiguity diagnostics, row helpers, and testing helpers are not all in one file.
4. Split `table/row.rs` into action rows, goto rows, default-row compression, and tests.
5. Move table-build profiling into `table/profile.rs`.
6. Replace local env compatibility with explicit compile/runtime option plumbing.
7. Add property tests for stack-advance fast paths.
8. Decide when to remove the hidden `compiler::glr` shim.
9. Document the table action algebra in publication prose.
10. Decide whether GLR-specific code should remain named `glr` or whether the public paper should call it an abstract parser backend.
