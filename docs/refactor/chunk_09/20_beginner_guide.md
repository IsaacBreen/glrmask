# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Beginner guide to applying this chunk manually

A beginner can apply this chunk by following these mechanical steps:

1. Create `src/parser/mod.rs`.
2. Move `src/compiler/glr` to `src/parser/glr`.
3. Create a new `src/compiler/glr/mod.rs` that only re-exports from `parser::glr`.
4. In `src/lib.rs`, add `pub(crate) mod parser;`.
5. Rename `src/parser/glr/parser` to `src/parser/glr/advance`.
6. In `src/parser/glr/mod.rs`, change `pub mod parser;` to `pub mod advance;`.
7. Search all Rust source for `crate::compiler::glr` and replace with `crate::parser::glr`.
8. Search all Rust source for `glr::parser` and replace with `glr::advance`.
9. Rename `stack_may_advance_on` to `stack_can_advance_on`.
10. Rename `stack_may_advance_on_any` to `stack_can_advance_on_any`.
11. Add `advance/options.rs` and move advance env reads there.
12. Add `table/options.rs` and move table optimizer env reads there.
13. Split large files into textual fragments.
14. Add docs and checks.

The most common mistake is to move files but forget to update imports in runtime commit.  Runtime commit is the proof that GLR is not compile-only.
