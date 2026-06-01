# Compile-repair strategy for Chunk 08

The user explicitly requested not to compile during these shape-first chunks.
When compile repair begins, use this order:

1. Run `cargo fmt` only after deciding whether module paths are final.
2. Run `cargo check -q` and fix module path errors first.
3. Fix privacy errors by narrowing visibility only after the code compiles with
   broad `pub(crate)`/`pub(super)` if necessary.
4. Fix unused imports last.  Shape refactors temporarily duplicate imports; do
   not optimize them until the module split is stable.
5. Run JSON importer unit tests separately before full crate tests.
6. Add regression tests for every path touched by loader splitting.
7. Only then run benchmarks.  Importer performance should not influence the
   semantic boundary until correctness claims are stable.

Likely compile-error classes:

- sibling module visibility in `schema/mod.rs` reexports;
- child-module privacy for `lower::string` helpers used by tests;
- unused imports due to the broad prelude in moved lower modules;
- module-path mistakes from `super::` to `crate::import::json_schema::...`;
- `#![deny(warnings)]` turning unused helper docs or imports into errors.

The repair should preserve the stage boundary.  Do not fix a compile error by
moving raw JSON parsing back into lowering or by reintroducing a flat `ast.rs`.
