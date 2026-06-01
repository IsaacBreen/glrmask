# Compile-repair strategy for a later pass

When the project is ready to compile after the architectural chunks, repair in
this order:

1. Run `cargo fmt` to normalize moved modules.
2. Run `cargo check --all-targets` and fix import/privacy errors only.
3. Do not change algorithms while fixing imports.
4. If `DenseMaskAcc` privacy is too narrow, add explicit accessors rather than
   making the entire type public.
5. If a moved helper is unused, decide whether it is genuinely dead or should be
   kept for diagnostics.  Prefer removal only in a warning-cleanup chunk.
6. Run tests for Mask/Commit equivalence on small grammars.
7. Run JSON Schema importer tests because they exercise the most runtime paths.
8. Run Python binding import smoke tests.
9. Only after correctness checks, run benchmarks.

Expected compile-error classes:

- missing `use` statements after module extraction;
- private fields in `DenseGssTransitionKey` or `DenseTokenSetIntersectionKey`;
- functions moved to `pub(super)` when a sibling module needs `pub(crate)`;
- rustdoc links affected by moved files;
- stale profile field naming warnings if diagnostics docs are strict.

Do not respond to these errors by undoing the boundary.  The boundary is the
point of the chunk.
