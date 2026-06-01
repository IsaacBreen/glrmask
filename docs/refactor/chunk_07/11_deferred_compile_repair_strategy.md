# Deferred compile-repair strategy

The user's instruction for this refactor series is explicit: do not compile prematurely. This chunk therefore performs static source restructuring only. When compile repair begins, use this strategy.

## First compile pass

Run `cargo check` and classify errors into these buckets:

1. missing module declarations;
2. visibility mismatch (`private item` or `private type`);
3. unused imports due to `#![deny(warnings)]`;
4. stale import paths from `crate::grammar_ir::ast::lower` or old grammar files;
5. test-only import errors;
6. rustfmt-only style issues.

Do not respond to all errors by flattening the tree again. Fix visibility/imports while preserving the conceptual boundaries.

## Expected likely repairs

### Visibility

Methods implemented in child modules may need `pub(super)` if called from `lower/mod.rs`.

### Renderer helpers

`regex_escape_byte`, `escape_byte`, and `u8set_to_class_def` are intentionally renderer helpers but are also used by lowering. If visibility errors arise, make them `pub(crate)` rather than moving them back to lowering.

### Compatibility shims

If `pub use` of a crate-private item fails, replace it with a crate-private wrapper or `pub(crate) use`.

### Tests

Extracted tests may need import path cleanup. Prefer importing from `grammar_ir` directly in new tests.

## Anti-repairs

Do not:

- put `Lowerer` back in `ast.rs`;
- put GLRM rendering back in the parser file;
- put named grammar transforms back under old `src/grammar/` implementation files;
- migrate every downstream import in the same compile-repair commit unless it is mechanically necessary.
