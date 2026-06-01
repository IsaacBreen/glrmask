# Chunk 07 risk register

## Risk 1: visibility errors across lower child modules

Methods moved into child modules use `impl super::Lowerer`. If a method is called from the parent module, it must be visible to the parent. This was addressed with `pub(super)` for the parent-called methods, but compile repair should check this first.

Severity: medium.  
Repair: adjust visibility, not structure.

## Risk 2: `#![deny(warnings)]` turns unused imports into hard errors

Moving functions often leaves imports behind. Because the crate denies warnings, ordinary unused imports become compile failures.

Severity: high during compile repair.  
Repair: delete unused imports after first `cargo check`.

## Risk 3: compatibility shim re-export visibility

The old `crate::grammar::ast::comma_sep_shape` re-export points at a crate-visible item. Depending on Rust visibility rules, it may need a wrapper function.

Severity: low/medium.  
Repair: replace the re-export with:

```rust
pub(crate) fn comma_sep_shape() -> crate::grammar_ir::ast::CommaSepShape {
    crate::grammar_ir::lower::separated_sequence::comma_sep_shape()
}
```

## Risk 4: extracted tests import from wrong parent

Moving inline `mod tests` to `tests.rs` changes what `super::` means if an extra nested module remains. This chunk removed the most obvious wrappers, but compile repair should check test modules.

Severity: low for library build, medium for tests.  
Repair: direct imports from `crate::grammar_ir`.

## Risk 5: renderer helpers become too broadly visible

`regex_escape_byte` and `u8set_to_class_def` are renderer-ish but also useful to lowering. Making them `pub(crate)` is acceptable for now, but long term they might belong in a small `escape` utility module.

Severity: low.  
Repair later: create `grammar_ir::render::escape` or `util::escape`.

## Risk 6: old import paths hide incomplete migration

Compatibility shims are useful but can hide stale conceptual naming. Later chunks should migrate imports gradually.

Severity: architectural.  
Repair: use the migration backlog.

## Risk 7: GLRM parser/renderer split changes private helper access

`to_glrm` was moved to `render::glrm` and re-exported. If a test expected private parser helpers from the same module, imports may need adjustment.

Severity: low.  
Repair: keep `pub use render::glrm::to_glrm` in `grammar_ir::glrm`.
