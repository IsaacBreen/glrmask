# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Import migration rule

Every new import should use:

```rust
use crate::parser::glr::...;
```

not:

```rust
use crate::compiler::glr::...;
```

## Compatibility shim

The only place where `compiler::glr` should appear in source is the shim:

```text
src/compiler/glr/mod.rs
```

That shim re-exports `parser::glr` and maps the old `compiler::glr::parser` path to `parser::glr::advance`.

## Why a shim is kept

The shim lets downstream internal tests, benchmarks, or un-migrated experimental files continue to resolve during later compile-repair work.  The shim must not grow logic.  If someone tries to edit `src/compiler/glr/mod.rs` to add behavior, that behavior belongs in `src/parser/glr` instead.

## Migration checklist

1. Search for `crate::compiler::glr`.
2. If found outside the shim, replace with `crate::parser::glr`.
3. Search for `glr::parser`.
4. Replace with `glr::advance` unless it is intentionally documenting the old path.
5. Search for `stack_may_advance`.
6. Replace with `stack_can_advance`.
