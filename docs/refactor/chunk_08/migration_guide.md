# Migration guide from the old JSON importer layout

| Old path | New path | Meaning |
|---|---|---|
| `ast.rs` | `schema/` | typed loaded schema syntax |
| `load.rs` | `load/mod.rs` | raw JSON value loading |
| `lower.rs` | `lower/mod.rs` | lowerer context, builtins, dispatch, refs, literals |
| `array.rs` | `lower/array.rs` | array lowering |
| `object.rs` | `lower/object.rs` | object lowering |
| `string.rs` | `lower/string.rs` | string lowering and regex conversion |
| `number.rs` | `lower/number.rs` | number/integer lowering |
| `combinators.rs` | `normalize/combinators.rs` | schema combinator algebra |
| `config.rs` | `options.rs` | importer-local options/env vars |
| `error.rs` | `diagnostics.rs` | import errors |
| `tests.rs` | `tests/mod.rs` | test root |

## Mechanical import rewrites

- `super::ast::*` became `super::schema::*` or `super::super::schema::*` depending on depth.
- `super::error::*` became `super::diagnostics::*` or `super::super::diagnostics::*`.
- `super::config::JsonSchemaConfig` became `super::options::JsonSchemaConfig`.
- Lowerer child modules import helper constructors from `super::{choice, lit, ...}`.
- Normalization code imports lowerer helpers from `super::super::lower::*`.

## Compile-repair expectations

This chunk was deliberately applied before compiling, by request.  The most
likely repair points once compilation begins are module visibility and sibling
imports in the moved `normalize/combinators.rs` and `lower/object.rs` files.  Do
not undo the phase tree to fix those errors; fix the module paths in place.
