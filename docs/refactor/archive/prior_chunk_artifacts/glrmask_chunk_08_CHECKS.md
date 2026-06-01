# Chunk 08 static checks

No compile/test/benchmark/rustfmt was run. Checks below are static source-shape checks only.

## Phase roots

- Missing expected phase files: none
- Old flat implementation files still present at JSON Schema root: none
- Loader files mentioning `GrammarExpr`: none
- Stale path markers (`super::ast`, `super::string::{`, `JsonSchemaOptions`): none

## File tree

```text
src/import/json_schema/diagnostics.rs
src/import/json_schema/load/README.md
src/import/json_schema/load/collect.rs
src/import/json_schema/load/keywords.rs
src/import/json_schema/load/mod.rs
src/import/json_schema/load/pointers.rs
src/import/json_schema/load/shape.rs
src/import/json_schema/load/typed.rs
src/import/json_schema/lower/README.md
src/import/json_schema/lower/array/README.md
src/import/json_schema/lower/array/mod.rs
src/import/json_schema/lower/mod.rs
src/import/json_schema/lower/number/README.md
src/import/json_schema/lower/number/mod.rs
src/import/json_schema/lower/object/README.md
src/import/json_schema/lower/object/mod.rs
src/import/json_schema/lower/string/README.md
src/import/json_schema/lower/string/mod.rs
src/import/json_schema/mod.rs
src/import/json_schema/normalize/README.md
src/import/json_schema/normalize/combinators.rs
src/import/json_schema/normalize/mod.rs
src/import/json_schema/options.rs
src/import/json_schema/schema/array.rs
src/import/json_schema/schema/assertions.rs
src/import/json_schema/schema/document.rs
src/import/json_schema/schema/mod.rs
src/import/json_schema/schema/object.rs
src/import/json_schema/schema/scalar.rs
src/import/json_schema/tests/mod.rs
```

## LOC

| File | Lines |
|---|---:|
| `src/import/json_schema/diagnostics.rs` | 56 |
| `src/import/json_schema/load/README.md` | 21 |
| `src/import/json_schema/load/collect.rs` | 145 |
| `src/import/json_schema/load/keywords.rs` | 279 |
| `src/import/json_schema/load/mod.rs` | 13 |
| `src/import/json_schema/load/pointers.rs` | 40 |
| `src/import/json_schema/load/shape.rs` | 54 |
| `src/import/json_schema/load/typed.rs` | 195 |
| `src/import/json_schema/lower/README.md` | 23 |
| `src/import/json_schema/lower/array/README.md` | 9 |
| `src/import/json_schema/lower/array/mod.rs` | 258 |
| `src/import/json_schema/lower/mod.rs` | 700 |
| `src/import/json_schema/lower/number/README.md` | 9 |
| `src/import/json_schema/lower/number/mod.rs` | 283 |
| `src/import/json_schema/lower/object/README.md` | 9 |
| `src/import/json_schema/lower/object/mod.rs` | 2693 |
| `src/import/json_schema/lower/string/README.md` | 9 |
| `src/import/json_schema/lower/string/mod.rs` | 1345 |
| `src/import/json_schema/mod.rs` | 74 |
| `src/import/json_schema/normalize/README.md` | 15 |
| `src/import/json_schema/normalize/combinators.rs` | 1902 |
| `src/import/json_schema/normalize/mod.rs` | 14 |
| `src/import/json_schema/options.rs` | 169 |
| `src/import/json_schema/schema/array.rs` | 21 |
| `src/import/json_schema/schema/assertions.rs` | 66 |
| `src/import/json_schema/schema/document.rs` | 21 |
| `src/import/json_schema/schema/mod.rs` | 51 |
| `src/import/json_schema/schema/object.rs` | 46 |
| `src/import/json_schema/schema/scalar.rs` | 39 |
| `src/import/json_schema/tests/mod.rs` | 4085 |


## Documentation payload

Chunk 08 docs under `docs/refactor/chunk_08/` contain 57 Markdown files and 3741 total lines.
