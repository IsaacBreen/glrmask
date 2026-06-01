# File-by-file ledger after Chunk 08

| File | Lines | Role | Next split |
|---|---:|---|---|
| `src/import/json_schema/diagnostics.rs` | 34 | error reporting | maybe add structured warning type |
| `src/import/json_schema/load/README.md` | 22 | raw schema loading | none |
| `src/import/json_schema/load/mod.rs` | 671 | raw schema loading | pointer/reference/keyword/schema_reader modules |
| `src/import/json_schema/lower/README.md` | 24 | supporting file | none immediate |
| `src/import/json_schema/lower/array/README.md` | 10 | array grammar lowering | homogeneous/tuple/terminal_fast_path modules |
| `src/import/json_schema/lower/array/mod.rs` | 259 | array grammar lowering | homogeneous/tuple/terminal_fast_path modules |
| `src/import/json_schema/lower/mod.rs` | 701 | lowerer context, builtins, refs, dispatch | context/names/builtins/refs/literals/dispatch modules |
| `src/import/json_schema/lower/number/README.md` | 10 | number grammar lowering | integer/decimal/range/multiple modules |
| `src/import/json_schema/lower/number/mod.rs` | 284 | number grammar lowering | integer/decimal/range/multiple modules |
| `src/import/json_schema/lower/object/README.md` | 10 | object grammar lowering | fixed/open/pattern/additional/any_of/shadow/trie modules |
| `src/import/json_schema/lower/object/mod.rs` | 2694 | object grammar lowering | fixed/open/pattern/additional/any_of/shadow/trie modules |
| `src/import/json_schema/lower/string/README.md` | 10 | string grammar lowering | pattern/format/utf8/value modules |
| `src/import/json_schema/lower/string/mod.rs` | 1346 | string grammar lowering | pattern/format/utf8/value modules |
| `src/import/json_schema/mod.rs` | 75 | supporting file | none immediate |
| `src/import/json_schema/normalize/README.md` | 16 | schema algebra | none |
| `src/import/json_schema/normalize/combinators.rs` | 1903 | schema algebra | all_of/any_of/merge/shape/factor modules |
| `src/import/json_schema/normalize/mod.rs` | 15 | schema algebra | none |
| `src/import/json_schema/options.rs` | 173 | importer configuration | none |
| `src/import/json_schema/schema/array.rs` | 22 | typed schema syntax | none immediate |
| `src/import/json_schema/schema/assertions.rs` | 67 | typed schema syntax | none immediate |
| `src/import/json_schema/schema/document.rs` | 22 | typed schema syntax | none immediate |
| `src/import/json_schema/schema/mod.rs` | 52 | typed schema syntax | none immediate |
| `src/import/json_schema/schema/object.rs` | 47 | typed schema syntax | none immediate |
| `src/import/json_schema/schema/scalar.rs` | 40 | typed schema syntax | none immediate |
| `src/import/json_schema/tests/mod.rs` | 4086 | tests | semantic family files |
