# Chunk 08 changeset — JSON Schema importer as typed lowering pipeline

## Scope

This chunk restructures `src/import/json_schema` around the mathematical phases of JSON Schema import:

```text
raw JSON schema value
  -> typed schema document
  -> schema combinator normalization/factoring
  -> grammar lowering
```

It does not compile, test, benchmark, or rustfmt.

## Source changes

- Rewrote `src/import/json_schema/mod.rs` as a small facade and phase map.
- Replaced the former flat schema AST with `schema/`:
  - `schema/mod.rs`
  - `schema/document.rs`
  - `schema/assertions.rs`
  - `schema/object.rs`
  - `schema/array.rs`
  - `schema/scalar.rs`
- Split raw JSON loading into `load/`:
  - `load/mod.rs` phase root
  - `load/typed.rs` orchestration and located schema-node construction
  - `load/keywords.rs` raw keyword parsing and supported-key checks
  - `load/collect.rs` definition/reference-target collection
  - `load/pointers.rs` local pointer and `$id` alias utilities
  - `load/shape.rs` syntactic shape predicates
- Moved grammar lowering into `lower/`:
  - `lower/mod.rs` context, JSON lexical builtins, refs, literal lowering, helper grammar constructors
  - `lower/object/mod.rs` object lowering
  - `lower/array/mod.rs` array lowering
  - `lower/string/mod.rs` string and JSON-string regex lowering
  - `lower/number/mod.rs` number/integer lowering
- Moved schema combinator algebra into `normalize/combinators.rs` with `normalize/mod.rs` re-exporting the important helpers.
- Renamed `config.rs` to `options.rs` and moved JSON-Schema-local env-var policy there.
- Renamed `error.rs` to `diagnostics.rs`.
- Moved `tests.rs` to `tests/mod.rs`.
- Added source-local README files for `load`, `lower`, `normalize`, and lower-family modules.
- Updated `docs/json_schema.md` and added a large self-contained Chunk 08 documentation set under `docs/refactor/chunk_08/`.

## Mathematical improvement

The source tree now reflects the central denotational fact:

```text
JSON Schema denotes values; glrmask grammars denote encoded texts.
```

The previous structure made that distinction implicit and easy to violate.  The new structure assigns each symbol to the layer where it belongs: schema syntax, loading/reference discovery, schema algebra normalization, or grammar lowering.

## Known deferred work

The largest implementation files are intentionally not fully split internally yet:

- `lower/object/mod.rs`
- `lower/string/mod.rs`
- `normalize/combinators.rs`
- `tests/mod.rs`

They now live in the right phase directory.  Subsequent repair/refinement chunks should split them internally without undoing the phase tree.

## Compile note

This chunk intentionally follows the instruction not to compile yet.  The most likely compile-repair tasks are module-path and visibility fixes caused by moving large existing files into phase directories.
