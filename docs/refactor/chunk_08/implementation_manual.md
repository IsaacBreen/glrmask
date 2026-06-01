# Chunk 08 implementation manual

This file is a mechanical application guide for the JSON Schema importer split.
It is deliberately explicit.  A future contributor should be able to reproduce,
review, or repair the chunk from this document alone.

## 0. Starting point

The starting point is the Chunk 07 tree.  The JSON Schema importer exists as a
flat module:

```text
src/import/json_schema/
  ast.rs
  load.rs
  lower.rs
  array.rs
  object.rs
  string.rs
  number.rs
  combinators.rs
  config.rs
  error.rs
  tests.rs
  mod.rs
```

The flat layout conflates four layers:

1. typed schema syntax;
2. raw JSON loading and keyword validation;
3. schema algebra for combinators;
4. grammar emission.

The goal is not merely shorter files.  The goal is to make the file tree encode
the mathematical domain of each symbol.

## 1. Create the phase directories

Create:

```text
src/import/json_schema/schema/
src/import/json_schema/load/
src/import/json_schema/normalize/
src/import/json_schema/lower/
src/import/json_schema/tests/
```

Then move the old files according to this rule:

```text
ast.rs         -> schema/
load.rs        -> load/mod.rs
combinators.rs -> normalize/combinators.rs
lower.rs       -> lower/mod.rs
object.rs      -> lower/object/mod.rs
array.rs       -> lower/array/mod.rs
string.rs      -> lower/string/mod.rs
number.rs      -> lower/number/mod.rs
config.rs      -> options.rs
error.rs       -> diagnostics.rs
tests.rs       -> tests/mod.rs
```

Do not leave active compatibility modules named `ast`, `config`, or `error`.
Those names encourage new code to keep using the old conceptual split.

## 2. Split `ast.rs` semantically

The old `ast.rs` was not simply moved.  It was split because the schema syntax
is the most important publication boundary.  Use the following homes:

```text
schema/document.rs   SchemaDocument, SchemaDefinition
schema/mod.rs        Schema, SchemaKind, re-exports
schema/assertions.rs SchemaAssertions
schema/object.rs     ObjectSchema, PropertySchema, PatternPropertySchema, AdditionalProperties
schema/array.rs      ArraySchema
schema/scalar.rs     SchemaType, StringSchema, NumberSchema
```

This split makes the high-prominence types easy to find and makes it hard to add
new fields to a random monolithic syntax file.

## 3. Rewrite module imports

After moving files, apply these path rewrites:

| Old import | New import |
|---|---|
| `super::ast::*` from phase roots | `super::schema::*` |
| `super::ast::*` from `lower/*` | `super::super::schema::*` |
| `super::error::*` from phase roots | `super::diagnostics::*` |
| `super::error::*` from `lower/*` | `super::super::diagnostics::*` |
| `super::config::JsonSchemaConfig` | `super::options::JsonSchemaConfig` |
| `super::combinators::*` from lowerers | `super::super::normalize::*` |
| `super::lower::*` inside lower child modules | `super::*` |

When compilation starts, any remaining errors should be solved by refining these
paths, not by moving code back to the old flat layout.

## 4. Preserve facade compatibility

The external facade remains:

```rust
pub fn schema_to_named_grammar(schema: &Value) -> Result<NamedGrammar, GlrMaskError>
```

The importer phase tree is internal.  Users and other crate modules should not
start depending on `schema::*`, `load::*`, or `lower::*` unless a future API
explicitly exposes them for diagnostics or testing.

## 5. Move env-var policy

JSON-Schema-specific env vars belong in `options.rs`, not `mod.rs` and not the
compile pipeline.  The phase facade may retain small wrapper functions because
`src/import/mod.rs` currently calls `json_schema::lower_exact_subtractions_enabled()`.

## 6. Decide where new code goes

Use this decision tree:

```text
Does it read raw JSON Schema serde_json::Value?
  yes -> load/
  no  -> continue

Does it define loaded schema data structures?
  yes -> schema/
  no  -> continue

Does it transform/compare schema denotations without emitting grammar?
  yes -> normalize/
  no  -> continue

Does it emit GrammarExpr or allocate grammar rule names?
  yes -> lower/
  no  -> options.rs or diagnostics.rs or tests/
```

## 7. Compile repair order

When instructed to compile later, repair in this order:

1. module path errors;
2. visibility errors;
3. rustfmt;
4. JSON importer unit tests;
5. full crate tests;
6. benchmarks only after tests are stable.

Do not make semantic changes while fixing import paths.  Keep semantic changes
as separately reviewable chunks.
