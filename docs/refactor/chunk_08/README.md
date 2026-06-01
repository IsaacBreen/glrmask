# Chunk 08 — JSON Schema importer restructuring

## Goal

Make the JSON Schema importer publishable by separating the mathematics of JSON
Schema from the mechanics of grammar emission.  The previous tree grouped files
by accident of implementation:

```text
ast.rs
load.rs
lower.rs
object.rs
array.rs
string.rs
number.rs
combinators.rs
```

That layout hid the central semantic distinction: JSON Schema is a language of
JSON **values**, while `glrmask` ultimately needs a grammar over JSON **texts**.
The new layout encodes that distinction directly.

## New phase tree

```text
src/import/json_schema/
  mod.rs
  schema/
  load/
  normalize/
  lower/
  options.rs
  diagnostics.rs
  tests/
```

## Exact conceptual pipeline

Let `V_json` be the set of JSON values.  Let `B*` be byte strings.  A JSON Schema
`S` denotes a set `[[S]] ⊆ V_json`.  A JSON text encoder/whitespace policy maps a
value `v` to a set of accepted byte strings `enc(v) ⊆ B*`.  The importer should
construct a grammar whose language is

```text
L(S) = ⋃_{v ∈ [[S]]} enc(v)
```

or, when exactness is not supported, a documented over-approximation `L'(S)`
such that `L(S) ⊆ L'(S)`.

## What changed in code

1. `schema/` now owns typed loaded schema syntax.  It is deliberately not named
   `ast` because the important object is not merely syntactic; it is a located,
   typed approximation of JSON Schema assertions.
2. `load/` now owns raw JSON loading.  It is the only layer whose job is to
   inspect arbitrary `serde_json::Value` schema objects.
3. `normalize/` now owns combinator algebra.  The old `combinators.rs` was large
   and semantically dense, so it now has a dedicated home.
4. `lower/` now owns grammar emission and contains the object/array/string/number
   lowerers as children.
5. `options.rs` owns JSON-Schema-local configuration and the JSON-Schema-local
   env vars.
6. `diagnostics.rs` owns importer errors.
7. `tests/` is a directory, preparing for eventual split by semantic family.

## What intentionally did not happen yet

This chunk performs the major structural boundary move without compiling.  It
keeps the largest old implementation bodies intact inside their new homes.
The next repair pass should split these files further and work through compiler
errors deliberately rather than prematurely optimizing for compile success.

## Definition of done for this chunk

- The JSON importer has explicit phase directories.
- The public facade remains `schema_to_named_grammar`.
- The schema syntax structs no longer live in a vague `ast.rs` file.
- The grammar-emitting code lives under `lower/`.
- The schema algebra code lives under `normalize/`.
- Options and diagnostics are separated from the facade.
- A future contributor can identify where each JSON Schema concept belongs.
