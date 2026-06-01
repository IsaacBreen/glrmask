# JSON Schema importer

The JSON Schema importer is a compiler from a value-level constraint language to
the crate's grammar IR.  It is deliberately not described as "parsing JSON
Schema into a grammar" because JSON Schema assertions denote sets of JSON values,
not token strings.  The importer therefore has four conceptual layers.

```text
serde_json::Value
  -> schema::SchemaDocument
  -> normalization/factoring over schema denotations
  -> grammar rules over encoded JSON text
  -> ordinary glrmask compile pipeline
```

## Source tree

```text
src/import/json_schema/
  mod.rs          facade and phase boundary
  schema/         typed loaded schema syntax
  load/           serde_json::Value -> schema::SchemaDocument
  normalize/      allOf/anyOf/oneOf factoring and safe broadening
  lower/          schema::SchemaDocument -> NamedGrammar
  options.rs      importer-local configuration and env vars
  diagnostics.rs  schema import errors
  tests/          semantic and regression tests
```

## Mathematical contract

For a schema `S`, let `J(S)` be the JSON values accepted by the JSON Schema
semantics supported by this importer.  Let `E(v)` be the canonical family of JSON
texts that encode value `v` under the whitespace policy emitted by the importer.
The ideal lowered grammar language is

```text
L(S) = union { E(v) | v in J(S) }.
```

Some JSON Schema features are lowered exactly, some are intentionally broadened,
and some are rejected.  Any broadening must satisfy

```text
L_exact(S) subseteq L_emitted(S)
```

and must be named in the semantic coverage matrix under
`docs/refactor/chunk_08/semantic_coverage_matrix.md`.

## Publication principle

The importer should be readable by someone who knows JSON Schema but not GLR
internals.  JSON Schema syntax, schema algebra, and grammar emission must remain
separate in the file tree.
