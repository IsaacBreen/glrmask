# Basic implementer manual

This manual assumes the reader knows basic Rust but not the importer.

## Adding a new schema keyword

1. Decide whether the keyword is an assertion, annotation, applicator, or
   reference/resolution keyword.
2. Add a typed field to `schema/` only if the importer will preserve it after
   loading.
3. Parse and type-check the raw JSON value in `load/keywords.rs` or a new loader
   file.
4. Add an error path for wrong types.  Do not let a malformed keyword become an
   ignored field.
5. Add a lowering function under `lower/` if the keyword affects accepted values.
6. Write a denotational comment stating exactness.
7. Add support-matrix documentation.
8. Add golden schema-to-grammar tests and, eventually, value-level oracle tests.

## Adding a new object-lowering optimization

1. State the object value-set transformation in math first.
2. Identify whether duplicate keys matter.
3. Identify whether patternProperties overlap matters.
4. Check whether additionalProperties complement changes.
5. Add a small helper with a name that describes the theorem, not the benchmark.
6. Put benchmark-specific constants near the optimization but explain why they do
   not change semantics.
7. Add a fallback path that is simpler and obviously correct.

## Adding a new environment option

Do not add raw env reads in lowering files.  Add a field to `JsonSchemaOptions`,
read it in `options.rs`, and document whether it changes only grammar shape or
also changes accepted language.  Publication-facing options should default to the
most semantically honest behavior, not the fastest benchmark behavior.

## Moving code

When moving a helper, ask:

- Does it inspect raw JSON? It belongs in `load/`.
- Does it define typed schema data? It belongs in `schema/`.
- Does it construct `GrammarExpr`? It belongs in `lower/`.
- Does it only define unsupported names or support policy? It belongs in
  `diagnostics.rs` or a future diagnostics submodule.
- Does it tune shape/performance? It belongs in `options.rs`.
