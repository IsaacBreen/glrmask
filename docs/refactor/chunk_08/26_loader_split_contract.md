# Loader split contract and application guide

This document is the mechanical contract for the JSON Schema loader split in
Chunk 08.  It is written as if the reader has only basic Rust knowledge and
needs to decide which file to open for a change.

## 1. Purpose of the loader

The loader converts raw `serde_json::Value` into the typed value model in
`schema::*`.  The loader is not allowed to know about grammar rules, automata,
DWA construction, runtime masks, token IDs, or parser states.  This is the
first publication-quality separation of concerns for JSON Schema import:

```text
Raw JSON object keys and arrays
  -> checked keyword values
  -> typed schema structs with locations
```

The loader may reject a schema.  It may not silently broaden a schema.  Silent
broadening belongs either in a named normalization function with a proof note or
in a named lowerer function with a test.

## 2. Files and ownership

### `load/mod.rs`

This file is now only a module index.  It should stay tiny.  It declares:

```text
collect
keywords
pointers
shape
typed
```

and re-exports `load_document`.  Do not put new behavior in `mod.rs`.

### `load/typed.rs`

This file owns orchestration.  Add code here only if the code is about deciding
which loader helper to call next.  Examples that belong here:

1. boolean schema dispatch,
2. object schema dispatch,
3. `$ref` plus sibling assertion wrapping,
4. construction of a `SchemaAssertions` aggregate,
5. deciding whether object/array/string/number keywords should be loaded for an
   untyped schema.

Examples that do not belong here:

1. parsing the internals of `properties`,
2. interpreting `patternProperties`,
3. parsing numeric bounds,
4. escaping JSON pointer path segments,
5. computing a grammar rule.

### `load/keywords.rs`

This file owns keyword spelling.  The spellings `properties`, `required`,
`additionalProperties`, `items`, `prefixItems`, `minLength`, `format`,
`multipleOf`, and similar should appear here and only here unless tests mention
them.  This gives publication readers one audit point for schema coverage.

Every new validation keyword must be categorized in exactly one way:

1. supported exactly,
2. supported with documented broadening outside the loader,
3. accepted as annotation only,
4. rejected as unsupported.

If a keyword is in category 2, `keywords.rs` should only load the typed syntax;
the broadening decision belongs in `normalize` or `lower`.

### `load/pointers.rs`

This file owns local pointer spelling and raw `$ref` discovery.  It is deliberately
not a resolver.  It can collect strings.  It can escape path segments.  It can
recognize document-local aliases.  It should not allocate grammar rule names.

### `load/collect.rs`

This file owns discovery of schema nodes that deserve names or may be reference
targets.  It bridges raw JSON traversal and typed schema construction by calling
`typed::load_schema_at`, but it still belongs to loading because it is collecting
schema syntax, not grammar syntax.

### `load/shape.rs`

This file owns predicates over loaded schema shapes.  Examples include:

- a singleton `allOf` ref wrapper with no siblings,
- a `oneOf` that illegally mixes refs and inline non-null branches,
- a null-only inline branch.

These are syntactic shape checks.  They do not prove semantic equivalence of
arbitrary schemas.

## 3. Step-by-step rule for adding a keyword

Suppose a contributor wants to add `maxContains`.

1. Add it to the support matrix before code.
2. Decide whether support is exact, broad, annotation-only, or rejected.
3. If rejected, add it to the unsupported keyword list with a test.
4. If loaded, add a typed field to `schema::ArraySchema` only if the field is
   semantically meaningful after loading.
5. Parse the raw value in `load/keywords.rs`.
6. Add a loader test showing invalid raw shapes are rejected at the schema
   location.
7. Add a normalization or lowering test showing the denotation contract.

No step should mention Terminal DWA, Parser DWA, token IDs, or runtime masks.
