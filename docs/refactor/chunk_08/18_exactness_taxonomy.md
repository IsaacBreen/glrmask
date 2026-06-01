# Exactness taxonomy for importer code comments

Every nontrivial JSON Schema lowering helper should eventually carry one of the
following labels in a doc comment.

## Exact

A transformation is exact when it preserves the value-level denotation exactly
under the documented schema subset.  Example:

```text
const: {"x": 1}
```

lowers to the byte-language for one JSON object literal, modulo the crate's
chosen serialization and whitespace convention.

## Exact under side condition

A transformation is exact only if a syntactic or semantic side condition holds.
Example:

```text
allOf [object-schema-a, object-schema-b]
```

can be merged exactly when object assertions compose monotonically and no branch
requires incompatible property values.

## Conservative overapproximation

A transformation is conservative when it admits every valid value but may admit
invalid values.  This is acceptable only if explicitly documented and never used
for claims requiring exact constrained decoding.

`oneOf` lowered as plain grammar choice is a candidate: if two branches overlap,
choice accepts values satisfying both branches even though `oneOf` should reject
them.

## Conservative underapproximation

A transformation is an underapproximation when it rejects some valid values but
never admits invalid values.  This can be useful for decoder control but should
not be advertised as JSON Schema support unless clearly labelled.

## Unsupported

A transformation is unsupported when the importer rejects the schema before
lowering.  Unsupported is better than silently approximate for publication,
because it keeps the theorem statement honest.

## Unknown

Unknown exactness is unacceptable for final publication.  It may be tolerated in
intermediate refactor chunks, but every unknown path must be tracked in the
backlog.
