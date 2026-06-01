# Blue-sky rewrite design

If this importer were rewritten from scratch, it would use three explicit IRs.

## IR 1 — Schema syntax graph

A graph of located schema nodes, local anchors, local ids, `$defs`, and raw
keyword assertions.  This IR is close to JSON Schema syntax and owns reference
resolution.

## IR 2 — Validation algebra

A normalized algebra of regular-ish JSON value constraints:

```text
Value = Null | Bool | Number(NumberConstraint) | String(StringConstraint)
      | Array(ArrayConstraint) | Object(ObjectConstraint)
      | Union(Vec<Value>) | Intersection(Vec<Value>) | Difference(Value, Value)
```

This IR would make `oneOf`, `not`, allOf merging, and broadening choices explicit
before grammar emission.

## IR 3 — JSON text grammar templates

A grammar-oriented IR that knows JSON lexical encoding, string escaping,
separator policy, object key-order strategies, and recursive references.  This
would lower to `grammar_ir::NamedGrammar`.

## Why not jump there now?

The current chunk prepares that architecture without rewriting the semantics all
at once.  It creates directories corresponding to these layers so future commits
can migrate functions gradually while keeping the publication target visible.
