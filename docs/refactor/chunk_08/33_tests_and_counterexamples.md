# Tests and counterexamples for JSON Schema import

This file lists tests that should exist after Chunk 08 is compile-repaired.  It
is deliberately phrased as oracles rather than implementation code.

## 1. Loader tests

1. Boolean `true` loads as `SchemaKind::Any`.
2. Boolean `false` loads as `SchemaKind::Never`.
3. Unsupported validation keyword reports its schema location.
4. `properties` must be an object.
5. `required` must be an array of strings.
6. `type` array deduplicates repeated types.
7. `multipleOf <= 0` is rejected.
8. Tuple-form `items` and `prefixItems` together are rejected.
9. Local `$defs` definitions are collected.
10. Property-local `$ref` targets are collected.

## 2. Normalization tests

1. `allOf` merges object required sets exactly.
2. `allOf` merges array bounds by max/min.
3. `allOf` with incompatible array bounds lowers to empty language or rejection.
4. `anyOf` object marker branches factor exactly when marker keys are exclusive.
5. Broad object collapse tests include an accepted counterexample outside exact
   semantics.
6. Recursive `allOf` refs are not inlined infinitely.

## 3. Lowering tests

1. Empty object schema lowers to `json_object`.
2. Closed required object emits every required key exactly once.
3. Additional properties exclude fixed keys.
4. Pattern properties intersect with fixed property schemas.
5. Bounded homogeneous arrays honor min and max.
6. Prefix arrays with no tail close after the prefix.
7. String min/max lengths respect split bounded string chunks.
8. Decoded regex `\\d` maps to JSON body digit language.
9. Recognized formats filter enum values.
10. Integer finite ranges enumerate when small and use regex when large.

## 4. End-to-end mask tests

Once compilation resumes, include small vocabs that make invalid continuations
visible.  For example, for a schema requiring key `"a"`, a vocab containing
`"b"` should be masked at the key position in a closed object.  These tests are
more valuable than only comparing emitted grammar strings because runtime masking
is the public behavior.

## 5. Counterexample discipline

For every broadening, write down one counterexample if practical:

```text
schema accepts:      set A
emitted grammar:     set B
counterexample:      x in B - A
reason allowed:      performance/unsupported exact check/etc.
```

If a counterexample is impossible to construct, the broadening is probably exact
and should be reclassified.
