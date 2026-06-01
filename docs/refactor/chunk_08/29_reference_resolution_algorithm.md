# Reference resolution algorithm and target resolver

Chunk 08 separates reference discovery from grammar rule allocation, but it does
not yet introduce a complete resolver object.  This document defines the target.

## 1. Current model

The loader records:

```text
SchemaDocument.root
SchemaDocument.definitions
SchemaDocument.ref_targets
```

The lowerer builds:

```text
definition_by_pointer: BTreeMap<String, &Schema>
definition_rules: BTreeMap<String, String>
```

This works, but the responsibilities are still split informally.

## 2. Target model

Introduce:

```rust
struct ReferenceGraph {
    root: SchemaId,
    nodes: Vec<SchemaNode>,
    aliases: BTreeMap<String, SchemaId>,
    edges: Vec<(SchemaId, SchemaId)>,
}
```

and:

```rust
struct GrammarRuleAllocator {
    pointer_to_rule: BTreeMap<String, String>,
    active_stack: Vec<String>,
}
```

The first object belongs to loading/resolution.  The second object belongs to
lowering.

## 3. Required behavior

### Local JSON Pointers

`#/a/b` should resolve by JSON Pointer semantics.  Escaping is:

```text
~0 -> ~
~1 -> /
```

The current code only needs escaping for generated locations; full unescaping
should be introduced if arbitrary incoming pointers are normalized.

### Fragment aliases

Local `$id` values such as `#node` are aliases.  They should point to the schema
node where they appear.  Absolute self references ending in `#` are currently
accepted as root aliases.  This behavior should be documented in tests.

### Recursive references

A recursive reference should allocate a grammar rule before lowering the target,
then fill that rule.  This is the same fixed-point idea used by recursive grammar
rules.  The target algorithm is:

```text
lower_ref(pointer):
  normalized = resolver.normalize(pointer)
  if normalized is root: return Ref("start")
  if allocator.has_rule(normalized): return Ref(rule)
  rule = allocator.reserve(normalized)
  expr = lower_schema(resolver.target(normalized))
  allocator.define(rule, expr)
  return Ref(rule)
```

## 4. Diagnostics

Every unresolved reference diagnostic should include:

1. the original reference string,
2. the normalized reference string if different,
3. the schema location containing the `$ref`,
4. whether the importer only supports local references.

The current diagnostic has only the reference string.  That is acceptable for the
structural chunk but should be upgraded before publication.
