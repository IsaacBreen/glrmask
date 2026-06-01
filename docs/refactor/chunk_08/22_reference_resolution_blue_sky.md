# Blue-sky reference resolver

The current importer keeps references as strings until lowering.  This is simple
but not ideal.  A publication-quality resolver would create a graph:

```text
SchemaGraph {
  nodes: Vec<SchemaNode>,
  root: NodeId,
  pointer_to_node: BTreeMap<JsonPointer, NodeId>,
  alias_to_node: BTreeMap<AnchorOrId, NodeId>,
}
```

Then `SchemaKind::Ref(String)` becomes either:

```rust
SchemaKind::Ref(NodeId)
```

or a separate `ResolvedSchema` type removes unresolved refs entirely.

Benefits:

- unresolved refs fail before lowering;
- recursion is explicit in the graph;
- diagnostics can report both reference site and target site;
- lowerer memoization uses `NodeId`, not normalized pointer strings;
- future dynamic refs/anchors have a natural place to live.

This also makes the proof statement cleaner.  The loader parses syntax; the
resolver constructs a graph; the lowerer maps graph nodes to grammar rules.
