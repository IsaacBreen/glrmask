# Reference resolution

## Why this package exists

This work package is part of the JSON Schema importer publication cleanup.  It is written so a future implementer can apply it without needing to reconstruct the whole design from memory.

## Required changes

1. Model local references as graph edges from located schema nodes to pointer targets.
2. Separate pointer syntax normalization from graph traversal.
3. Remote URI references must be rejected or explicitly unsupported before lowering.
4. Recursive references are grammar-recursive only after lower::refs allocates rule names.
5. Aliases from $id/id should be represented as named local targets with collision policy.

## Definition of done

- The source tree has an obvious home for: model local references as graph edges from located schema nodes to pointer targets.
- The source tree has an obvious home for: separate pointer syntax normalization from graph traversal.
- The source tree has an obvious home for: remote uri references must be rejected or explicitly unsupported before lowering.
- The source tree has an obvious home for: recursive references are grammar-recursive only after lower::refs allocates rule names.
- The source tree has an obvious home for: aliases from $id/id should be represented as named local targets with collision policy.

## Mathematical invariant

No change in this package may silently narrow the denotation of a supported schema.  If exactness is not preserved, the emitted grammar language must be a documented superset of the exact language or the schema must be rejected.
