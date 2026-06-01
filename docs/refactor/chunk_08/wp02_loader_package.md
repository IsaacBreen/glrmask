# Loader package

## Why this package exists

This work package is part of the JSON Schema importer publication cleanup.  It is written so a future implementer can apply it without needing to reconstruct the whole design from memory.

## Required changes

1. Loader parses raw serde_json::Value into schema::* only.
2. Every supported keyword should have a type-checking function.
3. Unsupported validation keywords should be rejected centrally.
4. Annotation keywords should be accepted only through a documented allowlist.
5. Reference targets should be collected once with no grammar-lowering side effects.

## Definition of done

- The source tree has an obvious home for: loader parses raw serde_json::value into schema::* only.
- The source tree has an obvious home for: every supported keyword should have a type-checking function.
- The source tree has an obvious home for: unsupported validation keywords should be rejected centrally.
- The source tree has an obvious home for: annotation keywords should be accepted only through a documented allowlist.
- The source tree has an obvious home for: reference targets should be collected once with no grammar-lowering side effects.

## Mathematical invariant

No change in this package may silently narrow the denotation of a supported schema.  If exactness is not preserved, the emitted grammar language must be a documented superset of the exact language or the schema must be rejected.
