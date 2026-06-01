# Object lowering

## Why this package exists

This work package is part of the JSON Schema importer publication cleanup.  It is written so a future implementer can apply it without needing to reconstruct the whole design from memory.

## Required changes

1. Separate fixed literal keys, pattern keys, additional keys, and residual open-object branches.
2. Name key-order enumeration machinery explicitly.
3. Make required-property satisfaction a first-class state invariant.
4. Document exact-subtraction use for excluded additional-property keys.
5. Split Snowplow/schema-specific heuristics out of generic object semantics.

## Definition of done

- The source tree has an obvious home for: separate fixed literal keys, pattern keys, additional keys, and residual open-object branches.
- The source tree has an obvious home for: name key-order enumeration machinery explicitly.
- The source tree has an obvious home for: make required-property satisfaction a first-class state invariant.
- The source tree has an obvious home for: document exact-subtraction use for excluded additional-property keys.
- The source tree has an obvious home for: split snowplow/schema-specific heuristics out of generic object semantics.

## Mathematical invariant

No change in this package may silently narrow the denotation of a supported schema.  If exactness is not preserved, the emitted grammar language must be a documented superset of the exact language or the schema must be rejected.
