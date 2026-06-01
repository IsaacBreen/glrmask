# Array lowering

## Why this package exists

This work package is part of the JSON Schema importer publication cleanup.  It is written so a future implementer can apply it without needing to reconstruct the whole design from memory.

## Required changes

1. Distinguish homogeneous arrays from tuple arrays.
2. Keep bounded-array terminal fast paths semantically equivalent to general array grammar.
3. Treat legacy tuple items as load-time compatibility, not lower-time special case.
4. Document separator and whitespace policy.
5. Add tests for prefixItems + items interactions.

## Definition of done

- The source tree has an obvious home for: distinguish homogeneous arrays from tuple arrays.
- The source tree has an obvious home for: keep bounded-array terminal fast paths semantically equivalent to general array grammar.
- The source tree has an obvious home for: treat legacy tuple items as load-time compatibility, not lower-time special case.
- The source tree has an obvious home for: document separator and whitespace policy.
- The source tree has an obvious home for: add tests for prefixitems + items interactions.

## Mathematical invariant

No change in this package may silently narrow the denotation of a supported schema.  If exactness is not preserved, the emitted grammar language must be a documented superset of the exact language or the schema must be rejected.
