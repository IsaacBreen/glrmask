# Test split

## Why this package exists

This work package is part of the JSON Schema importer publication cleanup.  It is written so a future implementer can apply it without needing to reconstruct the whole design from memory.

## Required changes

1. Split the 4k-line tests module by semantic family.
2. Keep regression schemas named by bug shape, not by incidental line number.
3. Add exactness tests comparing accepted/rejected sample JSON texts.
4. Add broadening tests that demonstrate valid examples remain accepted.
5. Add reference graph tests for local ids and recursive refs.

## Definition of done

- The source tree has an obvious home for: split the 4k-line tests module by semantic family.
- The source tree has an obvious home for: keep regression schemas named by bug shape, not by incidental line number.
- The source tree has an obvious home for: add exactness tests comparing accepted/rejected sample json texts.
- The source tree has an obvious home for: add broadening tests that demonstrate valid examples remain accepted.
- The source tree has an obvious home for: add reference graph tests for local ids and recursive refs.

## Mathematical invariant

No change in this package may silently narrow the denotation of a supported schema.  If exactness is not preserved, the emitted grammar language must be a documented superset of the exact language or the schema must be rejected.
