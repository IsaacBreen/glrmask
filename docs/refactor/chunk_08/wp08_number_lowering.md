# Number lowering

## Why this package exists

This work package is part of the JSON Schema importer publication cleanup.  It is written so a future implementer can apply it without needing to reconstruct the whole design from memory.

## Required changes

1. Replace f64 storage with exact decimal rational before final publication if feasible.
2. Classify integer multipleOf regexes by exactness.
3. Keep range regex construction isolated from schema loading.
4. Document JSON number lexical grammar and exponent support.
5. Add proof comments for finite enumeration thresholds.

## Definition of done

- The source tree has an obvious home for: replace f64 storage with exact decimal rational before final publication if feasible.
- The source tree has an obvious home for: classify integer multipleof regexes by exactness.
- The source tree has an obvious home for: keep range regex construction isolated from schema loading.
- The source tree has an obvious home for: document json number lexical grammar and exponent support.
- The source tree has an obvious home for: add proof comments for finite enumeration thresholds.

## Mathematical invariant

No change in this package may silently narrow the denotation of a supported schema.  If exactness is not preserved, the emitted grammar language must be a documented superset of the exact language or the schema must be rejected.
