# Publication comments

## Why this package exists

This work package is part of the JSON Schema importer publication cleanup.  It is written so a future implementer can apply it without needing to reconstruct the whole design from memory.

## Required changes

1. Replace comments like simple importer / old importer with precise history-neutral language.
2. Every fallback should say exact, broadening, or rejected.
3. Every heuristic threshold should state whether it affects semantics or only grammar size.
4. Use paper terminology: schema denotation, grammar language, encoded JSON text.
5. Avoid internal benchmark nicknames in core source comments.

## Definition of done

- The source tree has an obvious home for: replace comments like simple importer / old importer with precise history-neutral language.
- The source tree has an obvious home for: every fallback should say exact, broadening, or rejected.
- The source tree has an obvious home for: every heuristic threshold should state whether it affects semantics or only grammar size.
- The source tree has an obvious home for: use paper terminology: schema denotation, grammar language, encoded json text.
- The source tree has an obvious home for: avoid internal benchmark nicknames in core source comments.

## Mathematical invariant

No change in this package may silently narrow the denotation of a supported schema.  If exactness is not preserved, the emitted grammar language must be a documented superset of the exact language or the schema must be rejected.
