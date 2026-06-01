# Normalization algebra

## Why this package exists

This work package is part of the JSON Schema importer publication cleanup.  It is written so a future implementer can apply it without needing to reconstruct the whole design from memory.

## Required changes

1. Every rewrite needs an exactness or broadening proof.
2. allOf merge is intersection over JSON values, not intersection over GrammarExpr unless terminal-safe.
3. anyOf factoring should be treated as union factoring with explicit subsumption checks.
4. oneOf must be revisited because exclusive-one semantics is not ordinary choice.
5. not must remain shape-limited until complement construction is available.

## Definition of done

- The source tree has an obvious home for: every rewrite needs an exactness or broadening proof.
- The source tree has an obvious home for: allof merge is intersection over json values, not intersection over grammarexpr unless terminal-safe.
- The source tree has an obvious home for: anyof factoring should be treated as union factoring with explicit subsumption checks.
- The source tree has an obvious home for: oneof must be revisited because exclusive-one semantics is not ordinary choice.
- The source tree has an obvious home for: not must remain shape-limited until complement construction is available.

## Mathematical invariant

No change in this package may silently narrow the denotation of a supported schema.  If exactness is not preserved, the emitted grammar language must be a documented superset of the exact language or the schema must be rejected.
