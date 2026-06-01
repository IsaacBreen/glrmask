# String lowering

## Why this package exists

This work package is part of the JSON Schema importer publication cleanup.  It is written so a future implementer can apply it without needing to reconstruct the whole design from memory.

## Required changes

1. Keep decoded string semantics separate from encoded JSON body regexes.
2. Split pattern parsing, HIR conversion, UTF-8 byte range generation, and format regexes.
3. Audit all regex broadening on compile-limit failure.
4. Make unknown format handling explicit as annotation-ignore policy.
5. Add docs for minLength/maxLength counting units.

## Definition of done

- The source tree has an obvious home for: keep decoded string semantics separate from encoded json body regexes.
- The source tree has an obvious home for: split pattern parsing, hir conversion, utf-8 byte range generation, and format regexes.
- The source tree has an obvious home for: audit all regex broadening on compile-limit failure.
- The source tree has an obvious home for: make unknown format handling explicit as annotation-ignore policy.
- The source tree has an obvious home for: add docs for minlength/maxlength counting units.

## Mathematical invariant

No change in this package may silently narrow the denotation of a supported schema.  If exactness is not preserved, the emitted grammar language must be a documented superset of the exact language or the schema must be rejected.
