# Prioritized JSON importer backlog

## P0 — must happen before publication

1. Compile-repair the Chunk 08 module paths without flattening the tree.
2. Audit `oneOf` semantics and choose exact/reject/compatibility policy.
3. Audit conditionals and decide reject vs implement.
4. Add exactness/broadening comments at every fallback in `normalize/` and `lower/`.
5. Split tests by semantic family.

## P1 — high value

1. Split `lower/object/mod.rs` into named object strategies.
2. Split `lower/string/mod.rs` into pattern/format/utf8/value modules.
3. Split `normalize/combinators.rs` into allOf/anyOf/merge/shape/factor modules.
4. Split `load/mod.rs` into pointer/reference/keyword/schema-reader modules.
5. Add a public diagnostic report for broadened JSON Schema features.

## P2 — publication polish

1. Replace f64 numeric schema representation with exact decimals.
2. Rename schema-specific performance heuristics to generic strategy names.
3. Add examples for recursive schemas, objects, arrays, strings, and enums.
4. Document supported JSON Schema draft assumptions.
5. Add sample validator cross-check tests against an external JSON Schema validator in CI if feasible.
