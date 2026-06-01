# Risk register

## Risk 1: visibility mistakes after splitting

Because the chunk splits a large file into sibling modules, the most likely
first compile errors are missing `pub(super)` visibility or imports.

Mitigation: keep fixes local.  If a helper is used by exactly one sibling, expose
it as `pub(super)`, not `pub(crate)`.  Do not merge files to silence errors.

## Risk 2: accidental algorithm change through line moves

The split is intended to preserve algorithmic behavior.  There is risk if a
helper was moved into the wrong module and edited while moving.

Mitigation: compare old and new bodies by function name when compilation begins.
The core functions should be textually almost identical except imports and
visibility.

## Risk 3: cyclic sibling imports

`vocab_materialize.rs` calls legacy validation/fallback helpers, while
`legacy_materialize.rs` reuses small utility functions from `vocab_materialize`.
Rust can handle sibling references, but privacy/import mistakes may arise.

Mitigation: if this becomes annoying, extract the shared helper
`intern_state_terminal_label` and `used_state_class_ids` into a tiny
`materialize_common.rs` rather than widening visibility.

## Risk 4: public API leakage

The new `src/scan/` module is crate-private.  Making it public too early could
freeze internal representations before the refactor is complete.

Mitigation: keep `pub(crate) mod scan;` in `lib.rs` until the final API pass.

## Risk 5: confusion between `terminal_sequences.rs` and Terminal DWA

`terminal_sequences.rs` computes sparse CanMatch maps over a token trie.  It is
not the Terminal DWA.  The name is a compromise because the same walker is used
by Terminal-DWA pair partitioning to subtract future completions.

Mitigation: if confusion persists, rename it later to `sparse_can_match.rs`.

## Risk 6: hidden performance regression

Splitting files should not affect performance, but moving helpers across modules
can affect inlining if visibility attributes change.

Mitigation: after compile is restored, benchmark scan-relation construction on
large vocabularies before and after adding `#[inline]` annotations.

## Risk 7: documentation overpromising exact mathematical set semantics

Runtime `TokenizerExecResult` records terminal matches with widths and end
states.  `CompletedTerminals` is currently a conceptual wrapper and not a full
replacement.

Mitigation: documentation says it is vocabulary, not a complete implementation
migration.  Future code should add conversions only once exact semantics are
settled.
