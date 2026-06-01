# Chunk 11 review checklist

A reviewer should check the following before compile repair.

## State

- `runtime/state/mod.rs` contains the struct and no algorithmic Mask/Commit code.
- `runtime/state/cache.rs` contains only Mask cache/scratch types.
- `runtime/state/scratch.rs` contains only Commit scratch buffers.
- `runtime/state/inspect.rs` does not mutate the state.
- `runtime/state/force.rs` takes `&self` and uses clones for Commit.

## Mask

- `runtime/mask/dense_acc.rs` speaks in internal dense token sets.
- `runtime/mask/bitset.rs` speaks in original vocabulary `u32` mask words.
- `runtime/mask/constants.rs` contains thresholds only.
- `runtime/mask/mod.rs` still owns traversal and public methods.

## Commit

- `runtime/commit/options.rs` is the only place for template-DFA Commit env vars.
- `runtime/commit/parser_advance.rs` has no tokenizer logic.
- `runtime/commit/token_lookup.rs` has no parser logic.
- `runtime/commit/mask_assert.rs` has no transition logic.
- `runtime/commit/mod.rs` still increments generation after Commit methods.

## Naming

- New code should prefer `can` for exact admissibility predicates.
- `may` may remain in profile field names until a compatibility decision is made.
- New modules should use paper terms: Mask, Commit, Parser DWA, terminal,
  tokenizer state, parser stack frontier.

## No premature compile work

This chunk intentionally stops before compile/test repair.  If compile errors
appear later, fix them mechanically while preserving the boundaries above.
