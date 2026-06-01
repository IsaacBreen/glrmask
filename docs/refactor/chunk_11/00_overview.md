# Chunk 11 overview: runtime state, Mask, and Commit boundaries

Chunk 11 is the first runtime-online cleanup after the immutable artifact split.
The goal is to make the runtime read like the mathematical object in the paper:
a fixed compiled constraint plus a mutable prefix state, with two alternating
operations.

The three mathematical objects are now deliberately visible:

```text
Constraint        immutable compiled artifact
ConstraintState   mutable frontier for one generated prefix
Mask              read-only query on ConstraintState
Commit            transition relation on ConstraintState
```

This chunk does not try to optimize.  It tries to make the existing optimized
code legible by separating semantically different concepts.  A fast path is
allowed to be ugly internally, but the surrounding file should say which
mathematical relation it implements.

The concrete source changes are:

- `src/runtime/state.rs` became `src/runtime/state/`.
- mask cache and commit scratch are no longer adjacent to semantic state fields.
- read-only parser/frontier observations moved to `state/inspect.rs`.
- forced-token logic moved to `state/force.rs` and is explicitly marked as
  derived, not primitive.
- Mask dense accumulator logic moved to `runtime/mask/dense_acc.rs`.
- Mask bitset helpers moved to `runtime/mask/bitset.rs`.
- Mask thresholds moved to `runtime/mask/constants.rs`.
- Commit environment switches moved to `runtime/commit/options.rs`.
- Commit parser-stack advance dispatch moved to `runtime/commit/parser_advance.rs`.
- Commit-vs-Mask debug assertion moved to `runtime/commit/mask_assert.rs`.
- token-id-to-byte lookup moved to `runtime/commit/token_lookup.rs`.
- runtime module docs and per-subsystem README files were added.

The chunk intentionally leaves the large body of `runtime/commit/mod.rs` in
place.  That file contains many intertwined fast paths.  Pulling those apart
without a compile/test pass would risk losing invariants.  Instead, this chunk
cuts out the mathematically clean leaf boundaries first.  The remaining commit
file is now a phase graph with fewer unrelated concerns at its top.
