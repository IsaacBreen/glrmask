# Mask contract after the split

Mask has three layers.

First, it converts every active parser-stack configuration into an internal
accumulator.  This is where delayed exclusions and seed states matter.  The
new file `runtime/mask/dense_acc.rs` owns the dense internal-token accumulator
used for this walk.

Second, it traverses the Parser DWA.  The traversal queue and DWA transition
logic remain in `runtime/mask/mod.rs`, because this is still the central Mask
phase graph.  The queue policy remains in `runtime/mask/queue.rs`.

Third, it materializes internal-token results into the original vocabulary
bitset.  The packed-bitset layout helpers are now in `runtime/mask/bitset.rs`.
This is deliberately distinct from `runtime/bitmask_ops.rs`, which contains
lower-level dense/sparse operations over buffer words.

The public contract is unchanged:

```rust
ConstraintState::mask() -> Vec<u32>
ConstraintState::fill_mask(&mut [u32])
ConstraintState::fill_mask_profiled(&mut [u32]) -> MaskProfile
```

The mathematical contract is:

```text
A bit for original token t is set exactly when token t has at least one byte
string that can extend the current generated prefix to a prefix accepted by the
compiled grammar/lexer/parser constraint.
```

Fast paths may compute this by direct single-path traversal, queue traversal,
dense delta replay, grouped sparse materialization, or full dense copy.  Those
choices are implementation details.  The `dense_acc`, `bitset`, and `constants`
files make the categories easier to audit.
