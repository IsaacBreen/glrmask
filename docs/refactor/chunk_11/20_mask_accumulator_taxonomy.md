# Mask accumulator taxonomy

Mask currently has several accumulator representations:

- parser GSS frontiers from the state;
- dense internal-token accumulators carried while walking the Parser DWA;
- grouped sparse masks used to materialize original-token outputs;
- caller-visible packed `Vec<u32>` masks;
- cached merged dense words used for delta replay.

This chunk extracts `DenseMaskAcc` because it is the central representation used
between Parser-DWA traversal and final materialization.  It should not be
confused with the final `Vec<u32>` bitset.  The distinction is:

```text
DenseMaskAcc: u64 words over internal token ids
public mask:  u32 words over original token ids
```

The next Mask cleanup should separate traversal from finalization even more:

```text
mask/traverse.rs      Parser-DWA walk
mask/accumulator.rs   internal token accumulator
mask/finalize.rs      original-vocabulary materialization and cache write
mask/direct.rs        single-path direct fast path
mask/profiled.rs      profile-specific wrapping
```

For now, `dense_acc.rs` and `bitset.rs` remove the two easiest concepts from the
main phase graph.
