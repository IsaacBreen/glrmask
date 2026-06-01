# Detailed follow-up blueprint: direct partition

`src/compile/terminal_dwa/direct_partition/mod.rs` remains the largest unsplit file in this subsystem.  Chunk 04 does not split it, but it documents the exact future split so the next relevant chunk can be mechanical rather than impressionistic.

## Future file: `direct_partition/options.rs`

All direct-partition-specific env decisions.  It should not import DWA.

Move these symbols:
- `skip_max_length_for_partition`
- `skip_direct_partition_max_length_for_partition`
- `direct_partition_max_length_min_states`
- `should_skip_max_length_for_partition`
- `fast_projected_direct_partition_id_map_enabled`
- `fast_projected_direct_partition_id_map_max_tsids`
- `should_use_fast_projected_direct_partition_id_map`
- `compact_direct_partition_terminal_dwa_enabled`

Acceptance rule: after this move, the file should have one reason to change.  Imports should be made explicit rather than relying on `super::*`.

## Future file: `direct_partition/vocab_order.rs`

The stable ordering of tokens and prefix buckets.

Move these symbols:
- `DirectPartitionIdentityVocabOrder`
- `direct_partition_identity_vocab_order`
- `prepare_direct_partition_identity_vocab_order`
- `build_direct_partition_identity_vocab_map`
- `DirectPartitionSortedTokenBuckets`
- `build_direct_partition_sorted_token_buckets`

Acceptance rule: after this move, the file should have one reason to change.  Imports should be made explicit rather than relying on `super::*`.

## Future file: `direct_partition/range_accumulator.rs`

Token-id range accumulation and interning.

Move these symbols:
- `PreHashedRanges`
- `LazyRanges`
- `range_hash_val`
- `append_token_id_range`
- `append_token_id_span`
- `flush_end_rep_run`
- `merge_ranges_in_place`
- `shared_rangeset_from_unsorted_pairs`

Acceptance rule: after this move, the file should have one reason to change.  Imports should be made explicit rather than relying on `super::*`.

## Future file: `direct_partition/state_equivalence.rs`

State quotienting and id-map construction for direct partition.

Move these symbols:
- `count_direct_partition_equivalence_classes`
- `build_direct_partition_id_map`
- `state_to_representative_vector`
- `find_direct_partition_exact_state_equivalence_by_token_signatures`
- `merge_deferred_equivalent_tsids`
- `remap_deferred_arced_tsids`
- `apply_tsid_perm_to_id_map`

Acceptance rule: after this move, the file should have one reason to change.  Imports should be made explicit rather than relying on `super::*`.

## Future file: `direct_partition/signature.rs`

Signature computation used to distinguish tokenizer states.

Move these symbols:
- `collect_active_terminal_signature`
- `build_direct_partition_state_to_terminal_signature`
- `direct_partition_token_signature_profile_for_state`
- `append_direct_partition_signature_profile_run`
- `direct_partition_bucket_suffix_signature_profile`

Acceptance rule: after this move, the file should have one reason to change.  Imports should be made explicit rather than relying on `super::*`.

## Future file: `direct_partition/builder.rs`

The local direct Terminal-DWA construction itself.

Move these symbols:
- `build_direct_partition_terminal_dwa`
- `build_flat_transition_table`
- `collect_direct_partition_root_ranges_by_first_byte_lcp`
- `build_end_rep_group_masks`

Acceptance rule: after this move, the file should have one reason to change.  Imports should be made explicit rather than relying on `super::*`.

## Future file: `direct_partition/profile.rs`

Counters and profile-only structs.

Move these symbols:
- `TokenLengthStats`
- `token_length_stats`
- `token_length_stats_from_entries`
- `DirectPartitionIdMapProfile`
- `DirectPartitionTsidProfileMergeReport`
- `DirectPartitionTerminalBuildProfile`

Acceptance rule: after this move, the file should have one reason to change.  Imports should be made explicit rather than relying on `super::*`.

## Future direct-partition module shape

```rust
pub(crate) mod builder;
mod options;
mod profile;
mod range_accumulator;
mod signature;
mod state_equivalence;
mod vocab_order;
pub(crate) mod max_length;

pub(crate) use builder::{
    build_direct_partition_terminal_dwa,
    build_flat_transition_table,
    count_direct_partition_equivalence_classes,
};
pub(crate) use vocab_order::prepare_direct_partition_identity_vocab_order;
```

