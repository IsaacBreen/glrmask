# Terminal DWA algorithm deep dive after Chunk 04

## Reading order after this chunk

A new implementer should read Terminal-DWA construction in this order:

1. `src/compile/terminal_dwa/mod.rs`: the map of the subsystem.
2. `src/compile/terminal_dwa/types.rs`: the small shared vocabulary of profile and coloring types.
3. `src/compile/terminal_dwa/options.rs`: the build-policy knobs.
4. `src/compile/terminal_dwa/vocab_partition.rs`: how local sub-vocabs are chosen.
5. `src/compile/terminal_dwa/global_state_map.rs`: the shared tokenizer-state quotient.
6. `src/compile/terminal_dwa/builder.rs`: top-level orchestration.
7. `src/compile/terminal_dwa/partition.rs`: one local sub-vocab build.
8. `src/compile/terminal_dwa/direct_partition/mod.rs`: direct local builder.
9. `src/compile/terminal_dwa/pair_partition/mod.rs`: pair local builder.
10. `src/compile/terminal_dwa/merge.rs`: id-map and automaton reconciliation.

## Algorithm as a phase list

### Phase T0: classify support

The pipeline precomputes terminal path classification support in `compile/pipeline/terminal_scan.rs`.  The shared classify cache is passed into the Terminal-DWA builder so the expensive byte-set scan is not repeated unnecessarily.

### Phase T1: global tokenizer-state quotient

The pipeline builds `global_max_length_state_map` before the parallel Terminal-DWA and scan-relation build.  This is done by `global_state_map.rs` and timed as id-map work.

### Phase T2: vocabulary partition choice

`builder.rs` calls `vocab_partition::choose_terminal_dwa_sub_vocabs`.  That function returns an `Arc<[Vocab]>`.  It intentionally does not return any DWA artifacts.

The currently supported strategies are:

- `CharType`: seven coarse byte/character partitions.
- `PairPartitionCost`: cost-estimated partitions optimized directly for pair-partition complexity.
- `AutoPairPartitionCost`: choose between char-type and pair-cost partitions using safety thresholds.

### Phase T3: local partition builds

For each sub-vocab, `partition::build_partition_terminal_dwa` is called in parallel.  That function splits grammar terminals into direct and pair masks by terminal path length.  It then calls:

- `direct_partition::build_direct_partition_terminal_dwa` if direct terminals exist;
- `pair_partition::build_pair_partition_terminal_dwa` if multi-step terminals exist.

The two results are merged locally.

### Phase T4: global merge

If there is only one local partition, the local result is already compacted and can be returned directly.  If there is more than one partition, `merge::merge_id_maps_and_terminal_dwas` is called.  This reconciles local state/token ids and combines automata.

### Phase T5: profile aggregation

`builder.rs` keeps profile aggregation near the orchestration because it is about phase accounting, not automaton semantics.  The profile is not allowed to decide correctness.

## What still needs later cleanup

The largest remaining implementation blob in this subsystem is `direct_partition/mod.rs`.  It should eventually be split into:

- `direct_partition/options.rs`
- `direct_partition/vocab_order.rs`
- `direct_partition/state_equivalence.rs`
- `direct_partition/signature.rs`
- `direct_partition/range_accumulator.rs`
- `direct_partition/builder.rs`
- `direct_partition/profile.rs`

That split is deliberately not completed here because Chunk 04 focuses on the top-level Terminal-DWA boundary.  Doing both top-level split and full direct-partition surgery in the same patch would make review harder.
