# Chunk 04 implementation manual: Terminal DWA construction cleanup

Chunk 04 cleans up Terminal-DWA construction after the compile pipeline has been made explicit.  The previous state had the correct paper-facing directory name, but the top-level `terminal_dwa/mod.rs` still mixed at least five concerns: environment variables, vocabulary partitioning, global tokenizer-state quotienting, partition orchestration, and merge profiling.  That is mathematically misleading because those concerns live at different semantic levels.

The Terminal DWA itself is simple.  It is a weighted deterministic automaton over completed grammar-terminal strings.  Its evaluation on a terminal sequence returns the lexer-state/token pairs that can produce that sequence.  Everything else in this subsystem is a construction strategy for building that object without exploding intermediate automata.

This chunk therefore makes a strong source-tree claim:

- `builder.rs` may orchestrate but must not define policy.
- `options.rs` may read environment variables but must not perform construction.
- `vocab_partition.rs` may choose sub-vocabularies but must not build local automata.
- `global_state_map.rs` may compute tokenizer-state quotients but must not know about direct-vs-pair partitions.
- `partition.rs` may combine direct and pair local builders for one sub-vocabulary.
- `direct_partition/` and `pair_partition/` own the two local construction algorithms.
- `merge.rs` owns reconciliation of local id maps and local automata.

The chunk intentionally does not compile or benchmark.  The target is publication shape and semantic boundaries first.  Compile repair belongs later.


## Files changed

### `src/compile/terminal_dwa/mod.rs`

Before this chunk, `mod.rs` was a real implementation file of more than six hundred lines.  It declared submodules but also read environment variables, warmed caches, decided how to partition the vocabulary, computed the global max-length state map, and built the final mapped DWA.  A reader looking for the mathematical object had to pass through local performance heuristics.

After this chunk, `mod.rs` is a boundary file.  It documents the subsystem and lists the pieces.  It also re-exports only the build entry points used elsewhere:

```rust
pub(crate) use builder::{
    build_terminal_dwa,
    build_terminal_dwa_with_precomputed_global_max_length,
};
pub(crate) use global_state_map::build_global_max_length_state_map;
pub(crate) use vocab_partition::prepare_vocab_for_terminal_dwa;
```

That is the correct prominence.  The names a neighbouring phase should know are: prepare, build with precomputed support, build fresh, and global state-map support.  It should not know about pair partition cost objectives, auto thresholds, char-type sub-vocab cache internals, or profile line details.

### `src/compile/terminal_dwa/options.rs`

This is a new policy file.  It centralizes historical environment-variable decisions that affect the route to a Terminal DWA:

- `GLRMASK_PARTITION_SCHEME`
- `GLRMASK_PAIR_PARTITION_COST_FN`
- `GLRMASK_PAIR_PARTITION_COST_OBJECTIVE`
- `GLRMASK_PAIR_PARTITION_COST_PARTITIONS`
- `GLRMASK_PAIR_PARTITION_AUTO_SECOND_LARGEST_LIMIT`
- `GLRMASK_PAIR_PARTITION_AUTO_MAX_ESTIMATED_TERMINALS`
- `GLRMASK_PAIR_PARTITION_AUTO_MIN_ESTIMATED_TERMINALS`
- `GLRMASK_PAIR_PARTITION_AUTO_MIN_GRAMMAR_TERMINALS`
- `GLRMASK_USE_GLOBAL_MAX_LENGTH`

The important structural choice is the enum:

```rust
pub(crate) enum VocabPartitionScheme {
    CharType,
    PairPartitionCost,
    AutoPairPartitionCost,
}
```

That enum gives the code a real type for a mathematical/performance policy.  The old version repeatedly compared raw strings.  Raw strings are acceptable at the process boundary, but they should not be the language used inside the compiler.

### `src/compile/terminal_dwa/vocab_partition.rs`

This file owns the question: how do caller vocabulary tokens get split into local sub-vocabularies before local Terminal DWAs are built?

It contains:

- `CharTypeSubVocabs`, the derived-artifact cache for char-type sub-vocabs.
- `vocab_from_token_partitions`, a helper for turning token-id partitions into `Vocab`s.
- `build_char_type_sub_vocabs`, the old char-type path.
- `prepare_vocab_for_terminal_dwa`, the cache warmer previously in `mod.rs`.
- `choose_terminal_dwa_sub_vocabs`, the single public decision point.
- `choose_cost_partitioned_sub_vocabs`, the cost-partition path.
- `choose_auto_partitioned_sub_vocabs`, the auto strategy.

The strong boundary is that these functions return `Arc<[Vocab]>`.  They do not return a DWA.  They do not touch id-map merging.  They do not compact anything.  They produce only the construction partition.

### `src/compile/terminal_dwa/global_state_map.rs`

This file owns the global tokenizer-state quotient computed before both Terminal-DWA and scan-relation work.  It moved out of `mod.rs` because it is neither vocabulary partitioning nor local automaton construction.

It contains:

- the local `use_global_max_length` decision;
- the call to `run_state_equivalence_pipeline` with `StateEquivalenceScope::Global`;
- profile output for the global max-length phase.

This makes the relation between Terminal DWA and scan relation clearer.  Both phases can share the global state map, but neither phase should pretend the map is part of its denotation.

### `src/compile/terminal_dwa/builder.rs`

This file owns top-level orchestration.  Its algorithm is now readable as:

1. obtain or create shared classify cache;
2. choose sub-vocabularies;
3. allocate shared pair-partition caches;
4. build local partition DWAs in parallel;
5. collect dominant profile information;
6. handle the empty-vocabulary case;
7. perform the global merge if there is more than one local partition;
8. return the mapped DWA and phase profile.

This is still implementation code, not just a facade.  But it has a single conceptual job: assemble the Terminal DWA from already-named components.

## Why this chunk matters mathematically

The old `mod.rs` had the right final output but the wrong explanatory shape.  It mixed the denotation with several quotients.  There are at least four equivalence/partition operations in this subsystem:

1. caller-token vocabulary partitioning;
2. global tokenizer-state equivalence by max-token-length behaviour;
3. direct-partition exact state equivalence by token signatures;
4. pair-partition state/vocab equivalence used to construct the NWA/DWA.

Those are not interchangeable.  They live at different layers and have different proof obligations.  A publication-quality codebase should make the proof obligations visible in the file tree.

## Definition of done for this chunk

- `terminal_dwa/mod.rs` is a small boundary module, not an implementation blob.
- Environment-variable parsing for top-level Terminal-DWA policy is in `options.rs`.
- Vocabulary partitioning is in `vocab_partition.rs`.
- Global max-length state quotienting is in `global_state_map.rs`.
- Top-level build orchestration is in `builder.rs`.
- Existing callers can continue to call `crate::compile::terminal_dwa::build_terminal_dwa_with_precomputed_global_max_length` and `build_global_max_length_state_map` through re-exports.
- No compile/test/rustfmt work is performed in this chunk.
