# File-level and symbol-level surgery for Chunk 04

## Source movement table

| Before Chunk 04 | After Chunk 04 | Reason |
|---|---|---|
| `terminal_dwa/mod.rs` env parsing | `terminal_dwa/options.rs` | Build policy is not the denotation. |
| `terminal_dwa/mod.rs` char-type sub-vocab cache | `terminal_dwa/vocab_partition.rs` | Vocabulary partitioning is a construction choice. |
| `terminal_dwa/mod.rs` pair-cost partitioning logic | `terminal_dwa/vocab_partition.rs` | Cost strategy belongs beside partition strategy. |
| `terminal_dwa/mod.rs` auto partition decision | `terminal_dwa/vocab_partition.rs` | Auto policy chooses partitions, not automata. |
| `terminal_dwa/mod.rs` global max-length state map | `terminal_dwa/global_state_map.rs` | State quotienting is separate from vocabulary splitting. |
| `terminal_dwa/mod.rs` top-level build functions | `terminal_dwa/builder.rs` | Orchestration deserves an explicit file. |
| `terminal_dwa/mod.rs` public-ish exports | `terminal_dwa/mod.rs` re-exports | Boundary module stays small and prominent. |

## Functions introduced or moved

### `options.rs`

- `VocabPartitionScheme`
- `VocabPartitionScheme::as_str`
- `vocab_partition_scheme_from_env`
- `pair_partition_cost_fn_from_env`
- `pair_partition_objective_from_env`
- `pair_partition_count_from_env`
- `pair_partition_auto_second_largest_limit_from_env`
- `pair_partition_auto_max_estimated_pair_partition_terminals_from_env`
- `pair_partition_auto_min_estimated_pair_partition_terminals_from_env`
- `pair_partition_auto_min_grammar_terminals_from_env`
- `global_max_length_env_override`

### `vocab_partition.rs`

- `CharTypeSubVocabs`
- `vocab_from_token_partitions`
- `build_char_type_sub_vocabs`
- `prepare_vocab_for_terminal_dwa`
- `choose_terminal_dwa_sub_vocabs`
- `choose_cost_partitioned_sub_vocabs`
- `choose_auto_partitioned_sub_vocabs`

### `global_state_map.rs`

- `build_global_max_length_state_map`
- local `use_global_max_length`

### `builder.rs`

- `build_terminal_dwa_with_precomputed_global_max_length`
- `build_terminal_dwa`

## Prominence decisions

The most prominent file is now `mod.rs`, but it contains only names and a map.  The most prominent implementation file is `builder.rs`.  That is intentional: the top-level reader should see the whole construction before seeing any one optimization.

The cost functions and partition thresholds are deliberately less prominent.  They remain available, but only after the reader has understood that they are policy.

The direct and pair builders keep their existing module-level prominence because they correspond to real algorithmic branches.
