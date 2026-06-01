# Constraint field taxonomy

This table classifies every field in `src/runtime/artifact/compiled.rs`.

| field | type | classification |
| --- | --- | --- |
| `parser_dwa` | `DWA` | semantic/serialized |
| `table` | `GLRTable` | semantic/serialized |
| `terminal_display_names` | `Vec<String>` | semantic/serialized |
| `tokenizer` | `Tokenizer` | semantic/serialized |
| `ignore_terminal` | `Option<TerminalID>` | semantic/serialized |
| `can_match` | `CanMatchByTerminal` | semantic/serialized |
| `state_to_internal_tsid` | `Vec<u32>` | semantic/serialized |
| `internal_tsid_to_states` | `Vec<Vec<u32>>` | semantic/serialized |
| `template_dfas_by_terminal` | `TemplateDfasByTerminal` | semantic/serialized |
| `original_token_to_internal` | `Vec<u32>` | semantic/serialized |
| `internal_token_to_tokens` | `Vec<Vec<u32>>` | semantic/serialized |
| `eos_token_id` | `Option<u32>` | semantic/serialized |
| `json_u_prefix_token_id` | `Option<u32>` | semantic/serialized |
| `json_escape_prefix_buf_mask` | `Box<[u32]>` | derived runtime cache |
| `token_bytes` | `Arc<BTreeMap<u32, Vec<u8>>>` | semantic/serialized |
| `internal_token_bytes` | `BTreeMap<u32, Vec<u8>>` | semantic/serialized |
| `token_bytes_dense` | `Vec<Option<Box<[u8]>>>` | derived runtime cache |
| `internal_token_buf_masks` | `Vec<InternalTokenBufMasks>` | derived runtime cache |
| `word_group_buf_masks` | `Vec<Box<[u32]>>` | derived runtime cache |
| `pair_word_group_buf_masks` | `Vec<Box<[u32]>>` | derived runtime cache |
| `quad_word_group_buf_masks` | `Vec<Box<[u32]>>` | derived runtime cache |
| `super_word_group_buf_masks` | `Vec<Box<[u32]>>` | derived runtime cache |
| `mega_word_group_buf_masks` | `Vec<Box<[u32]>>` | derived runtime cache |
| `giga_word_group_buf_masks` | `Vec<Box<[u32]>>` | derived runtime cache |
| `word_group_sparse_masks` | `Vec<InternalTokenBufMasks>` | derived runtime cache |
| `word_group_prefix_buf_masks` | `Vec<Box<[u32]>>` | derived runtime cache |
| `word_group_sparse_prefix_entries` | `Vec<usize>` | derived runtime cache |
| `quad_group_sparse_masks` | `Vec<InternalTokenBufMasks>` | derived runtime cache |
| `byte_group_sparse_masks` | `Vec<InternalTokenBufMasks>` | derived runtime cache |
| `word_group_sparse_total_entries` | `usize` | derived runtime cache |
| `word_group_sparse_max_entries` | `usize` | derived runtime cache |
| `all_tokens_buf_mask` | `Box<[u32]>` | derived runtime cache |
| `internal_token_dense_words` | `usize` | derived runtime cache |
| `weight_token_dense_masks` | `DenseWeightMaskCache` | derived runtime cache |
| `weight_token_buf_masks` | `DenseWeightBufMaskCache` | derived runtime cache |
| `weight_token_sparse_buf_masks` | `SparseWeightBufMaskCache` | derived runtime cache |
| `seed_terminal_dense` | `SeedTerminalDenseMasks` | derived runtime cache |
| `seed_state_dense` | `SeedStateDenseMasks` | derived runtime cache |
| `seed_state_by_dense_hash` | `FxHashMap<u64, Vec<usize>>` | derived runtime cache |
| `seed_state_buf_masks` | `SeedStateBufMasks` | derived runtime cache |
| `seed_universe_dense` | `DenseWords` | derived runtime cache |
| `dwa_fast_transitions` | `FastDwaTransitions` | derived runtime cache |
| `tokenizer_fast_transitions` | `FastTokenizerTransitions` | derived runtime cache |
| `heavy_token_dense_masks` | `Vec<Option<Box<[u32]>>>` | derived runtime cache |
| `internal_token_buf_flat` | `Box<[(u16, u32)]>` | derived runtime cache |
| `internal_token_buf_offsets` | `Box<[u32]>` | derived runtime cache |
| `total_internal_buf_cost` | `usize` | derived runtime cache |
| `heavy_token_indices` | `Vec<usize>` | derived runtime cache |
| `heavy_total_cost` | `usize` | derived runtime cache |
| `light_avg_cost_x256` | `usize` | derived runtime cache |
| `internal_token_buf_op_costs` | `Vec<usize>` | derived runtime cache |
| `word_group_buf_op_costs` | `Vec<usize>` | derived runtime cache |
| `final_mask_mapping` | `FinalMaskMapping` | derived runtime cache |
