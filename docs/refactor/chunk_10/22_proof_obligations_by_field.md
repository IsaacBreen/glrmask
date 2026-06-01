# Proof obligations by `Constraint` field

## `parser_dwa: DWA`

- Classification: semantic artifact field.
- Serialization obligation: must round-trip or be derivable under an explicit compatibility rule.
- Cache obligation: cache rebuild may read this field but must not change its denotation.
- Review question: does any runtime algorithm mutate this field after finalization? If yes, it is misclassified.

## `table: GLRTable`

- Classification: semantic artifact field.
- Serialization obligation: must round-trip or be derivable under an explicit compatibility rule.
- Cache obligation: cache rebuild may read this field but must not change its denotation.
- Review question: does any runtime algorithm mutate this field after finalization? If yes, it is misclassified.

## `terminal_display_names: Vec<String>`

- Classification: semantic artifact field.
- Serialization obligation: must round-trip or be derivable under an explicit compatibility rule.
- Cache obligation: cache rebuild may read this field but must not change its denotation.
- Review question: does any runtime algorithm mutate this field after finalization? If yes, it is misclassified.

## `tokenizer: Tokenizer`

- Classification: semantic artifact field.
- Serialization obligation: must round-trip or be derivable under an explicit compatibility rule.
- Cache obligation: cache rebuild may read this field but must not change its denotation.
- Review question: does any runtime algorithm mutate this field after finalization? If yes, it is misclassified.

## `ignore_terminal: Option<TerminalID>`

- Classification: semantic artifact field.
- Serialization obligation: must round-trip or be derivable under an explicit compatibility rule.
- Cache obligation: cache rebuild may read this field but must not change its denotation.
- Review question: does any runtime algorithm mutate this field after finalization? If yes, it is misclassified.

## `can_match: CanMatchByTerminal`

- Classification: semantic artifact field.
- Serialization obligation: must round-trip or be derivable under an explicit compatibility rule.
- Cache obligation: cache rebuild may read this field but must not change its denotation.
- Review question: does any runtime algorithm mutate this field after finalization? If yes, it is misclassified.

## `state_to_internal_tsid: Vec<u32>`

- Classification: semantic artifact field.
- Serialization obligation: must round-trip or be derivable under an explicit compatibility rule.
- Cache obligation: cache rebuild may read this field but must not change its denotation.
- Review question: does any runtime algorithm mutate this field after finalization? If yes, it is misclassified.

## `internal_tsid_to_states: Vec<Vec<u32>>`

- Classification: semantic artifact field.
- Serialization obligation: must round-trip or be derivable under an explicit compatibility rule.
- Cache obligation: cache rebuild may read this field but must not change its denotation.
- Review question: does any runtime algorithm mutate this field after finalization? If yes, it is misclassified.

## `template_dfas_by_terminal: TemplateDfasByTerminal`

- Classification: semantic artifact field.
- Serialization obligation: must round-trip or be derivable under an explicit compatibility rule.
- Cache obligation: cache rebuild may read this field but must not change its denotation.
- Review question: does any runtime algorithm mutate this field after finalization? If yes, it is misclassified.

## `original_token_to_internal: Vec<u32>`

- Classification: semantic artifact field.
- Serialization obligation: must round-trip or be derivable under an explicit compatibility rule.
- Cache obligation: cache rebuild may read this field but must not change its denotation.
- Review question: does any runtime algorithm mutate this field after finalization? If yes, it is misclassified.

## `internal_token_to_tokens: Vec<Vec<u32>>`

- Classification: semantic artifact field.
- Serialization obligation: must round-trip or be derivable under an explicit compatibility rule.
- Cache obligation: cache rebuild may read this field but must not change its denotation.
- Review question: does any runtime algorithm mutate this field after finalization? If yes, it is misclassified.

## `eos_token_id: Option<u32>`

- Classification: semantic artifact field.
- Serialization obligation: must round-trip or be derivable under an explicit compatibility rule.
- Cache obligation: cache rebuild may read this field but must not change its denotation.
- Review question: does any runtime algorithm mutate this field after finalization? If yes, it is misclassified.

## `json_u_prefix_token_id: Option<u32>`

- Classification: semantic artifact field.
- Serialization obligation: must round-trip or be derivable under an explicit compatibility rule.
- Cache obligation: cache rebuild may read this field but must not change its denotation.
- Review question: does any runtime algorithm mutate this field after finalization? If yes, it is misclassified.

## `json_escape_prefix_buf_mask: Box<[u32]>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `token_bytes: Arc<BTreeMap<u32, Vec<u8>>>`

- Classification: semantic artifact field.
- Serialization obligation: must round-trip or be derivable under an explicit compatibility rule.
- Cache obligation: cache rebuild may read this field but must not change its denotation.
- Review question: does any runtime algorithm mutate this field after finalization? If yes, it is misclassified.

## `internal_token_bytes: BTreeMap<u32, Vec<u8>>`

- Classification: semantic artifact field.
- Serialization obligation: must round-trip or be derivable under an explicit compatibility rule.
- Cache obligation: cache rebuild may read this field but must not change its denotation.
- Review question: does any runtime algorithm mutate this field after finalization? If yes, it is misclassified.

## `token_bytes_dense: Vec<Option<Box<[u8]>>>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `internal_token_buf_masks: Vec<InternalTokenBufMasks>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `word_group_buf_masks: Vec<Box<[u32]>>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `pair_word_group_buf_masks: Vec<Box<[u32]>>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `quad_word_group_buf_masks: Vec<Box<[u32]>>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `super_word_group_buf_masks: Vec<Box<[u32]>>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `mega_word_group_buf_masks: Vec<Box<[u32]>>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `giga_word_group_buf_masks: Vec<Box<[u32]>>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `word_group_sparse_masks: Vec<InternalTokenBufMasks>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `word_group_prefix_buf_masks: Vec<Box<[u32]>>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `word_group_sparse_prefix_entries: Vec<usize>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `quad_group_sparse_masks: Vec<InternalTokenBufMasks>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `byte_group_sparse_masks: Vec<InternalTokenBufMasks>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `word_group_sparse_total_entries: usize`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `word_group_sparse_max_entries: usize`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `all_tokens_buf_mask: Box<[u32]>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `internal_token_dense_words: usize`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `weight_token_dense_masks: DenseWeightMaskCache`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `weight_token_buf_masks: DenseWeightBufMaskCache`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `weight_token_sparse_buf_masks: SparseWeightBufMaskCache`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `seed_terminal_dense: SeedTerminalDenseMasks`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `seed_state_dense: SeedStateDenseMasks`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `seed_state_by_dense_hash: FxHashMap<u64, Vec<usize>>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `seed_state_buf_masks: SeedStateBufMasks`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `seed_universe_dense: DenseWords`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `dwa_fast_transitions: FastDwaTransitions`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `tokenizer_fast_transitions: FastTokenizerTransitions`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `heavy_token_dense_masks: Vec<Option<Box<[u32]>>>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `internal_token_buf_flat: Box<[(u16, u32)]>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `internal_token_buf_offsets: Box<[u32]>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `total_internal_buf_cost: usize`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `heavy_token_indices: Vec<usize>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `heavy_total_cost: usize`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `light_avg_cost_x256: usize`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `internal_token_buf_op_costs: Vec<usize>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `word_group_buf_op_costs: Vec<usize>`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

## `final_mask_mapping: FinalMaskMapping`

- Classification: derived runtime cache.
- Serialization obligation: should be skipped or safe to rebuild.
- Rebuild obligation: must be a deterministic function of semantic fields and previous cache phases.
- Review question: if this field were deleted before serialization, could `load` reconstruct equivalent Mask/Commit behavior? If no, it is misclassified.

