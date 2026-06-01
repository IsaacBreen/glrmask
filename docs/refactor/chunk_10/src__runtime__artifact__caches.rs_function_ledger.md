# Function ledger for `src/runtime/artifact/caches.rs`

This ledger is generated for Chunk 10 review. It is not a semantic proof; it is a navigation map for manual inspection.

| line | function signature | reviewer classification |
| --- | --- | --- |
| 34 | `pub(crate) fn rebuild_runtime_caches_impl(&mut self) {` | cache rebuild |
| 204 | `fn compute_tokenizer_fast_transitions(&self) -> FastTokenizerTransitions {` | cache rebuild |
| 220 | `fn compute_buf_masks(&self) -> Vec<InternalTokenBufMasks> {` | cache rebuild |
| 261 | `fn compute_token_block_sparse_masks(&self, block_size: usize) -> (Vec<InternalTokenBufMasks>, usize, usize) {` | cache rebuild |
| 297 | `fn compute_sliding_word_group_dense_masks(&self, word_group_len: usize) -> Vec<Box<[u32]>> {` | cache rebuild |
| 321 | `fn compute_all_tokens_buf_mask(&self) -> Box<[u32]> {` | cache rebuild |
| 332 | `pub(crate) fn token_starts_json_escape_prefix(bytes: &[u8]) -> bool {` | cache rebuild |
| 338 | `fn compute_json_escape_prefix_buf_mask(&self) -> Box<[u32]> {` | cache rebuild |
| 352 | `fn compute_word_group_prefix_buf_masks(&self) -> Vec<Box<[u32]>> {` | cache rebuild |
| 366 | `fn compute_sparse_entry_prefix(groups: &[InternalTokenBufMasks]) -> Vec<usize> {` | cache rebuild |
| 377 | `fn dense_words_hash(words: &[u64]) -> u64 {` | cache rebuild |
| 386 | `fn compute_seed_state_hashes(` | cache rebuild |
| 399 | `pub(crate) fn seed_state_index_for_dense(&self, dense: &[u64]) -> Option<usize> {` | cache rebuild |
| 409 | `pub(crate) fn or_seed_state_dense_to_buf(&self, seed_idx: usize, buf: &mut [u32]) -> bool {` | cache rebuild |
| 418 | `pub(crate) fn has_seed_state_buf_mask(&self, seed_idx: usize) -> bool {` | cache rebuild |
| 424 | `pub(crate) fn or_seed_dense_token_set_to_buf(` | cache rebuild |
| 508 | `pub(crate) fn or_weight_token_set_to_buf_if_contained(` | cache rebuild |
| 538 | `pub(crate) fn or_dense_token_set_to_buf_sparse(` | cache rebuild |
| 583 | `pub(crate) fn has_weight_token_set_buf_if_contained(` | cache rebuild |
| 606 | `fn compute_weight_token_buf_masks(&self) -> DenseWeightBufMaskCache {` | cache rebuild |
| 627 | `fn dense_buf_to_sparse_entries(buf: &[u32]) -> Box<[(u16, u32)]> {` | cache rebuild |
| 641 | `fn compute_weight_token_sparse_buf_masks(&self) -> SparseWeightBufMaskCache {` | cache rebuild |
| 667 | `fn compute_seed_state_buf_masks(&self) -> SeedStateBufMasks {` | cache rebuild |
| 694 | `fn compute_heavy_token_dense_masks(&self) -> Vec<Option<Box<[u32]>>> {` | cache rebuild |
| 725 | `fn compute_flat_buf_masks(masks: &[InternalTokenBufMasks]) -> (Box<[(u16, u32)]>, Box<[u32]>) {` | cache rebuild |
| 739 | `fn compute_total_internal_buf_cost(` | cache rebuild |
| 756 | `fn compute_internal_token_buf_op_costs(` | cache rebuild |
| 773 | `fn compute_word_group_buf_op_costs(costs: &[usize]) -> Vec<usize> {` | cache rebuild |
| 780 | `fn compute_dense_token_bytes(&self) -> Vec<Option<Box<[u8]>>> {` | cache rebuild |
| 792 | `fn compute_fast_transitions(&self) -> FastDwaTransitions {` | cache rebuild |
| 807 | `fn compute_dense_token_masks(&self) -> (usize, DenseWeightMaskCache) {` | cache rebuild |
| 839 | `pub(crate) fn build_buf_masks(&mut self) {` | cache rebuild |
| 873 | `pub(crate) fn build_dense_token_bytes(&mut self) {` | cache rebuild |
| 878 | `pub(crate) fn build_fast_transitions(&mut self) {` | cache rebuild |
| 882 | `pub(crate) fn build_dense_token_masks(&mut self) {` | cache rebuild |
| 891 | `pub(crate) fn build_seed_dense_masks(&mut self) {` | cache rebuild |
| 917 | `fn extend_seed_state_dense_with_single_terminal_exclusions(&mut self) {` | cache rebuild |
| 961 | `fn collect_weight_token_sets<'a>(` | cache rebuild |
| 971 | `fn final_weight_token_dense_masks(&self) -> Vec<(&usize, &DenseWords)> {` | cache rebuild |
| 995 | `fn dense_words_from_internal_set_with_words(` | cache rebuild |
| 1010 | `fn dense_words_from_internal_set(&self, internal_tokens: &RangeSetBlaze<u32>) -> DenseWords {` | cache rebuild |
| 1021 | `fn build_internal_token_buf_mask(originals: &[u32]) -> InternalTokenBufMasks {` | cache rebuild |
| 1052 | `fn build_internal_token_buf_mask_unsorted(originals: &[u32]) -> InternalTokenBufMasks {` | cache rebuild |
| 1062 | `fn build_seed_terminal_dense_masks(&self) -> SeedTerminalDenseMasks {` | cache rebuild |
| 1088 | `fn precompute_node_reachable_dense(` | cache rebuild |
| 1117 | `fn walk_seed_trie(` | cache rebuild |
| 1165 | `fn build_seed_state_dense_masks(&self) -> SeedStateDenseMasks {` | cache rebuild |
