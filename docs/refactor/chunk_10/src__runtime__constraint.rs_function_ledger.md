# Function ledger for `src/runtime/constraint.rs`

This ledger is generated for Chunk 10 review. It is not a semantic proof; it is a navigation map for manual inspection.

| line | function signature | reviewer classification |
| --- | --- | --- |
| 18 | `pub fn table_ambiguous_actions(&self) -> Vec<TableAmbiguity> {` | artifact accessor / runtime helper |
| 23 | `pub fn table_has_ambiguity(&self) -> bool {` | artifact accessor / runtime helper |
| 28 | `pub fn terminal_display_names(&self) -> &[String] {` | artifact accessor / runtime helper |
| 33 | `pub fn terminal_display_name(&self, terminal_id: TerminalID) -> Option<&str> {` | artifact accessor / runtime helper |
| 39 | `pub(crate) fn internal_token_materialization_cost(&self, internal_token: usize) -> u64 {` | artifact accessor / runtime helper |
| 52 | `pub(crate) fn estimate_internal_dense_to_buf_cost(&self, dense: &[u64]) -> u64 {` | artifact accessor / runtime helper |
| 126 | `pub(crate) fn apply_internal_dense_delta_to_buf(` | artifact accessor / runtime helper |
| 275 | `fn or_internal_token_masks_to_buf(&self, internal_token: usize, buf: &mut [u32]) {` | artifact accessor / runtime helper |
| 282 | `fn sparse_word_group_entries_in(&self, start: usize, len: usize) -> usize {` | artifact accessor / runtime helper |
| 295 | `fn prefer_dense_buf_scan(buf_words: usize, sparse_entries: usize) -> bool {` | artifact accessor / runtime helper |
| 300 | `fn or_word_group_prefix_diff_to_buf(&self, start: usize, end: usize, buf: &mut [u32]) {` | artifact accessor / runtime helper |
| 327 | `fn andnot_word_group_prefix_diff_from_buf(&self, start: usize, end: usize, buf: &mut [u32]) {` | artifact accessor / runtime helper |
| 353 | `fn or_full_internal_word_run_to_buf(` | artifact accessor / runtime helper |
| 436 | `fn andnot_full_internal_word_run_from_buf(` | artifact accessor / runtime helper |
| 520 | `fn internal_token_buf_op_cost(&self, internal_token: usize, buf_len: usize) -> usize {` | artifact accessor / runtime helper |
| 535 | `fn internal_bits_buf_op_cost(&self, wi: usize, mut bits: u64, buf_len: usize) -> usize {` | artifact accessor / runtime helper |
| 547 | `pub(crate) fn internal_bits_grouped_buf_op_cost(` | artifact accessor / runtime helper |
| 583 | `fn or_internal_token_to_buf_fast(` | artifact accessor / runtime helper |
| 603 | `fn andnot_internal_token_from_buf_fast(` | artifact accessor / runtime helper |
| 622 | `fn or_internal_bits_to_buf_grouped(` | artifact accessor / runtime helper |
| 671 | `fn andnot_internal_bits_from_buf_grouped(` | artifact accessor / runtime helper |
| 726 | `pub(crate) fn or_internal_dense_to_buf(` | artifact accessor / runtime helper |
| 888 | `pub(crate) fn or_internal_dense_to_buf_fast(` | artifact accessor / runtime helper |
| 903 | `fn or_original_token_to_buf(&self, token_id: u32, buf: &mut [u32]) {` | artifact accessor / runtime helper |
| 918 | `fn json_escape_prefix_predicate_matches_supported_short_escapes() {` | artifact accessor / runtime helper |
