# Function ledger for `src/runtime/artifact/token_space.rs`

This ledger is generated for Chunk 10 review. It is not a semantic proof; it is a navigation map for manual inspection.

| line | function signature | reviewer classification |
| --- | --- | --- |
| 40 | `pub(crate) fn can_match_for_state(` | artifact accessor / runtime helper |
| 58 | `pub(crate) fn internal_tsid_for_state(&self, tokenizer_state: u32) -> u32 {` | artifact accessor / runtime helper |
| 65 | `pub(crate) fn internal_token_for_original(&self, token_id: u32) -> u32 {` | artifact accessor / runtime helper |
| 73 | `pub(crate) fn final_internal_token_for_original(&self, token_id: u32) -> Option<u32> {` | artifact accessor / runtime helper |
| 89 | `pub(crate) fn internal_token_universe(&self) -> RangeSetBlaze<u32> {` | artifact accessor / runtime helper |
| 100 | `pub(crate) fn expand_internal_token_set(` | artifact accessor / runtime helper |
| 112 | `pub(crate) fn initial_state_map(&self) -> BTreeMap<u32, ParserGSS> {` | artifact accessor / runtime helper |
| 118 | `fn collect_original_token_ids(&self, internal_tokens: &RangeSetBlaze<u32>) -> Vec<u32> {` | artifact accessor / runtime helper |
| 135 | `fn range_set_from_sorted_ids(ids: &[u32]) -> RangeSetBlaze<u32> {` | artifact accessor / runtime helper |
