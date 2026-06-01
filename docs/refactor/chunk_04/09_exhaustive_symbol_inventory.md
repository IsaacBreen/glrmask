# Exhaustive Terminal-DWA file and symbol inventory after Chunk 04

This file is generated from the patched source tree.  It is intentionally mechanical: every listed symbol is a review handle.  The point is not to explain every implementation detail perfectly; the point is to make the cleanup surface concrete enough that a reviewer can walk the subsystem file by file.

## File metrics

| File | LOC | Symbol count | Intended responsibility |
|---|---:|---:|---|
| `src/compile/terminal_dwa/builder.rs` | 274 | 2 | top-level orchestration |
| `src/compile/terminal_dwa/classify.rs` | 707 | 35 | terminal-path and token-byte classification |
| `src/compile/terminal_dwa/direct_partition/max_length.rs` | 434 | 14 | direct/single-step local construction internals |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 2338 | 57 | direct/single-step local construction internals |
| `src/compile/terminal_dwa/global_state_map.rs` | 79 | 2 | global tokenizer-state quotient |
| `src/compile/terminal_dwa/grammar_helpers.rs` | 264 | 7 | supporting terminal-DWA helper |
| `src/compile/terminal_dwa/merge.rs` | 756 | 15 | local/global id-map and DWA reconciliation |
| `src/compile/terminal_dwa/mod.rs` | 44 | 0 | subsystem boundary and re-exports |
| `src/compile/terminal_dwa/options.rs` | 114 | 13 | typed build policy / environment boundary |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/combined.rs` | 522 | 16 | pair/multi-step local construction internals |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs` | 295 | 20 | pair/multi-step local construction internals |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/disallowed_follows.rs` | 60 | 2 | pair/multi-step local construction internals |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/mod.rs` | 9 | 0 | pair/multi-step local construction internals |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/shared.rs` | 75 | 5 | pair/multi-step local construction internals |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 1051 | 31 | pair/multi-step local construction internals |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 583 | 26 | pair/multi-step local construction internals |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/mod.rs` | 7 | 0 | pair/multi-step local construction internals |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 492 | 22 | pair/multi-step local construction internals |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/mod.rs` | 96 | 2 | pair/multi-step local construction internals |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/pipeline.rs` | 171 | 13 | pair/multi-step local construction internals |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 2455 | 72 | pair/multi-step local construction internals |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/mod.rs` | 6 | 0 | pair/multi-step local construction internals |
| `src/compile/terminal_dwa/pair_partition/mod.rs` | 787 | 16 | pair/multi-step local construction internals |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 1031 | 48 | top-level orchestration |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 718 | 21 | pair/multi-step local construction internals |
| `src/compile/terminal_dwa/partition.rs` | 193 | 1 | single sub-vocab direct/pair local build |
| `src/compile/terminal_dwa/types.rs` | 86 | 13 | small shared types and profile counters |
| `src/compile/terminal_dwa/vocab_partition.rs` | 273 | 8 | caller vocabulary partition selection |

## `src/compile/terminal_dwa/builder.rs`

LOC: 274.  Symbols detected: 2.

| Line | Symbol | Review note |
|---:|---|---|
| 71 | `pub(crate) fn build_terminal_dwa_with_precomputed_global_max_length(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 236 | `pub(crate) fn build_terminal_dwa(` | builder/constructor; verify inputs and returned id-map witness are documented |

## `src/compile/terminal_dwa/classify.rs`

LOC: 707.  Symbols detected: 35.

| Line | Symbol | Review note |
|---:|---|---|
| 18 | `pub struct SharedClassifyBytesets {` | data witness/cache/profile; check whether name says what invariant it carries |
| 25 | `pub type SharedClassifyCache = std::sync::OnceLock<SharedClassifyBytesets>;` | type alias; ensure alias clarifies coordinate space |
| 28 | `struct VocabByteSet {` | data witness/cache/profile; check whether name says what invariant it carries |
| 32 | `impl crate::vocab::VocabDerivedArtifact for VocabByteSet {}` | implementation block; review whether methods belong with the type |
| 34 | `fn vocab_byte_set(vocab: &Vocab) -> U8Set {` | helper; check whether it belongs in current file or a later split |
| 49 | `pub(crate) fn prepare_vocab_for_terminal_classification(vocab: &Vocab) {` | helper; check whether it belongs in current file or a later split |
| 54 | `pub(crate) enum PairPartitionCostFn {` | closed choice; prefer this over raw strings where policy is internal |
| 61 | `impl PairPartitionCostFn {` | implementation block; review whether methods belong with the type |
| 62 | `pub(crate) fn as_str(self) -> &'static str {` | helper; check whether it belongs in current file or a later split |
| 73 | `pub(crate) enum PairPartitionObjective {` | closed choice; prefer this over raw strings where policy is internal |
| 78 | `impl PairPartitionObjective {` | implementation block; review whether methods belong with the type |
| 79 | `pub(crate) fn as_str(self) -> &'static str {` | helper; check whether it belongs in current file or a later split |
| 87 | `pub(crate) struct PairPartitionCostPartitioning {` | data witness/cache/profile; check whether name says what invariant it carries |
| 95 | `struct PairPartitionTokenGroup {` | data witness/cache/profile; check whether name says what invariant it carries |
| 101 | `struct PairPartitionBucket {` | data witness/cache/profile; check whether name says what invariant it carries |
| 107 | `impl PairPartitionBucket {` | implementation block; review whether methods belong with the type |
| 108 | `fn new() -> Self {` | helper; check whether it belongs in current file or a later split |
| 116 | `fn size(&self) -> usize {` | helper; check whether it belongs in current file or a later split |
| 120 | `fn pair_partition_count(&self) -> usize {` | helper; check whether it belongs in current file or a later split |
| 125 | `impl SharedClassifyBytesets {` | implementation block; review whether methods belong with the type |
| 131 | `pub fn build(tokenizer: &Tokenizer, num_terminals: u32) -> Self {` | helper; check whether it belongs in current file or a later split |
| 227 | `pub(crate) fn classify_vocab_char_type(bytes: &[u8]) -> u8 {` | helper; check whether it belongs in current file or a later split |
| 283 | `fn classify_nonalnum(bytes: &[u8]) -> u8 {` | helper; check whether it belongs in current file or a later split |
| 308 | `pub(crate) fn classify_terminal_path_lengths(` | helper; check whether it belongs in current file or a later split |
| 369 | `fn build_byte_terminal_reverse_index(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 388 | `fn token_pair_partition_terminals(` | helper; check whether it belongs in current file or a later split |
| 425 | `fn compute_partition_cost(` | helper; check whether it belongs in current file or a later split |
| 444 | `fn partition_metric_count(` | helper; check whether it belongs in current file or a later split |
| 457 | `fn objective_score(objective: PairPartitionObjective, costs: &[f64]) -> f64 {` | helper; check whether it belongs in current file or a later split |
| 464 | `fn compute_token_pair_partition_map(` | helper; check whether it belongs in current file or a later split |
| 490 | `pub(crate) fn partition_vocab_char_type_tokens(vocab: &Vocab) -> Vec<Vec<u32>> {` | helper; check whether it belongs in current file or a later split |
| 499 | `pub(crate) fn estimate_pair_partition_objective_for_token_partitions(` | helper; check whether it belongs in current file or a later split |
| 546 | `fn partition_token_pair_partition_map_by_cost(` | helper; check whether it belongs in current file or a later split |
| 673 | `pub(crate) fn partition_vocab_by_pair_partition_cost_with_token_map(` | helper; check whether it belongs in current file or a later split |
| 688 | `pub(crate) fn partition_vocab_by_pair_partition_cost(` | helper; check whether it belongs in current file or a later split |

## `src/compile/terminal_dwa/direct_partition/max_length.rs`

LOC: 434.  Symbols detected: 14.

| Line | Symbol | Review note |
|---:|---|---|
| 12 | `fn mix_u64(mut x: u64) -> u64 {` | helper; check whether it belongs in current file or a later split |
| 22 | `fn hash_filtered_sorted_set(values: &[usize], active_groups: Option<&[bool]>, tag: u64) -> u64 {` | helper; check whether it belongs in current file or a later split |
| 38 | `fn hash_state_label(state: &FlatDfaState, active_groups: Option<&[bool]>) -> u64 {` | helper; check whether it belongs in current file or a later split |
| 49 | `fn hash_transition_labels(label_hashes: &[u64]) -> u64 {` | helper; check whether it belongs in current file or a later split |
| 58 | `fn hash_transition_targets(targets: &[usize], prev_hashes: &[u64]) -> u64 {` | helper; check whether it belongs in current file or a later split |
| 66 | `fn build_state_shape(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 93 | `fn build_subset_mapping(states: &[usize], hashes: &[u64]) -> Vec<usize> {` | builder/constructor; verify inputs and returned id-map witness are documented |
| 119 | `fn count_distinct_hashes(hashes: &[u64]) -> usize {` | helper; check whether it belongs in current file or a later split |
| 127 | `fn find_state_equivalence_classes_kstep(` | helper; check whether it belongs in current file or a later split |
| 181 | `fn cheap_state_hash(` | helper; check whether it belongs in current file or a later split |
| 199 | `pub fn find_state_equivalence_classes<S: AsRef<[u8]>>(` | helper; check whether it belongs in current file or a later split |
| 223 | `fn build_state_shape_restricted(dfa: &FlatDfa, state_idx: usize, relevant_bytes: &[bool; 256]) -> (Vec<usize>, u64) {` | builder/constructor; verify inputs and returned id-map witness are documented |
| 246 | `fn find_state_equivalence_classes_kstep_restricted(` | helper; check whether it belongs in current file or a later split |
| 403 | `pub fn find_state_equivalence_classes_byte_restricted<S: AsRef<[u8]>>(` | helper; check whether it belongs in current file or a later split |

## `src/compile/terminal_dwa/direct_partition/mod.rs`

LOC: 2338.  Symbols detected: 57.

| Line | Symbol | Review note |
|---:|---|---|
| 18 | `struct PreHashedRanges {` | data witness/cache/profile; check whether name says what invariant it carries |
| 24 | `struct DirectPartitionIdentityVocabOrder {` | data witness/cache/profile; check whether name says what invariant it carries |
| 31 | `impl crate::vocab::VocabDerivedArtifact for DirectPartitionIdentityVocabOrder {}` | implementation block; review whether methods belong with the type |
| 33 | `fn direct_partition_identity_vocab_order(vocab: &Vocab) -> Arc<DirectPartitionIdentityVocabOrder> {` | helper; check whether it belongs in current file or a later split |
| 73 | `pub(crate) fn prepare_direct_partition_identity_vocab_order(vocab: &Vocab) {` | helper; check whether it belongs in current file or a later split |
| 77 | `fn skip_max_length_for_partition(partition_label: &str) -> bool {` | helper; check whether it belongs in current file or a later split |
| 100 | `fn skip_direct_partition_max_length_for_partition(partition_label: &str) -> bool {` | helper; check whether it belongs in current file or a later split |
| 123 | `fn direct_partition_max_length_min_states() -> usize {` | helper; check whether it belongs in current file or a later split |
| 134 | `fn should_skip_max_length_for_partition(` | helper; check whether it belongs in current file or a later split |
| 145 | `fn fast_projected_direct_partition_id_map_enabled() -> bool {` | configuration boundary; should remain outside denotation docs |
| 157 | `fn fast_projected_direct_partition_id_map_max_tsids() -> usize {` | helper; check whether it belongs in current file or a later split |
| 168 | `fn should_use_fast_projected_direct_partition_id_map(` | helper; check whether it belongs in current file or a later split |
| 186 | `fn range_hash_val(s: u32, e: u32) -> u64 {` | helper; check whether it belongs in current file or a later split |
| 191 | `impl PreHashedRanges {` | implementation block; review whether methods belong with the type |
| 192 | `fn new(ranges: Vec<(u32, u32)>) -> Self {` | helper; check whether it belongs in current file or a later split |
| 202 | `impl PartialEq for PreHashedRanges {` | implementation block; review whether methods belong with the type |
| 203 | `fn eq(&self, other: &Self) -> bool {` | helper; check whether it belongs in current file or a later split |
| 208 | `impl Eq for PreHashedRanges {}` | implementation block; review whether methods belong with the type |
| 210 | `impl Hash for PreHashedRanges {` | implementation block; review whether methods belong with the type |
| 211 | `fn hash<H: Hasher>(&self, state: &mut H) {` | helper; check whether it belongs in current file or a later split |
| 224 | `struct LazyRanges<'a> {` | data witness/cache/profile; check whether name says what invariant it carries |
| 231 | `fn new(refs: Vec<&'a [(u32, u32)]>) -> Self {` | helper; check whether it belongs in current file or a later split |
| 270 | `fn materialize(&self) -> Vec<(u32, u32)> {` | helper; check whether it belongs in current file or a later split |
| 291 | `fn eq(&self, other: &Self) -> bool {` | helper; check whether it belongs in current file or a later split |
| 325 | `fn compact_direct_partition_terminal_dwa_enabled() -> bool {` | configuration boundary; should remain outside denotation docs |
| 349 | `pub(crate) fn count_direct_partition_equivalence_classes(` | helper; check whether it belongs in current file or a later split |
| 403 | `pub(crate) fn build_direct_partition_terminal_dwa(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 549 | `fn build_direct_partition_id_map<'a>(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 776 | `fn build_direct_partition_identity_vocab_map(vocab: &Vocab) -> (ManyToOneIdMap, Arc<DirectPartitionIdentityVocabOrder>, f64) {` | builder/constructor; verify inputs and returned id-map witness are documented |
| 793 | `fn state_to_representative_vector(state_map: &ManyToOneIdMap, num_dfa_states: usize) -> Vec<u32> {` | helper; check whether it belongs in current file or a later split |
| 805 | `struct TokenLengthStats {` | data witness/cache/profile; check whether name says what invariant it carries |
| 813 | `fn token_length_stats(tokens: &[&[u8]]) -> TokenLengthStats {` | helper; check whether it belongs in current file or a later split |
| 842 | `fn token_length_stats_from_entries(tokens: &[(u32, Arc<[u8]>)]) -> TokenLengthStats {` | helper; check whether it belongs in current file or a later split |
| 871 | `fn find_direct_partition_exact_state_equivalence_by_token_signatures(` | helper; check whether it belongs in current file or a later split |
| 1004 | `fn direct_partition_bucket_suffix_signature_profile(` | helper; check whether it belongs in current file or a later split |
| 1049 | `struct DirectPartitionSortedTokenBuckets {` | data witness/cache/profile; check whether name says what invariant it carries |
| 1058 | `fn build_direct_partition_sorted_token_buckets(sorted_entries: &[(u32, Arc<[u8]>)]) -> DirectPartitionSortedTokenBuckets {` | builder/constructor; verify inputs and returned id-map witness are documented |
| 1110 | `fn collect_active_terminal_signature(` | helper; check whether it belongs in current file or a later split |
| 1131 | `fn build_direct_partition_state_to_terminal_signature(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 1154 | `fn direct_partition_token_signature_profile_for_state(` | helper; check whether it belongs in current file or a later split |
| 1215 | `fn append_direct_partition_signature_profile_run(profile: &mut Vec<(u32, u32, u32)>, sig_id: u32, token_id: u32) {` | helper; check whether it belongs in current file or a later split |
| 1225 | `fn build_direct_partition_terminal_dwa(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 2007 | `pub(crate) fn build_flat_transition_table(tokenizer: &Tokenizer) -> Vec<u32> {` | builder/constructor; verify inputs and returned id-map witness are documented |
| 2019 | `fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {` | helper; check whether it belongs in current file or a later split |
| 2028 | `fn append_token_id_range(token_ranges: &mut Vec<(u32, u32)>, token_id: u32) {` | helper; check whether it belongs in current file or a later split |
| 2032 | `fn append_token_id_span(token_ranges: &mut Vec<(u32, u32)>, start: u32, end: u32) {` | helper; check whether it belongs in current file or a later split |
| 2042 | `fn flush_end_rep_run(` | helper; check whether it belongs in current file or a later split |
| 2057 | `fn collect_direct_partition_root_ranges_by_first_byte_lcp(` | helper; check whether it belongs in current file or a later split |
| 2121 | `fn merge_ranges_in_place(ranges: &mut Vec<(u32, u32)>) {` | helper; check whether it belongs in current file or a later split |
| 2139 | `fn shared_rangeset_from_unsorted_pairs(ranges: &[(u32, u32)]) -> Option<Arc<RangeSetBlaze<u32>>> {` | helper; check whether it belongs in current file or a later split |
| 2151 | `fn build_end_rep_group_masks(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 2171 | `fn merge_deferred_equivalent_tsids(` | helper; check whether it belongs in current file or a later split |
| 2247 | `fn remap_deferred_arced_tsids(` | helper; check whether it belongs in current file or a later split |
| 2278 | `fn apply_tsid_perm_to_id_map(id_map: &mut ManyToOneIdMap, perm: &[u32], new_count: usize) {` | helper; check whether it belongs in current file or a later split |
| 2302 | `struct DirectPartitionIdMapProfile {` | data witness/cache/profile; check whether name says what invariant it carries |
| 2319 | `struct DirectPartitionTsidProfileMergeReport {` | data witness/cache/profile; check whether name says what invariant it carries |
| 2328 | `struct DirectPartitionTerminalBuildProfile {` | data witness/cache/profile; check whether name says what invariant it carries |

## `src/compile/terminal_dwa/global_state_map.rs`

LOC: 79.  Symbols detected: 2.

| Line | Symbol | Review note |
|---:|---|---|
| 22 | `fn use_global_max_length(tokenizer: &Tokenizer) -> bool {` | helper; check whether it belongs in current file or a later split |
| 29 | `pub(crate) fn build_global_max_length_state_map(` | builder/constructor; verify inputs and returned id-map witness are documented |

## `src/compile/terminal_dwa/grammar_helpers.rs`

LOC: 264.  Symbols detected: 7.

| Line | Symbol | Review note |
|---:|---|---|
| 13 | `pub(crate) fn compute_terminal_coloring(table: &GLRTable) -> TerminalColoring {` | helper; check whether it belongs in current file or a later split |
| 112 | `fn assert_row_colors_are_unique(table: &GLRTable, coloring: &TerminalColoring) {` | helper; check whether it belongs in current file or a later split |
| 128 | `fn terminal_coloring_keeps_action_row_terminals_distinct() {` | helper; check whether it belongs in current file or a later split |
| 147 | `fn terminal_coloring_handles_sparse_high_terminal_count() {` | helper; check whether it belongs in current file or a later split |
| 168 | `pub(crate) fn compute_ever_allowed_follows(grammar: &AnalyzedGrammar) -> Vec<Vec<TerminalID>> {` | helper; check whether it belongs in current file or a later split |
| 191 | `pub(crate) fn compute_always_allowed_follows(grammar: &AnalyzedGrammar) -> Vec<Vec<TerminalID>> {` | helper; check whether it belongs in current file or a later split |
| 217 | `fn occurrence_follow_set(` | helper; check whether it belongs in current file or a later split |

## `src/compile/terminal_dwa/merge.rs`

LOC: 756.  Symbols detected: 15.

| Line | Symbol | Review note |
|---:|---|---|
| 22 | `fn minimize_merged_terminal_dwa_enabled() -> bool {` | configuration boundary; should remain outside denotation docs |
| 34 | `fn compact_merged_terminal_dwa_enabled() -> bool {` | configuration boundary; should remain outside denotation docs |
| 47 | `pub(crate) fn merge_local_id_maps_and_terminal_dwas(` | helper; check whether it belongs in current file or a later split |
| 103 | `pub(crate) fn merge_id_maps_and_terminal_dwas(` | helper; check whether it belongs in current file or a later split |
| 257 | `fn build_unified_global_id_map(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 300 | `fn build_unified_global_token_id_map_generic(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 352 | `fn build_unified_global_token_id_map_disjoint(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 406 | `fn build_direct_local_to_global_token_map(local_to_global: &[u32]) -> Vec<Vec<u32>> {` | builder/constructor; verify inputs and returned id-map witness are documented |
| 419 | `fn reorder_classes(` | helper; check whether it belongs in current file or a later split |
| 452 | `fn reorder_classes_with_sentinel(` | helper; check whether it belongs in current file or a later split |
| 489 | `fn build_local_to_global_tsid_map(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 513 | `fn build_local_to_global_token_map(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 548 | `fn remap_nwa_with_maps(` | helper; check whether it belongs in current file or a later split |
| 597 | `fn remap_weight_cached(` | helper; check whether it belongs in current file or a later split |
| 618 | `fn remap_weight_general(` | helper; check whether it belongs in current file or a later split |

## `src/compile/terminal_dwa/mod.rs`

LOC: 44.  Symbols detected: 0.

No top-level symbols detected by the simple inventory regex.

## `src/compile/terminal_dwa/options.rs`

LOC: 114.  Symbols detected: 13.

| Line | Symbol | Review note |
|---:|---|---|
| 15 | `pub(crate) enum VocabPartitionScheme {` | closed choice; prefer this over raw strings where policy is internal |
| 25 | `impl VocabPartitionScheme {` | implementation block; review whether methods belong with the type |
| 26 | `pub(crate) fn as_str(self) -> &'static str {` | helper; check whether it belongs in current file or a later split |
| 35 | `fn parse_truthy(value: &str) -> bool {` | helper; check whether it belongs in current file or a later split |
| 40 | `pub(crate) fn vocab_partition_scheme_from_env() -> VocabPartitionScheme {` | configuration boundary; should remain outside denotation docs |
| 51 | `pub(crate) fn pair_partition_cost_fn_from_env() -> PairPartitionCostFn {` | configuration boundary; should remain outside denotation docs |
| 63 | `pub(crate) fn pair_partition_objective_from_env() -> PairPartitionObjective {` | configuration boundary; should remain outside denotation docs |
| 73 | `pub(crate) fn pair_partition_count_from_env() -> usize {` | configuration boundary; should remain outside denotation docs |
| 81 | `pub(crate) fn pair_partition_auto_second_largest_limit_from_env() -> usize {` | configuration boundary; should remain outside denotation docs |
| 89 | `pub(crate) fn pair_partition_auto_max_estimated_pair_partition_terminals_from_env() -> usize {` | configuration boundary; should remain outside denotation docs |
| 97 | `pub(crate) fn pair_partition_auto_min_estimated_pair_partition_terminals_from_env() -> usize {` | configuration boundary; should remain outside denotation docs |
| 104 | `pub(crate) fn pair_partition_auto_min_grammar_terminals_from_env() -> usize {` | configuration boundary; should remain outside denotation docs |
| 111 | `pub(crate) fn global_max_length_env_override() -> Option<bool> {` | configuration boundary; should remain outside denotation docs |

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/combined.rs`

LOC: 522.  Symbols detected: 16.

| Line | Symbol | Review note |
|---:|---|---|
| 24 | `fn deduplicate_tokens_by_byte_class<'a, S: AsRef<[u8]>>(` | helper; check whether it belongs in current file or a later split |
| 50 | `fn adjust_disallowed_follows(` | helper; check whether it belongs in current file or a later split |
| 66 | `fn build_state_map(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 92 | `fn build_state_map_composed(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 137 | `fn build_vocab_map(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 178 | `struct PreparedEquivalenceInputs<'a> {` | data witness/cache/profile; check whether name says what invariant it carries |
| 185 | `fn prepare_equivalence_inputs<'a>(` | helper; check whether it belongs in current file or a later split |
| 212 | `struct CombinedEquivalenceResult {` | data witness/cache/profile; check whether name says what invariant it carries |
| 217 | `pub(crate) struct CombinedEquivalenceProfile {` | data witness/cache/profile; check whether name says what invariant it carries |
| 238 | `struct TokenLengthStats {` | data witness/cache/profile; check whether name says what invariant it carries |
| 246 | `fn skip_max_length_for_partition(partition_label: &str) -> bool {` | helper; check whether it belongs in current file or a later split |
| 270 | `fn should_skip_max_length_for_partition(` | helper; check whether it belongs in current file or a later split |
| 282 | `fn token_length_stats(tokens: &[&[u8]]) -> TokenLengthStats {` | helper; check whether it belongs in current file or a later split |
| 311 | `fn build_internal_id_map_from_combined_result(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 330 | `pub(crate) fn analyze_equivalences_with_group_filter(` | helper; check whether it belongs in current file or a later split |
| 348 | `fn analyze_equivalences_impl(` | helper; check whether it belongs in current file or a later split |

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs`

LOC: 295.  Symbols detected: 20.

| Line | Symbol | Review note |
|---:|---|---|
| 7 | `fn build_transition_table(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 17 | `fn collect_group_ids(groups: impl Iterator<Item = usize>) -> Vec<usize> {` | helper; check whether it belongs in current file or a later split |
| 24 | `pub struct FlatDfaState {` | data witness/cache/profile; check whether name says what invariant it carries |
| 35 | `pub struct FlatDfa {` | data witness/cache/profile; check whether name says what invariant it carries |
| 43 | `pub(crate) fn compute_byte_classes(dfa: &FlatDfa) -> [u8; 256] {` | helper; check whether it belongs in current file or a later split |
| 98 | `impl FlatDfa {` | implementation block; review whether methods belong with the type |
| 101 | `pub fn trans(&self, state: usize, byte: usize) -> u32 {` | helper; check whether it belongs in current file or a later split |
| 107 | `pub fn transitions_for(&self, state: usize) -> &[u32] {` | helper; check whether it belongs in current file or a later split |
| 111 | `pub fn from_tokenizer(tokenizer: &Tokenizer) -> Self {` | helper; check whether it belongs in current file or a later split |
| 144 | `pub fn from_tokenizer_filtered(tokenizer: &Tokenizer, active_groups: &[bool]) -> Self {` | helper; check whether it belongs in current file or a later split |
| 184 | `pub fn from_flat_trans(` | helper; check whether it belongs in current file or a later split |
| 209 | `pub fn from_flat_trans_filtered(` | helper; check whether it belongs in current file or a later split |
| 246 | `pub struct TokenizerView {` | data witness/cache/profile; check whether name says what invariant it carries |
| 250 | `impl TokenizerView {` | implementation block; review whether methods belong with the type |
| 251 | `pub fn new(tokenizer: &Tokenizer) -> Self {` | helper; check whether it belongs in current file or a later split |
| 258 | `pub fn new_filtered(tokenizer: &Tokenizer, active_groups: &[bool]) -> Self {` | helper; check whether it belongs in current file or a later split |
| 266 | `pub fn new_from_flat_trans(` | helper; check whether it belongs in current file or a later split |
| 277 | `pub fn new_filtered_from_flat_trans(` | helper; check whether it belongs in current file or a later split |
| 287 | `pub fn dfa(&self) -> &FlatDfa {` | helper; check whether it belongs in current file or a later split |
| 291 | `pub fn initial_state_id(&self) -> usize {` | helper; check whether it belongs in current file or a later split |

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/disallowed_follows.rs`

LOC: 60.  Symbols detected: 2.

| Line | Symbol | Review note |
|---:|---|---|
| 6 | `pub(crate) fn normalize_disallowed_follows(` | helper; check whether it belongs in current file or a later split |
| 25 | `pub(crate) fn build_disallowed_follow_dfa(disallowed_follows: &[BitSet]) -> DFA {` | builder/constructor; verify inputs and returned id-map witness are documented |

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/mod.rs`

LOC: 9.  Symbols detected: 0.

No top-level symbols detected by the simple inventory regex.

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/shared.rs`

LOC: 75.  Symbols detected: 5.

| Line | Symbol | Review note |
|---:|---|---|
| 4 | `pub(crate) struct TokenDedup<'a> {` | data witness/cache/profile; check whether name says what invariant it carries |
| 10 | `pub(crate) fn hash_byte_class_seq(bytes: &[u8], byte_to_class: &[u8; 256]) -> u128 {` | helper; check whether it belongs in current file or a later split |
| 24 | `pub(crate) fn expand_vocab_classes(` | helper; check whether it belongs in current file or a later split |
| 51 | `pub(crate) fn representative_tokens_for_vocab_classes<'a>(` | helper; check whether it belongs in current file or a later split |
| 61 | `pub(crate) fn tokenizer_group_count(tokenizer: &TokenizerView) -> usize {` | helper; check whether it belongs in current file or a later split |

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs`

LOC: 1051.  Symbols detected: 31.

| Line | Symbol | Review note |
|---:|---|---|
| 14 | `pub type StateEquivalenceResult = BTreeSet<BTreeSet<usize>>;` | type alias; ensure alias clarifies coordinate space |
| 17 | `struct WalkFrame {` | data witness/cache/profile; check whether name says what invariant it carries |
| 24 | `fn bit_words(num_bits: usize) -> usize {` | helper; check whether it belongs in current file or a later split |
| 29 | `fn bitset_set(bits: &mut [u64], idx: usize) {` | helper; check whether it belongs in current file or a later split |
| 34 | `fn bitset_clear(bits: &mut [u64], idx: usize) {` | helper; check whether it belongs in current file or a later split |
| 39 | `fn clear_active_positions(positions: &mut [i32], active_bits: &mut [u64]) {` | helper; check whether it belongs in current file or a later split |
| 52 | `fn mix_u128(mut x: u128) -> u128 {` | helper; check whether it belongs in current file or a later split |
| 62 | `fn mix_tagged(hash: u128, tag: u128, value: u128) -> u128 {` | helper; check whether it belongs in current file or a later split |
| 66 | `fn hash_future_groups(future_groups: &[usize]) -> u128 {` | helper; check whether it belongs in current file or a later split |
| 74 | `fn hash_future_groups_filtered(future_groups: &[usize], disallowed: &BitSet) -> u128 {` | helper; check whether it belongs in current file or a later split |
| 89 | `struct FollowContextTable {` | data witness/cache/profile; check whether name says what invariant it carries |
| 94 | `impl FollowContextTable {` | implementation block; review whether methods belong with the type |
| 95 | `fn new(num_groups: usize, disallowed_follows: Option<&[BitSet]>) -> Self {` | helper; check whether it belongs in current file or a later split |
| 133 | `fn num_contexts(&self) -> usize {` | helper; check whether it belongs in current file or a later split |
| 138 | `fn context_for_gid(&self, gid: usize) -> usize {` | helper; check whether it belongs in current file or a later split |
| 143 | `fn allows_follow(&self, context: usize, gid: usize) -> bool {` | helper; check whether it belongs in current file or a later split |
| 149 | `struct SuffixNode {` | data witness/cache/profile; check whether name says what invariant it carries |
| 154 | `struct TokenSuffixHashes {` | data witness/cache/profile; check whether name says what invariant it carries |
| 160 | `impl TokenSuffixHashes {` | implementation block; review whether methods belong with the type |
| 162 | `fn get(&self, context: usize, pos: usize) -> u128 {` | helper; check whether it belongs in current file or a later split |
| 167 | `fn build_future_group_hashes_by_context(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 188 | `fn hash_suffix_node(` | helper; check whether it belongs in current file or a later split |
| 252 | `fn build_token_suffix_hashes(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 284 | `fn hash_trellis_node_from_positions(` | helper; check whether it belongs in current file or a later split |
| 335 | `fn build_strided_batches(total_tokens: usize, target_batch_size: usize) -> Vec<Vec<usize>> {` | builder/constructor; verify inputs and returned id-map witness are documented |
| 354 | `fn build_start_state_suffix_nodes(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 425 | `pub fn find_state_equivalence_classes_with_disallowed<S: AsRef<[u8]> + Sync>(` | helper; check whether it belongs in current file or a later split |
| 444 | `pub fn find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed<` | helper; check whether it belongs in current file or a later split |
| 468 | `fn find_state_equivalence_classes_ex_inner<S: AsRef<[u8]> + Sync>(` | helper; check whether it belongs in current file or a later split |
| 496 | `fn find_state_equivalence_classes_token_based<S: AsRef<[u8]> + Sync>(` | helper; check whether it belongs in current file or a later split |
| 1040 | `pub fn mapping_to_equivalence_classes(` | helper; check whether it belongs in current file or a later split |

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs`

LOC: 583.  Symbols detected: 26.

| Line | Symbol | Review note |
|---:|---|---|
| 15 | `struct ActiveTransitionTable {` | data witness/cache/profile; check whether name says what invariant it carries |
| 21 | `enum RefineMode {` | closed choice; prefer this over raw strings where policy is internal |
| 27 | `fn refine_mode() -> RefineMode {` | helper; check whether it belongs in current file or a later split |
| 36 | `fn is_full_state_query(states: &[usize], total_states: usize) -> bool {` | helper; check whether it belongs in current file or a later split |
| 47 | `fn mix_u64(mut x: u64) -> u64 {` | helper; check whether it belongs in current file or a later split |
| 57 | `fn hash_signature_row(row: &[u32]) -> u64 {` | helper; check whether it belongs in current file or a later split |
| 66 | `fn usize_to_u32(value: usize, what: &str) -> u32 {` | helper; check whether it belongs in current file or a later split |
| 71 | `fn is_active_group(group_id: usize, active_groups: Option<&[bool]>) -> bool {` | helper; check whether it belongs in current file or a later split |
| 77 | `fn filtered_group_ids(values: &[usize], active_groups: Option<&[bool]>) -> Vec<usize> {` | helper; check whether it belongs in current file or a later split |
| 85 | `fn build_filtered_finalizer_labels(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 95 | `fn build_filtered_possible_future_labels(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 105 | `fn build_has_any_transition_labels(dfa: &FlatDfa) -> Vec<bool> {` | builder/constructor; verify inputs and returned id-map witness are documented |
| 117 | `fn byte_is_relevant(byte: usize, relevant_bytes: Option<&[bool; 256]>) -> bool {` | helper; check whether it belongs in current file or a later split |
| 121 | `fn active_byte_representatives(` | helper; check whether it belongs in current file or a later split |
| 148 | `fn build_initial_label_partition(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 190 | `fn same_partition(left: &[u32], left_count: usize, right: &[u32], right_count: usize) -> bool {` | helper; check whether it belongs in current file or a later split |
| 221 | `fn build_active_transition_table(dfa: &FlatDfa, active_bytes: &[u8]) -> ActiveTransitionTable {` | builder/constructor; verify inputs and returned id-map witness are documented |
| 241 | `fn refine_once_sorted(` | helper; check whether it belongs in current file or a later split |
| 312 | `fn row_hash(` | helper; check whether it belongs in current file or a later split |
| 335 | `fn rows_equal(` | helper; check whether it belongs in current file or a later split |
| 369 | `fn refine_once_interned(` | helper; check whether it belongs in current file or a later split |
| 422 | `fn auto_prefers_sorted_refinement(` | helper; check whether it belongs in current file or a later split |
| 430 | `fn compute_kbounded_partition(` | helper; check whether it belongs in current file or a later split |
| 491 | `fn build_subset_mapping(states: &[usize], blocks: &[u32]) -> Vec<usize> {` | builder/constructor; verify inputs and returned id-map witness are documented |
| 520 | `pub(crate) fn find_state_equivalence_classes_kbounded(` | helper; check whether it belongs in current file or a later split |
| 543 | `pub fn find_state_equivalence_classes_byte_restricted<S: AsRef<[u8]>>(` | helper; check whether it belongs in current file or a later split |

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/mod.rs`

LOC: 7.  Symbols detected: 0.

No top-level symbols detected by the simple inventory regex.

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs`

LOC: 492.  Symbols detected: 22.

| Line | Symbol | Review note |
|---:|---|---|
| 13 | `pub(crate) enum MaxLengthMode {` | closed choice; prefer this over raw strings where policy is internal |
| 18 | `impl MaxLengthMode {` | implementation block; review whether methods belong with the type |
| 19 | `pub(crate) fn name(self) -> &'static str {` | helper; check whether it belongs in current file or a later split |
| 28 | `pub(crate) struct MaxLengthStatistic {` | data witness/cache/profile; check whether name says what invariant it carries |
| 34 | `fn mix_u64(mut x: u64) -> u64 {` | helper; check whether it belongs in current file or a later split |
| 43 | `fn is_active_group(group_id: usize, active_groups: Option<&[bool]>) -> bool {` | helper; check whether it belongs in current file or a later split |
| 49 | `fn filtered_terminals(` | helper; check whether it belongs in current file or a later split |
| 61 | `fn build_filtered_finalizer_labels(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 71 | `fn build_filtered_possible_future_labels(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 83 | `fn build_has_any_transition_labels(tokenizer: &Tokenizer) -> Vec<bool> {` | builder/constructor; verify inputs and returned id-map witness are documented |
| 92 | `fn build_initial_label_partition(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 134 | `fn byte_is_relevant(byte: usize, relevant_bytes: Option<&[bool; 256]>) -> bool {` | helper; check whether it belongs in current file or a later split |
| 138 | `fn active_byte_representatives(` | helper; check whether it belongs in current file or a later split |
| 165 | `fn compute_byte_classes(tokenizer: &Tokenizer) -> [u8; 256] {` | helper; check whether it belongs in current file or a later split |
| 219 | `fn same_partition(left: &[u32], left_count: usize, right: &[u32], right_count: usize) -> bool {` | helper; check whether it belongs in current file or a later split |
| 249 | `fn refine_once_sorted(` | helper; check whether it belongs in current file or a later split |
| 316 | `fn build_full_mapping_from_blocks(blocks: &[u32], num_states: usize) -> Vec<usize> {` | builder/constructor; verify inputs and returned id-map witness are documented |
| 335 | `fn build_subset_mapping(states: &[usize], full_mapping: &[usize]) -> Vec<usize> {` | builder/constructor; verify inputs and returned id-map witness are documented |
| 342 | `fn compute_kbounded_partition(` | helper; check whether it belongs in current file or a later split |
| 383 | `fn stable_refinement_blocks(` | helper; check whether it belongs in current file or a later split |
| 423 | `pub(crate) fn compute_statistic(vocab: &Vocab) -> MaxLengthStatistic {` | helper; check whether it belongs in current file or a later split |
| 438 | `pub(crate) fn compute_state_map(` | helper; check whether it belongs in current file or a later split |

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/mod.rs`

LOC: 96.  Symbols detected: 2.

| Line | Symbol | Review note |
|---:|---|---|
| 11 | `pub(crate) fn identity_state_map(num_states: usize) -> ManyToOneIdMap {` | helper; check whether it belongs in current file or a later split |
| 20 | `pub(crate) fn build_state_map_from_subset_representatives(` | builder/constructor; verify inputs and returned id-map witness are documented |

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/pipeline.rs`

LOC: 171.  Symbols detected: 13.

| Line | Symbol | Review note |
|---:|---|---|
| 11 | `pub(crate) enum StateEquivalencePassKind {` | closed choice; prefer this over raw strings where policy is internal |
| 15 | `impl StateEquivalencePassKind {` | implementation block; review whether methods belong with the type |
| 16 | `fn parse(value: &str) -> Result<Self, String> {` | helper; check whether it belongs in current file or a later split |
| 27 | `pub(crate) enum StateEquivalenceScope {` | closed choice; prefer this over raw strings where policy is internal |
| 33 | `pub(crate) struct StateEquivalencePipelineConfig {` | data witness/cache/profile; check whether name says what invariant it carries |
| 38 | `pub(crate) struct StateEquivalencePassProfile {` | data witness/cache/profile; check whether name says what invariant it carries |
| 47 | `pub(crate) struct StateEquivalencePipelineProfile {` | data witness/cache/profile; check whether name says what invariant it carries |
| 54 | `fn parse_passes(value: &str) -> Vec<StateEquivalencePassKind> {` | helper; check whether it belongs in current file or a later split |
| 66 | `fn resolve_pipeline_config(` | helper; check whether it belongs in current file or a later split |
| 82 | `pub(crate) fn resolve_global_pipeline_config(` | helper; check whether it belongs in current file or a later split |
| 93 | `pub(crate) fn resolve_pair_partition_pipeline_config(` | helper; check whether it belongs in current file or a later split |
| 104 | `pub(crate) fn run_state_equivalence_pipeline(` | helper; check whether it belongs in current file or a later split |
| 154 | `fn record_max_length_profile(` | helper; check whether it belongs in current file or a later split |

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs`

LOC: 2455.  Symbols detected: 72.

| Line | Symbol | Review note |
|---:|---|---|
| 21 | `pub type VocabEquivalenceResult = BTreeSet<Vec<usize>>;` | type alias; ensure alias clarifies coordinate space |
| 23 | `type EdgeList = SmallVec<[(usize, usize); 4]>;` | type alias; ensure alias clarifies coordinate space |
| 25 | `struct DagNode {` | data witness/cache/profile; check whether name says what invariant it carries |
| 46 | `struct Dfa {` | data witness/cache/profile; check whether name says what invariant it carries |
| 70 | `pub struct SharedVocabDfaBase {` | data witness/cache/profile; check whether name says what invariant it carries |
| 81 | `impl SharedVocabDfaBase {` | implementation block; review whether methods belong with the type |
| 83 | `pub fn build_from_dfa(dfa: &super::super::compat::FlatDfa) -> Self {` | builder/constructor; verify inputs and returned id-map witness are documented |
| 147 | `pub fn byte_to_class(&self) -> [u8; 256] {` | helper; check whether it belongs in current file or a later split |
| 154 | `pub fn is_compatible_with_dfa(&self, dfa: &super::super::compat::FlatDfa) -> bool {` | helper; check whether it belongs in current file or a later split |
| 172 | `pub type SharedVocabDfaCache = std::sync::OnceLock<SharedVocabDfaBase>;` | type alias; ensure alias clarifies coordinate space |
| 174 | `impl Dfa {` | implementation block; review whether methods belong with the type |
| 177 | `fn completion(&self, state: usize) -> u64 {` | helper; check whether it belongs in current file or a later split |
| 186 | `fn completion_with_disallowed(&self, state: usize, disallowed: Option<&BitSet>) -> u64 {` | helper; check whether it belongs in current file or a later split |
| 201 | `fn transition(&self, state: usize, byte: u8) -> u32 {` | helper; check whether it belongs in current file or a later split |
| 207 | `fn disallowed_for(&self, gid: usize) -> &BitSet {` | helper; check whether it belongs in current file or a later split |
| 213 | `struct Scratch {` | data witness/cache/profile; check whether name says what invariant it carries |
| 248 | `fn new_hasher() -> AHasher {` | helper; check whether it belongs in current file or a later split |
| 252 | `fn env_flag_enabled(name: &str) -> bool {` | configuration boundary; should remain outside denotation docs |
| 261 | `fn vocab_batch_size_override() -> Option<usize> {` | configuration boundary; should remain outside denotation docs |
| 268 | `fn vocab_verify_token_pair_override() -> Option<(usize, usize)> {` | configuration boundary; should remain outside denotation docs |
| 279 | `fn vocab_verify_token_pair_from_final_classes_enabled() -> bool {` | configuration boundary; should remain outside denotation docs |
| 283 | `fn vocab_state_group_size(num_states: usize, num_groups: usize) -> usize {` | helper; check whether it belongs in current file or a later split |
| 293 | `fn diversity_state_order_enabled() -> bool {` | configuration boundary; should remain outside denotation docs |
| 297 | `fn states_by_transition_diversity(dfa: &Dfa, states: &[usize]) -> Vec<usize> {` | helper; check whether it belongs in current file or a later split |
| 327 | `fn hash_group_list(iter: impl ExactSizeIterator<Item = usize>) -> u64 {` | helper; check whether it belongs in current file or a later split |
| 338 | `fn hash_filtered_group_list(groups: &[usize], disallowed: &BitSet) -> u64 {` | helper; check whether it belongs in current file or a later split |
| 356 | `fn build_dfa(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 373 | `fn build_dfa_with_group_filter(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 497 | `fn intersect_node_disallowed(` | helper; check whether it belongs in current file or a later split |
| 510 | `fn node_disallows_gid(scratch: &Scratch, pos: usize, gid: usize) -> bool {` | helper; check whether it belongs in current file or a later split |
| 520 | `fn ensure_position_slot<T>(slots: &mut Vec<Option<T>>, pos: usize) {` | helper; check whether it belongs in current file or a later split |
| 526 | `impl Scratch {` | implementation block; review whether methods belong with the type |
| 527 | `fn new(num_states: usize, num_groups: usize) -> Self {` | helper; check whether it belongs in current file or a later split |
| 554 | `fn mark_dirty_group(scratch: &mut Scratch, state_idx: usize, gid: usize) {` | helper; check whether it belongs in current file or a later split |
| 562 | `fn ensure_target_gids_map(` | helper; check whether it belongs in current file or a later split |
| 574 | `fn advance_seen_epoch(seen: &mut [u32], epoch: &mut u32) {` | helper; check whether it belongs in current file or a later split |
| 582 | `fn record_target_gid(` | helper; check whether it belongs in current file or a later split |
| 627 | `fn run_batch_inner(` | helper; check whether it belongs in current file or a later split |
| 747 | `fn collect_targets(` | helper; check whether it belongs in current file or a later split |
| 810 | `fn run_batch(` | helper; check whether it belongs in current file or a later split |
| 860 | `fn hash_suffixes(` | helper; check whether it belongs in current file or a later split |
| 979 | `fn run_suffix(` | helper; check whether it belongs in current file or a later split |
| 1036 | `fn try_hash_single_target_suffix(` | helper; check whether it belongs in current file or a later split |
| 1090 | `fn finish_token_signature(` | helper; check whether it belongs in current file or a later split |
| 1146 | `fn fill_state_observation_words_and_cleanup(` | helper; check whether it belongs in current file or a later split |
| 1202 | `fn compute_token_state_observation_words(` | helper; check whether it belongs in current file or a later split |
| 1231 | `fn first_distinguishing_state_for_token_pair_with_count<S: AsRef<[u8]>>(` | helper; check whether it belongs in current file or a later split |
| 1286 | `fn first_distinguishing_state_for_token_pair<S: AsRef<[u8]>>(` | helper; check whether it belongs in current file or a later split |
| 1307 | `fn log_vocab_pair_verification<S: AsRef<[u8]>>(` | helper; check whether it belongs in current file or a later split |
| 1357 | `fn run_vocab_row_cert_diag<S: AsRef<[u8]> + Sync>(` | helper; check whether it belongs in current file or a later split |
| 1424 | `fn token_signature(` | helper; check whether it belongs in current file or a later split |
| 1469 | `struct DepthChangeLog {` | data witness/cache/profile; check whether name says what invariant it carries |
| 1480 | `impl DepthChangeLog {` | implementation block; review whether methods belong with the type |
| 1481 | `fn new() -> Self {` | helper; check whether it belongs in current file or a later split |
| 1490 | `fn clear(&mut self) {` | helper; check whether it belongs in current file or a later split |
| 1498 | `struct TrieWalkState {` | data witness/cache/profile; check whether name says what invariant it carries |
| 1502 | `impl TrieWalkState {` | implementation block; review whether methods belong with the type |
| 1503 | `fn new() -> Self {` | helper; check whether it belongs in current file or a later split |
| 1509 | `fn ensure_depth(&mut self, depth: usize) {` | helper; check whether it belongs in current file or a later split |
| 1517 | `struct TrieWalkChunkStats {` | data witness/cache/profile; check whether name says what invariant it carries |
| 1537 | `impl TrieWalkChunkStats {` | implementation block; review whether methods belong with the type |
| 1538 | `fn add_assign(&mut self, other: Self) {` | helper; check whether it belongs in current file or a later split |
| 1560 | `struct DfsStepStats {` | data witness/cache/profile; check whether name says what invariant it carries |
| 1570 | `fn dfs_step(` | helper; check whether it belongs in current file or a later split |
| 1635 | `fn dfs_step_profiled(` | helper; check whether it belongs in current file or a later split |
| 1717 | `fn dfs_undo_depth(scratch: &mut Scratch, log: &DepthChangeLog) {` | helper; check whether it belongs in current file or a later split |
| 1733 | `fn dfs_backtrack(` | helper; check whether it belongs in current file or a later split |
| 1749 | `fn finish_token_signature_clean(` | helper; check whether it belongs in current file or a later split |
| 1782 | `fn finish_token_signature_no_cleanup(` | helper; check whether it belongs in current file or a later split |
| 1836 | `fn trie_walk_chunk_signatures<S: AsRef<[u8]> + Sync>(` | helper; check whether it belongs in current file or a later split |
| 2018 | `fn compact_dfa_for_tokens<S: AsRef<[u8]>>(` | helper; check whether it belongs in current file or a later split |
| 2149 | `pub fn find_vocab_equivalence_classes_with_group_filter<S: AsRef<[u8]> + Sync>(` | helper; check whether it belongs in current file or a later split |

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/mod.rs`

LOC: 6.  Symbols detected: 0.

No top-level symbols detected by the simple inventory regex.

## `src/compile/terminal_dwa/pair_partition/mod.rs`

LOC: 787.  Symbols detected: 16.

| Line | Symbol | Review note |
|---:|---|---|
| 47 | `struct SimplifyCacheKey {` | data witness/cache/profile; check whether name says what invariant it carries |
| 53 | `pub(crate) struct SharedSimplifyCache {` | data witness/cache/profile; check whether name says what invariant it carries |
| 57 | `struct SimplifyCacheEntry {` | data witness/cache/profile; check whether name says what invariant it carries |
| 62 | `impl SimplifyCacheEntry {` | implementation block; review whether methods belong with the type |
| 63 | `fn new() -> Self {` | helper; check whether it belongs in current file or a later split |
| 71 | `impl SharedSimplifyCache {` | implementation block; review whether methods belong with the type |
| 72 | `fn key(active_terminals: &[bool], relevant_bytes: &[bool; 256]) -> SimplifyCacheKey {` | helper; check whether it belongs in current file or a later split |
| 93 | `fn simplify_for_terminals(` | helper; check whether it belongs in current file or a later split |
| 146 | `fn project_initial_state_map_enabled() -> bool {` | configuration boundary; should remain outside denotation docs |
| 159 | `struct PairPartitionTokenLengthStats {` | data witness/cache/profile; check whether name says what invariant it carries |
| 168 | `fn pair_partition_token_length_stats(vocab: &Vocab) -> PairPartitionTokenLengthStats {` | helper; check whether it belongs in current file or a later split |
| 201 | `struct ProjectInitialStateMapProfile {` | data witness/cache/profile; check whether name says what invariant it carries |
| 212 | `impl ProjectInitialStateMapProfile {` | implementation block; review whether methods belong with the type |
| 213 | `fn unused(reason: &'static str, simplified_state_count: usize) -> Self {` | helper; check whether it belongs in current file or a later split |
| 227 | `fn project_initial_state_map_for_simplified_tokenizer(` | helper; check whether it belongs in current file or a later split |
| 370 | `pub(crate) fn build_pair_partition_terminal_dwa(` | builder/constructor; verify inputs and returned id-map witness are documented |

## `src/compile/terminal_dwa/pair_partition/nwa_builder.rs`

LOC: 1031.  Symbols detected: 48.

| Line | Symbol | Review note |
|---:|---|---|
| 27 | `type NwaState = u32;` | type alias; ensure alias clarifies coordinate space |
| 29 | `type TokenizerState = u32;` | type alias; ensure alias clarifies coordinate space |
| 30 | `type LeafTokenIds = SmallVec<[u32; 8]>;` | type alias; ensure alias clarifies coordinate space |
| 31 | `type FutureTerminalColorGroups = SmallVec<[(ColorId, SmallVec<[TerminalID; 4]>); 8]>;` | type alias; ensure alias clarifies coordinate space |
| 33 | `fn all_token_weight(internal_tsid: u32, max_token_id: u32) -> Weight {` | helper; check whether it belongs in current file or a later split |
| 41 | `pub(crate) struct NodesByTokenizerState {` | data witness/cache/profile; check whether name says what invariant it carries |
| 45 | `impl NodesByTokenizerState {` | implementation block; review whether methods belong with the type |
| 46 | `fn new() -> Self {` | helper; check whether it belongs in current file or a later split |
| 52 | `fn is_empty(&self) -> bool {` | helper; check whether it belongs in current file or a later split |
| 56 | `fn merge(&mut self, state: TokenizerState, nodes: &[NwaState]) {` | helper; check whether it belongs in current file or a later split |
| 60 | `fn first(&self, state: TokenizerState) -> Option<NwaState> {` | helper; check whether it belongs in current file or a later split |
| 64 | `fn push_one(&mut self, state: TokenizerState, node: NwaState) {` | helper; check whether it belongs in current file or a later split |
| 68 | `fn iter(&self) -> impl Iterator<Item = (TokenizerState, &[NwaState])> {` | helper; check whether it belongs in current file or a later split |
| 75 | `impl IntoIterator for NodesByTokenizerState {` | implementation block; review whether methods belong with the type |
| 76 | `type Item = (TokenizerState, Vec<NwaState>);` | type alias; ensure alias clarifies coordinate space |
| 77 | `type IntoIter = <FxHashMap<TokenizerState, Vec<NwaState>> as IntoIterator>::IntoIter;` | type alias; ensure alias clarifies coordinate space |
| 79 | `fn into_iter(self) -> Self::IntoIter {` | helper; check whether it belongs in current file or a later split |
| 84 | `pub(crate) struct TerminalNwaBuilder<'tok, 'cm, 'nwa> {` | data witness/cache/profile; check whether name says what invariant it carries |
| 112 | `struct BufferedLeafTransition {` | data witness/cache/profile; check whether name says what invariant it carries |
| 118 | `pub(crate) fn new(` | helper; check whether it belongs in current file or a later split |
| 161 | `fn fast_step(&mut self, state: u32, byte: u8) -> Option<u32> {` | helper; check whether it belongs in current file or a later split |
| 175 | `fn leaf_token_ids_for(&mut self, source: u32, label: TerminalID) -> &mut LeafTokenIds {` | helper; check whether it belongs in current file or a later split |
| 190 | `fn buffer_leaf_token_id(&mut self, source: u32, label: TerminalID, internal_token_id: u32) {` | helper; check whether it belongs in current file or a later split |
| 194 | `fn possible_future_terminals_for_state(&mut self, tokenizer_state: TokenizerState) -> Vec<TerminalID> {` | helper; check whether it belongs in current file or a later split |
| 205 | `fn populate_future_terminal_color_cache(&mut self, tokenizer_state: TokenizerState) {` | helper; check whether it belongs in current file or a later split |
| 234 | `fn ignore_terminal_possible_for_state(&mut self, tokenizer_state: TokenizerState) -> bool {` | helper; check whether it belongs in current file or a later split |
| 245 | `fn future_terminal_colors_for_state(` | helper; check whether it belongs in current file or a later split |
| 256 | `fn future_terminal_color_groups_for_state(` | helper; check whether it belongs in current file or a later split |
| 267 | `fn buffer_future_leaf_token_id(` | helper; check whether it belongs in current file or a later split |
| 282 | `fn add_future_leaf_token_from_sources(` | helper; check whether it belongs in current file or a later split |
| 313 | `fn add_future_weighted_match_from_sources(` | helper; check whether it belongs in current file or a later split |
| 355 | `fn cached_reachable_weight(&mut self, token_ids: &RangeSetBlaze<usize>) -> Weight {` | helper; check whether it belongs in current file or a later split |
| 367 | `fn token_set_weight_fast(&self, internal_token_ids: &RangeSetBlaze<usize>) -> Weight {` | helper; check whether it belongs in current file or a later split |
| 378 | `fn cached_leaf_weight(&mut self, mut token_ids: LeafTokenIds) -> Weight {` | helper; check whether it belongs in current file or a later split |
| 392 | `fn continuation_weight_for_match(` | helper; check whether it belongs in current file or a later split |
| 434 | `fn add_leaf_token_from_sources(` | helper; check whether it belongs in current file or a later split |
| 458 | `fn can_skip_self_loop_subtree(` | helper; check whether it belongs in current file or a later split |
| 476 | `fn emit_self_loop_leaf_only_subtree(` | helper; check whether it belongs in current file or a later split |
| 498 | `fn add_match_from_sources(` | helper; check whether it belongs in current file or a later split |
| 522 | `pub(crate) fn flush_transition_buffer(&mut self) {` | helper; check whether it belongs in current file or a later split |
| 656 | `pub(crate) fn build_direct_partition_fast(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 771 | `pub(crate) fn build_from_trie(` | builder/constructor; verify inputs and returned id-map witness are documented |
| 806 | `fn process_child_segment(` | helper; check whether it belongs in current file or a later split |
| 933 | `fn subtract_can_match(` | helper; check whether it belongs in current file or a later split |
| 942 | `fn ensure_continuation_state(` | helper; check whether it belongs in current file or a later split |
| 956 | `pub(crate) fn internal_vocab_entries(vocab: &Vocab, id_map: &InternalIdMap) -> Vec<(u32, Vec<u8>)> {` | helper; check whether it belongs in current file or a later split |
| 970 | `pub(crate) fn seed_root_nodes(` | helper; check whether it belongs in current file or a later split |
| 991 | `pub(crate) fn build_nwa_via_trie_walk<'a>(` | builder/constructor; verify inputs and returned id-map witness are documented |

## `src/compile/terminal_dwa/pair_partition/postprocess.rs`

LOC: 718.  Symbols detected: 21.

| Line | Symbol | Review note |
|---:|---|---|
| 21 | `fn structural_hash_nwa_state(state: &NWAStateType) -> u64 {` | helper; check whether it belongs in current file or a later split |
| 48 | `fn hash_weight(weight: &Weight, hasher: &mut impl Hasher) {` | helper; check whether it belongs in current file or a later split |
| 52 | `pub(crate) fn canonicalize_acyclic_nwa(nwa: &mut NWA) {` | helper; check whether it belongs in current file or a later split |
| 142 | `fn retain_nwa_states(nwa: &mut NWA, retain: &[bool], drop_empty_weights: bool) -> bool {` | helper; check whether it belongs in current file or a later split |
| 187 | `fn compute_forward_reachable(nwa: &NWA) -> Vec<bool> {` | helper; check whether it belongs in current file or a later split |
| 223 | `pub(crate) fn prune_unreachable_states(nwa: &mut NWA) -> bool {` | helper; check whether it belongs in current file or a later split |
| 231 | `fn topological_order(nwa: &NWA) -> Vec<usize> {` | helper; check whether it belongs in current file or a later split |
| 274 | `fn compute_coreachable_nwa(nwa: &NWA) -> Vec<bool> {` | helper; check whether it belongs in current file or a later split |
| 316 | `pub(crate) fn prune_non_coreachable_states(nwa: &mut NWA) -> bool {` | helper; check whether it belongs in current file or a later split |
| 326 | `fn propagate_incoming_labels(` | helper; check whether it belongs in current file or a later split |
| 370 | `fn propagate_collapse_context(` | helper; check whether it belongs in current file or a later split |
| 437 | `fn allowed_labels_by_state(` | helper; check whether it belongs in current file or a later split |
| 467 | `fn collapse_single_allowed_transitions(` | helper; check whether it belongs in current file or a later split |
| 543 | `pub(crate) fn collapse_always_allowed(` | helper; check whether it belongs in current file or a later split |
| 599 | `pub(crate) struct SharedDisallowedFollowDfaCache {` | data witness/cache/profile; check whether name says what invariant it carries |
| 603 | `impl SharedDisallowedFollowDfaCache {` | implementation block; review whether methods belong with the type |
| 604 | `pub(crate) fn new() -> Self {` | helper; check whether it belongs in current file or a later split |
| 608 | `fn get_or_build(` | helper; check whether it belongs in current file or a later split |
| 628 | `pub(crate) fn apply_disallowed_follow_constraints(` | helper; check whether it belongs in current file or a later split |
| 651 | `fn subtract_disallowed_dfa(nwa: &NWA, right: &DFA) -> NWA {` | helper; check whether it belongs in current file or a later split |
| 652 | `type ProdState = (u32, Option<u32>);` | type alias; ensure alias clarifies coordinate space |

## `src/compile/terminal_dwa/partition.rs`

LOC: 193.  Symbols detected: 1.

| Line | Symbol | Review note |
|---:|---|---|
| 31 | `pub(crate) fn build_partition_terminal_dwa(` | builder/constructor; verify inputs and returned id-map witness are documented |

## `src/compile/terminal_dwa/types.rs`

LOC: 86.  Symbols detected: 13.

| Line | Symbol | Review note |
|---:|---|---|
| 8 | `pub(crate) type ColorId = u32;` | type alias; ensure alias clarifies coordinate space |
| 14 | `pub(crate) struct TerminalColoring {` | data witness/cache/profile; check whether name says what invariant it carries |
| 19 | `impl TerminalColoring {` | implementation block; review whether methods belong with the type |
| 20 | `pub(crate) fn identity(num_terminals: usize) -> Self {` | helper; check whether it belongs in current file or a later split |
| 28 | `pub(crate) fn color_for(&self, terminal_id: TerminalID) -> ColorId {` | helper; check whether it belongs in current file or a later split |
| 38 | `pub(crate) struct TerminalDwaBuildProfile {` | data witness/cache/profile; check whether name says what invariant it carries |
| 44 | `pub(crate) struct TerminalDwaPhaseProfile {` | data witness/cache/profile; check whether name says what invariant it carries |
| 53 | `pub(crate) struct LocalIdMapTerminalDwa {` | data witness/cache/profile; check whether name says what invariant it carries |
| 59 | `impl TerminalDwaPhaseProfile {` | implementation block; review whether methods belong with the type |
| 60 | `pub(crate) fn total_ms(self) -> f64 {` | helper; check whether it belongs in current file or a later split |
| 64 | `pub(crate) fn add_assign(&mut self, other: Self) {` | helper; check whether it belongs in current file or a later split |
| 75 | `pub(crate) enum TerminalPathLength {` | closed choice; prefer this over raw strings where policy is internal |
| 84 | `pub(crate) fn compile_profile_enabled() -> bool {` | configuration boundary; should remain outside denotation docs |

## `src/compile/terminal_dwa/vocab_partition.rs`

LOC: 273.  Symbols detected: 8.

| Line | Symbol | Review note |
|---:|---|---|
| 27 | `struct CharTypeSubVocabs {` | data witness/cache/profile; check whether name says what invariant it carries |
| 31 | `impl crate::vocab::VocabDerivedArtifact for CharTypeSubVocabs {}` | implementation block; review whether methods belong with the type |
| 33 | `pub(crate) fn vocab_from_token_partitions(vocab: &Vocab, token_partitions: Vec<Vec<u32>>) -> Arc<[Vocab]> {` | helper; check whether it belongs in current file or a later split |
| 47 | `pub(crate) fn build_char_type_sub_vocabs(vocab: &Vocab) -> Arc<[Vocab]> {` | builder/constructor; verify inputs and returned id-map witness are documented |
| 72 | `pub(crate) fn prepare_vocab_for_terminal_dwa(vocab: &Vocab) {` | helper; check whether it belongs in current file or a later split |
| 89 | `pub(crate) fn choose_terminal_dwa_sub_vocabs(` | policy selection; must not perform automaton construction |
| 115 | `fn choose_cost_partitioned_sub_vocabs(` | policy selection; must not perform automaton construction |
| 153 | `fn choose_auto_partitioned_sub_vocabs(` | policy selection; must not perform automaton construction |

