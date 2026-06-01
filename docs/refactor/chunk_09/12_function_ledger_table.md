# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Table symbol ledger

| File | Line | Symbol |
| --- | ---: | --- |
| `src/parser/glr/table/action.rs` | 6 | `pub struct StackShift {` |
| `src/parser/glr/table/action.rs` | 12 | `pub struct StackShiftGuard {` |
| `src/parser/glr/table/action.rs` | 18 | `pub struct GuardedStackShift {` |
| `src/parser/glr/table/action.rs` | 26 | `pub enum Action {` |
| `src/parser/glr/table/action.rs` | 39 | `impl Action {` |
| `src/parser/glr/table/action.rs` | 113 | `mod tests {` |
| `src/parser/glr/table/build.rs` | 6 | `pub(super) fn build_table(grammar: &AnalyzedGrammar) -> GLRTable {` |
| `src/parser/glr/table/build.rs` | 93 | `fn replace_shifts_enabled() -> bool {` |
| `src/parser/glr/table/build.rs` | 97 | `fn replace_gotos_enabled() -> bool {` |
| `src/parser/glr/table/build.rs` | 105 | `fn local_forward_replace_enabled() -> bool {` |
| `src/parser/glr/table/build.rs` | 117 | `pub(super) struct Item {` |
| `src/parser/glr/table/build.rs` | 123 | `impl Item {` |
| `src/parser/glr/table/build.rs` | 135 | `pub(super) struct PendingAction {` |
| `src/parser/glr/table/build.rs` | 141 | `impl PendingAction {` |
| `src/parser/glr/table/build.rs` | 179 | `fn initialize_pending_and_goto(` |
| `src/parser/glr/table/build.rs` | 214 | `fn finish_table(` |
| `src/parser/glr/table/build.rs` | 248 | `fn lookahead_bit(term: TerminalID, num_terminals: u32) -> usize {` |
| `src/parser/glr/table/build.rs` | 256 | `fn bit_lookahead(bit: usize, num_terminals: u32) -> TerminalID {` |
| `src/parser/glr/table/build.rs` | 265 | `struct LR1ItemCore {` |
| `src/parser/glr/table/build.rs` | 272 | `impl LR1ItemCore {` |
| `src/parser/glr/table/build.rs` | 288 | `type LR1ItemSet = BTreeMap<LR1ItemCore, BitSet>;` |
| `src/parser/glr/table/build.rs` | 291 | `struct LR1Item {` |
| `src/parser/glr/table/build.rs` | 302 | `impl LR1Item {` |
| `src/parser/glr/table/build.rs` | 314 | `fn first_of_sequence_bits(` |
| `src/parser/glr/table/build.rs` | 345 | `fn first_bitsets(grammar: &AnalyzedGrammar) -> Vec<BitSet> {` |
| `src/parser/glr/table/build.rs` | 349 | `fn union_lookaheads(item_set: &mut LR1ItemSet, core: LR1ItemCore, lookaheads: &BitSet) -> BitSet {` |
| `src/parser/glr/table/build.rs` | 360 | `fn lr1_closure(` |
| `src/parser/glr/table/build.rs` | 402 | `fn item_set_key(items: &LR1ItemSet) -> Vec<(LR1ItemCore, BitSet)> {` |
| `src/parser/glr/table/build.rs` | 426 | `fn compute_transfer_items(` |
| `src/parser/glr/table/build.rs` | 501 | `fn build_lr1_item_sets(` |
| `src/parser/glr/table/build.rs` | 646 | `fn current_unique_reduce_len(` |
| `src/parser/glr/table/build.rs` | 669 | `fn grammar_has_recursion(rules: &[Rule]) -> bool {` |
| `src/parser/glr/table/build.rs` | 708 | `fn apply_local_forward_replace(` |
| `src/parser/glr/table/build.rs` | 879 | `fn inline_zero_pop_reduces(` |
| `src/parser/glr/table/build.rs` | 947 | `fn build_lr1_table(` |
| `src/parser/glr/table/build.rs` | 995 | `fn lr1_core_key(items: &LR1ItemSet) -> Vec<Item> {` |
| `src/parser/glr/table/build.rs` | 1002 | `fn build_ielr_table(` |
| `src/parser/glr/table/build.rs` | 1013 | `fn grouped_item_lookahead_counts(grammar: &AnalyzedGrammar) -> Vec<Vec<(u32, u32, u32, usize)>> {` |
| `src/parser/glr/table/build.rs` | 1027 | `mod tests {` |
| `src/parser/glr/table/mod.rs` | 10 | `mod action;` |
| `src/parser/glr/table/mod.rs` | 11 | `mod build;` |
| `src/parser/glr/table/mod.rs` | 12 | `mod optimize;` |
| `src/parser/glr/table/mod.rs` | 13 | `mod options;` |
| `src/parser/glr/table/mod.rs` | 14 | `mod row;` |
| `src/parser/glr/table/mod.rs` | 24 | `fn default_action_rows_enabled() -> bool {` |
| `src/parser/glr/table/mod.rs` | 29 | `pub struct GuardedShiftCellIndex {` |
| `src/parser/glr/table/mod.rs` | 37 | `pub struct GLRTable {` |
| `src/parser/glr/table/mod.rs` | 67 | `pub enum TableAmbiguityKind {` |
| `src/parser/glr/table/mod.rs` | 74 | `pub struct TableAmbiguity {` |
| `src/parser/glr/table/mod.rs` | 81 | `fn guarded_stack_shift_constraints(` |
| `src/parser/glr/table/mod.rs` | 102 | `fn guarded_stack_shifts_overlap(left: &GuardedStackShift, right: &GuardedStackShift) -> bool {` |
| `src/parser/glr/table/mod.rs` | 121 | `fn guarded_stack_shifts_are_ambiguous(shifts: &[GuardedStackShift]) -> bool {` |
| `src/parser/glr/table/mod.rs` | 132 | `fn action_ambiguity(action: &Action) -> Option<(TableAmbiguityKind, usize)> {` |
| `src/parser/glr/table/mod.rs` | 153 | `impl GLRTable {` |
| `src/parser/glr/table/mod.rs` | 450 | `fn action_presence_rows(action: &[ActionRow], num_terminals: u32) -> Vec<BitSet> {` |
| `src/parser/glr/table/mod.rs` | 458 | `fn action_presence_row(action_row: &ActionRow, num_terminals: u32) -> BitSet {` |
| `src/parser/glr/table/mod.rs` | 473 | `impl GLRTable {` |
| `src/parser/glr/table/mod.rs` | 487 | `pub(crate) mod testing {` |
| `src/parser/glr/table/mod.rs` | 522 | `mod ambiguity_tests {` |
| `src/parser/glr/table/optimize/guarded/action_exploration.rs` | 1 | `fn stack_effects_for_action(` |
| `src/parser/glr/table/optimize/guarded/action_exploration.rs` | 162 | `fn normalize_guarded_effects(effects: &mut Vec<GuardedStackShift>) {` |
| `src/parser/glr/table/optimize/guarded/action_materialize.rs` | 1 | `fn stack_effect_action(table: &GLRTable, mut effects: Vec<GuardedStackShift>) -> Option<Action> {` |
| `src/parser/glr/table/optimize/guarded/action_materialize.rs` | 27 | `fn try_inline_action_to_stack_shifts(` |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 1 | `struct StackEffectFrame {` |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 7 | `enum ReduceFrameResult {` |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 15 | `fn states_at_depth(` |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 43 | `fn normalize_states(mut states: Vec<u32>) -> Vec<u32> {` |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 49 | `fn add_guard_to_frame(` |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 70 | `fn pop_frame(frame: &mut StackEffectFrame, pop: u32) {` |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 80 | `fn push_transition_to_frame(frame: &mut StackEffectFrame, target: u32, replace: bool) {` |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 93 | `fn frame_to_guarded_shift(frame: StackEffectFrame) -> GuardedStackShift {` |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 101 | `fn stack_effect_action_key(action: &Action) -> StackEffectActionKey {` |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 114 | `fn stack_effect_action_tag(action: &Action) -> u8 {` |
| `src/parser/glr/table/optimize/guarded/reduce_frame.rs` | 1 | `fn apply_reduce_to_frame(` |
| `src/parser/glr/table/optimize/guarded/reduce_frame.rs` | 61 | `fn compose_guarded_shift_with_frame(` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization.rs` | 1 | `fn normalize_stack_shifts(shifts: &mut Vec<StackShift>) {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization.rs` | 6 | `fn canonicalize_stack_shift_predecessors_by_goto_superset(table: &GLRTable, shifts: &mut [StackShift]) {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization.rs` | 42 | `fn goto_row_is_target_compatible_subset(table: &GLRTable, subset: u32, superset: u32) -> bool {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization.rs` | 53 | `fn stack_shift_action(mut shifts: Vec<StackShift>) -> Option<Action> {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization_tests.rs` | 2 | `mod tests {` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 1 | `enum CellUpdate {` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 6 | `fn build_runtime_state_predecessors(` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 65 | `fn subset_key(subset: &BTreeSet<u32>) -> Vec<u32> {` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 69 | `fn union_state_subsets(` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 80 | `fn merge_shift_into_pending(` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 115 | `fn merge_action_into_pending(` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 170 | `fn build_merged_action_row(` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 207 | `fn build_merged_goto_row(` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 256 | `fn union_advance_rows(table: &GLRTable, subset: &BTreeSet<u32>) -> BitSet {` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 279 | `fn ensure_subset_state(` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 351 | `fn refresh_merged_states_depending_on(` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 400 | `fn unit_reduce_destination(` |
| `src/parser/glr/table/optimize/policy_adapter.rs` | 7 | `fn stack_shift_predecessor_canonicalization_enabled() -> bool {` |
| `src/parser/glr/table/optimize/policy_adapter.rs` | 11 | `fn recognizer_suffix_quotient_enabled() -> bool {` |
| `src/parser/glr/table/optimize/policy_adapter.rs` | 15 | `fn max_guarded_stack_effects() -> Option<usize> {` |
| `src/parser/glr/table/optimize/same_core_merge.rs` | 1 | `enum ActionSig {` |
| `src/parser/glr/table/optimize/same_core_merge.rs` | 15 | `struct RowSignature {` |
| `src/parser/glr/table/optimize/same_core_merge.rs` | 22 | `fn remap_action_to_partition(action: &Action, partition: &[u32]) -> ActionSig {` |
| `src/parser/glr/table/optimize/same_core_merge.rs` | 77 | `fn core_classes(core_keys: &[Vec<Item>]) -> Vec<u32> {` |
| `src/parser/glr/table/optimize/same_core_merge.rs` | 94 | `fn refine_same_core_partition(table: &GLRTable, core_keys: &[Vec<Item>]) -> Vec<u32> {` |
| `src/parser/glr/table/optimize/same_core_merge.rs` | 138 | `pub(super) fn merge_same_core_lr1_states(table: GLRTable, core_keys: &[Vec<Item>]) -> GLRTable {` |
| `src/parser/glr/table/optimize/stack_effect_keys.rs` | 1 | `enum StackEffectActionKey {` |
| `src/parser/glr/table/optimize/stack_effect_keys.rs` | 11 | `struct StackEffectKey {` |
| `src/parser/glr/table/optimize/stack_effect_keys.rs` | 20 | `struct StackEffectVisitKey {` |
| `src/parser/glr/table/optimize/stack_effect_keys.rs` | 28 | `struct StackEffectResult {` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 1 | `struct SuffixQuotient {` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 10 | `impl SuffixQuotient {` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 270 | `fn action_row_has_multi_stack_shifts(row: &ActionRow) -> bool {` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 274 | `fn action_has_multi_stack_shifts(action: &Action) -> bool {` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 348 | `fn normalize_suffixes(suffixes: &mut Vec<Vec<u32>>) {` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 353 | `fn normalize_guarded_effects_for_suffix_quotient(effects: &mut Vec<GuardedStackShift>) {` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 367 | `fn apply_goto_to_suffix(suffix: &[u32], target: u32, replace: bool) -> Vec<u32> {` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 381 | `fn unguarded_suffix_effect(` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 389 | `fn guarded_suffix_effect(` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 414 | `fn suffix_effect(` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 444 | `fn row_fingerprint(` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 474 | `fn rows_equal(` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 493 | `fn push_reachable_state(state: u32, reachable: &mut [bool], stack: &mut Vec<u32>) {` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 503 | `fn push_action_targets(action: &Action, reachable: &mut [bool], stack: &mut Vec<u32>) {` |
| `src/parser/glr/table/optimize/table_passes.rs` | 1 | `impl GLRTable {` |
| `src/parser/glr/table/optimize/unit_reductions.rs` | 1 | `fn try_inline_unit_reductions_for_cell(` |
| `src/parser/glr/table/optimize/unit_reductions.rs` | 49 | `fn try_inline_unit_reductions_for_cell_inner(` |
| `src/parser/glr/table/optimize/unit_reductions.rs` | 151 | `fn remap_action_targets(action: &Action, mapping: &[u32]) -> Action {` |
| `src/parser/glr/table/options.rs` | 14 | `pub(crate) struct GLRTableOptions {` |
| `src/parser/glr/table/options.rs` | 26 | `impl GLRTableOptions {` |
| `src/parser/glr/table/options.rs` | 72 | `pub(crate) fn table_options_from_env() -> GLRTableOptions {` |
| `src/parser/glr/table/options.rs` | 76 | `fn env_usize(name: &str, default: usize) -> usize {` |
| `src/parser/glr/table/options.rs` | 84 | `fn env_optional_usize(name: &str) -> Option<usize> {` |
| `src/parser/glr/table/options.rs` | 91 | `fn env_flag_enabled(name: &str, default: bool) -> bool {` |
| `src/parser/glr/table/row.rs` | 14 | `const INLINE_ROW_CAPACITY: usize = 8;` |
| `src/parser/glr/table/row.rs` | 17 | `pub(crate) enum SparseRow<K: Copy + Eq + Hash, V: Clone> {` |
| `src/parser/glr/table/row.rs` | 22 | `impl<K: Copy + Eq + Hash, V: Clone> Default for SparseRow<K, V> {` |
| `src/parser/glr/table/row.rs` | 28 | `impl<K: Copy + Eq + Hash, V: Clone> SparseRow<K, V> {` |
| `src/parser/glr/table/row.rs` | 129 | `impl<K: Copy + Eq + Hash, V: Clone + PartialEq> PartialEq for SparseRow<K, V> {` |
| `src/parser/glr/table/row.rs` | 138 | `impl<K: Copy + Eq + Hash, V: Clone + Eq> Eq for SparseRow<K, V> {}` |
| `src/parser/glr/table/row.rs` | 140 | `impl<K, V> Serialize for SparseRow<K, V>` |
| `src/parser/glr/table/row.rs` | 157 | `impl<'de, K, V> Deserialize<'de> for SparseRow<K, V>` |
| `src/parser/glr/table/row.rs` | 195 | `impl<'a, K: Copy + Eq + Hash, V: Clone> IntoIterator for &'a SparseRow<K, V> {` |
| `src/parser/glr/table/row.rs` | 204 | `impl<K: Copy + Eq + Hash, V: Clone> Index<&K> for SparseRow<K, V> {` |
| `src/parser/glr/table/row.rs` | 212 | `impl<K: Copy + Eq + Hash, V: Clone> FromIterator<(K, V)> for SparseRow<K, V> {` |
| `src/parser/glr/table/row.rs` | 222 | `pub(crate) enum SparseRowIter<'a, K: Copy + Eq + Hash, V: Clone> {` |
| `src/parser/glr/table/row.rs` | 227 | `impl<'a, K: Copy + Eq + Hash, V: Clone> Iterator for SparseRowIter<'a, K, V> {` |
| `src/parser/glr/table/row.rs` | 238 | `pub(crate) enum SparseRowKeys<'a, K: Copy + Eq + Hash, V: Clone> {` |
| `src/parser/glr/table/row.rs` | 243 | `impl<'a, K: Copy + Eq + Hash, V: Clone> Iterator for SparseRowKeys<'a, K, V> {` |
| `src/parser/glr/table/row.rs` | 254 | `pub(crate) enum SparseRowValues<'a, K: Copy + Eq + Hash, V: Clone> {` |
| `src/parser/glr/table/row.rs` | 259 | `impl<'a, K: Copy + Eq + Hash, V: Clone> Iterator for SparseRowValues<'a, K, V> {` |
| `src/parser/glr/table/row.rs` | 271 | `pub(crate) enum ActionRow {` |
| `src/parser/glr/table/row.rs` | 280 | `impl Default for ActionRow {` |
| `src/parser/glr/table/row.rs` | 286 | `impl ActionRow {` |
| `src/parser/glr/table/row.rs` | 518 | `impl PartialEq for ActionRow {` |
| `src/parser/glr/table/row.rs` | 528 | `impl Eq for ActionRow {}` |
| `src/parser/glr/table/row.rs` | 530 | `impl<'a> IntoIterator for &'a ActionRow {` |
| `src/parser/glr/table/row.rs` | 539 | `impl Index<&TerminalID> for ActionRow {` |
| `src/parser/glr/table/row.rs` | 547 | `impl FromIterator<(TerminalID, Action)> for ActionRow {` |
| `src/parser/glr/table/row.rs` | 553 | `pub(crate) enum ActionRowIter<'a> {` |
| `src/parser/glr/table/row.rs` | 558 | `impl<'a> Iterator for ActionRowIter<'a> {` |
| `src/parser/glr/table/row.rs` | 569 | `pub(crate) struct DefaultActionRowIter<'a> {` |
| `src/parser/glr/table/row.rs` | 576 | `impl<'a> Iterator for DefaultActionRowIter<'a> {` |
| `src/parser/glr/table/row.rs` | 593 | `pub(crate) struct ActionRowKeys<'a> {` |
| `src/parser/glr/table/row.rs` | 597 | `impl<'a> Iterator for ActionRowKeys<'a> {` |
| `src/parser/glr/table/row.rs` | 605 | `pub(crate) struct ActionRowValues<'a> {` |
| `src/parser/glr/table/row.rs` | 609 | `impl<'a> Iterator for ActionRowValues<'a> {` |
| `src/parser/glr/table/row.rs` | 617 | `pub(crate) type GotoRow = SparseRow<NonterminalID, (u32, bool)>;` |
| `src/parser/glr/table/row.rs` | 620 | `mod tests {` |
| `src/parser/glr/table/mod.rs` | 10 | `mod action;` |
| `src/parser/glr/table/mod.rs` | 11 | `mod build;` |
| `src/parser/glr/table/mod.rs` | 12 | `mod optimize;` |
| `src/parser/glr/table/mod.rs` | 13 | `mod options;` |
| `src/parser/glr/table/mod.rs` | 14 | `mod row;` |
| `src/parser/glr/table/mod.rs` | 24 | `fn default_action_rows_enabled() -> bool {` |
| `src/parser/glr/table/mod.rs` | 29 | `pub struct GuardedShiftCellIndex {` |
| `src/parser/glr/table/mod.rs` | 37 | `pub struct GLRTable {` |
| `src/parser/glr/table/mod.rs` | 67 | `pub enum TableAmbiguityKind {` |
| `src/parser/glr/table/mod.rs` | 74 | `pub struct TableAmbiguity {` |
| `src/parser/glr/table/mod.rs` | 81 | `fn guarded_stack_shift_constraints(` |
| `src/parser/glr/table/mod.rs` | 102 | `fn guarded_stack_shifts_overlap(left: &GuardedStackShift, right: &GuardedStackShift) -> bool {` |
| `src/parser/glr/table/mod.rs` | 121 | `fn guarded_stack_shifts_are_ambiguous(shifts: &[GuardedStackShift]) -> bool {` |
| `src/parser/glr/table/mod.rs` | 132 | `fn action_ambiguity(action: &Action) -> Option<(TableAmbiguityKind, usize)> {` |
| `src/parser/glr/table/mod.rs` | 153 | `impl GLRTable {` |
| `src/parser/glr/table/mod.rs` | 450 | `fn action_presence_rows(action: &[ActionRow], num_terminals: u32) -> Vec<BitSet> {` |
| `src/parser/glr/table/mod.rs` | 458 | `fn action_presence_row(action_row: &ActionRow, num_terminals: u32) -> BitSet {` |
| `src/parser/glr/table/mod.rs` | 473 | `impl GLRTable {` |
| `src/parser/glr/table/mod.rs` | 487 | `pub(crate) mod testing {` |
| `src/parser/glr/table/mod.rs` | 522 | `mod ambiguity_tests {` |
