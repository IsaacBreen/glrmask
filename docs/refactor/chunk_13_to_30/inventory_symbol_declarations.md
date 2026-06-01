# Symbol inventory after Chunk 13-30

| File | Line | Symbol | Declaration |
|---|---:|---|---|
| `src/api/options.rs` | 14 | `CompileOptions` | `pub struct CompileOptions {}` |
| `src/api/options.rs` | 22 | `RuntimeOptions` | `pub struct RuntimeOptions {}` |
| `src/api/state.rs` | 20 | `State` | `pub type State<'a> = ConstraintState<'a>;` |
| `src/automata/lexer/ast.rs` | 17 | `Expr` | `pub enum Expr {` |
| `src/automata/lexer/ast.rs` | 40 | `byte` | `pub fn byte(b: u8) -> Expr {` |
| `src/automata/lexer/ast.rs` | 44 | `bytes` | `pub fn bytes(bs: &[u8]) -> Expr {` |
| `src/automata/lexer/ast.rs` | 48 | `class` | `pub fn class(set: U8Set) -> Expr {` |
| `src/automata/lexer/ast.rs` | 52 | `dfa` | `pub fn dfa(dfa: DFA) -> Expr {` |
| `src/automata/lexer/ast.rs` | 56 | `seq` | `pub fn seq(exprs: Vec<Expr>) -> Expr {` |
| `src/automata/lexer/ast.rs` | 60 | `choice` | `pub fn choice(exprs: Vec<Expr>) -> Expr {` |
| `src/automata/lexer/ast.rs` | 64 | `exclude` | `pub fn exclude(expr: impl Into<Expr>, excluded: impl Into<Expr>) -> Expr {` |
| `src/automata/lexer/ast.rs` | 71 | `intersect` | `pub fn intersect(expr: impl Into<Expr>, other: impl Into<Expr>) -> Expr {` |
| `src/automata/lexer/ast.rs` | 78 | `repeat` | `pub fn repeat(expr: impl Into<Expr>, min: usize, max: Option<usize>) -> Expr {` |
| `src/automata/lexer/ast.rs` | 86 | `plus` | `pub fn plus(expr: impl Into<Expr>) -> Expr {` |
| `src/automata/lexer/ast.rs` | 90 | `star` | `pub fn star(expr: impl Into<Expr>) -> Expr {` |
| `src/automata/lexer/ast.rs` | 94 | `opt` | `pub fn opt(expr: impl Into<Expr>) -> Expr {` |
| `src/automata/lexer/ast.rs` | 98 | `eps` | `pub fn eps() -> Expr {` |
| `src/automata/lexer/ast.rs` | 102 | `optimize_repeat_expr` | `fn optimize_repeat_expr(expr: Expr, min: usize, max: Option<usize>) -> Expr {` |
| `src/automata/lexer/ast.rs` | 114 | `flatten_sequence_parts` | `fn flatten_sequence_parts(exprs: Vec<Expr>) -> Vec<Expr> {` |
| `src/automata/lexer/ast.rs` | 126 | `simplify_single_byte_classes` | `fn simplify_single_byte_classes(exprs: &mut [Expr]) {` |
| `src/automata/lexer/ast.rs` | 137 | `merge_adjacent_byte_sequences` | `fn merge_adjacent_byte_sequences(exprs: Vec<Expr>) -> Vec<Expr> {` |
| `src/automata/lexer/ast.rs` | 154 | `flatten_choice_parts` | `fn flatten_choice_parts(exprs: Vec<Expr>) -> Vec<Expr> {` |
| `src/automata/lexer/ast.rs` | 166 | `fold_choice_byte_classes` | `fn fold_choice_byte_classes(exprs: Vec<Expr>) -> Vec<Expr> {` |
| `src/automata/lexer/ast.rs` | 187 | `is_nullable` | `pub fn is_nullable(&self) -> bool {` |
| `src/automata/lexer/ast.rs` | 202 | `optimize` | `pub fn optimize(self) -> Self {` |
| `src/automata/lexer/ast.rs` | 226 | `strip_prefix` | `pub fn strip_prefix(&self, prefix: &Expr) -> Option<Expr> {` |
| `src/automata/lexer/ast.rs` | 249 | `make_seq` | `pub fn make_seq(exprs: Vec<Expr>) -> Expr {` |
| `src/automata/lexer/ast.rs` | 266 | `make_choice` | `pub fn make_choice(exprs: Vec<Expr>) -> Expr {` |
| `src/automata/lexer/ast.rs` | 287 | `from` | `fn from(s: &str) -> Self {` |
| `src/automata/lexer/compile.rs` | 16 | `ProductStateTuple` | `type ProductStateTuple = SmallVec<[(u32, u32); 12]>;` |
| `src/automata/lexer/compile.rs` | 18 | `common_prefix_factor` | `fn common_prefix_factor(exprs: &[Expr]) -> Option<(Expr, Vec<Expr>)> {` |
| `src/automata/lexer/compile.rs` | 19 | `candidate_prefix` | `fn candidate_prefix(expr: &Expr) -> Option<&Expr> {` |
| `src/automata/lexer/compile.rs` | 35 | `expr_contains_group_op` | `fn expr_contains_group_op(expr: &Expr) -> bool {` |
| `src/automata/lexer/compile.rs` | 45 | `split_top_level_group_ops` | `fn split_top_level_group_ops(expr: &Expr) -> (Expr, Vec<Expr>, Vec<Expr>) {` |
| `src/automata/lexer/compile.rs` | 65 | `materialize_nested_group_ops` | `fn materialize_nested_group_ops(expr: Expr) -> Expr {` |
| `src/automata/lexer/compile.rs` | 94 | `ExclusionCompilePlan` | `struct ExclusionCompilePlan {` |
| `src/automata/lexer/compile.rs` | 101 | `build_exclusion_compile_plan` | `fn build_exclusion_compile_plan(exprs: &[Expr]) -> ExclusionCompilePlan {` |
| `src/automata/lexer/compile.rs` | 173 | `expr_accepts_empty` | `fn expr_accepts_empty(expr: &Expr) -> bool {` |
| `src/automata/lexer/compile.rs` | 190 | `expr_u8set` | `fn expr_u8set(expr: &Expr) -> U8Set {` |
| `src/automata/lexer/compile.rs` | 214 | `highest_power_of_two_leq` | `fn highest_power_of_two_leq(value: usize) -> usize {` |
| `src/automata/lexer/compile.rs` | 219 | `RepeatCompiler` | `struct RepeatCompiler<'expr, 'nfa> {` |
| `src/automata/lexer/compile.rs` | 227 | `new` | `fn new(expr: &'expr Expr, nfa: &'nfa mut NFA) -> Self {` |
| `src/automata/lexer/compile.rs` | 236 | `compile_power` | `fn compile_power(&mut self, copies: usize, end: u32) -> u32 {` |
| `src/automata/lexer/compile.rs` | 257 | `compile_exact` | `fn compile_exact(&mut self, copies: usize, end: u32) -> u32 {` |
| `src/automata/lexer/compile.rs` | 267 | `compile_upto` | `fn compile_upto(&mut self, copies: usize, end: u32) -> u32 {` |
| `src/automata/lexer/compile.rs` | 291 | `append_byte_sequence_expr` | `fn append_byte_sequence_expr(bytes: &[u8], nfa: &mut NFA, start: u32, end: u32) {` |
| `src/automata/lexer/compile.rs` | 308 | `append_dfa_expr` | `fn append_dfa_expr(dfa: &DFA, nfa: &mut NFA, start: u32, end: u32) {` |
| `src/automata/lexer/compile.rs` | 326 | `append_sequence_expr` | `fn append_sequence_expr(parts: &[Expr], nfa: &mut NFA, start: u32, end: u32) {` |
| `src/automata/lexer/compile.rs` | 343 | `append_choice_expr` | `fn append_choice_expr(options: &[Expr], nfa: &mut NFA, start: u32, end: u32) {` |
| `src/automata/lexer/compile.rs` | 356 | `compile_expr_to_dfa` | `fn compile_expr_to_dfa(expr: &Expr) -> DFA {` |
| `src/automata/lexer/compile.rs` | 362 | `productive_dfa_states` | `fn productive_dfa_states(dfa: &DFA) -> Vec<bool> {` |
| `src/automata/lexer/compile.rs` | 391 | `dfa_is_nonnullable_and_prefix_free` | `fn dfa_is_nonnullable_and_prefix_free(dfa: &DFA) -> bool {` |
| `src/automata/lexer/compile.rs` | 411 | `compile_direct_bounded_repeat_base_dfa_unconditionally` | `fn compile_direct_bounded_repeat_base_dfa_unconditionally(expr: &Expr) -> Option<DFA> {` |
| `src/automata/lexer/compile.rs` | 420 | `compile_direct_bounded_repeat_base_dfa` | `fn compile_direct_bounded_repeat_base_dfa(expr: &Expr, max: usize) -> Option<DFA> {` |
| `src/automata/lexer/compile.rs` | 427 | `build_bounded_repeat_dfa` | `fn build_bounded_repeat_dfa(expr: &Expr, min: usize, max: usize) -> Option<DFA> {` |
| `src/automata/lexer/compile.rs` | 471 | `collect_suffix_bytes` | `fn collect_suffix_bytes(exprs: &[Expr]) -> Option<Vec<u8>> {` |
| `src/automata/lexer/compile.rs` | 490 | `build_bounded_repeat_with_suffix_dfa` | `fn build_bounded_repeat_with_suffix_dfa(parts: &[Expr]) -> Option<(DFA, bool)> {` |
| `src/automata/lexer/compile.rs` | 619 | `build_bounded_repeat_with_regex_suffix` | `fn build_bounded_repeat_with_regex_suffix(parts: &[Expr]) -> Option<(DFA, bool)> {` |
| `src/automata/lexer/compile.rs` | 790 | `prepend_literal_prefix_to_dfa` | `fn prepend_literal_prefix_to_dfa(prefix_bytes: &[u8], tail_dfa: DFA) -> Option<DFA> {` |
| `src/automata/lexer/compile.rs` | 832 | `build_prefixed_bounded_repeat_with_suffix_dfa` | `fn build_prefixed_bounded_repeat_with_suffix_dfa(parts: &[Expr]) -> Option<(DFA, bool)> {` |
| `src/automata/lexer/compile.rs` | 884 | `append_bounded_repeat_expr` | `fn append_bounded_repeat_expr(expr: &Expr, min: usize, max: usize, nfa: &mut NFA, start: u32, end: u32) {` |
| `src/automata/lexer/compile.rs` | 901 | `append_unbounded_repeat_expr` | `fn append_unbounded_repeat_expr(` |
| `src/automata/lexer/compile.rs` | 930 | `append_compiled_expr` | `fn append_compiled_expr(expr: &Expr, nfa: &mut NFA, start: u32, end: u32) {` |
| `src/automata/lexer/compile.rs` | 955 | `Regex` | `pub struct Regex {` |
| `src/automata/lexer/compile.rs` | 960 | `num_states` | `pub fn num_states(&self) -> usize {` |
| `src/automata/lexer/compile.rs` | 964 | `num_transitions` | `pub fn num_transitions(&self) -> usize {` |
| `src/automata/lexer/compile.rs` | 968 | `step` | `pub fn step(&self, state: u32, byte: u8) -> Option<u32> {` |
| `src/automata/lexer/compile.rs` | 972 | `get_u8set` | `pub fn get_u8set(&self, state: u32) -> U8Set {` |
| `src/automata/lexer/compile.rs` | 977 | `dfa_transition_count` | `fn dfa_transition_count(dfa: &DFA) -> usize {` |
| `src/automata/lexer/compile.rs` | 985 | `build` | `pub fn build(self) -> Regex {` |
| `src/automata/lexer/compile.rs` | 993 | `compile_single_expr_dfa` | `fn compile_single_expr_dfa(expr: &Expr) -> DFA {` |
| `src/automata/lexer/compile.rs` | 1008 | `compile_with_plan` | `fn compile_with_plan(plan: ExclusionCompilePlan) -> DFA {` |
| `src/automata/lexer/compile.rs` | 1067 | `build_regex` | `pub fn build_regex(exprs: &[Expr]) -> Regex {` |
| `src/automata/lexer/compile.rs` | 1073 | `product_state_metadata` | `fn product_state_metadata(` |
| `src/automata/lexer/compile.rs` | 1109 | `explicit_dead_sink_state` | `fn explicit_dead_sink_state(dfa: &DFA) -> Option<u32> {` |
| `src/automata/lexer/compile.rs` | 1133 | `expr_is_epsilon_only` | `fn expr_is_epsilon_only(expr: &Expr) -> bool {` |
| `src/automata/lexer/compile.rs` | 1148 | `optional_choice_non_epsilon` | `fn optional_choice_non_epsilon(expr: &Expr) -> Option<&Expr> {` |
| `src/automata/lexer/compile.rs` | 1164 | `optional_tail_parts` | `fn optional_tail_parts(expr: &Expr) -> Option<Vec<Expr>> {` |
| `src/automata/lexer/compile.rs` | 1173 | `mark_state_accepting` | `fn mark_state_accepting(dfa: &mut DFA, state_id: u32) {` |
| `src/automata/lexer/compile.rs` | 1183 | `compile_product_component_dfa_direct` | `fn compile_product_component_dfa_direct(expr: &Expr) -> Option<(DFA, bool)> {` |
| `src/automata/lexer/compile.rs` | 1206 | `compile_product_component_dfa` | `fn compile_product_component_dfa(expr: &Expr) -> DFA {` |
| `src/automata/lexer/compile.rs` | 1210 | `ProductComponent` | `enum ProductComponent {` |
| `src/automata/lexer/compile.rs` | 1219 | `ProductComponentClassTransitions` | `enum ProductComponentClassTransitions {` |
| `src/automata/lexer/compile.rs` | 1225 | `partition_dfa` | `fn partition_dfa(&self) -> &DFA {` |
| `src/automata/lexer/compile.rs` | 1232 | `dead_state` | `fn dead_state(&self) -> Option<u32> {` |
| `src/automata/lexer/compile.rs` | 1240 | `compile_product_component` | `fn compile_product_component(expr: &Expr) -> ProductComponent {` |
| `src/automata/lexer/compile.rs` | 1262 | `build_product_dfa` | `fn build_product_dfa(exprs: &[Expr]) -> DFA {` |
| `src/automata/lexer/compile.rs` | 1441 | `compute_product_equivalence_classes` | `fn compute_product_equivalence_classes(components: &[ProductComponent]) -> (Vec<u8>, Vec<Vec<u8>>) {` |
| `src/automata/lexer/compile.rs` | 1478 | `build_product_class_transitions_for_dfa` | `fn build_product_class_transitions_for_dfa(dfa: &DFA, class_map: &[u8]) -> Vec<Vec<(u8, u32)>> {` |
| `src/automata/lexer/compile.rs` | 1493 | `build_product_class_transitions` | `fn build_product_class_transitions(` |
| `src/automata/lexer/compile.rs` | 1514 | `refine_u8_partitions` | `fn refine_u8_partitions(partitions: Vec<U8Set>, split: U8Set) -> Vec<U8Set> {` |
| `src/automata/lexer/compile.rs` | 1532 | `build_regex_nfa` | `pub fn build_regex_nfa(exprs: &[Expr]) -> NFA {` |
| `src/automata/lexer/compile.rs` | 1536 | `build_regex_nfa_impl` | `fn build_regex_nfa_impl(exprs: &[Expr]) -> NFA {` |
| `src/automata/lexer/compile.rs` | 1578 | `byte_expr` | `fn byte_expr(byte: u8) -> Expr {` |
| `src/automata/lexer/compile.rs` | 1582 | `byte_choice` | `fn byte_choice(bytes: &[u8]) -> Expr {` |
| `src/automata/lexer/compile.rs` | 1586 | `terminal_matches` | `fn terminal_matches(expr: Expr, input: &[u8]) -> bool {` |
| `src/automata/lexer/compile.rs` | 1600 | `nested_exclude_in_exclusion_branch_compiles` | `fn nested_exclude_in_exclusion_branch_compiles() {` |
| `src/automata/lexer/compile.rs` | 1620 | `nested_intersect_in_exclusion_branch_compiles` | `fn nested_intersect_in_exclusion_branch_compiles() {` |
| `src/automata/lexer/compile.rs` | 1640 | `standalone_exact_repeat_matches_only_at_full_length` | `fn standalone_exact_repeat_matches_only_at_full_length() {` |
| `src/automata/lexer/compile.rs` | 1673 | `product_exact_repeat_matches_only_at_full_length` | `fn product_exact_repeat_matches_only_at_full_length() {` |
| `src/automata/lexer/compile.rs` | 1708 | `product_vbr_exact_repeat_matches_only_at_full_length` | `fn product_vbr_exact_repeat_matches_only_at_full_length() {` |
| `src/automata/lexer/compile.rs` | 1743 | `glrm_chunk16_terminal_family_keeps_exact_repeat_nonfinal_until_16` | `fn glrm_chunk16_terminal_family_keeps_exact_repeat_nonfinal_until_16() {` |
| `src/automata/lexer/compile.rs` | 1800 | `product_vbr_with_literal_prefix_uses_direct_bounded_repeat_tail` | `fn product_vbr_with_literal_prefix_uses_direct_bounded_repeat_tail() {` |
| `src/automata/lexer/compile.rs` | 1841 | `product_vbr_with_literal_prefix_and_regex_suffix_matches` | `fn product_vbr_with_literal_prefix_and_regex_suffix_matches() {` |
| `src/automata/lexer/compile.rs` | 1902 | `prefixed_bounded_repeat_with_regex_suffix_uses_direct_path_without_repeat_cutoff` | `fn prefixed_bounded_repeat_with_regex_suffix_uses_direct_path_without_repeat_cutoff() {` |
| `src/automata/lexer/compile.rs` | 1968 | `prefixed_optional_word_list_expr` | `fn prefixed_optional_word_list_expr(max_pairs: usize) -> Expr {` |
| `src/automata/lexer/compile.rs` | 1996 | `prefixed_optional_choice_uses_direct_component_path_for_bounded_repeat_suffix` | `fn prefixed_optional_choice_uses_direct_component_path_for_bounded_repeat_suffix() {` |
| `src/automata/lexer/compile.rs` | 2012 | `prefixed_optional_word_list_semantics` | `fn prefixed_optional_word_list_semantics() {` |
| `src/automata/lexer/determinize.rs` | 16 | `sparse_to_sorted_vec` | `fn sparse_to_sorted_vec(set: &SparseStateSet) -> Vec<u32> {` |
| `src/automata/lexer/determinize.rs` | 30 | `compute_subset_metadata` | `fn compute_subset_metadata(` |
| `src/automata/lexer/determinize.rs` | 49 | `compute_reachable_groups` | `fn compute_reachable_groups(nfa: &NFA, group_count: usize) -> Vec<BitSet> {` |
| `src/automata/lexer/determinize.rs` | 225 | `is_epsilon_free_deterministic` | `fn is_epsilon_free_deterministic(nfa: &NFA) -> bool {` |
| `src/automata/lexer/determinize.rs` | 242 | `determinize_epsilon_free_deterministic` | `fn determinize_epsilon_free_deterministic(nfa: &NFA, group_count: usize, reachable_groups: &[BitSet]) -> DFA {` |
| `src/automata/lexer/determinize.rs` | 285 | `build_remapped_transitions` | `fn build_remapped_transitions(nfa: &NFA, class_map: &[u8]) -> Vec<Vec<(U8Set, u32)>> {` |
| `src/automata/lexer/determinize.rs` | 309 | `dfs_selective_post_order` | `fn dfs_selective_post_order(` |
| `src/automata/lexer/determinize.rs` | 340 | `precompute_epsilon_closures` | `fn precompute_epsilon_closures(` |
| `src/automata/lexer/determinize.rs` | 392 | `build_start_closure` | `fn build_start_closure(` |
| `src/automata/lexer/determinize.rs` | 420 | `collect_transition_targets` | `fn collect_transition_targets(` |
| `src/automata/lexer/determinize.rs` | 441 | `fast_singleton_without_epsilon` | `fn fast_singleton_without_epsilon(` |
| `src/automata/lexer/determinize.rs` | 461 | `expand_transition_closure` | `fn expand_transition_closure(` |
| `src/automata/lexer/determinize.rs` | 512 | `to_dfa` | `pub fn to_dfa(&self) -> DFA {` |
| `src/automata/lexer/dfa.rs` | 11 | `GroupId` | `pub type GroupId = u32;` |
| `src/automata/lexer/dfa.rs` | 14 | `resized_bitset` | `fn resized_bitset(bits: &BitSet, num_groups: usize) -> BitSet {` |
| `src/automata/lexer/dfa.rs` | 22 | `project_bitset` | `fn project_bitset(bits: &BitSet, num_groups: usize) -> BitSet {` |
| `src/automata/lexer/dfa.rs` | 30 | `excluded_group_indices` | `fn excluded_group_indices(` |
| `src/automata/lexer/dfa.rs` | 50 | `intersection_missing_group_indices` | `fn intersection_missing_group_indices(` |
| `src/automata/lexer/dfa.rs` | 71 | `DFAState` | `pub struct DFAState {` |
| `src/automata/lexer/dfa.rs` | 78 | `DFA` | `pub struct DFA {` |
| `src/automata/lexer/dfa.rs` | 84 | `new` | `pub fn new(num_states: usize) -> Self {` |
| `src/automata/lexer/dfa.rs` | 91 | `num_states` | `pub fn num_states(&self) -> usize {` |
| `src/automata/lexer/dfa.rs` | 95 | `add_state` | `pub(super) fn add_state(&mut self) -> u32 {` |
| `src/automata/lexer/dfa.rs` | 106 | `ensure_group_capacity` | `pub(crate) fn ensure_group_capacity(&mut self, num_groups: usize) {` |
| `src/automata/lexer/dfa.rs` | 115 | `add_transition` | `pub(super) fn add_transition(&mut self, from: u32, byte: u8, to: u32) {` |
| `src/automata/lexer/dfa.rs` | 121 | `set_transitions_from_sorted_entries` | `pub(crate) fn set_transitions_from_sorted_entries(` |
| `src/automata/lexer/dfa.rs` | 131 | `clear_finalizers_for_state` | `pub(super) fn clear_finalizers_for_state(&mut self, state: u32) -> BitSet {` |
| `src/automata/lexer/dfa.rs` | 140 | `overwrite_state_metadata` | `pub(crate) fn overwrite_state_metadata(` |
| `src/automata/lexer/dfa.rs` | 152 | `set_group_u8set` | `pub(crate) fn set_group_u8set(&mut self, group_id: GroupId, set: U8Set) {` |
| `src/automata/lexer/dfa.rs` | 158 | `step` | `pub fn step(&self, state: u32, byte: u8) -> Option<u32> {` |
| `src/automata/lexer/dfa.rs` | 164 | `get_u8set` | `pub fn get_u8set(&self, state: u32) -> U8Set {` |
| `src/automata/lexer/dfa.rs` | 174 | `get_transition` | `pub fn get_transition(&self, state: u32, byte: u8) -> u32 {` |
| `src/automata/lexer/dfa.rs` | 178 | `group_id_to_u8set` | `pub fn group_id_to_u8set(&self, group_id: GroupId) -> &U8Set {` |
| `src/automata/lexer/dfa.rs` | 182 | `finalizers` | `pub fn finalizers(&self, state: u32) -> &BitSet {` |
| `src/automata/lexer/dfa.rs` | 186 | `possible_future_group_ids` | `pub(crate) fn possible_future_group_ids(&self, state: u32) -> &BitSet {` |
| `src/automata/lexer/dfa.rs` | 190 | `states` | `pub fn states(&self) -> &[DFAState] {` |
| `src/automata/lexer/dfa.rs` | 194 | `states_mut` | `pub(super) fn states_mut(&mut self) -> &mut Vec<DFAState> {` |
| `src/automata/lexer/dfa.rs` | 198 | `num_groups` | `pub(super) fn num_groups(&self) -> usize {` |
| `src/automata/lexer/dfa.rs` | 202 | `set_possible_future_group_ids` | `pub(super) fn set_possible_future_group_ids(&mut self, state: u32, ids: BitSet) {` |
| `src/automata/lexer/dfa.rs` | 208 | `mask_possible_futures` | `pub(crate) fn mask_possible_futures(&mut self, mask: &BitSet) {` |
| `src/automata/lexer/dfa.rs` | 215 | `clone_state` | `pub(super) fn clone_state(&mut self, source: u32) -> u32 {` |
| `src/automata/lexer/dfa.rs` | 224 | `redirect_transitions` | `pub(super) fn redirect_transitions(&mut self, old_target: u32, new_target: u32) {` |
| `src/automata/lexer/dfa.rs` | 234 | `apply_group_exclusions` | `pub(crate) fn apply_group_exclusions(` |
| `src/automata/lexer/dfa.rs` | 254 | `apply_group_intersections` | `pub(crate) fn apply_group_intersections(` |
| `src/automata/lexer/dfa.rs` | 270 | `project_groups` | `pub(crate) fn project_groups(&self, num_groups: usize) -> DFA {` |
| `src/automata/lexer/dfa.rs` | 295 | `state_mut` | `fn state_mut(&mut self, state: u32) -> Option<&mut DFAState> {` |
| `src/automata/lexer/dfa.rs` | 299 | `resize_state_group_bits` | `fn resize_state_group_bits(state: &mut DFAState, num_groups: usize) {` |
| `src/automata/lexer/lightweight/nfa.rs` | 9 | `TransitionTable` | `type TransitionTable = [u32; 256];` |
| `src/automata/lexer/lightweight/nfa.rs` | 12 | `ProductState` | `struct ProductState {` |
| `src/automata/lexer/lightweight/nfa.rs` | 18 | `State` | `pub struct State {` |
| `src/automata/lexer/lightweight/nfa.rs` | 25 | `Nfa` | `pub struct Nfa {` |
| `src/automata/lexer/lightweight/nfa.rs` | 33 | `new` | `pub fn new(num_states: usize) -> Self {` |
| `src/automata/lexer/lightweight/nfa.rs` | 42 | `with_flags` | `pub fn with_flags(` |
| `src/automata/lexer/lightweight/nfa.rs` | 56 | `num_states` | `pub fn num_states(&self) -> usize {` |
| `src/automata/lexer/lightweight/nfa.rs` | 60 | `is_deterministic` | `pub fn is_deterministic(&self) -> bool {` |
| `src/automata/lexer/lightweight/nfa.rs` | 64 | `is_minimal` | `pub fn is_minimal(&self) -> bool {` |
| `src/automata/lexer/lightweight/nfa.rs` | 68 | `add_state` | `pub fn add_state(&mut self) -> u32 {` |
| `src/automata/lexer/lightweight/nfa.rs` | 76 | `add_transition` | `pub fn add_transition(&mut self, from: u32, byte: u8, to: u32) {` |
| `src/automata/lexer/lightweight/nfa.rs` | 80 | `add_u8set_transition` | `pub fn add_u8set_transition(&mut self, from: u32, set: U8Set, to: u32) {` |
| `src/automata/lexer/lightweight/nfa.rs` | 88 | `add_epsilon` | `pub fn add_epsilon(&mut self, from: u32, to: u32) {` |
| `src/automata/lexer/lightweight/nfa.rs` | 96 | `set_end` | `pub fn set_end(&mut self, state: u32, is_end: bool) {` |
| `src/automata/lexer/lightweight/nfa.rs` | 103 | `step` | `pub fn step(&self, state: u32, byte: u8) -> Option<u32> {` |
| `src/automata/lexer/lightweight/nfa.rs` | 113 | `accepting_states` | `pub fn accepting_states(&self) -> impl Iterator<Item = u32> + '_ {` |
| `src/automata/lexer/lightweight/nfa.rs` | 121 | `determinize` | `pub fn determinize(&self) -> Self {` |
| `src/automata/lexer/lightweight/nfa.rs` | 131 | `minimize` | `pub fn minimize(&self) -> Self {` |
| `src/automata/lexer/lightweight/nfa.rs` | 142 | `epsilon` | `pub fn epsilon() -> Self {` |
| `src/automata/lexer/lightweight/nfa.rs` | 155 | `from_minimal_lexer_dfa` | `pub fn from_minimal_lexer_dfa(dfa: &LexerDfa) -> Self {` |
| `src/automata/lexer/lightweight/nfa.rs` | 159 | `to_lexer_dfa` | `pub fn to_lexer_dfa(&self) -> LexerDfa {` |
| `src/automata/lexer/lightweight/nfa.rs` | 163 | `concatenate` | `pub fn concatenate(&self, rhs: &Self) -> Self {` |
| `src/automata/lexer/lightweight/nfa.rs` | 190 | `union` | `pub fn union(&self, rhs: &Self) -> Self {` |
| `src/automata/lexer/lightweight/nfa.rs` | 224 | `subtract` | `pub fn subtract(&self, rhs: &Self) -> Self {` |
| `src/automata/lexer/lightweight/nfa.rs` | 263 | `transition_tables` | `fn transition_tables(&self) -> Vec<TransitionTable> {` |
| `src/automata/lexer/lightweight/nfa.rs` | 279 | `to_lexer_nfa` | `fn to_lexer_nfa(&self) -> LexerNfa {` |
| `src/automata/lexer/lightweight/nfa.rs` | 298 | `to_lexer_dfa_impl` | `fn to_lexer_dfa_impl(&self) -> LexerDfa {` |
| `src/automata/lexer/lightweight/nfa.rs` | 322 | `from_lexer_dfa_impl` | `fn from_lexer_dfa_impl(dfa: &LexerDfa, minimal: bool) -> Self {` |
| `src/automata/lexer/lightweight/nfa.rs` | 345 | `as_deterministic` | `fn as_deterministic(&self) -> Self {` |
| `src/automata/lexer/lightweight/nfa.rs` | 353 | `as_minimal` | `fn as_minimal(&self) -> Self {` |
| `src/automata/lexer/lightweight/nfa.rs` | 361 | `product_accepts` | `fn product_accepts(product: ProductState, lhs: &Self, rhs: &Self) -> bool {` |
| `src/automata/lexer/lightweight/nfa.rs` | 370 | `product_successors` | `fn product_successors(` |
| `src/automata/lexer/lightweight/nfa.rs` | 397 | `product_state_id` | `fn product_state_id(` |
| `src/automata/lexer/lightweight/nfa.rs` | 413 | `group_target_bytes` | `fn group_target_bytes(transitions: &[(U8Set, u32)]) -> HashMap<u32, BTreeSet<u8>> {` |
| `src/automata/lexer/lightweight/nfa.rs` | 423 | `sorted_transition_entries` | `fn sorted_transition_entries(target_bytes: HashMap<u32, BTreeSet<u8>>) -> Vec<(u8, u32)> {` |
| `src/automata/lexer/lightweight/nfa.rs` | 434 | `group_dfa_transition_bytes` | `fn group_dfa_transition_bytes(state: &super::super::dfa::DFAState) -> HashMap<u32, U8Set> {` |
| `src/automata/lexer/minimize.rs` | 16 | `TopologyPrerefine` | `enum TopologyPrerefine {` |
| `src/automata/lexer/minimize.rs` | 25 | `partition_by_finalizers` | `fn partition_by_finalizers(dfa: &DFA) -> (Vec<u32>, Vec<Vec<u32>>) {` |
| `src/automata/lexer/minimize.rs` | 45 | `clear_possible_futures_for_minimization` | `fn clear_possible_futures_for_minimization(dfa: &mut DFA) {` |
| `src/automata/lexer/minimize.rs` | 50 | `dedup_adjacency` | `fn dedup_adjacency(dfa: &DFA) -> Vec<Vec<usize>> {` |
| `src/automata/lexer/minimize.rs` | 63 | `compute_post_order` | `fn compute_post_order(adj: &[Vec<usize>]) -> Vec<usize> {` |
| `src/automata/lexer/minimize.rs` | 96 | `reverse_adjacency` | `fn reverse_adjacency(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {` |
| `src/automata/lexer/minimize.rs` | 106 | `compute_kosaraju_scc_ids` | `fn compute_kosaraju_scc_ids(adj: &[Vec<usize>], post_order: &[usize]) -> Vec<u32> {` |
| `src/automata/lexer/minimize.rs` | 133 | `build_blocks_from_labels` | `fn build_blocks_from_labels(labels: &[u32], num_labels: u32) -> (Vec<u32>, Vec<Vec<u32>>) {` |
| `src/automata/lexer/minimize.rs` | 145 | `has_self_loops` | `fn has_self_loops(dfa: &DFA) -> bool {` |
| `src/automata/lexer/minimize.rs` | 164 | `iterative_signature_refine` | `fn iterative_signature_refine(dfa: &DFA, initial_blocks: Vec<Vec<u32>>, max_iterations: u32) -> Option<Vec<Vec<u32>>> {` |
| `src/automata/lexer/minimize.rs` | 222 | `topology_prerefine_partition` | `fn topology_prerefine_partition(dfa: &DFA, partition: &[u32]) -> TopologyPrerefine {` |
| `src/automata/lexer/minimize.rs` | 269 | `build_inverse_transitions` | `fn build_inverse_transitions(dfa: &DFA) -> Vec<Vec<(u8, u32)>> {` |
| `src/automata/lexer/minimize.rs` | 279 | `hopcroft_refine_partition` | `fn hopcroft_refine_partition(` |
| `src/automata/lexer/minimize.rs` | 420 | `compute_tarjan_scc_ids` | `fn compute_tarjan_scc_ids(adj: &[Vec<usize>]) -> (Vec<u32>, u32) {` |
| `src/automata/lexer/minimize.rs` | 483 | `distinct_fingerprint_count` | `pub(crate) fn distinct_fingerprint_count(&self) -> usize {` |
| `src/automata/lexer/minimize.rs` | 509 | `minimize` | `pub fn minimize(&self) -> DFA {` |
| `src/automata/lexer/minimize.rs` | 516 | `minimize_with_state_mapping` | `pub fn minimize_with_state_mapping(&self) -> (DFA, Vec<u32>) {` |
| `src/automata/lexer/minimize.rs` | 526 | `minimize_with_state_mapping_preserve_all_states` | `pub fn minimize_with_state_mapping_preserve_all_states(&self) -> (DFA, Vec<u32>) {` |
| `src/automata/lexer/minimize.rs` | 530 | `minimize_impl` | `fn minimize_impl(&self, drop_unreachable: bool) -> (DFA, Vec<u32>) {` |
| `src/automata/lexer/minimize.rs` | 587 | `remove_unreachable_states_with_mapping` | `fn remove_unreachable_states_with_mapping(&mut self) -> Vec<u32> {` |
| `src/automata/lexer/minimize.rs` | 593 | `remove_unreachable_states_with_roots_with_mapping` | `fn remove_unreachable_states_with_roots_with_mapping(&mut self, extra_roots: &[u32]) -> Vec<u32> {` |
| `src/automata/lexer/minimize.rs` | 653 | `recompute_possible_futures` | `pub(crate) fn recompute_possible_futures(&mut self) {` |
| `src/automata/lexer/minimize.rs` | 758 | `rebuild_from_blocks` | `fn rebuild_from_blocks(&self, partition_blocks: Vec<Vec<u32>>) -> DFA {` |
| `src/automata/lexer/minimize.rs` | 763 | `rebuild_from_blocks_with_mapping` | `fn rebuild_from_blocks_with_mapping(&self, mut partition_blocks: Vec<Vec<u32>>) -> (DFA, Vec<u32>) {` |
| `src/automata/lexer/minimize.rs` | 813 | `compose_mappings` | `fn compose_mappings(first: &[u32], second: &[u32]) -> Vec<u32> {` |
| `src/automata/lexer/nfa.rs` | 12 | `NFAState` | `pub struct NFAState {` |
| `src/automata/lexer/nfa.rs` | 19 | `new` | `fn new() -> Self {` |
| `src/automata/lexer/nfa.rs` | 29 | `CompactNFA` | `pub(crate) struct CompactNFA {` |
| `src/automata/lexer/nfa.rs` | 35 | `NFA` | `pub struct NFA {` |
| `src/automata/lexer/nfa.rs` | 40 | `build_states` | `fn build_states(count: usize) -> Vec<NFAState> {` |
| `src/automata/lexer/nfa.rs` | 49 | `new` | `pub fn new(num_states: usize) -> Self {` |
| `src/automata/lexer/nfa.rs` | 57 | `num_states` | `pub fn num_states(&self) -> usize {` |
| `src/automata/lexer/nfa.rs` | 61 | `add_state` | `pub fn add_state(&mut self) -> u32 {` |
| `src/automata/lexer/nfa.rs` | 67 | `add_transition` | `pub fn add_transition(&mut self, from: u32, byte: u8, to: u32) {` |
| `src/automata/lexer/nfa.rs` | 71 | `add_u8set_transition` | `pub fn add_u8set_transition(&mut self, from: u32, set: U8Set, to: u32) {` |
| `src/automata/lexer/nfa.rs` | 77 | `add_epsilon` | `pub fn add_epsilon(&mut self, from: u32, to: u32) {` |
| `src/automata/lexer/nfa.rs` | 83 | `add_finalizer` | `pub fn add_finalizer(&mut self, state: u32, group_id: GroupId) {` |
| `src/automata/lexer/nfa.rs` | 89 | `epsilon_closure` | `pub fn epsilon_closure(&self, states: &BTreeSet<u32>) -> BTreeSet<u32> {` |
| `src/automata/lexer/nfa.rs` | 104 | `condense_epsilon_sccs` | `pub fn condense_epsilon_sccs(&mut self) {` |
| `src/automata/lexer/nfa.rs` | 175 | `build_compact_nfa` | `pub(crate) fn build_compact_nfa(&self) -> CompactNFA {` |
| `src/automata/lexer/nfa.rs` | 191 | `compute_equivalence_classes` | `pub(crate) fn compute_equivalence_classes(&self) -> (Vec<u8>, usize, Vec<Vec<u8>>) {` |
| `src/automata/lexer/nfa.rs` | 208 | `build_condensed_states` | `fn build_condensed_states(&self, scc_map: &[usize], scc_count: usize) -> Vec<NFAState> {` |
| `src/automata/lexer/nfa.rs` | 216 | `merge_condensed_state` | `fn merge_condensed_state(` |
| `src/automata/lexer/nfa.rs` | 239 | `dedup_epsilon_transitions` | `fn dedup_epsilon_transitions(states: &mut [NFAState]) {` |
| `src/automata/lexer/nfa.rs` | 246 | `refine_partitions` | `fn refine_partitions(partitions: Vec<U8Set>, split: U8Set) -> Vec<U8Set> {` |
| `src/automata/lexer/nfa.rs` | 261 | `build_equivalence_class_outputs` | `fn build_equivalence_class_outputs(partitions: &[U8Set]) -> (Vec<u8>, Vec<Vec<u8>>) {` |
| `src/automata/lexer/regex.rs` | 10 | `choice_or_single` | `fn choice_or_single(mut options: Vec<Expr>) -> Expr {` |
| `src/automata/lexer/regex.rs` | 18 | `sequence_or_single` | `fn sequence_or_single(mut parts: Vec<Expr>) -> Expr {` |
| `src/automata/lexer/regex.rs` | 26 | `repeat_expr` | `fn repeat_expr(expr: Expr, min: usize, max: Option<usize>) -> Expr {` |
| `src/automata/lexer/regex.rs` | 34 | `ascii_digit_set` | `fn ascii_digit_set() -> U8Set {` |
| `src/automata/lexer/regex.rs` | 38 | `ascii_space_set` | `fn ascii_space_set() -> U8Set {` |
| `src/automata/lexer/regex.rs` | 42 | `ascii_word_set` | `fn ascii_word_set() -> U8Set {` |
| `src/automata/lexer/regex.rs` | 46 | `escaped_class_set` | `fn escaped_class_set(escaped: u8) -> Option<U8Set> {` |
| `src/automata/lexer/regex.rs` | 55 | `parse_regex` | `pub fn parse_regex(pattern: &str, utf8: bool) -> Expr {` |
| `src/automata/lexer/regex.rs` | 65 | `unescape_literal` | `pub(crate) fn unescape_literal(input: &[u8]) -> Vec<u8> {` |
| `src/automata/lexer/regex.rs` | 85 | `parse_alternation` | `fn parse_alternation(input: &[u8], pos: usize, utf8: bool) -> (Expr, usize) {` |
| `src/automata/lexer/regex.rs` | 96 | `parse_sequence` | `fn parse_sequence(input: &[u8], pos: usize, utf8: bool) -> (Expr, usize) {` |
| `src/automata/lexer/regex.rs` | 112 | `parse_quantified` | `fn parse_quantified(input: &[u8], pos: usize, utf8: bool) -> (Expr, usize) {` |
| `src/automata/lexer/regex.rs` | 144 | `consume_lazy_suffix` | `fn consume_lazy_suffix(input: &[u8], pos: usize) -> usize {` |
| `src/automata/lexer/regex.rs` | 152 | `parse_repetition_bounds` | `fn parse_repetition_bounds(input: &[u8], pos: usize) -> (usize, Option<usize>, usize) {` |
| `src/automata/lexer/regex.rs` | 172 | `parse_usize` | `fn parse_usize(input: &[u8], pos: usize) -> (usize, usize) {` |
| `src/automata/lexer/regex.rs` | 182 | `parse_atom` | `fn parse_atom(input: &[u8], pos: usize, utf8: bool) -> (Expr, usize) {` |
| `src/automata/lexer/regex.rs` | 202 | `parse_group` | `fn parse_group(input: &[u8], pos: usize, utf8: bool) -> (Expr, usize) {` |
| `src/automata/lexer/regex.rs` | 211 | `consume_group_prefix` | `fn consume_group_prefix(input: &[u8], pos: usize) -> usize {` |
| `src/automata/lexer/regex.rs` | 231 | `consume_named_group_name` | `fn consume_named_group_name(input: &[u8], mut pos: usize) -> Option<usize> {` |
| `src/automata/lexer/regex.rs` | 243 | `parse_char_class_byte` | `fn parse_char_class_byte(input: &[u8], pos: usize) -> Option<(u8, usize)> {` |
| `src/automata/lexer/regex.rs` | 255 | `parse_char_class` | `fn parse_char_class(input: &[u8], pos: usize, utf8: bool) -> (Expr, usize) {` |
| `src/automata/lexer/regex.rs` | 301 | `parse_escape_class_set` | `fn parse_escape_class_set(input: &[u8], pos: usize) -> Option<(U8Set, usize)> {` |
| `src/automata/lexer/regex.rs` | 309 | `utf8_aware_negated_ascii_class` | `fn utf8_aware_negated_ascii_class(excluded: U8Set) -> Expr {` |
| `src/automata/lexer/regex.rs` | 367 | `parse_escape` | `fn parse_escape(input: &[u8], pos: usize, utf8: bool) -> (Expr, usize) {` |
| `src/automata/lexer/regex.rs` | 383 | `negated_ascii_class` | `fn negated_ascii_class(excluded: U8Set, utf8: bool) -> Expr {` |
| `src/automata/lexer/regex.rs` | 391 | `parse_escape_byte` | `fn parse_escape_byte(input: &[u8], pos: usize) -> u8 {` |
| `src/automata/lexer/regex.rs` | 406 | `escape_len` | `fn escape_len(input: &[u8], pos: usize) -> usize {` |
| `src/automata/lexer/regex.rs` | 414 | `hex_digit` | `fn hex_digit(b: u8) -> u8 {` |
| `src/automata/lexer/tokenizer.rs` | 16 | `Tokenizer` | `pub struct Tokenizer {` |
| `src/automata/lexer/tokenizer.rs` | 27 | `TokenizerMatch` | `pub struct TokenizerMatch {` |
| `src/automata/lexer/tokenizer.rs` | 34 | `TokenizerExecResult` | `pub struct TokenizerExecResult {` |
| `src/automata/lexer/tokenizer.rs` | 39 | `into_longest_matches` | `fn into_longest_matches(matches: FxHashMap<TerminalID, (usize, u32)>) -> Vec<TokenizerMatch> {` |
| `src/automata/lexer/tokenizer.rs` | 50 | `group_matches_by_width` | `fn group_matches_by_width(matches: Vec<TokenizerMatch>) -> Vec<(usize, BTreeSet<TerminalID>)> {` |
| `src/automata/lexer/tokenizer.rs` | 58 | `TerminalFilteredDfa` | `struct TerminalFilteredDfa {` |
| `src/automata/lexer/tokenizer.rs` | 66 | `start_state` | `pub fn start_state(&self) -> u32 {` |
| `src/automata/lexer/tokenizer.rs` | 74 | `isolate_start_state_and_drain_nullable_terminals` | `pub fn isolate_start_state_and_drain_nullable_terminals(&mut self) -> BTreeSet<TerminalID> {` |
| `src/automata/lexer/tokenizer.rs` | 89 | `isolate_start_state` | `fn isolate_start_state(&mut self) {` |
| `src/automata/lexer/tokenizer.rs` | 98 | `step` | `pub fn step(&self, state: u32, byte: u8) -> Option<u32> {` |
| `src/automata/lexer/tokenizer.rs` | 102 | `get_transition` | `pub fn get_transition(&self, state: u32, byte: u8) -> u32 {` |
| `src/automata/lexer/tokenizer.rs` | 106 | `run` | `pub fn run(&self, input: &[u8]) -> u32 {` |
| `src/automata/lexer/tokenizer.rs` | 113 | `matched_terminals` | `pub fn matched_terminals(&self, state: u32) -> BTreeSet<TerminalID> {` |
| `src/automata/lexer/tokenizer.rs` | 117 | `matched_terminals_iter` | `pub(crate) fn matched_terminals_iter(` |
| `src/automata/lexer/tokenizer.rs` | 127 | `possible_future_terminals_iter` | `pub(crate) fn possible_future_terminals_iter(` |
| `src/automata/lexer/tokenizer.rs` | 137 | `possible_future_terminals` | `pub fn possible_future_terminals(&self, state: u32) -> &BitSet {` |
| `src/automata/lexer/tokenizer.rs` | 141 | `is_end` | `pub fn is_end(&self, state: u32) -> bool {` |
| `src/automata/lexer/tokenizer.rs` | 145 | `num_states` | `pub fn num_states(&self) -> u32 {` |
| `src/automata/lexer/tokenizer.rs` | 149 | `execute_from_state_all_widths` | `pub(crate) fn execute_from_state_all_widths(` |
| `src/automata/lexer/tokenizer.rs` | 165 | `execute_from_state` | `pub fn execute_from_state(&self, input: &[u8], start: u32) -> TokenizerExecResult {` |
| `src/automata/lexer/tokenizer.rs` | 177 | `execute_from_state_end_only` | `pub(crate) fn execute_from_state_end_only(&self, input: &[u8], start: u32) -> Option<u32> {` |
| `src/automata/lexer/tokenizer.rs` | 181 | `execute_all_matches` | `pub fn execute_all_matches(&self, input: &[u8], start: u32) -> TokenizerResult {` |
| `src/automata/lexer/tokenizer.rs` | 190 | `initial_state` | `pub fn initial_state(&self) -> u32 {` |
| `src/automata/lexer/tokenizer.rs` | 194 | `initial_state_id` | `pub fn initial_state_id(&self) -> u32 {` |
| `src/automata/lexer/tokenizer.rs` | 198 | `tokens_accessible_from_state` | `pub fn tokens_accessible_from_state(&self, state: u32) -> &BitSet {` |
| `src/automata/lexer/tokenizer.rs` | 223 | `scan_terminal_matches_from_state` | `pub fn scan_terminal_matches_from_state(` |
| `src/automata/lexer/tokenizer.rs` | 259 | `has_incoming_start_transitions` | `fn has_incoming_start_transitions(&self, start: u32) -> bool {` |
| `src/automata/lexer/tokenizer.rs` | 266 | `record_all_matches` | `fn record_all_matches(&self, matches: &mut Vec<TokenizerMatch>, state: u32, width: usize) {` |
| `src/automata/lexer/tokenizer.rs` | 274 | `record_longest_matches` | `fn record_longest_matches(` |
| `src/automata/lexer/tokenizer.rs` | 285 | `scan_input` | `fn scan_input<R>(` |
| `src/automata/lexer/tokenizer.rs` | 301 | `filter_dfa_for_terminals` | `fn filter_dfa_for_terminals(` |
| `src/automata/lexer/tokenizer.rs` | 400 | `simplify_for_terminals` | `pub fn simplify_for_terminals(` |
| `src/automata/lexer/tokenizer.rs` | 488 | `TokenizerResult` | `pub struct TokenizerResult {` |
| `src/automata/unweighted/determinize.rs` | 14 | `subset_is_accepting` | `fn subset_is_accepting(nfa: &NFA, subset: &[u32]) -> bool {` |
| `src/automata/unweighted/determinize.rs` | 18 | `gather_label_targets` | `fn gather_label_targets(nfa: &NFA, subset: &[u32]) -> BTreeMap<Label, BTreeSet<u32>> {` |
| `src/automata/unweighted/determinize.rs` | 31 | `get_or_create_subset_state` | `fn get_or_create_subset_state(` |
| `src/automata/unweighted/determinize.rs` | 47 | `epsilon_closure` | `fn epsilon_closure(nfa: &NFA, seeds: &[u32]) -> BTreeSet<u32> {` |
| `src/automata/unweighted/determinize.rs` | 65 | `determinize` | `pub fn determinize(nfa: &NFA) -> DFA {` |
| `src/automata/unweighted/dfa.rs` | 3 | `Label` | `pub type Label = i32;` |
| `src/automata/unweighted/dfa.rs` | 6 | `DFAState` | `pub struct DFAState {` |
| `src/automata/unweighted/dfa.rs` | 12 | `DFA` | `pub struct DFA {` |
| `src/automata/unweighted/dfa.rs` | 17 | `has_self_loop` | `fn has_self_loop(state_id: usize, state: &DFAState) -> bool {` |
| `src/automata/unweighted/dfa.rs` | 21 | `visit_successors` | `fn visit_successors(` |
| `src/automata/unweighted/dfa.rs` | 47 | `new` | `pub fn new() -> Self {` |
| `src/automata/unweighted/dfa.rs` | 54 | `num_states` | `pub fn num_states(&self) -> usize {` |
| `src/automata/unweighted/dfa.rs` | 58 | `add_state` | `pub fn add_state(&mut self) -> u32 {` |
| `src/automata/unweighted/dfa.rs` | 64 | `add_transition` | `pub fn add_transition(&mut self, from: u32, label: Label, to: u32) {` |
| `src/automata/unweighted/dfa.rs` | 70 | `set_accepting` | `pub fn set_accepting(&mut self, state: u32, is_accepting: bool) {` |
| `src/automata/unweighted/dfa.rs` | 77 | `is_acyclic` | `pub fn is_acyclic(&self) -> bool {` |
| `src/automata/unweighted/dfa.rs` | 97 | `fmt` | `fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {` |
| `src/automata/unweighted/minimize_acyclic.rs` | 13 | `StateSignature` | `struct StateSignature {` |
| `src/automata/unweighted/minimize_acyclic.rs` | 19 | `reverse_topological_order` | `fn reverse_topological_order(dfa: &DFA) -> Vec<usize> {` |
| `src/automata/unweighted/minimize_acyclic.rs` | 20 | `dfs` | `fn dfs(state_id: usize, dfa: &DFA, visited: &mut [bool], order: &mut Vec<usize>) {` |
| `src/automata/unweighted/minimize_acyclic.rs` | 40 | `state_signature` | `fn state_signature(` |
| `src/automata/unweighted/minimize_acyclic.rs` | 65 | `build_minimized_acyclic_dfa` | `fn build_minimized_acyclic_dfa(` |
| `src/automata/unweighted/minimize_acyclic.rs` | 116 | `minimize_acyclic` | `pub fn minimize_acyclic(dfa: &DFA) -> DFA {` |
| `src/automata/unweighted/minimize_cyclic.rs` | 11 | `collect_reachable_alphabet` | `fn collect_reachable_alphabet(dfa: &DFA, reachable: &[usize]) -> Vec<Label> {` |
| `src/automata/unweighted/minimize_cyclic.rs` | 21 | `dense_reachable_states` | `fn dense_reachable_states(reachable: &[usize]) -> (HashMap<usize, usize>, Vec<usize>) {` |
| `src/automata/unweighted/minimize_cyclic.rs` | 31 | `initial_partition` | `fn initial_partition(dfa: &DFA, dense_to_state: &[usize], dead: usize) -> Vec<usize> {` |
| `src/automata/unweighted/minimize_cyclic.rs` | 55 | `refine_partition` | `fn refine_partition(` |
| `src/automata/unweighted/minimize_cyclic.rs` | 91 | `build_minimized_cyclic_dfa` | `fn build_minimized_cyclic_dfa(` |
| `src/automata/unweighted/minimize_cyclic.rs` | 152 | `minimize_cyclic` | `pub fn minimize_cyclic(dfa: &DFA) -> DFA {` |
| `src/automata/unweighted/minimize_cyclic.rs` | 197 | `reachable_states` | `fn reachable_states(dfa: &DFA) -> Vec<usize> {` |
| `src/automata/unweighted/nfa.rs` | 12 | `visit_successors` | `fn visit_successors(state: &NFAState, mut visit: impl FnMut(u32)) {` |
| `src/automata/unweighted/nfa.rs` | 25 | `NFAState` | `pub struct NFAState {` |
| `src/automata/unweighted/nfa.rs` | 36 | `NFA` | `pub struct NFA {` |
| `src/automata/unweighted/nfa.rs` | 43 | `new` | `pub fn new() -> Self {` |
| `src/automata/unweighted/nfa.rs` | 51 | `new_empty` | `pub fn new_empty() -> Self {` |
| `src/automata/unweighted/nfa.rs` | 59 | `add_state` | `pub fn add_state(&mut self) -> u32 {` |
| `src/automata/unweighted/nfa.rs` | 66 | `num_states` | `pub fn num_states(&self) -> usize {` |
| `src/automata/unweighted/nfa.rs` | 71 | `add_transition` | `pub fn add_transition(&mut self, from: u32, label: Label, to: u32) {` |
| `src/automata/unweighted/nfa.rs` | 80 | `add_epsilon` | `pub fn add_epsilon(&mut self, from: u32, to: u32) {` |
| `src/automata/unweighted/nfa.rs` | 85 | `set_accepting` | `pub fn set_accepting(&mut self, state: u32) {` |
| `src/automata/unweighted/nfa.rs` | 90 | `is_accepting` | `pub fn is_accepting(&self, state: u32) -> bool {` |
| `src/automata/unweighted/nfa.rs` | 98 | `is_acyclic` | `pub fn is_acyclic(&self) -> bool {` |
| `src/automata/unweighted/nfa.rs` | 102 | `visit` | `fn visit(s: usize, states: &[NFAState], color: &mut [u8]) -> bool {` |
| `src/automata/unweighted/nfa.rs` | 140 | `fmt` | `fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {` |
| `src/automata/unweighted/subtract.rs` | 12 | `ProductState` | `struct ProductState {` |
| `src/automata/unweighted/subtract.rs` | 17 | `get_or_create_product_state` | `fn get_or_create_product_state(` |
| `src/automata/unweighted/subtract.rs` | 32 | `subtract` | `pub fn subtract(left: &DFA, right: &DFA) -> DFA {` |
| `src/automata/weighted/determinize.rs` | 15 | `union_state_weight` | `fn union_state_weight(weights: &mut FxHashMap<u32, Weight>, state_id: u32, add: Weight) {` |
| `src/automata/weighted/determinize.rs` | 31 | `subset_final_weight` | `fn subset_final_weight(nwa: &NWA, subset_entries: &[(u32, Weight)]) -> Weight {` |
| `src/automata/weighted/determinize.rs` | 41 | `seed_start_subset` | `fn seed_start_subset(nwa: &NWA) -> FxHashMap<u32, Weight> {` |
| `src/automata/weighted/determinize.rs` | 49 | `determinize` | `pub fn determinize(nwa: &NWA) -> Result<DWA, GlrMaskError> {` |
| `src/automata/weighted/determinize.rs` | 58 | `canonicalize` | `fn canonicalize(subset: &FxHashMap<u32, Weight>) -> Vec<(u32, Weight)> {` |
| `src/automata/weighted/determinize.rs` | 67 | `epsilon_closure` | `fn epsilon_closure(nwa: &NWA, seed: FxHashMap<u32, Weight>) -> FxHashMap<u32, Weight> {` |
| `src/automata/weighted/dwa.rs` | 10 | `DWAState` | `pub struct DWAState {` |
| `src/automata/weighted/dwa.rs` | 16 | `DWA` | `pub struct DWA {` |
| `src/automata/weighted/dwa.rs` | 22 | `DwaStats` | `pub struct DwaStats {` |
| `src/automata/weighted/dwa.rs` | 34 | `EncodedTokenSet` | `type EncodedTokenSet = Vec<[u32; 2]>;` |
| `src/automata/weighted/dwa.rs` | 38 | `WeightPoolEntry` | `struct WeightPoolEntry {` |
| `src/automata/weighted/dwa.rs` | 45 | `DWAStateSerde` | `struct DWAStateSerde {` |
| `src/automata/weighted/dwa.rs` | 53 | `DWASerde` | `struct DWASerde {` |
| `src/automata/weighted/dwa.rs` | 63 | `serialize` | `fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {` |
| `src/automata/weighted/dwa.rs` | 142 | `deserialize` | `fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {` |
| `src/automata/weighted/dwa.rs` | 216 | `new` | `pub fn new(_num_tsids: u32, _max_token: u32) -> Self {` |
| `src/automata/weighted/dwa.rs` | 224 | `states` | `pub fn states(&self) -> &[DWAState] {` |
| `src/automata/weighted/dwa.rs` | 229 | `states_mut` | `pub fn states_mut(&mut self) -> &mut Vec<DWAState> {` |
| `src/automata/weighted/dwa.rs` | 234 | `start_state` | `pub fn start_state(&self) -> u32 {` |
| `src/automata/weighted/dwa.rs` | 238 | `from_parts` | `pub fn from_parts(states: Vec<DWAState>, start_state: u32) -> Self {` |
| `src/automata/weighted/dwa.rs` | 242 | `set_start_state` | `pub fn set_start_state(&mut self, state: u32) {` |
| `src/automata/weighted/dwa.rs` | 246 | `add_state` | `pub fn add_state(&mut self) -> u32 {` |
| `src/automata/weighted/dwa.rs` | 252 | `num_states` | `pub fn num_states(&self) -> u32 {` |
| `src/automata/weighted/dwa.rs` | 256 | `num_transitions` | `pub fn num_transitions(&self) -> usize {` |
| `src/automata/weighted/dwa.rs` | 260 | `stats` | `pub fn stats(&self) -> DwaStats {` |
| `src/automata/weighted/dwa.rs` | 306 | `set_final_weight` | `pub fn set_final_weight(&mut self, state: u32, weight: Weight) {` |
| `src/automata/weighted/dwa.rs` | 312 | `add_transition` | `pub fn add_transition(&mut self, from: u32, label: Label, to: u32, weight: Weight) {` |
| `src/automata/weighted/dwa.rs` | 318 | `eval_word` | `pub fn eval_word(&self, word: &[Label]) -> Weight {` |
| `src/automata/weighted/dwa.rs` | 335 | `clip_weights` | `pub fn clip_weights(&mut self, max_token: u32) {` |
| `src/automata/weighted/dwa.rs` | 349 | `labels` | `pub fn labels(&self) -> Vec<Label> {` |
| `src/automata/weighted/dwa.rs` | 358 | `is_acyclic` | `pub fn is_acyclic(&self) -> bool {` |
| `src/automata/weighted/dwa.rs` | 359 | `for_each_successor` | `fn for_each_successor(state: &DWAState, mut visit: impl FnMut(u32)) {` |
| `src/automata/weighted/dwa.rs` | 379 | `visit` | `fn visit(state_id: usize, states: &[DWAState], colors: &mut [u8]) -> bool {` |
| `src/automata/weighted/dwa.rs` | 421 | `to_nwa` | `pub fn to_nwa(&self) -> super::nwa::NWA {` |
| `src/automata/weighted/dwa.rs` | 443 | `fmt_dwa_states` | `fn fmt_dwa_states(` |
| `src/automata/weighted/dwa.rs` | 471 | `fmt` | `fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {` |
| `src/automata/weighted/dwa.rs` | 478 | `eq` | `fn eq(&self, other: &Self) -> bool {` |
| `src/automata/weighted/dwa.rs` | 484 | `eq` | `fn eq(&self, other: &Self) -> bool {` |
| `src/automata/weighted/minimize.rs` | 7 | `should_skip_minimization` | `fn should_skip_minimization(dwa: &DWA) -> bool {` |
| `src/automata/weighted/minimize.rs` | 11 | `minimize_if_acyclic` | `fn minimize_if_acyclic(dwa: &DWA, minimize: impl FnOnce(&DWA) -> DWA) -> DWA {` |
| `src/automata/weighted/minimize.rs` | 19 | `minimize` | `pub fn minimize(dwa: &DWA) -> DWA {` |
| `src/automata/weighted/minimize_acyclic.rs` | 16 | `Label` | `type Label = i32;` |
| `src/automata/weighted/minimize_acyclic.rs` | 20 | `weighted_dwa_minimize_profile_enabled` | `fn weighted_dwa_minimize_profile_enabled() -> bool {` |
| `src/automata/weighted/minimize_acyclic.rs` | 25 | `mapped_target` | `fn mapped_target(old_to_new: &[u32], target: u32) -> Option<u32> {` |
| `src/automata/weighted/minimize_acyclic.rs` | 29 | `compute_reachable_from_start` | `fn compute_reachable_from_start(dwa: &DWA, start_state: usize) -> Vec<bool> {` |
| `src/automata/weighted/minimize_acyclic.rs` | 53 | `weight_body_id` | `fn weight_body_id(weight: &Weight) -> usize {` |
| `src/automata/weighted/minimize_acyclic.rs` | 57 | `intersection_memo_key` | `fn intersection_memo_key(left: &Weight, right: &Weight) -> (usize, usize) {` |
| `src/automata/weighted/minimize_acyclic.rs` | 67 | `memoized_intersection` | `fn memoized_intersection(` |
| `src/automata/weighted/minimize_acyclic.rs` | 99 | `push_weights` | `pub fn push_weights(dwa: &mut DWA) -> (bool, Option<Vec<usize>>, Vec<Weight>) {` |
| `src/automata/weighted/minimize_acyclic.rs` | 163 | `compute_topo_order` | `fn compute_topo_order(dwa: &DWA) -> Option<Vec<usize>> {` |
| `src/automata/weighted/minimize_acyclic.rs` | 208 | `compute_heights` | `fn compute_heights(dwa: &DWA, topo: &[usize]) -> Vec<usize> {` |
| `src/automata/weighted/minimize_acyclic.rs` | 227 | `ProductiveTransition` | `struct ProductiveTransition {` |
| `src/automata/weighted/minimize_acyclic.rs` | 233 | `compute_productive_transitions` | `fn compute_productive_transitions(dwa: &DWA, needed: &[Weight]) -> Vec<Vec<ProductiveTransition>> {` |
| `src/automata/weighted/minimize_acyclic.rs` | 268 | `weights_equal_on_domain` | `fn weights_equal_on_domain(w_a: &Weight, w_b: &Weight, domain: &Weight) -> bool {` |
| `src/automata/weighted/minimize_acyclic.rs` | 358 | `token_sets_agree_on_domain` | `fn token_sets_agree_on_domain(` |
| `src/automata/weighted/minimize_acyclic.rs` | 411 | `token_sets_intersect_three` | `fn token_sets_intersect_three(` |
| `src/automata/weighted/minimize_acyclic.rs` | 446 | `token_sets_agree_on_domain_intersection` | `fn token_sets_agree_on_domain_intersection(` |
| `src/automata/weighted/minimize_acyclic.rs` | 522 | `weight_is_disjoint_from_domain_intersection` | `fn weight_is_disjoint_from_domain_intersection(` |
| `src/automata/weighted/minimize_acyclic.rs` | 596 | `weights_equal_on_domain_intersection` | `fn weights_equal_on_domain_intersection(` |
| `src/automata/weighted/minimize_acyclic.rs` | 732 | `are_compatible` | `fn are_compatible(` |
| `src/automata/weighted/minimize_acyclic.rs` | 921 | `ClassProfile` | `struct ClassProfile {` |
| `src/automata/weighted/minimize_acyclic.rs` | 927 | `build_class_profile` | `fn build_class_profile(` |
| `src/automata/weighted/minimize_acyclic.rs` | 950 | `sorted_targets_compatible` | `fn sorted_targets_compatible(class_targets: &[(Label, u32)], group_targets: &[(Label, u32)]) -> bool {` |
| `src/automata/weighted/minimize_acyclic.rs` | 973 | `sorted_weights_compatible_on_domain_intersection` | `fn sorted_weights_compatible_on_domain_intersection(` |
| `src/automata/weighted/minimize_acyclic.rs` | 1040 | `targets_compatible_with_group_map` | `fn targets_compatible_with_group_map(` |
| `src/automata/weighted/minimize_acyclic.rs` | 1051 | `transition_weights_compatible_on_overlap` | `fn transition_weights_compatible_on_overlap(` |
| `src/automata/weighted/minimize_acyclic.rs` | 1097 | `merge_sorted_targets` | `fn merge_sorted_targets(existing: &mut Vec<(Label, u32)>, add: &[(Label, u32)]) {` |
| `src/automata/weighted/minimize_acyclic.rs` | 1130 | `merge_sorted_weights` | `fn merge_sorted_weights(existing: &mut Vec<(Label, Weight)>, add: &[(Label, Weight)]) {` |
| `src/automata/weighted/minimize_acyclic.rs` | 1175 | `build_and_color_hybrid` | `fn build_and_color_hybrid(` |
| `src/automata/weighted/minimize_acyclic.rs` | 1272 | `OverlapMergeGroup` | `struct OverlapMergeGroup {` |
| `src/automata/weighted/minimize_acyclic.rs` | 1489 | `partition_refine_coloring_raw` | `fn partition_refine_coloring_raw(` |
| `src/automata/weighted/minimize_acyclic.rs` | 1557 | `states_raw_equal` | `fn states_raw_equal(` |
| `src/automata/weighted/minimize_acyclic.rs` | 1580 | `try_all_compatible_height_0_coloring` | `fn try_all_compatible_height_0_coloring(` |
| `src/automata/weighted/minimize_acyclic.rs` | 1621 | `MergedStateBuilder` | `struct MergedStateBuilder {` |
| `src/automata/weighted/minimize_acyclic.rs` | 1627 | `default` | `fn default() -> Self {` |
| `src/automata/weighted/minimize_acyclic.rs` | 1636 | `add_final_weight` | `fn add_final_weight(&mut self, weight: &Weight) {` |
| `src/automata/weighted/minimize_acyclic.rs` | 1640 | `add_transition` | `fn add_transition(&mut self, label: Label, target: u32, weight: Weight) {` |
| `src/automata/weighted/minimize_acyclic.rs` | 1654 | `finalize_for_reuse` | `fn finalize_for_reuse(&mut self) {` |
| `src/automata/weighted/minimize_acyclic.rs` | 1667 | `batch_build_weight` | `fn batch_build_weight(pending: Vec<Weight>) -> Weight {` |
| `src/automata/weighted/minimize_acyclic.rs` | 1685 | `merge_state_into_builder` | `fn merge_state_into_builder(` |
| `src/automata/weighted/minimize_acyclic.rs` | 1720 | `reconstruct_dwa` | `fn reconstruct_dwa(start_old: usize, old_to_new: &[u32], builders: Vec<MergedStateBuilder>) -> DWA {` |
| `src/automata/weighted/minimize_acyclic.rs` | 1745 | `canonical_dead_dwa` | `fn canonical_dead_dwa() -> DWA {` |
| `src/automata/weighted/minimize_acyclic.rs` | 1754 | `minimize_acyclic` | `pub fn minimize_acyclic(dwa: &DWA) -> DWA {` |
| `src/automata/weighted/minimize_acyclic.rs` | 1975 | `token_set` | `fn token_set(ranges: &[(u32, u32)]) -> RangeSetBlaze<u32> {` |
| `src/automata/weighted/minimize_acyclic.rs` | 1979 | `weight` | `fn weight(entries: &[(u32, &[(u32, u32)])]) -> Weight {` |
| `src/automata/weighted/minimize_acyclic.rs` | 1988 | `assert_disjoint_matches_overlap` | `fn assert_disjoint_matches_overlap(weight: &Weight, left: &Weight, right: &Weight) {` |
| `src/automata/weighted/minimize_acyclic.rs` | 1996 | `assert_equal_matches_overlap` | `fn assert_equal_matches_overlap(a: &Weight, b: &Weight, left: &Weight, right: &Weight) {` |
| `src/automata/weighted/minimize_acyclic.rs` | 2004 | `transition_map` | `fn transition_map(entries: &[(Label, Weight)]) -> FxHashMap<Label, Weight> {` |
| `src/automata/weighted/minimize_acyclic.rs` | 2009 | `transition_compat_accepts_matching_label_equal_on_overlap_but_different_outside` | `fn transition_compat_accepts_matching_label_equal_on_overlap_but_different_outside() {` |
| `src/automata/weighted/minimize_acyclic.rs` | 2022 | `transition_compat_rejects_class_only_label_active_on_overlap` | `fn transition_compat_rejects_class_only_label_active_on_overlap() {` |
| `src/automata/weighted/minimize_acyclic.rs` | 2035 | `transition_compat_rejects_group_only_label_active_on_overlap` | `fn transition_compat_rejects_group_only_label_active_on_overlap() {` |
| `src/automata/weighted/minimize_acyclic.rs` | 2048 | `transition_compat_rejects_same_target_shape_with_extra_active_label` | `fn transition_compat_rejects_same_target_shape_with_extra_active_label() {` |
| `src/automata/weighted/minimize_acyclic.rs` | 2064 | `transition_compat_accepts_class_and_group_weights_disjoint_from_overlap` | `fn transition_compat_accepts_class_and_group_weights_disjoint_from_overlap() {` |
| `src/automata/weighted/minimize_acyclic.rs` | 2077 | `minimize_acyclic_helpers_match_materialized_overlap_for_empty_tsid_intersection` | `fn minimize_acyclic_helpers_match_materialized_overlap_for_empty_tsid_intersection() {` |
| `src/automata/weighted/minimize_acyclic.rs` | 2089 | `minimize_acyclic_helpers_match_materialized_overlap_for_disjoint_token_domains` | `fn minimize_acyclic_helpers_match_materialized_overlap_for_disjoint_token_domains() {` |
| `src/automata/weighted/minimize_acyclic.rs` | 2101 | `minimize_acyclic_helpers_match_materialized_overlap_when_weight_range_is_missing` | `fn minimize_acyclic_helpers_match_materialized_overlap_when_weight_range_is_missing() {` |
| `src/automata/weighted/minimize_acyclic.rs` | 2113 | `minimize_acyclic_helpers_match_materialized_overlap_for_equal_weights_by_value` | `fn minimize_acyclic_helpers_match_materialized_overlap_for_equal_weights_by_value() {` |
| `src/automata/weighted/minimize_acyclic.rs` | 2125 | `minimize_acyclic_helpers_match_materialized_overlap_when_difference_is_outside_overlap` | `fn minimize_acyclic_helpers_match_materialized_overlap_when_difference_is_outside_overlap() {` |
| `src/automata/weighted/minimize_acyclic.rs` | 2137 | `minimize_acyclic_helpers_match_materialized_overlap_when_difference_is_inside_overlap` | `fn minimize_acyclic_helpers_match_materialized_overlap_when_difference_is_inside_overlap() {` |
| `src/automata/weighted/nwa.rs` | 5 | `Label` | `pub type Label = i32;` |
| `src/automata/weighted/nwa.rs` | 8 | `NWAState` | `pub struct NWAState {` |
| `src/automata/weighted/nwa.rs` | 15 | `NWA` | `pub struct NWA {` |
| `src/automata/weighted/nwa.rs` | 21 | `NwaBody` | `pub struct NwaBody {` |
| `src/automata/weighted/nwa.rs` | 26 | `union` | `pub fn union(left: &Self, right: &Self) -> Self {` |
| `src/automata/weighted/nwa.rs` | 34 | `prune_empty_outgoing` | `fn prune_empty_outgoing(state: &mut NWAState) {` |
| `src/automata/weighted/nwa.rs` | 42 | `new` | `pub fn new(_num_tsids: u32, _max_token: u32) -> Self {` |
| `src/automata/weighted/nwa.rs` | 50 | `states` | `pub fn states(&self) -> &[NWAState] {` |
| `src/automata/weighted/nwa.rs` | 55 | `states_mut` | `pub fn states_mut(&mut self) -> &mut Vec<NWAState> {` |
| `src/automata/weighted/nwa.rs` | 60 | `start_states` | `pub fn start_states(&self) -> &[u32] {` |
| `src/automata/weighted/nwa.rs` | 64 | `from_parts` | `pub fn from_parts(states: Vec<NWAState>, start_states: Vec<u32>) -> Self {` |
| `src/automata/weighted/nwa.rs` | 68 | `set_start_states` | `pub fn set_start_states(&mut self, states: Vec<u32>) {` |
| `src/automata/weighted/nwa.rs` | 72 | `start_states_mut` | `pub fn start_states_mut(&mut self) -> &mut Vec<u32> {` |
| `src/automata/weighted/nwa.rs` | 76 | `add_state` | `pub fn add_state(&mut self) -> u32 {` |
| `src/automata/weighted/nwa.rs` | 82 | `num_states` | `pub fn num_states(&self) -> u32 {` |
| `src/automata/weighted/nwa.rs` | 86 | `set_final_weight` | `pub fn set_final_weight(&mut self, state: u32, weight: Weight) {` |
| `src/automata/weighted/nwa.rs` | 92 | `add_transition` | `pub fn add_transition(&mut self, from: u32, label: Label, to: u32, weight: Weight) {` |
| `src/automata/weighted/nwa.rs` | 98 | `add_epsilon` | `pub fn add_epsilon(&mut self, from: u32, to: u32, weight: Weight) {` |
| `src/automata/weighted/nwa.rs` | 109 | `subtract_final_weights_from_outgoing` | `pub fn subtract_final_weights_from_outgoing(&mut self) {` |
| `src/automata/weighted/nwa.rs` | 130 | `num_transitions` | `pub fn num_transitions(&self) -> usize {` |
| `src/automata/weighted/nwa.rs` | 137 | `body` | `pub fn body(&self) -> NwaBody {` |
| `src/automata/weighted/nwa.rs` | 143 | `append_with_body` | `pub fn append_with_body(&mut self, other: &NWA) -> NwaBody {` |
| `src/automata/weighted/nwa.rs` | 165 | `concatenate_in_place` | `pub fn concatenate_in_place(&mut self, left: &NWA, right_body: &NwaBody) -> NwaBody {` |
| `src/automata/weighted/nwa.rs` | 182 | `union_in_place` | `pub fn union_in_place(&mut self, other: &NWA, existing_body: &NwaBody) -> NwaBody {` |
| `src/automata/weighted/nwa.rs` | 187 | `reverse` | `pub fn reverse(&self) -> Self {` |
| `src/automata/weighted/nwa.rs` | 225 | `is_acyclic` | `pub fn is_acyclic(&self) -> bool {` |
| `src/automata/weighted/nwa.rs` | 226 | `for_each_successor` | `fn for_each_successor(state: &NWAState, mut visit: impl FnMut(u32)) {` |
| `src/automata/weighted/nwa.rs` | 249 | `visit` | `fn visit(state_id: usize, states: &[NWAState], colors: &mut [u8]) -> bool {` |
| `src/automata/weighted/nwa.rs` | 291 | `fmt_nwa_states` | `fn fmt_nwa_states(` |
| `src/automata/weighted/nwa.rs` | 328 | `fmt` | `fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {` |
| `src/compile/options.rs` | 10 | `env_flag_enabled` | `pub(crate) fn env_flag_enabled(name: &str) -> bool {` |
| `src/compile/options.rs` | 20 | `env_flag_enabled_by_default` | `pub(crate) fn env_flag_enabled_by_default(name: &str) -> bool {` |
| `src/compile/options.rs` | 30 | `compact_can_match_before_reconcile_enabled` | `pub(crate) fn compact_can_match_before_reconcile_enabled() -> bool {` |
| `src/compile/options.rs` | 35 | `tokenizer_detail_profile_enabled` | `pub(crate) fn tokenizer_detail_profile_enabled() -> bool {` |
| `src/compile/options.rs` | 46 | `DwaCanMatchMode` | `pub(crate) enum DwaCanMatchMode {` |
| `src/compile/options.rs` | 62 | `does_terminal_reconcile` | `pub(crate) fn does_terminal_reconcile(self) -> bool {` |
| `src/compile/options.rs` | 72 | `does_terminal_compact` | `pub(crate) fn does_terminal_compact(self) -> bool {` |
| `src/compile/options.rs` | 80 | `does_parser_compact` | `pub(crate) fn does_parser_compact(self) -> bool {` |
| `src/compile/options.rs` | 91 | `dwa_can_match_mode` | `pub(crate) fn dwa_can_match_mode() -> DwaCanMatchMode {` |
| `src/compile/options.rs` | 133 | `compile_thread_count` | `pub(crate) fn compile_thread_count() -> Option<usize> {` |
| `src/compile/parser_dwa/builder.rs` | 43 | `ParserDwaBuildInputs` | `pub(crate) struct ParserDwaBuildInputs<'a> {` |
| `src/compile/parser_dwa/builder.rs` | 53 | `ParserDwaBuildOutput` | `pub(crate) struct ParserDwaBuildOutput {` |
| `src/compile/parser_dwa/builder.rs` | 59 | `build_parser_dwa_from_terminal_dwa_with_templates` | `pub(crate) fn build_parser_dwa_from_terminal_dwa_with_templates(` |
| `src/compile/parser_dwa/builder.rs` | 201 | `build_parser_dwa_from_terminal_dwa_with_precomputed_templates` | `pub(crate) fn build_parser_dwa_from_terminal_dwa_with_precomputed_templates(` |
| `src/compile/parser_dwa/compose_nwa.rs` | 27 | `dwa_to_nwa` | `fn dwa_to_nwa(dwa: &DWA) -> NWA {` |
| `src/compile/parser_dwa/compose_nwa.rs` | 48 | `compute_productive_terminal_states` | `fn compute_productive_terminal_states(summaries: &StateSummaries) -> Vec<bool> {` |
| `src/compile/parser_dwa/compose_nwa.rs` | 90 | `append_weighted_template_redirecting_finals` | `fn append_weighted_template_redirecting_finals(` |
| `src/compile/parser_dwa/compose_nwa.rs` | 121 | `append_bundle_redirecting_finals` | `fn append_bundle_redirecting_finals(` |
| `src/compile/parser_dwa/compose_nwa.rs` | 142 | `append_branch_fragment` | `fn append_branch_fragment(` |
| `src/compile/parser_dwa/compose_nwa.rs` | 190 | `build_parser_nwa_from_terminal_dwa` | `pub(crate) fn build_parser_nwa_from_terminal_dwa(` |
| `src/compile/parser_dwa/determinize/epsilon.rs` | 14 | `local_epsilon_closure` | `pub(super) fn local_epsilon_closure(` |
| `src/compile/parser_dwa/determinize/fallback.rs` | 20 | `determinize_parser_dwa_with_fallbacks` | `pub(crate) fn determinize_parser_dwa_with_fallbacks(` |
| `src/compile/parser_dwa/determinize/fallback.rs` | 25 | `subset_key` | `fn subset_key(entries: &[(u32, Weight)]) -> Vec<(u32, usize)> {` |
| `src/compile/parser_dwa/determinize/outgoing.rs` | 14 | `build_possible_outgoing_ids_by_state` | `pub(crate) fn build_possible_outgoing_ids_by_state(` |
| `src/compile/parser_dwa/determinize/outgoing.rs` | 19 | `OutgoingIds` | `enum OutgoingIds {` |
| `src/compile/parser_dwa/determinize/support.rs` | 22 | `determinize_with_supports` | `pub(crate) fn determinize_with_supports(` |
| `src/compile/parser_dwa/determinize/support.rs` | 26 | `subset_key` | `fn subset_key(entries: &[(u32, Weight)]) -> Vec<(u32, usize)> {` |
| `src/compile/parser_dwa/labels.rs` | 8 | `parser_state_label` | `pub(crate) fn parser_state_label(label: i32, num_parser_states: u32) -> Option<u32> {` |
| `src/compile/parser_dwa/optimize.rs` | 18 | `union_final_weight` | `fn union_final_weight(slot: &mut Option<Weight>, add: Weight) -> bool {` |
| `src/compile/parser_dwa/optimize.rs` | 40 | `optimize_parser_dwa_defaults` | `pub(crate) fn optimize_parser_dwa_defaults(` |
| `src/compile/parser_dwa/optimize.rs` | 229 | `subtract_final_weights_from_outgoing_dwa` | `pub(crate) fn subtract_final_weights_from_outgoing_dwa(dwa: &mut DWA) {` |
| `src/compile/parser_dwa/options.rs` | 9 | `ParserDwaOptions` | `pub(crate) struct ParserDwaOptions {` |
| `src/compile/parser_dwa/options.rs` | 19 | `from_environment` | `pub(crate) fn from_environment(` |
| `src/compile/parser_dwa/options.rs` | 32 | `skip_parser_dwa_minimization_env_override` | `fn skip_parser_dwa_minimization_env_override() -> Option<bool> {` |
| `src/compile/parser_dwa/options.rs` | 44 | `should_skip_parser_dwa_minimization` | `pub(crate) fn should_skip_parser_dwa_minimization(` |
| `src/compile/parser_dwa/profiling.rs` | 12 | `elapsed_ms` | `pub(crate) fn elapsed_ms(started_at: Instant) -> f64 {` |
| `src/compile/parser_dwa/profiling.rs` | 16 | `parser_dwa_compose_detail_enabled` | `pub(crate) fn parser_dwa_compose_detail_enabled() -> bool {` |
| `src/compile/parser_dwa/profiling.rs` | 23 | `ParserNwaBuildProfile` | `pub(crate) struct ParserNwaBuildProfile {` |
| `src/compile/parser_dwa/profiling.rs` | 30 | `ParserDwaComposeDetailProfile` | `pub(crate) struct ParserDwaComposeDetailProfile {` |
| `src/compile/parser_dwa/profiling.rs` | 60 | `accumulate_bundle_profile` | `pub(crate) fn accumulate_bundle_profile(&mut self, bundle_profile: &BundleBuildProfile) {` |
| `src/compile/parser_dwa/profiling.rs` | 75 | `ParserDwaProfile` | `pub(crate) struct ParserDwaProfile {` |
| `src/compile/parser_dwa/profiling.rs` | 101 | `empty` | `pub(crate) fn empty(` |
| `src/compile/parser_dwa/profiling.rs` | 119 | `emit_detail` | `pub(crate) fn emit_detail(&self) {` |
| `src/compile/parser_dwa/profiling.rs` | 148 | `emit_parser_bundle_profile` | `pub(crate) fn emit_parser_bundle_profile(bundle_id: usize, bundle_profile: &BundleBuildProfile) {` |
| `src/compile/parser_dwa/profiling.rs` | 186 | `emit_parser_dwa_compose_profiles` | `pub(crate) fn emit_parser_dwa_compose_profiles(detail: &ParserDwaComposeDetailProfile) {` |
| `src/compile/parser_dwa/terminal_projection.rs` | 21 | `group_terminal_edges_by_target` | `fn group_terminal_edges_by_target(` |
| `src/compile/parser_dwa/terminal_projection.rs` | 47 | `bundle_signature` | `fn bundle_signature(bundle: &TerminalBundle) -> BundleSignature {` |
| `src/compile/parser_dwa/terminal_projection.rs` | 54 | `terminal_template_has_acceptance` | `fn terminal_template_has_acceptance(template: &NWA) -> bool {` |
| `src/compile/parser_dwa/terminal_projection.rs` | 58 | `terminal_bundle_has_acceptance` | `fn terminal_bundle_has_acceptance(bundle: &TerminalBundle, templates: &Templates) -> bool {` |
| `src/compile/parser_dwa/terminal_projection.rs` | 68 | `build_state_summaries` | `pub(crate) fn build_state_summaries(` |
| `src/compile/parser_dwa/terminal_projection.rs` | 117 | `compute_productive_terminal_states` | `pub(crate) fn compute_productive_terminal_states(summaries: &StateSummaries) -> Vec<bool> {` |
| `src/compile/parser_dwa/types.rs` | 21 | `TerminalBundle` | `pub(crate) type TerminalBundle = BTreeMap<TerminalID, Weight>;` |
| `src/compile/parser_dwa/types.rs` | 24 | `BundleSignature` | `pub(crate) type BundleSignature = Vec<(TerminalID, Weight)>;` |
| `src/compile/parser_dwa/types.rs` | 28 | `TargetContribs` | `pub(crate) type TargetContribs = SmallVec<[(u32, Weight); 4]>;` |
| `src/compile/parser_dwa/types.rs` | 32 | `add_target_contribution` | `pub(crate) fn add_target_contribution(contribs: &mut TargetContribs, target: u32, add: Weight) {` |
| `src/compile/parser_dwa/types.rs` | 49 | `extend_target_contribs` | `pub(crate) fn extend_target_contribs(dst: &mut TargetContribs, src: &TargetContribs) {` |
| `src/compile/parser_dwa/types.rs` | 58 | `Branch` | `pub(crate) struct Branch {` |
| `src/compile/parser_dwa/types.rs` | 65 | `StateSummary` | `pub(crate) struct StateSummary {` |
| `src/compile/parser_dwa/types.rs` | 72 | `StateSummaries` | `pub(crate) struct StateSummaries {` |
| `src/compile/parser_dwa/types.rs` | 81 | `DeterminizedDwaWithSupports` | `pub(crate) struct DeterminizedDwaWithSupports {` |
| `src/compile/parser_dwa/types.rs` | 88 | `CachedClosure` | `pub(crate) struct CachedClosure {` |
| `src/compile/parser_dwa/types.rs` | 96 | `PossibleOutgoingIds` | `pub(crate) enum PossibleOutgoingIds {` |
| `src/compile/pipeline/analysis.rs` | 24 | `build_grammar_analysis` | `pub(crate) fn build_grammar_analysis(` |
| `src/compile/pipeline/analysis.rs` | 100 | `compute_disallowed_follows` | `pub(crate) fn compute_disallowed_follows(grammar: &AnalyzedGrammar) -> BTreeMap<u32, BitSet> {` |
| `src/compile/pipeline/context.rs` | 28 | `OwnedCompileInput` | `pub(crate) struct OwnedCompileInput<'vocab> {` |
| `src/compile/pipeline/context.rs` | 34 | `PreparedCompileInput` | `pub(crate) struct PreparedCompileInput<'vocab> {` |
| `src/compile/pipeline/context.rs` | 41 | `GrammarAnalysisOutput` | `pub(crate) struct GrammarAnalysisOutput {` |
| `src/compile/pipeline/context.rs` | 50 | `TerminalScanSupport` | `pub(crate) struct TerminalScanSupport {` |
| `src/compile/pipeline/context.rs` | 62 | `TerminalAndScanOutput` | `pub(crate) struct TerminalAndScanOutput {` |
| `src/compile/pipeline/context.rs` | 68 | `TemplateOutput` | `pub(crate) struct TemplateOutput {` |
| `src/compile/pipeline/context.rs` | 74 | `ReconciledArtifacts` | `pub(crate) struct ReconciledArtifacts {` |
| `src/compile/pipeline/counts.rs` | 9 | `interned_range_count_for_weight_refs` | `pub(crate) fn interned_range_count_for_weight_refs(weight_refs: &[&Weight]) -> usize {` |
| `src/compile/pipeline/counts.rs` | 14 | `interned_range_count_for_artifact` | `pub(crate) fn interned_range_count_for_artifact<T: WeightRefs>(artifact: &mut T) -> usize {` |
| `src/compile/pipeline/counts.rs` | 20 | `joint_interned_range_count_for_artifacts` | `pub(crate) fn joint_interned_range_count_for_artifacts<L, R>(left: &mut L, right: &mut R) -> usize` |
| `src/compile/pipeline/finalize.rs` | 23 | `finalize_runtime_constraint` | `pub(crate) fn finalize_runtime_constraint(` |
| `src/compile/pipeline/mod.rs` | 47 | `compile_owned` | `pub(crate) fn compile_owned(grammar: GrammarDef, vocab: &Vocab) -> Constraint {` |
| `src/compile/pipeline/mod.rs` | 59 | `compile_owned_profiled` | `pub(crate) fn compile_owned_profiled(` |
| `src/compile/pipeline/mod.rs` | 75 | `compile_prepared` | `pub(crate) fn compile_prepared(prepared_grammar: GrammarDef, vocab: &Vocab) -> Constraint {` |
| `src/compile/pipeline/mod.rs` | 80 | `compile_prepared_with_profile` | `pub(crate) fn compile_prepared_with_profile(` |
| `src/compile/pipeline/phases.rs` | 9 | `CompilePhase` | `pub(crate) enum CompilePhase {` |
| `src/compile/pipeline/phases.rs` | 35 | `label` | `pub(crate) fn label(self) -> &'static str {` |
| `src/compile/pipeline/phases.rs` | 51 | `description` | `pub(crate) fn description(self) -> &'static str {` |
| `src/compile/pipeline/reconcile.rs` | 31 | `reconcile_and_build_parser_dwa` | `pub(crate) fn reconcile_and_build_parser_dwa(` |
| `src/compile/pipeline/templates.rs` | 20 | `build_templates` | `pub(crate) fn build_templates(` |
| `src/compile/pipeline/terminal_scan.rs` | 30 | `precompute_terminal_scan_support` | `pub(crate) fn precompute_terminal_scan_support(` |
| `src/compile/pipeline/terminal_scan.rs` | 74 | `build_terminal_dwa_and_scan_relation` | `pub(crate) fn build_terminal_dwa_and_scan_relation(` |
| `src/compile/profiling.rs` | 16 | `compile_profile_summary_enabled` | `pub(crate) fn compile_profile_summary_enabled() -> bool {` |
| `src/compile/profiling.rs` | 21 | `compile_profile_enabled` | `pub(crate) fn compile_profile_enabled() -> bool {` |
| `src/compile/profiling.rs` | 26 | `elapsed_ms` | `pub(crate) fn elapsed_ms(started_at: Instant) -> f64 {` |
| `src/compile/profiling.rs` | 37 | `CompilePhaseProfile` | `pub(crate) struct CompilePhaseProfile {` |
| `src/compile/profiling.rs` | 74 | `CompileProfileSink` | `pub(crate) trait CompileProfileSink {` |
| `src/compile/profiling.rs` | 75 | `emit_line` | `fn emit_line(&mut self, line: &str);` |
| `src/compile/profiling.rs` | 79 | `StderrCompileProfileSink` | `pub(crate) struct StderrCompileProfileSink;` |
| `src/compile/profiling.rs` | 82 | `emit_line` | `fn emit_line(&mut self, line: &str) {` |
| `src/compile/profiling.rs` | 88 | `compile_profile_summary_line` | `pub(crate) fn compile_profile_summary_line(` |
| `src/compile/profiling.rs` | 136 | `emit_compile_profile_summary_to_sink` | `pub(crate) fn emit_compile_profile_summary_to_sink(` |
| `src/compile/profiling.rs` | 149 | `emit_compile_profile_summary` | `pub(crate) fn emit_compile_profile_summary(` |
| `src/compile/profiling.rs` | 162 | `emit_template_profile_summary` | `pub(crate) fn emit_template_profile_summary(` |
| `src/compile/scan_relation/collector.rs` | 21 | `TokenRange` | `pub(crate) type TokenRange = (u32, u32);` |
| `src/compile/scan_relation/collector.rs` | 30 | `TerminalRangeGroup` | `pub(crate) struct TerminalRangeGroup {` |
| `src/compile/scan_relation/collector.rs` | 35 | `IntervalCanMatchMap` | `pub(crate) type IntervalCanMatchMap = Vec<TerminalRangeGroup>;` |
| `src/compile/scan_relation/collector.rs` | 37 | `TrieClassBuildResult` | `pub(crate) struct TrieClassBuildResult {` |
| `src/compile/scan_relation/collector.rs` | 44 | `expand_to_states` | `pub(crate) fn expand_to_states(&self, entries: &[u32]) -> BTreeMap<u32, IntervalCanMatchMap> {` |
| `src/compile/scan_relation/collector.rs` | 54 | `SegmentOutcome` | `struct SegmentOutcome { terminals_id: u32, end_state: Option<u32> }` |
| `src/compile/scan_relation/collector.rs` | 56 | `SegmentOutcomeCache` | `enum SegmentOutcomeCache {` |
| `src/compile/scan_relation/collector.rs` | 62 | `default` | `fn default() -> Self {` |
| `src/compile/scan_relation/collector.rs` | 68 | `TerminalSetInterner` | `struct TerminalSetInterner {` |
| `src/compile/scan_relation/collector.rs` | 75 | `intern_slice` | `fn intern_slice(&mut self, terminals: &[TerminalID]) -> u32 {` |
| `src/compile/scan_relation/collector.rs` | 78 | `intern_vec` | `fn intern_vec(&mut self, mut terminals: Vec<TerminalID>) -> u32 {` |
| `src/compile/scan_relation/collector.rs` | 87 | `intern_mask` | `fn intern_mask(&mut self, mask: u128) -> u32 {` |
| `src/compile/scan_relation/collector.rs` | 107 | `get` | `fn get(&self, id: u32) -> &[TerminalID] { &self.sets[id as usize] }` |
| `src/compile/scan_relation/collector.rs` | 114 | `NodeClasses` | `struct NodeClasses { classes: Vec<u32>, class_maps: Vec<Arc<IntervalCanMatchMap>> }` |
| `src/compile/scan_relation/collector.rs` | 117 | `BuildTimings` | `struct BuildTimings {` |
| `src/compile/scan_relation/collector.rs` | 163 | `add_assign` | `fn add_assign(&mut self, other: Self) {` |
| `src/compile/scan_relation/collector.rs` | 210 | `canonical_terminal_box` | `fn canonical_terminal_box(terminals: &[TerminalID]) -> Option<Box<[TerminalID]>> {` |
| `src/compile/scan_relation/collector.rs` | 216 | `append_range` | `fn append_range(map: &mut IntervalCanMatchMap, terminals: &[TerminalID], range: TokenRange) {` |
| `src/compile/scan_relation/collector.rs` | 225 | `append_ranges` | `fn append_ranges(map: &mut IntervalCanMatchMap, terminals: &[TerminalID], ranges: &[TokenRange]) {` |
| `src/compile/scan_relation/collector.rs` | 233 | `merge_interval_maps` | `fn merge_interval_maps(into: &mut IntervalCanMatchMap, other: &IntervalCanMatchMap) {` |
| `src/compile/scan_relation/collector.rs` | 237 | `normalize_ranges` | `fn normalize_ranges(ranges: &mut Vec<TokenRange>) {` |
| `src/compile/scan_relation/collector.rs` | 250 | `normalize_interval_map` | `fn normalize_interval_map(map: &mut IntervalCanMatchMap) {` |
| `src/compile/scan_relation/collector.rs` | 274 | `reachable_ranges` | `fn reachable_ranges(node: &VocabPrefixTreeNode) -> Box<[TokenRange]> {` |
| `src/compile/scan_relation/collector.rs` | 283 | `next_nonzero_generation` | `fn next_nonzero_generation(generation: &mut u32, stamps: &mut [u32]) -> u32 {` |
| `src/compile/scan_relation/collector.rs` | 292 | `dense_segment_cache_min_entries` | `fn dense_segment_cache_min_entries() -> usize {` |
| `src/compile/scan_relation/collector.rs` | 299 | `promote_segment_outcome_cache` | `fn promote_segment_outcome_cache(` |
| `src/compile/scan_relation/collector.rs` | 321 | `segment_outcomes_for_states` | `fn segment_outcomes_for_states(` |
| `src/compile/scan_relation/collector.rs` | 447 | `mix_signature_word` | `fn mix_signature_word(hash: u64, word: u32) -> u64 {` |
| `src/compile/scan_relation/collector.rs` | 451 | `SignatureEntry` | `struct SignatureEntry { state_pos: usize, class_id: u32 }` |
| `src/compile/scan_relation/collector.rs` | 452 | `ChildBuildData` | `struct ChildBuildData { outcomes: Vec<SegmentOutcome>, child_class_ids: Vec<u32>, reachable: Box<[TokenRange]>, result: NodeClasses }` |
| `src/compile/scan_relation/collector.rs` | 453 | `ChildPendingData` | `struct ChildPendingData<'a> { child: &'a VocabPrefixTreeNode, outcomes: Vec<SegmentOutcome>, descend_positions: Vec<u32>, child_active_states: Vec<u32>, reachable: Box<[TokenRange]> }` |
| `src/compile/scan_relation/collector.rs` | 455 | `build_node` | `fn build_node(` |
| `src/compile/scan_relation/collector.rs` | 670 | `collect_can_match_interval_trie_class_build_with_classes` | `pub(crate) fn collect_can_match_interval_trie_class_build_with_classes(` |
| `src/compile/scan_relation/compute.rs` | 32 | `compute_scan_relation` | `pub(crate) fn compute_scan_relation(` |
| `src/compile/scan_relation/compute.rs` | 45 | `compute_scan_relation_with_artifacts` | `fn compute_scan_relation_with_artifacts(` |
| `src/compile/scan_relation/compute.rs` | 128 | `compute_scan_relation_for_vocab` | `pub(crate) fn compute_scan_relation_for_vocab(` |
| `src/compile/scan_relation/compute.rs` | 178 | `prepare_vocab_for_scan_relation` | `pub(crate) fn prepare_vocab_for_scan_relation(vocab: &Vocab) {` |
| `src/compile/scan_relation/legacy_materialize.rs` | 13 | `ExpandedIntervalCanMatchMap` | `type ExpandedIntervalCanMatchMap = BTreeMap<TerminalID, Vec<(u32, u32)>>;` |
| `src/compile/scan_relation/legacy_materialize.rs` | 16 | `LegacySweepEvent` | `struct LegacySweepEvent {` |
| `src/compile/scan_relation/legacy_materialize.rs` | 21 | `normalize_token_ranges` | `fn normalize_token_ranges(ranges: &mut Vec<(u32, u32)>) {` |
| `src/compile/scan_relation/legacy_materialize.rs` | 38 | `append_expanded_ranges` | `fn append_expanded_ranges(` |
| `src/compile/scan_relation/legacy_materialize.rs` | 48 | `normalize_expanded_interval_map` | `fn normalize_expanded_interval_map(map: &mut ExpandedIntervalCanMatchMap) {` |
| `src/compile/scan_relation/legacy_materialize.rs` | 55 | `expand_interval_class_maps` | `fn expand_interval_class_maps(` |
| `src/compile/scan_relation/legacy_materialize.rs` | 70 | `push_legacy_sweep_event` | `fn push_legacy_sweep_event(` |
| `src/compile/scan_relation/legacy_materialize.rs` | 81 | `build_legacy_sweep_events` | `fn build_legacy_sweep_events(` |
| `src/compile/scan_relation/legacy_materialize.rs` | 115 | `apply_legacy_sweep_events` | `fn apply_legacy_sweep_events(` |
| `src/compile/scan_relation/legacy_materialize.rs` | 137 | `build_legacy_scan_relation_vocab_and_weights_from_interval_maps` | `pub(super) fn build_legacy_scan_relation_vocab_and_weights_from_interval_maps(` |
| `src/compile/scan_relation/legacy_materialize.rs` | 214 | `validate_group_scan_relation_vocab_outputs` | `pub(super) fn validate_group_scan_relation_vocab_outputs(` |
| `src/compile/scan_relation/ordered_vocab.rs` | 11 | `OrderedVocab` | `pub(super) struct OrderedVocab {` |
| `src/compile/scan_relation/ordered_vocab.rs` | 18 | `OrderedVocabTrieArtifacts` | `pub(super) struct OrderedVocabTrieArtifacts {` |
| `src/compile/scan_relation/ordered_vocab.rs` | 26 | `OrderedVocabCacheFingerprint` | `pub(super) struct OrderedVocabCacheFingerprint {` |
| `src/compile/scan_relation/ordered_vocab.rs` | 34 | `OrderedVocabCacheEntry` | `pub(super) struct OrderedVocabCacheEntry {` |
| `src/compile/scan_relation/ordered_vocab.rs` | 41 | `OrderedVocabCacheStatus` | `pub(super) enum OrderedVocabCacheStatus {` |
| `src/compile/scan_relation/ordered_vocab.rs` | 48 | `as_str` | `fn as_str(self) -> &'static str {` |
| `src/compile/scan_relation/ordered_vocab.rs` | 58 | `OrderedVocabCacheProfile` | `pub(super) struct OrderedVocabCacheProfile {` |
| `src/compile/scan_relation/ordered_vocab.rs` | 68 | `build_internal_token_bytes_from_groups` | `pub(crate) fn build_internal_token_bytes_from_groups(` |
| `src/compile/scan_relation/ordered_vocab.rs` | 78 | `build_ordered_vocab` | `fn build_ordered_vocab(token_bytes: &BTreeMap<u32, Vec<u8>>) -> OrderedVocab {` |
| `src/compile/scan_relation/ordered_vocab.rs` | 105 | `build_ordered_vocab_prefix_tree` | `fn build_ordered_vocab_prefix_tree(ordered_vocab: &OrderedVocab) -> VocabPrefixTree {` |
| `src/compile/scan_relation/ordered_vocab.rs` | 110 | `ordered_vocab_cache_enabled` | `fn ordered_vocab_cache_enabled() -> bool {` |
| `src/compile/scan_relation/ordered_vocab.rs` | 122 | `ordered_vocab_cache_capacity` | `fn ordered_vocab_cache_capacity() -> usize {` |
| `src/compile/scan_relation/ordered_vocab.rs` | 132 | `ordered_vocab_cache` | `fn ordered_vocab_cache() -> &'static Mutex<Vec<OrderedVocabCacheEntry>> {` |
| `src/compile/scan_relation/ordered_vocab.rs` | 137 | `ordered_vocab_cache_fingerprint` | `fn ordered_vocab_cache_fingerprint(` |
| `src/compile/scan_relation/ordered_vocab.rs` | 160 | `ordered_vocab_cache_source_matches` | `fn ordered_vocab_cache_source_matches(` |
| `src/compile/scan_relation/ordered_vocab.rs` | 206 | `ordered_vocab_cache_source_original_to_ordered` | `fn ordered_vocab_cache_source_original_to_ordered(` |
| `src/compile/scan_relation/ordered_vocab.rs` | 220 | `compile_profile_requested` | `fn compile_profile_requested() -> bool {` |
| `src/compile/scan_relation/ordered_vocab.rs` | 225 | `emit_ordered_vocab_cache_profile` | `pub(super) fn emit_ordered_vocab_cache_profile(profile: OrderedVocabCacheProfile) {` |
| `src/compile/scan_relation/ordered_vocab.rs` | 241 | `get_ordered_vocab_trie_artifacts` | `pub(super) fn get_ordered_vocab_trie_artifacts(` |
| `src/compile/scan_relation/ordered_vocab.rs` | 349 | `get_ordered_vocab_trie_artifacts_for_vocab` | `pub(super) fn get_ordered_vocab_trie_artifacts_for_vocab(` |
| `src/compile/scan_relation/ordered_vocab.rs` | 397 | `dense_word_count` | `pub(crate) fn dense_word_count(token_slots: u32) -> usize { (token_slots as usize + 63) / 64 }` |
| `src/compile/scan_relation/ordered_vocab.rs` | 400 | `max_original_token_slot` | `pub(crate) fn max_original_token_slot(token_bytes: &BTreeMap<u32, Vec<u8>>) -> u32 {` |
| `src/compile/scan_relation/ordered_vocab.rs` | 404 | `range_set_from_sorted_ids` | `pub(super) fn range_set_from_sorted_ids(ids: &[u32]) -> RangeSetBlaze<u32> {` |
| `src/compile/scan_relation/ordered_vocab.rs` | 417 | `range_set_from_u128_mask` | `pub(super) fn range_set_from_u128_mask(mask: u128) -> RangeSetBlaze<u32> {` |
| `src/compile/scan_relation/profile.rs` | 9 | `profile_summary_enabled` | `pub(crate) fn profile_summary_enabled() -> bool {` |
| `src/compile/scan_relation/profile.rs` | 13 | `elapsed_ms` | `pub(crate) fn elapsed_ms(started_at: Instant) -> f64 {` |
| `src/compile/scan_relation/profile.rs` | 18 | `CanMatchProfile` | `pub(crate) struct CanMatchProfile {` |
| `src/compile/scan_relation/root_collect.rs` | 11 | `group_scan_relation_vocab_validation_enabled` | `fn group_scan_relation_vocab_validation_enabled() -> bool {` |
| `src/compile/scan_relation/root_collect.rs` | 17 | `group_scan_relation_vocab_legacy_enabled` | `fn group_scan_relation_vocab_legacy_enabled() -> bool {` |
| `src/compile/scan_relation/root_collect.rs` | 23 | `sparse_root_collect_enabled` | `pub(super) fn sparse_root_collect_enabled() -> bool {` |
| `src/compile/scan_relation/root_collect.rs` | 29 | `sparse_root_state_limit` | `pub(super) fn sparse_root_state_limit() -> usize {` |
| `src/compile/scan_relation/root_collect.rs` | 36 | `sparse_root_terminal_limit` | `pub(super) fn sparse_root_terminal_limit() -> usize {` |
| `src/compile/scan_relation/root_collect.rs` | 43 | `root_terminal_union_count` | `pub(super) fn root_terminal_union_count(tokenizer: &Tokenizer, states: &[u32]) -> usize {` |
| `src/compile/scan_relation/root_collect.rs` | 61 | `interval_map_from_sparse_matches` | `fn interval_map_from_sparse_matches(` |
| `src/compile/scan_relation/root_collect.rs` | 95 | `collect_sparse_root_can_match` | `pub(super) fn collect_sparse_root_can_match(` |
| `src/compile/scan_relation/terminal_sequences.rs` | 13 | `CanMatchMap` | `type CanMatchMap = FxHashMap<TerminalID, RangeSetBlaze<u32>>;` |
| `src/compile/scan_relation/terminal_sequences.rs` | 15 | `reachable_u32` | `fn reachable_u32(node: &VocabPrefixTreeNode) -> RangeSetBlaze<u32> {` |
| `src/compile/scan_relation/terminal_sequences.rs` | 23 | `merge_token_ids` | `fn merge_token_ids(into: &mut RangeSetBlaze<u32>, other: &RangeSetBlaze<u32>) {` |
| `src/compile/scan_relation/terminal_sequences.rs` | 27 | `merge_can_match_maps` | `fn merge_can_match_maps(into: &mut CanMatchMap, other: &CanMatchMap) {` |
| `src/compile/scan_relation/terminal_sequences.rs` | 34 | `CanMatchComputer` | `pub(crate) struct CanMatchComputer<'a> {` |
| `src/compile/scan_relation/terminal_sequences.rs` | 44 | `new` | `pub(crate) fn new(tokenizer: &'a Tokenizer) -> Self {` |
| `src/compile/scan_relation/terminal_sequences.rs` | 48 | `new_with_canonical_state` | `pub(crate) fn new_with_canonical_state(` |
| `src/compile/scan_relation/terminal_sequences.rs` | 63 | `fast_step` | `fn fast_step(&mut self, state: u32, byte: u8) -> Option<u32> {` |
| `src/compile/scan_relation/terminal_sequences.rs` | 77 | `reachable_for_node` | `fn reachable_for_node(&mut self, node: &VocabPrefixTreeNode) -> Rc<RangeSetBlaze<u32>> {` |
| `src/compile/scan_relation/terminal_sequences.rs` | 88 | `can_skip_self_loop_subtree` | `fn can_skip_self_loop_subtree(` |
| `src/compile/scan_relation/terminal_sequences.rs` | 106 | `can_match_for_node` | `pub(crate) fn can_match_for_node(` |
| `src/compile/scan_relation/types.rs` | 23 | `RuntimeCanMatchByTerminal` | `pub(crate) type RuntimeCanMatchByTerminal = BTreeMap<TerminalID, Weight>;` |
| `src/compile/scan_relation/types.rs` | 26 | `SignatureClassId` | `pub(crate) type SignatureClassId = u32;` |
| `src/compile/scan_relation/types.rs` | 29 | `StateTerminalLabel` | `pub(super) type StateTerminalLabel = (u32, TerminalID);` |
| `src/compile/scan_relation/types.rs` | 33 | `ScanRelationVocabMap` | `pub(crate) struct ScanRelationVocabMap {` |
| `src/compile/scan_relation/types.rs` | 49 | `ScanRelationConfig` | `pub(crate) struct ScanRelationConfig;` |
| `src/compile/scan_relation/types.rs` | 53 | `ScanRelationProfile` | `pub(crate) struct ScanRelationProfile {` |
| `src/compile/scan_relation/types.rs` | 60 | `ScanRelationComputation` | `pub(crate) struct ScanRelationComputation {` |
| `src/compile/scan_relation/types.rs` | 66 | `SweepEvent` | `pub(super) struct SweepEvent {` |
| `src/compile/scan_relation/types.rs` | 72 | `SweepGroup` | `pub(super) struct SweepGroup {` |
| `src/compile/scan_relation/types.rs` | 77 | `SweepBuildStats` | `pub(super) struct SweepBuildStats {` |
| `src/compile/scan_relation/vocab_equivalence.rs` | 12 | `scan_relation_vocab_equiv_enabled` | `pub(super) fn scan_relation_vocab_equiv_enabled() -> bool {` |
| `src/compile/scan_relation/vocab_equivalence.rs` | 22 | `CanMatchTokenOutcome` | `struct CanMatchTokenOutcome {` |
| `src/compile/scan_relation/vocab_equivalence.rs` | 28 | `mix_can_match_signature_word` | `fn mix_can_match_signature_word(hash: u64, word: u64) -> u64 {` |
| `src/compile/scan_relation/vocab_equivalence.rs` | 36 | `can_match_signature_hash` | `fn can_match_signature_hash(outcomes: &[CanMatchTokenOutcome]) -> u64 {` |
| `src/compile/scan_relation/vocab_equivalence.rs` | 45 | `can_match_signature_matches` | `fn can_match_signature_matches(signature: &[u128], outcomes: &[CanMatchTokenOutcome]) -> bool {` |
| `src/compile/scan_relation/vocab_equivalence.rs` | 53 | `intern_can_match_token_signature` | `fn intern_can_match_token_signature(` |
| `src/compile/scan_relation/vocab_equivalence.rs` | 77 | `advance_can_match_token_outcomes` | `fn advance_can_match_token_outcomes(` |
| `src/compile/scan_relation/vocab_equivalence.rs` | 106 | `CanMatchVocabEquivBuilder` | `struct CanMatchVocabEquivBuilder<'a> {` |
| `src/compile/scan_relation/vocab_equivalence.rs` | 118 | `new` | `fn new(` |
| `src/compile/scan_relation/vocab_equivalence.rs` | 135 | `record_token` | `fn record_token(&mut self, ordered_token_id: usize, outcomes: &[CanMatchTokenOutcome]) {` |
| `src/compile/scan_relation/vocab_equivalence.rs` | 160 | `visit` | `fn visit(&mut self, node: &VocabPrefixTreeNode, outcomes: &[CanMatchTokenOutcome]) {` |
| `src/compile/scan_relation/vocab_equivalence.rs` | 175 | `finish` | `fn finish(mut self) -> ManyToOneIdMap {` |
| `src/compile/scan_relation/vocab_equivalence.rs` | 188 | `compute_scan_relation_vocab_equivalence_map` | `pub(super) fn compute_scan_relation_vocab_equivalence_map(` |
| `src/compile/scan_relation/vocab_equivalence.rs` | 227 | `compute_scan_relation_vocab_equivalence_map_fast` | `pub(super) fn compute_scan_relation_vocab_equivalence_map_fast(` |
| `src/compile/scan_relation/vocab_materialize.rs` | 20 | `used_state_class_ids` | `pub(super) fn used_state_class_ids(state_classes: &[u32]) -> Vec<u32> {` |
| `src/compile/scan_relation/vocab_materialize.rs` | 27 | `next_nonzero_stamp` | `fn next_nonzero_stamp(generation: &mut u32, stamps: &mut [u32]) -> u32 {` |
| `src/compile/scan_relation/vocab_materialize.rs` | 36 | `push_sweep_event` | `fn push_sweep_event(events: &mut [Vec<SweepEvent>], event_positions: &mut Vec<u32>, position: u32, event: SweepEvent) {` |
| `src/compile/scan_relation/vocab_materialize.rs` | 42 | `intern_state_terminal_label` | `pub(super) fn intern_state_terminal_label(` |
| `src/compile/scan_relation/vocab_materialize.rs` | 57 | `build_sweep_events` | `fn build_sweep_events(` |
| `src/compile/scan_relation/vocab_materialize.rs` | 115 | `active_group_hash` | `fn active_group_hash(group_id: u32) -> u64 {` |
| `src/compile/scan_relation/vocab_materialize.rs` | 122 | `insert_active_group_id` | `fn insert_active_group_id(` |
| `src/compile/scan_relation/vocab_materialize.rs` | 137 | `remove_active_group_id` | `fn remove_active_group_id(` |
| `src/compile/scan_relation/vocab_materialize.rs` | 155 | `apply_sweep_events` | `fn apply_sweep_events(` |
| `src/compile/scan_relation/vocab_materialize.rs` | 179 | `active_group_key_matches` | `fn active_group_key_matches(` |
| `src/compile/scan_relation/vocab_materialize.rs` | 190 | `build_signature_from_active_groups` | `fn build_signature_from_active_groups(` |
| `src/compile/scan_relation/vocab_materialize.rs` | 215 | `build_signature_from_active_group_ids` | `fn build_signature_from_active_group_ids(` |
| `src/compile/scan_relation/vocab_materialize.rs` | 240 | `build_scan_relation_vocab_and_weights_from_interval_maps` | `pub(super) fn build_scan_relation_vocab_and_weights_from_interval_maps(` |
| `src/compile/terminal_dwa/builder.rs` | 71 | `build_terminal_dwa_with_precomputed_global_max_length` | `pub(crate) fn build_terminal_dwa_with_precomputed_global_max_length(` |
| `src/compile/terminal_dwa/builder.rs` | 236 | `build_terminal_dwa` | `pub(crate) fn build_terminal_dwa(` |
| `src/compile/terminal_dwa/classify.rs` | 18 | `SharedClassifyBytesets` | `pub struct SharedClassifyBytesets {` |
| `src/compile/terminal_dwa/classify.rs` | 25 | `SharedClassifyCache` | `pub type SharedClassifyCache = std::sync::OnceLock<SharedClassifyBytesets>;` |
| `src/compile/terminal_dwa/classify.rs` | 28 | `VocabByteSet` | `struct VocabByteSet {` |
| `src/compile/terminal_dwa/classify.rs` | 34 | `vocab_byte_set` | `fn vocab_byte_set(vocab: &Vocab) -> U8Set {` |
| `src/compile/terminal_dwa/classify.rs` | 49 | `prepare_vocab_for_terminal_classification` | `pub(crate) fn prepare_vocab_for_terminal_classification(vocab: &Vocab) {` |
| `src/compile/terminal_dwa/classify.rs` | 54 | `PairPartitionCostFn` | `pub(crate) enum PairPartitionCostFn {` |
| `src/compile/terminal_dwa/classify.rs` | 62 | `as_str` | `pub(crate) fn as_str(self) -> &'static str {` |
| `src/compile/terminal_dwa/classify.rs` | 73 | `PairPartitionObjective` | `pub(crate) enum PairPartitionObjective {` |
| `src/compile/terminal_dwa/classify.rs` | 79 | `as_str` | `pub(crate) fn as_str(self) -> &'static str {` |
| `src/compile/terminal_dwa/classify.rs` | 87 | `PairPartitionCostPartitioning` | `pub(crate) struct PairPartitionCostPartitioning {` |
| `src/compile/terminal_dwa/classify.rs` | 95 | `PairPartitionTokenGroup` | `struct PairPartitionTokenGroup {` |
| `src/compile/terminal_dwa/classify.rs` | 101 | `PairPartitionBucket` | `struct PairPartitionBucket {` |
| `src/compile/terminal_dwa/classify.rs` | 108 | `new` | `fn new() -> Self {` |
| `src/compile/terminal_dwa/classify.rs` | 116 | `size` | `fn size(&self) -> usize {` |
| `src/compile/terminal_dwa/classify.rs` | 120 | `pair_partition_count` | `fn pair_partition_count(&self) -> usize {` |
| `src/compile/terminal_dwa/classify.rs` | 131 | `build` | `pub fn build(tokenizer: &Tokenizer, num_terminals: u32) -> Self {` |
| `src/compile/terminal_dwa/classify.rs` | 227 | `classify_vocab_char_type` | `pub(crate) fn classify_vocab_char_type(bytes: &[u8]) -> u8 {` |
| `src/compile/terminal_dwa/classify.rs` | 283 | `classify_nonalnum` | `fn classify_nonalnum(bytes: &[u8]) -> u8 {` |
| `src/compile/terminal_dwa/classify.rs` | 308 | `classify_terminal_path_lengths` | `pub(crate) fn classify_terminal_path_lengths(` |
| `src/compile/terminal_dwa/classify.rs` | 369 | `build_byte_terminal_reverse_index` | `fn build_byte_terminal_reverse_index(` |
| `src/compile/terminal_dwa/classify.rs` | 388 | `token_pair_partition_terminals` | `fn token_pair_partition_terminals(` |
| `src/compile/terminal_dwa/classify.rs` | 425 | `compute_partition_cost` | `fn compute_partition_cost(` |
| `src/compile/terminal_dwa/classify.rs` | 444 | `partition_metric_count` | `fn partition_metric_count(` |
| `src/compile/terminal_dwa/classify.rs` | 457 | `objective_score` | `fn objective_score(objective: PairPartitionObjective, costs: &[f64]) -> f64 {` |
| `src/compile/terminal_dwa/classify.rs` | 464 | `compute_token_pair_partition_map` | `fn compute_token_pair_partition_map(` |
| `src/compile/terminal_dwa/classify.rs` | 490 | `partition_vocab_char_type_tokens` | `pub(crate) fn partition_vocab_char_type_tokens(vocab: &Vocab) -> Vec<Vec<u32>> {` |
| `src/compile/terminal_dwa/classify.rs` | 499 | `estimate_pair_partition_objective_for_token_partitions` | `pub(crate) fn estimate_pair_partition_objective_for_token_partitions(` |
| `src/compile/terminal_dwa/classify.rs` | 546 | `partition_token_pair_partition_map_by_cost` | `fn partition_token_pair_partition_map_by_cost(` |
| `src/compile/terminal_dwa/classify.rs` | 673 | `partition_vocab_by_pair_partition_cost_with_token_map` | `pub(crate) fn partition_vocab_by_pair_partition_cost_with_token_map(` |
| `src/compile/terminal_dwa/classify.rs` | 688 | `partition_vocab_by_pair_partition_cost` | `pub(crate) fn partition_vocab_by_pair_partition_cost(` |
| `src/compile/terminal_dwa/direct_partition/max_length.rs` | 12 | `mix_u64` | `fn mix_u64(mut x: u64) -> u64 {` |
| `src/compile/terminal_dwa/direct_partition/max_length.rs` | 22 | `hash_filtered_sorted_set` | `fn hash_filtered_sorted_set(values: &[usize], active_groups: Option<&[bool]>, tag: u64) -> u64 {` |
| `src/compile/terminal_dwa/direct_partition/max_length.rs` | 38 | `hash_state_label` | `fn hash_state_label(state: &FlatDfaState, active_groups: Option<&[bool]>) -> u64 {` |
| `src/compile/terminal_dwa/direct_partition/max_length.rs` | 49 | `hash_transition_labels` | `fn hash_transition_labels(label_hashes: &[u64]) -> u64 {` |
| `src/compile/terminal_dwa/direct_partition/max_length.rs` | 58 | `hash_transition_targets` | `fn hash_transition_targets(targets: &[usize], prev_hashes: &[u64]) -> u64 {` |
| `src/compile/terminal_dwa/direct_partition/max_length.rs` | 66 | `build_state_shape` | `fn build_state_shape(` |
| `src/compile/terminal_dwa/direct_partition/max_length.rs` | 93 | `build_subset_mapping` | `fn build_subset_mapping(states: &[usize], hashes: &[u64]) -> Vec<usize> {` |
| `src/compile/terminal_dwa/direct_partition/max_length.rs` | 119 | `count_distinct_hashes` | `fn count_distinct_hashes(hashes: &[u64]) -> usize {` |
| `src/compile/terminal_dwa/direct_partition/max_length.rs` | 127 | `find_state_equivalence_classes_kstep` | `fn find_state_equivalence_classes_kstep(` |
| `src/compile/terminal_dwa/direct_partition/max_length.rs` | 181 | `cheap_state_hash` | `fn cheap_state_hash(` |
| `src/compile/terminal_dwa/direct_partition/max_length.rs` | 199 | `find_state_equivalence_classes` | `pub fn find_state_equivalence_classes<S: AsRef<[u8]>>(` |
| `src/compile/terminal_dwa/direct_partition/max_length.rs` | 223 | `build_state_shape_restricted` | `fn build_state_shape_restricted(dfa: &FlatDfa, state_idx: usize, relevant_bytes: &[bool; 256]) -> (Vec<usize>, u64) {` |
| `src/compile/terminal_dwa/direct_partition/max_length.rs` | 246 | `find_state_equivalence_classes_kstep_restricted` | `fn find_state_equivalence_classes_kstep_restricted(` |
| `src/compile/terminal_dwa/direct_partition/max_length.rs` | 403 | `find_state_equivalence_classes_byte_restricted` | `pub fn find_state_equivalence_classes_byte_restricted<S: AsRef<[u8]>>(` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 18 | `PreHashedRanges` | `struct PreHashedRanges {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 24 | `DirectPartitionIdentityVocabOrder` | `struct DirectPartitionIdentityVocabOrder {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 33 | `direct_partition_identity_vocab_order` | `fn direct_partition_identity_vocab_order(vocab: &Vocab) -> Arc<DirectPartitionIdentityVocabOrder> {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 73 | `prepare_direct_partition_identity_vocab_order` | `pub(crate) fn prepare_direct_partition_identity_vocab_order(vocab: &Vocab) {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 77 | `skip_max_length_for_partition` | `fn skip_max_length_for_partition(partition_label: &str) -> bool {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 100 | `skip_direct_partition_max_length_for_partition` | `fn skip_direct_partition_max_length_for_partition(partition_label: &str) -> bool {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 123 | `direct_partition_max_length_min_states` | `fn direct_partition_max_length_min_states() -> usize {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 134 | `should_skip_max_length_for_partition` | `fn should_skip_max_length_for_partition(` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 145 | `fast_projected_direct_partition_id_map_enabled` | `fn fast_projected_direct_partition_id_map_enabled() -> bool {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 157 | `fast_projected_direct_partition_id_map_max_tsids` | `fn fast_projected_direct_partition_id_map_max_tsids() -> usize {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 168 | `should_use_fast_projected_direct_partition_id_map` | `fn should_use_fast_projected_direct_partition_id_map(` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 186 | `range_hash_val` | `fn range_hash_val(s: u32, e: u32) -> u64 {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 192 | `new` | `fn new(ranges: Vec<(u32, u32)>) -> Self {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 203 | `eq` | `fn eq(&self, other: &Self) -> bool {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 211 | `hash` | `fn hash<H: Hasher>(&self, state: &mut H) {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 224 | `LazyRanges` | `struct LazyRanges<'a> {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 231 | `new` | `fn new(refs: Vec<&'a [(u32, u32)]>) -> Self {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 270 | `materialize` | `fn materialize(&self) -> Vec<(u32, u32)> {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 291 | `eq` | `fn eq(&self, other: &Self) -> bool {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 325 | `compact_direct_partition_terminal_dwa_enabled` | `fn compact_direct_partition_terminal_dwa_enabled() -> bool {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 349 | `count_direct_partition_equivalence_classes` | `pub(crate) fn count_direct_partition_equivalence_classes(` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 403 | `build_direct_partition_terminal_dwa` | `pub(crate) fn build_direct_partition_terminal_dwa(` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 549 | `build_direct_partition_id_map` | `fn build_direct_partition_id_map<'a>(` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 776 | `build_direct_partition_identity_vocab_map` | `fn build_direct_partition_identity_vocab_map(vocab: &Vocab) -> (ManyToOneIdMap, Arc<DirectPartitionIdentityVocabOrder>, f64) {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 793 | `state_to_representative_vector` | `fn state_to_representative_vector(state_map: &ManyToOneIdMap, num_dfa_states: usize) -> Vec<u32> {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 805 | `TokenLengthStats` | `struct TokenLengthStats {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 813 | `token_length_stats` | `fn token_length_stats(tokens: &[&[u8]]) -> TokenLengthStats {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 842 | `token_length_stats_from_entries` | `fn token_length_stats_from_entries(tokens: &[(u32, Arc<[u8]>)]) -> TokenLengthStats {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 871 | `find_direct_partition_exact_state_equivalence_by_token_signatures` | `fn find_direct_partition_exact_state_equivalence_by_token_signatures(` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 1004 | `direct_partition_bucket_suffix_signature_profile` | `fn direct_partition_bucket_suffix_signature_profile(` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 1049 | `DirectPartitionSortedTokenBuckets` | `struct DirectPartitionSortedTokenBuckets {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 1058 | `build_direct_partition_sorted_token_buckets` | `fn build_direct_partition_sorted_token_buckets(sorted_entries: &[(u32, Arc<[u8]>)]) -> DirectPartitionSortedTokenBuckets {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 1110 | `collect_active_terminal_signature` | `fn collect_active_terminal_signature(` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 1131 | `build_direct_partition_state_to_terminal_signature` | `fn build_direct_partition_state_to_terminal_signature(` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 1154 | `direct_partition_token_signature_profile_for_state` | `fn direct_partition_token_signature_profile_for_state(` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 1215 | `append_direct_partition_signature_profile_run` | `fn append_direct_partition_signature_profile_run(profile: &mut Vec<(u32, u32, u32)>, sig_id: u32, token_id: u32) {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 1225 | `build_direct_partition_terminal_dwa` | `fn build_direct_partition_terminal_dwa(` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 2007 | `build_flat_transition_table` | `pub(crate) fn build_flat_transition_table(tokenizer: &Tokenizer) -> Vec<u32> {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 2019 | `common_prefix_len` | `fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 2028 | `append_token_id_range` | `fn append_token_id_range(token_ranges: &mut Vec<(u32, u32)>, token_id: u32) {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 2032 | `append_token_id_span` | `fn append_token_id_span(token_ranges: &mut Vec<(u32, u32)>, start: u32, end: u32) {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 2042 | `flush_end_rep_run` | `fn flush_end_rep_run(` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 2057 | `collect_direct_partition_root_ranges_by_first_byte_lcp` | `fn collect_direct_partition_root_ranges_by_first_byte_lcp(` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 2121 | `merge_ranges_in_place` | `fn merge_ranges_in_place(ranges: &mut Vec<(u32, u32)>) {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 2139 | `shared_rangeset_from_unsorted_pairs` | `fn shared_rangeset_from_unsorted_pairs(ranges: &[(u32, u32)]) -> Option<Arc<RangeSetBlaze<u32>>> {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 2151 | `build_end_rep_group_masks` | `fn build_end_rep_group_masks(` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 2171 | `merge_deferred_equivalent_tsids` | `fn merge_deferred_equivalent_tsids(` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 2247 | `remap_deferred_arced_tsids` | `fn remap_deferred_arced_tsids(` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 2278 | `apply_tsid_perm_to_id_map` | `fn apply_tsid_perm_to_id_map(id_map: &mut ManyToOneIdMap, perm: &[u32], new_count: usize) {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 2302 | `DirectPartitionIdMapProfile` | `struct DirectPartitionIdMapProfile {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 2319 | `DirectPartitionTsidProfileMergeReport` | `struct DirectPartitionTsidProfileMergeReport {` |
| `src/compile/terminal_dwa/direct_partition/mod.rs` | 2328 | `DirectPartitionTerminalBuildProfile` | `struct DirectPartitionTerminalBuildProfile {` |
| `src/compile/terminal_dwa/global_state_map.rs` | 22 | `use_global_max_length` | `fn use_global_max_length(tokenizer: &Tokenizer) -> bool {` |
| `src/compile/terminal_dwa/global_state_map.rs` | 29 | `build_global_max_length_state_map` | `pub(crate) fn build_global_max_length_state_map(` |
| `src/compile/terminal_dwa/grammar_helpers.rs` | 13 | `compute_terminal_coloring` | `pub(crate) fn compute_terminal_coloring(table: &GLRTable) -> TerminalColoring {` |
| `src/compile/terminal_dwa/grammar_helpers.rs` | 112 | `assert_row_colors_are_unique` | `fn assert_row_colors_are_unique(table: &GLRTable, coloring: &TerminalColoring) {` |
| `src/compile/terminal_dwa/grammar_helpers.rs` | 128 | `terminal_coloring_keeps_action_row_terminals_distinct` | `fn terminal_coloring_keeps_action_row_terminals_distinct() {` |
| `src/compile/terminal_dwa/grammar_helpers.rs` | 147 | `terminal_coloring_handles_sparse_high_terminal_count` | `fn terminal_coloring_handles_sparse_high_terminal_count() {` |
| `src/compile/terminal_dwa/grammar_helpers.rs` | 168 | `compute_ever_allowed_follows` | `pub(crate) fn compute_ever_allowed_follows(grammar: &AnalyzedGrammar) -> Vec<Vec<TerminalID>> {` |
| `src/compile/terminal_dwa/grammar_helpers.rs` | 191 | `compute_always_allowed_follows` | `pub(crate) fn compute_always_allowed_follows(grammar: &AnalyzedGrammar) -> Vec<Vec<TerminalID>> {` |
| `src/compile/terminal_dwa/grammar_helpers.rs` | 217 | `occurrence_follow_set` | `fn occurrence_follow_set(` |
| `src/compile/terminal_dwa/merge.rs` | 22 | `minimize_merged_terminal_dwa_enabled` | `fn minimize_merged_terminal_dwa_enabled() -> bool {` |
| `src/compile/terminal_dwa/merge.rs` | 34 | `compact_merged_terminal_dwa_enabled` | `fn compact_merged_terminal_dwa_enabled() -> bool {` |
| `src/compile/terminal_dwa/merge.rs` | 47 | `merge_local_id_maps_and_terminal_dwas` | `pub(crate) fn merge_local_id_maps_and_terminal_dwas(` |
| `src/compile/terminal_dwa/merge.rs` | 103 | `merge_id_maps_and_terminal_dwas` | `pub(crate) fn merge_id_maps_and_terminal_dwas(` |
| `src/compile/terminal_dwa/merge.rs` | 257 | `build_unified_global_id_map` | `fn build_unified_global_id_map(` |
| `src/compile/terminal_dwa/merge.rs` | 300 | `build_unified_global_token_id_map_generic` | `fn build_unified_global_token_id_map_generic(` |
| `src/compile/terminal_dwa/merge.rs` | 352 | `build_unified_global_token_id_map_disjoint` | `fn build_unified_global_token_id_map_disjoint(` |
| `src/compile/terminal_dwa/merge.rs` | 406 | `build_direct_local_to_global_token_map` | `fn build_direct_local_to_global_token_map(local_to_global: &[u32]) -> Vec<Vec<u32>> {` |
| `src/compile/terminal_dwa/merge.rs` | 419 | `reorder_classes` | `fn reorder_classes(` |
| `src/compile/terminal_dwa/merge.rs` | 452 | `reorder_classes_with_sentinel` | `fn reorder_classes_with_sentinel(` |
| `src/compile/terminal_dwa/merge.rs` | 489 | `build_local_to_global_tsid_map` | `fn build_local_to_global_tsid_map(` |
| `src/compile/terminal_dwa/merge.rs` | 513 | `build_local_to_global_token_map` | `fn build_local_to_global_token_map(` |
| `src/compile/terminal_dwa/merge.rs` | 548 | `remap_nwa_with_maps` | `fn remap_nwa_with_maps(` |
| `src/compile/terminal_dwa/merge.rs` | 597 | `remap_weight_cached` | `fn remap_weight_cached(` |
| `src/compile/terminal_dwa/merge.rs` | 618 | `remap_weight_general` | `fn remap_weight_general(` |
| `src/compile/terminal_dwa/options.rs` | 15 | `VocabPartitionScheme` | `pub(crate) enum VocabPartitionScheme {` |
| `src/compile/terminal_dwa/options.rs` | 26 | `as_str` | `pub(crate) fn as_str(self) -> &'static str {` |
| `src/compile/terminal_dwa/options.rs` | 35 | `parse_truthy` | `fn parse_truthy(value: &str) -> bool {` |
| `src/compile/terminal_dwa/options.rs` | 40 | `vocab_partition_scheme_from_env` | `pub(crate) fn vocab_partition_scheme_from_env() -> VocabPartitionScheme {` |
| `src/compile/terminal_dwa/options.rs` | 51 | `pair_partition_cost_fn_from_env` | `pub(crate) fn pair_partition_cost_fn_from_env() -> PairPartitionCostFn {` |
| `src/compile/terminal_dwa/options.rs` | 63 | `pair_partition_objective_from_env` | `pub(crate) fn pair_partition_objective_from_env() -> PairPartitionObjective {` |
| `src/compile/terminal_dwa/options.rs` | 73 | `pair_partition_count_from_env` | `pub(crate) fn pair_partition_count_from_env() -> usize {` |
| `src/compile/terminal_dwa/options.rs` | 81 | `pair_partition_auto_second_largest_limit_from_env` | `pub(crate) fn pair_partition_auto_second_largest_limit_from_env() -> usize {` |
| `src/compile/terminal_dwa/options.rs` | 89 | `pair_partition_auto_max_estimated_pair_partition_terminals_from_env` | `pub(crate) fn pair_partition_auto_max_estimated_pair_partition_terminals_from_env() -> usize {` |
| `src/compile/terminal_dwa/options.rs` | 97 | `pair_partition_auto_min_estimated_pair_partition_terminals_from_env` | `pub(crate) fn pair_partition_auto_min_estimated_pair_partition_terminals_from_env() -> usize {` |
| `src/compile/terminal_dwa/options.rs` | 104 | `pair_partition_auto_min_grammar_terminals_from_env` | `pub(crate) fn pair_partition_auto_min_grammar_terminals_from_env() -> usize {` |
| `src/compile/terminal_dwa/options.rs` | 111 | `global_max_length_env_override` | `pub(crate) fn global_max_length_env_override() -> Option<bool> {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/combined.rs` | 24 | `deduplicate_tokens_by_byte_class` | `fn deduplicate_tokens_by_byte_class<'a, S: AsRef<[u8]>>(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/combined.rs` | 50 | `adjust_disallowed_follows` | `fn adjust_disallowed_follows(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/combined.rs` | 66 | `build_state_map` | `fn build_state_map(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/combined.rs` | 92 | `build_state_map_composed` | `fn build_state_map_composed(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/combined.rs` | 137 | `build_vocab_map` | `fn build_vocab_map(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/combined.rs` | 178 | `PreparedEquivalenceInputs` | `struct PreparedEquivalenceInputs<'a> {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/combined.rs` | 185 | `prepare_equivalence_inputs` | `fn prepare_equivalence_inputs<'a>(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/combined.rs` | 212 | `CombinedEquivalenceResult` | `struct CombinedEquivalenceResult {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/combined.rs` | 217 | `CombinedEquivalenceProfile` | `pub(crate) struct CombinedEquivalenceProfile {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/combined.rs` | 238 | `TokenLengthStats` | `struct TokenLengthStats {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/combined.rs` | 246 | `skip_max_length_for_partition` | `fn skip_max_length_for_partition(partition_label: &str) -> bool {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/combined.rs` | 270 | `should_skip_max_length_for_partition` | `fn should_skip_max_length_for_partition(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/combined.rs` | 282 | `token_length_stats` | `fn token_length_stats(tokens: &[&[u8]]) -> TokenLengthStats {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/combined.rs` | 311 | `build_internal_id_map_from_combined_result` | `fn build_internal_id_map_from_combined_result(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/combined.rs` | 330 | `analyze_equivalences_with_group_filter` | `pub(crate) fn analyze_equivalences_with_group_filter(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/combined.rs` | 348 | `analyze_equivalences_impl` | `fn analyze_equivalences_impl(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs` | 7 | `build_transition_table` | `fn build_transition_table(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs` | 17 | `collect_group_ids` | `fn collect_group_ids(groups: impl Iterator<Item = usize>) -> Vec<usize> {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs` | 24 | `FlatDfaState` | `pub struct FlatDfaState {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs` | 35 | `FlatDfa` | `pub struct FlatDfa {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs` | 43 | `compute_byte_classes` | `pub(crate) fn compute_byte_classes(dfa: &FlatDfa) -> [u8; 256] {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs` | 101 | `trans` | `pub fn trans(&self, state: usize, byte: usize) -> u32 {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs` | 107 | `transitions_for` | `pub fn transitions_for(&self, state: usize) -> &[u32] {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs` | 111 | `from_tokenizer` | `pub fn from_tokenizer(tokenizer: &Tokenizer) -> Self {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs` | 144 | `from_tokenizer_filtered` | `pub fn from_tokenizer_filtered(tokenizer: &Tokenizer, active_groups: &[bool]) -> Self {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs` | 184 | `from_flat_trans` | `pub fn from_flat_trans(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs` | 209 | `from_flat_trans_filtered` | `pub fn from_flat_trans_filtered(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs` | 246 | `TokenizerView` | `pub struct TokenizerView {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs` | 251 | `new` | `pub fn new(tokenizer: &Tokenizer) -> Self {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs` | 258 | `new_filtered` | `pub fn new_filtered(tokenizer: &Tokenizer, active_groups: &[bool]) -> Self {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs` | 266 | `new_from_flat_trans` | `pub fn new_from_flat_trans(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs` | 277 | `new_filtered_from_flat_trans` | `pub fn new_filtered_from_flat_trans(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs` | 287 | `dfa` | `pub fn dfa(&self) -> &FlatDfa {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs` | 291 | `initial_state_id` | `pub fn initial_state_id(&self) -> usize {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/disallowed_follows.rs` | 6 | `normalize_disallowed_follows` | `pub(crate) fn normalize_disallowed_follows(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/disallowed_follows.rs` | 25 | `build_disallowed_follow_dfa` | `pub(crate) fn build_disallowed_follow_dfa(disallowed_follows: &[BitSet]) -> DFA {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/shared.rs` | 4 | `TokenDedup` | `pub(crate) struct TokenDedup<'a> {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/shared.rs` | 10 | `hash_byte_class_seq` | `pub(crate) fn hash_byte_class_seq(bytes: &[u8], byte_to_class: &[u8; 256]) -> u128 {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/shared.rs` | 24 | `expand_vocab_classes` | `pub(crate) fn expand_vocab_classes(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/shared.rs` | 51 | `representative_tokens_for_vocab_classes` | `pub(crate) fn representative_tokens_for_vocab_classes<'a>(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/shared.rs` | 61 | `tokenizer_group_count` | `pub(crate) fn tokenizer_group_count(tokenizer: &TokenizerView) -> usize {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 14 | `StateEquivalenceResult` | `pub type StateEquivalenceResult = BTreeSet<BTreeSet<usize>>;` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 17 | `WalkFrame` | `struct WalkFrame {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 24 | `bit_words` | `fn bit_words(num_bits: usize) -> usize {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 29 | `bitset_set` | `fn bitset_set(bits: &mut [u64], idx: usize) {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 34 | `bitset_clear` | `fn bitset_clear(bits: &mut [u64], idx: usize) {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 39 | `clear_active_positions` | `fn clear_active_positions(positions: &mut [i32], active_bits: &mut [u64]) {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 52 | `mix_u128` | `fn mix_u128(mut x: u128) -> u128 {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 62 | `mix_tagged` | `fn mix_tagged(hash: u128, tag: u128, value: u128) -> u128 {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 66 | `hash_future_groups` | `fn hash_future_groups(future_groups: &[usize]) -> u128 {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 74 | `hash_future_groups_filtered` | `fn hash_future_groups_filtered(future_groups: &[usize], disallowed: &BitSet) -> u128 {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 89 | `FollowContextTable` | `struct FollowContextTable {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 95 | `new` | `fn new(num_groups: usize, disallowed_follows: Option<&[BitSet]>) -> Self {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 133 | `num_contexts` | `fn num_contexts(&self) -> usize {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 138 | `context_for_gid` | `fn context_for_gid(&self, gid: usize) -> usize {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 143 | `allows_follow` | `fn allows_follow(&self, context: usize, gid: usize) -> bool {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 149 | `SuffixNode` | `struct SuffixNode {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 154 | `TokenSuffixHashes` | `struct TokenSuffixHashes {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 162 | `get` | `fn get(&self, context: usize, pos: usize) -> u128 {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 167 | `build_future_group_hashes_by_context` | `fn build_future_group_hashes_by_context(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 188 | `hash_suffix_node` | `fn hash_suffix_node(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 252 | `build_token_suffix_hashes` | `fn build_token_suffix_hashes(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 284 | `hash_trellis_node_from_positions` | `fn hash_trellis_node_from_positions(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 335 | `build_strided_batches` | `fn build_strided_batches(total_tokens: usize, target_batch_size: usize) -> Vec<Vec<usize>> {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 354 | `build_start_state_suffix_nodes` | `fn build_start_state_suffix_nodes(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 425 | `find_state_equivalence_classes_with_disallowed` | `pub fn find_state_equivalence_classes_with_disallowed<S: AsRef<[u8]> + Sync>(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 444 | `find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed` | `pub fn find_state_equivalence_classes_ex_with_rep_confirmation_and_disallowed<` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 468 | `find_state_equivalence_classes_ex_inner` | `fn find_state_equivalence_classes_ex_inner<S: AsRef<[u8]> + Sync>(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 496 | `find_state_equivalence_classes_token_based` | `fn find_state_equivalence_classes_token_based<S: AsRef<[u8]> + Sync>(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs` | 1040 | `mapping_to_equivalence_classes` | `pub fn mapping_to_equivalence_classes(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 15 | `ActiveTransitionTable` | `struct ActiveTransitionTable {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 21 | `RefineMode` | `enum RefineMode {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 27 | `refine_mode` | `fn refine_mode() -> RefineMode {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 36 | `is_full_state_query` | `fn is_full_state_query(states: &[usize], total_states: usize) -> bool {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 47 | `mix_u64` | `fn mix_u64(mut x: u64) -> u64 {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 57 | `hash_signature_row` | `fn hash_signature_row(row: &[u32]) -> u64 {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 66 | `usize_to_u32` | `fn usize_to_u32(value: usize, what: &str) -> u32 {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 71 | `is_active_group` | `fn is_active_group(group_id: usize, active_groups: Option<&[bool]>) -> bool {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 77 | `filtered_group_ids` | `fn filtered_group_ids(values: &[usize], active_groups: Option<&[bool]>) -> Vec<usize> {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 85 | `build_filtered_finalizer_labels` | `fn build_filtered_finalizer_labels(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 95 | `build_filtered_possible_future_labels` | `fn build_filtered_possible_future_labels(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 105 | `build_has_any_transition_labels` | `fn build_has_any_transition_labels(dfa: &FlatDfa) -> Vec<bool> {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 117 | `byte_is_relevant` | `fn byte_is_relevant(byte: usize, relevant_bytes: Option<&[bool; 256]>) -> bool {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 121 | `active_byte_representatives` | `fn active_byte_representatives(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 148 | `build_initial_label_partition` | `fn build_initial_label_partition(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 190 | `same_partition` | `fn same_partition(left: &[u32], left_count: usize, right: &[u32], right_count: usize) -> bool {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 221 | `build_active_transition_table` | `fn build_active_transition_table(dfa: &FlatDfa, active_bytes: &[u8]) -> ActiveTransitionTable {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 241 | `refine_once_sorted` | `fn refine_once_sorted(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 312 | `row_hash` | `fn row_hash(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 335 | `rows_equal` | `fn rows_equal(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 369 | `refine_once_interned` | `fn refine_once_interned(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 422 | `auto_prefers_sorted_refinement` | `fn auto_prefers_sorted_refinement(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 430 | `compute_kbounded_partition` | `fn compute_kbounded_partition(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 491 | `build_subset_mapping` | `fn build_subset_mapping(states: &[usize], blocks: &[u32]) -> Vec<usize> {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 520 | `find_state_equivalence_classes_kbounded` | `pub(crate) fn find_state_equivalence_classes_kbounded(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs` | 543 | `find_state_equivalence_classes_byte_restricted` | `pub fn find_state_equivalence_classes_byte_restricted<S: AsRef<[u8]>>(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 13 | `MaxLengthMode` | `pub(crate) enum MaxLengthMode {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 19 | `name` | `pub(crate) fn name(self) -> &'static str {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 28 | `MaxLengthStatistic` | `pub(crate) struct MaxLengthStatistic {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 34 | `mix_u64` | `fn mix_u64(mut x: u64) -> u64 {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 43 | `is_active_group` | `fn is_active_group(group_id: usize, active_groups: Option<&[bool]>) -> bool {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 49 | `filtered_terminals` | `fn filtered_terminals(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 61 | `build_filtered_finalizer_labels` | `fn build_filtered_finalizer_labels(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 71 | `build_filtered_possible_future_labels` | `fn build_filtered_possible_future_labels(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 83 | `build_has_any_transition_labels` | `fn build_has_any_transition_labels(tokenizer: &Tokenizer) -> Vec<bool> {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 92 | `build_initial_label_partition` | `fn build_initial_label_partition(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 134 | `byte_is_relevant` | `fn byte_is_relevant(byte: usize, relevant_bytes: Option<&[bool; 256]>) -> bool {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 138 | `active_byte_representatives` | `fn active_byte_representatives(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 165 | `compute_byte_classes` | `fn compute_byte_classes(tokenizer: &Tokenizer) -> [u8; 256] {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 219 | `same_partition` | `fn same_partition(left: &[u32], left_count: usize, right: &[u32], right_count: usize) -> bool {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 249 | `refine_once_sorted` | `fn refine_once_sorted(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 316 | `build_full_mapping_from_blocks` | `fn build_full_mapping_from_blocks(blocks: &[u32], num_states: usize) -> Vec<usize> {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 335 | `build_subset_mapping` | `fn build_subset_mapping(states: &[usize], full_mapping: &[usize]) -> Vec<usize> {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 342 | `compute_kbounded_partition` | `fn compute_kbounded_partition(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 383 | `stable_refinement_blocks` | `fn stable_refinement_blocks(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 423 | `compute_statistic` | `pub(crate) fn compute_statistic(vocab: &Vocab) -> MaxLengthStatistic {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs` | 438 | `compute_state_map` | `pub(crate) fn compute_state_map(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/mod.rs` | 11 | `identity_state_map` | `pub(crate) fn identity_state_map(num_states: usize) -> ManyToOneIdMap {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/mod.rs` | 20 | `build_state_map_from_subset_representatives` | `pub(crate) fn build_state_map_from_subset_representatives(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/pipeline.rs` | 11 | `StateEquivalencePassKind` | `pub(crate) enum StateEquivalencePassKind {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/pipeline.rs` | 16 | `parse` | `fn parse(value: &str) -> Result<Self, String> {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/pipeline.rs` | 27 | `StateEquivalenceScope` | `pub(crate) enum StateEquivalenceScope {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/pipeline.rs` | 33 | `StateEquivalencePipelineConfig` | `pub(crate) struct StateEquivalencePipelineConfig {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/pipeline.rs` | 38 | `StateEquivalencePassProfile` | `pub(crate) struct StateEquivalencePassProfile {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/pipeline.rs` | 47 | `StateEquivalencePipelineProfile` | `pub(crate) struct StateEquivalencePipelineProfile {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/pipeline.rs` | 54 | `parse_passes` | `fn parse_passes(value: &str) -> Vec<StateEquivalencePassKind> {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/pipeline.rs` | 66 | `resolve_pipeline_config` | `fn resolve_pipeline_config(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/pipeline.rs` | 82 | `resolve_global_pipeline_config` | `pub(crate) fn resolve_global_pipeline_config(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/pipeline.rs` | 93 | `resolve_pair_partition_pipeline_config` | `pub(crate) fn resolve_pair_partition_pipeline_config(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/pipeline.rs` | 104 | `run_state_equivalence_pipeline` | `pub(crate) fn run_state_equivalence_pipeline(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/pipeline.rs` | 154 | `record_max_length_profile` | `fn record_max_length_profile(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 21 | `VocabEquivalenceResult` | `pub type VocabEquivalenceResult = BTreeSet<Vec<usize>>;` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 23 | `EdgeList` | `type EdgeList = SmallVec<[(usize, usize); 4]>;` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 25 | `DagNode` | `struct DagNode {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 46 | `Dfa` | `struct Dfa {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 70 | `SharedVocabDfaBase` | `pub struct SharedVocabDfaBase {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 83 | `build_from_dfa` | `pub fn build_from_dfa(dfa: &super::super::compat::FlatDfa) -> Self {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 147 | `byte_to_class` | `pub fn byte_to_class(&self) -> [u8; 256] {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 154 | `is_compatible_with_dfa` | `pub fn is_compatible_with_dfa(&self, dfa: &super::super::compat::FlatDfa) -> bool {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 172 | `SharedVocabDfaCache` | `pub type SharedVocabDfaCache = std::sync::OnceLock<SharedVocabDfaBase>;` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 177 | `completion` | `fn completion(&self, state: usize) -> u64 {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 186 | `completion_with_disallowed` | `fn completion_with_disallowed(&self, state: usize, disallowed: Option<&BitSet>) -> u64 {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 201 | `transition` | `fn transition(&self, state: usize, byte: u8) -> u32 {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 207 | `disallowed_for` | `fn disallowed_for(&self, gid: usize) -> &BitSet {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 213 | `Scratch` | `struct Scratch {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 248 | `new_hasher` | `fn new_hasher() -> AHasher {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 252 | `env_flag_enabled` | `fn env_flag_enabled(name: &str) -> bool {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 261 | `vocab_batch_size_override` | `fn vocab_batch_size_override() -> Option<usize> {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 268 | `vocab_verify_token_pair_override` | `fn vocab_verify_token_pair_override() -> Option<(usize, usize)> {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 279 | `vocab_verify_token_pair_from_final_classes_enabled` | `fn vocab_verify_token_pair_from_final_classes_enabled() -> bool {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 283 | `vocab_state_group_size` | `fn vocab_state_group_size(num_states: usize, num_groups: usize) -> usize {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 293 | `diversity_state_order_enabled` | `fn diversity_state_order_enabled() -> bool {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 297 | `states_by_transition_diversity` | `fn states_by_transition_diversity(dfa: &Dfa, states: &[usize]) -> Vec<usize> {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 327 | `hash_group_list` | `fn hash_group_list(iter: impl ExactSizeIterator<Item = usize>) -> u64 {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 338 | `hash_filtered_group_list` | `fn hash_filtered_group_list(groups: &[usize], disallowed: &BitSet) -> u64 {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 356 | `build_dfa` | `fn build_dfa(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 373 | `build_dfa_with_group_filter` | `fn build_dfa_with_group_filter(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 497 | `intersect_node_disallowed` | `fn intersect_node_disallowed(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 510 | `node_disallows_gid` | `fn node_disallows_gid(scratch: &Scratch, pos: usize, gid: usize) -> bool {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 520 | `ensure_position_slot` | `fn ensure_position_slot<T>(slots: &mut Vec<Option<T>>, pos: usize) {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 527 | `new` | `fn new(num_states: usize, num_groups: usize) -> Self {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 554 | `mark_dirty_group` | `fn mark_dirty_group(scratch: &mut Scratch, state_idx: usize, gid: usize) {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 562 | `ensure_target_gids_map` | `fn ensure_target_gids_map(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 574 | `advance_seen_epoch` | `fn advance_seen_epoch(seen: &mut [u32], epoch: &mut u32) {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 582 | `record_target_gid` | `fn record_target_gid(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 627 | `run_batch_inner` | `fn run_batch_inner(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 747 | `collect_targets` | `fn collect_targets(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 810 | `run_batch` | `fn run_batch(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 860 | `hash_suffixes` | `fn hash_suffixes(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 979 | `run_suffix` | `fn run_suffix(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1036 | `try_hash_single_target_suffix` | `fn try_hash_single_target_suffix(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1090 | `finish_token_signature` | `fn finish_token_signature(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1146 | `fill_state_observation_words_and_cleanup` | `fn fill_state_observation_words_and_cleanup(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1202 | `compute_token_state_observation_words` | `fn compute_token_state_observation_words(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1231 | `first_distinguishing_state_for_token_pair_with_count` | `fn first_distinguishing_state_for_token_pair_with_count<S: AsRef<[u8]>>(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1286 | `first_distinguishing_state_for_token_pair` | `fn first_distinguishing_state_for_token_pair<S: AsRef<[u8]>>(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1307 | `log_vocab_pair_verification` | `fn log_vocab_pair_verification<S: AsRef<[u8]>>(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1357 | `run_vocab_row_cert_diag` | `fn run_vocab_row_cert_diag<S: AsRef<[u8]> + Sync>(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1424 | `token_signature` | `fn token_signature(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1469 | `DepthChangeLog` | `struct DepthChangeLog {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1481 | `new` | `fn new() -> Self {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1490 | `clear` | `fn clear(&mut self) {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1498 | `TrieWalkState` | `struct TrieWalkState {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1503 | `new` | `fn new() -> Self {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1509 | `ensure_depth` | `fn ensure_depth(&mut self, depth: usize) {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1517 | `TrieWalkChunkStats` | `struct TrieWalkChunkStats {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1538 | `add_assign` | `fn add_assign(&mut self, other: Self) {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1560 | `DfsStepStats` | `struct DfsStepStats {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1570 | `dfs_step` | `fn dfs_step(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1635 | `dfs_step_profiled` | `fn dfs_step_profiled(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1717 | `dfs_undo_depth` | `fn dfs_undo_depth(scratch: &mut Scratch, log: &DepthChangeLog) {` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1733 | `dfs_backtrack` | `fn dfs_backtrack(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1749 | `finish_token_signature_clean` | `fn finish_token_signature_clean(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1782 | `finish_token_signature_no_cleanup` | `fn finish_token_signature_no_cleanup(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 1836 | `trie_walk_chunk_signatures` | `fn trie_walk_chunk_signatures<S: AsRef<[u8]> + Sync>(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 2018 | `compact_dfa_for_tokens` | `fn compact_dfa_for_tokens<S: AsRef<[u8]>>(` |
| `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs` | 2149 | `find_vocab_equivalence_classes_with_group_filter` | `pub fn find_vocab_equivalence_classes_with_group_filter<S: AsRef<[u8]> + Sync>(` |
| `src/compile/terminal_dwa/pair_partition/mod.rs` | 47 | `SimplifyCacheKey` | `struct SimplifyCacheKey {` |
| `src/compile/terminal_dwa/pair_partition/mod.rs` | 53 | `SharedSimplifyCache` | `pub(crate) struct SharedSimplifyCache {` |
| `src/compile/terminal_dwa/pair_partition/mod.rs` | 57 | `SimplifyCacheEntry` | `struct SimplifyCacheEntry {` |
| `src/compile/terminal_dwa/pair_partition/mod.rs` | 63 | `new` | `fn new() -> Self {` |
| `src/compile/terminal_dwa/pair_partition/mod.rs` | 72 | `key` | `fn key(active_terminals: &[bool], relevant_bytes: &[bool; 256]) -> SimplifyCacheKey {` |
| `src/compile/terminal_dwa/pair_partition/mod.rs` | 93 | `simplify_for_terminals` | `fn simplify_for_terminals(` |
| `src/compile/terminal_dwa/pair_partition/mod.rs` | 146 | `project_initial_state_map_enabled` | `fn project_initial_state_map_enabled() -> bool {` |
| `src/compile/terminal_dwa/pair_partition/mod.rs` | 159 | `PairPartitionTokenLengthStats` | `struct PairPartitionTokenLengthStats {` |
| `src/compile/terminal_dwa/pair_partition/mod.rs` | 168 | `pair_partition_token_length_stats` | `fn pair_partition_token_length_stats(vocab: &Vocab) -> PairPartitionTokenLengthStats {` |
| `src/compile/terminal_dwa/pair_partition/mod.rs` | 201 | `ProjectInitialStateMapProfile` | `struct ProjectInitialStateMapProfile {` |
| `src/compile/terminal_dwa/pair_partition/mod.rs` | 213 | `unused` | `fn unused(reason: &'static str, simplified_state_count: usize) -> Self {` |
| `src/compile/terminal_dwa/pair_partition/mod.rs` | 227 | `project_initial_state_map_for_simplified_tokenizer` | `fn project_initial_state_map_for_simplified_tokenizer(` |
| `src/compile/terminal_dwa/pair_partition/mod.rs` | 370 | `build_pair_partition_terminal_dwa` | `pub(crate) fn build_pair_partition_terminal_dwa(` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 27 | `NwaState` | `type NwaState = u32;` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 29 | `TokenizerState` | `type TokenizerState = u32;` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 30 | `LeafTokenIds` | `type LeafTokenIds = SmallVec<[u32; 8]>;` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 31 | `FutureTerminalColorGroups` | `type FutureTerminalColorGroups = SmallVec<[(ColorId, SmallVec<[TerminalID; 4]>); 8]>;` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 33 | `all_token_weight` | `fn all_token_weight(internal_tsid: u32, max_token_id: u32) -> Weight {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 41 | `NodesByTokenizerState` | `pub(crate) struct NodesByTokenizerState {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 46 | `new` | `fn new() -> Self {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 52 | `is_empty` | `fn is_empty(&self) -> bool {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 56 | `merge` | `fn merge(&mut self, state: TokenizerState, nodes: &[NwaState]) {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 60 | `first` | `fn first(&self, state: TokenizerState) -> Option<NwaState> {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 64 | `push_one` | `fn push_one(&mut self, state: TokenizerState, node: NwaState) {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 68 | `iter` | `fn iter(&self) -> impl Iterator<Item = (TokenizerState, &[NwaState])> {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 76 | `Item` | `type Item = (TokenizerState, Vec<NwaState>);` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 77 | `IntoIter` | `type IntoIter = <FxHashMap<TokenizerState, Vec<NwaState>> as IntoIterator>::IntoIter;` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 79 | `into_iter` | `fn into_iter(self) -> Self::IntoIter {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 84 | `TerminalNwaBuilder` | `pub(crate) struct TerminalNwaBuilder<'tok, 'cm, 'nwa> {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 112 | `BufferedLeafTransition` | `struct BufferedLeafTransition {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 118 | `new` | `pub(crate) fn new(` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 161 | `fast_step` | `fn fast_step(&mut self, state: u32, byte: u8) -> Option<u32> {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 175 | `leaf_token_ids_for` | `fn leaf_token_ids_for(&mut self, source: u32, label: TerminalID) -> &mut LeafTokenIds {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 190 | `buffer_leaf_token_id` | `fn buffer_leaf_token_id(&mut self, source: u32, label: TerminalID, internal_token_id: u32) {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 194 | `possible_future_terminals_for_state` | `fn possible_future_terminals_for_state(&mut self, tokenizer_state: TokenizerState) -> Vec<TerminalID> {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 205 | `populate_future_terminal_color_cache` | `fn populate_future_terminal_color_cache(&mut self, tokenizer_state: TokenizerState) {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 234 | `ignore_terminal_possible_for_state` | `fn ignore_terminal_possible_for_state(&mut self, tokenizer_state: TokenizerState) -> bool {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 245 | `future_terminal_colors_for_state` | `fn future_terminal_colors_for_state(` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 256 | `future_terminal_color_groups_for_state` | `fn future_terminal_color_groups_for_state(` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 267 | `buffer_future_leaf_token_id` | `fn buffer_future_leaf_token_id(` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 282 | `add_future_leaf_token_from_sources` | `fn add_future_leaf_token_from_sources(` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 313 | `add_future_weighted_match_from_sources` | `fn add_future_weighted_match_from_sources(` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 355 | `cached_reachable_weight` | `fn cached_reachable_weight(&mut self, token_ids: &RangeSetBlaze<usize>) -> Weight {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 367 | `token_set_weight_fast` | `fn token_set_weight_fast(&self, internal_token_ids: &RangeSetBlaze<usize>) -> Weight {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 378 | `cached_leaf_weight` | `fn cached_leaf_weight(&mut self, mut token_ids: LeafTokenIds) -> Weight {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 392 | `continuation_weight_for_match` | `fn continuation_weight_for_match(` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 434 | `add_leaf_token_from_sources` | `fn add_leaf_token_from_sources(` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 458 | `can_skip_self_loop_subtree` | `fn can_skip_self_loop_subtree(` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 476 | `emit_self_loop_leaf_only_subtree` | `fn emit_self_loop_leaf_only_subtree(` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 498 | `add_match_from_sources` | `fn add_match_from_sources(` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 522 | `flush_transition_buffer` | `pub(crate) fn flush_transition_buffer(&mut self) {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 656 | `build_direct_partition_fast` | `pub(crate) fn build_direct_partition_fast(` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 771 | `build_from_trie` | `pub(crate) fn build_from_trie(` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 806 | `process_child_segment` | `fn process_child_segment(` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 933 | `subtract_can_match` | `fn subtract_can_match(` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 942 | `ensure_continuation_state` | `fn ensure_continuation_state(` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 956 | `internal_vocab_entries` | `pub(crate) fn internal_vocab_entries(vocab: &Vocab, id_map: &InternalIdMap) -> Vec<(u32, Vec<u8>)> {` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 970 | `seed_root_nodes` | `pub(crate) fn seed_root_nodes(` |
| `src/compile/terminal_dwa/pair_partition/nwa_builder.rs` | 991 | `build_nwa_via_trie_walk` | `pub(crate) fn build_nwa_via_trie_walk<'a>(` |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 21 | `structural_hash_nwa_state` | `fn structural_hash_nwa_state(state: &NWAStateType) -> u64 {` |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 48 | `hash_weight` | `fn hash_weight(weight: &Weight, hasher: &mut impl Hasher) {` |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 52 | `canonicalize_acyclic_nwa` | `pub(crate) fn canonicalize_acyclic_nwa(nwa: &mut NWA) {` |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 142 | `retain_nwa_states` | `fn retain_nwa_states(nwa: &mut NWA, retain: &[bool], drop_empty_weights: bool) -> bool {` |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 187 | `compute_forward_reachable` | `fn compute_forward_reachable(nwa: &NWA) -> Vec<bool> {` |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 223 | `prune_unreachable_states` | `pub(crate) fn prune_unreachable_states(nwa: &mut NWA) -> bool {` |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 231 | `topological_order` | `fn topological_order(nwa: &NWA) -> Vec<usize> {` |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 274 | `compute_coreachable_nwa` | `fn compute_coreachable_nwa(nwa: &NWA) -> Vec<bool> {` |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 316 | `prune_non_coreachable_states` | `pub(crate) fn prune_non_coreachable_states(nwa: &mut NWA) -> bool {` |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 326 | `propagate_incoming_labels` | `fn propagate_incoming_labels(` |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 370 | `propagate_collapse_context` | `fn propagate_collapse_context(` |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 437 | `allowed_labels_by_state` | `fn allowed_labels_by_state(` |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 467 | `collapse_single_allowed_transitions` | `fn collapse_single_allowed_transitions(` |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 543 | `collapse_always_allowed` | `pub(crate) fn collapse_always_allowed(` |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 599 | `SharedDisallowedFollowDfaCache` | `pub(crate) struct SharedDisallowedFollowDfaCache {` |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 604 | `new` | `pub(crate) fn new() -> Self {` |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 608 | `get_or_build` | `fn get_or_build(` |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 628 | `apply_disallowed_follow_constraints` | `pub(crate) fn apply_disallowed_follow_constraints(` |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 651 | `subtract_disallowed_dfa` | `fn subtract_disallowed_dfa(nwa: &NWA, right: &DFA) -> NWA {` |
| `src/compile/terminal_dwa/pair_partition/postprocess.rs` | 652 | `ProdState` | `type ProdState = (u32, Option<u32>);` |
| `src/compile/terminal_dwa/partition.rs` | 31 | `build_partition_terminal_dwa` | `pub(crate) fn build_partition_terminal_dwa(` |
| `src/compile/terminal_dwa/types.rs` | 8 | `ColorId` | `pub(crate) type ColorId = u32;` |
| `src/compile/terminal_dwa/types.rs` | 14 | `TerminalColoring` | `pub(crate) struct TerminalColoring {` |
| `src/compile/terminal_dwa/types.rs` | 20 | `identity` | `pub(crate) fn identity(num_terminals: usize) -> Self {` |
| `src/compile/terminal_dwa/types.rs` | 28 | `color_for` | `pub(crate) fn color_for(&self, terminal_id: TerminalID) -> ColorId {` |
| `src/compile/terminal_dwa/types.rs` | 38 | `TerminalDwaBuildProfile` | `pub(crate) struct TerminalDwaBuildProfile {` |
| `src/compile/terminal_dwa/types.rs` | 44 | `TerminalDwaPhaseProfile` | `pub(crate) struct TerminalDwaPhaseProfile {` |
| `src/compile/terminal_dwa/types.rs` | 53 | `LocalIdMapTerminalDwa` | `pub(crate) struct LocalIdMapTerminalDwa {` |
| `src/compile/terminal_dwa/types.rs` | 60 | `total_ms` | `pub(crate) fn total_ms(self) -> f64 {` |
| `src/compile/terminal_dwa/types.rs` | 64 | `add_assign` | `pub(crate) fn add_assign(&mut self, other: Self) {` |
| `src/compile/terminal_dwa/types.rs` | 75 | `TerminalPathLength` | `pub(crate) enum TerminalPathLength {` |
| `src/compile/terminal_dwa/types.rs` | 84 | `compile_profile_enabled` | `pub(crate) fn compile_profile_enabled() -> bool {` |
| `src/compile/terminal_dwa/vocab_partition.rs` | 27 | `CharTypeSubVocabs` | `struct CharTypeSubVocabs {` |
| `src/compile/terminal_dwa/vocab_partition.rs` | 33 | `vocab_from_token_partitions` | `pub(crate) fn vocab_from_token_partitions(vocab: &Vocab, token_partitions: Vec<Vec<u32>>) -> Arc<[Vocab]> {` |
| `src/compile/terminal_dwa/vocab_partition.rs` | 47 | `build_char_type_sub_vocabs` | `pub(crate) fn build_char_type_sub_vocabs(vocab: &Vocab) -> Arc<[Vocab]> {` |
| `src/compile/terminal_dwa/vocab_partition.rs` | 72 | `prepare_vocab_for_terminal_dwa` | `pub(crate) fn prepare_vocab_for_terminal_dwa(vocab: &Vocab) {` |
| `src/compile/terminal_dwa/vocab_partition.rs` | 89 | `choose_terminal_dwa_sub_vocabs` | `pub(crate) fn choose_terminal_dwa_sub_vocabs(` |
| `src/compile/terminal_dwa/vocab_partition.rs` | 115 | `choose_cost_partitioned_sub_vocabs` | `fn choose_cost_partitioned_sub_vocabs(` |
| `src/compile/terminal_dwa/vocab_partition.rs` | 153 | `choose_auto_partitioned_sub_vocabs` | `fn choose_auto_partitioned_sub_vocabs(` |
| `src/compile/thread_pool.rs` | 20 | `run_with_compile_thread_pool` | `pub(crate) fn run_with_compile_thread_pool<F, R>(f: F) -> R` |
| `src/compile/tokenizer.rs` | 20 | `build_tokenizer` | `pub(crate) fn build_tokenizer(grammar: &GrammarDef) -> Tokenizer {` |
| `src/compile/tokenizer.rs` | 29 | `build_tokenizer_from_exprs` | `pub(crate) fn build_tokenizer_from_exprs(exprs: &[Expr]) -> Tokenizer {` |
| `src/compile/tokenizer.rs` | 39 | `terminal_expr` | `fn terminal_expr(terminal: &Terminal) -> Expr {` |
| `src/compile/tokenizer.rs` | 47 | `emit_tokenizer_detail` | `fn emit_tokenizer_detail(grammar: &GrammarDef, exprs: &[Expr]) {` |
| `src/compile/tokenizer.rs` | 71 | `tokenizer_is_independent_of_vocab` | `pub(crate) fn tokenizer_is_independent_of_vocab(_: &GrammarDef, _: &Vocab) -> bool {` |
| `src/compiler/compile.rs` | 15 | `prepare_vocab_for_compile` | `pub(crate) fn prepare_vocab_for_compile(vocab: &crate::Vocab) {` |
| `src/compiler/grammar/transforms.rs` | 13 | `env_var_enabled` | `fn env_var_enabled(key: &str, default: bool) -> bool {` |
| `src/compiler/grammar/transforms.rs` | 22 | `compile_profile_enabled` | `fn compile_profile_enabled() -> bool {` |
| `src/compiler/grammar/transforms.rs` | 27 | `elapsed_ms` | `fn elapsed_ms(started_at: Instant) -> f64 {` |
| `src/compiler/grammar/transforms.rs` | 31 | `emit_grammar_transform_profile` | `fn emit_grammar_transform_profile(` |
| `src/compiler/grammar/transforms.rs` | 57 | `expand_nullable_terminals` | `pub(crate) fn expand_nullable_terminals(` |
| `src/compiler/grammar/transforms.rs` | 113 | `remap_terminal_id` | `fn remap_terminal_id(terminal: &Terminal, new_id: TerminalID) -> Terminal {` |
| `src/compiler/grammar/transforms.rs` | 131 | `terminal_is_nullable` | `fn terminal_is_nullable(terminal: &Terminal) -> bool {` |
| `src/compiler/grammar/transforms.rs` | 139 | `nullable_terminals_for_grammar` | `fn nullable_terminals_for_grammar(grammar: &GrammarDef) -> BTreeSet<TerminalID> {` |
| `src/compiler/grammar/transforms.rs` | 148 | `TerminalIdentity` | `enum TerminalIdentity {` |
| `src/compiler/grammar/transforms.rs` | 154 | `terminal_identity` | `fn terminal_identity(terminal: &Terminal, is_ignore: bool) -> TerminalIdentity {` |
| `src/compiler/grammar/transforms.rs` | 175 | `compact_unused_terminals` | `pub(crate) fn compact_unused_terminals(grammar: &mut GrammarDef) {` |
| `src/compiler/grammar/transforms.rs` | 223 | `remap_terminal_names` | `fn remap_terminal_names(` |
| `src/compiler/grammar/transforms.rs` | 233 | `inline_single_use_nonterminals` | `pub(crate) fn inline_single_use_nonterminals(` |
| `src/compiler/grammar/transforms.rs` | 393 | `remove_cyclic_inline_candidates` | `fn remove_cyclic_inline_candidates(` |
| `src/compiler/grammar/transforms.rs` | 396 | `reaches_start` | `fn reaches_start(` |
| `src/compiler/grammar/transforms.rs` | 439 | `inline_post_bound_single_use_nonterminals` | `fn inline_post_bound_single_use_nonterminals(` |
| `src/compiler/grammar/transforms.rs` | 563 | `bound_runtime_reduction_length` | `pub(crate) fn bound_runtime_reduction_length(` |
| `src/compiler/grammar/transforms.rs` | 654 | `collect_protected_nonterminals` | `fn collect_protected_nonterminals(grammar: &GrammarDef) -> BTreeSet<NonterminalID> {` |
| `src/compiler/grammar/transforms.rs` | 665 | `prepare_grammar_transforms_only` | `pub(crate) fn prepare_grammar_transforms_only(grammar: GrammarDef) -> GrammarDef {` |
| `src/compiler/grammar/transforms.rs` | 685 | `prepare_grammar_transforms_impl` | `fn prepare_grammar_transforms_impl(` |
| `src/compiler/grammar/transforms.rs` | 835 | `nt` | `fn nt(id: NonterminalID) -> Symbol {` |
| `src/compiler/grammar/transforms.rs` | 839 | `t` | `fn t(id: TerminalID) -> Symbol {` |
| `src/compiler/grammar/transforms.rs` | 844 | `inline_single_use_nonterminals_skips_candidate_cycles` | `fn inline_single_use_nonterminals_skips_candidate_cycles() {` |
| `src/compiler/grammar/transforms.rs` | 882 | `inline_single_use_nonterminals_still_expands_acyclic_chains` | `fn inline_single_use_nonterminals_still_expands_acyclic_chains() {` |
| `src/compiler/stages/resolve_negatives.rs` | 14 | `QueryKey` | `type QueryKey = (u32, i32);` |
| `src/compiler/stages/resolve_negatives.rs` | 15 | `CancellationTask` | `type CancellationTask = (u32, u32, i32);` |
| `src/compiler/stages/resolve_negatives.rs` | 16 | `QueryWeights` | `type QueryWeights = Vec<Option<FxHashMap<QueryKey, Weight>>>;` |
| `src/compiler/stages/resolve_negatives.rs` | 17 | `QueuedQueries` | `type QueuedQueries = Vec<SmallQueuedQueries>;` |
| `src/compiler/stages/resolve_negatives.rs` | 18 | `DerivedEpsilons` | `type DerivedEpsilons = Vec<Option<FxHashMap<u32, Weight>>>;` |
| `src/compiler/stages/resolve_negatives.rs` | 19 | `SubsetMemo` | `type SubsetMemo = FxHashMap<(usize, usize), bool>;` |
| `src/compiler/stages/resolve_negatives.rs` | 22 | `SmallQueuedQueries` | `enum SmallQueuedQueries {` |
| `src/compiler/stages/resolve_negatives.rs` | 30 | `insert` | `fn insert(&mut self, query_key: QueryKey) -> bool {` |
| `src/compiler/stages/resolve_negatives.rs` | 52 | `remove` | `fn remove(&mut self, query_key: &QueryKey) {` |
| `src/compiler/stages/resolve_negatives.rs` | 77 | `merge_weight` | `fn merge_weight(entry: &mut Weight, add: Weight) -> bool {` |
| `src/compiler/stages/resolve_negatives.rs` | 94 | `intersect_with_single_weight_hint` | `fn intersect_with_single_weight_hint(` |
| `src/compiler/stages/resolve_negatives.rs` | 112 | `intersect_or_clone_right_if_subset` | `fn intersect_or_clone_right_if_subset(left: &Weight, right: &Weight) -> Weight {` |
| `src/compiler/stages/resolve_negatives.rs` | 120 | `intersect_or_clone_right_if_subset_cached` | `fn intersect_or_clone_right_if_subset_cached(` |
| `src/compiler/stages/resolve_negatives.rs` | 137 | `PredEdge` | `struct PredEdge<'a> {` |
| `src/compiler/stages/resolve_negatives.rs` | 143 | `GuardedFinalWeight` | `struct GuardedFinalWeight {` |
| `src/compiler/stages/resolve_negatives.rs` | 154 | `initial` | `fn initial(weight: Weight) -> Option<Self> {` |
| `src/compiler/stages/resolve_negatives.rs` | 161 | `is_guarded_by` | `fn is_guarded_by(&self, edge_weight: &Weight) -> bool {` |
| `src/compiler/stages/resolve_negatives.rs` | 167 | `intersection_with_edge` | `fn intersection_with_edge(&self, edge_weight: &Weight) -> Option<Self> {` |
| `src/compiler/stages/resolve_negatives.rs` | 198 | `merge_guarded_final_weight` | `fn merge_guarded_final_weight(` |
| `src/compiler/stages/resolve_negatives.rs` | 234 | `union_guarded_pending` | `fn union_guarded_pending(` |
| `src/compiler/stages/resolve_negatives.rs` | 258 | `enqueue_cancellation_task` | `fn enqueue_cancellation_task(` |
| `src/compiler/stages/resolve_negatives.rs` | 271 | `record_query_weight` | `fn record_query_weight(` |
| `src/compiler/stages/resolve_negatives.rs` | 284 | `queue_query_weight` | `fn queue_query_weight(` |
| `src/compiler/stages/resolve_negatives.rs` | 298 | `propagate_query_through_derived_epsilons` | `fn propagate_query_through_derived_epsilons(` |
| `src/compiler/stages/resolve_negatives.rs` | 351 | `extend_derived_epsilons` | `fn extend_derived_epsilons(` |
| `src/compiler/stages/resolve_negatives.rs` | 420 | `collect_non_empty_derived_epsilons` | `fn collect_non_empty_derived_epsilons(` |
| `src/compiler/stages/resolve_negatives.rs` | 439 | `compute_cancellations_range` | `pub(crate) fn compute_cancellations_range(` |
| `src/compiler/stages/resolve_negatives.rs` | 446 | `compute_cancellations_range_inner` | `fn compute_cancellations_range_inner(` |
| `src/compiler/stages/resolve_negatives.rs` | 585 | `apply_cancellations_range` | `pub(crate) fn apply_cancellations_range(nwa: &mut NWA, range: std::ops::Range<u32>) {` |
| `src/compiler/stages/resolve_negatives.rs` | 591 | `is_live_finality_edge` | `fn is_live_finality_edge(target_state: u32, weight: &Weight, state_count: usize) -> bool {` |
| `src/compiler/stages/resolve_negatives.rs` | 595 | `build_finality_preds_and_outdegree` | `fn build_finality_preds_and_outdegree<'a>(nwa: &'a NWA) -> (Vec<Vec<PredEdge<'a>>>, Vec<usize>) {` |
| `src/compiler/stages/resolve_negatives.rs` | 633 | `build_finality_reverse_topo_order` | `fn build_finality_reverse_topo_order(` |
| `src/compiler/stages/resolve_negatives.rs` | 660 | `collect_initial_final_weights` | `fn collect_initial_final_weights(nwa: &NWA) -> Vec<Option<GuardedFinalWeight>> {` |
| `src/compiler/stages/resolve_negatives.rs` | 672 | `write_final_weights` | `fn write_final_weights(nwa: &mut NWA, reachable_final_weights: Vec<Option<GuardedFinalWeight>>) {` |
| `src/compiler/stages/resolve_negatives.rs` | 680 | `propagate_final_weights_to_predecessors` | `fn propagate_final_weights_to_predecessors(` |
| `src/compiler/stages/resolve_negatives.rs` | 700 | `apply_finality_fixpoint_worklist` | `fn apply_finality_fixpoint_worklist(` |
| `src/compiler/stages/resolve_negatives.rs` | 734 | `apply_finality_fixpoint_acyclic` | `fn apply_finality_fixpoint_acyclic(` |
| `src/compiler/stages/resolve_negatives.rs` | 765 | `apply_finality_fixpoint` | `pub(crate) fn apply_finality_fixpoint(nwa: &mut NWA) {` |
| `src/compiler/stages/resolve_negatives.rs` | 787 | `remove_negative_transitions` | `pub(crate) fn remove_negative_transitions(nwa: &mut NWA) {` |
| `src/compiler/stages/resolve_negatives.rs` | 793 | `has_live_final_weight` | `fn has_live_final_weight(state: &NWAState) -> bool {` |
| `src/compiler/stages/resolve_negatives.rs` | 797 | `has_non_default_transitions` | `fn has_non_default_transitions(state: &NWAState) -> bool {` |
| `src/compiler/stages/resolve_negatives.rs` | 804 | `is_terminal_shape_candidate` | `fn is_terminal_shape_candidate(state: &NWAState) -> bool {` |
| `src/compiler/stages/resolve_negatives.rs` | 810 | `default_targets_are_terminal_and_redundant` | `fn default_targets_are_terminal_and_redundant(state: &NWAState, terminal_states: &[bool]) -> bool {` |
| `src/compiler/stages/resolve_negatives.rs` | 826 | `grow_terminal_state_set` | `fn grow_terminal_state_set(nwa: &NWA, terminal_states: &mut [bool]) {` |
| `src/compiler/stages/resolve_negatives.rs` | 846 | `prune_terminal_default_targets` | `fn prune_terminal_default_targets(nwa: &mut NWA, terminal_states: &[bool]) {` |
| `src/compiler/stages/resolve_negatives.rs` | 864 | `remove_redundant_default_transitions` | `pub(crate) fn remove_redundant_default_transitions(nwa: &mut NWA) {` |
| `src/compiler/stages/resolve_negatives.rs` | 875 | `resolve_negative_codes_in_nwa` | `pub(crate) fn resolve_negative_codes_in_nwa(nwa: &mut NWA) {` |
| `src/config/env.rs` | 3 | `flag` | `pub(crate) fn flag(name: &str) -> bool { std::env::var_os(name).is_some() }` |
| `src/config/env.rs` | 4 | `truthy` | `pub(crate) fn truthy(name: &str) -> Option<bool> { let value=std::env::var(name).ok()?; match value.trim().to_ascii_lowercase().as_str() { "1"\|"true"\|"yes"\|"on"=>Some(true), "0"\|"false"\|"no"\|"off"=>Some(false), _=>None } }` |
| `src/config/env.rs` | 5 | `usize_var` | `pub(crate) fn usize_var(name: &str) -> Option<usize> { std::env::var(name).ok()?.trim().parse().ok() }` |
| `src/config/env.rs` | 6 | `u64_var` | `pub(crate) fn u64_var(name: &str) -> Option<u64> { std::env::var(name).ok()?.trim().parse().ok() }` |
| `src/config/env.rs` | 7 | `string_var` | `pub(crate) fn string_var(name: &str) -> Option<String> { std::env::var(name).ok() }` |
| `src/diagnostics/cache.rs` | 9 | `clear_stale_weights` | `pub fn clear_stale_weights() {` |
| `src/diagnostics/cache.rs` | 14 | `clear_weight_interners` | `pub fn clear_weight_interners() {` |
| `src/diagnostics/cache.rs` | 19 | `clear_weight_op_caches` | `pub fn clear_weight_op_caches() {` |
| `src/diagnostics/frontend.rs` | 13 | `compile_grammar_def_json` | `pub fn compile_grammar_def_json(grammar_def_json: &str, vocab: &Vocab) -> Result<Constraint> {` |
| `src/diagnostics/frontend.rs` | 22 | `prepare_vocab_for_compile` | `pub fn prepare_vocab_for_compile(vocab: &Vocab) {` |
| `src/diagnostics/frontend.rs` | 30 | `dump_json_schema_grammar_glrm` | `pub fn dump_json_schema_grammar_glrm(schema_json: &str) -> Result<String> {` |
| `src/diagnostics/logging.rs` | 3 | `emit_stderr` | `pub(crate) fn emit_stderr(message: impl AsRef<str>) { eprintln!("{}", message.as_ref()); }` |
| `src/ds/char_transitions.rs` | 7 | `CharTransitions` | `pub struct CharTransitions<T> {` |
| `src/ds/char_transitions.rs` | 13 | `entry_index` | `fn entry_index(&self, key: u8) -> Result<usize, usize> {` |
| `src/ds/char_transitions.rs` | 17 | `new` | `pub fn new() -> Self {` |
| `src/ds/char_transitions.rs` | 21 | `from_sorted_entries` | `pub fn from_sorted_entries(entries: Vec<(u8, T)>) -> Self {` |
| `src/ds/char_transitions.rs` | 25 | `len` | `pub fn len(&self) -> usize {` |
| `src/ds/char_transitions.rs` | 29 | `is_empty` | `pub fn is_empty(&self) -> bool {` |
| `src/ds/char_transitions.rs` | 33 | `clear` | `pub fn clear(&mut self) {` |
| `src/ds/char_transitions.rs` | 37 | `insert` | `pub fn insert(&mut self, key: u8, value: T) -> Option<T> {` |
| `src/ds/char_transitions.rs` | 47 | `get` | `pub fn get(&self, key: u8) -> Option<&T> {` |
| `src/ds/char_transitions.rs` | 53 | `get_mut` | `pub fn get_mut(&mut self, key: u8) -> Option<&mut T> {` |
| `src/ds/char_transitions.rs` | 59 | `contains_key` | `pub fn contains_key(&self, key: u8) -> bool {` |
| `src/ds/char_transitions.rs` | 63 | `iter` | `pub fn iter(&self) -> CharTransitionsIter<'_, T> {` |
| `src/ds/char_transitions.rs` | 69 | `iter_mut` | `pub fn iter_mut(&mut self) -> CharTransitionsIterMut<'_, T> {` |
| `src/ds/char_transitions.rs` | 75 | `values` | `pub fn values(&self) -> impl Iterator<Item = &T> {` |
| `src/ds/char_transitions.rs` | 81 | `Output` | `type Output = T;` |
| `src/ds/char_transitions.rs` | 83 | `index` | `fn index(&self, key: u8) -> &Self::Output {` |
| `src/ds/char_transitions.rs` | 89 | `index_mut` | `fn index_mut(&mut self, key: u8) -> &mut Self::Output {` |
| `src/ds/char_transitions.rs` | 94 | `CharTransitionsIter` | `pub struct CharTransitionsIter<'a, T> {` |
| `src/ds/char_transitions.rs` | 99 | `Item` | `type Item = (u8, &'a T);` |
| `src/ds/char_transitions.rs` | 101 | `next` | `fn next(&mut self) -> Option<Self::Item> {` |
| `src/ds/char_transitions.rs` | 106 | `CharTransitionsIterMut` | `pub struct CharTransitionsIterMut<'a, T> {` |
| `src/ds/char_transitions.rs` | 111 | `Item` | `type Item = (u8, &'a mut T);` |
| `src/ds/char_transitions.rs` | 113 | `next` | `fn next(&mut self) -> Option<Self::Item> {` |
| `src/ds/char_transitions.rs` | 119 | `Item` | `type Item = (u8, &'a T);` |
| `src/ds/char_transitions.rs` | 120 | `IntoIter` | `type IntoIter = CharTransitionsIter<'a, T>;` |
| `src/ds/char_transitions.rs` | 122 | `into_iter` | `fn into_iter(self) -> Self::IntoIter {` |
| `src/ds/char_transitions.rs` | 128 | `Item` | `type Item = (u8, &'a mut T);` |
| `src/ds/char_transitions.rs` | 129 | `IntoIter` | `type IntoIter = CharTransitionsIterMut<'a, T>;` |
| `src/ds/char_transitions.rs` | 131 | `into_iter` | `fn into_iter(self) -> Self::IntoIter {` |
| `src/ds/char_transitions.rs` | 137 | `extend` | `fn extend<I>(&mut self, iter: I)` |
| `src/ds/char_transitions.rs` | 148 | `from_iter` | `fn from_iter<I: IntoIterator<Item = (u8, T)>>(iter: I) -> Self {` |
| `src/ds/char_transitions.rs` | 156 | `fmt` | `fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {` |
| `src/ds/compressed_state_set.rs` | 4 | `hash_sparse_word` | `fn hash_sparse_word(word_index: usize, word: u64) -> u64 {` |
| `src/ds/compressed_state_set.rs` | 10 | `pop_lowest_state_bit` | `fn pop_lowest_state_bit(word: &mut u64, word_index: u32) -> Option<usize> {` |
| `src/ds/compressed_state_set.rs` | 20 | `SparseStateSet` | `pub struct SparseStateSet {` |
| `src/ds/compressed_state_set.rs` | 26 | `new` | `pub fn new(num_bits: usize) -> Self {` |
| `src/ds/compressed_state_set.rs` | 34 | `insert` | `pub fn insert(&mut self, bit: usize) -> bool {` |
| `src/ds/compressed_state_set.rs` | 49 | `insert_many` | `pub fn insert_many(&mut self, states: &[u32]) {` |
| `src/ds/compressed_state_set.rs` | 55 | `clear` | `pub fn clear(&mut self) {` |
| `src/ds/compressed_state_set.rs` | 64 | `CompressedStateSet` | `pub struct CompressedStateSet {` |
| `src/ds/compressed_state_set.rs` | 71 | `eq` | `fn eq(&self, other: &Self) -> bool {` |
| `src/ds/compressed_state_set.rs` | 77 | `hash` | `fn hash<H: Hasher>(&self, state: &mut H) {` |
| `src/ds/compressed_state_set.rs` | 83 | `new` | `pub fn new() -> Self {` |
| `src/ds/compressed_state_set.rs` | 91 | `from_sparse` | `pub fn from_sparse(sparse: &SparseStateSet) -> Self {` |
| `src/ds/compressed_state_set.rs` | 98 | `reuse_from_sparse` | `pub fn reuse_from_sparse(sparse: &SparseStateSet, buffer: &mut Self) {` |
| `src/ds/compressed_state_set.rs` | 121 | `iter` | `pub fn iter(&self) -> CompressedStateSetIter<'_> {` |
| `src/ds/compressed_state_set.rs` | 131 | `len` | `pub fn len(&self) -> usize {` |
| `src/ds/compressed_state_set.rs` | 139 | `CompressedStateSetIter` | `pub struct CompressedStateSetIter<'a> {` |
| `src/ds/compressed_state_set.rs` | 147 | `Item` | `type Item = usize;` |
| `src/ds/compressed_state_set.rs` | 149 | `next` | `fn next(&mut self) -> Option<Self::Item> {` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 16 | `ArcArrayVec` | `pub struct ArcArrayVec<T> {` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 24 | `new` | `pub fn new() -> Self {` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 33 | `as_slice` | `pub fn as_slice(&self) -> &[T] {` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 39 | `iter` | `pub fn iter(&self) -> std::slice::Iter<'_, T> {` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 45 | `len` | `pub fn len(&self) -> usize {` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 51 | `is_empty` | `pub fn is_empty(&self) -> bool {` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 57 | `last` | `pub fn last(&self) -> Option<&T> {` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 67 | `take` | `pub fn take(&self, n: usize) -> Self {` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 76 | `truncate` | `pub fn truncate(&mut self, new_len: usize) {` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 84 | `unit` | `pub fn unit(val: T) -> Self {` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 93 | `from_vec` | `pub fn from_vec(v: Vec<T>) -> Self {` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 102 | `append` | `pub fn append(&self, other: &Self) -> Self {` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 116 | `to_vec` | `pub fn to_vec(&self) -> Vec<T> {` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 122 | `try_push` | `pub fn try_push(&mut self, val: T) -> bool {` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 137 | `eq` | `fn eq(&self, other: &Self) -> bool {` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 148 | `hash` | `fn hash<H: Hasher>(&self, state: &mut H) {` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 154 | `default` | `fn default() -> Self {` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 165 | `unit` | `fn unit(val: T) -> Self { ArcArrayVec::unit(val) }` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 167 | `from_vec` | `fn from_vec(v: Vec<T>) -> Self { ArcArrayVec::from_vec(v) }` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 169 | `len` | `fn len(&self) -> usize { self.nw }` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 171 | `last` | `fn last(&self) -> Option<&T> { ArcArrayVec::last(self) }` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 173 | `take` | `fn take(&self, n: usize) -> Self { ArcArrayVec::take(self, n) }` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 175 | `truncate` | `fn truncate(&mut self, new_len: usize) { ArcArrayVec::truncate(self, new_len) }` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 177 | `try_push` | `fn try_push(&mut self, val: T) -> bool { ArcArrayVec::try_push(self, val) }` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 178 | `try_harder_push` | `fn try_harder_push(&mut self, val: T) -> bool {` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 189 | `append` | `fn append(&self, other: &Self) -> Self { ArcArrayVec::append(self, other) }` |
| `src/ds/stack_vecs/arc_array_vec.rs` | 190 | `to_vec` | `fn to_vec(&self) -> Vec<T> { ArcArrayVec::to_vec(self) }` |
| `src/ds/stack_vecs/dispatch.rs` | 10 | `Variant` | `enum Variant {` |
| `src/ds/stack_vecs/dispatch.rs` | 17 | `selected_variant` | `fn selected_variant() -> Variant {` |
| `src/ds/stack_vecs/dispatch.rs` | 39 | `DynStackVec` | `pub enum DynStackVec<T: Clone> {` |
| `src/ds/stack_vecs/dispatch.rs` | 45 | `unit` | `pub fn unit(val: T) -> Self {` |
| `src/ds/stack_vecs/dispatch.rs` | 51 | `from_vec` | `pub fn from_vec(v: Vec<T>) -> Self {` |
| `src/ds/stack_vecs/dispatch.rs` | 58 | `len` | `pub fn len(&self) -> usize {` |
| `src/ds/stack_vecs/dispatch.rs` | 63 | `is_empty` | `pub fn is_empty(&self) -> bool {` |
| `src/ds/stack_vecs/dispatch.rs` | 68 | `last` | `pub fn last(&self) -> Option<&T> {` |
| `src/ds/stack_vecs/dispatch.rs` | 73 | `take` | `pub fn take(&self, n: usize) -> Self {` |
| `src/ds/stack_vecs/dispatch.rs` | 78 | `truncate` | `pub fn truncate(&mut self, new_len: usize) {` |
| `src/ds/stack_vecs/dispatch.rs` | 83 | `try_push` | `pub fn try_push(&mut self, val: T) -> bool {` |
| `src/ds/stack_vecs/dispatch.rs` | 88 | `try_harder_push` | `pub fn try_harder_push(&mut self, val: T) -> bool {` |
| `src/ds/stack_vecs/dispatch.rs` | 92 | `append` | `pub fn append(&self, other: &Self) -> Self {` |
| `src/ds/stack_vecs/dispatch.rs` | 99 | `try_append` | `pub fn try_append(&self, other: &Self) -> Option<Self> {` |
| `src/ds/stack_vecs/dispatch.rs` | 107 | `capacity` | `pub fn capacity(&self) -> usize {` |
| `src/ds/stack_vecs/dispatch.rs` | 111 | `to_vec` | `pub fn to_vec(&self) -> Vec<T> {` |
| `src/ds/stack_vecs/dispatch.rs` | 116 | `iter` | `pub fn iter(&self) -> DynIter<'_, T> {` |
| `src/ds/stack_vecs/dispatch.rs` | 124 | `eq` | `fn eq(&self, other: &Self) -> bool {` |
| `src/ds/stack_vecs/dispatch.rs` | 135 | `hash` | `fn hash<H: Hasher>(&self, state: &mut H) {` |
| `src/ds/stack_vecs/dispatch.rs` | 142 | `fmt` | `fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {` |
| `src/ds/stack_vecs/dispatch.rs` | 150 | `default` | `fn default() -> Self {` |
| `src/ds/stack_vecs/dispatch.rs` | 168 | `DynIter` | `pub enum DynIter<'a, T> {` |
| `src/ds/stack_vecs/dispatch.rs` | 173 | `Item` | `type Item = &'a T;` |
| `src/ds/stack_vecs/dispatch.rs` | 175 | `next` | `fn next(&mut self) -> Option<&'a T> {` |
| `src/ds/stack_vecs/dispatch.rs` | 181 | `size_hint` | `fn size_hint(&self) -> (usize, Option<usize>) {` |
| `src/ds/stack_vecs/dispatch.rs` | 190 | `next_back` | `fn next_back(&mut self) -> Option<&'a T> {` |
| `src/ds/stack_vecs/stack_vec.rs` | 8 | `StackVec` | `pub trait StackVec<T>: Clone + PartialEq + Eq + Hash + Default` |
| `src/ds/stack_vecs/stack_vec.rs` | 13 | `unit` | `fn unit(val: T) -> Self;` |
| `src/ds/stack_vecs/stack_vec.rs` | 16 | `from_vec` | `fn from_vec(v: Vec<T>) -> Self;` |
| `src/ds/stack_vecs/stack_vec.rs` | 19 | `len` | `fn len(&self) -> usize;` |
| `src/ds/stack_vecs/stack_vec.rs` | 22 | `is_empty` | `fn is_empty(&self) -> bool {` |
| `src/ds/stack_vecs/stack_vec.rs` | 27 | `last` | `fn last(&self) -> Option<&T>;` |
| `src/ds/stack_vecs/stack_vec.rs` | 31 | `take` | `fn take(&self, n: usize) -> Self;` |
| `src/ds/stack_vecs/stack_vec.rs` | 34 | `truncate` | `fn truncate(&mut self, new_len: usize);` |
| `src/ds/stack_vecs/stack_vec.rs` | 40 | `try_push` | `fn try_push(&mut self, val: T) -> bool;` |
| `src/ds/stack_vecs/stack_vec.rs` | 46 | `try_harder_push` | `fn try_harder_push(&mut self, val: T) -> bool {` |
| `src/ds/stack_vecs/stack_vec.rs` | 51 | `append` | `fn append(&self, other: &Self) -> Self;` |
| `src/ds/stack_vecs/stack_vec.rs` | 55 | `try_append` | `fn try_append(&self, other: &Self) -> Option<Self> {` |
| `src/ds/stack_vecs/stack_vec.rs` | 60 | `capacity` | `fn capacity(&self) -> usize {` |
| `src/ds/stack_vecs/stack_vec.rs` | 65 | `to_vec` | `fn to_vec(&self) -> Vec<T>;` |
| `src/ds/stack_vecs/vec_stack_vec.rs` | 8 | `VecStackVec` | `pub struct VecStackVec<T>(Vec<T>);` |
| `src/ds/stack_vecs/vec_stack_vec.rs` | 13 | `iter` | `pub fn iter(&self) -> std::slice::Iter<'_, T> {` |
| `src/ds/stack_vecs/vec_stack_vec.rs` | 19 | `as_slice` | `pub fn as_slice(&self) -> &[T] {` |
| `src/ds/stack_vecs/vec_stack_vec.rs` | 25 | `eq` | `fn eq(&self, other: &Self) -> bool {` |
| `src/ds/stack_vecs/vec_stack_vec.rs` | 33 | `hash` | `fn hash<H: Hasher>(&self, state: &mut H) {` |
| `src/ds/stack_vecs/vec_stack_vec.rs` | 39 | `default` | `fn default() -> Self {` |
| `src/ds/stack_vecs/vec_stack_vec.rs` | 47 | `unit` | `fn unit(val: T) -> Self {` |
| `src/ds/stack_vecs/vec_stack_vec.rs` | 52 | `from_vec` | `fn from_vec(v: Vec<T>) -> Self {` |
| `src/ds/stack_vecs/vec_stack_vec.rs` | 57 | `len` | `fn len(&self) -> usize {` |
| `src/ds/stack_vecs/vec_stack_vec.rs` | 62 | `last` | `fn last(&self) -> Option<&T> {` |
| `src/ds/stack_vecs/vec_stack_vec.rs` | 66 | `take` | `fn take(&self, n: usize) -> Self {` |
| `src/ds/stack_vecs/vec_stack_vec.rs` | 72 | `truncate` | `fn truncate(&mut self, new_len: usize) {` |
| `src/ds/stack_vecs/vec_stack_vec.rs` | 77 | `try_push` | `fn try_push(&mut self, val: T) -> bool {` |
| `src/ds/stack_vecs/vec_stack_vec.rs` | 82 | `append` | `fn append(&self, other: &Self) -> Self {` |
| `src/ds/stack_vecs/vec_stack_vec.rs` | 90 | `to_vec` | `fn to_vec(&self) -> Vec<T> {` |
| `src/ds/vocab_prefix_tree.rs` | 6 | `VocabPrefixTreeNode` | `pub struct VocabPrefixTreeNode {` |
| `src/ds/vocab_prefix_tree.rs` | 21 | `VocabPrefixTreeChildIter` | `pub struct VocabPrefixTreeChildIter<'a> {` |
| `src/ds/vocab_prefix_tree.rs` | 27 | `Item` | `type Item = (&'a [u8], &'a VocabPrefixTreeNode);` |
| `src/ds/vocab_prefix_tree.rs` | 30 | `next` | `fn next(&mut self) -> Option<Self::Item> {` |
| `src/ds/vocab_prefix_tree.rs` | 36 | `size_hint` | `fn size_hint(&self) -> (usize, Option<usize>) {` |
| `src/ds/vocab_prefix_tree.rs` | 45 | `new` | `fn new(token_id: usize, prefix: Box<[u8]>, has_token: bool) -> Self {` |
| `src/ds/vocab_prefix_tree.rs` | 57 | `token_id` | `pub fn token_id(&self) -> usize {` |
| `src/ds/vocab_prefix_tree.rs` | 62 | `has_token` | `pub fn has_token(&self) -> bool {` |
| `src/ds/vocab_prefix_tree.rs` | 67 | `prefix_length` | `pub fn prefix_length(&self) -> usize {` |
| `src/ds/vocab_prefix_tree.rs` | 72 | `prefix` | `pub fn prefix(&self) -> &[u8] {` |
| `src/ds/vocab_prefix_tree.rs` | 77 | `children` | `pub fn children(&self) -> &[VocabPrefixTreeNode] {` |
| `src/ds/vocab_prefix_tree.rs` | 82 | `iter_children` | `pub fn iter_children(&self) -> VocabPrefixTreeChildIter<'_> {` |
| `src/ds/vocab_prefix_tree.rs` | 90 | `reachable_token_ids` | `pub fn reachable_token_ids(&self) -> &RangeSetBlaze<usize> {` |
| `src/ds/vocab_prefix_tree.rs` | 95 | `subtree_bytes` | `pub fn subtree_bytes(&self) -> &[u64; 4] {` |
| `src/ds/vocab_prefix_tree.rs` | 100 | `child_edge_label` | `fn child_edge_label<'a>(&'a self, child: &'a VocabPrefixTreeNode) -> &'a [u8] {` |
| `src/ds/vocab_prefix_tree.rs` | 105 | `child_key_byte` | `fn child_key_byte(&self, child: &VocabPrefixTreeNode) -> u8 {` |
| `src/ds/vocab_prefix_tree.rs` | 110 | `find_child` | `fn find_child(&self, next_byte: u8) -> Option<&VocabPrefixTreeNode> {` |
| `src/ds/vocab_prefix_tree.rs` | 119 | `insert_bytes_into_mask` | `fn insert_bytes_into_mask(mask: &mut [u64; 4], bytes: &[u8]) {` |
| `src/ds/vocab_prefix_tree.rs` | 125 | `merge_reachable_token_ids` | `fn merge_reachable_token_ids(` |
| `src/ds/vocab_prefix_tree.rs` | 132 | `merge_child_metadata` | `fn merge_child_metadata(` |
| `src/ds/vocab_prefix_tree.rs` | 145 | `next_matching_child` | `fn next_matching_child<'tree, 'bytes>(` |
| `src/ds/vocab_prefix_tree.rs` | 157 | `fmt` | `fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {` |
| `src/ds/vocab_prefix_tree.rs` | 158 | `format_bytes` | `fn format_bytes(bytes: &[u8]) -> String {` |
| `src/ds/vocab_prefix_tree.rs` | 189 | `VocabPrefixTree` | `pub struct VocabPrefixTree {` |
| `src/ds/vocab_prefix_tree.rs` | 197 | `new` | `pub fn new() -> Self {` |
| `src/ds/vocab_prefix_tree.rs` | 205 | `build` | `pub fn build(tokens: &[(usize, Vec<u8>)]) -> Self {` |
| `src/ds/vocab_prefix_tree.rs` | 209 | `sort_and_dedup_tokens` | `fn sort_and_dedup_tokens(tokens: &mut Vec<(usize, Vec<u8>)>) {` |
| `src/ds/vocab_prefix_tree.rs` | 226 | `build_children` | `fn build_children(` |
| `src/ds/vocab_prefix_tree.rs` | 244 | `build_owned` | `pub fn build_owned(mut tokens: Vec<(usize, Vec<u8>)>) -> Self {` |
| `src/ds/vocab_prefix_tree.rs` | 251 | `build_presorted` | `pub fn build_presorted(tokens: &[(usize, &[u8])]) -> Self {` |
| `src/ds/vocab_prefix_tree.rs` | 291 | `lcp_len` | `fn lcp_len(a: &[u8], b: &[u8], from: usize) -> usize {` |
| `src/ds/vocab_prefix_tree.rs` | 300 | `build_subtree` | `fn build_subtree(entries: &[(usize, &[u8])], parent_prefix_len: usize) -> VocabPrefixTreeNode {` |
| `src/ds/vocab_prefix_tree.rs` | 337 | `find_token` | `pub fn find_token(&self, bytes: &[u8]) -> Option<usize> {` |
| `src/ds/vocab_prefix_tree.rs` | 357 | `find_longest_prefix_token` | `pub fn find_longest_prefix_token<'s>(&'s self, bytes: &[u8]) -> Option<(usize, &'s [u8])> {` |
| `src/ds/vocab_prefix_tree.rs` | 384 | `has_empty_string_token` | `pub fn has_empty_string_token(&self) -> bool {` |
| `src/ds/vocab_prefix_tree.rs` | 389 | `root_children` | `pub fn root_children(&self) -> VocabPrefixTreeChildIter<'_> {` |
| `src/ds/vocab_prefix_tree.rs` | 394 | `max_token_id` | `pub fn max_token_id(&self) -> usize {` |
| `src/ds/vocab_prefix_tree.rs` | 400 | `default` | `fn default() -> Self {` |
| `src/error.rs` | 6 | `Error` | `pub enum Error {` |
| `src/error.rs` | 20 | `GlrMaskError` | `pub type GlrMaskError = Error;` |
| `src/error.rs` | 22 | `Result` | `pub type Result<T> = std::result::Result<T, Error>;` |
| `src/grammar_ir/ast.rs` | 21 | `GrammarExpr` | `pub enum GrammarExpr {` |
| `src/grammar_ir/ast.rs` | 78 | `CommaSepShape` | `pub enum CommaSepShape {` |
| `src/grammar_ir/ast.rs` | 90 | `NamedRule` | `pub struct NamedRule {` |
| `src/grammar_ir/ast.rs` | 101 | `NamedGrammar` | `pub struct NamedGrammar {` |
| `src/grammar_ir/ast.rs` | 111 | `terminal_names_set` | `pub fn terminal_names_set(&self) -> HashSet<String> {` |
| `src/grammar_ir/ast.rs` | 124 | `prune_unreachable` | `pub fn prune_unreachable(&self) -> Self {` |
| `src/grammar_ir/ast.rs` | 125 | `collect_refs` | `fn collect_refs(expr: &GrammarExpr, out: &mut HashSet<String>) {` |
| `src/grammar_ir/ast.rs` | 195 | `to_lark` | `pub fn to_lark(&self) -> String {` |
| `src/grammar_ir/expr_nfa.rs` | 18 | `ExprNFA` | `pub struct ExprNFA {` |
| `src/grammar_ir/expr_nfa.rs` | 24 | `new` | `pub fn new(nfa: NFA, symbols: Vec<GrammarExpr>) -> Self {` |
| `src/grammar_ir/expr_nfa.rs` | 28 | `into_determinized_and_minimized` | `pub fn into_determinized_and_minimized(self) -> Self {` |
| `src/grammar_ir/expr_nfa.rs` | 49 | `determinize` | `pub fn determinize(&self) -> DFA {` |
| `src/grammar_ir/expr_nfa.rs` | 53 | `determinize_and_minimize` | `pub fn determinize_and_minimize(&self) -> DFA {` |
| `src/grammar_ir/expr_nfa.rs` | 57 | `symbol_for_label` | `pub fn symbol_for_label(&self, label: Label) -> Option<&GrammarExpr> {` |
| `src/grammar_ir/expr_nfa.rs` | 68 | `ExprNfaBuilder` | `pub struct ExprNfaBuilder {` |
| `src/grammar_ir/expr_nfa.rs` | 75 | `default` | `fn default() -> Self {` |
| `src/grammar_ir/expr_nfa.rs` | 81 | `new` | `pub fn new() -> Self {` |
| `src/grammar_ir/expr_nfa.rs` | 89 | `add_state` | `pub fn add_state(&mut self) -> u32 {` |
| `src/grammar_ir/expr_nfa.rs` | 93 | `start_state` | `pub fn start_state(&self) -> u32 {` |
| `src/grammar_ir/expr_nfa.rs` | 97 | `add_start_state` | `pub fn add_start_state(&mut self, state: u32) {` |
| `src/grammar_ir/expr_nfa.rs` | 103 | `set_accepting` | `pub fn set_accepting(&mut self, state: u32) {` |
| `src/grammar_ir/expr_nfa.rs` | 107 | `add_epsilon` | `pub fn add_epsilon(&mut self, from: u32, to: u32) {` |
| `src/grammar_ir/expr_nfa.rs` | 111 | `add_symbol` | `pub fn add_symbol(&mut self, symbol: GrammarExpr) -> Label {` |
| `src/grammar_ir/expr_nfa.rs` | 122 | `add_transition` | `pub fn add_transition(&mut self, from: u32, symbol: GrammarExpr, to: u32) -> Label {` |
| `src/grammar_ir/expr_nfa.rs` | 128 | `add_labeled_transition` | `pub fn add_labeled_transition(&mut self, from: u32, label: Label, to: u32) {` |
| `src/grammar_ir/expr_nfa.rs` | 132 | `into_nfa_and_symbols` | `pub fn into_nfa_and_symbols(self) -> (NFA, Vec<GrammarExpr>) {` |
| `src/grammar_ir/expr_nfa.rs` | 136 | `build` | `pub fn build(self) -> ExprNFA {` |
| `src/grammar_ir/expr_nfa.rs` | 142 | `minimize_dfa` | `pub fn minimize_dfa(dfa: &DFA) -> DFA {` |
| `src/grammar_ir/expr_nfa.rs` | 150 | `subset_is_accepting` | `fn subset_is_accepting(nfa: &NFA, subset: &[u32]) -> bool {` |
| `src/grammar_ir/expr_nfa.rs` | 154 | `epsilon_closure` | `fn epsilon_closure(nfa: &NFA, seeds: &[u32]) -> BTreeSet<u32> {` |
| `src/grammar_ir/expr_nfa.rs` | 173 | `gather_label_targets` | `fn gather_label_targets(nfa: &NFA, subset: &[u32]) -> BTreeMap<Label, BTreeSet<u32>> {` |
| `src/grammar_ir/expr_nfa.rs` | 189 | `get_or_create_subset_state` | `fn get_or_create_subset_state(` |
| `src/grammar_ir/expr_nfa.rs` | 204 | `determinize_nfa` | `pub fn determinize_nfa(nfa: &NFA) -> DFA {` |
| `src/grammar_ir/expr_nfa.rs` | 253 | `lowers_expr_nfa_transition_symbols` | `fn lowers_expr_nfa_transition_symbols() {` |
| `src/grammar_ir/expr_nfa.rs` | 281 | `builder_preserves_nfa_and_exposes_determinize_minimize` | `fn builder_preserves_nfa_and_exposes_determinize_minimize() {` |
| `src/grammar_ir/flat.rs` | 12 | `GrammarDef` | `pub struct GrammarDef {` |
| `src/grammar_ir/flat.rs` | 24 | `NonterminalID` | `pub type NonterminalID = u32;` |
| `src/grammar_ir/flat.rs` | 26 | `TerminalID` | `pub type TerminalID = u32;` |
| `src/grammar_ir/flat.rs` | 29 | `Rule` | `pub struct Rule {` |
| `src/grammar_ir/flat.rs` | 35 | `Symbol` | `pub enum Symbol {` |
| `src/grammar_ir/flat.rs` | 41 | `fmt` | `fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {` |
| `src/grammar_ir/flat.rs` | 50 | `nonterminal_id` | `fn nonterminal_id(&self) -> Option<NonterminalID> {` |
| `src/grammar_ir/flat.rs` | 59 | `Terminal` | `pub enum Terminal {` |
| `src/grammar_ir/flat.rs` | 69 | `nonterminal_ids` | `fn nonterminal_ids(&self) -> impl Iterator<Item = NonterminalID> + '_ {` |
| `src/grammar_ir/flat.rs` | 76 | `id` | `pub fn id(&self) -> TerminalID {` |
| `src/grammar_ir/flat.rs` | 85 | `name` | `pub fn name(&self) -> String {` |
| `src/grammar_ir/flat.rs` | 95 | `num_terminals` | `pub fn num_terminals(&self) -> u32 {` |
| `src/grammar_ir/flat.rs` | 99 | `num_nonterminals` | `pub fn num_nonterminals(&self) -> u32 {` |
| `src/grammar_ir/flat.rs` | 108 | `terminal_display_name` | `pub fn terminal_display_name(&self, terminal: TerminalID) -> String {` |
| `src/grammar_ir/flat.rs` | 116 | `terminal_by_id` | `fn terminal_by_id(&self, terminal: TerminalID) -> Option<&Terminal> {` |
| `src/grammar_ir/glrm/mod.rs` | 16 | `from_glrm` | `pub fn from_glrm(input: &str) -> Result<NamedGrammar, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 25 | `Tok` | `enum Tok {` |
| `src/grammar_ir/glrm/mod.rs` | 76 | `Lexer` | `struct Lexer<'a> {` |
| `src/grammar_ir/glrm/mod.rs` | 82 | `new` | `fn new(src: &'a str) -> Self {` |
| `src/grammar_ir/glrm/mod.rs` | 86 | `peek` | `fn peek(&self) -> Option<u8> {` |
| `src/grammar_ir/glrm/mod.rs` | 90 | `peek2` | `fn peek2(&self) -> Option<u8> {` |
| `src/grammar_ir/glrm/mod.rs` | 94 | `advance` | `fn advance(&mut self) -> Option<u8> {` |
| `src/grammar_ir/glrm/mod.rs` | 100 | `skip_whitespace_and_comments` | `fn skip_whitespace_and_comments(&mut self) {` |
| `src/grammar_ir/glrm/mod.rs` | 132 | `lex_string` | `fn lex_string(&mut self, delim: u8) -> Result<Vec<u8>, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 162 | `lex_regex` | `fn lex_regex(&mut self) -> Result<String, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 183 | `lex_char_class` | `fn lex_char_class(&mut self) -> Result<(String, bool), GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 227 | `lex_ident` | `fn lex_ident(&mut self, first: u8) -> String {` |
| `src/grammar_ir/glrm/mod.rs` | 237 | `lex_int` | `fn lex_int(&mut self, first: u8) -> Result<usize, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 246 | `tokenize` | `fn tokenize(mut self) -> Result<Vec<Tok>, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 318 | `GlrmParser` | `struct GlrmParser {` |
| `src/grammar_ir/glrm/mod.rs` | 325 | `peek` | `fn peek(&self) -> &Tok {` |
| `src/grammar_ir/glrm/mod.rs` | 329 | `advance` | `fn advance(&mut self) -> &Tok {` |
| `src/grammar_ir/glrm/mod.rs` | 335 | `consume` | `fn consume(&mut self, expected: &Tok) -> Result<(), GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 344 | `expect_ident` | `fn expect_ident(&mut self) -> Result<String, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 351 | `expect_int` | `fn expect_int(&mut self) -> Result<usize, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 358 | `parse_grammar` | `fn parse_grammar(&mut self) -> Result<NamedGrammar, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 410 | `parse_rule` | `fn parse_rule(&mut self, is_terminal: bool, is_internal: bool) -> Result<NamedRule, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 419 | `parse_expr_nfa_rule` | `fn parse_expr_nfa_rule(&mut self) -> Result<NamedRule, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 513 | `parse_expr_nfa_transition_expr` | `fn parse_expr_nfa_transition_expr(&mut self) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 530 | `parse_nt_expr` | `fn parse_nt_expr(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 553 | `parse_nt_exclude` | `fn parse_nt_exclude(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 575 | `parse_nt_intersect` | `fn parse_nt_intersect(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 588 | `parse_nt_exclude_rhs` | `fn parse_nt_exclude_rhs(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 611 | `parse_nt_seq` | `fn parse_nt_seq(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 628 | `can_start_nt_atom` | `fn can_start_nt_atom(&self) -> bool {` |
| `src/grammar_ir/glrm/mod.rs` | 637 | `parse_nt_postfix` | `fn parse_nt_postfix(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 680 | `parse_sepseq_item` | `fn parse_sepseq_item(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 732 | `apply_nt_quantifier` | `fn apply_nt_quantifier(&mut self, atom: GrammarExpr) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 763 | `parse_nt_atom` | `fn parse_nt_atom(&mut self, allow_raw_regex: bool) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 815 | `hex_digit` | `fn hex_digit(b: u8) -> Result<u8, GlrMaskError> {` |
| `src/grammar_ir/glrm/mod.rs` | 824 | `ensure_nfa_state` | `fn ensure_nfa_state(nfa: &mut NFA, state: u32) {` |
| `src/grammar_ir/glrm/mod.rs` | 830 | `intern_expr_nfa_symbol` | `fn intern_expr_nfa_symbol(` |
| `src/grammar_ir/glrm/mod.rs` | 844 | `err` | `fn err(msg: &str) -> GlrMaskError {` |
| `src/grammar_ir/glrm/tests.rs` | 7 | `single_path_terminal_names` | `fn single_path_terminal_names(` |
| `src/grammar_ir/glrm/tests.rs` | 30 | `parses_named_expr_nfa_definition` | `fn parses_named_expr_nfa_definition() {` |
| `src/grammar_ir/glrm/tests.rs` | 56 | `dumps_expr_nfa_as_own_definition` | `fn dumps_expr_nfa_as_own_definition() {` |
| `src/grammar_ir/glrm/tests.rs` | 77 | `expr_nfa_transition_symbols_accept_full_expressions` | `fn expr_nfa_transition_symbols_accept_full_expressions() {` |
| `src/grammar_ir/glrm/tests.rs` | 99 | `exclude_rhs_sequence_requires_parentheses` | `fn exclude_rhs_sequence_requires_parentheses() {` |
| `src/grammar_ir/glrm/tests.rs` | 113 | `grouped_exclude_rhs_preserves_parenthesized_ref` | `fn grouped_exclude_rhs_preserves_parenthesized_ref() {` |
| `src/grammar_ir/glrm/tests.rs` | 140 | `lowering_subtracts_exact_nonterminal_alternatives` | `fn lowering_subtracts_exact_nonterminal_alternatives() {` |
| `src/grammar_ir/glrm/tests.rs` | 179 | `lowering_rejects_parenthesized_ref_without_exact_alternative` | `fn lowering_rejects_parenthesized_ref_without_exact_alternative() {` |
| `src/grammar_ir/glrm/tests.rs` | 195 | `rejects_nested_expr_nfa_at_lowering` | `fn rejects_nested_expr_nfa_at_lowering() {` |
| `src/grammar_ir/lower/exact_subtraction.rs` | 15 | `exact_subtraction_alternatives` | `fn exact_subtraction_alternatives(` |
| `src/grammar_ir/lower/exact_subtraction.rs` | 46 | `canonical_exact_expr` | `fn canonical_exact_expr(&self, expr: &GrammarExpr) -> GrammarExpr {` |
| `src/grammar_ir/lower/exact_subtraction.rs` | 52 | `canonical_exact_expr_inner` | `fn canonical_exact_expr_inner(` |
| `src/grammar_ir/lower/exact_subtraction.rs` | 146 | `exact_nonterminal_subtraction_expr` | `pub(super) fn exact_nonterminal_subtraction_expr(` |
| `src/grammar_ir/lower/expr_nfa_lower.rs` | 13 | `expr_nfa_state_nonterminals` | `pub(super) fn expr_nfa_state_nonterminals(` |
| `src/grammar_ir/lower/expr_nfa_lower.rs` | 39 | `emit_expr_dfa_leftlinear` | `pub(super) fn emit_expr_dfa_leftlinear(` |
| `src/grammar_ir/lower/expr_nfa_lower.rs` | 87 | `emit_expr_dfa_leftlinear_nonnullable` | `pub(super) fn emit_expr_dfa_leftlinear_nonnullable(` |
| `src/grammar_ir/lower/expr_nfa_lower.rs` | 169 | `emit_expr_nfa` | `pub(super) fn emit_expr_nfa(&mut self, lhs: NonterminalID, expr_nfa: &ExprNFA) -> Result<(), GlrMaskError> {` |
| `src/grammar_ir/lower/expr_nfa_lower.rs` | 174 | `emit_expr_nfa_nonnullable` | `pub(super) fn emit_expr_nfa_nonnullable(` |
| `src/grammar_ir/lower/mod.rs` | 43 | `char_class_pattern` | `fn char_class_pattern(def: &str, negate: bool) -> String {` |
| `src/grammar_ir/lower/mod.rs` | 51 | `Lowerer` | `pub(super) struct Lowerer {` |
| `src/grammar_ir/lower/mod.rs` | 79 | `new` | `fn new() -> Self {` |
| `src/grammar_ir/lower/mod.rs` | 101 | `nonterminal_id` | `fn nonterminal_id(&mut self, name: &str) -> NonterminalID {` |
| `src/grammar_ir/lower/mod.rs` | 111 | `fresh_nonterminal` | `fn fresh_nonterminal(&mut self, hint: &str) -> (String, NonterminalID) {` |
| `src/grammar_ir/lower/mod.rs` | 118 | `expr_is_nullable` | `fn expr_is_nullable(&self, expr: &GrammarExpr) -> bool {` |
| `src/grammar_ir/lower/mod.rs` | 122 | `strip_grouping` | `fn strip_grouping(expr: &GrammarExpr) -> &GrammarExpr {` |
| `src/grammar_ir/lower/mod.rs` | 129 | `top_level_alternatives` | `fn top_level_alternatives(expr: &GrammarExpr) -> Vec<GrammarExpr> {` |
| `src/grammar_ir/lower/mod.rs` | 139 | `resolve_terminal_expr` | `fn resolve_terminal_expr(` |
| `src/grammar_ir/lower/mod.rs` | 156 | `nonnullable_terminal_symbol` | `fn nonnullable_terminal_symbol(` |
| `src/grammar_ir/lower/mod.rs` | 193 | `lower_nonnullable_named_rule` | `fn lower_nonnullable_named_rule(&mut self, name: &str) -> Result<Symbol, GlrMaskError> {` |
| `src/grammar_ir/lower/mod.rs` | 243 | `lower_nonnullable_expr_symbol` | `fn lower_nonnullable_expr_symbol(` |
| `src/grammar_ir/lower/mod.rs` | 272 | `emit_nonnullable_sequence` | `fn emit_nonnullable_sequence(` |
| `src/grammar_ir/lower/mod.rs` | 304 | `emit_nonnullable_expr` | `fn emit_nonnullable_expr(` |
| `src/grammar_ir/lower/mod.rs` | 397 | `terminal_id` | `fn terminal_id(&mut self, name: &str, pattern: &str, utf8: bool) -> TerminalID {` |
| `src/grammar_ir/lower/mod.rs` | 423 | `lower_expr` | `fn lower_expr(&mut self, expr: &GrammarExpr) -> Symbol {` |
| `src/grammar_ir/lower/mod.rs` | 424 | `emit` | `fn emit(lowerer: &mut Lowerer, lhs: NonterminalID, expr: &GrammarExpr) -> Result<(), GlrMaskError> {` |
| `src/grammar_ir/lower/mod.rs` | 501 | `lower_expr_terminalish` | `fn lower_expr_terminalish(&mut self, expr: &GrammarExpr) -> Result<Symbol, GlrMaskError> {` |
| `src/grammar_ir/lower/mod.rs` | 568 | `register_terminal_expr` | `fn register_terminal_expr(&mut self, name: &str, expr: Expr) -> TerminalID {` |
| `src/grammar_ir/lower/mod.rs` | 583 | `lower` | `pub fn lower(grammar: &NamedGrammar) -> Result<GrammarDef, GlrMaskError> {` |
| `src/grammar_ir/lower/repeat.rs` | 11 | `RepeatTreeShape` | `pub(super) enum RepeatTreeShape {` |
| `src/grammar_ir/lower/repeat.rs` | 22 | `repeat_tree_shape` | `pub(super) fn repeat_tree_shape() -> RepeatTreeShape {` |
| `src/grammar_ir/lower/repeat.rs` | 29 | `repeat_tree_shape_from_value` | `pub(super) fn repeat_tree_shape_from_value(value: &str) -> RepeatTreeShape {` |
| `src/grammar_ir/lower/repeat.rs` | 39 | `right_repeat_range_front_bucket` | `pub(super) fn right_repeat_range_front_bucket() -> usize {` |
| `src/grammar_ir/lower/repeat.rs` | 46 | `left_repeat_range_back_bucket` | `pub(super) fn left_repeat_range_back_bucket() -> usize {` |
| `src/grammar_ir/lower/repeat.rs` | 53 | `exact_repeat_split` | `pub(super) fn exact_repeat_split(count: usize, shape: RepeatTreeShape) -> (usize, usize) {` |
| `src/grammar_ir/lower/repeat.rs` | 65 | `range_repeat_split` | `pub(super) fn range_repeat_split(min: usize, max: usize, shape: RepeatTreeShape) -> (usize, usize) {` |
| `src/grammar_ir/lower/repeat.rs` | 79 | `highest_power_of_two_le` | `pub(super) fn highest_power_of_two_le(n: usize) -> usize {` |
| `src/grammar_ir/lower/repeat.rs` | 82 | `repeat_exact_nonterminal` | `pub(super) fn repeat_exact_nonterminal(` |
| `src/grammar_ir/lower/repeat.rs` | 122 | `repeat_max_nonterminal` | `pub(super) fn repeat_max_nonterminal(` |
| `src/grammar_ir/lower/repeat.rs` | 204 | `repeat_min1_max_nonterminal` | `pub(super) fn repeat_min1_max_nonterminal(&mut self, symbol: &Symbol, max: usize) -> NonterminalID {` |
| `src/grammar_ir/lower/repeat.rs` | 230 | `repeat_range_nonterminal` | `pub(super) fn repeat_range_nonterminal(` |
| `src/grammar_ir/lower/repeat.rs` | 336 | `repeat_range_nonterminal_countdown` | `pub(super) fn repeat_range_nonterminal_countdown(` |
| `src/grammar_ir/lower/repeat.rs` | 370 | `repeat_range_nonterminal_balanced` | `pub(super) fn repeat_range_nonterminal_balanced(` |
| `src/grammar_ir/lower/repeat.rs` | 400 | `emit_repeat_range` | `pub(super) fn emit_repeat_range(` |
| `src/grammar_ir/lower/separated_sequence.rs` | 14 | `comma_sep_shape` | `pub(crate) fn comma_sep_shape() -> CommaSepShape {` |
| `src/grammar_ir/lower/separated_sequence.rs` | 32 | `lower_separated_sequence_repetition_item_nonempty_symbol` | `pub(super) fn lower_separated_sequence_repetition_item_nonempty_symbol(` |
| `src/grammar_ir/lower/separated_sequence.rs` | 100 | `lower_separated_sequence_item_nonempty_symbol` | `pub(super) fn lower_separated_sequence_item_nonempty_symbol(` |
| `src/grammar_ir/lower/separated_sequence.rs` | 127 | `lower_separated_sequence_inner` | `pub(super) fn lower_separated_sequence_inner(` |
| `src/grammar_ir/lower/terminal_expr.rs` | 22 | `grammar_expr_to_expr` | `pub(super) fn grammar_expr_to_expr(` |
| `src/grammar_ir/lower/terminal_expr.rs` | 121 | `grammar_expr_is_nullable` | `pub(super) fn grammar_expr_is_nullable(` |
| `src/grammar_ir/lower/terminal_expr.rs` | 198 | `compute_rule_nullability` | `pub(super) fn compute_rule_nullability(grammar: &NamedGrammar) -> HashMap<String, bool> {` |
| `src/grammar_ir/lower/terminal_expr.rs` | 222 | `validate_expr_nfa_placement` | `pub(super) fn validate_expr_nfa_placement(grammar: &NamedGrammar) -> Result<(), GlrMaskError> {` |
| `src/grammar_ir/lower/terminal_expr.rs` | 223 | `walk` | `fn walk(expr: &GrammarExpr, top_level: bool, rule_name: &str) -> Result<(), GlrMaskError> {` |
| `src/grammar_ir/lower/terminal_expr.rs` | 288 | `expr_to_grammar_expr` | `pub fn expr_to_grammar_expr(expr: &Expr) -> GrammarExpr {` |
| `src/grammar_ir/lower/terminal_expr.rs` | 343 | `dedup_rules_preserving_first_occurrence` | `pub(super) fn dedup_rules_preserving_first_occurrence(rules: &mut Vec<Rule>) {` |
| `src/grammar_ir/lower/tests.rs` | 6 | `nonterminal` | `fn nonterminal(name: &str, expr: GrammarExpr) -> NamedRule {` |
| `src/grammar_ir/lower/tests.rs` | 15 | `terminal` | `fn terminal(name: &str, expr: GrammarExpr) -> NamedRule {` |
| `src/grammar_ir/lower/tests.rs` | 24 | `literal` | `fn literal(text: &str) -> GrammarExpr {` |
| `src/grammar_ir/lower/tests.rs` | 28 | `subtract` | `fn subtract(lhs: &str, exclude: GrammarExpr) -> GrammarExpr {` |
| `src/grammar_ir/lower/tests.rs` | 36 | `exact_subtraction_matches_nonterminal_alias_body` | `fn exact_subtraction_matches_nonterminal_alias_body() {` |
| `src/grammar_ir/lower/tests.rs` | 67 | `exact_subtraction_canonicalization_is_cycle_safe` | `fn exact_subtraction_canonicalization_is_cycle_safe() {` |
| `src/grammar_ir/lower/tests.rs` | 95 | `lower_deduplicates_identical_rules` | `fn lower_deduplicates_identical_rules() {` |
| `src/grammar_ir/lower/tests.rs` | 115 | `nonnullable_sequence_with_nonnullable_part_reduces_rules` | `fn nonnullable_sequence_with_nonnullable_part_reduces_rules() {` |
| `src/grammar_ir/render/glrm.rs` | 10 | `to_glrm` | `pub fn to_glrm(grammar: &NamedGrammar) -> String {` |
| `src/grammar_ir/render/glrm.rs` | 39 | `dump_expr_nfa` | `fn dump_expr_nfa(expr_nfa: &ExprNFA) -> String {` |
| `src/grammar_ir/render/glrm.rs` | 79 | `dump_nt_expr` | `fn dump_nt_expr(expr: &GrammarExpr, needs_parens: bool) -> String {` |
| `src/grammar_ir/render/glrm.rs` | 126 | `dump_nt_seq` | `fn dump_nt_seq(expr: &GrammarExpr) -> String {` |
| `src/grammar_ir/render/glrm.rs` | 138 | `dump_nt_postfix` | `fn dump_nt_postfix(expr: &GrammarExpr) -> String {` |
| `src/grammar_ir/render/glrm.rs` | 154 | `dump_nt_atom` | `fn dump_nt_atom(expr: &GrammarExpr) -> String {` |
| `src/grammar_ir/render/glrm.rs` | 223 | `dump_set_operand` | `fn dump_set_operand(expr: &GrammarExpr) -> String {` |
| `src/grammar_ir/render/glrm.rs` | 234 | `escape_bytes_for_string` | `fn escape_bytes_for_string(bytes: &[u8]) -> String {` |
| `src/grammar_ir/render/glrm.rs` | 250 | `escape_regex_for_slash` | `fn escape_regex_for_slash(pat: &str) -> String {` |
| `src/grammar_ir/render/lark.rs` | 11 | `to_lark` | `pub fn to_lark(grammar: &NamedGrammar) -> String {` |
| `src/grammar_ir/render/lark.rs` | 50 | `grammar_expr_to_lark` | `pub(crate) fn grammar_expr_to_lark(expr: &GrammarExpr, out: &mut String, parens: bool) {` |
| `src/grammar_ir/render/lark.rs` | 54 | `grammar_expr_to_lark_with_indent` | `fn grammar_expr_to_lark_with_indent(` |
| `src/grammar_ir/render/lark.rs` | 201 | `u8set_to_class_def` | `pub(crate) fn u8set_to_class_def(set: &U8Set) -> String {` |
| `src/grammar_ir/render/lark.rs` | 226 | `push_class_char` | `fn push_class_char(out: &mut String, b: u8) {` |
| `src/grammar_ir/render/lark.rs` | 238 | `escape_byte` | `pub(crate) fn escape_byte(b: u8) -> String {` |
| `src/grammar_ir/render/lark.rs` | 250 | `regex_escape_byte` | `pub(crate) fn regex_escape_byte(b: u8) -> String {` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 14 | `ExactSubtractionLoweringStats` | `pub struct ExactSubtractionLoweringStats {` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 22 | `lower_exact_subtractions` | `pub fn lower_exact_subtractions(` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 89 | `ResolvedSubtraction` | `struct ResolvedSubtraction {` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 94 | `ExactSubtractionResolver` | `struct ExactSubtractionResolver<'a> {` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 100 | `resolve_site` | `fn resolve_site(&self, expr: &GrammarExpr) -> Result<Option<ResolvedSubtraction>> {` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 145 | `exact_subtraction_alternatives` | `fn exact_subtraction_alternatives(` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 176 | `canonical_exact_expr` | `fn canonical_exact_expr(&self, expr: &GrammarExpr) -> GrammarExpr {` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 182 | `canonical_exact_expr_inner` | `fn canonical_exact_expr_inner(` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 280 | `SiteCollector` | `struct SiteCollector {` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 286 | `collect_expr` | `fn collect_expr(&mut self, expr: &GrammarExpr, resolver: &ExactSubtractionResolver<'_>) -> Result<()> {` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 340 | `LhsCollection` | `struct LhsCollection {` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 346 | `add_removal_set` | `fn add_removal_set(&mut self, removed_indices: Vec<usize>) {` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 353 | `GeneratedHelpers` | `struct GeneratedHelpers {` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 361 | `build_helpers_for_lhs` | `fn build_helpers_for_lhs(` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 449 | `Partition` | `struct Partition {` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 453 | `SegmentTreeBuilder` | `struct SegmentTreeBuilder {` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 461 | `node_ref` | `fn node_ref(` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 487 | `cover_included_partitions` | `fn cover_included_partitions(` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 518 | `collect_cover_refs` | `fn collect_cover_refs(` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 544 | `rewrite_expr` | `fn rewrite_expr(` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 608 | `strip_grouping` | `fn strip_grouping(expr: &GrammarExpr) -> &GrammarExpr {` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 615 | `top_level_alternatives` | `fn top_level_alternatives(expr: &GrammarExpr) -> Vec<GrammarExpr> {` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 625 | `named_helper_rule` | `fn named_helper_rule(name: String, expr: GrammarExpr) -> NamedRule {` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 634 | `choice_or_single` | `fn choice_or_single(mut options: Vec<GrammarExpr>) -> GrammarExpr {` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 642 | `sanitize_name_component` | `fn sanitize_name_component(name: &str) -> String {` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 654 | `NameAllocator` | `struct NameAllocator {` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 660 | `new` | `fn new<I>(existing_names: I) -> Self` |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 671 | `alloc` | `fn alloc(&mut self, prefix: &str) -> String {` |
| `src/grammar_ir/transforms/exact_subtraction/tests.rs` | 12 | `EnvVarGuard` | `struct EnvVarGuard {` |
| `src/grammar_ir/transforms/exact_subtraction/tests.rs` | 18 | `set` | `fn set(key: &'static str, value: &str) -> Self {` |
| `src/grammar_ir/transforms/exact_subtraction/tests.rs` | 26 | `unset` | `fn unset(key: &'static str) -> Self {` |
| `src/grammar_ir/transforms/exact_subtraction/tests.rs` | 36 | `drop` | `fn drop(&mut self) {` |
| `src/grammar_ir/transforms/exact_subtraction/tests.rs` | 48 | `nonterminal` | `fn nonterminal(name: &str, expr: GrammarExpr) -> NamedRule {` |
| `src/grammar_ir/transforms/exact_subtraction/tests.rs` | 57 | `terminal` | `fn terminal(name: &str, expr: GrammarExpr) -> NamedRule {` |
| `src/grammar_ir/transforms/exact_subtraction/tests.rs` | 66 | `literal` | `fn literal(text: &str) -> GrammarExpr {` |
| `src/grammar_ir/transforms/exact_subtraction/tests.rs` | 70 | `subtract` | `fn subtract(lhs: &str, exclude: GrammarExpr) -> GrammarExpr {` |
| `src/grammar_ir/transforms/exact_subtraction/tests.rs` | 77 | `find_rule` | `fn find_rule<'a>(grammar: &'a NamedGrammar, name: &str) -> &'a NamedRule {` |
| `src/grammar_ir/transforms/exact_subtraction/tests.rs` | 85 | `contains_exclude` | `fn contains_exclude(expr: &GrammarExpr) -> bool {` |
| `src/grammar_ir/transforms/exact_subtraction/tests.rs` | 114 | `exact_subtraction_rewrites_sites_into_shared_helpers` | `fn exact_subtraction_rewrites_sites_into_shared_helpers() {` |
| `src/grammar_ir/transforms/exact_subtraction/tests.rs` | 167 | `exact_subtraction_partitions_alternatives_by_shared_signature` | `fn exact_subtraction_partitions_alternatives_by_shared_signature() {` |
| `src/grammar_ir/transforms/exact_subtraction/tests.rs` | 214 | `exact_subtraction_errors_on_missing_exact_alternative` | `fn exact_subtraction_errors_on_missing_exact_alternative() {` |
| `src/grammar_ir/transforms/exact_subtraction/tests.rs` | 232 | `exact_subtraction_matches_nonterminal_alias_body` | `fn exact_subtraction_matches_nonterminal_alias_body() {` |
| `src/grammar_ir/transforms/exact_subtraction/tests.rs` | 267 | `exact_subtraction_canonicalization_is_cycle_safe` | `fn exact_subtraction_canonicalization_is_cycle_safe() {` |
| `src/grammar_ir/transforms/exact_subtraction/tests.rs` | 295 | `exact_subtraction_json_schema_dump_uses_helpers_when_enabled` | `fn exact_subtraction_json_schema_dump_uses_helpers_when_enabled() {` |
| `src/grammar_ir/transforms/exact_subtraction/tests.rs` | 330 | `exact_subtraction_json_schema_dump_keeps_direct_subtraction_when_disabled` | `fn exact_subtraction_json_schema_dump_keeps_direct_subtraction_when_disabled() {` |
| `src/grammar_ir/transforms/factor.rs` | 16 | `contains_regex_features` | `fn contains_regex_features(expr: &GrammarExpr) -> bool {` |
| `src/grammar_ir/transforms/factor.rs` | 45 | `epsilon_expr` | `fn epsilon_expr() -> GrammarExpr {` |
| `src/grammar_ir/transforms/factor.rs` | 49 | `colon_literal` | `fn colon_literal() -> GrammarExpr {` |
| `src/grammar_ir/transforms/factor.rs` | 53 | `factor_named_grammar` | `pub fn factor_named_grammar(grammar: NamedGrammar) -> NamedGrammar {` |
| `src/grammar_ir/transforms/factor.rs` | 63 | `ChoiceFactorer` | `struct ChoiceFactorer {` |
| `src/grammar_ir/transforms/factor.rs` | 73 | `new` | `fn new(rules: Vec<NamedRule>, terminals: &HashSet<String>) -> Self {` |
| `src/grammar_ir/transforms/factor.rs` | 110 | `factor_all` | `fn factor_all(mut self) -> Vec<NamedRule> {` |
| `src/grammar_ir/transforms/factor.rs` | 135 | `should_factor_rule` | `fn should_factor_rule(&self, name: &str, expr: &GrammarExpr) -> bool {` |
| `src/grammar_ir/transforms/factor.rs` | 141 | `factor_expr` | `fn factor_expr(&mut self, expr: GrammarExpr, rule_name: &str) -> GrammarExpr {` |
| `src/grammar_ir/transforms/factor.rs` | 174 | `factor_choice` | `fn factor_choice(&mut self, alternatives: Vec<GrammarExpr>, rule_name: &str) -> GrammarExpr {` |
| `src/grammar_ir/transforms/factor.rs` | 234 | `is_safe_alternative` | `fn is_safe_alternative(&self, expr: &GrammarExpr) -> bool {` |
| `src/grammar_ir/transforms/factor.rs` | 239 | `collect_refs` | `fn collect_refs(expr: &GrammarExpr) -> HashSet<String> {` |
| `src/grammar_ir/transforms/factor.rs` | 245 | `collect_refs_impl` | `fn collect_refs_impl(expr: &GrammarExpr, refs: &mut HashSet<String>) {` |
| `src/grammar_ir/transforms/factor.rs` | 290 | `group_by_tail` | `fn group_by_tail(&self, alternatives: &[GrammarExpr]) -> HashMap<GrammarExpr, Vec<GrammarExpr>> {` |
| `src/grammar_ir/transforms/factor.rs` | 304 | `has_tail_pattern` | `fn has_tail_pattern(expr: &GrammarExpr) -> bool {` |
| `src/grammar_ir/transforms/factor.rs` | 308 | `extract_tail_pattern` | `fn extract_tail_pattern(expr: &GrammarExpr) -> Option<(GrammarExpr, GrammarExpr)> {` |
| `src/grammar_ir/transforms/factor.rs` | 331 | `is_complex_head` | `fn is_complex_head(expr: &GrammarExpr) -> bool {` |
| `src/grammar_ir/transforms/factor.rs` | 345 | `create_helper_rule` | `fn create_helper_rule(&mut self, alternatives: Vec<GrammarExpr>, base_name: String) -> String {` |
| `src/grammar_ir/transforms/factor.rs` | 377 | `find_recursive_rules` | `fn find_recursive_rules(rules: &HashMap<String, GrammarExpr>) -> HashSet<String> {` |
| `src/grammar_ir/transforms/factor.rs` | 396 | `collect_refs_static` | `fn collect_refs_static(expr: &GrammarExpr, refs: &mut HashSet<String>) {` |
| `src/grammar_ir/transforms/factor.rs` | 441 | `can_reach_self` | `fn can_reach_self(start: &str, deps: &HashMap<String, Vec<String>>) -> bool {` |
| `src/grammar_ir/transforms/simplify.rs` | 12 | `simplify_named_grammar` | `pub fn simplify_named_grammar(grammar: &mut NamedGrammar) -> SimplifyStats {` |
| `src/grammar_ir/transforms/simplify.rs` | 34 | `SimplifyStats` | `pub struct SimplifyStats {` |
| `src/grammar_ir/transforms/simplify.rs` | 44 | `simplify_expr` | `fn simplify_expr(expr: GrammarExpr, stats: &mut SimplifyStats) -> GrammarExpr {` |
| `src/grammar_ir/transforms/simplify.rs` | 84 | `simplify_sequence` | `fn simplify_sequence(parts: Vec<GrammarExpr>, stats: &mut SimplifyStats) -> GrammarExpr {` |
| `src/grammar_ir/transforms/simplify.rs` | 109 | `simplify_choice` | `fn simplify_choice(options: Vec<GrammarExpr>, stats: &mut SimplifyStats) -> GrammarExpr {` |
| `src/grammar_ir/transforms/simplify.rs` | 131 | `simplify_optional` | `fn simplify_optional(inner: GrammarExpr, stats: &mut SimplifyStats) -> GrammarExpr {` |
| `src/grammar_ir/transforms/simplify.rs` | 149 | `simplify_repeat` | `fn simplify_repeat(inner: GrammarExpr, stats: &mut SimplifyStats) -> GrammarExpr {` |
| `src/grammar_ir/transforms/simplify.rs` | 163 | `simplify_repeat_one` | `fn simplify_repeat_one(inner: GrammarExpr, stats: &mut SimplifyStats) -> GrammarExpr {` |
| `src/grammar_ir/transforms/simplify.rs` | 181 | `simplify_repeat_range` | `fn simplify_repeat_range(` |
| `src/grammar_ir/transforms/simplify.rs` | 209 | `inline_single_use_sequence_and_choice_rules` | `fn inline_single_use_sequence_and_choice_rules(grammar: &mut NamedGrammar, stats: &mut SimplifyStats) {` |
| `src/grammar_ir/transforms/simplify.rs` | 240 | `reference_counts` | `fn reference_counts(grammar: &NamedGrammar) -> HashMap<String, usize> {` |
| `src/grammar_ir/transforms/simplify.rs` | 248 | `collect_ref_counts` | `fn collect_ref_counts(expr: &GrammarExpr, counts: &mut HashMap<String, usize>) {` |
| `src/grammar_ir/transforms/simplify.rs` | 295 | `protected_rule_names` | `fn protected_rule_names(grammar: &NamedGrammar) -> HashSet<String> {` |
| `src/grammar_ir/transforms/simplify.rs` | 309 | `inline_refs_in_expr` | `fn inline_refs_in_expr(` |
| `src/grammar_ir/transforms/simplify.rs` | 370 | `inline_sequence_refs` | `fn inline_sequence_refs(` |
| `src/grammar_ir/transforms/simplify.rs` | 397 | `inline_choice_refs` | `fn inline_choice_refs(` |
| `src/grammar_ir/transforms/simplify.rs` | 424 | `single_use_ref_to_sequence` | `fn single_use_ref_to_sequence(` |
| `src/grammar_ir/transforms/simplify.rs` | 443 | `single_use_ref_to_choice` | `fn single_use_ref_to_choice(` |
| `src/grammar_ir/transforms/simplify.rs` | 466 | `lit` | `fn lit(s: &str) -> GrammarExpr {` |
| `src/grammar_ir/transforms/simplify.rs` | 470 | `nt` | `fn nt(name: &str, expr: GrammarExpr) -> NamedRule {` |
| `src/grammar_ir/transforms/simplify.rs` | 480 | `flattens_singleton_and_nested_sequences_and_choices` | `fn flattens_singleton_and_nested_sequences_and_choices() {` |
| `src/grammar_ir/transforms/simplify.rs` | 505 | `inlines_single_use_sequence_rule_into_longer_sequence` | `fn inlines_single_use_sequence_rule_into_longer_sequence() {` |
| `src/grammar_ir/transforms/simplify.rs` | 526 | `does_not_inline_single_use_sequence_rule_as_whole_rule` | `fn does_not_inline_single_use_sequence_rule_as_whole_rule() {` |
| `src/grammar_ir/transforms/simplify.rs` | 542 | `inlines_single_use_choice_rule_into_longer_choice` | `fn inlines_single_use_choice_rule_into_longer_choice() {` |
| `src/grammar_ir/transforms/simplify.rs` | 563 | `simplifies_safe_repeat_shapes` | `fn simplifies_safe_repeat_shapes() {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 16 | `promote_choice_terminals_exact` | `pub fn promote_choice_terminals_exact(` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 63 | `PromotionStats` | `pub struct PromotionStats {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 71 | `PathStep` | `enum PathStep {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 88 | `Candidate` | `struct Candidate {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 97 | `CandidateCollector` | `struct CandidateCollector {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 105 | `new` | `fn new(include_non_literal_terminals: bool) -> Self {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 114 | `collect` | `fn collect(&mut self, grammar: &NamedGrammar) {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 131 | `collect_expr` | `fn collect_expr(&mut self, rule_idx: usize, expr: &GrammarExpr, path: &mut Vec<PathStep>) {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 243 | `eligible_atom` | `fn eligible_atom(&self, expr: &GrammarExpr) -> Option<GrammarExpr> {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 259 | `atom_id` | `fn atom_id(&mut self, atom: GrammarExpr) -> usize {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 270 | `solve_exact` | `fn solve_exact(atom_total_counts: &[usize], candidates: &[Candidate]) -> Vec<usize> {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 294 | `ExactSearch` | `struct ExactSearch<'a> {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 305 | `visit` | `fn visit(&mut self, idx: usize) {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 327 | `include` | `fn include(&mut self, idx: usize) {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 334 | `exclude` | `fn exclude(&mut self, idx: usize) {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 341 | `standalone_atoms` | `fn standalone_atoms(&self) -> usize {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 349 | `unavoidable_standalone_atoms` | `fn unavoidable_standalone_atoms(&self, idx: usize) -> usize {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 360 | `promoted_cost` | `fn promoted_cost(` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 379 | `TerminalNameGenerator` | `struct TerminalNameGenerator {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 385 | `new` | `fn new(used: BTreeSet<String>) -> Self {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 389 | `next` | `fn next(&mut self) -> String {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 400 | `replace_candidate` | `fn replace_candidate(grammar: &mut NamedGrammar, candidate: &Candidate, terminal_name: &str) {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 427 | `expr_at_path_mut` | `fn expr_at_path_mut<'a>(mut expr: &'a mut GrammarExpr, path: &[PathStep]) -> &'a mut GrammarExpr {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 464 | `lit` | `fn lit(s: &str) -> GrammarExpr {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 468 | `nt` | `fn nt(name: &str, expr: GrammarExpr) -> NamedRule {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 477 | `terminal_rule_count` | `fn terminal_rule_count(grammar: &NamedGrammar) -> usize {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 482 | `promotes_mixed_choice_literal_subset_when_it_reduces_terminals` | `fn promotes_mixed_choice_literal_subset_when_it_reduces_terminals() {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 508 | `does_not_promote_dense_pair_cover_when_standalone_literals_are_cheaper` | `fn does_not_promote_dense_pair_cover_when_standalone_literals_are_cheaper() {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 531 | `does_not_rewrite_cost_neutral_ties` | `fn does_not_rewrite_cost_neutral_ties() {` |
| `src/grammar_ir/transforms/terminal_choice.rs` | 549 | `non_literal_terminal_atoms_are_opt_in` | `fn non_literal_terminal_atoms_are_opt_in() {` |
| `src/import/ebnf/mod.rs` | 8 | `is_terminal_name` | `fn is_terminal_name(name: &str) -> bool {` |
| `src/import/ebnf/mod.rs` | 16 | `Token` | `enum Token {` |
| `src/import/ebnf/mod.rs` | 31 | `Lexer` | `struct Lexer<'a> {` |
| `src/import/ebnf/mod.rs` | 37 | `new` | `fn new(input: &'a str) -> Self {` |
| `src/import/ebnf/mod.rs` | 44 | `peek` | `fn peek(&self) -> Option<u8> {` |
| `src/import/ebnf/mod.rs` | 48 | `advance` | `fn advance(&mut self) -> Option<u8> {` |
| `src/import/ebnf/mod.rs` | 54 | `skip_whitespace_inline` | `fn skip_whitespace_inline(&mut self) {` |
| `src/import/ebnf/mod.rs` | 64 | `skip_comment` | `fn skip_comment(&mut self) {` |
| `src/import/ebnf/mod.rs` | 73 | `lex_string` | `fn lex_string(&mut self, quote: u8) -> Result<String, GlrMaskError> {` |
| `src/import/ebnf/mod.rs` | 115 | `lex_char_class` | `fn lex_char_class(&mut self) -> Result<(String, bool), GlrMaskError> {` |
| `src/import/ebnf/mod.rs` | 140 | `lex_ident` | `fn lex_ident(&mut self, first: u8) -> String {` |
| `src/import/ebnf/mod.rs` | 153 | `lex_separator` | `fn lex_separator(&mut self) {` |
| `src/import/ebnf/mod.rs` | 161 | `lex_literal_token` | `fn lex_literal_token(&mut self, quote: u8) -> Result<Token, GlrMaskError> {` |
| `src/import/ebnf/mod.rs` | 165 | `lex_char_class_token` | `fn lex_char_class_token(&mut self) -> Result<Token, GlrMaskError> {` |
| `src/import/ebnf/mod.rs` | 170 | `lex_ident_token` | `fn lex_ident_token(&mut self, first: u8) -> Token {` |
| `src/import/ebnf/mod.rs` | 174 | `tokenize` | `fn tokenize(&mut self) -> Result<Vec<Token>, GlrMaskError> {` |
| `src/import/ebnf/mod.rs` | 207 | `hex_digit` | `fn hex_digit(b: u8) -> Result<u8, GlrMaskError> {` |
| `src/import/ebnf/mod.rs` | 216 | `is_ebnf_ident_start` | `fn is_ebnf_ident_start(byte: u8) -> bool {` |
| `src/import/ebnf/mod.rs` | 220 | `is_ebnf_ident_continue` | `fn is_ebnf_ident_continue(byte: u8) -> bool {` |
| `src/import/ebnf/mod.rs` | 224 | `apply_postfix_operator` | `fn apply_postfix_operator(atom: GrammarExpr, token: Option<&Token>) -> GrammarExpr {` |
| `src/import/ebnf/mod.rs` | 233 | `Parser` | `struct Parser {` |
| `src/import/ebnf/mod.rs` | 239 | `new` | `fn new(tokens: Vec<Token>) -> Self {` |
| `src/import/ebnf/mod.rs` | 243 | `peek` | `fn peek(&self) -> Option<&Token> {` |
| `src/import/ebnf/mod.rs` | 247 | `advance` | `fn advance(&mut self) -> Option<Token> {` |
| `src/import/ebnf/mod.rs` | 253 | `expect` | `fn expect(&mut self, expected: &Token) -> Result<(), GlrMaskError> {` |
| `src/import/ebnf/mod.rs` | 267 | `skip_newlines` | `fn skip_newlines(&mut self) {` |
| `src/import/ebnf/mod.rs` | 273 | `parse_rule_name` | `fn parse_rule_name(&mut self) -> Result<String, GlrMaskError> {` |
| `src/import/ebnf/mod.rs` | 286 | `parse_rule` | `fn parse_rule(&mut self) -> Result<NamedRule, GlrMaskError> {` |
| `src/import/ebnf/mod.rs` | 299 | `parse_grammar` | `fn parse_grammar(&mut self) -> Result<NamedGrammar, GlrMaskError> {` |
| `src/import/ebnf/mod.rs` | 314 | `parse_alternatives` | `fn parse_alternatives(&mut self) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/import/ebnf/mod.rs` | 323 | `parse_sequence` | `fn parse_sequence(&mut self) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/import/ebnf/mod.rs` | 331 | `is_unit_start` | `fn is_unit_start(&self) -> bool {` |
| `src/import/ebnf/mod.rs` | 342 | `parse_unit` | `fn parse_unit(&mut self) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/import/ebnf/mod.rs` | 354 | `parse_group` | `fn parse_group(&mut self) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/import/ebnf/mod.rs` | 360 | `parse_atom` | `fn parse_atom(&mut self) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/import/ebnf/mod.rs` | 376 | `parse_ebnf` | `pub fn parse_ebnf(input: &str) -> Result<GrammarDef, GlrMaskError> {` |
| `src/import/ebnf/mod.rs` | 382 | `parse_ebnf_to_named` | `pub fn parse_ebnf_to_named(input: &str) -> Result<NamedGrammar, GlrMaskError> {` |
| `src/import/json_schema/diagnostics.rs` | 8 | `ImportResult` | `pub(crate) type ImportResult<T> = Result<T, SchemaImportError>;` |
| `src/import/json_schema/diagnostics.rs` | 11 | `SchemaImportError` | `pub(crate) struct SchemaImportError {` |
| `src/import/json_schema/diagnostics.rs` | 16 | `new` | `pub(crate) fn new(message: impl Into<String>) -> Self {` |
| `src/import/json_schema/diagnostics.rs` | 20 | `at` | `pub(crate) fn at(location: &str, message: impl AsRef<str>) -> Self {` |
| `src/import/json_schema/diagnostics.rs` | 24 | `message` | `pub(crate) fn message(&self) -> &str {` |
| `src/import/json_schema/diagnostics.rs` | 30 | `from` | `fn from(value: SchemaImportError) -> Self {` |
| `src/import/json_schema/diagnostics.rs` | 54 | `is_documented_unsupported_keyword` | `pub(crate) fn is_documented_unsupported_keyword(keyword: &str) -> bool {` |
| `src/import/json_schema/load/collect.rs` | 14 | `collect_definitions` | `pub(super) fn collect_definitions(` |
| `src/import/json_schema/load/collect.rs` | 58 | `collect_ref_targets` | `pub(super) fn collect_ref_targets(` |
| `src/import/json_schema/load/keywords.rs` | 18 | `validate_supported_keys` | `pub(super) fn validate_supported_keys(object: &Map<String, Value>, location: &str) -> ImportResult<()> {` |
| `src/import/json_schema/load/keywords.rs` | 31 | `is_unsupported_validation_key` | `fn is_unsupported_validation_key(key: &str) -> bool {` |
| `src/import/json_schema/load/keywords.rs` | 35 | `load_types` | `pub(super) fn load_types(object: &Map<String, Value>, location: &str) -> ImportResult<Option<Vec<SchemaType>>> {` |
| `src/import/json_schema/load/keywords.rs` | 59 | `parse_type_name` | `fn parse_type_name(name: &str, location: &str) -> ImportResult<SchemaType> {` |
| `src/import/json_schema/load/keywords.rs` | 72 | `load_enum_values` | `pub(super) fn load_enum_values(object: &Map<String, Value>, location: &str) -> ImportResult<Option<Vec<Value>>> {` |
| `src/import/json_schema/load/keywords.rs` | 83 | `load_object_keywords` | `pub(super) fn load_object_keywords(` |
| `src/import/json_schema/load/keywords.rs` | 144 | `load_array_keywords` | `pub(super) fn load_array_keywords(` |
| `src/import/json_schema/load/keywords.rs` | 197 | `load_string_keywords` | `pub(super) fn load_string_keywords(object: &Map<String, Value>, location: &str) -> ImportResult<StringSchema> {` |
| `src/import/json_schema/load/keywords.rs` | 206 | `load_number_keywords` | `pub(super) fn load_number_keywords(` |
| `src/import/json_schema/load/keywords.rs` | 248 | `read_usize_keyword` | `fn read_usize_keyword(object: &Map<String, Value>, key: &str, location: &str) -> ImportResult<Option<usize>> {` |
| `src/import/json_schema/load/keywords.rs` | 260 | `read_f64_keyword` | `fn read_f64_keyword(object: &Map<String, Value>, key: &str, location: &str) -> ImportResult<Option<f64>> {` |
| `src/import/json_schema/load/keywords.rs` | 270 | `read_string_keyword` | `fn read_string_keyword(object: &Map<String, Value>, key: &str, location: &str) -> ImportResult<Option<String>> {` |
| `src/import/json_schema/load/pointers.rs` | 9 | `collect_all_ref_pointers` | `pub(super) fn collect_all_ref_pointers(value: &Value, refs: &mut std::collections::BTreeSet<String>) {` |
| `src/import/json_schema/load/pointers.rs` | 24 | `local_id_alias` | `pub(super) fn local_id_alias(object: &Map<String, Value>, location: &str) -> Option<String> {` |
| `src/import/json_schema/load/pointers.rs` | 38 | `escape_pointer_segment` | `pub(super) fn escape_pointer_segment(segment: &str) -> String {` |
| `src/import/json_schema/load/shape.rs` | 8 | `singleton_all_of_ref_without_siblings` | `pub(super) fn singleton_all_of_ref_without_siblings(assertions: &SchemaAssertions) -> Option<&str> {` |
| `src/import/json_schema/load/shape.rs` | 25 | `one_of_mixes_ref_and_inline_branches` | `pub(super) fn one_of_mixes_ref_and_inline_branches(branches: &[Schema]) -> bool {` |
| `src/import/json_schema/load/shape.rs` | 38 | `schema_is_null_only_inline_branch` | `pub(super) fn schema_is_null_only_inline_branch(schema: &Schema) -> bool {` |
| `src/import/json_schema/load/typed.rs` | 21 | `load_document` | `pub(crate) fn load_document(root: &Value) -> ImportResult<SchemaDocument> {` |
| `src/import/json_schema/load/typed.rs` | 58 | `load_schema_at` | `pub(super) fn load_schema_at(value: &Value, location: &str) -> ImportResult<Schema> {` |
| `src/import/json_schema/load/typed.rs` | 67 | `load_object_schema` | `fn load_object_schema(object: &Map<String, Value>, location: &str) -> ImportResult<Schema> {` |
| `src/import/json_schema/load/typed.rs` | 95 | `load_assertions` | `fn load_assertions(object: &Map<String, Value>, location: &str) -> ImportResult<SchemaAssertions> {` |
| `src/import/json_schema/load/typed.rs` | 128 | `load_schema_array` | `fn load_schema_array(` |
| `src/import/json_schema/load/typed.rs` | 146 | `load_schema_member` | `fn load_schema_member(` |
| `src/import/json_schema/load/typed.rs` | 157 | `should_load_object_assertion` | `fn should_load_object_assertion(object: &Map<String, Value>, types: Option<&[SchemaType]>) -> bool {` |
| `src/import/json_schema/load/typed.rs` | 170 | `should_load_array_assertion` | `fn should_load_array_assertion(object: &Map<String, Value>, types: Option<&[SchemaType]>) -> bool {` |
| `src/import/json_schema/load/typed.rs` | 177 | `should_load_string_assertion` | `fn should_load_string_assertion(object: &Map<String, Value>, types: Option<&[SchemaType]>) -> bool {` |
| `src/import/json_schema/load/typed.rs` | 184 | `should_load_number_assertion` | `fn should_load_number_assertion(object: &Map<String, Value>, types: Option<&[SchemaType]>) -> bool {` |
| `src/import/json_schema/load/typed.rs` | 192 | `type_mentions` | `fn type_mentions(types: Option<&[SchemaType]>, wanted: SchemaType) -> bool {` |
| `src/import/json_schema/lower/array/mod.rs` | 9 | `lower_array` | `pub(crate) fn lower_array(&mut self, schema: &ArraySchema) -> ImportResult<GrammarExpr> {` |
| `src/import/json_schema/lower/array/mod.rs` | 45 | `should_terminalize_bounded_scalar_array` | `fn should_terminalize_bounded_scalar_array(&self, max_items: usize) -> bool {` |
| `src/import/json_schema/lower/array/mod.rs` | 49 | `array_body` | `fn array_body(&self, item: GrammarExpr, min: usize, max: Option<usize>) -> GrammarExpr {` |
| `src/import/json_schema/lower/array/mod.rs` | 79 | `bounded_homogeneous_array_exprnfa` | `fn bounded_homogeneous_array_exprnfa(` |
| `src/import/json_schema/lower/array/mod.rs` | 118 | `bounded_homogeneous_array_terminal` | `fn bounded_homogeneous_array_terminal(` |
| `src/import/json_schema/lower/array/mod.rs` | 150 | `unbounded_homogeneous_array_terminal` | `fn unbounded_homogeneous_array_terminal(&mut self, item: GrammarExpr) -> GrammarExpr {` |
| `src/import/json_schema/lower/array/mod.rs` | 162 | `lower_tuple_array_body` | `fn lower_tuple_array_body(&mut self, schema: &ArraySchema) -> ImportResult<GrammarExpr> {` |
| `src/import/json_schema/lower/array/mod.rs` | 211 | `fixed_array_items` | `fn fixed_array_items(&mut self, items: &[super::super::schema::Schema]) -> ImportResult<GrammarExpr> {` |
| `src/import/json_schema/lower/array/mod.rs` | 225 | `tuple_tail_items` | `fn tuple_tail_items(` |
| `src/import/json_schema/lower/array/mod.rs` | 253 | `bounded_array_object_item_candidate` | `fn bounded_array_object_item_candidate(schema: &super::super::schema::Schema) -> bool {` |
| `src/import/json_schema/lower/mod.rs` | 46 | `lower_document` | `pub(crate) fn lower_document(` |
| `src/import/json_schema/lower/mod.rs` | 54 | `Lowerer` | `pub(crate) struct Lowerer<'a> {` |
| `src/import/json_schema/lower/mod.rs` | 76 | `new` | `fn new(document: &'a SchemaDocument, config: JsonSchemaConfig) -> Self {` |
| `src/import/json_schema/lower/mod.rs` | 111 | `finish` | `fn finish(mut self) -> ImportResult<NamedGrammar> {` |
| `src/import/json_schema/lower/mod.rs` | 117 | `install_json_builtins` | `fn install_json_builtins(&mut self) {` |
| `src/import/json_schema/lower/mod.rs` | 189 | `item_separator_expr` | `pub(crate) fn item_separator_expr(&self) -> GrammarExpr {` |
| `src/import/json_schema/lower/mod.rs` | 193 | `key_separator_expr` | `pub(crate) fn key_separator_expr(&self) -> GrammarExpr {` |
| `src/import/json_schema/lower/mod.rs` | 197 | `separator_regex` | `fn separator_regex(&self, separator: &str) -> String {` |
| `src/import/json_schema/lower/mod.rs` | 204 | `json_string_char_regex` | `fn json_string_char_regex(&self) -> String {` |
| `src/import/json_schema/lower/mod.rs` | 208 | `lower_schema` | `pub(crate) fn lower_schema(&mut self, schema: &Schema) -> ImportResult<GrammarExpr> {` |
| `src/import/json_schema/lower/mod.rs` | 217 | `lower_ref` | `pub(crate) fn lower_ref(&mut self, pointer: &str) -> ImportResult<GrammarExpr> {` |
| `src/import/json_schema/lower/mod.rs` | 237 | `resolve_ref_target` | `pub(crate) fn resolve_ref_target(&self, pointer: &str) -> ImportResult<&'a Schema> {` |
| `src/import/json_schema/lower/mod.rs` | 245 | `lower_assertions` | `fn lower_assertions(` |
| `src/import/json_schema/lower/mod.rs` | 306 | `selected_types` | `fn selected_types(` |
| `src/import/json_schema/lower/mod.rs` | 326 | `lower_for_type` | `fn lower_for_type(` |
| `src/import/json_schema/lower/mod.rs` | 358 | `inferred_constrained_types` | `fn inferred_constrained_types(&self, assertions: &SchemaAssertions) -> Vec<SchemaType> {` |
| `src/import/json_schema/lower/mod.rs` | 375 | `lower_untyped_single_family_assertions` | `fn lower_untyped_single_family_assertions(` |
| `src/import/json_schema/lower/mod.rs` | 406 | `lower_json_literal` | `pub(crate) fn lower_json_literal(&mut self, value: &Value) -> GrammarExpr {` |
| `src/import/json_schema/lower/mod.rs` | 453 | `add_nonterminal_rule` | `pub(crate) fn add_nonterminal_rule(&mut self, name: &str, expr: GrammarExpr) {` |
| `src/import/json_schema/lower/mod.rs` | 463 | `add_terminal_rule` | `pub(crate) fn add_terminal_rule(&mut self, name: &str, expr: GrammarExpr) {` |
| `src/import/json_schema/lower/mod.rs` | 473 | `add_internal_terminal_rule` | `pub(crate) fn add_internal_terminal_rule(&mut self, name: &str, expr: GrammarExpr) {` |
| `src/import/json_schema/lower/mod.rs` | 483 | `fresh_rule_name` | `pub(crate) fn fresh_rule_name(&mut self, prefix: &str) -> String {` |
| `src/import/json_schema/lower/mod.rs` | 494 | `large_string_enum_regex_literals` | `fn large_string_enum_regex_literals(assertions: &SchemaAssertions) -> ImportResult<Option<Vec<String>>> {` |
| `src/import/json_schema/lower/mod.rs` | 544 | `string_enum_regex` | `fn string_enum_regex(encoded_literals: &[String]) -> String {` |
| `src/import/json_schema/lower/mod.rs` | 555 | `factored_small_string_enum_expr` | `fn factored_small_string_enum_expr(values: &[&Value]) -> Option<GrammarExpr> {` |
| `src/import/json_schema/lower/mod.rs` | 576 | `collect_shared_ap_exclusion_plan` | `fn collect_shared_ap_exclusion_plan(document: &SchemaDocument) -> (BTreeSet<String>, Vec<String>) {` |
| `src/import/json_schema/lower/mod.rs` | 591 | `collect_shared_ap_exclusions_from_schema` | `fn collect_shared_ap_exclusions_from_schema(` |
| `src/import/json_schema/lower/mod.rs` | 643 | `normalize_local_ref` | `pub(crate) fn normalize_local_ref(pointer: &str) -> ImportResult<String> {` |
| `src/import/json_schema/lower/mod.rs` | 655 | `is_local_fragment_alias` | `fn is_local_fragment_alias(pointer: &str) -> bool {` |
| `src/import/json_schema/lower/mod.rs` | 659 | `is_absolute_self_ref_alias` | `fn is_absolute_self_ref_alias(pointer: &str) -> bool {` |
| `src/import/json_schema/lower/mod.rs` | 663 | `r` | `pub(crate) fn r(name: &str) -> GrammarExpr {` |
| `src/import/json_schema/lower/mod.rs` | 667 | `lit` | `pub(crate) fn lit(text: &str) -> GrammarExpr {` |
| `src/import/json_schema/lower/mod.rs` | 671 | `lit_bytes` | `pub(crate) fn lit_bytes(bytes: Vec<u8>) -> GrammarExpr {` |
| `src/import/json_schema/lower/mod.rs` | 675 | `seq` | `pub(crate) fn seq(mut parts: Vec<GrammarExpr>) -> GrammarExpr {` |
| `src/import/json_schema/lower/mod.rs` | 683 | `choice` | `pub(crate) fn choice(mut alternatives: Vec<GrammarExpr>) -> GrammarExpr {` |
| `src/import/json_schema/lower/mod.rs` | 698 | `never` | `pub(crate) fn never() -> GrammarExpr {` |
| `src/import/json_schema/lower/number/mod.rs` | 14 | `lower_number` | `pub(crate) fn lower_number(&mut self, schema: &NumberSchema) -> ImportResult<GrammarExpr> {` |
| `src/import/json_schema/lower/number/mod.rs` | 59 | `lower_integer` | `fn lower_integer(&mut self, schema: &NumberSchema) -> ImportResult<GrammarExpr> {` |
| `src/import/json_schema/lower/number/mod.rs` | 107 | `integer_lower_bound` | `fn integer_lower_bound(schema: &NumberSchema) -> Option<i64> {` |
| `src/import/json_schema/lower/number/mod.rs` | 119 | `integer_upper_bound` | `fn integer_upper_bound(schema: &NumberSchema) -> Option<i64> {` |
| `src/import/json_schema/lower/number/mod.rs` | 131 | `integer_satisfies_multiple` | `fn integer_satisfies_multiple(value: i64, multiple: Option<f64>) -> bool {` |
| `src/import/json_schema/lower/number/mod.rs` | 139 | `bounded_integer_multiple_choice` | `fn bounded_integer_multiple_choice(` |
| `src/import/json_schema/lower/number/mod.rs` | 162 | `ceil_div_i64` | `fn ceil_div_i64(value: i64, divisor: i64) -> i64 {` |
| `src/import/json_schema/lower/number/mod.rs` | 168 | `integer_multiple_expr` | `fn integer_multiple_expr(multiple: f64) -> Option<GrammarExpr> {` |
| `src/import/json_schema/lower/number/mod.rs` | 172 | `positive_integer_multiple_value` | `fn positive_integer_multiple_value(multiple: f64) -> Option<u64> {` |
| `src/import/json_schema/lower/number/mod.rs` | 180 | `positive_integer_multiple_i64` | `fn positive_integer_multiple_i64(multiple: f64) -> Option<i64> {` |
| `src/import/json_schema/lower/number/mod.rs` | 185 | `power_of_ten_multiple_regex` | `fn power_of_ten_multiple_regex(multiple: f64) -> Option<String> {` |
| `src/import/json_schema/lower/number/mod.rs` | 206 | `decimal_multiple_regex` | `fn decimal_multiple_regex(multiple: f64) -> Option<String> {` |
| `src/import/json_schema/lower/number/mod.rs` | 212 | `DecimalStep` | `struct DecimalStep {` |
| `src/import/json_schema/lower/number/mod.rs` | 218 | `parse_decimal_step` | `fn parse_decimal_step(multiple: f64) -> Option<DecimalStep> {` |
| `src/import/json_schema/lower/number/mod.rs` | 247 | `decimal_fraction_regex` | `fn decimal_fraction_regex(step: &DecimalStep) -> Option<String> {` |
| `src/import/json_schema/lower/object/mod.rs` | 28 | `ObjectItem` | `struct ObjectItem {` |
| `src/import/json_schema/lower/object/mod.rs` | 38 | `AnyOfFixedObjectItem` | `struct AnyOfFixedObjectItem {` |
| `src/import/json_schema/lower/object/mod.rs` | 44 | `AnyOfFixedObjectVariant` | `struct AnyOfFixedObjectVariant {` |
| `src/import/json_schema/lower/object/mod.rs` | 48 | `AnyOfObjectVariant` | `struct AnyOfObjectVariant {` |
| `src/import/json_schema/lower/object/mod.rs` | 57 | `AnyOfFixedObjectState` | `struct AnyOfFixedObjectState {` |
| `src/import/json_schema/lower/object/mod.rs` | 64 | `AnyOfObjectPhase` | `enum AnyOfObjectPhase {` |
| `src/import/json_schema/lower/object/mod.rs` | 71 | `ShadowOwnerState` | `enum ShadowOwnerState {` |
| `src/import/json_schema/lower/object/mod.rs` | 82 | `AnyOfObjectState` | `struct AnyOfObjectState {` |
| `src/import/json_schema/lower/object/mod.rs` | 90 | `is_obviously_object_valued_property` | `fn is_obviously_object_valued_property(schema: &Schema) -> bool {` |
| `src/import/json_schema/lower/object/mod.rs` | 102 | `obvious_object_valued_property_count` | `fn obvious_object_valued_property_count(schema: &ObjectSchema) -> usize {` |
| `src/import/json_schema/lower/object/mod.rs` | 110 | `is_unconstrained_open_object_schema` | `fn is_unconstrained_open_object_schema(schema: &ObjectSchema) -> bool {` |
| `src/import/json_schema/lower/object/mod.rs` | 120 | `len` | `fn len(&self) -> usize {` |
| `src/import/json_schema/lower/object/mod.rs` | 124 | `advance_cursor` | `fn advance_cursor(&self, cursor: usize, key: &str) -> Option<usize> {` |
| `src/import/json_schema/lower/object/mod.rs` | 135 | `close_allowed` | `fn close_allowed(&self, cursor: usize) -> bool {` |
| `src/import/json_schema/lower/object/mod.rs` | 139 | `legal_next_keys` | `fn legal_next_keys(&self, cursor: usize) -> Vec<&str> {` |
| `src/import/json_schema/lower/object/mod.rs` | 150 | `value_expr_for_key` | `fn value_expr_for_key(&self, key: &str) -> Option<GrammarExpr> {` |
| `src/import/json_schema/lower/object/mod.rs` | 159 | `len` | `fn len(&self) -> usize {` |
| `src/import/json_schema/lower/object/mod.rs` | 163 | `advance_cursor` | `fn advance_cursor(&self, cursor: usize, key: &str) -> Option<usize> {` |
| `src/import/json_schema/lower/object/mod.rs` | 174 | `close_allowed` | `fn close_allowed(&self, cursor: usize) -> bool {` |
| `src/import/json_schema/lower/object/mod.rs` | 178 | `legal_next_keys` | `fn legal_next_keys(&self, cursor: usize) -> Vec<&str> {` |
| `src/import/json_schema/lower/object/mod.rs` | 189 | `value_expr_for_key` | `fn value_expr_for_key(&self, key: &str) -> Option<GrammarExpr> {` |
| `src/import/json_schema/lower/object/mod.rs` | 196 | `has_required_items` | `fn has_required_items(&self) -> bool {` |
| `src/import/json_schema/lower/object/mod.rs` | 202 | `lower_object` | `pub(crate) fn lower_object(&mut self, schema: &ObjectSchema) -> ImportResult<GrammarExpr> {` |
| `src/import/json_schema/lower/object/mod.rs` | 206 | `lower_object_requiring_any_property` | `pub(crate) fn lower_object_requiring_any_property(` |
| `src/import/json_schema/lower/object/mod.rs` | 214 | `lower_object_with_exclusive_properties` | `pub(crate) fn lower_object_with_exclusive_properties(` |
| `src/import/json_schema/lower/object/mod.rs` | 223 | `try_lower_closed_object_any_of_variants` | `pub(crate) fn try_lower_closed_object_any_of_variants(` |
| `src/import/json_schema/lower/object/mod.rs` | 251 | `try_lower_open_object_any_of_variants` | `pub(crate) fn try_lower_open_object_any_of_variants(` |
| `src/import/json_schema/lower/object/mod.rs` | 277 | `try_lower_ref_string_path_object_any_of` | `pub(crate) fn try_lower_ref_string_path_object_any_of(` |
| `src/import/json_schema/lower/object/mod.rs` | 332 | `resolve_branch_schema` | `fn resolve_branch_schema<'b>(&'b self, schema: &'b Schema) -> ImportResult<&'b Schema> {` |
| `src/import/json_schema/lower/object/mod.rs` | 339 | `is_path_recursive_open_object_branch` | `fn is_path_recursive_open_object_branch(` |
| `src/import/json_schema/lower/object/mod.rs` | 386 | `lower_object_internal` | `fn lower_object_internal(` |
| `src/import/json_schema/lower/object/mod.rs` | 677 | `dynamic_pair_list_body` | `fn dynamic_pair_list_body(` |
| `src/import/json_schema/lower/object/mod.rs` | 711 | `schema_has_huge_bounded_string` | `fn schema_has_huge_bounded_string(&self, schema: &Schema) -> bool {` |
| `src/import/json_schema/lower/object/mod.rs` | 728 | `try_lower_pattern_map_pair_list_object` | `fn try_lower_pattern_map_pair_list_object(` |
| `src/import/json_schema/lower/object/mod.rs` | 804 | `try_lower_wrapper_pattern_map_anyof_value` | `fn try_lower_wrapper_pattern_map_anyof_value(` |
| `src/import/json_schema/lower/object/mod.rs` | 832 | `lower_fixed_object_body_exprnfa` | `fn lower_fixed_object_body_exprnfa(` |
| `src/import/json_schema/lower/object/mod.rs` | 938 | `collect_closed_any_of_object_variant` | `fn collect_closed_any_of_object_variant(` |
| `src/import/json_schema/lower/object/mod.rs` | 1017 | `collect_open_any_of_object_variant` | `fn collect_open_any_of_object_variant(` |
| `src/import/json_schema/lower/object/mod.rs` | 1024 | `collect_open_any_of_object_variant_inner` | `fn collect_open_any_of_object_variant_inner(` |
| `src/import/json_schema/lower/object/mod.rs` | 1138 | `add_expr_nfa_symbol_path` | `fn add_expr_nfa_symbol_path(` |
| `src/import/json_schema/lower/object/mod.rs` | 1160 | `split_additional_key_colon_transition` | `fn split_additional_key_colon_transition(symbol: GrammarExpr) -> Vec<GrammarExpr> {` |
| `src/import/json_schema/lower/object/mod.rs` | 1171 | `is_shared_additional_key_colon_choice` | `fn is_shared_additional_key_colon_choice(alternatives: &[GrammarExpr]) -> bool {` |
| `src/import/json_schema/lower/object/mod.rs` | 1186 | `is_shared_additional_key_colon_base_ref` | `fn is_shared_additional_key_colon_base_ref(expr: &GrammarExpr) -> bool {` |
| `src/import/json_schema/lower/object/mod.rs` | 1190 | `is_shared_additional_key_colon_addback` | `fn is_shared_additional_key_colon_addback(expr: &GrammarExpr) -> bool {` |
| `src/import/json_schema/lower/object/mod.rs` | 1206 | `split_object_pair_symbols` | `fn split_object_pair_symbols(pair: &GrammarExpr) -> ImportResult<[GrammarExpr; 2]> {` |
| `src/import/json_schema/lower/object/mod.rs` | 1217 | `split_object_pair_symbol_paths` | `fn split_object_pair_symbol_paths(pair: &GrammarExpr) -> ImportResult<Vec<[GrammarExpr; 2]>> {` |
| `src/import/json_schema/lower/object/mod.rs` | 1227 | `lower_closed_any_of_object_variants_expr_nfa` | `fn lower_closed_any_of_object_variants_expr_nfa(` |
| `src/import/json_schema/lower/object/mod.rs` | 1318 | `is_json_value_expr` | `fn is_json_value_expr(expr: &GrammarExpr) -> bool {` |
| `src/import/json_schema/lower/object/mod.rs` | 1322 | `is_json_string_expr` | `fn is_json_string_expr(expr: &GrammarExpr) -> bool {` |
| `src/import/json_schema/lower/object/mod.rs` | 1326 | `is_json_string_constrained_expr` | `fn is_json_string_constrained_expr(expr: &GrammarExpr) -> bool {` |
| `src/import/json_schema/lower/object/mod.rs` | 1330 | `non_string_json_value_expr` | `fn non_string_json_value_expr() -> GrammarExpr {` |
| `src/import/json_schema/lower/object/mod.rs` | 1340 | `invalid_residual_value_for_owner` | `fn invalid_residual_value_for_owner(owner_value_expr: &GrammarExpr) -> Option<GrammarExpr> {` |
| `src/import/json_schema/lower/object/mod.rs` | 1356 | `select_shadow_owner_for_variant` | `fn select_shadow_owner_for_variant(` |
| `src/import/json_schema/lower/object/mod.rs` | 1399 | `shadow_owner_suppresses_close` | `fn shadow_owner_suppresses_close(` |
| `src/import/json_schema/lower/object/mod.rs` | 1416 | `shadow_owner_can_take_additional` | `fn shadow_owner_can_take_additional(` |
| `src/import/json_schema/lower/object/mod.rs` | 1429 | `advance_shadow_owner_on_key` | `fn advance_shadow_owner_on_key(` |
| `src/import/json_schema/lower/object/mod.rs` | 1473 | `lower_open_any_of_object_variants_expr_nfa` | `fn lower_open_any_of_object_variants_expr_nfa(` |
| `src/import/json_schema/lower/object/mod.rs` | 1822 | `lower_fixed_object_body_exprnfa_without_group` | `fn lower_fixed_object_body_exprnfa_without_group(` |
| `src/import/json_schema/lower/object/mod.rs` | 1962 | `split_literal_key_symbol` | `fn split_literal_key_symbol(symbol: GrammarExpr) -> Vec<GrammarExpr> {` |
| `src/import/json_schema/lower/object/mod.rs` | 1985 | `object_pair_path_symbols` | `fn object_pair_path_symbols(` |
| `src/import/json_schema/lower/object/mod.rs` | 1994 | `lower_snowplow_large_pattern_object_key_trie` | `fn lower_snowplow_large_pattern_object_key_trie(` |
| `src/import/json_schema/lower/object/mod.rs` | 2163 | `lower_large_optional_open_object_fused_prefix_chain` | `fn lower_large_optional_open_object_fused_prefix_chain(` |
| `src/import/json_schema/lower/object/mod.rs` | 2217 | `lower_large_closed_object_prefix_chain` | `fn lower_large_closed_object_prefix_chain(&mut self, items: &[ObjectItem]) -> GrammarExpr {` |
| `src/import/json_schema/lower/object/mod.rs` | 2253 | `lower_large_closed_object_fixed_pair_loop` | `fn lower_large_closed_object_fixed_pair_loop(` |
| `src/import/json_schema/lower/object/mod.rs` | 2327 | `lower_required_prefix_open_object_pair_loop` | `fn lower_required_prefix_open_object_pair_loop(` |
| `src/import/json_schema/lower/object/mod.rs` | 2429 | `lower_property_item` | `fn lower_property_item(` |
| `src/import/json_schema/lower/object/mod.rs` | 2456 | `lower_object_property_value_schema` | `fn lower_object_property_value_schema(&mut self, schema: &Schema) -> ImportResult<GrammarExpr> {` |
| `src/import/json_schema/lower/object/mod.rs` | 2498 | `object_with_required_synthetic_properties` | `fn object_with_required_synthetic_properties(` |
| `src/import/json_schema/lower/object/mod.rs` | 2548 | `is_ref_string_open_object_branch` | `fn is_ref_string_open_object_branch(schema: &Schema) -> bool {` |
| `src/import/json_schema/lower/object/mod.rs` | 2584 | `all_of_has_explicit_object_only_type` | `fn all_of_has_explicit_object_only_type(branches: &[Schema]) -> bool {` |
| `src/import/json_schema/lower/object/mod.rs` | 2588 | `schema_has_explicit_object_only_type` | `fn schema_has_explicit_object_only_type(schema: &Schema) -> bool {` |
| `src/import/json_schema/lower/object/mod.rs` | 2602 | `is_plain_array_branch` | `fn is_plain_array_branch(schema: &Schema) -> bool {` |
| `src/import/json_schema/lower/object/mod.rs` | 2627 | `is_string_schema` | `fn is_string_schema(schema: &Schema) -> bool {` |
| `src/import/json_schema/lower/object/mod.rs` | 2652 | `property_matches_pattern` | `fn property_matches_pattern(pattern: &str, property_name: &str) -> ImportResult<bool> {` |
| `src/import/json_schema/lower/object/mod.rs` | 2656 | `pattern_schema_for_property` | `fn pattern_schema_for_property(property_schema: &Schema, pattern_schema: &Schema) -> Schema {` |
| `src/import/json_schema/lower/object/mod.rs` | 2673 | `single_numeric_property_type` | `fn single_numeric_property_type(property_schema: &Schema) -> Option<SchemaType> {` |
| `src/import/json_schema/lower/object/mod.rs` | 2684 | `has_non_numeric_assertions` | `fn has_non_numeric_assertions(assertions: &SchemaAssertions) -> bool {` |
| `src/import/json_schema/lower/string/mod.rs` | 19 | `lower_string` | `pub(crate) fn lower_string(&mut self, schema: &StringSchema) -> ImportResult<GrammarExpr> {` |
| `src/import/json_schema/lower/string/mod.rs` | 37 | `lower_inline_bounded_array_string_item_expr` | `pub(crate) fn lower_inline_bounded_array_string_item_expr(` |
| `src/import/json_schema/lower/string/mod.rs` | 83 | `lower_constrained_string_terminal_expr` | `fn lower_constrained_string_terminal_expr(` |
| `src/import/json_schema/lower/string/mod.rs` | 133 | `lower_string_expr` | `fn lower_string_expr(&mut self, schema: &StringSchema) -> ImportResult<GrammarExpr> {` |
| `src/import/json_schema/lower/string/mod.rs` | 157 | `should_split_bounded_string` | `fn should_split_bounded_string(&self, min: usize, max: usize) -> bool {` |
| `src/import/json_schema/lower/string/mod.rs` | 165 | `string_char_exact_ref` | `fn string_char_exact_ref(&mut self, count: usize) -> GrammarExpr {` |
| `src/import/json_schema/lower/string/mod.rs` | 188 | `string_char_upto_ref` | `fn string_char_upto_ref(&mut self, max: usize) -> GrammarExpr {` |
| `src/import/json_schema/lower/string/mod.rs` | 211 | `string_char_upto_close_ref` | `fn string_char_upto_close_ref(&mut self, max: usize) -> GrammarExpr {` |
| `src/import/json_schema/lower/string/mod.rs` | 225 | `string_char_exact_open_ref` | `fn string_char_exact_open_ref(&mut self, count: usize) -> GrammarExpr {` |
| `src/import/json_schema/lower/string/mod.rs` | 235 | `string_char_upto_wrapped_ref` | `fn string_char_upto_wrapped_ref(&mut self, max: usize) -> GrammarExpr {` |
| `src/import/json_schema/lower/string/mod.rs` | 242 | `split_string_exact_expr` | `fn split_string_exact_expr(&mut self, count: usize) -> GrammarExpr {` |
| `src/import/json_schema/lower/string/mod.rs` | 261 | `split_string_upto_close_expr` | `fn split_string_upto_close_expr(&mut self, max: usize) -> GrammarExpr {` |
| `src/import/json_schema/lower/string/mod.rs` | 289 | `lower_split_bounded_string` | `fn lower_split_bounded_string(&mut self, min: usize, max: usize) -> GrammarExpr {` |
| `src/import/json_schema/lower/string/mod.rs` | 303 | `split_bounded_string_terminal_expr` | `fn split_bounded_string_terminal_expr(&mut self, min: usize, max: usize) -> GrammarExpr {` |
| `src/import/json_schema/lower/string/mod.rs` | 344 | `lower_string_literal` | `pub(crate) fn lower_string_literal(&mut self, text: &str) -> GrammarExpr {` |
| `src/import/json_schema/lower/string/mod.rs` | 352 | `lower_literal_key_colon` | `pub(crate) fn lower_literal_key_colon(&mut self, key: &str) -> GrammarExpr {` |
| `src/import/json_schema/lower/string/mod.rs` | 356 | `lower_literal_key_colon_with_prefix` | `pub(crate) fn lower_literal_key_colon_with_prefix(` |
| `src/import/json_schema/lower/string/mod.rs` | 369 | `lower_pattern_key_colon_expr` | `fn lower_pattern_key_colon_expr(&mut self, pattern: &str) -> ImportResult<GrammarExpr> {` |
| `src/import/json_schema/lower/string/mod.rs` | 373 | `pattern_key_colon_full_language` | `fn pattern_key_colon_full_language(&mut self, pattern: &str) -> ImportResult<GrammarExpr> {` |
| `src/import/json_schema/lower/string/mod.rs` | 381 | `lower_pattern_key_colon_terminal` | `pub(crate) fn lower_pattern_key_colon_terminal(` |
| `src/import/json_schema/lower/string/mod.rs` | 408 | `pattern_overlapping_literal_keys` | `fn pattern_overlapping_literal_keys(&mut self, pattern: &str) -> ImportResult<Vec<String>> {` |
| `src/import/json_schema/lower/string/mod.rs` | 432 | `pattern_local_overlapping_literal_keys` | `fn pattern_local_overlapping_literal_keys(` |
| `src/import/json_schema/lower/string/mod.rs` | 453 | `shared_pattern_overlap_literal_rule` | `fn shared_pattern_overlap_literal_rule(&mut self, pattern: &str) -> ImportResult<Option<GrammarExpr>> {` |
| `src/import/json_schema/lower/string/mod.rs` | 475 | `lower_pattern_key_colon_appearance` | `pub(crate) fn lower_pattern_key_colon_appearance(` |
| `src/import/json_schema/lower/string/mod.rs` | 510 | `lower_additional_key_colon` | `pub(crate) fn lower_additional_key_colon(` |
| `src/import/json_schema/lower/string/mod.rs` | 552 | `use_shared_additional_key_colon` | `fn use_shared_additional_key_colon(&self) -> bool {` |
| `src/import/json_schema/lower/string/mod.rs` | 556 | `lower_additional_key_colon_expanded_addback` | `fn lower_additional_key_colon_expanded_addback(` |
| `src/import/json_schema/lower/string/mod.rs` | 590 | `lower_additional_key_colon_literal_only` | `fn lower_additional_key_colon_literal_only(` |
| `src/import/json_schema/lower/string/mod.rs` | 613 | `shared_additional_excluded_key_colon` | `fn shared_additional_excluded_key_colon(&mut self) -> ImportResult<GrammarExpr> {` |
| `src/import/json_schema/lower/string/mod.rs` | 634 | `shared_additional_key_colon_base` | `fn shared_additional_key_colon_base(&mut self) -> ImportResult<GrammarExpr> {` |
| `src/import/json_schema/lower/string/mod.rs` | 673 | `lower_pattern_key_colon_addback` | `fn lower_pattern_key_colon_addback(` |
| `src/import/json_schema/lower/string/mod.rs` | 714 | `string_body_for_length` | `fn string_body_for_length(&self, min: usize, max: Option<usize>) -> GrammarExpr {` |
| `src/import/json_schema/lower/string/mod.rs` | 728 | `repeat_exact_string_char` | `fn repeat_exact_string_char(&self, count: usize) -> GrammarExpr {` |
| `src/import/json_schema/lower/string/mod.rs` | 756 | `string_pattern_as_body_regex` | `fn string_pattern_as_body_regex(pattern: &str) -> ImportResult<String> {` |
| `src/import/json_schema/lower/string/mod.rs` | 764 | `preprocess_ascii_shorthand` | `fn preprocess_ascii_shorthand(pattern: &str) -> String {` |
| `src/import/json_schema/lower/string/mod.rs` | 806 | `string_pattern_hir_as_body_regex` | `fn string_pattern_hir_as_body_regex(hir: &Hir) -> ImportResult<String> {` |
| `src/import/json_schema/lower/string/mod.rs` | 847 | `string_pattern_branch_as_body_regex` | `fn string_pattern_branch_as_body_regex(hir: Hir) -> ImportResult<String> {` |
| `src/import/json_schema/lower/string/mod.rs` | 856 | `lower_string_pattern_branch_parts` | `fn lower_string_pattern_branch_parts(hir: Hir) -> ImportResult<(String, bool, bool)> {` |
| `src/import/json_schema/lower/string/mod.rs` | 862 | `wrap_lowered_string_pattern_branch` | `fn wrap_lowered_string_pattern_branch(lowered: &str, anchored_start: bool, anchored_end: bool) -> String {` |
| `src/import/json_schema/lower/string/mod.rs` | 872 | `quoted_string_body_regex` | `fn quoted_string_body_regex(body_regex: &str) -> String {` |
| `src/import/json_schema/lower/string/mod.rs` | 879 | `recognized_string_format_body_regex` | `fn recognized_string_format_body_regex(format: Option<&str>) -> Option<&'static str> {` |
| `src/import/json_schema/lower/string/mod.rs` | 905 | `pattern_key_colon_regex` | `fn pattern_key_colon_regex(pattern: &str) -> ImportResult<String> {` |
| `src/import/json_schema/lower/string/mod.rs` | 910 | `strip_outer_captures` | `fn strip_outer_captures(mut hir: Hir) -> Hir {` |
| `src/import/json_schema/lower/string/mod.rs` | 919 | `strip_outer_start_anchor` | `fn strip_outer_start_anchor(hir: Hir) -> Option<Hir> {` |
| `src/import/json_schema/lower/string/mod.rs` | 934 | `strip_outer_end_anchor` | `fn strip_outer_end_anchor(hir: Hir) -> Option<Hir> {` |
| `src/import/json_schema/lower/string/mod.rs` | 949 | `strip_outer_anchors` | `fn strip_outer_anchors(hir: Hir) -> (Hir, bool, bool) {` |
| `src/import/json_schema/lower/string/mod.rs` | 988 | `is_start_look` | `fn is_start_look(look: Look) -> bool {` |
| `src/import/json_schema/lower/string/mod.rs` | 992 | `is_end_look` | `fn is_end_look(look: Look) -> bool {` |
| `src/import/json_schema/lower/string/mod.rs` | 996 | `lower_decoded_regex_hir_to_json_body_regex` | `fn lower_decoded_regex_hir_to_json_body_regex(hir: &Hir) -> ImportResult<String> {` |
| `src/import/json_schema/lower/string/mod.rs` | 1035 | `lower_decoded_repetition_to_json_body_regex` | `fn lower_decoded_repetition_to_json_body_regex(repetition: &Repetition) -> ImportResult<String> {` |
| `src/import/json_schema/lower/string/mod.rs` | 1054 | `lower_decoded_class_to_json_body_regex` | `fn lower_decoded_class_to_json_body_regex(class: &Class) -> String {` |
| `src/import/json_schema/lower/string/mod.rs` | 1123 | `utf8_sequence_to_regex_string` | `fn utf8_sequence_to_regex_string(seq: &regex_syntax::utf8::Utf8Sequence) -> String {` |
| `src/import/json_schema/lower/string/mod.rs` | 1135 | `unicode_range_to_utf8_regex_string` | `fn unicode_range_to_utf8_regex_string(start: char, end: char) -> String {` |
| `src/import/json_schema/lower/string/mod.rs` | 1146 | `is_unicode_decimal_digit_class` | `fn is_unicode_decimal_digit_class(class: &Class) -> bool {` |
| `src/import/json_schema/lower/string/mod.rs` | 1170 | `is_dot_like_unicode_class` | `fn is_dot_like_unicode_class(class: &Class) -> bool {` |
| `src/import/json_schema/lower/string/mod.rs` | 1182 | `push_safe_raw_char_ranges` | `fn push_safe_raw_char_ranges(start: char, end: char, output: &mut Vec<String>) {` |
| `src/import/json_schema/lower/string/mod.rs` | 1205 | `decoded_class_contains` | `fn decoded_class_contains(class: &Class, ch: char) -> bool {` |
| `src/import/json_schema/lower/string/mod.rs` | 1219 | `class_contains_general_non_ascii_non_whitespace` | `fn class_contains_general_non_ascii_non_whitespace(class: &Class) -> bool {` |
| `src/import/json_schema/lower/string/mod.rs` | 1225 | `regex_char_class_range` | `fn regex_char_class_range(start: char, end: char) -> String {` |
| `src/import/json_schema/lower/string/mod.rs` | 1235 | `escape_regex_class_char` | `fn escape_regex_class_char(ch: char) -> String {` |
| `src/import/json_schema/lower/string/mod.rs` | 1245 | `json_body_char_regex_for_decoded_char` | `fn json_body_char_regex_for_decoded_char(ch: char) -> String {` |
| `src/import/json_schema/lower/string/mod.rs` | 1255 | `json_string_body_char_regex` | `fn json_string_body_char_regex() -> &'static str {` |
| `src/import/json_schema/lower/string/mod.rs` | 1259 | `json_string_body_non_ascii_non_whitespace_regex` | `fn json_string_body_non_ascii_non_whitespace_regex() -> &'static str {` |
| `src/import/json_schema/lower/string/mod.rs` | 1263 | `json_string_body_dot_regex` | `fn json_string_body_dot_regex() -> &'static str {` |
| `src/import/json_schema/lower/string/mod.rs` | 1267 | `is_safe_raw_json_string_char` | `fn is_safe_raw_json_string_char(ch: char) -> bool {` |
| `src/import/json_schema/lower/string/mod.rs` | 1271 | `property_name_matches_pattern` | `pub(crate) fn property_name_matches_pattern(pattern: &str, property_name: &str) -> ImportResult<bool> {` |
| `src/import/json_schema/lower/string/mod.rs` | 1278 | `is_regex_compile_limit_error` | `fn is_regex_compile_limit_error(error: &SchemaImportError) -> bool {` |
| `src/import/json_schema/lower/string/mod.rs` | 1282 | `string_value_satisfies_schema` | `pub(crate) fn string_value_satisfies_schema(` |
| `src/import/json_schema/lower/string/mod.rs` | 1317 | `preprocess_ascii_shorthand_rewrites_generic_word_shorthand` | `fn preprocess_ascii_shorthand_rewrites_generic_word_shorthand() {` |
| `src/import/json_schema/lower/string/mod.rs` | 1323 | `preprocess_ascii_shorthand_preserves_escaped_word_shorthand` | `fn preprocess_ascii_shorthand_preserves_escaped_word_shorthand() {` |
| `src/import/json_schema/lower/string/mod.rs` | 1328 | `lowered_bounded_free_text_pattern_rejects_leading_space_slash` | `fn lowered_bounded_free_text_pattern_rejects_leading_space_slash() {` |
| `src/import/json_schema/lower/string/mod.rs` | 1337 | `lowered_optional_decimal_pattern_rejects_backslash_digit_string` | `fn lowered_optional_decimal_pattern_rejects_backslash_digit_string() {` |
| `src/import/json_schema/mod.rs` | 55 | `schema_to_named_grammar` | `pub fn schema_to_named_grammar(schema: &Value) -> Result<NamedGrammar, GlrMaskError> {` |
| `src/import/json_schema/mod.rs` | 62 | `simplify_grammar_enabled` | `pub(crate) fn simplify_grammar_enabled() -> bool {` |
| `src/import/json_schema/mod.rs` | 67 | `lower_exact_subtractions_enabled` | `pub(crate) fn lower_exact_subtractions_enabled() -> bool {` |
| `src/import/json_schema/mod.rs` | 72 | `promote_literal_choices_enabled` | `pub(crate) fn promote_literal_choices_enabled() -> bool {` |
| `src/import/json_schema/normalize/combinators.rs` | 25 | `lower_any_of` | `pub(crate) fn lower_any_of(` |
| `src/import/json_schema/normalize/combinators.rs` | 86 | `try_merge_single_object_any_of_with_siblings` | `fn try_merge_single_object_any_of_with_siblings(` |
| `src/import/json_schema/normalize/combinators.rs` | 110 | `lower_one_of` | `pub(crate) fn lower_one_of(&mut self, assertions: &SchemaAssertions) -> ImportResult<GrammarExpr> {` |
| `src/import/json_schema/normalize/combinators.rs` | 120 | `lower_all_of` | `pub(crate) fn lower_all_of(&mut self, assertions: &SchemaAssertions) -> ImportResult<GrammarExpr> {` |
| `src/import/json_schema/normalize/combinators.rs` | 217 | `try_lower_single_ref_with_object_siblings` | `fn try_lower_single_ref_with_object_siblings(` |
| `src/import/json_schema/normalize/combinators.rs` | 265 | `inline_all_of_refs` | `fn inline_all_of_refs(&self, branches: &[Schema]) -> ImportResult<Vec<Schema>> {` |
| `src/import/json_schema/normalize/combinators.rs` | 272 | `inline_all_of_refs_for_any_of_factoring` | `fn inline_all_of_refs_for_any_of_factoring(` |
| `src/import/json_schema/normalize/combinators.rs` | 286 | `schema_transitively_refs_pointer` | `fn schema_transitively_refs_pointer(` |
| `src/import/json_schema/normalize/combinators.rs` | 356 | `inline_all_of_ref_target` | `fn inline_all_of_ref_target(&self, pointer: &str, fallback: &Schema) -> ImportResult<Schema> {` |
| `src/import/json_schema/normalize/combinators.rs` | 385 | `try_inline_object_like_all_of_target` | `fn try_inline_object_like_all_of_target(&self, target: &Schema) -> ImportResult<Option<Schema>> {` |
| `src/import/json_schema/normalize/combinators.rs` | 400 | `try_rewrite_all_of_object_choice_target` | `fn try_rewrite_all_of_object_choice_target(&self, target: &Schema) -> ImportResult<Option<Schema>> {` |
| `src/import/json_schema/normalize/combinators.rs` | 449 | `inline_refs_in_all_of_branch` | `fn inline_refs_in_all_of_branch(&self, branch: &Schema) -> ImportResult<Schema> {` |
| `src/import/json_schema/normalize/combinators.rs` | 468 | `try_merge_all_of_single_ref_object_branches` | `fn try_merge_all_of_single_ref_object_branches(` |
| `src/import/json_schema/normalize/combinators.rs` | 505 | `drop_subsumed_open_object_any_of_branches` | `fn drop_subsumed_open_object_any_of_branches(` |
| `src/import/json_schema/normalize/combinators.rs` | 555 | `object_branch_resolved` | `fn object_branch_resolved<'schema>(` |
| `src/import/json_schema/normalize/combinators.rs` | 565 | `object_schema_subsumes` | `fn object_schema_subsumes(` |
| `src/import/json_schema/normalize/combinators.rs` | 612 | `schema_subsumes` | `fn schema_subsumes(` |
| `src/import/json_schema/normalize/combinators.rs` | 691 | `all_of_intersection_terminal_safe` | `fn all_of_intersection_terminal_safe(expr: &GrammarExpr) -> bool {` |
| `src/import/json_schema/normalize/combinators.rs` | 727 | `explicit_all_of_type_intersection` | `fn explicit_all_of_type_intersection(branches: &[Schema]) -> Option<BTreeSet<SchemaType>> {` |
| `src/import/json_schema/normalize/combinators.rs` | 748 | `untyped_single_family_assertion` | `fn untyped_single_family_assertion(schema: &Schema) -> Option<SchemaType> {` |
| `src/import/json_schema/normalize/combinators.rs` | 781 | `family_overlaps_types` | `fn family_overlaps_types(family: SchemaType, types: &BTreeSet<SchemaType>) -> bool {` |
| `src/import/json_schema/normalize/combinators.rs` | 790 | `drop_vacuous_untyped_family_branches` | `fn drop_vacuous_untyped_family_branches(branches: Vec<Schema>) -> Option<Vec<Schema>> {` |
| `src/import/json_schema/normalize/combinators.rs` | 809 | `flatten_pure_all_of_branches` | `fn flatten_pure_all_of_branches(branches: Vec<Schema>) -> Vec<Schema> {` |
| `src/import/json_schema/normalize/combinators.rs` | 825 | `collapse_pure_single_choice_branches` | `fn collapse_pure_single_choice_branches(branches: Vec<Schema>) -> Vec<Schema> {` |
| `src/import/json_schema/normalize/combinators.rs` | 838 | `try_factor_required_property_any_of` | `fn try_factor_required_property_any_of(` |
| `src/import/json_schema/normalize/combinators.rs` | 884 | `try_factor_closed_object_variant_any_of` | `fn try_factor_closed_object_variant_any_of(` |
| `src/import/json_schema/normalize/combinators.rs` | 989 | `try_factor_mutually_exclusive_property_not_any_of` | `fn try_factor_mutually_exclusive_property_not_any_of(` |
| `src/import/json_schema/normalize/combinators.rs` | 1041 | `mutually_exclusive_property_not_branch` | `fn mutually_exclusive_property_not_branch(schema: &Schema) -> Option<(&PropertySchema, String)> {` |
| `src/import/json_schema/normalize/combinators.rs` | 1078 | `single_required_object_not_name` | `fn single_required_object_not_name(schema: &Schema) -> Option<&str> {` |
| `src/import/json_schema/normalize/combinators.rs` | 1110 | `single_required_object_branch_name` | `fn single_required_object_branch_name(schema: &Schema) -> Option<&str> {` |
| `src/import/json_schema/normalize/combinators.rs` | 1143 | `closed_object_variant_branch` | `fn closed_object_variant_branch(schema: &Schema) -> Option<&ObjectSchema> {` |
| `src/import/json_schema/normalize/combinators.rs` | 1176 | `open_object_any_of_covers_json_object` | `pub(crate) fn open_object_any_of_covers_json_object(branches: &[Schema]) -> bool {` |
| `src/import/json_schema/normalize/combinators.rs` | 1211 | `object_branch` | `fn object_branch(schema: &Schema) -> Option<&ObjectSchema> {` |
| `src/import/json_schema/normalize/combinators.rs` | 1236 | `property_schema_by_name` | `fn property_schema_by_name<'a>(object: &'a ObjectSchema, name: &str) -> Option<&'a Schema> {` |
| `src/import/json_schema/normalize/combinators.rs` | 1244 | `schema_subsumption_key` | `fn schema_subsumption_key(schema: &Schema) -> ImportResult<String> {` |
| `src/import/json_schema/normalize/combinators.rs` | 1251 | `pure_any_of_assertions` | `fn pure_any_of_assertions(assertions: &SchemaAssertions) -> bool {` |
| `src/import/json_schema/normalize/combinators.rs` | 1265 | `broad_string_assertions` | `fn broad_string_assertions(assertions: &SchemaAssertions) -> Option<&super::super::schema::StringSchema> {` |
| `src/import/json_schema/normalize/combinators.rs` | 1288 | `string_literal_values` | `fn string_literal_values(assertions: &SchemaAssertions) -> Option<Vec<&serde_json::Value>> {` |
| `src/import/json_schema/normalize/combinators.rs` | 1311 | `schemas_shape_equivalent` | `fn schemas_shape_equivalent(left: &Schema, right: &Schema) -> bool {` |
| `src/import/json_schema/normalize/combinators.rs` | 1332 | `option_schemas_shape_equivalent` | `fn option_schemas_shape_equivalent(left: Option<&Schema>, right: Option<&Schema>) -> bool {` |
| `src/import/json_schema/normalize/combinators.rs` | 1340 | `option_objects_shape_equivalent` | `fn option_objects_shape_equivalent(left: Option<&ObjectSchema>, right: Option<&ObjectSchema>) -> bool {` |
| `src/import/json_schema/normalize/combinators.rs` | 1348 | `object_schemas_shape_equivalent` | `fn object_schemas_shape_equivalent(left: &ObjectSchema, right: &ObjectSchema) -> bool {` |
| `src/import/json_schema/normalize/combinators.rs` | 1371 | `additional_properties_shape_equivalent` | `fn additional_properties_shape_equivalent(` |
| `src/import/json_schema/normalize/combinators.rs` | 1385 | `option_arrays_shape_equivalent` | `fn option_arrays_shape_equivalent(` |
| `src/import/json_schema/normalize/combinators.rs` | 1401 | `option_strings_shape_equivalent` | `fn option_strings_shape_equivalent(` |
| `src/import/json_schema/normalize/combinators.rs` | 1417 | `option_numbers_shape_equivalent` | `fn option_numbers_shape_equivalent(` |
| `src/import/json_schema/normalize/combinators.rs` | 1435 | `schema_slices_shape_equivalent` | `fn schema_slices_shape_equivalent(left: &[Schema], right: &[Schema]) -> bool {` |
| `src/import/json_schema/normalize/combinators.rs` | 1443 | `sibling_assertion_schema` | `fn sibling_assertion_schema(assertions: &SchemaAssertions) -> Option<Schema> {` |
| `src/import/json_schema/normalize/combinators.rs` | 1452 | `branch_with_siblings` | `fn branch_with_siblings(branch: Schema, siblings: Option<Schema>) -> Schema {` |
| `src/import/json_schema/normalize/combinators.rs` | 1464 | `push_object_only_type_into_branch` | `fn push_object_only_type_into_branch(branch: &Schema) -> Option<Schema> {` |
| `src/import/json_schema/normalize/combinators.rs` | 1489 | `schema_contains_ref` | `fn schema_contains_ref(schema: &Schema) -> bool {` |
| `src/import/json_schema/normalize/combinators.rs` | 1502 | `schema_has_explicit_object_only_type` | `fn schema_has_explicit_object_only_type(schema: &Schema) -> bool {` |
| `src/import/json_schema/normalize/combinators.rs` | 1512 | `try_merge_all_of_objects` | `pub(crate) fn try_merge_all_of_objects(branches: &[Schema]) -> Option<ObjectSchema> {` |
| `src/import/json_schema/normalize/combinators.rs` | 1521 | `plain_object_schema` | `fn plain_object_schema(schema: &Schema) -> Option<&ObjectSchema> {` |
| `src/import/json_schema/normalize/combinators.rs` | 1542 | `ChoiceKind` | `enum ChoiceKind {` |
| `src/import/json_schema/normalize/combinators.rs` | 1547 | `pure_choice_branch` | `fn pure_choice_branch(schema: &Schema) -> Option<(ChoiceKind, &[Schema])> {` |
| `src/import/json_schema/normalize/combinators.rs` | 1570 | `distribute_all_of_over_single_object_choice` | `fn distribute_all_of_over_single_object_choice(` |
| `src/import/json_schema/normalize/combinators.rs` | 1613 | `schema_is_object_like` | `fn schema_is_object_like(schema: &Schema) -> bool {` |
| `src/import/json_schema/normalize/combinators.rs` | 1617 | `merge_all_of_object_like_schema` | `fn merge_all_of_object_like_schema(branches: &[Schema]) -> Option<Schema> {` |
| `src/import/json_schema/normalize/combinators.rs` | 1658 | `object_like_schema` | `fn object_like_schema(schema: &Schema) -> Option<Schema> {` |
| `src/import/json_schema/normalize/combinators.rs` | 1703 | `merge_all_of_array_like_schema` | `fn merge_all_of_array_like_schema(branches: &[Schema]) -> Option<Schema> {` |
| `src/import/json_schema/normalize/combinators.rs` | 1742 | `plain_array_schema` | `fn plain_array_schema(schema: &Schema) -> Option<(&ArraySchema, bool)> {` |
| `src/import/json_schema/normalize/combinators.rs` | 1765 | `array_is_bounds_only` | `fn array_is_bounds_only(array: &ArraySchema) -> bool {` |
| `src/import/json_schema/normalize/combinators.rs` | 1770 | `merge_array_bounds` | `fn merge_array_bounds(left: &mut ArraySchema, right: &ArraySchema) {` |
| `src/import/json_schema/normalize/combinators.rs` | 1780 | `merge_two_objects` | `fn merge_two_objects(left: &ObjectSchema, right: &ObjectSchema) -> ObjectSchema {` |
| `src/import/json_schema/normalize/combinators.rs` | 1810 | `merge_property_schemas` | `fn merge_property_schemas(left: Schema, right: Schema) -> Schema {` |
| `src/import/json_schema/normalize/combinators.rs` | 1820 | `is_vacuous_json_value_schema` | `fn is_vacuous_json_value_schema(schema: &Schema) -> bool {` |
| `src/import/json_schema/normalize/combinators.rs` | 1850 | `is_vacuous_object_schema` | `fn is_vacuous_object_schema(schema: &Schema) -> bool {` |
| `src/import/json_schema/normalize/combinators.rs` | 1874 | `merge_additional_properties` | `fn merge_additional_properties(` |
| `src/import/json_schema/normalize/combinators.rs` | 1894 | `all_of_schema` | `pub(crate) fn all_of_schema(left: Schema, right: Schema) -> Schema {` |
| `src/import/json_schema/options.rs` | 7 | `JsonSchemaConfig` | `pub(crate) struct JsonSchemaConfig {` |
| `src/import/json_schema/options.rs` | 16 | `QuoteMerge` | `pub(crate) struct QuoteMerge {` |
| `src/import/json_schema/options.rs` | 22 | `MergeFamily` | `pub(crate) struct MergeFamily {` |
| `src/import/json_schema/options.rs` | 29 | `ObjectMergeConfig` | `pub(crate) struct ObjectMergeConfig {` |
| `src/import/json_schema/options.rs` | 35 | `default` | `fn default() -> Self {` |
| `src/import/json_schema/options.rs` | 64 | `from_env` | `pub(crate) fn from_env() -> Self {` |
| `src/import/json_schema/options.rs` | 117 | `read_usize` | `fn read_usize(name: &str) -> Option<usize> {` |
| `src/import/json_schema/options.rs` | 121 | `read_quote_merge` | `fn read_quote_merge(open_name: &str, close_name: &str, default: QuoteMerge) -> QuoteMerge {` |
| `src/import/json_schema/options.rs` | 128 | `read_bool` | `fn read_bool(name: &str) -> Option<bool> {` |
| `src/import/json_schema/options.rs` | 140 | `simplify_grammar_enabled` | `pub(crate) fn simplify_grammar_enabled() -> bool {` |
| `src/import/json_schema/options.rs` | 150 | `lower_exact_subtractions_enabled` | `pub(crate) fn lower_exact_subtractions_enabled() -> bool {` |
| `src/import/json_schema/options.rs` | 167 | `promote_literal_choices_enabled` | `pub(crate) fn promote_literal_choices_enabled() -> bool {` |
| `src/import/json_schema/schema/array.rs` | 5 | `ArraySchema` | `pub(crate) struct ArraySchema {` |
| `src/import/json_schema/schema/array.rs` | 13 | `default` | `fn default() -> Self {` |
| `src/import/json_schema/schema/assertions.rs` | 12 | `SchemaAssertions` | `pub(crate) struct SchemaAssertions {` |
| `src/import/json_schema/schema/assertions.rs` | 27 | `is_empty` | `pub(crate) fn is_empty(&self) -> bool {` |
| `src/import/json_schema/schema/assertions.rs` | 41 | `has_value_assertions_without_combinators` | `pub(crate) fn has_value_assertions_without_combinators(&self) -> bool {` |
| `src/import/json_schema/schema/assertions.rs` | 51 | `clone_without_combinators` | `pub(crate) fn clone_without_combinators(&self) -> Self {` |
| `src/import/json_schema/schema/document.rs` | 10 | `SchemaDocument` | `pub(crate) struct SchemaDocument {` |
| `src/import/json_schema/schema/document.rs` | 18 | `SchemaDefinition` | `pub(crate) struct SchemaDefinition {` |
| `src/import/json_schema/schema/mod.rs` | 21 | `Schema` | `pub(crate) struct Schema {` |
| `src/import/json_schema/schema/mod.rs` | 28 | `SchemaKind` | `pub(crate) enum SchemaKind {` |
| `src/import/json_schema/schema/mod.rs` | 40 | `any` | `pub(crate) fn any(location: impl Into<String>) -> Self {` |
| `src/import/json_schema/schema/mod.rs` | 44 | `never` | `pub(crate) fn never(location: impl Into<String>) -> Self {` |
| `src/import/json_schema/schema/mod.rs` | 48 | `assertions` | `pub(crate) fn assertions(location: impl Into<String>, assertions: SchemaAssertions) -> Self {` |
| `src/import/json_schema/schema/object.rs` | 7 | `ObjectSchema` | `pub(crate) struct ObjectSchema {` |
| `src/import/json_schema/schema/object.rs` | 17 | `default` | `fn default() -> Self {` |
| `src/import/json_schema/schema/object.rs` | 30 | `PropertySchema` | `pub(crate) struct PropertySchema {` |
| `src/import/json_schema/schema/object.rs` | 36 | `PatternPropertySchema` | `pub(crate) struct PatternPropertySchema {` |
| `src/import/json_schema/schema/object.rs` | 42 | `AdditionalProperties` | `pub(crate) enum AdditionalProperties {` |
| `src/import/json_schema/schema/scalar.rs` | 3 | `SchemaType` | `pub(crate) enum SchemaType {` |
| `src/import/json_schema/schema/scalar.rs` | 19 | `StringSchema` | `pub(crate) struct StringSchema {` |
| `src/import/json_schema/schema/scalar.rs` | 32 | `NumberSchema` | `pub(crate) struct NumberSchema {` |
| `src/import/json_schema/tests/mod.rs` | 16 | `EnvVarGuard` | `struct EnvVarGuard {` |
| `src/import/json_schema/tests/mod.rs` | 22 | `set` | `fn set(key: &'static str, value: &str) -> Self {` |
| `src/import/json_schema/tests/mod.rs` | 30 | `unset` | `fn unset(key: &'static str) -> Self {` |
| `src/import/json_schema/tests/mod.rs` | 40 | `drop` | `fn drop(&mut self) {` |
| `src/import/json_schema/tests/mod.rs` | 52 | `start_expr` | `fn start_expr(grammar: &NamedGrammar) -> &GrammarExpr {` |
| `src/import/json_schema/tests/mod.rs` | 62 | `exact_subtraction_lowering_env_var_defaults_true_and_accepts_falsey_values` | `fn exact_subtraction_lowering_env_var_defaults_true_and_accepts_falsey_values() {` |
| `src/import/json_schema/tests/mod.rs` | 77 | `contains_separated_sequence` | `fn contains_separated_sequence(expr: &GrammarExpr) -> bool {` |
| `src/import/json_schema/tests/mod.rs` | 105 | `contains_expr_nfa` | `fn contains_expr_nfa(expr: &GrammarExpr) -> bool {` |
| `src/import/json_schema/tests/mod.rs` | 133 | `count_rules_with_prefix` | `fn count_rules_with_prefix(grammar: &NamedGrammar, prefix: &str) -> usize {` |
| `src/import/json_schema/tests/mod.rs` | 137 | `byte_vocab` | `fn byte_vocab() -> Vocab {` |
| `src/import/json_schema/tests/mod.rs` | 145 | `schema_accepts_bytes` | `fn schema_accepts_bytes(schema: &serde_json::Value, input: &[u8]) -> bool {` |
| `src/import/json_schema/tests/mod.rs` | 153 | `parser_path_count_after_bytes` | `fn parser_path_count_after_bytes(schema: &serde_json::Value, input: &[u8], limit: usize) -> usize {` |
| `src/import/json_schema/tests/mod.rs` | 163 | `contains_exclude` | `fn contains_exclude(expr: &GrammarExpr) -> bool {` |
| `src/import/json_schema/tests/mod.rs` | 187 | `contains_ref_with_prefix` | `fn contains_ref_with_prefix(expr: &GrammarExpr, prefix: &str) -> bool {` |
| `src/import/json_schema/tests/mod.rs` | 218 | `find_all_pop1_stackshifts` | `fn find_all_pop1_stackshifts(table: &GLRTable) -> Option<(u32, u32, Action)> {` |
| `src/import/json_schema/tests/mod.rs` | 235 | `recursive_array_additional_properties_schema_does_not_reproduce_all_pop1_stackshifts` | `fn recursive_array_additional_properties_schema_does_not_reproduce_all_pop1_stackshifts() {` |
| `src/import/json_schema/tests/mod.rs` | 271 | `contains_intersect` | `fn contains_intersect(expr: &GrammarExpr) -> bool {` |
| `src/import/json_schema/tests/mod.rs` | 295 | `contains_intersect_with_separated_sequence` | `fn contains_intersect_with_separated_sequence(expr: &GrammarExpr) -> bool {` |
| `src/import/json_schema/tests/mod.rs` | 332 | `contains_ref_named` | `fn contains_ref_named(expr: &GrammarExpr, name: &str) -> bool {` |
| `src/import/json_schema/tests/mod.rs` | 363 | `contains_literal_bytes` | `fn contains_literal_bytes(expr: &GrammarExpr, bytes: &[u8]) -> bool {` |
| `src/import/json_schema/tests/mod.rs` | 398 | `closed_object_lowers_to_prefix_chain_body` | `fn closed_object_lowers_to_prefix_chain_body() {` |
| `src/import/json_schema/tests/mod.rs` | 418 | `large_optional_closed_object_uses_expr_nfa_body` | `fn large_optional_closed_object_uses_expr_nfa_body() {` |
| `src/import/json_schema/tests/mod.rs` | 438 | `required_prefix_open_object_uses_pair_loop_body` | `fn required_prefix_open_object_uses_pair_loop_body() {` |
| `src/import/json_schema/tests/mod.rs` | 465 | `open_additional_map_min_properties_requires_dynamic_pair` | `fn open_additional_map_min_properties_requires_dynamic_pair() {` |
| `src/import/json_schema/tests/mod.rs` | 483 | `closed_fixed_object_min_properties_requires_one_optional_after_required` | `fn closed_fixed_object_min_properties_requires_one_optional_after_required() {` |
| `src/import/json_schema/tests/mod.rs` | 503 | `closed_fixed_object_min_max_properties_exactly_one_optional` | `fn closed_fixed_object_min_max_properties_exactly_one_optional() {` |
| `src/import/json_schema/tests/mod.rs` | 521 | `closed_fixed_object_max_properties_caps_optional_after_required` | `fn closed_fixed_object_max_properties_caps_optional_after_required() {` |
| `src/import/json_schema/tests/mod.rs` | 540 | `open_additional_map_max_properties_emits_bounded_dynamic_body` | `fn open_additional_map_max_properties_emits_bounded_dynamic_body() {` |
| `src/import/json_schema/tests/mod.rs` | 554 | `required_property_covered_by_pattern_properties_is_synthesized` | `fn required_property_covered_by_pattern_properties_is_synthesized() {` |
| `src/import/json_schema/tests/mod.rs` | 571 | `required_property_matching_multiple_patterns_applies_all_pattern_schemas` | `fn required_property_matching_multiple_patterns_applies_all_pattern_schemas() {` |
| `src/import/json_schema/tests/mod.rs` | 590 | `required_property_not_covered_by_closed_object_lowers_to_empty_language` | `fn required_property_not_covered_by_closed_object_lowers_to_empty_language() {` |
| `src/import/json_schema/tests/mod.rs` | 607 | `fixed_property_still_intersects_matching_pattern_property` | `fn fixed_property_still_intersects_matching_pattern_property() {` |
| `src/import/json_schema/tests/mod.rs` | 628 | `open_no_pattern_object_lowers_to_expr_nfa_body` | `fn open_no_pattern_object_lowers_to_expr_nfa_body() {` |
| `src/import/json_schema/tests/mod.rs` | 647 | `large_optional_open_object_uses_fused_prefix_chain_rules` | `fn large_optional_open_object_uses_fused_prefix_chain_rules() {` |
| `src/import/json_schema/tests/mod.rs` | 668 | `large_optional_open_object_allow_any_scalars_uses_expr_nfa_body` | `fn large_optional_open_object_allow_any_scalars_uses_expr_nfa_body() {` |
| `src/import/json_schema/tests/mod.rs` | 687 | `large_optional_open_object_allow_any_object_valued_at_16_uses_expr_nfa_body` | `fn large_optional_open_object_allow_any_object_valued_at_16_uses_expr_nfa_body() {` |
| `src/import/json_schema/tests/mod.rs` | 715 | `large_optional_open_object_allow_any_object_valued_at_32_uses_expr_nfa_body` | `fn large_optional_open_object_allow_any_object_valued_at_32_uses_expr_nfa_body() {` |
| `src/import/json_schema/tests/mod.rs` | 743 | `large_required_open_object_does_not_use_fused_prefix_chain_rules` | `fn large_required_open_object_does_not_use_fused_prefix_chain_rules() {` |
| `src/import/json_schema/tests/mod.rs` | 763 | `pattern_property_object_still_uses_separated_sequence` | `fn pattern_property_object_still_uses_separated_sequence() {` |
| `src/import/json_schema/tests/mod.rs` | 778 | `large_optional_open_object_with_pattern_properties_uses_fused_prefix_chain_rules` | `fn large_optional_open_object_with_pattern_properties_uses_fused_prefix_chain_rules() {` |
| `src/import/json_schema/tests/mod.rs` | 803 | `allof_drops_vacuous_untyped_object_branch_for_typed_property` | `fn allof_drops_vacuous_untyped_object_branch_for_typed_property() {` |
| `src/import/json_schema/tests/mod.rs` | 826 | `large_snowplow_like_pattern_property_object_uses_expr_nfa_body` | `fn large_snowplow_like_pattern_property_object_uses_expr_nfa_body() {` |
| `src/import/json_schema/tests/mod.rs` | 852 | `shared_additional_key_colon_terminal_is_emitted_once` | `fn shared_additional_key_colon_terminal_is_emitted_once() {` |
| `src/import/json_schema/tests/mod.rs` | 882 | `additional_properties_factoring_uses_shared_key_colon_terminal` | `fn additional_properties_factoring_uses_shared_key_colon_terminal() {` |
| `src/import/json_schema/tests/mod.rs` | 905 | `huge_shared_additional_exclusion_set_uses_expanded_literal_addback` | `fn huge_shared_additional_exclusion_set_uses_expanded_literal_addback() {` |
| `src/import/json_schema/tests/mod.rs` | 925 | `shared_additional_excluded_key_skips_closed_object_keys` | `fn shared_additional_excluded_key_skips_closed_object_keys() {` |
| `src/import/json_schema/tests/mod.rs` | 961 | `arrays_use_item_schema_and_min_max_items` | `fn arrays_use_item_schema_and_min_max_items() {` |
| `src/import/json_schema/tests/mod.rs` | 976 | `bounded_object_arrays_use_exprnfa_rule` | `fn bounded_object_arrays_use_exprnfa_rule() {` |
| `src/import/json_schema/tests/mod.rs` | 998 | `bounded_pattern_string_arrays_use_terminal_rule` | `fn bounded_pattern_string_arrays_use_terminal_rule() {` |
| `src/import/json_schema/tests/mod.rs` | 1018 | `large_bounded_pattern_string_arrays_do_not_use_terminal_rule` | `fn large_bounded_pattern_string_arrays_do_not_use_terminal_rule() {` |
| `src/import/json_schema/tests/mod.rs` | 1038 | `unbounded_plain_string_arrays_use_terminal_rule` | `fn unbounded_plain_string_arrays_use_terminal_rule() {` |
| `src/import/json_schema/tests/mod.rs` | 1054 | `prefix_items_lower_with_no_tail` | `fn prefix_items_lower_with_no_tail() {` |
| `src/import/json_schema/tests/mod.rs` | 1077 | `legacy_tuple_items_use_additional_items_tail` | `fn legacy_tuple_items_use_additional_items_tail() {` |
| `src/import/json_schema/tests/mod.rs` | 1099 | `plain_items_ignore_additional_items_without_tuple` | `fn plain_items_ignore_additional_items_without_tuple() {` |
| `src/import/json_schema/tests/mod.rs` | 1113 | `map_shaped_min_properties_lowers_as_bounded_pattern_map` | `fn map_shaped_min_properties_lowers_as_bounded_pattern_map() {` |
| `src/import/json_schema/tests/mod.rs` | 1128 | `small_bounded_string_pattern_ignores_length_bounds` | `fn small_bounded_string_pattern_ignores_length_bounds() {` |
| `src/import/json_schema/tests/mod.rs` | 1155 | `large_bounded_string_pattern_ignores_length_bounds` | `fn large_bounded_string_pattern_ignores_length_bounds() {` |
| `src/import/json_schema/tests/mod.rs` | 1184 | `string_pattern_lowers_ascii_digit_subranges` | `fn string_pattern_lowers_ascii_digit_subranges() {` |
| `src/import/json_schema/tests/mod.rs` | 1198 | `terminalized_dot_pattern_lowers_utf8_lead_byte_alternatives` | `fn terminalized_dot_pattern_lowers_utf8_lead_byte_alternatives() {` |
| `src/import/json_schema/tests/mod.rs` | 1219 | `json_string_char_terminal_requires_valid_utf8_sequences` | `fn json_string_char_terminal_requires_valid_utf8_sequences() {` |
| `src/import/json_schema/tests/mod.rs` | 1230 | `medium_bounded_string_uses_split_chunk_rules_by_default` | `fn medium_bounded_string_uses_split_chunk_rules_by_default() {` |
| `src/import/json_schema/tests/mod.rs` | 1257 | `bounded_pattern_map_respects_min_and_max_properties` | `fn bounded_pattern_map_respects_min_and_max_properties() {` |
| `src/import/json_schema/tests/mod.rs` | 1273 | `unsupported_nonredundant_max_properties_broadens` | `fn unsupported_nonredundant_max_properties_broadens() {` |
| `src/import/json_schema/tests/mod.rs` | 1288 | `unsupported_nonredundant_min_properties_broadens` | `fn unsupported_nonredundant_min_properties_broadens() {` |
| `src/import/json_schema/tests/mod.rs` | 1304 | `oversized_pattern_properties_overlap_check_broadens` | `fn oversized_pattern_properties_overlap_check_broadens() {` |
| `src/import/json_schema/tests/mod.rs` | 1331 | `medium_bounded_string_terminalizes_with_env_override` | `fn medium_bounded_string_terminalizes_with_env_override() {` |
| `src/import/json_schema/tests/mod.rs` | 1359 | `moderately_bounded_string_terminalizes_by_default` | `fn moderately_bounded_string_terminalizes_by_default() {` |
| `src/import/json_schema/tests/mod.rs` | 1379 | `split_bounded_string_chunks_do_not_overlap_at_boundary` | `fn split_bounded_string_chunks_do_not_overlap_at_boundary() {` |
| `src/import/json_schema/tests/mod.rs` | 1398 | `very_large_bounded_string_still_uses_split_chunk_rules` | `fn very_large_bounded_string_still_uses_split_chunk_rules() {` |
| `src/import/json_schema/tests/mod.rs` | 1422 | `decoded_string_patterns_are_matched_against_json_string_bodies` | `fn decoded_string_patterns_are_matched_against_json_string_bodies() {` |
| `src/import/json_schema/tests/mod.rs` | 1451 | `uuid_format_lowers_to_constrained_terminal` | `fn uuid_format_lowers_to_constrained_terminal() {` |
| `src/import/json_schema/tests/mod.rs` | 1474 | `date_time_format_lowers_to_constrained_terminal` | `fn date_time_format_lowers_to_constrained_terminal() {` |
| `src/import/json_schema/tests/mod.rs` | 1498 | `date_format_lowers_to_constrained_terminal` | `fn date_format_lowers_to_constrained_terminal() {` |
| `src/import/json_schema/tests/mod.rs` | 1522 | `email_format_lowers_to_constrained_terminal` | `fn email_format_lowers_to_constrained_terminal() {` |
| `src/import/json_schema/tests/mod.rs` | 1545 | `email_format_with_large_max_length_does_not_preserve_length_envelope` | `fn email_format_with_large_max_length_does_not_preserve_length_envelope() {` |
| `src/import/json_schema/tests/mod.rs` | 1570 | `hostname_ipv4_ipv6_formats_lower_to_constrained_terminals` | `fn hostname_ipv4_ipv6_formats_lower_to_constrained_terminals() {` |
| `src/import/json_schema/tests/mod.rs` | 1599 | `uri_format_lowers_to_constrained_terminal` | `fn uri_format_lowers_to_constrained_terminal() {` |
| `src/import/json_schema/tests/mod.rs` | 1622 | `string_pattern_is_intersected_with_format` | `fn string_pattern_is_intersected_with_format() {` |
| `src/import/json_schema/tests/mod.rs` | 1638 | `object_nonterminals_reference_terminalized_key_and_string_patterns` | `fn object_nonterminals_reference_terminalized_key_and_string_patterns() {` |
| `src/import/json_schema/tests/mod.rs` | 1687 | `overlapping_literal_and_pattern_keys_still_lower_with_shared_factoring` | `fn overlapping_literal_and_pattern_keys_still_lower_with_shared_factoring() {` |
| `src/import/json_schema/tests/mod.rs` | 1707 | `json_separators_are_canonical_space_separated` | `fn json_separators_are_canonical_space_separated() {` |
| `src/import/json_schema/tests/mod.rs` | 1724 | `legacy_id_metadata_is_accepted` | `fn legacy_id_metadata_is_accepted() {` |
| `src/import/json_schema/tests/mod.rs` | 1743 | `local_ref_to_property_schema_is_loaded` | `fn local_ref_to_property_schema_is_loaded() {` |
| `src/import/json_schema/tests/mod.rs` | 1757 | `default_object_named_properties_is_not_scanned_for_ref_targets` | `fn default_object_named_properties_is_not_scanned_for_ref_targets() {` |
| `src/import/json_schema/tests/mod.rs` | 1772 | `property_named_definitions_is_not_definition_container` | `fn property_named_definitions_is_not_definition_container() {` |
| `src/import/json_schema/tests/mod.rs` | 1790 | `unknown_format_is_ignored_as_annotation` | `fn unknown_format_is_ignored_as_annotation() {` |
| `src/import/json_schema/tests/mod.rs` | 1801 | `date_time_string_value_satisfaction_filters_invalid_literals` | `fn date_time_string_value_satisfaction_filters_invalid_literals() {` |
| `src/import/json_schema/tests/mod.rs` | 1815 | `date_string_value_satisfaction_filters_invalid_literals` | `fn date_string_value_satisfaction_filters_invalid_literals() {` |
| `src/import/json_schema/tests/mod.rs` | 1829 | `uuid_string_value_satisfaction_filters_invalid_literals` | `fn uuid_string_value_satisfaction_filters_invalid_literals() {` |
| `src/import/json_schema/tests/mod.rs` | 1844 | `email_string_value_satisfaction_filters_invalid_literals` | `fn email_string_value_satisfaction_filters_invalid_literals() {` |
| `src/import/json_schema/tests/mod.rs` | 1857 | `host_string_value_satisfaction_filters_invalid_literals` | `fn host_string_value_satisfaction_filters_invalid_literals() {` |
| `src/import/json_schema/tests/mod.rs` | 1882 | `uri_string_value_satisfaction_filters_invalid_literals` | `fn uri_string_value_satisfaction_filters_invalid_literals() {` |
| `src/import/json_schema/tests/mod.rs` | 1899 | `unknown_metadata_keys_are_ignored` | `fn unknown_metadata_keys_are_ignored() {` |
| `src/import/json_schema/tests/mod.rs` | 1911 | `conditional_keywords_are_ignored_for_broad_lowering` | `fn conditional_keywords_are_ignored_for_broad_lowering() {` |
| `src/import/json_schema/tests/mod.rs` | 1934 | `oneof_lowers_as_choice` | `fn oneof_lowers_as_choice() {` |
| `src/import/json_schema/tests/mod.rs` | 1947 | `oneof_single_ref_wrapper_is_supported` | `fn oneof_single_ref_wrapper_is_supported() {` |
| `src/import/json_schema/tests/mod.rs` | 1962 | `fragment_id_ref_alias_lowers` | `fn fragment_id_ref_alias_lowers() {` |
| `src/import/json_schema/tests/mod.rs` | 1983 | `absolute_root_id_self_ref_lowers` | `fn absolute_root_id_self_ref_lowers() {` |
| `src/import/json_schema/tests/mod.rs` | 1998 | `oneof_ref_and_null_is_supported` | `fn oneof_ref_and_null_is_supported() {` |
| `src/import/json_schema/tests/mod.rs` | 2020 | `oneof_mixed_ref_and_inline_errors` | `fn oneof_mixed_ref_and_inline_errors() {` |
| `src/import/json_schema/tests/mod.rs` | 2042 | `unsupported_not_shape_errors` | `fn unsupported_not_shape_errors() {` |
| `src/import/json_schema/tests/mod.rs` | 2053 | `anyof_property_not_mutual_exclusion_lowers_as_exclusive_group` | `fn anyof_property_not_mutual_exclusion_lowers_as_exclusive_group() {` |
| `src/import/json_schema/tests/mod.rs` | 2084 | `enum_and_const_lower_to_exact_json_literals` | `fn enum_and_const_lower_to_exact_json_literals() {` |
| `src/import/json_schema/tests/mod.rs` | 2095 | `string_const_splits_open_quote_from_literal_body` | `fn string_const_splits_open_quote_from_literal_body() {` |
| `src/import/json_schema/tests/mod.rs` | 2107 | `object_const_uses_json_separator_rules` | `fn object_const_uses_json_separator_rules() {` |
| `src/import/json_schema/tests/mod.rs` | 2123 | `large_string_enum_at_root_uses_raw_regex` | `fn large_string_enum_at_root_uses_raw_regex() {` |
| `src/import/json_schema/tests/mod.rs` | 2135 | `small_string_enum_at_root_uses_factored_suffix_choice` | `fn small_string_enum_at_root_uses_factored_suffix_choice() {` |
| `src/import/json_schema/tests/mod.rs` | 2156 | `snowplow_style_string_enum_uses_factored_suffix_choice` | `fn snowplow_style_string_enum_uses_factored_suffix_choice() {` |
| `src/import/json_schema/tests/mod.rs` | 2181 | `patterned_string_enum_does_not_use_raw_regex_fast_path` | `fn patterned_string_enum_does_not_use_raw_regex_fast_path() {` |
| `src/import/json_schema/tests/mod.rs` | 2197 | `mixed_type_enum_does_not_use_raw_regex_fast_path` | `fn mixed_type_enum_does_not_use_raw_regex_fast_path() {` |
| `src/import/json_schema/tests/mod.rs` | 2206 | `integer_power_of_ten_multiple_lowers_to_regex` | `fn integer_power_of_ten_multiple_lowers_to_regex() {` |
| `src/import/json_schema/tests/mod.rs` | 2215 | `unbounded_integer_multiple_of_three_lowers_broadly` | `fn unbounded_integer_multiple_of_three_lowers_broadly() {` |
| `src/import/json_schema/tests/mod.rs` | 2223 | `lower_bounded_integer_multiple_of_twelve_lowers_to_range` | `fn lower_bounded_integer_multiple_of_twelve_lowers_to_range() {` |
| `src/import/json_schema/tests/mod.rs` | 2234 | `bounded_integer_multiple_of_sixteen_lowers_without_enumerating_large_range` | `fn bounded_integer_multiple_of_sixteen_lowers_without_enumerating_large_range() {` |
| `src/import/json_schema/tests/mod.rs` | 2250 | `non_integer_integer_multiple_of_remains_unsupported` | `fn non_integer_integer_multiple_of_remains_unsupported() {` |
| `src/import/json_schema/tests/mod.rs` | 2257 | `finite_integer_range_multiple_lowers_to_literals` | `fn finite_integer_range_multiple_lowers_to_literals() {` |
| `src/import/json_schema/tests/mod.rs` | 2271 | `bounded_number_lowers_to_range_regex_not_plain_json_number` | `fn bounded_number_lowers_to_range_regex_not_plain_json_number() {` |
| `src/import/json_schema/tests/mod.rs` | 2285 | `large_bounded_integer_lowers_to_range_regex_not_plain_json_integer` | `fn large_bounded_integer_lowers_to_range_regex_not_plain_json_integer() {` |
| `src/import/json_schema/tests/mod.rs` | 2299 | `number_integer_union_uses_json_number_once` | `fn number_integer_union_uses_json_number_once() {` |
| `src/import/json_schema/tests/mod.rs` | 2309 | `anyof_lowers_to_choice` | `fn anyof_lowers_to_choice() {` |
| `src/import/json_schema/tests/mod.rs` | 2323 | `anyof_allows_sibling_assertions` | `fn anyof_allows_sibling_assertions() {` |
| `src/import/json_schema/tests/mod.rs` | 2337 | `anyof_required_property_object_factors_into_single_expr_nfa_body` | `fn anyof_required_property_object_factors_into_single_expr_nfa_body() {` |
| `src/import/json_schema/tests/mod.rs` | 2360 | `anyof_required_sets_with_object_sibling_type_do_not_allow_non_objects` | `fn anyof_required_sets_with_object_sibling_type_do_not_allow_non_objects() {` |
| `src/import/json_schema/tests/mod.rs` | 2385 | `anyof_closed_object_variants_factor_into_single_expr_nfa_body` | `fn anyof_closed_object_variants_factor_into_single_expr_nfa_body() {` |
| `src/import/json_schema/tests/mod.rs` | 2422 | `anyof_required_property_factoring_falls_back_for_nontrivial_branch` | `fn anyof_required_property_factoring_falls_back_for_nontrivial_branch() {` |
| `src/import/json_schema/tests/mod.rs` | 2445 | `anyof_open_objects_with_disjoint_optional_properties_collapses_to_json_object` | `fn anyof_open_objects_with_disjoint_optional_properties_collapses_to_json_object() {` |
| `src/import/json_schema/tests/mod.rs` | 2472 | `unconstrained_object_collapses_to_json_object` | `fn unconstrained_object_collapses_to_json_object() {` |
| `src/import/json_schema/tests/mod.rs` | 2485 | `empty_properties_object_collapses_to_json_object` | `fn empty_properties_object_collapses_to_json_object() {` |
| `src/import/json_schema/tests/mod.rs` | 2499 | `constrained_open_objects_do_not_collapse_to_json_object` | `fn constrained_open_objects_do_not_collapse_to_json_object() {` |
| `src/import/json_schema/tests/mod.rs` | 2523 | `anyof_open_objects_with_shared_optional_property_does_not_collapse_to_json_object` | `fn anyof_open_objects_with_shared_optional_property_does_not_collapse_to_json_object() {` |
| `src/import/json_schema/tests/mod.rs` | 2549 | `anyof_nested_object_allof_refs_factor_into_single_body` | `fn anyof_nested_object_allof_refs_factor_into_single_body() {` |
| `src/import/json_schema/tests/mod.rs` | 2627 | `pattern_map_anyof_open_objects_with_disjoint_optional_properties_collapses_value_to_json_object` | `fn pattern_map_anyof_open_objects_with_disjoint_optional_properties_collapses_value_to_json_object()` |
| `src/import/json_schema/tests/mod.rs` | 2664 | `anyof_closed_object_variant_factoring_falls_back_for_two_variant_properties` | `fn anyof_closed_object_variant_factoring_falls_back_for_two_variant_properties() {` |
| `src/import/json_schema/tests/mod.rs` | 2693 | `anyof_closed_object_variant_factoring_falls_back_for_mismatched_common_schema` | `fn anyof_closed_object_variant_factoring_falls_back_for_mismatched_common_schema() {` |
| `src/import/json_schema/tests/mod.rs` | 2722 | `anyof_closed_object_variants_with_shared_required_prefix_use_exact_variant_nfa` | `fn anyof_closed_object_variants_with_shared_required_prefix_use_exact_variant_nfa() {` |
| `src/import/json_schema/tests/mod.rs` | 2755 | `anyof_untyped_closed_object_variants_keep_non_object_alternatives` | `fn anyof_untyped_closed_object_variants_keep_non_object_alternatives() {` |
| `src/import/json_schema/tests/mod.rs` | 2791 | `anyof_untyped_closed_object_variants_with_sibling_required_use_exact_variant_nfa` | `fn anyof_untyped_closed_object_variants_with_sibling_required_use_exact_variant_nfa() {` |
| `src/import/json_schema/tests/mod.rs` | 2830 | `anyof_explicit_object_variants_do_not_add_non_object_alternatives` | `fn anyof_explicit_object_variants_do_not_add_non_object_alternatives() {` |
| `src/import/json_schema/tests/mod.rs` | 2857 | `untyped_plain_object_assertions_keep_non_object_alternatives` | `fn untyped_plain_object_assertions_keep_non_object_alternatives() {` |
| `src/import/json_schema/tests/mod.rs` | 2882 | `explicit_plain_object_assertions_remain_object_only` | `fn explicit_plain_object_assertions_remain_object_only() {` |
| `src/import/json_schema/tests/mod.rs` | 2897 | `untyped_object_and_array_assertions_do_not_take_plain_object_fallback` | `fn untyped_object_and_array_assertions_do_not_take_plain_object_fallback() {` |
| `src/import/json_schema/tests/mod.rs` | 2911 | `anyof_required_property_factoring_falls_back_for_unknown_required_name` | `fn anyof_required_property_factoring_falls_back_for_unknown_required_name() {` |
| `src/import/json_schema/tests/mod.rs` | 2933 | `allof_merges_plain_object_branches` | `fn allof_merges_plain_object_branches() {` |
| `src/import/json_schema/tests/mod.rs` | 2958 | `allof_merges_array_ref_with_min_items_assertion` | `fn allof_merges_array_ref_with_min_items_assertion() {` |
| `src/import/json_schema/tests/mod.rs` | 2980 | `allof_merges_array_bounds_before_ref_branch` | `fn allof_merges_array_bounds_before_ref_branch() {` |
| `src/import/json_schema/tests/mod.rs` | 3002 | `allof_array_min_max_items_merge_clamps_bounds` | `fn allof_array_min_max_items_merge_clamps_bounds() {` |
| `src/import/json_schema/tests/mod.rs` | 3025 | `allof_array_merge_preserves_non_array_type_union_guard` | `fn allof_array_merge_preserves_non_array_type_union_guard() {` |
| `src/import/json_schema/tests/mod.rs` | 3042 | `allof_flattens_nested_object_allof_before_intersect` | `fn allof_flattens_nested_object_allof_before_intersect() {` |
| `src/import/json_schema/tests/mod.rs` | 3072 | `allof_collapses_single_anyof_ref_before_intersect` | `fn allof_collapses_single_anyof_ref_before_intersect() {` |
| `src/import/json_schema/tests/mod.rs` | 3104 | `recursive_ref_in_allof_is_not_inlined` | `fn recursive_ref_in_allof_is_not_inlined() {` |
| `src/import/json_schema/tests/mod.rs` | 3133 | `allof_drops_vacuous_json_value_property_when_refined` | `fn allof_drops_vacuous_json_value_property_when_refined() {` |
| `src/import/json_schema/tests/mod.rs` | 3168 | `allof_drops_vacuous_object_property_when_refined` | `fn allof_drops_vacuous_object_property_when_refined() {` |
| `src/import/json_schema/tests/mod.rs` | 3200 | `allof_distributes_over_object_anyof_before_lowering` | `fn allof_distributes_over_object_anyof_before_lowering() {` |
| `src/import/json_schema/tests/mod.rs` | 3233 | `allof_ref_to_nested_object_oneof_with_siblings_lowers` | `fn allof_ref_to_nested_object_oneof_with_siblings_lowers() {` |
| `src/import/json_schema/tests/mod.rs` | 3287 | `unsafe_allof_object_ref_intersection_broadens_to_choice` | `fn unsafe_allof_object_ref_intersection_broadens_to_choice() {` |
| `src/import/json_schema/tests/mod.rs` | 3312 | `unsafe_allof_array_separated_sequence_broadens_to_choice` | `fn unsafe_allof_array_separated_sequence_broadens_to_choice() {` |
| `src/import/json_schema/tests/mod.rs` | 3331 | `terminal_safe_allof_keeps_intersection` | `fn terminal_safe_allof_keeps_intersection() {` |
| `src/import/json_schema/tests/mod.rs` | 3346 | `oneof_object_branches_with_root_type_object_and_required_anyof_lowers` | `fn oneof_object_branches_with_root_type_object_and_required_anyof_lowers() {` |
| `src/import/json_schema/tests/mod.rs` | 3394 | `open_object_anyof_uses_single_object_body_nfa` | `fn open_object_anyof_uses_single_object_body_nfa() {` |
| `src/import/json_schema/tests/mod.rs` | 3482 | `array_items_anyof_allof_ref_alias_variants_lower_to_shared_open_object_body` | `fn array_items_anyof_allof_ref_alias_variants_lower_to_shared_open_object_body() {` |
| `src/import/json_schema/tests/mod.rs` | 3577 | `sibling_pattern_addback_subtracts_local_pattern_language_for_o10297_shape` | `fn sibling_pattern_addback_subtracts_local_pattern_language_for_o10297_shape() {` |
| `src/import/json_schema/tests/mod.rs` | 3625 | `anyof_drops_subsumed_open_object_branch_for_o83993_shape` | `fn anyof_drops_subsumed_open_object_branch_for_o83993_shape() {` |
| `src/import/json_schema/tests/mod.rs` | 3662 | `anyof_drops_recursive_open_object_branches_subsumed_by_base_node` | `fn anyof_drops_recursive_open_object_branches_subsumed_by_base_node() {` |
| `src/import/json_schema/tests/mod.rs` | 3745 | `anyof_does_not_drop_open_object_branch_that_widens_base_property` | `fn anyof_does_not_drop_open_object_branch_that_widens_base_property() {` |
| `src/import/json_schema/tests/mod.rs` | 3771 | `shadow_author_author_path_schema` | `fn shadow_author_author_path_schema() -> serde_json::Value {` |
| `src/import/json_schema/tests/mod.rs` | 3799 | `shadow_owner_owned_object_close_suppresses_residual_duplicate` | `fn shadow_owner_owned_object_close_suppresses_residual_duplicate() {` |
| `src/import/json_schema/tests/mod.rs` | 3808 | `shadow_owner_missing_required_key_keeps_residual_open_branch` | `fn shadow_owner_missing_required_key_keeps_residual_open_branch() {` |
| `src/import/json_schema/tests/mod.rs` | 3815 | `shadow_owner_invalid_owner_fixed_type_keeps_residual_open_branch` | `fn shadow_owner_invalid_owner_fixed_type_keeps_residual_open_branch() {` |
| `src/import/json_schema/tests/mod.rs` | 3822 | `shadow_owner_invalid_date_time_string_keeps_residual_string_subtraction` | `fn shadow_owner_invalid_date_time_string_keeps_residual_string_subtraction() {` |
| `src/import/json_schema/tests/mod.rs` | 3832 | `shadow_owner_out_of_order_fixed_fields_keep_residual_open_branch` | `fn shadow_owner_out_of_order_fixed_fields_keep_residual_open_branch() {` |
| `src/import/json_schema/tests/mod.rs` | 3842 | `shadow_owner_skips_residual_with_unsafe_additional_constraints` | `fn shadow_owner_skips_residual_with_unsafe_additional_constraints() {` |
| `src/import/json_schema/tests/mod.rs` | 3871 | `shadow_owner_allows_unsupported_optional_owner_fields` | `fn shadow_owner_allows_unsupported_optional_owner_fields() {` |
| `src/import/json_schema/tests/mod.rs` | 3903 | `shadow_owner_ref_branch_context_uses_factored_open_object_body` | `fn shadow_owner_ref_branch_context_uses_factored_open_object_body() {` |
| `src/import/json_schema/tests/mod.rs` | 3962 | `single_anyof_object_ref_with_sibling_properties_merges_before_lowering` | `fn single_anyof_object_ref_with_sibling_properties_merges_before_lowering() {` |
| `src/import/json_schema/tests/mod.rs` | 3989 | `ref_with_sibling_assertions_is_intersected` | `fn ref_with_sibling_assertions_is_intersected() {` |
| `src/import/json_schema/tests/mod.rs` | 4005 | `singleton_allof_ref_without_siblings_reuses_ref_rule` | `fn singleton_allof_ref_without_siblings_reuses_ref_rule() {` |
| `src/import/json_schema/tests/mod.rs` | 4031 | `singleton_allof_ref_with_noop_object_siblings_reuses_ref_rule` | `fn singleton_allof_ref_with_noop_object_siblings_reuses_ref_rule() {` |
| `src/import/json_schema/tests/mod.rs` | 4059 | `singleton_allof_ref_with_restrictive_additional_properties_skips_fast_path` | `fn singleton_allof_ref_with_restrictive_additional_properties_skips_fast_path() {` |
| `src/import/lark/mod.rs` | 10 | `Token` | `enum Token {` |
| `src/import/lark/mod.rs` | 34 | `Lexer` | `struct Lexer<'a> {` |
| `src/import/lark/mod.rs` | 40 | `new` | `fn new(input: &'a str) -> Self {` |
| `src/import/lark/mod.rs` | 47 | `peek` | `fn peek(&self) -> Option<u8> {` |
| `src/import/lark/mod.rs` | 51 | `advance` | `fn advance(&mut self) -> Option<u8> {` |
| `src/import/lark/mod.rs` | 57 | `skip_whitespace_inline` | `fn skip_whitespace_inline(&mut self) {` |
| `src/import/lark/mod.rs` | 67 | `skip_comment` | `fn skip_comment(&mut self) {` |
| `src/import/lark/mod.rs` | 76 | `lex_string` | `fn lex_string(&mut self, quote: u8) -> Result<String, GlrMaskError> {` |
| `src/import/lark/mod.rs` | 114 | `lex_regex` | `fn lex_regex(&mut self) -> Result<String, GlrMaskError> {` |
| `src/import/lark/mod.rs` | 131 | `lex_ident` | `fn lex_ident(&mut self, first: u8) -> String {` |
| `src/import/lark/mod.rs` | 145 | `lex_number` | `fn lex_number(&mut self, first: u8) -> usize {` |
| `src/import/lark/mod.rs` | 158 | `tokenize` | `fn tokenize(&mut self) -> Result<Vec<Token>, GlrMaskError> {` |
| `src/import/lark/mod.rs` | 290 | `bounded_repeat_expr` | `fn bounded_repeat_expr(atom: GrammarExpr, min: usize, max: Option<usize>) -> GrammarExpr {` |
| `src/import/lark/mod.rs` | 300 | `escape_char_class_byte` | `fn escape_char_class_byte(b: u8) -> String {` |
| `src/import/lark/mod.rs` | 311 | `literal_range_expr` | `fn literal_range_expr(start: &str, end: &str) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/import/lark/mod.rs` | 340 | `Parser` | `struct Parser {` |
| `src/import/lark/mod.rs` | 345 | `is_lark_terminal_name` | `fn is_lark_terminal_name(name: &str) -> bool {` |
| `src/import/lark/mod.rs` | 352 | `lark_start_rule_name` | `fn lark_start_rule_name(rules: &[NamedRule]) -> String {` |
| `src/import/lark/mod.rs` | 360 | `mark_lark_terminal_rules` | `fn mark_lark_terminal_rules(rules: &mut [NamedRule]) {` |
| `src/import/lark/mod.rs` | 366 | `synthesize_lark_ignore_rule` | `fn synthesize_lark_ignore_rule(` |
| `src/import/lark/mod.rs` | 384 | `expand_lark_expr_list` | `fn expand_lark_expr_list(` |
| `src/import/lark/mod.rs` | 409 | `expand_lark_boxed_expr` | `fn expand_lark_boxed_expr(` |
| `src/import/lark/mod.rs` | 429 | `validate_lark_terminal_refs` | `fn validate_lark_terminal_refs(` |
| `src/import/lark/mod.rs` | 469 | `expand_lark_terminal_rule` | `fn expand_lark_terminal_rule(` |
| `src/import/lark/mod.rs` | 504 | `expand_lark_expr` | `fn expand_lark_expr(` |
| `src/import/lark/mod.rs` | 708 | `normalize_lark_named` | `fn normalize_lark_named(grammar: NamedGrammar) -> Result<NamedGrammar, GlrMaskError> {` |
| `src/import/lark/mod.rs` | 792 | `new` | `fn new(tokens: Vec<Token>) -> Self {` |
| `src/import/lark/mod.rs` | 796 | `peek` | `fn peek(&self) -> Option<&Token> {` |
| `src/import/lark/mod.rs` | 800 | `peek_nth` | `fn peek_nth(&self, n: usize) -> Option<&Token> {` |
| `src/import/lark/mod.rs` | 804 | `advance` | `fn advance(&mut self) -> Option<Token> {` |
| `src/import/lark/mod.rs` | 810 | `expect_token` | `fn expect_token(&mut self, expected: &Token) -> Result<(), GlrMaskError> {` |
| `src/import/lark/mod.rs` | 824 | `skip_newlines` | `fn skip_newlines(&mut self) {` |
| `src/import/lark/mod.rs` | 830 | `parse_rule_name` | `fn parse_rule_name(&mut self) -> Result<String, GlrMaskError> {` |
| `src/import/lark/mod.rs` | 847 | `skip_rule_priority` | `fn skip_rule_priority(&mut self) {` |
| `src/import/lark/mod.rs` | 853 | `parse_bounded_repeat` | `fn parse_bounded_repeat(&mut self, atom: GrammarExpr) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/import/lark/mod.rs` | 883 | `parse_literal_or_range` | `fn parse_literal_or_range(&mut self, literal: String) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/import/lark/mod.rs` | 905 | `parse_ignore_directive` | `fn parse_ignore_directive(` |
| `src/import/lark/mod.rs` | 919 | `parse_rule` | `fn parse_rule(&mut self) -> Result<NamedRule, GlrMaskError> {` |
| `src/import/lark/mod.rs` | 932 | `parse_grammar` | `fn parse_grammar(&mut self) -> Result<NamedGrammar, GlrMaskError> {` |
| `src/import/lark/mod.rs` | 957 | `parse_alternatives` | `fn parse_alternatives(&mut self) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/import/lark/mod.rs` | 971 | `consume_alternative_separator` | `fn consume_alternative_separator(&mut self) -> bool {` |
| `src/import/lark/mod.rs` | 990 | `consume_alias_if_present` | `fn consume_alias_if_present(&mut self) -> Result<(), GlrMaskError> {` |
| `src/import/lark/mod.rs` | 1008 | `parse_sequence` | `fn parse_sequence(&mut self) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/import/lark/mod.rs` | 1018 | `is_unit_start` | `fn is_unit_start(&self) -> bool {` |
| `src/import/lark/mod.rs` | 1031 | `parse_unit` | `fn parse_unit(&mut self) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/import/lark/mod.rs` | 1055 | `parse_atom` | `fn parse_atom(&mut self) -> Result<GrammarExpr, GlrMaskError> {` |
| `src/import/lark/mod.rs` | 1079 | `parse_lark` | `pub fn parse_lark(input: &str) -> Result<GrammarDef, GlrMaskError> {` |
| `src/import/lark/mod.rs` | 1085 | `parse_lark_to_named` | `pub fn parse_lark_to_named(input: &str) -> Result<NamedGrammar, GlrMaskError> {` |
| `src/import/mod.rs` | 16 | `GrammarParser` | `type GrammarParser = fn(&str) -> crate::Result<GrammarDef>;` |
| `src/import/mod.rs` | 17 | `NamedGrammarParser` | `type NamedGrammarParser = fn(&str) -> crate::Result<ast::NamedGrammar>;` |
| `src/import/mod.rs` | 19 | `choice_or_single` | `pub(crate) fn choice_or_single(mut options: Vec<ast::GrammarExpr>) -> ast::GrammarExpr {` |
| `src/import/mod.rs` | 27 | `sequence_or_single` | `pub(crate) fn sequence_or_single(mut items: Vec<ast::GrammarExpr>) -> ast::GrammarExpr {` |
| `src/import/mod.rs` | 35 | `emit_import_phase_start` | `fn emit_import_phase_start(name: &'static str) -> Option<std::time::Instant> {` |
| `src/import/mod.rs` | 44 | `emit_import_phase_end` | `fn emit_import_phase_end(name: &'static str, started_at: Option<std::time::Instant>) {` |
| `src/import/mod.rs` | 54 | `lower_factored_named_grammar` | `fn lower_factored_named_grammar(` |
| `src/import/mod.rs` | 93 | `compile_from_source` | `fn compile_from_source(` |
| `src/import/mod.rs` | 116 | `parse_json_schema_to_named` | `fn parse_json_schema_to_named(schema_json: &str) -> crate::Result<ast::NamedGrammar> {` |
| `src/import/mod.rs` | 145 | `from_ebnf` | `pub fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Self> {` |
| `src/import/mod.rs` | 162 | `from_lark` | `pub fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Self> {` |
| `src/import/mod.rs` | 184 | `from_json_schema` | `pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {` |
| `src/import/mod.rs` | 193 | `from_glrm_grammar` | `pub fn from_glrm_grammar(glrm: &str, vocab: &crate::Vocab) -> crate::Result<Self> {` |
| `src/import/numeric_range/mod.rs` | 6 | `Result` | `type Result<T> = std::result::Result<T, String>;` |
| `src/import/numeric_range/mod.rs` | 8 | `mk_or` | `fn mk_or(parts: Vec<String>) -> String {` |
| `src/import/numeric_range/mod.rs` | 16 | `mk_or_opt` | `fn mk_or_opt(parts: Vec<String>) -> Option<String> {` |
| `src/import/numeric_range/mod.rs` | 20 | `push_optional_part` | `fn push_optional_part(parts: &mut Vec<String>, part: Option<String>) {` |
| `src/import/numeric_range/mod.rs` | 26 | `negative_regex` | `fn negative_regex(pattern: String) -> String {` |
| `src/import/numeric_range/mod.rs` | 30 | `num_digits` | `fn num_digits(n: i64) -> usize {` |
| `src/import/numeric_range/mod.rs` | 36 | `rx_int_range` | `pub fn rx_int_range(left: Option<i64>, right: Option<i64>) -> Result<String> {` |
| `src/import/numeric_range/mod.rs` | 137 | `lexi_x_to_9` | `fn lexi_x_to_9(x: &str, incl: bool) -> Result<String> {` |
| `src/import/numeric_range/mod.rs` | 173 | `lexi_0_to_x` | `fn lexi_0_to_x(x: &str, incl: bool) -> Result<String> {` |
| `src/import/numeric_range/mod.rs` | 205 | `lexi_range` | `fn lexi_range(ld: &str, rd: &str, ld_incl: bool, rd_incl: bool) -> Result<String> {` |
| `src/import/numeric_range/mod.rs` | 251 | `float_to_str` | `fn float_to_str(f: f64) -> String {` |
| `src/import/numeric_range/mod.rs` | 256 | `escape_float_str` | `fn escape_float_str(s: &str) -> String {` |
| `src/import/numeric_range/mod.rs` | 260 | `exact_float_regex` | `fn exact_float_regex(value: f64) -> String {` |
| `src/import/numeric_range/mod.rs` | 264 | `NonnegativeDecimalBounds` | `struct NonnegativeDecimalBounds {` |
| `src/import/numeric_range/mod.rs` | 271 | `decimal_fraction` | `fn decimal_fraction(rendered: &str) -> String {` |
| `src/import/numeric_range/mod.rs` | 275 | `parse_decimal_integer` | `fn parse_decimal_integer(rendered: &str, label: &str) -> Result<i64> {` |
| `src/import/numeric_range/mod.rs` | 284 | `nonnegative_decimal_bounds` | `fn nonnegative_decimal_bounds(left: f64, right: f64) -> Result<NonnegativeDecimalBounds> {` |
| `src/import/numeric_range/mod.rs` | 303 | `pad_decimal_fractions` | `fn pad_decimal_fractions(left_fraction: &mut String, right_fraction: &mut String) {` |
| `src/import/numeric_range/mod.rs` | 313 | `FloatRangeMode` | `enum FloatRangeMode {` |
| `src/import/numeric_range/mod.rs` | 318 | `same_integer_float_part` | `fn same_integer_float_part(` |
| `src/import/numeric_range/mod.rs` | 344 | `lower_float_part` | `fn lower_float_part(` |
| `src/import/numeric_range/mod.rs` | 379 | `middle_float_part` | `fn middle_float_part(bounds: &NonnegativeDecimalBounds, mode: FloatRangeMode) -> Result<Option<String>> {` |
| `src/import/numeric_range/mod.rs` | 391 | `upper_float_part` | `fn upper_float_part(` |
| `src/import/numeric_range/mod.rs` | 422 | `collect_nonnegative_float_parts` | `fn collect_nonnegative_float_parts(` |
| `src/import/numeric_range/mod.rs` | 437 | `rx_float_range` | `pub fn rx_float_range(` |
| `src/import/numeric_range/mod.rs` | 528 | `rx_noninteger_float_range` | `pub fn rx_noninteger_float_range(` |
| `src/import/numeric_range/mod.rs` | 637 | `nonneg_float_range` | `fn nonneg_float_range(` |
| `src/import/numeric_range/mod.rs` | 662 | `nonneg_float_range_no_ints` | `fn nonneg_float_range_no_ints(` |
| `src/parser/glr/accumulator.rs` | 12 | `TerminalsDisallowed` | `pub struct TerminalsDisallowed(pub(crate) Arc<BTreeMap<u32, BTreeSet<u32>>>);` |
| `src/parser/glr/accumulator.rs` | 15 | `new` | `pub fn new() -> Self {` |
| `src/parser/glr/accumulator.rs` | 19 | `is_subset_of` | `pub fn is_subset_of(&self, other: &Self) -> bool {` |
| `src/parser/glr/accumulator.rs` | 38 | `with_insert` | `pub fn with_insert(&self, state: u32, terminal: u32) -> Self {` |
| `src/parser/glr/accumulator.rs` | 47 | `Target` | `type Target = BTreeMap<u32, BTreeSet<u32>>;` |
| `src/parser/glr/accumulator.rs` | 48 | `deref` | `fn deref(&self) -> &Self::Target {` |
| `src/parser/glr/accumulator.rs` | 54 | `eq` | `fn eq(&self, other: &Self) -> bool {` |
| `src/parser/glr/accumulator.rs` | 62 | `hash` | `fn hash<H: Hasher>(&self, state: &mut H) {` |
| `src/parser/glr/accumulator.rs` | 68 | `merge` | `fn merge(&self, other: &Self) -> Self {` |
| `src/parser/glr/accumulator.rs` | 94 | `subsumes` | `fn subsumes(&self, other: &Self) -> bool {` |
| `src/parser/glr/advance/applicability.rs` | 1 | `stack_can_advance_on` | `pub(crate) fn stack_can_advance_on(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> bool {` |
| `src/parser/glr/advance/applicability.rs` | 22 | `stack_may_apply_guarded_shifts` | `fn stack_may_apply_guarded_shifts(stack: &ParserGSS, shifts: &[GuardedStackShift]) -> bool {` |
| `src/parser/glr/advance/applicability_any.rs` | 1 | `stack_can_advance_on_any` | `pub(crate) fn stack_can_advance_on_any(` |
| `src/parser/glr/advance/applicability_any.rs` | 74 | `stacks_finished` | `pub(crate) fn stacks_finished(table: &GLRTable, stack: &ParserGSS) -> bool {` |
| `src/parser/glr/advance/deterministic.rs` | 1 | `advance_deterministically` | `fn advance_deterministically(` |
| `src/parser/glr/advance/deterministic_profiled.rs` | 1 | `advance_deterministically_profiled` | `fn advance_deterministically_profiled(` |
| `src/parser/glr/advance/deterministic_vstack.rs` | 1 | `advance_deterministically_from_vstack_raw` | `fn advance_deterministically_from_vstack_raw(` |
| `src/parser/glr/advance/deterministic_vstack.rs` | 85 | `advance_deterministically_from_vstack` | `fn advance_deterministically_from_vstack(` |
| `src/parser/glr/advance/deterministic_vstack.rs` | 94 | `advance_reduce_branch` | `fn advance_reduce_branch(` |
| `src/parser/glr/advance/deterministic_vstack.rs` | 119 | `single_concrete_path_as_vstack` | `fn single_concrete_path_as_vstack(` |
| `src/parser/glr/advance/deterministic_vstack.rs` | 133 | `advance_split_from_vstack` | `fn advance_split_from_vstack(` |
| `src/parser/glr/advance/entry_points.rs` | 1 | `advance_stacks` | `pub(crate) fn advance_stacks(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> ParserGSS {` |
| `src/parser/glr/advance/entry_points.rs` | 7 | `advance_stacks_owned` | `pub(crate) fn advance_stacks_owned(table: &GLRTable, stack: ParserGSS, token: TerminalID) -> ParserGSS {` |
| `src/parser/glr/advance/entry_points.rs` | 11 | `advance_stacks_profiled` | `pub(crate) fn advance_stacks_profiled(` |
| `src/parser/glr/advance/entry_points.rs` | 139 | `advance_stacks_core` | `fn advance_stacks_core(table: &GLRTable, mut gss: ParserGSS, token: TerminalID) -> ParserGSS {` |
| `src/parser/glr/advance/entry_points.rs` | 172 | `try_collapse_small_reduce_fanout` | `fn try_collapse_small_reduce_fanout(` |
| `src/parser/glr/advance/entry_points.rs` | 219 | `pure_frontier_shift` | `fn pure_frontier_shift(action: &Action) -> Option<(u32, bool)> {` |
| `src/parser/glr/advance/fast_paths.rs` | 1 | `advance_pure_frontier_shifts` | `fn advance_pure_frontier_shifts(` |
| `src/parser/glr/advance/fast_paths.rs` | 28 | `try_advance_single_alt_pop1_common_suffix_stackshift_wave` | `fn try_advance_single_alt_pop1_common_suffix_stackshift_wave(` |
| `src/parser/glr/advance/fast_paths.rs` | 64 | `try_advance_pop1_reduce_plus_stackshift_wave` | `fn try_advance_pop1_reduce_plus_stackshift_wave(` |
| `src/parser/glr/advance/fast_paths.rs` | 133 | `rebuild_floor_cross_from_shifts` | `fn rebuild_floor_cross_from_shifts(` |
| `src/parser/glr/advance/fast_paths.rs` | 148 | `push_states` | `fn push_states(mut gss: ParserGSS, states: &[u32]) -> ParserGSS {` |
| `src/parser/glr/advance/fast_paths.rs` | 155 | `common_stack_shift_suffix_len` | `fn common_stack_shift_suffix_len(pushes: &[&[u32]]) -> usize {` |
| `src/parser/glr/advance/fast_paths.rs` | 172 | `apply_push_sequences` | `fn apply_push_sequences(base: ParserGSS, pushes: &[&[u32]]) -> ParserGSS {` |
| `src/parser/glr/advance/fast_paths.rs` | 197 | `apply_stack_shifts` | `fn apply_stack_shifts(gss: ParserGSS, shifts: &[StackShift]) -> ParserGSS {` |
| `src/parser/glr/advance/fast_paths.rs` | 255 | `apply_guarded_stack_shifts_fast` | `pub(crate) fn apply_guarded_stack_shifts_fast(` |
| `src/parser/glr/advance/guarded_shifts.rs` | 1 | `apply_guarded_stack_shifts` | `fn apply_guarded_stack_shifts(` |
| `src/parser/glr/advance/guarded_shifts.rs` | 57 | `indexed_guarded_shift_candidates` | `fn indexed_guarded_shift_candidates(` |
| `src/parser/glr/advance/guarded_shifts.rs` | 90 | `apply_guarded_stack_shifts_to_vstack` | `fn apply_guarded_stack_shifts_to_vstack(` |
| `src/parser/glr/advance/guarded_shifts.rs` | 101 | `state_after_popping` | `fn state_after_popping(` |
| `src/parser/glr/advance/guarded_shifts.rs` | 114 | `consider_guarded_shift` | `fn consider_guarded_shift<'a>(` |
| `src/parser/glr/advance/guarded_shifts.rs` | 196 | `virtual_stack_satisfies_guards` | `fn virtual_stack_satisfies_guards(` |
| `src/parser/glr/advance/guarded_shifts.rs` | 226 | `virtual_stack_may_apply_guarded_shift` | `fn virtual_stack_may_apply_guarded_shift(` |
| `src/parser/glr/advance/mod.rs` | 30 | `ParserGSS` | `pub type ParserGSS = LeveledGSS<u32, TerminalsDisallowed>;` |
| `src/parser/glr/advance/mod.rs` | 32 | `ReduceSources` | `type ReduceSources = SmallVec<[(u32, ParserGSS); 4]>;` |
| `src/parser/glr/advance/mod.rs` | 33 | `ReduceBranches` | `type ReduceBranches = SmallVec<[(ParserGSS, u32, bool); 4]>;` |
| `src/parser/glr/advance/mod.rs` | 34 | `FloorCrossShift` | `type FloorCrossShift = (u32, u32, bool);` |
| `src/parser/glr/advance/mod.rs` | 40 | `advance_options` | `fn advance_options() -> &'static ParserAdvanceOptions {` |
| `src/parser/glr/advance/mod.rs` | 44 | `guarded_stack_to_stacks_fallback_disabled` | `fn guarded_stack_to_stacks_fallback_disabled() -> bool {` |
| `src/parser/glr/advance/mod.rs` | 48 | `stack_effect_to_stacks_fallback_disabled` | `fn stack_effect_to_stacks_fallback_disabled() -> bool {` |
| `src/parser/glr/advance/mod.rs` | 52 | `advance_trace_enabled` | `fn advance_trace_enabled() -> bool {` |
| `src/parser/glr/advance/nondeterministic.rs` | 1 | `advance_nondeterministically` | `fn advance_nondeterministically(` |
| `src/parser/glr/advance/nondeterministic_profiled.rs` | 1 | `advance_nondeterministically_profiled` | `fn advance_nondeterministically_profiled(` |
| `src/parser/glr/advance/options.rs` | 14 | `ParserAdvanceOptions` | `pub(crate) struct ParserAdvanceOptions {` |
| `src/parser/glr/advance/options.rs` | 35 | `from_env` | `pub(crate) fn from_env() -> Self {` |
| `src/parser/glr/advance/options.rs` | 47 | `global` | `pub(crate) fn global() -> &'static Self {` |
| `src/parser/glr/advance/options.rs` | 53 | `env_flag_enabled` | `fn env_flag_enabled(name: &str) -> bool {` |
| `src/parser/glr/advance/profile.rs` | 2 | `AdvanceTrace` | `pub struct AdvanceTrace {` |
| `src/parser/glr/advance/profile.rs` | 8 | `AdvanceTraceWave` | `pub struct AdvanceTraceWave {` |
| `src/parser/glr/advance/profile.rs` | 15 | `AdvanceTraceStep` | `pub struct AdvanceTraceStep {` |
| `src/parser/glr/advance/profile.rs` | 24 | `AdvanceTraceReduce` | `pub struct AdvanceTraceReduce {` |
| `src/parser/glr/advance/profile.rs` | 33 | `AdvanceTraceGoto` | `pub struct AdvanceTraceGoto {` |
| `src/parser/glr/advance/profile.rs` | 40 | `AdvanceProfile` | `pub struct AdvanceProfile {` |
| `src/parser/glr/advance/profile_trace.rs` | 1 | `trace_action_kind` | `fn trace_action_kind(action: Option<&Action>) -> &'static str {` |
| `src/parser/glr/advance/profile_trace.rs` | 14 | `trace_reduce_summary` | `fn trace_reduce_summary(` |
| `src/parser/glr/advance/profile_trace.rs` | 49 | `trace_action_summary` | `fn trace_action_summary(` |
| `src/parser/glr/advance/profile_trace.rs` | 92 | `AdvancedBranch` | `enum AdvancedBranch {` |
| `src/parser/glr/advance/profile_trace.rs` | 98 | `into_gss` | `fn into_gss(self) -> ParserGSS {` |
| `src/parser/glr/advance/reduce_sources.rs` | 1 | `reduce_sources_from_isolated` | `fn reduce_sources_from_isolated(gss: &ParserGSS, rhs_len: usize) -> ReduceSources {` |
| `src/parser/glr/advance/reduce_sources.rs` | 19 | `reduce_branches_from_isolated` | `fn reduce_branches_from_isolated(` |
| `src/parser/glr/advance/reduce_sources.rs` | 46 | `merge_into` | `fn merge_into(dst: &mut ParserGSS, branch: ParserGSS) {` |
| `src/parser/glr/advance/tests.rs` | 17 | `advance_stacks_matches_reduce_fanout_collapse_fast_path` | `fn advance_stacks_matches_reduce_fanout_collapse_fast_path() {` |
| `src/parser/glr/advance/tests.rs` | 50 | `advance_stacks_selective_pure_frontier_shift_keeps_only_actionable_top` | `fn advance_stacks_selective_pure_frontier_shift_keeps_only_actionable_top() {` |
| `src/parser/glr/advance/tests.rs` | 82 | `pop1_reduce_plus_stackshift_wave_fast_path_matches_snowplow_shape` | `fn pop1_reduce_plus_stackshift_wave_fast_path_matches_snowplow_shape() {` |
| `src/parser/glr/advance/tests.rs` | 132 | `pop1_reduce_plus_stackshift_wave_rejects_cross_product_base` | `fn pop1_reduce_plus_stackshift_wave_rejects_cross_product_base() {` |
| `src/parser/glr/advance/tests.rs` | 170 | `can_advance_consults_admission_rows_not_execution_actions` | `fn can_advance_consults_admission_rows_not_execution_actions() {` |
| `src/parser/glr/advance/tests.rs` | 190 | `can_advance_rechecks_guarded_stack_shifts_against_concrete_stack` | `fn can_advance_rechecks_guarded_stack_shifts_against_concrete_stack() {` |
| `src/parser/glr/advance/tests.rs` | 225 | `advance_stacks_materializes_single_concrete_path_for_split` | `fn advance_stacks_materializes_single_concrete_path_for_split() {` |
| `src/parser/glr/advance/tests.rs` | 274 | `indexed_guarded_vstack_matches_linear_guarded_vstack` | `fn indexed_guarded_vstack_matches_linear_guarded_vstack() {` |
| `src/parser/glr/analysis/fixed_point_sets.rs` | 1 | `compute_nullable` | `fn compute_nullable(rules: &[Rule], num_nt: u32) -> BTreeSet<NonterminalID> {` |
| `src/parser/glr/analysis/fixed_point_sets.rs` | 23 | `compute_first` | `fn compute_first(` |
| `src/parser/glr/analysis/fixed_point_sets.rs` | 63 | `compute_follow` | `fn compute_follow(` |
| `src/parser/glr/analysis/fixed_point_sets.rs` | 126 | `terminal_bit` | `fn terminal_bit(terminal: TerminalID, num_terminals: u32) -> usize {` |
| `src/parser/glr/analysis/fixed_point_sets.rs` | 134 | `filter_graph_to_reachable` | `fn filter_graph_to_reachable(` |
| `src/parser/glr/analysis/left_recursion.rs` | 1 | `eliminate_hidden_left_recursion` | `fn eliminate_hidden_left_recursion(` |
| `src/parser/glr/analysis/left_recursion.rs` | 135 | `expand_cycle_head_paths` | `fn expand_cycle_head_paths(` |
| `src/parser/glr/analysis/left_recursion.rs` | 141 | `expand` | `fn expand(` |
| `src/parser/glr/analysis/left_recursion.rs` | 193 | `nullable_prefix_len` | `fn nullable_prefix_len(rhs: &[Symbol], nullable: &BTreeSet<NonterminalID>) -> usize {` |
| `src/parser/glr/analysis/model.rs` | 1 | `AnalyzedGrammar` | `pub struct AnalyzedGrammar {` |
| `src/parser/glr/analysis/model.rs` | 15 | `from_grammar_def` | `pub fn from_grammar_def(g: &GrammarDef) -> Self {` |
| `src/parser/glr/analysis/model.rs` | 70 | `terminal_display_name` | `pub fn terminal_display_name(&self, terminal: TerminalID) -> &str {` |
| `src/parser/glr/analysis/model.rs` | 79 | `check_table_build_normal_form` | `pub fn check_table_build_normal_form(&self) -> Result<(), String> {` |
| `src/parser/glr/analysis/model.rs` | 97 | `debug_check_grammar_preconditions` | `pub fn debug_check_grammar_preconditions(&self) -> Result<(), String> {` |
| `src/parser/glr/analysis/model.rs` | 101 | `check_no_nullable_nonterminals` | `pub fn check_no_nullable_nonterminals(&self) -> Result<(), String> {` |
| `src/parser/glr/analysis/model.rs` | 124 | `check_no_reachable_zero_length_productions` | `pub fn check_no_reachable_zero_length_productions(&self) -> Result<(), String> {` |
| `src/parser/glr/analysis/model.rs` | 144 | `check_recursion_boundedness` | `pub fn check_recursion_boundedness(&self) -> Result<(), String> {` |
| `src/parser/glr/analysis/model.rs` | 184 | `reachable_nonterminals` | `fn reachable_nonterminals(&self) -> BTreeSet<NonterminalID> {` |
| `src/parser/glr/analysis/model.rs` | 209 | `eliminate_right_recursion` | `pub(crate) fn eliminate_right_recursion(` |
| `src/parser/glr/analysis/normalize.rs` | 1 | `normalize_grammar` | `pub fn normalize_grammar(rules: &mut Vec<Rule>, start: NonterminalID) {` |
| `src/parser/glr/analysis/normalize.rs` | 169 | `replace_rules_with_resync` | `fn replace_rules_with_resync(` |
| `src/parser/glr/analysis/normalize.rs` | 178 | `with_resynced_next_nonterminal` | `fn with_resynced_next_nonterminal(` |
| `src/parser/glr/analysis/normalize.rs` | 187 | `resync_next_nonterminal` | `fn resync_next_nonterminal(rules: &[Rule], next_nt: &std::cell::Cell<u32>) {` |
| `src/parser/glr/analysis/null_production_inline.rs` | 1 | `inline_null_productions_exhaustive` | `fn inline_null_productions_exhaustive(rules: &[Rule], num_nt: u32) -> Vec<Rule> {` |
| `src/parser/glr/analysis/null_production_inline.rs` | 200 | `find_nullable_runs` | `fn find_nullable_runs(` |
| `src/parser/glr/analysis/nullable_run_compress.rs` | 1 | `compute_nonempty_productive` | `fn compute_nonempty_productive(rules: &[Rule], num_nt: u32) -> BTreeSet<NonterminalID> {` |
| `src/parser/glr/analysis/nullable_run_compress.rs` | 38 | `compress_nullable_runs_with_optional_tree` | `fn compress_nullable_runs_with_optional_tree(rules: &[Rule], num_nt: u32) -> Vec<Rule> {` |
| `src/parser/glr/analysis/nullable_run_compress.rs` | 253 | `build_non_nullable_tree` | `fn build_non_nullable_tree(` |
| `src/parser/glr/analysis/nullable_run_compress.rs` | 350 | `get_or_create_non_nullable_nt` | `fn get_or_create_non_nullable_nt(` |
| `src/parser/glr/analysis/nullable_run_compress.rs` | 419 | `inline_null_productions` | `pub(crate) fn inline_null_productions(rules: &[Rule], num_nt: u32) -> Vec<Rule> {` |
| `src/parser/glr/analysis/options.rs` | 7 | `analysis_profile_enabled` | `pub(crate) fn analysis_profile_enabled() -> bool {` |
| `src/parser/glr/analysis/profile.rs` | 1 | `compile_profile_enabled` | `fn compile_profile_enabled() -> bool {` |
| `src/parser/glr/analysis/profile.rs` | 5 | `elapsed_ms` | `fn elapsed_ms(started_at: Instant) -> f64 {` |
| `src/parser/glr/analysis/profile.rs` | 9 | `emit_normalize_profile` | `fn emit_normalize_profile(` |
| `src/parser/glr/analysis/profile.rs` | 38 | `emit_inline_null_profile` | `fn emit_inline_null_profile(` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 1 | `remove_unreachable_rules` | `fn remove_unreachable_rules(rules: &[Rule], start: NonterminalID) -> Vec<Rule> {` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 33 | `build_rhs_by_lhs` | `fn build_rhs_by_lhs(rules: &[Rule]) -> BTreeMap<NonterminalID, BTreeSet<Vec<Symbol>>> {` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 44 | `compute_expandable_single_productions` | `fn compute_expandable_single_productions(` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 48 | `VisitState` | `enum VisitState {` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 54 | `visit` | `fn visit(` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 115 | `flatten_rhs_symbols` | `fn flatten_rhs_symbols(` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 123 | `flatten_symbol` | `fn flatten_symbol(` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 189 | `RuleDedupKey` | `enum RuleDedupKey<'a> {` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 195 | `lhs` | `fn lhs(&self) -> NonterminalID {` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 201 | `rhs` | `fn rhs(&self) -> &[Symbol] {` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 210 | `eq` | `fn eq(&self, other: &Self) -> bool {` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 218 | `hash` | `fn hash<H: Hasher>(&self, state: &mut H) {` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 224 | `dedup_rules` | `fn dedup_rules(rules: &mut Vec<Rule>) {` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 257 | `is_reflexive_unit_rule` | `fn is_reflexive_unit_rule(rule: &Rule) -> bool {` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 261 | `merge_identical_nonterminals` | `pub(crate) fn merge_identical_nonterminals(` |
| `src/parser/glr/analysis/right_recursion.rs` | 1 | `max_nt_id` | `fn max_nt_id(rules: &[Rule]) -> u32 {` |
| `src/parser/glr/analysis/right_recursion.rs` | 14 | `add_boundary_nonterminals` | `fn add_boundary_nonterminals<'a>(` |
| `src/parser/glr/analysis/right_recursion.rs` | 32 | `build_right_reachability_graph` | `fn build_right_reachability_graph(` |
| `src/parser/glr/analysis/right_recursion.rs` | 47 | `find_indirect_rr_cycle` | `fn find_indirect_rr_cycle(` |
| `src/parser/glr/analysis/right_recursion.rs` | 53 | `find_cycle` | `fn find_cycle(` |
| `src/parser/glr/analysis/right_recursion.rs` | 58 | `dfs` | `fn dfs(` |
| `src/parser/glr/analysis/right_recursion.rs` | 123 | `build_left_reachability_graph` | `fn build_left_reachability_graph(` |
| `src/parser/glr/analysis/right_recursion.rs` | 136 | `find_indirect_lr_cycle` | `fn find_indirect_lr_cycle(` |
| `src/parser/glr/analysis/right_recursion.rs` | 142 | `find_nontrivial_sccs` | `fn find_nontrivial_sccs(` |
| `src/parser/glr/analysis/right_recursion.rs` | 229 | `find_cycle_excluding_self_loops` | `fn find_cycle_excluding_self_loops(` |
| `src/parser/glr/analysis/right_recursion.rs` | 240 | `inline_right_end` | `fn inline_right_end(` |
| `src/parser/glr/analysis/right_recursion.rs` | 278 | `find_right_end_position` | `fn find_right_end_position(` |
| `src/parser/glr/analysis/right_recursion.rs` | 293 | `is_direct_right_recursive` | `fn is_direct_right_recursive(rule: &Rule) -> bool {` |
| `src/parser/glr/analysis/right_recursion.rs` | 307 | `resolve_direct_rr_single_nt` | `fn resolve_direct_rr_single_nt(` |
| `src/parser/glr/analysis/right_recursion.rs` | 357 | `resolve_direct_rr_batched` | `fn resolve_direct_rr_batched(` |
| `src/parser/glr/analysis/tests.rs` | 5 | `analyzed_grammar` | `fn analyzed_grammar(rules: Vec<Rule>, start: NonterminalID) -> AnalyzedGrammar {` |
| `src/parser/glr/analysis/tests.rs` | 17 | `bounded_language` | `fn bounded_language(` |
| `src/parser/glr/analysis/tests.rs` | 23 | `rhs_language` | `fn rhs_language(` |
| `src/parser/glr/analysis/tests.rs` | 74 | `table_build_normal_form_rejects_nullable_zero_length_rules` | `fn table_build_normal_form_rejects_nullable_zero_length_rules() {` |
| `src/parser/glr/analysis/tests.rs` | 83 | `table_build_normal_form_rejects_direct_right_recursion` | `fn table_build_normal_form_rejects_direct_right_recursion() {` |
| `src/parser/glr/analysis/tests.rs` | 103 | `table_build_normal_form_rejects_indirect_left_recursion` | `fn table_build_normal_form_rejects_indirect_left_recursion() {` |
| `src/parser/glr/analysis/tests.rs` | 127 | `table_build_normal_form_accepts_simple_nonnullable_grammar` | `fn table_build_normal_form_accepts_simple_nonnullable_grammar() {` |
| `src/parser/glr/analysis/tests.rs` | 140 | `nontrivial_sccs_include_multiple_disjoint_cycles_and_skip_self_loops` | `fn nontrivial_sccs_include_multiple_disjoint_cycles_and_skip_self_loops() {` |
| `src/parser/glr/analysis/tests.rs` | 158 | `nullable_run_compression_preserves_nullable_only_nonempty_derivations` | `fn nullable_run_compression_preserves_nullable_only_nonempty_derivations() {` |
| `src/parser/glr/labels.rs` | 3 | `encode_positive_label` | `pub(crate) fn encode_positive_label(state: u32) -> i32 {` |
| `src/parser/glr/labels.rs` | 7 | `encode_negative_label` | `pub(crate) fn encode_negative_label(state: u32) -> i32 {` |
| `src/parser/glr/labels.rs` | 11 | `is_negative_label` | `pub(crate) fn is_negative_label(label: i32) -> bool {` |
| `src/parser/glr/labels.rs` | 15 | `negative_to_positive_label` | `pub(crate) fn negative_to_positive_label(label: i32) -> i32 {` |
| `src/parser/glr/table/action.rs` | 6 | `StackShift` | `pub struct StackShift {` |
| `src/parser/glr/table/action.rs` | 12 | `StackShiftGuard` | `pub struct StackShiftGuard {` |
| `src/parser/glr/table/action.rs` | 18 | `GuardedStackShift` | `pub struct GuardedStackShift {` |
| `src/parser/glr/table/action.rs` | 26 | `Action` | `pub enum Action {` |
| `src/parser/glr/table/action.rs` | 41 | `shift_target` | `pub fn shift_target(&self) -> Option<u32> {` |
| `src/parser/glr/table/action.rs` | 56 | `shift_is_replace` | `pub fn shift_is_replace(&self) -> bool {` |
| `src/parser/glr/table/action.rs` | 69 | `for_each_stack_shift` | `pub fn for_each_stack_shift(&self, mut f: impl FnMut(u32, &[u32])) {` |
| `src/parser/glr/table/action.rs` | 90 | `for_each_reduce` | `pub fn for_each_reduce(&self, mut f: impl FnMut(NonterminalID, u32)) {` |
| `src/parser/glr/table/action.rs` | 103 | `reduce_count` | `pub fn reduce_count(&self) -> usize {` |
| `src/parser/glr/table/action.rs` | 117 | `guarded_stack_shifts_bincode_roundtrip_preserves_empty_guards` | `fn guarded_stack_shifts_bincode_roundtrip_preserves_empty_guards() {` |
| `src/parser/glr/table/build.rs` | 6 | `build_table` | `pub(super) fn build_table(grammar: &AnalyzedGrammar) -> GLRTable {` |
| `src/parser/glr/table/build.rs` | 93 | `replace_shifts_enabled` | `fn replace_shifts_enabled() -> bool {` |
| `src/parser/glr/table/build.rs` | 97 | `replace_gotos_enabled` | `fn replace_gotos_enabled() -> bool {` |
| `src/parser/glr/table/build.rs` | 105 | `local_forward_replace_enabled` | `fn local_forward_replace_enabled() -> bool {` |
| `src/parser/glr/table/build.rs` | 117 | `Item` | `pub(super) struct Item {` |
| `src/parser/glr/table/build.rs` | 124 | `new` | `pub(super) fn new(rule: u32, dot: u32, stack_depth: u32) -> Self {` |
| `src/parser/glr/table/build.rs` | 128 | `next_symbol` | `fn next_symbol<'a>(&self, rules: &'a [Rule]) -> Option<&'a Symbol> {` |
| `src/parser/glr/table/build.rs` | 135 | `PendingAction` | `pub(super) struct PendingAction {` |
| `src/parser/glr/table/build.rs` | 142 | `push_shift` | `pub(super) fn push_shift(&mut self, target: u32, is_replace: bool) {` |
| `src/parser/glr/table/build.rs` | 149 | `push_reduce` | `pub(super) fn push_reduce(&mut self, nt: NonterminalID, len: u32) {` |
| `src/parser/glr/table/build.rs` | 153 | `push_accept` | `pub(super) fn push_accept(&mut self) {` |
| `src/parser/glr/table/build.rs` | 157 | `maybe_finish` | `pub(super) fn maybe_finish(mut self) -> Option<Action> {` |
| `src/parser/glr/table/build.rs` | 173 | `finish` | `pub(super) fn finish(self) -> Action {` |
| `src/parser/glr/table/build.rs` | 179 | `initialize_pending_and_goto` | `fn initialize_pending_and_goto(` |
| `src/parser/glr/table/build.rs` | 214 | `finish_table` | `fn finish_table(` |
| `src/parser/glr/table/build.rs` | 248 | `lookahead_bit` | `fn lookahead_bit(term: TerminalID, num_terminals: u32) -> usize {` |
| `src/parser/glr/table/build.rs` | 256 | `bit_lookahead` | `fn bit_lookahead(bit: usize, num_terminals: u32) -> TerminalID {` |
| `src/parser/glr/table/build.rs` | 265 | `LR1ItemCore` | `struct LR1ItemCore {` |
| `src/parser/glr/table/build.rs` | 273 | `new` | `fn new(rule: u32, dot: u32, stack_depth: u32) -> Self {` |
| `src/parser/glr/table/build.rs` | 282 | `next_symbol` | `fn next_symbol<'a>(&self, rules: &'a [Rule]) -> Option<&'a Symbol> {` |
| `src/parser/glr/table/build.rs` | 288 | `LR1ItemSet` | `type LR1ItemSet = BTreeMap<LR1ItemCore, BitSet>;` |
| `src/parser/glr/table/build.rs` | 291 | `LR1Item` | `struct LR1Item {` |
| `src/parser/glr/table/build.rs` | 303 | `new` | `fn new(rule: u32, dot: u32, lookahead: TerminalID, stack_depth: u32) -> Self {` |
| `src/parser/glr/table/build.rs` | 307 | `next_symbol` | `fn next_symbol<'a>(&self, rules: &'a [Rule]) -> Option<&'a Symbol> {` |
| `src/parser/glr/table/build.rs` | 314 | `first_of_sequence_bits` | `fn first_of_sequence_bits(` |
| `src/parser/glr/table/build.rs` | 345 | `first_bitsets` | `fn first_bitsets(grammar: &AnalyzedGrammar) -> Vec<BitSet> {` |
| `src/parser/glr/table/build.rs` | 349 | `union_lookaheads` | `fn union_lookaheads(item_set: &mut LR1ItemSet, core: LR1ItemCore, lookaheads: &BitSet) -> BitSet {` |
| `src/parser/glr/table/build.rs` | 360 | `lr1_closure` | `fn lr1_closure(` |
| `src/parser/glr/table/build.rs` | 402 | `item_set_key` | `fn item_set_key(items: &LR1ItemSet) -> Vec<(LR1ItemCore, BitSet)> {` |
| `src/parser/glr/table/build.rs` | 426 | `compute_transfer_items` | `fn compute_transfer_items(` |
| `src/parser/glr/table/build.rs` | 501 | `build_lr1_item_sets` | `fn build_lr1_item_sets(` |
| `src/parser/glr/table/build.rs` | 646 | `current_unique_reduce_len` | `fn current_unique_reduce_len(` |
| `src/parser/glr/table/build.rs` | 669 | `grammar_has_recursion` | `fn grammar_has_recursion(rules: &[Rule]) -> bool {` |
| `src/parser/glr/table/build.rs` | 683 | `dfs` | `fn dfs(node: usize, adj: &[Vec<usize>], color: &mut [u8]) -> bool {` |
| `src/parser/glr/table/build.rs` | 708 | `apply_local_forward_replace` | `fn apply_local_forward_replace(` |
| `src/parser/glr/table/build.rs` | 879 | `inline_zero_pop_reduces` | `fn inline_zero_pop_reduces(` |
| `src/parser/glr/table/build.rs` | 947 | `build_lr1_table` | `fn build_lr1_table(` |
| `src/parser/glr/table/build.rs` | 995 | `lr1_core_key` | `fn lr1_core_key(items: &LR1ItemSet) -> Vec<Item> {` |
| `src/parser/glr/table/build.rs` | 1002 | `build_ielr_table` | `fn build_ielr_table(` |
| `src/parser/glr/table/build.rs` | 1013 | `grouped_item_lookahead_counts` | `fn grouped_item_lookahead_counts(grammar: &AnalyzedGrammar) -> Vec<Vec<(u32, u32, u32, usize)>> {` |
| `src/parser/glr/table/build.rs` | 1033 | `multi_lookahead_grammar` | `fn multi_lookahead_grammar() -> AnalyzedGrammar {` |
| `src/parser/glr/table/build.rs` | 1066 | `grouped_lr1_items_merge_multiple_lookaheads_on_one_core` | `fn grouped_lr1_items_merge_multiple_lookaheads_on_one_core() {` |
| `src/parser/glr/table/build.rs` | 1082 | `grouped_lr1_items_still_emit_expected_lowered_shift_actions` | `fn grouped_lr1_items_still_emit_expected_lowered_shift_actions() {` |
| `src/parser/glr/table/mod.rs` | 24 | `default_action_rows_enabled` | `fn default_action_rows_enabled() -> bool {` |
| `src/parser/glr/table/mod.rs` | 29 | `GuardedShiftCellIndex` | `pub struct GuardedShiftCellIndex {` |
| `src/parser/glr/table/mod.rs` | 37 | `GLRTable` | `pub struct GLRTable {` |
| `src/parser/glr/table/mod.rs` | 67 | `TableAmbiguityKind` | `pub enum TableAmbiguityKind {` |
| `src/parser/glr/table/mod.rs` | 74 | `TableAmbiguity` | `pub struct TableAmbiguity {` |
| `src/parser/glr/table/mod.rs` | 81 | `guarded_stack_shift_constraints` | `fn guarded_stack_shift_constraints(` |
| `src/parser/glr/table/mod.rs` | 102 | `guarded_stack_shifts_overlap` | `fn guarded_stack_shifts_overlap(left: &GuardedStackShift, right: &GuardedStackShift) -> bool {` |
| `src/parser/glr/table/mod.rs` | 121 | `guarded_stack_shifts_are_ambiguous` | `fn guarded_stack_shifts_are_ambiguous(shifts: &[GuardedStackShift]) -> bool {` |
| `src/parser/glr/table/mod.rs` | 132 | `action_ambiguity` | `fn action_ambiguity(action: &Action) -> Option<(TableAmbiguityKind, usize)> {` |
| `src/parser/glr/table/mod.rs` | 154 | `build` | `pub fn build(grammar: &AnalyzedGrammar) -> Self {` |
| `src/parser/glr/table/mod.rs` | 159 | `terminal_bit` | `fn terminal_bit(&self, terminal: TerminalID) -> Option<usize> {` |
| `src/parser/glr/table/mod.rs` | 170 | `has_advance_rows` | `fn has_advance_rows(&self) -> bool {` |
| `src/parser/glr/table/mod.rs` | 174 | `rebuild_advance_rows_from_actions` | `pub(crate) fn rebuild_advance_rows_from_actions(&mut self) {` |
| `src/parser/glr/table/mod.rs` | 178 | `rebuild_guarded_shift_index` | `pub(crate) fn rebuild_guarded_shift_index(&mut self) {` |
| `src/parser/glr/table/mod.rs` | 231 | `guarded_shift_index` | `pub(crate) fn guarded_shift_index(` |
| `src/parser/glr/table/mod.rs` | 242 | `advance_row_allows` | `pub(crate) fn advance_row_allows(&self, state: u32, terminal: TerminalID) -> bool {` |
| `src/parser/glr/table/mod.rs` | 260 | `advance_row_intersects` | `pub(crate) fn advance_row_intersects(&self, state: u32, terminals: &BitSet) -> bool {` |
| `src/parser/glr/table/mod.rs` | 279 | `compress_default_action_rows` | `pub(crate) fn compress_default_action_rows(&mut self) {` |
| `src/parser/glr/table/mod.rs` | 286 | `action` | `pub fn action(&self, state: u32, terminal: TerminalID) -> Option<&Action> {` |
| `src/parser/glr/table/mod.rs` | 292 | `ambiguous_actions` | `pub fn ambiguous_actions(&self) -> Vec<TableAmbiguity> {` |
| `src/parser/glr/table/mod.rs` | 309 | `has_ambiguity` | `pub fn has_ambiguity(&self) -> bool {` |
| `src/parser/glr/table/mod.rs` | 317 | `goto_target` | `pub fn goto_target(&self, state: u32, nt: NonterminalID) -> Option<(u32, bool)> {` |
| `src/parser/glr/table/mod.rs` | 323 | `validate_structure` | `pub(super) fn validate_structure(&self, context: &str) {` |
| `src/parser/glr/table/mod.rs` | 443 | `nonterminal_display_name` | `pub fn nonterminal_display_name(&self, nt: NonterminalID) -> Option<&str> {` |
| `src/parser/glr/table/mod.rs` | 450 | `action_presence_rows` | `fn action_presence_rows(action: &[ActionRow], num_terminals: u32) -> Vec<BitSet> {` |
| `src/parser/glr/table/mod.rs` | 458 | `action_presence_row` | `fn action_presence_row(action_row: &ActionRow, num_terminals: u32) -> BitSet {` |
| `src/parser/glr/table/mod.rs` | 474 | `extend_advance_rows_from_actions` | `pub(crate) fn extend_advance_rows_from_actions(&mut self) {` |
| `src/parser/glr/table/mod.rs` | 492 | `build_test_table` | `pub(crate) fn build_test_table(` |
| `src/parser/glr/table/mod.rs` | 531 | `build_table_from_glrm` | `fn build_table_from_glrm(glrm: &str) -> GLRTable {` |
| `src/parser/glr/table/mod.rs` | 536 | `build_table_from_named_grammar` | `fn build_table_from_named_grammar(named: &NamedGrammar) -> GLRTable {` |
| `src/parser/glr/table/mod.rs` | 542 | `build_expr_nfa_optional_pair_suffix_grammar_with_value` | `fn build_expr_nfa_optional_pair_suffix_grammar_with_value(` |
| `src/parser/glr/table/mod.rs` | 589 | `glrm_recursive_array_colorpalette_minimal_stackshift_mre` | `fn glrm_recursive_array_colorpalette_minimal_stackshift_mre() -> &'static str {` |
| `src/parser/glr/table/mod.rs` | 599 | `build_direct_recursive_array_colorpalette_minimal_grammar` | `fn build_direct_recursive_array_colorpalette_minimal_grammar() -> NamedGrammar {` |
| `src/parser/glr/table/mod.rs` | 679 | `assert_table_has_all_pop1_stack_shift_ambiguity` | `fn assert_table_has_all_pop1_stack_shift_ambiguity(table: &GLRTable) {` |
| `src/parser/glr/table/mod.rs` | 703 | `assert_table_has_no_ambiguity` | `fn assert_table_has_no_ambiguity(table: &GLRTable) {` |
| `src/parser/glr/table/mod.rs` | 726 | `assert_table_has_no_all_pop1_stack_shift_ambiguity` | `fn assert_table_has_no_all_pop1_stack_shift_ambiguity(table: &GLRTable) {` |
| `src/parser/glr/table/mod.rs` | 750 | `ambiguous_actions_reports_split_and_stack_shift_fanout` | `fn ambiguous_actions_reports_split_and_stack_shift_fanout() {` |
| `src/parser/glr/table/mod.rs` | 809 | `guarded_stack_shifts_with_disjoint_guards_are_not_ambiguous` | `fn guarded_stack_shifts_with_disjoint_guards_are_not_ambiguous() {` |
| `src/parser/glr/table/mod.rs` | 844 | `validate_structure_panics_on_invalid_action_target` | `fn validate_structure_panics_on_invalid_action_target() {` |
| `src/parser/glr/table/mod.rs` | 857 | `expr_nfa_optional_pair_suffixes_with_value_ref_have_no_table_ambiguity` | `fn expr_nfa_optional_pair_suffixes_with_value_ref_have_no_table_ambiguity() {` |
| `src/parser/glr/table/mod.rs` | 874 | `glrm_recursive_array_colorpalette_minimal_does_not_reproduce_all_pop1_stack_shifts` | `fn glrm_recursive_array_colorpalette_minimal_does_not_reproduce_all_pop1_stack_shifts() {` |
| `src/parser/glr/table/mod.rs` | 881 | `direct_recursive_array_colorpalette_minimal_avoids_all_pop1_stack_shifts` | `fn direct_recursive_array_colorpalette_minimal_avoids_all_pop1_stack_shifts() {` |
| `src/parser/glr/table/optimize/guarded/action_exploration.rs` | 1 | `stack_effects_for_action` | `fn stack_effects_for_action(` |
| `src/parser/glr/table/optimize/guarded/action_exploration.rs` | 162 | `normalize_guarded_effects` | `fn normalize_guarded_effects(effects: &mut Vec<GuardedStackShift>) {` |
| `src/parser/glr/table/optimize/guarded/action_materialize.rs` | 1 | `stack_effect_action` | `fn stack_effect_action(table: &GLRTable, mut effects: Vec<GuardedStackShift>) -> Option<Action> {` |
| `src/parser/glr/table/optimize/guarded/action_materialize.rs` | 27 | `try_inline_action_to_stack_shifts` | `fn try_inline_action_to_stack_shifts(` |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 1 | `StackEffectFrame` | `struct StackEffectFrame {` |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 7 | `ReduceFrameResult` | `enum ReduceFrameResult {` |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 15 | `states_at_depth` | `fn states_at_depth(` |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 43 | `normalize_states` | `fn normalize_states(mut states: Vec<u32>) -> Vec<u32> {` |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 49 | `add_guard_to_frame` | `fn add_guard_to_frame(` |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 70 | `pop_frame` | `fn pop_frame(frame: &mut StackEffectFrame, pop: u32) {` |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 80 | `push_transition_to_frame` | `fn push_transition_to_frame(frame: &mut StackEffectFrame, target: u32, replace: bool) {` |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 93 | `frame_to_guarded_shift` | `fn frame_to_guarded_shift(frame: StackEffectFrame) -> GuardedStackShift {` |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 101 | `stack_effect_action_key` | `fn stack_effect_action_key(action: &Action) -> StackEffectActionKey {` |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 114 | `stack_effect_action_tag` | `fn stack_effect_action_tag(action: &Action) -> u8 {` |
| `src/parser/glr/table/optimize/guarded/reduce_frame.rs` | 1 | `apply_reduce_to_frame` | `fn apply_reduce_to_frame(` |
| `src/parser/glr/table/optimize/guarded/reduce_frame.rs` | 61 | `compose_guarded_shift_with_frame` | `fn compose_guarded_shift_with_frame(` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization.rs` | 1 | `normalize_stack_shifts` | `fn normalize_stack_shifts(shifts: &mut Vec<StackShift>) {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization.rs` | 6 | `canonicalize_stack_shift_predecessors_by_goto_superset` | `fn canonicalize_stack_shift_predecessors_by_goto_superset(table: &GLRTable, shifts: &mut [StackShift]) {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization.rs` | 42 | `goto_row_is_target_compatible_subset` | `fn goto_row_is_target_compatible_subset(table: &GLRTable, subset: u32, superset: u32) -> bool {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization.rs` | 53 | `stack_shift_action` | `fn stack_shift_action(mut shifts: Vec<StackShift>) -> Option<Action> {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization_tests.rs` | 5 | `table_with_stack_shifts` | `fn table_with_stack_shifts(` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization_tests.rs` | 34 | `stack_shifts_at_start` | `fn stack_shifts_at_start(table: &GLRTable) -> Vec<StackShift> {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization_tests.rs` | 42 | `canonicalizes_stack_shift_predecessor_to_goto_superset` | `fn canonicalizes_stack_shift_predecessor_to_goto_superset() {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization_tests.rs` | 72 | `leaves_stack_shift_predecessors_unchanged_when_canonicalization_is_disabled` | `fn leaves_stack_shift_predecessors_unchanged_when_canonicalization_is_disabled() {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization_tests.rs` | 108 | `does_not_canonicalize_stack_shift_predecessors_when_shared_goto_target_differs` | `fn does_not_canonicalize_stack_shift_predecessors_when_shared_goto_target_differs() {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization_tests.rs` | 144 | `does_not_canonicalize_empty_goto_row_to_nonempty_superset` | `fn does_not_canonicalize_empty_goto_row_to_nonempty_superset() {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization_tests.rs` | 177 | `canonicalizes_buried_middle_stack_shift_predecessor_to_goto_superset` | `fn canonicalizes_buried_middle_stack_shift_predecessor_to_goto_superset() {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization_tests.rs` | 207 | `does_not_canonicalize_top_pushed_state_even_when_goto_rows_are_compatible` | `fn does_not_canonicalize_top_pushed_state_even_when_goto_rows_are_compatible() {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization_tests.rs` | 243 | `reduce_frame_allows_origin_dependent_multiple_goto_targets` | `fn reduce_frame_allows_origin_dependent_multiple_goto_targets() {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization_tests.rs` | 297 | `reduce_frame_allows_origin_dependent_single_goto_target` | `fn reduce_frame_allows_origin_dependent_single_goto_target() {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization_tests.rs` | 340 | `inline_action_to_stack_shifts_keeps_multishift_replacement_reduce_chain` | `fn inline_action_to_stack_shifts_keeps_multishift_replacement_reduce_chain() {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization_tests.rs` | 400 | `inline_action_to_stack_shifts_handles_replace_shift_and_replace_goto` | `fn inline_action_to_stack_shifts_handles_replace_shift_and_replace_goto() {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization_tests.rs` | 460 | `inline_action_to_stack_shifts_guards_divergent_replace_gotos_by_predecessor` | `fn inline_action_to_stack_shifts_guards_divergent_replace_gotos_by_predecessor() {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization_tests.rs` | 523 | `compatible_goto_unit_destination_still_refuses_replace_goto` | `fn compatible_goto_unit_destination_still_refuses_replace_goto() {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization_tests.rs` | 547 | `suffix_quotient_collapses_same_pop_stack_shift_fanout` | `fn suffix_quotient_collapses_same_pop_stack_shift_fanout() {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization_tests.rs` | 610 | `suffix_quotient_preserves_guarded_stack_shift_guards` | `fn suffix_quotient_preserves_guarded_stack_shift_guards() {` |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization_tests.rs` | 666 | `suffix_quotient_rolls_back_nested_created_states_on_outer_failure` | `fn suffix_quotient_rolls_back_nested_created_states_on_outer_failure() {` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 1 | `CellUpdate` | `enum CellUpdate {` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 6 | `build_runtime_state_predecessors` | `fn build_runtime_state_predecessors(` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 65 | `subset_key` | `fn subset_key(subset: &BTreeSet<u32>) -> Vec<u32> {` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 69 | `union_state_subsets` | `fn union_state_subsets(` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 80 | `merge_shift_into_pending` | `fn merge_shift_into_pending(` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 115 | `merge_action_into_pending` | `fn merge_action_into_pending(` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 170 | `build_merged_action_row` | `fn build_merged_action_row(` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 207 | `build_merged_goto_row` | `fn build_merged_goto_row(` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 256 | `union_advance_rows` | `fn union_advance_rows(table: &GLRTable, subset: &BTreeSet<u32>) -> BitSet {` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 279 | `ensure_subset_state` | `fn ensure_subset_state(` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 351 | `refresh_merged_states_depending_on` | `fn refresh_merged_states_depending_on(` |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 400 | `unit_reduce_destination` | `fn unit_reduce_destination(` |
| `src/parser/glr/table/optimize/policy_adapter.rs` | 7 | `stack_shift_predecessor_canonicalization_enabled` | `fn stack_shift_predecessor_canonicalization_enabled() -> bool {` |
| `src/parser/glr/table/optimize/policy_adapter.rs` | 11 | `recognizer_suffix_quotient_enabled` | `fn recognizer_suffix_quotient_enabled() -> bool {` |
| `src/parser/glr/table/optimize/policy_adapter.rs` | 15 | `max_guarded_stack_effects` | `fn max_guarded_stack_effects() -> Option<usize> {` |
| `src/parser/glr/table/optimize/same_core_merge.rs` | 1 | `ActionSig` | `enum ActionSig {` |
| `src/parser/glr/table/optimize/same_core_merge.rs` | 15 | `RowSignature` | `struct RowSignature {` |
| `src/parser/glr/table/optimize/same_core_merge.rs` | 22 | `remap_action_to_partition` | `fn remap_action_to_partition(action: &Action, partition: &[u32]) -> ActionSig {` |
| `src/parser/glr/table/optimize/same_core_merge.rs` | 77 | `core_classes` | `fn core_classes(core_keys: &[Vec<Item>]) -> Vec<u32> {` |
| `src/parser/glr/table/optimize/same_core_merge.rs` | 94 | `refine_same_core_partition` | `fn refine_same_core_partition(table: &GLRTable, core_keys: &[Vec<Item>]) -> Vec<u32> {` |
| `src/parser/glr/table/optimize/same_core_merge.rs` | 138 | `merge_same_core_lr1_states` | `pub(super) fn merge_same_core_lr1_states(table: GLRTable, core_keys: &[Vec<Item>]) -> GLRTable {` |
| `src/parser/glr/table/optimize/stack_effect_keys.rs` | 1 | `StackEffectActionKey` | `enum StackEffectActionKey {` |
| `src/parser/glr/table/optimize/stack_effect_keys.rs` | 11 | `StackEffectKey` | `struct StackEffectKey {` |
| `src/parser/glr/table/optimize/stack_effect_keys.rs` | 20 | `StackEffectVisitKey` | `struct StackEffectVisitKey {` |
| `src/parser/glr/table/optimize/stack_effect_keys.rs` | 28 | `StackEffectResult` | `struct StackEffectResult {` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 1 | `SuffixQuotient` | `struct SuffixQuotient {` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 11 | `new` | `fn new() -> Self {` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 25 | `normalize_action` | `fn normalize_action(&mut self, table: &mut GLRTable, action: Action) -> Option<Action> {` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 60 | `quotient_effect_groups` | `fn quotient_effect_groups(` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 115 | `ensure_suffix_state` | `fn ensure_suffix_state(` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 194 | `build_suffix_action_row` | `fn build_suffix_action_row(` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 236 | `build_suffix_goto_row` | `fn build_suffix_goto_row(` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 270 | `action_row_has_multi_stack_shifts` | `fn action_row_has_multi_stack_shifts(row: &ActionRow) -> bool {` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 274 | `action_has_multi_stack_shifts` | `fn action_has_multi_stack_shifts(action: &Action) -> bool {` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 277 | `ensure_suffix_target` | `fn ensure_suffix_target(` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 289 | `collect_effects_for_suffix_action` | `fn collect_effects_for_suffix_action(` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 348 | `normalize_suffixes` | `fn normalize_suffixes(suffixes: &mut Vec<Vec<u32>>) {` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 353 | `normalize_guarded_effects_for_suffix_quotient` | `fn normalize_guarded_effects_for_suffix_quotient(effects: &mut Vec<GuardedStackShift>) {` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 367 | `apply_goto_to_suffix` | `fn apply_goto_to_suffix(suffix: &[u32], target: u32, replace: bool) -> Vec<u32> {` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 381 | `unguarded_suffix_effect` | `fn unguarded_suffix_effect(` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 389 | `guarded_suffix_effect` | `fn guarded_suffix_effect(` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 414 | `suffix_effect` | `fn suffix_effect(` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 444 | `row_fingerprint` | `fn row_fingerprint(` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 474 | `rows_equal` | `fn rows_equal(` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 493 | `push_reachable_state` | `fn push_reachable_state(state: u32, reachable: &mut [bool], stack: &mut Vec<u32>) {` |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 503 | `push_action_targets` | `fn push_action_targets(action: &Action, reachable: &mut [bool], stack: &mut Vec<u32>) {` |
| `src/parser/glr/table/optimize/table_passes.rs` | 2 | `canonicalize_stack_shift_predecessors` | `pub(super) fn canonicalize_stack_shift_predecessors(&mut self) {` |
| `src/parser/glr/table/optimize/table_passes.rs` | 8 | `canonicalize_stack_shift_predecessors_with_enabled` | `fn canonicalize_stack_shift_predecessors_with_enabled(&mut self, enabled: bool) {` |
| `src/parser/glr/table/optimize/table_passes.rs` | 39 | `merge_identical_rows` | `pub(super) fn merge_identical_rows(&mut self) {` |
| `src/parser/glr/table/optimize/table_passes.rs` | 125 | `prune_unreachable_states` | `pub(super) fn prune_unreachable_states(&mut self) {` |
| `src/parser/glr/table/optimize/table_passes.rs` | 197 | `collapse_sr_unit_reductions_with_compatible_gotos` | `pub(super) fn collapse_sr_unit_reductions_with_compatible_gotos(&mut self) {` |
| `src/parser/glr/table/optimize/table_passes.rs` | 297 | `merge_recognizer_equivalent` | `pub(super) fn merge_recognizer_equivalent(&mut self) {` |
| `src/parser/glr/table/optimize/table_passes.rs` | 758 | `quotient_recognizer_stack_suffixes` | `pub(super) fn quotient_recognizer_stack_suffixes(&mut self) {` |
| `src/parser/glr/table/optimize/unit_reductions.rs` | 1 | `try_inline_unit_reductions_for_cell` | `fn try_inline_unit_reductions_for_cell(` |
| `src/parser/glr/table/optimize/unit_reductions.rs` | 49 | `try_inline_unit_reductions_for_cell_inner` | `fn try_inline_unit_reductions_for_cell_inner(` |
| `src/parser/glr/table/optimize/unit_reductions.rs` | 151 | `remap_action_targets` | `fn remap_action_targets(action: &Action, mapping: &[u32]) -> Action {` |
| `src/parser/glr/table/options.rs` | 14 | `GLRTableOptions` | `pub(crate) struct GLRTableOptions {` |
| `src/parser/glr/table/options.rs` | 41 | `from_env` | `pub(crate) fn from_env() -> Self {` |
| `src/parser/glr/table/options.rs` | 72 | `table_options_from_env` | `pub(crate) fn table_options_from_env() -> GLRTableOptions {` |
| `src/parser/glr/table/options.rs` | 76 | `env_usize` | `fn env_usize(name: &str, default: usize) -> usize {` |
| `src/parser/glr/table/options.rs` | 84 | `env_optional_usize` | `fn env_optional_usize(name: &str) -> Option<usize> {` |
| `src/parser/glr/table/options.rs` | 91 | `env_flag_enabled` | `fn env_flag_enabled(name: &str, default: bool) -> bool {` |
| `src/parser/glr/table/row.rs` | 17 | `SparseRow` | `pub(crate) enum SparseRow<K: Copy + Eq + Hash, V: Clone> {` |
| `src/parser/glr/table/row.rs` | 23 | `default` | `fn default() -> Self {` |
| `src/parser/glr/table/row.rs` | 30 | `len` | `pub(crate) fn len(&self) -> usize {` |
| `src/parser/glr/table/row.rs` | 38 | `is_empty` | `pub(crate) fn is_empty(&self) -> bool {` |
| `src/parser/glr/table/row.rs` | 43 | `get` | `pub(crate) fn get(&self, key: &K) -> Option<&V> {` |
| `src/parser/glr/table/row.rs` | 54 | `get_mut` | `pub(crate) fn get_mut(&mut self, key: &K) -> Option<&mut V> {` |
| `src/parser/glr/table/row.rs` | 64 | `insert` | `pub(crate) fn insert(&mut self, key: K, value: V) -> Option<V> {` |
| `src/parser/glr/table/row.rs` | 89 | `remove` | `pub(crate) fn remove(&mut self, key: &K) -> Option<V> {` |
| `src/parser/glr/table/row.rs` | 100 | `contains_key` | `pub(crate) fn contains_key(&self, key: &K) -> bool {` |
| `src/parser/glr/table/row.rs` | 105 | `iter` | `pub(crate) fn iter(&self) -> SparseRowIter<'_, K, V> {` |
| `src/parser/glr/table/row.rs` | 113 | `keys` | `pub(crate) fn keys(&self) -> SparseRowKeys<'_, K, V> {` |
| `src/parser/glr/table/row.rs` | 121 | `values` | `pub(crate) fn values(&self) -> SparseRowValues<'_, K, V> {` |
| `src/parser/glr/table/row.rs` | 130 | `eq` | `fn eq(&self, other: &Self) -> bool {` |
| `src/parser/glr/table/row.rs` | 145 | `serialize` | `fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>` |
| `src/parser/glr/table/row.rs` | 162 | `deserialize` | `fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>` |
| `src/parser/glr/table/row.rs` | 166 | `SparseRowVisitor` | `struct SparseRowVisitor<K, V>(PhantomData<(K, V)>);` |
| `src/parser/glr/table/row.rs` | 173 | `Value` | `type Value = SparseRow<K, V>;` |
| `src/parser/glr/table/row.rs` | 175 | `expecting` | `fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {` |
| `src/parser/glr/table/row.rs` | 179 | `visit_map` | `fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>` |
| `src/parser/glr/table/row.rs` | 196 | `Item` | `type Item = (&'a K, &'a V);` |
| `src/parser/glr/table/row.rs` | 197 | `IntoIter` | `type IntoIter = SparseRowIter<'a, K, V>;` |
| `src/parser/glr/table/row.rs` | 199 | `into_iter` | `fn into_iter(self) -> Self::IntoIter {` |
| `src/parser/glr/table/row.rs` | 205 | `Output` | `type Output = V;` |
| `src/parser/glr/table/row.rs` | 207 | `index` | `fn index(&self, index: &K) -> &Self::Output {` |
| `src/parser/glr/table/row.rs` | 213 | `from_iter` | `fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {` |
| `src/parser/glr/table/row.rs` | 222 | `SparseRowIter` | `pub(crate) enum SparseRowIter<'a, K: Copy + Eq + Hash, V: Clone> {` |
| `src/parser/glr/table/row.rs` | 228 | `Item` | `type Item = (&'a K, &'a V);` |
| `src/parser/glr/table/row.rs` | 230 | `next` | `fn next(&mut self) -> Option<Self::Item> {` |
| `src/parser/glr/table/row.rs` | 238 | `SparseRowKeys` | `pub(crate) enum SparseRowKeys<'a, K: Copy + Eq + Hash, V: Clone> {` |
| `src/parser/glr/table/row.rs` | 244 | `Item` | `type Item = &'a K;` |
| `src/parser/glr/table/row.rs` | 246 | `next` | `fn next(&mut self) -> Option<Self::Item> {` |
| `src/parser/glr/table/row.rs` | 254 | `SparseRowValues` | `pub(crate) enum SparseRowValues<'a, K: Copy + Eq + Hash, V: Clone> {` |
| `src/parser/glr/table/row.rs` | 260 | `Item` | `type Item = &'a V;` |
| `src/parser/glr/table/row.rs` | 262 | `next` | `fn next(&mut self) -> Option<Self::Item> {` |
| `src/parser/glr/table/row.rs` | 271 | `ActionRow` | `pub(crate) enum ActionRow {` |
| `src/parser/glr/table/row.rs` | 281 | `default` | `fn default() -> Self {` |
| `src/parser/glr/table/row.rs` | 288 | `is_default_compressed` | `pub(crate) fn is_default_compressed(&self) -> bool {` |
| `src/parser/glr/table/row.rs` | 293 | `len` | `pub(crate) fn len(&self) -> usize {` |
| `src/parser/glr/table/row.rs` | 308 | `is_empty` | `pub(crate) fn is_empty(&self) -> bool {` |
| `src/parser/glr/table/row.rs` | 313 | `get` | `pub(crate) fn get(&self, key: &TerminalID) -> Option<&Action> {` |
| `src/parser/glr/table/row.rs` | 333 | `get_mut` | `pub(crate) fn get_mut(&mut self, key: &TerminalID) -> Option<&mut Action> {` |
| `src/parser/glr/table/row.rs` | 343 | `insert` | `pub(crate) fn insert(&mut self, key: TerminalID, value: Action) -> Option<Action> {` |
| `src/parser/glr/table/row.rs` | 372 | `remove` | `pub(crate) fn remove(&mut self, key: &TerminalID) -> Option<Action> {` |
| `src/parser/glr/table/row.rs` | 401 | `contains_key` | `pub(crate) fn contains_key(&self, key: &TerminalID) -> bool {` |
| `src/parser/glr/table/row.rs` | 406 | `iter` | `pub(crate) fn iter(&self) -> ActionRowIter<'_> {` |
| `src/parser/glr/table/row.rs` | 423 | `keys` | `pub(crate) fn keys(&self) -> ActionRowKeys<'_> {` |
| `src/parser/glr/table/row.rs` | 428 | `values` | `pub(crate) fn values(&self) -> ActionRowValues<'_> {` |
| `src/parser/glr/table/row.rs` | 432 | `compress_default` | `pub(crate) fn compress_default(&mut self, num_terminals: TerminalID) {` |
| `src/parser/glr/table/row.rs` | 489 | `expand_default_to_sparse` | `fn expand_default_to_sparse(&mut self) {` |
| `src/parser/glr/table/row.rs` | 519 | `eq` | `fn eq(&self, other: &Self) -> bool {` |
| `src/parser/glr/table/row.rs` | 531 | `Item` | `type Item = (TerminalID, &'a Action);` |
| `src/parser/glr/table/row.rs` | 532 | `IntoIter` | `type IntoIter = ActionRowIter<'a>;` |
| `src/parser/glr/table/row.rs` | 534 | `into_iter` | `fn into_iter(self) -> Self::IntoIter {` |
| `src/parser/glr/table/row.rs` | 540 | `Output` | `type Output = Action;` |
| `src/parser/glr/table/row.rs` | 542 | `index` | `fn index(&self, index: &TerminalID) -> &Self::Output {` |
| `src/parser/glr/table/row.rs` | 548 | `from_iter` | `fn from_iter<T: IntoIterator<Item = (TerminalID, Action)>>(iter: T) -> Self {` |
| `src/parser/glr/table/row.rs` | 553 | `ActionRowIter` | `pub(crate) enum ActionRowIter<'a> {` |
| `src/parser/glr/table/row.rs` | 559 | `Item` | `type Item = (TerminalID, &'a Action);` |
| `src/parser/glr/table/row.rs` | 561 | `next` | `fn next(&mut self) -> Option<Self::Item> {` |
| `src/parser/glr/table/row.rs` | 569 | `DefaultActionRowIter` | `pub(crate) struct DefaultActionRowIter<'a> {` |
| `src/parser/glr/table/row.rs` | 577 | `Item` | `type Item = (TerminalID, &'a Action);` |
| `src/parser/glr/table/row.rs` | 579 | `next` | `fn next(&mut self) -> Option<Self::Item> {` |
| `src/parser/glr/table/row.rs` | 593 | `ActionRowKeys` | `pub(crate) struct ActionRowKeys<'a> {` |
| `src/parser/glr/table/row.rs` | 598 | `Item` | `type Item = TerminalID;` |
| `src/parser/glr/table/row.rs` | 600 | `next` | `fn next(&mut self) -> Option<Self::Item> {` |
| `src/parser/glr/table/row.rs` | 605 | `ActionRowValues` | `pub(crate) struct ActionRowValues<'a> {` |
| `src/parser/glr/table/row.rs` | 610 | `Item` | `type Item = &'a Action;` |
| `src/parser/glr/table/row.rs` | 612 | `next` | `fn next(&mut self) -> Option<Self::Item> {` |
| `src/parser/glr/table/row.rs` | 617 | `GotoRow` | `pub(crate) type GotoRow = SparseRow<NonterminalID, (u32, bool)>;` |
| `src/parser/glr/table/row.rs` | 624 | `shift` | `fn shift(target: u32) -> Action {` |
| `src/parser/glr/table/row.rs` | 629 | `default_row_lookup_and_iter_handle_null_and_override_exceptions` | `fn default_row_lookup_and_iter_handle_null_and_override_exceptions() {` |
| `src/parser/glr/table/row.rs` | 654 | `default_row_insert_and_remove_track_null_exceptions` | `fn default_row_insert_and_remove_track_null_exceptions() {` |
| `src/parser/glr/table/row.rs` | 671 | `default_row_keys_iterate_effective_present_terminals` | `fn default_row_keys_iterate_effective_present_terminals() {` |
| `src/parser/glr/table/row.rs` | 685 | `compress_default_prefers_default_row_when_structurally_smaller` | `fn compress_default_prefers_default_row_when_structurally_smaller() {` |
| `src/parser/glr/table/row.rs` | 702 | `table_compression_preserves_lookup_equivalence` | `fn table_compression_preserves_lookup_equivalence() {` |
| `src/parser/gss/mod.rs` | 10 | `SV` | `type SV<T> = DynStackVec<T>;` |
| `src/parser/gss/mod.rs` | 12 | `Merge` | `pub trait Merge: Clone {` |
| `src/parser/gss/mod.rs` | 13 | `merge` | `fn merge(&self, other: &Self) -> Self;` |
| `src/parser/gss/mod.rs` | 15 | `subsumes` | `fn subsumes(&self, _other: &Self) -> bool {` |
| `src/parser/gss/mod.rs` | 24 | `CompactMap` | `enum CompactMap<K: Clone + Eq + Hash, V: Clone> {` |
| `src/parser/gss/mod.rs` | 31 | `new` | `fn new() -> Self {` |
| `src/parser/gss/mod.rs` | 36 | `unit` | `fn unit(key: K, value: V) -> Self {` |
| `src/parser/gss/mod.rs` | 43 | `len` | `fn len(&self) -> usize {` |
| `src/parser/gss/mod.rs` | 51 | `is_empty` | `fn is_empty(&self) -> bool {` |
| `src/parser/gss/mod.rs` | 59 | `get` | `fn get(&self, key: &K) -> Option<&V> {` |
| `src/parser/gss/mod.rs` | 67 | `get_mut` | `fn get_mut(&mut self, key: &K) -> Option<&mut V> {` |
| `src/parser/gss/mod.rs` | 74 | `insert` | `fn insert(&mut self, key: K, value: V) -> Option<V> {` |
| `src/parser/gss/mod.rs` | 102 | `contains_key` | `fn contains_key(&self, key: &K) -> bool {` |
| `src/parser/gss/mod.rs` | 109 | `keys` | `fn keys(&self) -> CompactMapKeys<'_, K, V> {` |
| `src/parser/gss/mod.rs` | 116 | `ptr_eq` | `fn ptr_eq(&self, other: &Self) -> bool {` |
| `src/parser/gss/mod.rs` | 123 | `remove` | `fn remove(&mut self, key: &K) -> Option<V> {` |
| `src/parser/gss/mod.rs` | 136 | `values` | `fn values(&self) -> CompactMapValues<'_, K, V> {` |
| `src/parser/gss/mod.rs` | 143 | `iter` | `fn iter(&self) -> CompactMapIter<'_, K, V> {` |
| `src/parser/gss/mod.rs` | 151 | `CompactMapKeys` | `enum CompactMapKeys<'a, K, V> {` |
| `src/parser/gss/mod.rs` | 157 | `Item` | `type Item = &'a K;` |
| `src/parser/gss/mod.rs` | 158 | `next` | `fn next(&mut self) -> Option<Self::Item> {` |
| `src/parser/gss/mod.rs` | 164 | `size_hint` | `fn size_hint(&self) -> (usize, Option<usize>) {` |
| `src/parser/gss/mod.rs` | 172 | `CompactMapValues` | `enum CompactMapValues<'a, K, V> {` |
| `src/parser/gss/mod.rs` | 178 | `Item` | `type Item = &'a V;` |
| `src/parser/gss/mod.rs` | 179 | `next` | `fn next(&mut self) -> Option<Self::Item> {` |
| `src/parser/gss/mod.rs` | 185 | `size_hint` | `fn size_hint(&self) -> (usize, Option<usize>) {` |
| `src/parser/gss/mod.rs` | 193 | `CompactMapIter` | `enum CompactMapIter<'a, K, V> {` |
| `src/parser/gss/mod.rs` | 199 | `Item` | `type Item = (&'a K, &'a V);` |
| `src/parser/gss/mod.rs` | 200 | `next` | `fn next(&mut self) -> Option<Self::Item> {` |
| `src/parser/gss/mod.rs` | 206 | `size_hint` | `fn size_hint(&self) -> (usize, Option<usize>) {` |
| `src/parser/gss/mod.rs` | 215 | `Item` | `type Item = (&'a K, &'a V);` |
| `src/parser/gss/mod.rs` | 216 | `IntoIter` | `type IntoIter = CompactMapIter<'a, K, V>;` |
| `src/parser/gss/mod.rs` | 217 | `into_iter` | `fn into_iter(self) -> Self::IntoIter {` |
| `src/parser/gss/mod.rs` | 226 | `CompactOrdMap` | `enum CompactOrdMap<V: Clone> {` |
| `src/parser/gss/mod.rs` | 233 | `new` | `fn new() -> Self {` |
| `src/parser/gss/mod.rs` | 238 | `unit` | `fn unit(key: u32, value: V) -> Self {` |
| `src/parser/gss/mod.rs` | 245 | `len` | `fn len(&self) -> usize {` |
| `src/parser/gss/mod.rs` | 253 | `is_empty` | `fn is_empty(&self) -> bool {` |
| `src/parser/gss/mod.rs` | 261 | `get` | `fn get(&self, key: &u32) -> Option<&V> {` |
| `src/parser/gss/mod.rs` | 268 | `insert` | `fn insert(&mut self, key: u32, value: V) -> Option<V> {` |
| `src/parser/gss/mod.rs` | 295 | `keys` | `fn keys(&self) -> CompactOrdMapKeys<'_, V> {` |
| `src/parser/gss/mod.rs` | 302 | `get_max` | `fn get_max(&self) -> Option<(&u32, &V)> {` |
| `src/parser/gss/mod.rs` | 311 | `iter` | `fn iter(&self) -> CompactOrdMapIter<'_, V> {` |
| `src/parser/gss/mod.rs` | 318 | `values` | `fn values(&self) -> CompactOrdMapValues<'_, V> {` |
| `src/parser/gss/mod.rs` | 326 | `CompactOrdMapIter` | `enum CompactOrdMapIter<'a, V> {` |
| `src/parser/gss/mod.rs` | 332 | `Item` | `type Item = (&'a u32, &'a V);` |
| `src/parser/gss/mod.rs` | 333 | `next` | `fn next(&mut self) -> Option<Self::Item> {` |
| `src/parser/gss/mod.rs` | 339 | `size_hint` | `fn size_hint(&self) -> (usize, Option<usize>) {` |
| `src/parser/gss/mod.rs` | 347 | `CompactOrdMapValues` | `enum CompactOrdMapValues<'a, V> {` |
| `src/parser/gss/mod.rs` | 353 | `Item` | `type Item = &'a V;` |
| `src/parser/gss/mod.rs` | 354 | `next` | `fn next(&mut self) -> Option<Self::Item> {` |
| `src/parser/gss/mod.rs` | 360 | `size_hint` | `fn size_hint(&self) -> (usize, Option<usize>) {` |
| `src/parser/gss/mod.rs` | 369 | `from_iter` | `fn from_iter<I: IntoIterator<Item = (u32, V)>>(iter: I) -> Self {` |
| `src/parser/gss/mod.rs` | 378 | `CompactOrdMapKeys` | `enum CompactOrdMapKeys<'a, V> {` |
| `src/parser/gss/mod.rs` | 384 | `Item` | `type Item = &'a u32;` |
| `src/parser/gss/mod.rs` | 385 | `next` | `fn next(&mut self) -> Option<Self::Item> {` |
| `src/parser/gss/mod.rs` | 391 | `size_hint` | `fn size_hint(&self) -> (usize, Option<usize>) {` |
| `src/parser/gss/mod.rs` | 400 | `Item` | `type Item = (&'a u32, &'a V);` |
| `src/parser/gss/mod.rs` | 401 | `IntoIter` | `type IntoIter = CompactOrdMapIter<'a, V>;` |
| `src/parser/gss/mod.rs` | 402 | `into_iter` | `fn into_iter(self) -> Self::IntoIter {` |
| `src/parser/gss/mod.rs` | 407 | `Children` | `type Children<T, N> = CompactMap<T, CompactOrdMap<Arc<N>>>;` |
| `src/parser/gss/mod.rs` | 415 | `Segment` | `struct Segment<T: Clone + Eq + Hash> {` |
| `src/parser/gss/mod.rs` | 424 | `clone` | `fn clone(&self) -> Self {` |
| `src/parser/gss/mod.rs` | 436 | `eq` | `fn eq(&self, other: &Self) -> bool {` |
| `src/parser/gss/mod.rs` | 447 | `Lower` | `enum Lower<T: Clone + Eq + Hash> {` |
| `src/parser/gss/mod.rs` | 461 | `lower_node_id` | `fn lower_node_id<T: Clone + Eq + Hash>(node: &Arc<Lower<T>>) -> usize {` |
| `src/parser/gss/mod.rs` | 470 | `empty` | `fn empty(&self) -> bool {` |
| `src/parser/gss/mod.rs` | 478 | `max_depth` | `fn max_depth(&self) -> u32 {` |
| `src/parser/gss/mod.rs` | 486 | `segments_len` | `fn segments_len(&self) -> usize {` |
| `src/parser/gss/mod.rs` | 497 | `chain_step` | `fn chain_step(&self) -> Option<(&Arc<Lower<T>>, usize)> {` |
| `src/parser/gss/mod.rs` | 515 | `append_chain_values_top_first` | `fn append_chain_values_top_first(&self, out: &mut SmallVec<[T; 16]>) {` |
| `src/parser/gss/mod.rs` | 530 | `children` | `fn children(&self) -> Children<T, Lower<T>> {` |
| `src/parser/gss/mod.rs` | 542 | `into_parts` | `fn into_parts(self) -> (Children<T, Lower<T>>, bool, u32) {` |
| `src/parser/gss/mod.rs` | 573 | `children_len` | `fn children_len(&self) -> usize {` |
| `src/parser/gss/mod.rs` | 582 | `children_is_empty` | `fn children_is_empty(&self) -> bool {` |
| `src/parser/gss/mod.rs` | 590 | `children_contains_key` | `fn children_contains_key(&self, key: &T) -> bool {` |
| `src/parser/gss/mod.rs` | 598 | `ensure_general` | `fn ensure_general(&mut self) {` |
| `src/parser/gss/mod.rs` | 612 | `is_segment` | `fn is_segment(&self) -> bool {` |
| `src/parser/gss/mod.rs` | 619 | `segment_top_value` | `fn segment_top_value(&self) -> &T {` |
| `src/parser/gss/mod.rs` | 629 | `segment_next` | `fn segment_next(&self) -> &Arc<Lower<T>> {` |
| `src/parser/gss/mod.rs` | 639 | `segment_values` | `fn segment_values(&self) -> &SV<T> {` |
| `src/parser/gss/mod.rs` | 648 | `segment_rest_arc` | `fn segment_rest_arc(&self) -> Arc<Lower<T>> {` |
| `src/parser/gss/mod.rs` | 670 | `Interface` | `struct Interface<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {` |
| `src/parser/gss/mod.rs` | 676 | `UpperBranch` | `struct UpperBranch<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {` |
| `src/parser/gss/mod.rs` | 683 | `Upper` | `enum Upper<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {` |
| `src/parser/gss/mod.rs` | 689 | `max_depth` | `fn max_depth(&self) -> u32 {` |
| `src/parser/gss/mod.rs` | 696 | `children_keys` | `fn children_keys(&self) -> SmallVec<[T; 8]> {` |
| `src/parser/gss/mod.rs` | 706 | `single_child_key` | `fn single_child_key(&self) -> Option<T> {` |
| `src/parser/gss/mod.rs` | 728 | `single_child_key_without_empty` | `fn single_child_key_without_empty(&self) -> Option<T> {` |
| `src/parser/gss/mod.rs` | 752 | `LeveledGSSSummary` | `pub struct LeveledGSSSummary {` |
| `src/parser/gss/mod.rs` | 768 | `merge_optional_acc` | `fn merge_optional_acc<A: Merge + Clone>(a: &Option<A>, b: &Option<A>) -> Option<A> {` |
| `src/parser/gss/mod.rs` | 777 | `max_depth_from_children` | `fn max_depth_from_children<T, N, F>(children: &Children<T, N>, depth_of: F) -> u32` |
| `src/parser/gss/mod.rs` | 790 | `merge_children` | `fn merge_children<T, N, F>(c1: &Children<T, N>, c2: &Children<T, N>, merge_fn: F) -> Children<T, N>` |
| `src/parser/gss/mod.rs` | 818 | `new_lower` | `fn new_lower<T: Clone + Eq + Hash>(children: Children<T, Lower<T>>, empty: bool) -> Arc<Lower<T>> {` |
| `src/parser/gss/mod.rs` | 840 | `new_segment` | `fn new_segment<T: Clone + Eq + Hash>(values: SV<T>, next: Arc<Lower<T>>) -> Arc<Lower<T>> {` |
| `src/parser/gss/mod.rs` | 866 | `new_interface` | `fn new_interface<T, A>(inner: Arc<Lower<T>>, acc: A) -> Arc<Upper<T, A>>` |
| `src/parser/gss/mod.rs` | 874 | `new_branch` | `fn new_branch<T, A>(` |
| `src/parser/gss/mod.rs` | 890 | `empty_upper_inner` | `fn empty_upper_inner<T, A>() -> Arc<Upper<T, A>>` |
| `src/parser/gss/mod.rs` | 900 | `truncate_lower` | `fn truncate_lower<T: Clone + Eq + Hash>(` |
| `src/parser/gss/mod.rs` | 975 | `truncate_upper` | `fn truncate_upper<T, A>(` |
| `src/parser/gss/mod.rs` | 1065 | `merge_lower` | `fn merge_lower<T: Clone + Eq + Hash>(l1: &Arc<Lower<T>>, l2: &Arc<Lower<T>>) -> Arc<Lower<T>> {` |
| `src/parser/gss/mod.rs` | 1120 | `interface_to_upperbranch` | `fn interface_to_upperbranch<T, A>(it: &Arc<Interface<T, A>>) -> Arc<UpperBranch<T, A>>` |
| `src/parser/gss/mod.rs` | 1162 | `nonempty_deterministic_top_step` | `fn nonempty_deterministic_top_step<T>(lower: &Arc<Lower<T>>) -> Option<(T, Arc<Lower<T>>)>` |
| `src/parser/gss/mod.rs` | 1183 | `shared_nonempty_deterministic_prefix` | `fn shared_nonempty_deterministic_prefix<T>(` |
| `src/parser/gss/mod.rs` | 1214 | `merge_upperbranches` | `fn merge_upperbranches<T, A>(` |
| `src/parser/gss/mod.rs` | 1235 | `merge_interfaces` | `fn merge_interfaces<T, A>(a: &Arc<Interface<T, A>>, b: &Arc<Interface<T, A>>) -> Arc<Upper<T, A>>` |
| `src/parser/gss/mod.rs` | 1282 | `merge_upper` | `fn merge_upper<T, A>(u1: &Arc<Upper<T, A>>, u2: &Arc<Upper<T, A>>) -> Arc<Upper<T, A>>` |
| `src/parser/gss/mod.rs` | 1300 | `try_promote` | `fn try_promote<T, A>(node: &Arc<Upper<T, A>>) -> Arc<Upper<T, A>>` |
| `src/parser/gss/mod.rs` | 1360 | `empty_upper` | `fn empty_upper<T, A>() -> LeveledGSS<T, A>` |
| `src/parser/gss/mod.rs` | 1384 | `LeveledGSS` | `pub struct LeveledGSS<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {` |
| `src/parser/gss/mod.rs` | 1407 | `VirtualStack` | `pub struct VirtualStack<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {` |
| `src/parser/gss/mod.rs` | 1416 | `PushMode` | `enum PushMode {` |
| `src/parser/gss/mod.rs` | 1426 | `push_mode` | `fn push_mode() -> PushMode {` |
| `src/parser/gss/mod.rs` | 1438 | `top` | `pub fn top(&self) -> Option<&T> {` |
| `src/parser/gss/mod.rs` | 1445 | `top_after_popping` | `pub fn top_after_popping(&self, mut remaining: usize) -> Option<&T> {` |
| `src/parser/gss/mod.rs` | 1473 | `flush_pending` | `fn flush_pending(&mut self) {` |
| `src/parser/gss/mod.rs` | 1481 | `realize_push` | `fn realize_push(&mut self, value: T) {` |
| `src/parser/gss/mod.rs` | 1503 | `pop` | `pub fn pop(&mut self, mut remaining: usize) -> usize {` |
| `src/parser/gss/mod.rs` | 1539 | `push` | `pub fn push(&mut self, value: T) {` |
| `src/parser/gss/mod.rs` | 1546 | `parent_of_top` | `pub fn parent_of_top(&self) -> Option<T> {` |
| `src/parser/gss/mod.rs` | 1552 | `replace_top` | `pub fn replace_top(&mut self, value: T) -> bool {` |
| `src/parser/gss/mod.rs` | 1573 | `len` | `pub fn len(&self) -> usize {` |
| `src/parser/gss/mod.rs` | 1578 | `into_gss` | `pub fn into_gss(mut self) -> LeveledGSS<T, A> {` |
| `src/parser/gss/mod.rs` | 1590 | `into_gss_after_popping` | `pub fn into_gss_after_popping(mut self, n: usize) -> LeveledGSS<T, A> {` |
| `src/parser/gss/mod.rs` | 1601 | `into_gss_after_popping_and_pushing_branches` | `pub fn into_gss_after_popping_and_pushing_branches<'a, I>(` |
| `src/parser/gss/mod.rs` | 1652 | `into_gss_after_popping_and_pushing_single_branches` | `pub fn into_gss_after_popping_and_pushing_single_branches<'a, I>(` |
| `src/parser/gss/mod.rs` | 1693 | `eq` | `fn eq(&self, other: &Self) -> bool {` |
| `src/parser/gss/mod.rs` | 1701 | `fmt` | `fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {` |
| `src/parser/gss/mod.rs` | 1710 | `ptr_eq` | `pub fn ptr_eq(&self, other: &Self) -> bool {` |
| `src/parser/gss/mod.rs` | 1714 | `ptr_key` | `pub fn ptr_key(&self) -> usize {` |
| `src/parser/gss/mod.rs` | 1718 | `single_interface_lower_id` | `pub(crate) fn single_interface_lower_id(&self) -> Option<usize> {` |
| `src/parser/gss/mod.rs` | 1727 | `empty` | `pub fn empty() -> Self {` |
| `src/parser/gss/mod.rs` | 1731 | `from_stacks` | `pub fn from_stacks(stacks: &[(Vec<T>, A)]) -> Self {` |
| `src/parser/gss/mod.rs` | 1743 | `Entry` | `struct Entry<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash> {` |
| `src/parser/gss/mod.rs` | 1749 | `default` | `fn default() -> Self {` |
| `src/parser/gss/mod.rs` | 1780 | `build_lower` | `fn build_lower<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(` |
| `src/parser/gss/mod.rs` | 1796 | `build_upper` | `fn build_upper<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(` |
| `src/parser/gss/mod.rs` | 1856 | `from_single_stack` | `pub fn from_single_stack(values: Vec<T>, acc: A) -> Self {` |
| `src/parser/gss/mod.rs` | 1866 | `to_stacks` | `pub fn to_stacks(&self) -> Vec<(Vec<T>, A)> {` |
| `src/parser/gss/mod.rs` | 1869 | `dfs_lower` | `fn dfs_lower<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(` |
| `src/parser/gss/mod.rs` | 1903 | `dfs_upper` | `fn dfs_upper<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(` |
| `src/parser/gss/mod.rs` | 1963 | `apply_stack_effects_to_single_concrete_path` | `pub fn apply_stack_effects_to_single_concrete_path<'a, I>(` |
| `src/parser/gss/mod.rs` | 2034 | `apply_shared_pop_push_branches` | `pub fn apply_shared_pop_push_branches<'a, I>(` |
| `src/parser/gss/mod.rs` | 2047 | `apply_shared_pop_push_single_branches` | `pub fn apply_shared_pop_push_single_branches<'a, I>(` |
| `src/parser/gss/mod.rs` | 2060 | `apply_guarded_stack_effects_to_single_concrete_path` | `pub fn apply_guarded_stack_effects_to_single_concrete_path<'a, I, G>(` |
| `src/parser/gss/mod.rs` | 2125 | `push` | `pub fn push(&self, value: T) -> Self {` |
| `src/parser/gss/mod.rs` | 2153 | `remap_top_values` | `pub fn remap_top_values<I>(&self, shifts: I) -> Self` |
| `src/parser/gss/mod.rs` | 2250 | `remap_top_values_owned` | `pub fn remap_top_values_owned<I>(self, shifts: I) -> Self` |
| `src/parser/gss/mod.rs` | 2349 | `try_apply_selective_top_pure_shifts` | `pub fn try_apply_selective_top_pure_shifts<I>(&self, shifts: I) -> Option<Self>` |
| `src/parser/gss/mod.rs` | 2366 | `lower_with_top` | `fn lower_with_top<T: Clone + Eq + Hash>(` |
| `src/parser/gss/mod.rs` | 2389 | `apply_top_pure_shifts` | `pub fn apply_top_pure_shifts<I>(&self, shifts: I) -> Self` |
| `src/parser/gss/mod.rs` | 2398 | `insert_lower_child` | `fn insert_lower_child<T: Clone + Eq + Hash>(` |
| `src/parser/gss/mod.rs` | 2489 | `absorb_push_same_acc` | `pub fn absorb_push_same_acc(self, value: T, base: &Self) -> Self {` |
| `src/parser/gss/mod.rs` | 2503 | `absorb_vstack_same_acc` | `pub fn absorb_vstack_same_acc(mut self, stack: &VirtualStack<T, A>) -> Self {` |
| `src/parser/gss/mod.rs` | 2556 | `absorb_vstack_same_acc_owned` | `pub fn absorb_vstack_same_acc_owned(mut self, mut stack: VirtualStack<T, A>) -> Self {` |
| `src/parser/gss/mod.rs` | 2608 | `absorb_push_interface_inplace` | `fn absorb_push_interface_inplace(` |
| `src/parser/gss/mod.rs` | 2650 | `popn` | `pub fn popn(&self, n: isize) -> Self {` |
| `src/parser/gss/mod.rs` | 2664 | `popn_lower` | `fn popn_lower<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(` |
| `src/parser/gss/mod.rs` | 2722 | `popn_upper` | `fn popn_upper<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(` |
| `src/parser/gss/mod.rs` | 2784 | `popn_single_interface_path` | `fn popn_single_interface_path(&self, n: isize) -> Option<Self> {` |
| `src/parser/gss/mod.rs` | 2854 | `pop` | `pub fn pop(&self) -> Self {` |
| `src/parser/gss/mod.rs` | 2860 | `pop1_common_interface_base` | `pub fn pop1_common_interface_base(&self) -> Option<Self> {` |
| `src/parser/gss/mod.rs` | 2901 | `for_each_decomposed` | `pub fn for_each_decomposed(&self, mut f: impl FnMut(T, Self)) {` |
| `src/parser/gss/mod.rs` | 2965 | `try_virtual_stack` | `pub fn try_virtual_stack(&self) -> Option<VirtualStack<T, A>> {` |
| `src/parser/gss/mod.rs` | 2977 | `is_empty` | `pub fn is_empty(&self) -> bool {` |
| `src/parser/gss/mod.rs` | 2984 | `max_depth` | `pub fn max_depth(&self) -> u32 {` |
| `src/parser/gss/mod.rs` | 2988 | `summary` | `pub fn summary(&self) -> LeveledGSSSummary {` |
| `src/parser/gss/mod.rs` | 3100 | `isolate` | `pub fn isolate(&self, value: Option<T>) -> Self {` |
| `src/parser/gss/mod.rs` | 3195 | `apply` | `pub fn apply<B, F>(&self, mut func: F) -> LeveledGSS<T, B>` |
| `src/parser/gss/mod.rs` | 3202 | `map_acc` | `fn map_acc<A, B, F>(a: &A, memo: &mut StdHashMap<A, B>, f: &mut F) -> B` |
| `src/parser/gss/mod.rs` | 3216 | `transform` | `fn transform<T, A, B, F>(` |
| `src/parser/gss/mod.rs` | 3256 | `apply_and_prune` | `pub fn apply_and_prune<B, M>(&self, mut mutator: M) -> LeveledGSS<T, B>` |
| `src/parser/gss/mod.rs` | 3273 | `mutate_acc` | `fn mutate_acc<A, B, M>(` |
| `src/parser/gss/mod.rs` | 3293 | `transform` | `fn transform<T, A, B, M>(` |
| `src/parser/gss/mod.rs` | 3347 | `apply_transform_and_decompose` | `pub fn apply_transform_and_decompose<B, M>(` |
| `src/parser/gss/mod.rs` | 3357 | `mutate_acc_td` | `fn mutate_acc_td<A, B, M>(` |
| `src/parser/gss/mod.rs` | 3377 | `transform_td` | `fn transform_td<T, A, B, M>(` |
| `src/parser/gss/mod.rs` | 3507 | `apply_and_prune_no_promote` | `pub fn apply_and_prune_no_promote(&self, mut mutator: impl FnMut(&A) -> Option<A>) -> Self {` |
| `src/parser/gss/mod.rs` | 3518 | `mutate_acc_np` | `fn mutate_acc_np<A, M>(` |
| `src/parser/gss/mod.rs` | 3537 | `transform_np` | `fn transform_np<T, A, M>(` |
| `src/parser/gss/mod.rs` | 3591 | `merge` | `pub fn merge(&self, other: &Self) -> Self {` |
| `src/parser/gss/mod.rs` | 3600 | `merge_many` | `pub fn merge_many(gsses: impl IntoIterator<Item = Self>) -> Self {` |
| `src/parser/gss/mod.rs` | 3623 | `fuse` | `pub fn fuse(&self, levels: Option<isize>) -> Self {` |
| `src/parser/gss/mod.rs` | 3652 | `fuse_lower` | `fn fuse_lower<T, A>(` |
| `src/parser/gss/mod.rs` | 3728 | `fuse_upper` | `fn fuse_upper<T, A>(` |
| `src/parser/gss/mod.rs` | 3817 | `peek` | `pub fn peek(&self) -> HashSet<T> {` |
| `src/parser/gss/mod.rs` | 3821 | `peek_values` | `pub fn peek_values(&self) -> SmallVec<[T; 8]> {` |
| `src/parser/gss/mod.rs` | 3827 | `for_each_top_value` | `pub fn for_each_top_value<F: FnMut(T)>(&self, mut f: F) {` |
| `src/parser/gss/mod.rs` | 3847 | `single_top_value` | `pub fn single_top_value(&self) -> Option<T> {` |
| `src/parser/gss/mod.rs` | 3851 | `single_exclusive_top_value` | `pub fn single_exclusive_top_value(&self) -> Option<T> {` |
| `src/parser/gss/mod.rs` | 3855 | `path_count_at_most` | `pub fn path_count_at_most(&self, limit: usize) -> usize {` |
| `src/parser/gss/mod.rs` | 3860 | `capped_add` | `fn capped_add(acc: usize, value: usize, limit: usize) -> usize {` |
| `src/parser/gss/mod.rs` | 3864 | `count_lower` | `fn count_lower<T>(` |
| `src/parser/gss/mod.rs` | 3881 | `count_lower_inner` | `fn count_lower_inner<T>(` |
| `src/parser/gss/mod.rs` | 3909 | `count_upper` | `fn count_upper<T, A>(` |
| `src/parser/gss/mod.rs` | 3974 | `is_single_path` | `pub fn is_single_path(&self) -> bool {` |
| `src/parser/gss/mod.rs` | 3978 | `single_path_top_first_and_acc` | `pub fn single_path_top_first_and_acc(&self, out: &mut SmallVec<[T; 16]>) -> Option<A> {` |
| `src/parser/gss/mod.rs` | 3979 | `push_lower_path` | `fn push_lower_path<T>(node: &Arc<Lower<T>>, out: &mut SmallVec<[T; 16]>) -> bool` |
| `src/parser/gss/mod.rs` | 4007 | `push_upper_path` | `fn push_upper_path<T, A>(` |
| `src/parser/gss/mod.rs` | 4048 | `reduce_acc` | `pub fn reduce_acc(&self) -> Option<A> {` |
| `src/parser/gss/mod.rs` | 4085 | `for_each_acc` | `pub fn for_each_acc(&self, mut f: impl FnMut(&A)) {` |
| `src/parser/gss/mod.rs` | 4088 | `VisitedPtrs` | `enum VisitedPtrs {` |
| `src/parser/gss/mod.rs` | 4094 | `new` | `fn new() -> Self {` |
| `src/parser/gss/mod.rs` | 4098 | `insert` | `fn insert(&mut self, ptr: usize) -> bool {` |
| `src/parser/gss/mod.rs` | 4121 | `walk` | `fn walk<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(` |
| `src/parser/gss/mod.rs` | 4153 | `all_accs_satisfy` | `pub fn all_accs_satisfy(&self, pred: impl Fn(&A) -> bool) -> bool {` |
| `src/parser/gss/mod.rs` | 4154 | `check` | `fn check<T: Clone + Eq + Hash, A: Merge + Clone + Eq + Hash>(` |
| `src/parser/gss/mod.rs` | 4180 | `truncate` | `pub fn truncate(&self, max_len: isize) -> Self {` |
| `src/parser/gss/mod.rs` | 4204 | `get_node_info_lower` | `fn get_node_info_lower(node: &Arc<Lower<T>>) -> String` |
| `src/parser/gss/mod.rs` | 4219 | `get_node_info_upper` | `fn get_node_info_upper(node: &Arc<Upper<T, A>>) -> String` |
| `src/parser/gss/mod.rs` | 4252 | `format_recursive_lower` | `fn format_recursive_lower(` |
| `src/parser/gss/mod.rs` | 4304 | `format_recursive_upper` | `fn format_recursive_upper(` |
| `src/parser/gss/mod.rs` | 4432 | `TestAcc` | `struct TestAcc(u32);` |
| `src/parser/gss/mod.rs` | 4435 | `merge` | `fn merge(&self, other: &Self) -> Self {` |
| `src/parser/gss/mod.rs` | 4441 | `apply_shared_pop_push_branches_matches_virtual_stack_branch_builder` | `fn apply_shared_pop_push_branches_matches_virtual_stack_branch_builder() {` |
| `src/parser/gss/mod.rs` | 4459 | `apply_shared_pop_push_single_branches_deduplicates_targets` | `fn apply_shared_pop_push_single_branches_deduplicates_targets() {` |
| `src/parser/gss/mod.rs` | 4480 | `selective_top_pure_shift_extracts_one_shared_prefix_path` | `fn selective_top_pure_shift_extracts_one_shared_prefix_path() {` |
| `src/parser/gss/mod.rs` | 4499 | `generic_top_pure_shift_matches_selective_shared_prefix_shape` | `fn generic_top_pure_shift_matches_selective_shared_prefix_shape() {` |
| `src/parser/gss/mod.rs` | 4517 | `bench_generic_top_pure_shift_shared_prefix_shape` | `fn bench_generic_top_pure_shift_shared_prefix_shape() {` |
| `src/runtime/artifact/accessors.rs` | 29 | `start` | `pub fn start(&self) -> ConstraintState<'_> {` |
| `src/runtime/artifact/accessors.rs` | 44 | `mask_len` | `pub fn mask_len(&self) -> usize {` |
| `src/runtime/artifact/accessors.rs` | 57 | `internal_to_original_token_ids` | `pub fn internal_to_original_token_ids(&self) -> &[Vec<u32>] {` |
| `src/runtime/artifact/accessors.rs` | 65 | `original_to_internal_token_ids` | `pub fn original_to_internal_token_ids(&self) -> &[u32] {` |
