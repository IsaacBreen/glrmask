# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Analysis symbol ledger

| File | Line | Symbol |
| --- | ---: | --- |
| `src/parser/glr/analysis/fixed_point_sets.rs` | 1 | `fn compute_nullable(rules: &[Rule], num_nt: u32) -> BTreeSet<NonterminalID> {` |
| `src/parser/glr/analysis/fixed_point_sets.rs` | 23 | `fn compute_first(` |
| `src/parser/glr/analysis/fixed_point_sets.rs` | 63 | `fn compute_follow(` |
| `src/parser/glr/analysis/fixed_point_sets.rs` | 126 | `fn terminal_bit(terminal: TerminalID, num_terminals: u32) -> usize {` |
| `src/parser/glr/analysis/fixed_point_sets.rs` | 134 | `fn filter_graph_to_reachable(` |
| `src/parser/glr/analysis/left_recursion.rs` | 1 | `fn eliminate_hidden_left_recursion(` |
| `src/parser/glr/analysis/left_recursion.rs` | 135 | `fn expand_cycle_head_paths(` |
| `src/parser/glr/analysis/left_recursion.rs` | 193 | `fn nullable_prefix_len(rhs: &[Symbol], nullable: &BTreeSet<NonterminalID>) -> usize {` |
| `src/parser/glr/analysis/model.rs` | 1 | `pub struct AnalyzedGrammar {` |
| `src/parser/glr/analysis/model.rs` | 14 | `impl AnalyzedGrammar {` |
| `src/parser/glr/analysis/model.rs` | 209 | `pub(crate) fn eliminate_right_recursion(` |
| `src/parser/glr/analysis/normalize.rs` | 1 | `pub fn normalize_grammar(rules: &mut Vec<Rule>, start: NonterminalID) {` |
| `src/parser/glr/analysis/normalize.rs` | 169 | `fn replace_rules_with_resync(` |
| `src/parser/glr/analysis/normalize.rs` | 178 | `fn with_resynced_next_nonterminal(` |
| `src/parser/glr/analysis/normalize.rs` | 187 | `fn resync_next_nonterminal(rules: &[Rule], next_nt: &std::cell::Cell<u32>) {` |
| `src/parser/glr/analysis/null_production_inline.rs` | 1 | `fn inline_null_productions_exhaustive(rules: &[Rule], num_nt: u32) -> Vec<Rule> {` |
| `src/parser/glr/analysis/null_production_inline.rs` | 200 | `fn find_nullable_runs(` |
| `src/parser/glr/analysis/nullable_run_compress.rs` | 1 | `fn compute_nonempty_productive(rules: &[Rule], num_nt: u32) -> BTreeSet<NonterminalID> {` |
| `src/parser/glr/analysis/nullable_run_compress.rs` | 38 | `fn compress_nullable_runs_with_optional_tree(rules: &[Rule], num_nt: u32) -> Vec<Rule> {` |
| `src/parser/glr/analysis/nullable_run_compress.rs` | 253 | `fn build_non_nullable_tree(` |
| `src/parser/glr/analysis/nullable_run_compress.rs` | 350 | `fn get_or_create_non_nullable_nt(` |
| `src/parser/glr/analysis/nullable_run_compress.rs` | 419 | `pub(crate) fn inline_null_productions(rules: &[Rule], num_nt: u32) -> Vec<Rule> {` |
| `src/parser/glr/analysis/options.rs` | 7 | `pub(crate) fn analysis_profile_enabled() -> bool {` |
| `src/parser/glr/analysis/profile.rs` | 1 | `fn compile_profile_enabled() -> bool {` |
| `src/parser/glr/analysis/profile.rs` | 5 | `fn elapsed_ms(started_at: Instant) -> f64 {` |
| `src/parser/glr/analysis/profile.rs` | 9 | `fn emit_normalize_profile(` |
| `src/parser/glr/analysis/profile.rs` | 38 | `fn emit_inline_null_profile(` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 1 | `fn remove_unreachable_rules(rules: &[Rule], start: NonterminalID) -> Vec<Rule> {` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 33 | `fn build_rhs_by_lhs(rules: &[Rule]) -> BTreeMap<NonterminalID, BTreeSet<Vec<Symbol>>> {` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 44 | `fn compute_expandable_single_productions(` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 115 | `fn flatten_rhs_symbols(` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 189 | `enum RuleDedupKey<'a> {` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 194 | `impl RuleDedupKey<'_> {` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 209 | `impl PartialEq for RuleDedupKey<'_> {` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 215 | `impl Eq for RuleDedupKey<'_> {}` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 217 | `impl Hash for RuleDedupKey<'_> {` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 224 | `fn dedup_rules(rules: &mut Vec<Rule>) {` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 257 | `fn is_reflexive_unit_rule(rule: &Rule) -> bool {` |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 261 | `pub(crate) fn merge_identical_nonterminals(` |
| `src/parser/glr/analysis/right_recursion.rs` | 1 | `fn max_nt_id(rules: &[Rule]) -> u32 {` |
| `src/parser/glr/analysis/right_recursion.rs` | 14 | `fn add_boundary_nonterminals<'a>(` |
| `src/parser/glr/analysis/right_recursion.rs` | 32 | `fn build_right_reachability_graph(` |
| `src/parser/glr/analysis/right_recursion.rs` | 47 | `fn find_indirect_rr_cycle(` |
| `src/parser/glr/analysis/right_recursion.rs` | 53 | `fn find_cycle(` |
| `src/parser/glr/analysis/right_recursion.rs` | 123 | `fn build_left_reachability_graph(` |
| `src/parser/glr/analysis/right_recursion.rs` | 136 | `fn find_indirect_lr_cycle(` |
| `src/parser/glr/analysis/right_recursion.rs` | 142 | `fn find_nontrivial_sccs(` |
| `src/parser/glr/analysis/right_recursion.rs` | 229 | `fn find_cycle_excluding_self_loops(` |
| `src/parser/glr/analysis/right_recursion.rs` | 240 | `fn inline_right_end(` |
| `src/parser/glr/analysis/right_recursion.rs` | 278 | `fn find_right_end_position(` |
| `src/parser/glr/analysis/right_recursion.rs` | 293 | `fn is_direct_right_recursive(rule: &Rule) -> bool {` |
| `src/parser/glr/analysis/right_recursion.rs` | 307 | `fn resolve_direct_rr_single_nt(` |
| `src/parser/glr/analysis/right_recursion.rs` | 357 | `fn resolve_direct_rr_batched(` |
| `src/parser/glr/analysis/tests.rs` | 1 | `mod tests {` |
| `src/parser/glr/analysis.rs` | 8 | `pub const EOF: TerminalID = u32::MAX;` |
