# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Advance symbol ledger

| File | Line | Symbol |
| --- | ---: | --- |
| `src/parser/glr/advance/applicability.rs` | 1 | `pub(crate) fn stack_can_advance_on(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> bool {` |
| `src/parser/glr/advance/applicability.rs` | 22 | `fn stack_may_apply_guarded_shifts(stack: &ParserGSS, shifts: &[GuardedStackShift]) -> bool {` |
| `src/parser/glr/advance/applicability_any.rs` | 1 | `pub(crate) fn stack_can_advance_on_any(` |
| `src/parser/glr/advance/applicability_any.rs` | 74 | `pub(crate) fn stacks_finished(table: &GLRTable, stack: &ParserGSS) -> bool {` |
| `src/parser/glr/advance/deterministic.rs` | 1 | `fn advance_deterministically(` |
| `src/parser/glr/advance/deterministic_profiled.rs` | 1 | `fn advance_deterministically_profiled(` |
| `src/parser/glr/advance/deterministic_vstack.rs` | 1 | `fn advance_deterministically_from_vstack_raw(` |
| `src/parser/glr/advance/deterministic_vstack.rs` | 85 | `fn advance_deterministically_from_vstack(` |
| `src/parser/glr/advance/deterministic_vstack.rs` | 94 | `fn advance_reduce_branch(` |
| `src/parser/glr/advance/deterministic_vstack.rs` | 119 | `fn single_concrete_path_as_vstack(` |
| `src/parser/glr/advance/deterministic_vstack.rs` | 133 | `fn advance_split_from_vstack(` |
| `src/parser/glr/advance/entry_points.rs` | 1 | `pub(crate) fn advance_stacks(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> ParserGSS {` |
| `src/parser/glr/advance/entry_points.rs` | 7 | `pub(crate) fn advance_stacks_owned(table: &GLRTable, stack: ParserGSS, token: TerminalID) -> ParserGSS {` |
| `src/parser/glr/advance/entry_points.rs` | 11 | `pub(crate) fn advance_stacks_profiled(` |
| `src/parser/glr/advance/entry_points.rs` | 139 | `fn advance_stacks_core(table: &GLRTable, mut gss: ParserGSS, token: TerminalID) -> ParserGSS {` |
| `src/parser/glr/advance/entry_points.rs` | 172 | `fn try_collapse_small_reduce_fanout(` |
| `src/parser/glr/advance/entry_points.rs` | 219 | `fn pure_frontier_shift(action: &Action) -> Option<(u32, bool)> {` |
| `src/parser/glr/advance/fast_paths.rs` | 1 | `fn advance_pure_frontier_shifts(` |
| `src/parser/glr/advance/fast_paths.rs` | 28 | `fn try_advance_single_alt_pop1_common_suffix_stackshift_wave(` |
| `src/parser/glr/advance/fast_paths.rs` | 64 | `fn try_advance_pop1_reduce_plus_stackshift_wave(` |
| `src/parser/glr/advance/fast_paths.rs` | 133 | `fn rebuild_floor_cross_from_shifts(` |
| `src/parser/glr/advance/fast_paths.rs` | 148 | `fn push_states(mut gss: ParserGSS, states: &[u32]) -> ParserGSS {` |
| `src/parser/glr/advance/fast_paths.rs` | 155 | `fn common_stack_shift_suffix_len(pushes: &[&[u32]]) -> usize {` |
| `src/parser/glr/advance/fast_paths.rs` | 172 | `fn apply_push_sequences(base: ParserGSS, pushes: &[&[u32]]) -> ParserGSS {` |
| `src/parser/glr/advance/fast_paths.rs` | 197 | `fn apply_stack_shifts(gss: ParserGSS, shifts: &[StackShift]) -> ParserGSS {` |
| `src/parser/glr/advance/fast_paths.rs` | 255 | `pub(crate) fn apply_guarded_stack_shifts_fast(` |
| `src/parser/glr/advance/guarded_shifts.rs` | 1 | `fn apply_guarded_stack_shifts(` |
| `src/parser/glr/advance/guarded_shifts.rs` | 57 | `fn indexed_guarded_shift_candidates(` |
| `src/parser/glr/advance/guarded_shifts.rs` | 90 | `fn apply_guarded_stack_shifts_to_vstack(` |
| `src/parser/glr/advance/guarded_shifts.rs` | 196 | `fn virtual_stack_satisfies_guards(` |
| `src/parser/glr/advance/guarded_shifts.rs` | 226 | `fn virtual_stack_may_apply_guarded_shift(` |
| `src/parser/glr/advance/mod.rs` | 16 | `mod options;` |
| `src/parser/glr/advance/mod.rs` | 17 | `mod profile;` |
| `src/parser/glr/advance/mod.rs` | 30 | `pub type ParserGSS = LeveledGSS<u32, TerminalsDisallowed>;` |
| `src/parser/glr/advance/mod.rs` | 32 | `type ReduceSources = SmallVec<[(u32, ParserGSS); 4]>;` |
| `src/parser/glr/advance/mod.rs` | 33 | `type ReduceBranches = SmallVec<[(ParserGSS, u32, bool); 4]>;` |
| `src/parser/glr/advance/mod.rs` | 34 | `type FloorCrossShift = (u32, u32, bool);` |
| `src/parser/glr/advance/mod.rs` | 36 | `const SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH: usize = 64;` |
| `src/parser/glr/advance/mod.rs` | 37 | `const GUARDED_STACK_TO_STACKS_MAX_DEPTH: usize = 64;` |
| `src/parser/glr/advance/mod.rs` | 38 | `const SMALL_REDUCE_FANOUT_COLLAPSE_MAX_BRANCHES: usize = 8;` |
| `src/parser/glr/advance/mod.rs` | 40 | `fn advance_options() -> &'static ParserAdvanceOptions {` |
| `src/parser/glr/advance/mod.rs` | 44 | `fn guarded_stack_to_stacks_fallback_disabled() -> bool {` |
| `src/parser/glr/advance/mod.rs` | 48 | `fn stack_effect_to_stacks_fallback_disabled() -> bool {` |
| `src/parser/glr/advance/mod.rs` | 52 | `fn advance_trace_enabled() -> bool {` |
| `src/parser/glr/advance/nondeterministic.rs` | 1 | `fn advance_nondeterministically(` |
| `src/parser/glr/advance/nondeterministic_profiled.rs` | 1 | `fn advance_nondeterministically_profiled(` |
| `src/parser/glr/advance/options.rs` | 14 | `pub(crate) struct ParserAdvanceOptions {` |
| `src/parser/glr/advance/options.rs` | 26 | `impl ParserAdvanceOptions {` |
| `src/parser/glr/advance/options.rs` | 53 | `fn env_flag_enabled(name: &str) -> bool {` |
| `src/parser/glr/advance/profile.rs` | 2 | `pub struct AdvanceTrace {` |
| `src/parser/glr/advance/profile.rs` | 8 | `pub struct AdvanceTraceWave {` |
| `src/parser/glr/advance/profile.rs` | 15 | `pub struct AdvanceTraceStep {` |
| `src/parser/glr/advance/profile.rs` | 24 | `pub struct AdvanceTraceReduce {` |
| `src/parser/glr/advance/profile.rs` | 33 | `pub struct AdvanceTraceGoto {` |
| `src/parser/glr/advance/profile.rs` | 40 | `pub struct AdvanceProfile {` |
| `src/parser/glr/advance/profile_trace.rs` | 1 | `fn trace_action_kind(action: Option<&Action>) -> &'static str {` |
| `src/parser/glr/advance/profile_trace.rs` | 14 | `fn trace_reduce_summary(` |
| `src/parser/glr/advance/profile_trace.rs` | 49 | `fn trace_action_summary(` |
| `src/parser/glr/advance/profile_trace.rs` | 92 | `enum AdvancedBranch {` |
| `src/parser/glr/advance/profile_trace.rs` | 97 | `impl AdvancedBranch {` |
| `src/parser/glr/advance/reduce_sources.rs` | 1 | `fn reduce_sources_from_isolated(gss: &ParserGSS, rhs_len: usize) -> ReduceSources {` |
| `src/parser/glr/advance/reduce_sources.rs` | 19 | `fn reduce_branches_from_isolated(` |
| `src/parser/glr/advance/reduce_sources.rs` | 46 | `fn merge_into(dst: &mut ParserGSS, branch: ParserGSS) {` |
| `src/parser/glr/advance/tests.rs` | 2 | `mod tests {` |
| `src/parser/glr/advance/mod.rs` | 16 | `mod options;` |
| `src/parser/glr/advance/mod.rs` | 17 | `mod profile;` |
| `src/parser/glr/advance/mod.rs` | 30 | `pub type ParserGSS = LeveledGSS<u32, TerminalsDisallowed>;` |
| `src/parser/glr/advance/mod.rs` | 32 | `type ReduceSources = SmallVec<[(u32, ParserGSS); 4]>;` |
| `src/parser/glr/advance/mod.rs` | 33 | `type ReduceBranches = SmallVec<[(ParserGSS, u32, bool); 4]>;` |
| `src/parser/glr/advance/mod.rs` | 34 | `type FloorCrossShift = (u32, u32, bool);` |
| `src/parser/glr/advance/mod.rs` | 36 | `const SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH: usize = 64;` |
| `src/parser/glr/advance/mod.rs` | 37 | `const GUARDED_STACK_TO_STACKS_MAX_DEPTH: usize = 64;` |
| `src/parser/glr/advance/mod.rs` | 38 | `const SMALL_REDUCE_FANOUT_COLLAPSE_MAX_BRANCHES: usize = 8;` |
| `src/parser/glr/advance/mod.rs` | 40 | `fn advance_options() -> &'static ParserAdvanceOptions {` |
| `src/parser/glr/advance/mod.rs` | 44 | `fn guarded_stack_to_stacks_fallback_disabled() -> bool {` |
| `src/parser/glr/advance/mod.rs` | 48 | `fn stack_effect_to_stacks_fallback_disabled() -> bool {` |
| `src/parser/glr/advance/mod.rs` | 52 | `fn advance_trace_enabled() -> bool {` |
