# Function and symbol move ledger

## `types.rs`
- `ParserStatesByTokenizer`
- `SMALL_NORMALIZED_MATCH_LINEAR_SCAN_MAX`
- `NormalizedMatch`
- `SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH`
- `AdvanceResultCache`
- `LinearFastPathResult`
- `DirectLinearStep`

## `acceptance.rs`
- `ActionableTerminals`
- `is_ignored_terminal`
- `is_actionable_terminal`
- `collect_unique_actionable_matches`

## `initial_scan.rs`
- `InitialCommitScan::collect`
- `InitialCommitScan::take_exec_result`

## `pruning.rs`
- `state_has_nonempty_accumulators`
- `end_state_can_advance`
- `prune_initial_states`
- `prune_single_initial_state_for_exec`
- `prune_single_initial_state_for_terminal`
- `apply_future_terminal_disallow`

## `queue.rs`
- `merge_parser_state`
- `queue_parser_state`
- `finalize_pending_state`
- `merge_small_parser_state`

## `single_top.rs`
- `apply_single_top_action_fast`
- `apply_single_path_reduce_chain_fast`

## `terminal_advance.rs`
- `advance_terminal_match`

## `fast_path.rs`
- `commit_bytes_fast_path`
- `commit_bytes_full_width_fast_path`
- `commit_bytes_small_queue_fast_path`
- `choose_direct_linear_step`
- `commit_bytes_direct_linear_fast_path`
- `commit_bytes_linear_fast_path`

## `profiled.rs`
- `parser_stacks_only`
- `record_per_advance_entry`
- `commit_bytes_fast_path_profiled`
- `commit_bytes_impl_profiled`
- `final_stacks`
- `commit_bytes_linear_fast_path_profiled`

## `general.rs`
- `commit_bytes_impl`

## `api.rs`
- `ConstraintState::commit_token`
- `ConstraintState::commit_token_timed_ns`
- `ConstraintState::commit_token_profiled`
- `ConstraintState::commit_token_per_advance`
- `ConstraintState::commit_bytes`
- `ConstraintState::commit_tokens`

## How to use this ledger

When repairing compile errors after this no-compile source split, start from this ledger. If an unresolved name appears, locate the owning file here and either import it explicitly or qualify it through the owning module. The intended ownership should not be changed merely to silence an error.
