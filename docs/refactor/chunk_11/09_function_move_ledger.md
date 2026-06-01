# Function and type move ledger

Moved from `runtime/state.rs` to `runtime/state/cache.rs`:

- `MaskCacheData`
- `MaskScratch`

Moved from `runtime/state.rs` to `runtime/state/scratch.rs`:

- `CommitBuffers`
- `CommitBuffers::clone`
- `CommitBuffers::clear_all`

Moved from `runtime/state.rs` to `runtime/state/inspect.rs`:

- `ConstraintState::is_complete`
- `ConstraintState::is_finished`
- `ConstraintState::parser_root_count`
- `ConstraintState::parser_path_count`
- `ConstraintState::has_parser_ambiguity`
- `ConstraintState::debug_parser_stacks`

Moved from `runtime/state.rs` to `runtime/state/force.rs`:

- `ForcedFirstByte`
- `GreedyTokenizationStep`
- `ConstraintState::force`
- `ConstraintState::force_by_bytes`
- `ConstraintState::single_token_force`
- `ConstraintState::compute_forced_byte_prefix`
- `ConstraintState::forced_first_byte`
- `ConstraintState::tokenize_forced_with_stop`
- `ConstraintState::greedy_tokenization_step`
- `is_token_set`
- `single_allowed_token`
- `for_each_set_bit`

Moved from `runtime/mask/mod.rs` to `runtime/mask/dense_acc.rs`:

- `DenseTokenMaskCache`
- `DenseMaskGSS`
- `DenseTokenSetIntersectionKey`
- `DenseGssTransitionKey`
- `DenseMaskAcc`
- `DenseMaskAcc` methods
- `Merge for DenseMaskAcc`
- dense-accumulator unit tests

Moved from `runtime/mask/mod.rs` to `runtime/mask/bitset.rs`:

- `update_eos_mask`
- `set_token_bit`
- `is_token_bit_set`
- `for_each_set_token_bit`
- `eos_mask_bit`

Moved from `runtime/mask/mod.rs` to `runtime/mask/constants.rs`:

- `DELTA_SEED_MIN_SAVINGS`
- `MASK_SINGLE_PATH_DIRECT_MAX_DEPTH`
- `MASK_SINGLE_PATH_DIRECT_MAX_TOTAL_PATHS`

Moved from `runtime/commit/mod.rs` to `runtime/commit/options.rs`:

- `TEMPLATE_ADVANCE_ENABLED`
- `VALIDATE_TEMPLATE_ADVANCE_ENABLED`
- `template_advance_enabled`
- `validate_template_advance_enabled`

Moved from `runtime/commit/mod.rs` to `runtime/commit/parser_advance.rs`:

- `advance_parser_stacks`
- `advance_parser_stacks_owned`
- `advance_parser_stacks_profiled`

Moved from `runtime/commit/mod.rs` to `runtime/commit/token_lookup.rs`:

- `token_bytes_for_id`

Moved from `runtime/commit/mod.rs` to `runtime/commit/mask_assert.rs`:

- `commit_mask_assert_enabled`
- `token_in_mask`
- `snapshot_mask_membership`
- `format_token_bytes`
- `assert_mask_commit_equivalence`

Renamed inside Commit:

- `end_state_may_advance` -> `end_state_can_advance`
- local `may_advance` boolean -> `can_advance`

Profile field names such as `may_advance_ns` were intentionally not renamed in
this chunk because they are serialized/observed diagnostic names and may require
a compatibility decision.
