# Public API compatibility notes

This chunk is intended to preserve public API names.

Unchanged public methods include:

- `ConstraintState::mask`
- `ConstraintState::fill_mask`
- `ConstraintState::fill_mask_timed_ns`
- `ConstraintState::fill_mask_profiled`
- `ConstraintState::fill_mask_and_internal_token_ids`
- `ConstraintState::commit_token`
- `ConstraintState::commit_token_timed_ns`
- `ConstraintState::commit_token_profiled`
- `ConstraintState::commit_token_per_advance`
- `ConstraintState::commit_bytes`
- `ConstraintState::commit_tokens`
- `ConstraintState::force`
- `ConstraintState::is_complete`
- `ConstraintState::is_finished`

Changed private/internal names include:

- `end_state_may_advance` -> `end_state_can_advance`.

No Python binding surface should need to change for this chunk.  If a Python
binding compile error appears, it is likely due to a Rust import or module path,
not an intended API change.
