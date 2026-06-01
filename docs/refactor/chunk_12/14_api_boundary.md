# Public API boundary

`api.rs` now owns the public `ConstraintState` methods:

- `commit_token`
- `commit_token_timed_ns`
- `commit_token_profiled`
- `commit_token_per_advance`
- `commit_bytes`
- `commit_tokens`

The API boundary performs vocabulary token lookup, optional Mask/Commit equivalence assertion, generation counter updates, and dispatch to unprofiled/profiled internal transition functions.

The API boundary should not contain queue logic, scanner normalization, parser-table shortcuts, or delayed-exclusion algorithms. Those are now in separate internal modules.
