# Beginner step-by-step guide to applying this chunk manually

This section is intentionally mechanical.

1. Create a directory `src/runtime/state/`.
2. Move `MaskCacheData` and `MaskScratch` into `state/cache.rs`.
3. Move `CommitBuffers` and its impl blocks into `state/scratch.rs`.
4. Keep the `ConstraintState` struct in `state/mod.rs`.
5. Move read-only methods such as `is_complete` and `parser_path_count` into
   `state/inspect.rs`.
6. Move `force` and its private helpers into `state/force.rs`.
7. Delete the old flat `state.rs`.
8. Confirm `runtime/mod.rs` still says `mod state;`.
9. In `runtime/mask`, create `dense_acc.rs`, `bitset.rs`, and `constants.rs`.
10. Move `DenseMaskAcc` and its tests into `dense_acc.rs`.
11. Move packed-mask helper functions into `bitset.rs`.
12. Move threshold constants into `constants.rs`.
13. Import those helpers from `mask/mod.rs`.
14. In `runtime/commit`, create `options.rs`, `parser_advance.rs`,
    `mask_assert.rs`, and `token_lookup.rs`.
15. Move the corresponding helper functions.
16. Import them from `commit/mod.rs`.
17. Rename `end_state_may_advance` to `end_state_can_advance` and update calls.
18. Add README files explaining the new layout.
19. Do not run a compiler yet if following the project instruction for this
    cleanup series.  First review the shape.

The key idea is not moving text.  The key idea is that each moved group has a
single reason to exist.
