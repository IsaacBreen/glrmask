# Beginner step-by-step manual

To understand Commit after this chunk, read files in this order:

1. `src/runtime/commit/README.md` for the map/transition model.
2. `src/runtime/commit/api.rs` to see how user calls enter Commit.
3. `src/runtime/commit/general.rs` to see the reference transition.
4. `src/runtime/commit/acceptance.rs` to understand which scanner matches count.
5. `src/runtime/commit/pruning.rs` to understand delayed longest-match exclusions.
6. `src/runtime/commit/queue.rs` to understand how branch states are merged.
7. `src/runtime/commit/parser_advance.rs` and `single_top.rs` for parser stack effects.
8. `src/runtime/commit/fast_path.rs` only after the reference path is clear.
9. `src/runtime/commit/profiled.rs` only after the unprofiled path is clear.

Do not start with `fast_path.rs`. It is performance code and only makes sense after the mathematical transition is understood.
