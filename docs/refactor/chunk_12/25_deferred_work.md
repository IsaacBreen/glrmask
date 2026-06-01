# Deferred work inside Commit

This chunk intentionally stops at the first major Commit boundary. The next Commit-specific refinements should be:

1. Split `fast_path.rs` into `fast_path/full_width.rs`, `fast_path/small_queue.rs`, `fast_path/linear.rs`, and `fast_path/single_token.rs`.
2. Split `profiled.rs` into `profiled/reference.rs`, `profiled/fast_path.rs`, and `profiled/per_advance.rs`.
3. Replace `use super::*` with explicit imports.
4. Add an observer trait so profiled and unprofiled queue walks share the same transition code.
5. Define an internal `CommitTransition` object that owns `constraint`, `bytes`, `buffers`, and phase-local caches.
