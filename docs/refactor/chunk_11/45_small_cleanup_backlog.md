# Small cleanup backlog discovered while doing Chunk 11

- Consider renaming profile fields containing `may_advance` after compatibility
  review.
- Consider removing `for_each_set_token_bit` if it remains unused after later
  diagnostics work.
- Split `runtime/commit/mod.rs` by fast path.
- Split `runtime/mask_mapping.rs` into a named materialization subsystem.
- Decide whether `RuntimeOptions` should supersede runtime env vars.
- Add examples in rustdoc for `commit_bytes` and `force`.
- Add a short note in the README explaining packed mask layout.
- Audit all `expect` calls in forced-token logic and decide whether they encode
  invariants or should become errors.
- Add comments to every Commit fast path with precondition/fallback/reference.
- Move any remaining direct `std::env` reads out of algorithm files.
