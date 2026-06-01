# Patch application manual

To apply this chunk manually:

1. Start from the Chunk 11 source tree.
2. Replace `src/runtime/commit/mod.rs` with the new routing module.
3. Add the new files: `api.rs`, `acceptance.rs`, `fast_path.rs`, `general.rs`, `initial_scan.rs`, `profiled.rs`, `pruning.rs`, `queue.rs`, `single_top.rs`, `terminal_advance.rs`, and `types.rs`.
4. Keep existing `mask_assert.rs`, `options.rs`, `parser_advance.rs`, `profile.rs`, `template_advance.rs`, `tokenizer_scan.rs`, and `token_lookup.rs`.
5. Update `src/runtime/commit/README.md`.
6. Apply compile repair later without changing algorithmic control flow.

The safest review method is to diff old `mod.rs` against the union of the new files and confirm every old item appears in the move ledger.
