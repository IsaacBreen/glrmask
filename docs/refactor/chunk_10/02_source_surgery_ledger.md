# Source surgery ledger

## Deleted top-level runtime files

The following old files were removed because their concepts now live inside the
artifact namespace:

```text
src/runtime/artifact.rs        -> src/runtime/artifact/compiled.rs + submodules
src/runtime/finalize.rs        -> src/runtime/artifact/finalize.rs
src/runtime/serde.rs           -> src/runtime/artifact/serialization.rs
src/runtime/token_space.rs     -> src/runtime/artifact/token_space.rs
```

This is not cosmetic.  It prevents the runtime root from pretending that
serialization, finalization, and token-space quotients are unrelated peer
features.  They are all properties of the compiled artifact.

## New files

### `src/runtime/artifact/mod.rs`

Declares the artifact namespace and gives the intended reading order.

### `src/runtime/artifact/compiled.rs`

Owns the `Constraint` struct and `CompiledArtifactParts` constructor input.  The
source now states that the public `Constraint` is currently the compiled
artifact storage object, while leaving room for a future `Arc<CompiledArtifact>`
owner split.

### `src/runtime/artifact/cache_types.rs`

Owns type aliases and the `RuntimeCaches` aggregate.  These are named as cache
storage, not semantic runtime state.

### `src/runtime/artifact/caches.rs`

Owns all rebuild logic for derived cache fields.  This code moved out of
`constraint.rs`, which now no longer carries the burden of compile/load
finalization.

### `src/runtime/artifact/token_space.rs`

Owns conversion between original and internal token/state coordinate systems.

### `src/runtime/artifact/templates.rs`

Owns commit-time template DFA bundles.

### `src/runtime/artifact/dense.rs`

Owns the shared dense bit-vector type.

### `src/runtime/artifact/finalize.rs`

Owns the explicit cache-finalization entry point.

### `src/runtime/artifact/serialization.rs`

Owns save/load and version metadata.

### `src/runtime/artifact/accessors.rs`

Owns public and internal artifact accessors such as `start`, `mask_len`, and
parser/table/token-space diagnostics.

### `src/runtime/bitmask_ops.rs`

Owns primitive output-mask operations.  This is deliberately not in
`artifact/`, because both artifact cache building and mask materialization need
word-level Boolean operations.

## Updated file

### `src/compile/pipeline/finalize.rs`

Before this chunk, the compile finalizer constructed `Constraint` with every
semantic field and every empty cache field listed manually.  That made the
compile pipeline depend on runtime-cache layout.

After this chunk, it constructs `CompiledArtifactParts` and calls
`Constraint::from_compiled_parts`.  Cache field defaults are now owned by the
runtime artifact module.

### `src/runtime/mod.rs`

The runtime module now declares `bitmask_ops` and stops declaring top-level
`finalize`, `serde`, and `token_space` modules.

## Moved method groups

### Cache rebuild group

Moved from `runtime/constraint.rs` to `runtime/artifact/caches.rs`:

- `rebuild_runtime_caches_impl`
- `compute_tokenizer_fast_transitions`
- `compute_buf_masks`
- `compute_token_block_sparse_masks`
- `compute_sliding_word_group_dense_masks`
- `compute_all_tokens_buf_mask`
- `compute_json_escape_prefix_buf_mask`
- `compute_weight_token_buf_masks`
- `compute_weight_token_sparse_buf_masks`
- `compute_seed_state_buf_masks`
- `compute_heavy_token_dense_masks`
- `compute_flat_buf_masks`
- `compute_total_internal_buf_cost`
- `compute_internal_token_buf_op_costs`
- `compute_word_group_buf_op_costs`
- `compute_dense_token_bytes`
- `compute_fast_transitions`
- `compute_dense_token_masks`
- `build_seed_dense_masks`
- seed dense trie helpers

### Accessor group

Moved to `runtime/artifact/accessors.rs`:

- `start`
- `mask_len`
- `internal_to_original_token_ids`
- `original_to_internal_token_ids`
- `num_parser_states`
- `parser_dwa`
- `can_match_for_state_internal`
- `max_original_token_id`

### Token-space group

Moved from `runtime/token_space.rs` to `runtime/artifact/token_space.rs`.

### Serialization group

Moved from `runtime/serde.rs` to `runtime/artifact/serialization.rs` and
upgraded to a versioned envelope.
