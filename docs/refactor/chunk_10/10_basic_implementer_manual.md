# Basic implementer manual

This document is for someone applying or repairing the chunk manually.

## Step 1 — create artifact directory

Create:

```text
src/runtime/artifact/
```

Move the old `src/runtime/artifact.rs` contents into smaller files.  Do not
leave both `artifact.rs` and `artifact/mod.rs`, because Rust treats them as two
competing module definitions.

## Step 2 — move the struct

Move `pub struct Constraint` to:

```text
src/runtime/artifact/compiled.rs
```

Keep the public type name `Constraint` for now.  Do not rename it to
`CompiledArtifact` in this chunk, because many impl blocks and bindings still
refer to `Constraint`.

## Step 3 — add the handoff struct

Add `CompiledArtifactParts` in the same file.  It should include compile outputs
only.  It should not include empty cache vectors.

## Step 4 — create the constructor

Add:

```rust
Constraint::from_compiled_parts(parts)
```

This method should install empty cache storage via `RuntimeCaches::default()`.

## Step 5 — update compile finalization

In `src/compile/pipeline/finalize.rs`, replace the giant direct `Constraint {
... }` literal with:

```rust
let mut constraint = Constraint::from_compiled_parts(CompiledArtifactParts { ... });
constraint.rebuild_runtime_caches();
```

## Step 6 — move serialization

Move `save` and `load` from `runtime/serde.rs` to
`runtime/artifact/serialization.rs`.  Add the versioned envelope and keep legacy
fallback.

## Step 7 — move token-space methods

Move `runtime/token_space.rs` to `runtime/artifact/token_space.rs`.  Add type
aliases documenting original/internal token/state ids.

## Step 8 — move cache rebuild methods

Move `rebuild_runtime_caches_impl` and its helper methods out of
`runtime/constraint.rs` into `runtime/artifact/caches.rs`.

## Step 9 — extract bitmask ops

Move low-level OR/AND-NOT/COPY helpers to `runtime/bitmask_ops.rs` and import
them from both artifact cache building and remaining mask materialization code.

## Step 10 — update module declarations

Update `runtime/mod.rs`:

```rust
mod artifact;
mod bitmask_ops;
mod commit;
mod constraint;
mod mask;
pub mod mask_mapping;
mod state;
```

Remove old declarations for `finalize`, `serde`, and `token_space`.

## Step 11 — verify shape without compiling

Run static shape checks:

```text
find src/runtime/artifact -maxdepth 1 -type f
rg "mod serde|mod finalize|mod token_space" src/runtime/mod.rs
rg "from_compiled_parts" src/compile/pipeline/finalize.rs
```

This chunk intentionally stops there.
