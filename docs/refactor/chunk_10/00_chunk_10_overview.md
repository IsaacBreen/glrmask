# Chunk 10 — Runtime artifact and finalization cleanup

## Purpose

This chunk makes the runtime artifact legible as a mathematical object rather
than as a bag of fields accidentally required by mask and commit performance
paths.

The previous state had `src/runtime/artifact.rs` containing the `Constraint`
struct and many aliases, while `src/runtime/constraint.rs` mixed:

1. grammar diagnostics;
2. public lifecycle methods;
3. runtime cache construction;
4. token-space queries;
5. dense/sparse mask materialization;
6. output-mask micro-operations;
7. seed-mask construction;
8. serialization-adjacent rebuild logic.

That mixture is not publication-ready.  It hides the actual mathematical object
that the paper describes: a compiled constraint artifact.  It also makes it hard
to say which fields are semantic and which fields are merely derived caches.

## New reading order

The runtime artifact is now organized as:

```text
src/runtime/artifact/
  mod.rs             module boundary and reading order
  README.md          human-facing artifact explanation
  compiled.rs        serialized semantic object and constructor input
  token_space.rs     original/internal token and tokenizer-state quotients
  templates.rs       commit-time stack-effect recognizer bundle
  dense.rs           shared dense bit-vector representation
  cache_types.rs     named derived-cache field aggregate
  caches.rs          cache rebuild algorithms
  accessors.rs       public and crate-internal artifact accessors
  finalize.rs        cache finalization entry point
  serialization.rs   versioned save/load envelope with legacy fallback
```

The runtime root now also has:

```text
src/runtime/bitmask_ops.rs
```

for low-level Boolean operations over output masks.

## Main mathematical distinction

A compiled grammar constraint has two layers:

```text
semantic artifact
  Parser DWA
  GLR table
  tokenizer DFA
  CanMatch relation
  terminal display names
  token bytes
  token-space quotients
  template DFAs

runtime caches
  dense/sparse conversion tables
  fast transition tables
  seed masks
  weight-to-mask caches
  heavy-token shortcuts
  grouped word/nibble/byte masks
```

The semantic artifact is what `save` must preserve.  The runtime caches are a
choice of implementation and may be rebuilt at any time.  This chunk records
that distinction in the source tree.

## What changed in source

1. `src/runtime/artifact.rs` became a directory.
2. The `Constraint` struct moved to `src/runtime/artifact/compiled.rs`.
3. A new `CompiledArtifactParts` struct is the compile-to-runtime handoff.
4. Compile finalization now calls `Constraint::from_compiled_parts(...)` rather
   than constructing every cache field manually.
5. Runtime cache types moved to `cache_types.rs`.
6. Runtime cache rebuild logic moved from `constraint.rs` to `artifact/caches.rs`.
7. Token-space mapping methods moved from `runtime/token_space.rs` to
   `artifact/token_space.rs`.
8. Serialization moved from `runtime/serde.rs` to `artifact/serialization.rs`.
9. Save/load now uses a versioned envelope and can still read the legacy direct
   bincode representation.
10. Low-level mask-buffer OR/AND-NOT/COPY helpers moved to `runtime/bitmask_ops.rs`.

## What did not change

This chunk deliberately does not restructure the actual Mask algorithm or Commit
algorithm.  Those are subsequent chunks.  The large files still present are:

```text
src/runtime/commit/mod.rs
src/runtime/mask/mod.rs
src/runtime/mask_mapping.rs
```

Those remain large because they are not artifact-finalization concerns.  They
should be handled by the dedicated Mask, Commit, and final-mask mapping chunks.

## Definition of done for this chunk

This chunk is complete when:

- the compiled artifact has a named module directory;
- semantic fields and derived caches are documented separately;
- compile finalization no longer knows every cache field;
- serialization has explicit version metadata;
- old top-level runtime files for serde/finalize/token-space are gone;
- the cache rebuild entry point is artifact-local;
- the patch is self-contained and can be applied after Chunk 09.

No compilation, test, rustfmt, or benchmark pass was intentionally run.
