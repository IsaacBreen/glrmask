# Chunk 10 changeset — Runtime artifact and finalization cleanup

## Summary

This chunk restructures the runtime artifact boundary after Chunk 09.  It makes
`Constraint` readable as a compiled artifact with derived runtime caches, moves
serialization/finalization/token-space logic under `src/runtime/artifact/`, and
introduces a compile-to-runtime handoff struct so the compile pipeline no longer
constructs every cache field directly.

No compile, test, benchmark, or rustfmt pass was run.

## Source changes

### Runtime artifact directory

Replaced the old single file:

```text
src/runtime/artifact.rs
```

with:

```text
src/runtime/artifact/mod.rs
src/runtime/artifact/README.md
src/runtime/artifact/compiled.rs
src/runtime/artifact/cache_types.rs
src/runtime/artifact/caches.rs
src/runtime/artifact/accessors.rs
src/runtime/artifact/token_space.rs
src/runtime/artifact/templates.rs
src/runtime/artifact/dense.rs
src/runtime/artifact/finalize.rs
src/runtime/artifact/serialization.rs
```

### Compiled artifact handoff

Added:

```rust
CompiledArtifactParts
Constraint::from_compiled_parts(parts)
```

The compile finalizer now packages semantic compile outputs into
`CompiledArtifactParts` and delegates cache-field initialization to the runtime
artifact module.

### Runtime caches

Added:

```rust
RuntimeCaches
```

as a named aggregate for fields that are derived from the semantic artifact.
The existing `Constraint` struct still stores those fields directly for now, but
construction now passes through `RuntimeCaches::default()`.

Moved cache rebuild logic from `src/runtime/constraint.rs` into:

```text
src/runtime/artifact/caches.rs
```

### Token-space boundary

Moved token-space quotient methods from top-level runtime into:

```text
src/runtime/artifact/token_space.rs
```

and added coordinate aliases:

```rust
OriginalTokenId
InternalTokenId
OriginalTokenizerStateId
InternalTokenizerStateId
```

### Serialization boundary

Moved save/load into:

```text
src/runtime/artifact/serialization.rs
```

and added a versioned envelope:

```rust
SerializedArtifactEnvelope
SerializedArtifactFeatures
SERIALIZATION_FORMAT_VERSION
SERIALIZATION_MAGIC
```

`load` keeps a legacy fallback for old direct-bincode artifacts.

### Low-level bitmask operations

Moved shared output-mask Boolean operations into:

```text
src/runtime/bitmask_ops.rs
```

so cache building and remaining mask materialization code do not depend on
private helpers hidden inside `constraint.rs`.

### Compile finalization

Updated:

```text
src/compile/pipeline/finalize.rs
```

to construct:

```rust
Constraint::from_compiled_parts(CompiledArtifactParts { ... })
```

instead of directly spelling every semantic and cache field in a giant
`Constraint { ... }` literal.

## Documentation added

Added a large self-contained documentation set under:

```text
docs/refactor/chunk_10/
```

including:

- mathematical model of the runtime artifact;
- source surgery ledger;
- compiled-artifact handoff contract;
- runtime cache taxonomy;
- serialization policy;
- finalization algorithm;
- token-space contract;
- bitmask operation boundary;
- reviewer checklist;
- basic implementer manual;
- deferred work map;
- field taxonomy and field-level proof obligations;
- risk register;
- future compile-repair guide;
- cross-chunk dependency map;
- function ledgers;
- manual audit commands.

## Static checks

See:

```text
glrmask_chunk_10_CHECKS.md
```

All static shape checks passed.

## Patch stats

```text
files_changed=61
insertions=5427
deletions=1888
```

## Important remaining work

This chunk intentionally leaves these later chunks untouched:

- `runtime/constraint.rs` dense-to-buffer materialization split;
- `runtime/mask/mod.rs` algorithm split;
- `runtime/commit/mod.rs` algorithm split;
- template DFA subsystem promotion;
- unsafe bitmask operation audit;
- compile/test/rustfmt repair.
