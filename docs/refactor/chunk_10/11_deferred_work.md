# Deferred work after Chunk 10

Chunk 10 completes the artifact/finalization boundary but deliberately leaves
several runtime cleanups for later chunks.

## Deferred to Chunk 11 — `Constraint` implementation split

`runtime/constraint.rs` still owns dense-to-buffer materialization helpers and
some artifact diagnostics.  Chunk 11 should split it into:

```text
runtime/constraint/mod.rs
runtime/constraint/diagnostics.rs
runtime/constraint/materialize.rs
runtime/constraint/delta.rs
runtime/constraint/json_escape.rs
```

## Deferred to Chunk 12 — Mask runtime split

`runtime/mask/mod.rs` is still large.  It should be split by the mathematical
steps of Mask:

```text
seed
walk_parser_dwa
merge_weights
materialize
cache
profile
```

## Deferred to Chunk 13 — Commit runtime split

`runtime/commit/mod.rs` is still large.  It should be split by:

```text
scan bytes
collect terminal matches
advance parser stack relation
apply delayed exclusions
merge parser GSS
profile/validate
```

## Deferred to Chunk 14 — template DFA subsystem

Template DFAs still live partly as commit accelerators.  They should become the
first-class stack-effect recognizer subsystem shared by Parser-DWA compilation
and Commit.

## Deferred to Chunk 15 — weights/masks/pair sets

`runtime/bitmask_ops.rs`, `mask_mapping.rs`, and weight-related caches should be
unified under a more principled bitset/mask algebra.

## Deferred compile concerns

No attempt was made to fix import warnings or run rustfmt.  The next compile
repair pass should start from the static checks and then resolve errors in
module-order order:

1. runtime artifact imports;
2. compile finalizer imports;
3. moved cache helper visibility;
4. serialization envelope derives;
5. any stale path from old top-level runtime modules.
