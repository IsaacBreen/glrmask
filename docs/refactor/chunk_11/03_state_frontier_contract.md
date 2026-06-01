# ConstraintState frontier contract

`ConstraintState` now lives in `runtime/state/mod.rs` and has a narrow role:
store the mutable runtime configuration.

The semantic fields are:

```rust
constraint: &Constraint
state: BTreeMap<u32, ParserGSS>
generation: u64
```

The borrowed `Constraint` is not owned by the state, but it is part of every
operation's environment.  The `state` map is the actual frontier.  The generation
counter is not part of the accepted language, but it tracks semantic mutation:
every Commit increments it, so a cached Mask result can be invalidated cheaply.

The nonsemantic fields are:

```rust
buffers: CommitBuffers
mask_cache: Mutex<Option<MaskCacheData>>
mask_scratch: Mutex<MaskScratch>
```

The clone implementation preserves the logical frontier but resets caches and
scratch.  This is the right semantics: cloning a state should produce another
state that accepts the same continuations, not another state with the same heap
allocation history.

The following invariant should hold for every public runtime method:

```text
Discarding CommitBuffers, MaskCacheData, and MaskScratch does not change the
set of accepted future byte strings.
```

This invariant is now visible in the file tree because cache and scratch types
are defined in different modules from the `ConstraintState` struct itself.
