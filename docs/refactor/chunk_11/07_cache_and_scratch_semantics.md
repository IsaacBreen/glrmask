# Cache and scratch semantics

`runtime/state/cache.rs` contains `MaskCacheData` and `MaskScratch`.
`runtime/state/scratch.rs` contains `CommitBuffers`.

These types have the same semantic status: they are replaceable allocation
state.

The publication invariant is:

```text
For any state S, clearing all cache and scratch buffers yields an observationally
equivalent state S~ such that Mask_C(S) = Mask_C(S~) and Commit_C(S, b) accepts
exactly when Commit_C(S~, b) accepts.
```

This is why `CommitBuffers::clone` returns `Default` instead of copying all
maps.  This is also why cloned `ConstraintState` values start with empty mask
caches and default mask scratch.

Reviewers should look for violations of this invariant.  Red flags include:

- a cache field used to decide whether a terminal is semantically allowed;
- a scratch map whose previous contents affect a later result after `clear_all`;
- a generation update omitted after a mutation;
- a cached mask copied without checking generation;
- a clone implementation that accidentally shares mutable cache data.

The file split does not prove the invariant, but it gives reviewers a smaller
surface to inspect.
