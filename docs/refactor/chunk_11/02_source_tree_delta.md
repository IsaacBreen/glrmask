# Source tree delta

Before this chunk, the runtime namespace had a few very large flat files:

```text
runtime/state.rs
runtime/mask/mod.rs
runtime/commit/mod.rs
```

After this chunk, the runtime online state has first-order submodules:

```text
runtime/
  README.md
  mod.rs
  artifact/
  bitmask_ops.rs
  commit/
    README.md
    mod.rs
    mask_assert.rs
    options.rs
    parser_advance.rs
    profile.rs
    template_advance.rs
    tokenizer_scan.rs
    token_lookup.rs
  mask/
    README.md
    mod.rs
    bitset.rs
    constants.rs
    dense_acc.rs
    profile.rs
    queue.rs
  mask_mapping.rs
  state/
    README.md
    mod.rs
    cache.rs
    force.rs
    inspect.rs
    scratch.rs
```

Interpretation of the new directories:

- `state/` is the live prefix object.
- `mask/` is a query over that object.
- `commit/` is a transition on that object.
- `artifact/` is the immutable compiled object the query/transition borrow.
- `mask_mapping.rs` remains an output-materialization subsystem and is a good
  candidate for a later split.

This is not merely cosmetic.  It prevents a common conceptual mistake: treating
`ConstraintState` as a bag of runtime implementation tricks.  `ConstraintState`
represents a mathematical frontier.  The cache and scratch modules make clear
which fields are not part of the denotation.
