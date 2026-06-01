# Static checks explained

The package includes static checks rather than compile checks.  They are shape
checks:

- Does `src/runtime/state/` exist?
- Is the old `src/runtime/state.rs` gone?
- Are Commit helper modules present?
- Are Mask helper modules present?
- Did the `end_state_may_advance` name disappear?
- Are direct Commit template-DFA env reads localized?
- Are the largest runtime files identified for later cleanup?

These checks cannot prove the code compiles.  They are intended to catch the
kind of error this chunk cares about: accidentally leaving a concept in the old
monolithic location, or forgetting to create the expected boundary file.

A later compile-repair pass should run:

```text
cargo fmt
cargo check --all-targets
cargo test
maturin develop / python import smoke test
```

but this chunk deliberately does not run those commands.
