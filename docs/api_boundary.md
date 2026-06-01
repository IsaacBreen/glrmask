# Public API boundary

This document records the publication-facing API decision introduced in chunk 01.
The goal is to make the crate read like the paper: the public surface is the
mathematical interaction between a compiled constraint and a generation state;
compiler/runtime machinery remains internal unless explicitly exported for
diagnostics.

## Core objects

| Object | Public name | Meaning |
| --- | --- | --- |
| Vocabulary | `Vocab` | Original token ids mapped to byte strings. The public token-id space is always the user's token-id space. |
| Compiled constraint | `Constraint` | Immutable artifact obtained by compiling one grammar against one `Vocab`. It contains parser, scanner, DWA, and mask-materialization artifacts. |
| Generation state | `ConstraintState` / `State` | Mutable frontier after some byte prefix has been committed. It borrows a `Constraint`. |
| Mask profile | `MaskProfile` | Timing/diagnostic decomposition of the Mask operation. |
| Commit profile | `CommitProfile` | Timing/diagnostic decomposition of the Commit operation. |

## Runtime algebra

The public decoding algebra is intentionally small:

```text
Constraint × ε              -> ConstraintState
ConstraintState             -> Mask over original token ids
ConstraintState × token id  -> ConstraintState'
ConstraintState × bytes     -> ConstraintState'
```

Everything else in the implementation exists to make those maps fast.

In paper terms, `fill_mask` evaluates the current active parser-stack frontier
against the Parser DWA and materializes a Boolean mask over the original
vocabulary.  `commit_token` scans the selected token's bytes, records terminal
boundaries, applies parser advances for completed terminal sequences, and
replaces the state frontier.

## Root exports

The crate root now re-exports the stable facade:

```rust
pub use glrmask::{
    Constraint,
    ConstraintState,
    State,
    Vocab,
    Error,
    Result,
    CompileOptions,
    RuntimeOptions,
    MaskProfile,
    CommitProfile,
};
```

Lower-level trace/profile types remain exported because the current Python
bindings and benchmark harnesses inspect them, but they are grouped under
`api::profiles` in the source.

## Diagnostics boundary

Functions that expose implementation levers are no longer defined in `lib.rs`.
They live under `diagnostics`:

```rust
glrmask::diagnostics::cache::clear_stale_weights();
glrmask::diagnostics::cache::clear_weight_interners();
glrmask::diagnostics::cache::clear_weight_op_caches();

glrmask::diagnostics::frontend::prepare_vocab_for_compile(&vocab);
glrmask::diagnostics::frontend::compile_grammar_def_json(json, &vocab)?;
glrmask::diagnostics::frontend::dump_json_schema_grammar_glrm(schema)?;
```

Root-level compatibility shims for frontend diagnostics remain for now, hidden in
documentation, because existing tests and local harnesses may still call them.
Cache clearing is intentionally not re-exported at root.

## Internal token-space diagnostics

The publication API now uses names that describe the actual quotient map induced
by runtime token-space compaction:

```rust
constraint.internal_to_original_token_ids();
constraint.original_to_internal_token_ids();
state.fill_mask_and_internal_token_ids(&mut mask_words);
```

The benchmark-era aliases have been removed from the publication-facing surface;
local harnesses should migrate to these names.

## What is intentionally not public

The following are implementation objects and should not appear in the root API:

- weight interners and weight operation caches;
- exact Terminal-DWA construction stages;
- Parser-DWA builder internals;
- template-DFA compilation internals;
- compact artifact layout strategies;
- GLR table optimization passes;
- environment-variable configuration knobs.

When these are needed for benchmarking or paper validation, expose them through
`diagnostics`, `api::profiles`, or future feature-gated compatibility modules,
not by expanding the root facade.
