# Chunk 03 application instructions

These instructions are intentionally mechanical.  They tell a basic maintainer how to apply or check the chunk without needing to understand all algorithms.

## Step 1: create new compile support modules

Create these files:

```text
src/compile/options.rs
src/compile/profiling.rs
src/compile/thread_pool.rs
src/compile/tokenizer.rs
```

Paste the implementations from this package.  Confirm that `src/compile/options.rs` contains `DwaCanMatchMode`; `src/compile/profiling.rs` contains `CompilePhaseProfile`; `src/compile/thread_pool.rs` contains `run_with_compile_thread_pool`; and `src/compile/tokenizer.rs` contains `build_tokenizer`.

## Step 2: create the pipeline directory

Create:

```text
src/compile/pipeline/
```

Inside it create:

```text
analysis.rs
context.rs
counts.rs
finalize.rs
mod.rs
phases.rs
reconcile.rs
templates.rs
terminal_scan.rs
```

Do not skip `context.rs`.  It is the point of the chunk.  The code must have named intermediate objects, not just smaller functions.

## Step 3: update module declarations

In `src/compile/mod.rs`, declare:

```rust
pub(crate) mod options;
pub(crate) mod pipeline;
pub(crate) mod profiling;
pub(crate) mod thread_pool;
pub(crate) mod tokenizer;
```

Keep existing declarations for `terminal_dwa`, `scan_relation`, and `parser_dwa`.

## Step 4: shrink compiler-facing compatibility modules

Change `src/compiler/compile.rs` into a facade that re-exports the new compile and profiling entry points.  Leave `prepare_vocab_for_compile` there so existing diagnostics keep working.

Change `src/compiler/mod.rs` so it no longer declares the old `pipeline` module as the implementation center.

## Step 5: update import frontend paths

In `src/import/mod.rs`, change imports from `crate::compiler::compile` and `crate::compiler::compile_owned` to `crate::compile::pipeline` and `crate::compile::profiling`.

## Step 6: centralize profile helpers

Make `src/compile/terminal_dwa/types.rs::compile_profile_enabled` delegate to `crate::compile::profiling::compile_profile_enabled()`.  Make `src/compile/scan_relation/profile.rs::profile_summary_enabled` delegate to `crate::compile::profiling::compile_profile_summary_enabled()`.

## Step 7: runtime template-vector type

Expose `TemplateDfasByTerminal` crate-privately from `src/runtime/mod.rs`, because finalization needs to name the runtime vector type without reaching into `runtime::artifact` directly.

## Step 8: do not compile yet

This chunk is intentionally structural.  Do not start cargo error-chasing until the larger sequence of shape refactors has been applied.  If someone compiles too early, they will be tempted to undo architecture in order to silence immediate import/borrow warnings.

## Step 9: review the shape

The pass is successful if a reader can open `src/compile/pipeline/mod.rs` and understand the whole compile story in under one minute.  The detailed algorithms are allowed to remain complex, but the orchestration must read as a phase graph.

## Step 10: record deferred work

The following are deliberately deferred:

- fully typed public `CompileOptions`,
- structured compile report replacing flat `CompilePhaseProfile`,
- restoring phase-level parallel execution with local profile deltas,
- moving all terminal-DWA environment flags out of terminal-DWA internals,
- splitting GLR table internals from parser stack-effect contracts,
- compile/test/rustfmt cleanup.
