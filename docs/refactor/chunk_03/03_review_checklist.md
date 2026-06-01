# Chunk 03 review checklist

## Shape checks

- [ ] `src/compile/pipeline/mod.rs` is under 250 lines.
- [ ] `src/compile/pipeline/mod.rs` contains only orchestration and public phase-graph entry points.
- [ ] `src/compile/pipeline/context.rs` names intermediate objects.
- [ ] `src/compile/pipeline/phases.rs` lists the compile phases in order.
- [ ] Runtime `Constraint` field initialization is isolated in `src/compile/pipeline/finalize.rs`.
- [ ] Parser-DWA/CanMatch/Terminal-DWA ID reconciliation is isolated in `src/compile/pipeline/reconcile.rs`.
- [ ] Tokenizer construction is isolated in `src/compile/tokenizer.rs`.
- [ ] Compile-thread-pool policy is isolated in `src/compile/thread_pool.rs`.
- [ ] Compile option/environment parsing is isolated in `src/compile/options.rs`.
- [ ] Compile profile rendering is isolated in `src/compile/profiling.rs`.

## Mathematical checks

- [ ] Docs distinguish Terminal DWA from scan relation / CanMatch.
- [ ] Docs distinguish templates as stack-effect recognizers, not LR-specific machinery.
- [ ] Reconciliation docs state that all final runtime weights share one internal coordinate system.
- [ ] Finalization docs state that runtime caches are derived representation, not semantics.
- [ ] Pipeline docs do not imply grammar frontend, tokenizer, parser table, and runtime caches are one monolithic step.

## Mechanical checks

- [ ] Search `src/compile/pipeline` for `std::env`; no matches expected.
- [ ] Search `src/compile/pipeline` for `eprintln!`; no matches expected.
- [ ] Search `src/compiler/pipeline.rs`; it should not contain the old 800+ line implementation.
- [ ] Search for `CompilePhaseProfile`; canonical definition should be in `src/compile/profiling.rs`.
- [ ] Search for `DwaCanMatchMode`; canonical definition should be in `src/compile/options.rs`.
- [ ] Search for `build_tokenizer`; canonical implementation should be in `src/compile/tokenizer.rs`.

## Deferred validation checks

These are not part of this no-compile chunk but must happen later:

- [ ] Run `cargo fmt`.
- [ ] Run `cargo check`.
- [ ] Fix borrow/import fallout without collapsing the phase graph.
- [ ] Run unit tests.
- [ ] Run Python binding import smoke test.
- [ ] Compare benchmark profile-line field names against existing scripts.
- [ ] Compare compile artifact statistics before/after refactor.
