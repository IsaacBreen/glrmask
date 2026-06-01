# Chunk 03 symbol move table

| Old symbol / responsibility | Old location | New location | Reason | Notes for later chunks |
| --- | --- | --- | --- | --- |
| `compile_owned` | `src/compiler/pipeline.rs` via `compiler::compile` | `src/compile/pipeline/mod.rs` | the orchestration is a compile-object concern, not GLR internals | compatibility shim remains in `compiler::compile` |
| `compile_owned_profiled` | `src/compiler/pipeline.rs` | `src/compile/pipeline/mod.rs` | profile is a report on the phase graph | later can return a richer report type |
| `CompilePhaseProfile` | `src/compiler/pipeline.rs` | `src/compile/profiling.rs` | report shape is independent of orchestration implementation | still intentionally field-compatible |
| `emit_compile_profile_summary` | `src/compiler/pipeline.rs` | `src/compile/profiling.rs` | profile emission is a side effect; side effects are not phase logic | supports sink abstraction |
| `env_flag_enabled` | `src/compiler/pipeline.rs` | `src/compile/options.rs` | option resolution must be centralized | public `CompileOptions` can eventually replace env vars |
| `env_flag_enabled_by_default` | `src/compiler/pipeline.rs` | `src/compile/options.rs` | same | used for pre-reconcile CanMatch compaction |
| `DwaCanMatchMode` | `src/compiler/pipeline.rs` | `src/compile/options.rs` | this is a compile strategy decision, not a local variable | should eventually be an enum field on internal options |
| `compile_thread_count` | `src/compiler/pipeline.rs` | `src/compile/options.rs` | environment-backed compile decision | thread-pool construction moved separately |
| `run_with_compile_thread_pool` | `src/compiler/pipeline.rs` | `src/compile/thread_pool.rs` | execution policy is not phase math | keeps optional private rayon pool |
| `build_tokenizer` | `src/compiler/pipeline.rs` | `src/compile/tokenizer.rs` | tokenizer is an explicit grammar object | now documented as vocab-independent |
| `build_tokenizer_from_exprs` | `src/compiler/pipeline.rs` | `src/compile/tokenizer.rs` | same | kept crate-private |
| `terminal_expr` | `src/compiler/pipeline.rs` | `src/compile/tokenizer.rs` | terminal lowering belongs next to tokenizer construction | remains private |
| `compute_disallowed_follows` | `src/compiler/pipeline.rs` | `src/compile/pipeline/analysis.rs` | derived grammar/table fact | may later move to `compile/parser_facts` |
| interned-range accounting helpers | `src/compiler/pipeline.rs` | `src/compile/pipeline/counts.rs` | reconciliation accounting is independent utility | later can become artifact-size report module |
| `finalize_constraint` | `src/compiler/pipeline.rs` | `src/compile/pipeline/finalize.rs` | runtime layout knowledge should be isolated | finalization owns cache rebuild |
| template profile printing | inline in pipeline | `src/compile/profiling.rs` | profile side-effect centralized | line format preserved |
| old `compiler::pipeline` module | `src/compiler/pipeline.rs` full implementation | tiny compatibility shim / orphan path | prevents old file from continuing as conceptual center | remove after downstream imports are cleaned |
