# Chunk 03 file-level surgery

| File | Change |
| --- | --- |
| `src/compile/mod.rs` | Declares compile-object modules, making `pipeline`, `options`, `profiling`, `thread_pool`, and `tokenizer` peers of `terminal_dwa`, `scan_relation`, and `parser_dwa`. |
| `src/compile/options.rs` | Centralizes environment-backed compile decisions, including boolean flags, tokenizer-detail profiling, DWA/CanMatch reconciliation mode, and compile thread count. |
| `src/compile/profiling.rs` | Defines `CompilePhaseProfile`, profile summary rendering, explicit profile sinks, and template profile emission. |
| `src/compile/thread_pool.rs` | Owns the optional compile-specific rayon thread pool. |
| `src/compile/tokenizer.rs` | Owns grammar-terminal-to-tokenizer construction and documents tokenizer/vocab independence. |
| `src/compile/pipeline/mod.rs` | Small phase orchestrator: prepare grammar, analyze, build terminal/scan/template artifacts, reconcile, finalize. |
| `src/compile/pipeline/phases.rs` | Defines the ordered phase vocabulary and phase labels/descriptions. |
| `src/compile/pipeline/context.rs` | Defines typed intermediate outputs for phase boundaries. |
| `src/compile/pipeline/analysis.rs` | Builds tokenizer plus parser/table facts and computes disallowed follows. |
| `src/compile/pipeline/terminal_scan.rs` | Builds shared classification support, Terminal DWA, and scan relation. |
| `src/compile/pipeline/templates.rs` | Builds stack-effect templates and commit-specialized template DFAs. |
| `src/compile/pipeline/reconcile.rs` | Builds Parser DWA and reconciles Parser-DWA/CanMatch/Terminal-DWA ID spaces. |
| `src/compile/pipeline/finalize.rs` | Assembles runtime `Constraint` and rebuilds runtime caches. |
| `src/compile/pipeline/counts.rs` | Holds interned-range accounting utilities for reconciliation. |
| `src/compiler/compile.rs` | Becomes a compatibility facade into `compile::pipeline` and `compile::profiling`. |
| `src/compiler/mod.rs` | Stops declaring the old `compiler::pipeline` implementation as a central module. |
| `src/compiler/pipeline.rs` | Is reduced to a deprecated compatibility shim if included again later; it no longer contains the implementation. |
| `src/runtime/mod.rs` | Re-exports `TemplateDfasByTerminal` crate-privately so finalization can name the runtime template vector type. |
| `src/import/mod.rs` | Imports compile/profile entry points from the new compile namespace. |
