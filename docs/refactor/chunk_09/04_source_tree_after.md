# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## GLR source tree after Chunk 09

```text
src/parser/glr/accumulator.rs
src/parser/glr/advance/applicability.rs
src/parser/glr/advance/applicability_any.rs
src/parser/glr/advance/deterministic.rs
src/parser/glr/advance/deterministic_profiled.rs
src/parser/glr/advance/deterministic_vstack.rs
src/parser/glr/advance/entry_points.rs
src/parser/glr/advance/fast_paths.rs
src/parser/glr/advance/guarded_shifts.rs
src/parser/glr/advance/mod.rs
src/parser/glr/advance/nondeterministic.rs
src/parser/glr/advance/nondeterministic_profiled.rs
src/parser/glr/advance/options.rs
src/parser/glr/advance/profile.rs
src/parser/glr/advance/profile_trace.rs
src/parser/glr/advance/reduce_sources.rs
src/parser/glr/advance/tests.rs
src/parser/glr/analysis/fixed_point_sets.rs
src/parser/glr/analysis/left_recursion.rs
src/parser/glr/analysis/model.rs
src/parser/glr/analysis/normalize.rs
src/parser/glr/analysis/null_production_inline.rs
src/parser/glr/analysis/nullable_run_compress.rs
src/parser/glr/analysis/options.rs
src/parser/glr/analysis/profile.rs
src/parser/glr/analysis/reachability_unit_dedup.rs
src/parser/glr/analysis/right_recursion.rs
src/parser/glr/analysis/tests.rs
src/parser/glr/analysis.rs
src/parser/glr/labels.rs
src/parser/glr/mod.rs
src/parser/glr/table/action.rs
src/parser/glr/table/build.rs
src/parser/glr/table/mod.rs
src/parser/glr/table/optimize/guarded/action_exploration.rs
src/parser/glr/table/optimize/guarded/action_materialize.rs
src/parser/glr/table/optimize/guarded/frame_model.rs
src/parser/glr/table/optimize/guarded/reduce_frame.rs
src/parser/glr/table/optimize/guarded/stack_shift_canonicalization.rs
src/parser/glr/table/optimize/guarded/stack_shift_canonicalization_tests.rs
src/parser/glr/table/optimize/guarded_stack_effects.rs
src/parser/glr/table/optimize/merged_state_quotient.rs
src/parser/glr/table/optimize/policy_adapter.rs
src/parser/glr/table/optimize/same_core_merge.rs
src/parser/glr/table/optimize/stack_effect_keys.rs
src/parser/glr/table/optimize/suffix_quotient.rs
src/parser/glr/table/optimize/table_passes.rs
src/parser/glr/table/optimize/unit_reductions.rs
src/parser/glr/table/optimize.rs
src/parser/glr/table/options.rs
src/parser/glr/table/row.rs
```

## Reading notes

The smallest facade files are intentional.  For example, `analysis.rs`, `advance/mod.rs`, and `table/optimize.rs` use textual includes to expose subtopics without prematurely changing helper visibility.  When the compile-repair phase begins, these textual fragments are the natural units to convert into true Rust modules.
