# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Target parser-domain shape

The long-term parser tree should read as follows:

```text
src/parser/
  mod.rs
  glr/
    mod.rs
    README.md
    accumulator.rs
    labels.rs
    analysis.rs
    analysis/
      profile.rs
      model.rs
      right_recursion.rs
      null_production_inline.rs
      nullable_run_compress.rs
      left_recursion.rs
      reachability_unit_dedup.rs
      normalize.rs
      fixed_point_sets.rs
      tests.rs
    table/
      mod.rs
      action.rs
      row.rs
      build.rs
      options.rs
      optimize.rs
      optimize/
        policy_adapter.rs
        stack_effect_keys.rs
        table_passes.rs
        suffix_quotient.rs
        merged_state_quotient.rs
        unit_reductions.rs
        same_core_merge.rs
        guarded/
          frame_model.rs
          reduce_frame.rs
          action_exploration.rs
          action_materialize.rs
          stack_shift_canonicalization.rs
          stack_shift_canonicalization_tests.rs
    advance/
      mod.rs
      options.rs
      profile.rs
      profile_trace.rs
      entry_points.rs
      fast_paths.rs
      guarded_shifts.rs
      reduce_sources.rs
      deterministic_vstack.rs
      deterministic_profiled.rs
      nondeterministic_profiled.rs
      nondeterministic.rs
      deterministic.rs
      applicability.rs
      applicability_any.rs
      tests.rs
```

## Why this is the right abstraction

A grammar frontend should not know how GLR table optimization works.  A Terminal-DWA builder should not know how runtime stack advancement is profiled.  Runtime commit should not import from a compiler namespace to answer an exact parser-stack question.  `parser::glr` is the mathematical common denominator.

## Subsystem ownership

- `analysis` owns grammar normal forms and set computations.
- `table` owns LR/GLR state/action/goto construction and action-row optimization.
- `advance` owns execution of one terminal against a persistent parser GSS.
- `accumulator` owns path-carried terminal exclusions.
- `labels` owns terminal-label encoding conventions.
