# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Applied source changes

### GLR namespace move

`src/compiler/glr` was moved to `src/parser/glr`.  A small compatibility shim remains at `src/compiler/glr/mod.rs`.

### Public module declaration

`src/lib.rs` now declares:

```rust
pub(crate) mod parser;
```

The module is crate-private because GLR is an implementation detail, not a public API promise.

### Import migration

All source imports found in this tree were migrated from `crate::compiler::glr` to `crate::parser::glr`, except the compatibility shim itself.

### Parser execution rename

The old module name `parser` was misleading because it did not define the whole parser.  It defined stack advancement.  The module is now:

```text
src/parser/glr/advance/
```

### Exact predicate rename

The exact applicability predicates were renamed:

```text
stack_may_advance_on     -> stack_can_advance_on
stack_may_advance_on_any -> stack_can_advance_on_any
```

### Option isolation

Direct environment-variable reads were moved out of core algorithm bodies into:

```text
src/parser/glr/analysis/options.rs
src/parser/glr/advance/options.rs
src/parser/glr/table/options.rs
```

### File metrics after split

| File | Lines |
| --- | ---: |
| `src/parser/glr/accumulator.rs` | 97 |
| `src/parser/glr/advance/applicability.rs` | 38 |
| `src/parser/glr/advance/applicability_any.rs` | 85 |
| `src/parser/glr/advance/deterministic.rs` | 121 |
| `src/parser/glr/advance/deterministic_profiled.rs` | 147 |
| `src/parser/glr/advance/deterministic_vstack.rs` | 180 |
| `src/parser/glr/advance/entry_points.rs` | 229 |
| `src/parser/glr/advance/fast_paths.rs` | 283 |
| `src/parser/glr/advance/guarded_shifts.rs` | 236 |
| `src/parser/glr/advance/mod.rs` | 68 |
| `src/parser/glr/advance/nondeterministic.rs` | 122 |
| `src/parser/glr/advance/nondeterministic_profiled.rs` | 177 |
| `src/parser/glr/advance/options.rs` | 60 |
| `src/parser/glr/advance/profile.rs` | 70 |
| `src/parser/glr/advance/profile_trace.rs` | 104 |
| `src/parser/glr/advance/reduce_sources.rs` | 56 |
| `src/parser/glr/advance/tests.rs` | 374 |
| `src/parser/glr/analysis/fixed_point_sets.rs` | 152 |
| `src/parser/glr/analysis/left_recursion.rs` | 198 |
| `src/parser/glr/analysis/model.rs` | 242 |
| `src/parser/glr/analysis/normalize.rs` | 189 |
| `src/parser/glr/analysis/null_production_inline.rs` | 230 |
| `src/parser/glr/analysis/nullable_run_compress.rs` | 445 |
| `src/parser/glr/analysis/options.rs` | 10 |
| `src/parser/glr/analysis/profile.rs` | 53 |
| `src/parser/glr/analysis/reachability_unit_dedup.rs` | 646 |
| `src/parser/glr/analysis/right_recursion.rs` | 413 |
| `src/parser/glr/analysis/tests.rs` | 210 |
| `src/parser/glr/analysis.rs` | 20 |
| `src/parser/glr/labels.rs` | 17 |
| `src/parser/glr/mod.rs` | 19 |
| `src/parser/glr/table/action.rs` | 139 |
| `src/parser/glr/table/build.rs` | 1093 |
| `src/parser/glr/table/mod.rs` | 888 |
| `src/parser/glr/table/optimize/guarded/action_exploration.rs` | 175 |
| `src/parser/glr/table/optimize/guarded/action_materialize.rs` | 77 |
| `src/parser/glr/table/optimize/guarded/frame_model.rs` | 123 |
| `src/parser/glr/table/optimize/guarded/reduce_frame.rs` | 93 |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization.rs` | 70 |
| `src/parser/glr/table/optimize/guarded/stack_shift_canonicalization_tests.rs` | 718 |
| `src/parser/glr/table/optimize/guarded_stack_effects.rs` | 11 |
| `src/parser/glr/table/optimize/merged_state_quotient.rs` | 436 |
| `src/parser/glr/table/optimize/policy_adapter.rs` | 18 |
| `src/parser/glr/table/optimize/same_core_merge.rs` | 196 |
| `src/parser/glr/table/optimize/stack_effect_keys.rs` | 31 |
| `src/parser/glr/table/optimize/suffix_quotient.rs` | 533 |
| `src/parser/glr/table/optimize/table_passes.rs` | 792 |
| `src/parser/glr/table/optimize/unit_reductions.rs` | 204 |
| `src/parser/glr/table/optimize.rs` | 28 |
| `src/parser/glr/table/options.rs` | 98 |
| `src/parser/glr/table/row.rs` | 735 |
