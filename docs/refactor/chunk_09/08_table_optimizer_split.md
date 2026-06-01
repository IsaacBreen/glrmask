# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Optimizer split

`table/optimize.rs` is now a small textual facade.  It includes fragments representing distinct table-optimization ideas.

## Fragments

- `policy_adapter.rs`: bridges the table option object into the historical function names used inside the pass code.
- `stack_effect_keys.rs`: hash/equality keys for symbolic stack-effect exploration.
- `table_passes.rs`: `GLRTable` methods that schedule optimization passes.
- `suffix_quotient.rs`: quotienting of recognizer-equivalent stack suffixes.
- `merged_state_quotient.rs`: state-subset merging and synthetic row construction.
- `guarded/frame_model.rs`: symbolic stack-effect frame representation.
- `guarded/reduce_frame.rs`: reduction composition over symbolic frames.
- `guarded/action_exploration.rs`: recursive exploration of action semantics into stack effects.
- `guarded/action_materialize.rs`: normalization back into table actions.
- `guarded/stack_shift_canonicalization.rs`: concrete stack-shift canonicalization.
- `unit_reductions.rs`: unit-reduction inlining.
- `same_core_merge.rs`: LR-state merging with common cores.

## Why not true modules yet?

The optimizer has many private helper dependencies.  Splitting it into true modules in the same pass would require hundreds of visibility edits.  That is not mathematically difficult, but it is noisy and risky before compiling.  Textual fragments give reviewers the conceptual split now and defer visibility mechanics to a compile-repair pass.

## Critical optimization invariants

1. Merging states must remap every action target, goto target, advance bit, and forwarded shift consistently.
2. Stack-shift canonicalization must preserve lower-stack observability.
3. Guarded stack-effect construction may add guards but must not remove concrete accepting paths.
4. Unit-reduction inlining must preserve lookahead applicability.
5. Suffix quotienting must not equate stack suffixes that differ under later goto observation.
