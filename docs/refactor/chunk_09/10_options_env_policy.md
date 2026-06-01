# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Environment policy after Chunk 09

This chunk does not remove environment-variable compatibility.  It localizes it.

## New local option objects

```rust
ParserAdvanceOptions
GLRTableOptions
```

These are crate-private because they are not yet part of the public API.  Their job is to stop algorithm bodies from directly reading process-global environment variables.

## Parser advance options

- `disable_guarded_stack_to_stacks_fallback`
- `disable_stack_effect_to_stacks_fallback`
- `trace_enabled`

## GLR table options

- `default_action_rows`
- `stack_shift_predecessor_canonicalization`
- `recognizer_suffix_quotient`
- `recognizer_suffix_quotient_max_states`
- `recognizer_suffix_quotient_max_alts`
- `recognizer_suffix_quotient_max_width`
- `max_guarded_stack_effects`
- `unit_reduction_inlining`
- `profile_table_build`

## Deferred decision

A later configuration chunk should decide whether these options become fields of public `CompileOptions` / `RuntimeOptions`, stay private env compatibility hooks, or become explicit benchmark-only features.
