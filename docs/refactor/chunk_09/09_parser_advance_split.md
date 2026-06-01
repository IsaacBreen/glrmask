# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Parser advance split

The old `compiler::glr::parser::mod.rs` mixed entry points, tracing, fast paths, deterministic execution, nondeterministic execution, guarded shifts, tests, and exact applicability predicates.  The module is now `parser::glr::advance` and its body is split into fragments.

## Fragment responsibilities

- `options.rs`: stack-advance policy and env compatibility.
- `profile.rs`: public/profile structs already re-exported by the API facade.
- `profile_trace.rs`: trace construction helpers.
- `entry_points.rs`: public crate-private advance entry points.
- `fast_paths.rs`: special-case stack-advance paths.
- `guarded_shifts.rs`: guarded stack-effect execution.
- `reduce_sources.rs`: reduction source enumeration.
- `deterministic_vstack.rs`: virtual-stack deterministic advancement.
- `deterministic_profiled.rs`: profiled deterministic advancement.
- `nondeterministic_profiled.rs`: profiled nondeterministic advancement.
- `nondeterministic.rs`: non-profiled nondeterministic advancement.
- `deterministic.rs`: non-profiled deterministic advancement.
- `applicability.rs`: exact one-terminal applicability predicate.
- `applicability_any.rs`: exact set-valued applicability predicate plus finish check.
- `tests.rs`: existing parser-advance tests.

## Exactness of `can_advance`

The old `may_advance` name suggested a conservative approximation.  The implementation comments already said it was exact.  The new name makes the predicate match its semantics.  This matters because delayed longest-match exclusions and partial lexer states depend on exact parser admission: a false positive can admit invalid tokens, while a false negative can mask all valid continuations.
