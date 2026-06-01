# `parser::glr`

This directory owns the GLR-specific parser machinery. It is not part of the
paper's public terminology in the same way that Terminal DWA and Parser DWA are,
but it supplies the stack-effect semantics used to build and query those objects.

The important boundary is temporal:

- `analysis/` and `table/` are compile-time construction machinery.
- `advance/` is runtime-used parser-stack transition machinery.
- `accumulator.rs` and `labels.rs` define small parser-domain data carried by
  both sides.

The old path was `compiler::glr`. That path now exists only as a compatibility
shim. New code should import from `parser::glr`.
