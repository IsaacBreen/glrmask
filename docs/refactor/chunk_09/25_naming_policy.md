# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Naming policy

### Use `advance`, not `parser`, for one-terminal stack execution

A parser is a complete recognizer for a grammar.  The old module did not own the entire parser; it owned the one-terminal transition relation over GSS states.  `advance` is the right name.

### Use `can`, not `may`, for exact predicates

`may` suggests a conservative over-approximation.  The predicate is exact after guarded stack-shift evaluation.  `can` is clearer.

### Use `parser::glr`, not `compiler::glr`

GLR is a parser-domain backend used by both compile and runtime.  `compiler::glr` should exist only while old paths are being migrated.

### Keep GLR crate-private

Publication-facing users should see `Constraint`, `ConstraintState`, `Vocab`, Mask, Commit, and profiles.  They should not need to instantiate GLR tables directly.
