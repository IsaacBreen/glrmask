# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Proof obligations introduced or clarified by Chunk 09

### Namespace move

The namespace move has no semantic effect if every import points to the same moved definitions and the compatibility shim contains no behavior.  Review obligation: verify the shim only re-exports.

### Textual source split

The include-based splits are semantics-preserving because each fragment is textually included into the same module scope.  Review obligation: verify fragment order does not create duplicate item definitions and that no item was dropped while slicing.

### `may` to `can` rename

The predicate rename is semantics-preserving if every call site was migrated and no old alias with different behavior remains.  Review obligation: search for `stack_may_advance` and confirm zero source hits.

### Options extraction

Moving env reads into option objects is semantics-preserving if default values and env variable names match the old behavior.  Review obligation: compare each old env variable name and polarity against the new option field.

### Parser/table boundary

Moving GLR out of `compiler` is mathematically correct only if runtime users import the same relation.  Review obligation: runtime commit/mask should import `parser::glr::advance` and `parser::glr::table`, not `compiler::glr`.
