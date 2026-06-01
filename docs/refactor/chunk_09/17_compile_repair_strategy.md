# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Compile-repair strategy for this chunk

When the user eventually says to compile, repair in this order:

1. Resolve module path mistakes caused by the namespace move.
2. Resolve missing imports from textual fragments.
3. Resolve duplicate imports created by includes.
4. Resolve `pub(crate)` / private visibility issues only if the compiler proves they are necessary.
5. Resolve warnings from unused imports created by compatibility shims.
6. Only after typechecking, run rustfmt.
7. Only after rustfmt, run tests.
8. Only after tests, run benchmarks.

## Do not do this prematurely

Do not fold the textual fragments back into monolithic files just because the first compile produces an error.  The source split is the point of this chunk.  Fix imports/visibility instead.

## Expected kinds of errors

- Missing `super::*` imports in included fragments.
- Unused compatibility imports under `#![deny(warnings)]`.
- Duplicate `mod tests` if two included optimizer test fragments define the same module name in the same scope.
- Paths in old docs or tests referring to `compiler::glr`.

## Repair preference

Prefer adding explicit imports to fragments over relying on huge wildcard imports.  The final post-repair state should make dependencies clearer than this initial structural patch.
