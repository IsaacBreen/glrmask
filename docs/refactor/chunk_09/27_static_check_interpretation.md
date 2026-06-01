# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Static checks used for this chunk

The package includes a checks file with simple grep/LOC assertions.  These checks are not a substitute for compilation.  They answer only shape questions:

- Does the new parser namespace exist?
- Is the old compiler namespace reduced to a shim?
- Are stale imports absent from source?
- Were exact predicates renamed?
- Are direct env reads localized to option files?
- Were large GLR files split into fragments?

## How to interpret failures

A failure means the patch does not match the intended shape.  It does not necessarily mean the algorithm is wrong.  Fix shape failures before compiling, because compile errors from the wrong tree shape are noisy and misleading.
