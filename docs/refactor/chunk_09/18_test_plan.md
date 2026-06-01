# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Test plan for GLR parser/table cleanup

### Unit tests to prioritize

1. Analysis normalization tests.
2. GLR table construction tests.
3. Stack-shift canonicalization tests.
4. Parser advance fast-path equivalence tests.
5. JSON Schema importer tests that indirectly exercise GLR table construction.
6. Runtime commit tests that exercise `stack_can_advance_on`.

### New tests to add later

#### Exact applicability

Construct a table with guarded stack shifts where the top state supports a terminal but lower-stack guards reject it.  Assert `stack_can_advance_on` returns false.

#### Admission vs execution rows

Construct a table where the execution row has an action but the admission bit is cleared.  Assert `stack_can_advance_on` returns false.

#### Compatibility shim

A small crate-internal test may import both `parser::glr::table::GLRTable` and `compiler::glr::table::GLRTable` and assert they name the same type.  This test can be removed when the shim is removed.

#### Optimizer remap consistency

After a state merge, assert that action targets, goto targets, advance rows, and forwarded-shift pairs are all remapped consistently.

### Benchmark plan

Do not benchmark until all compile repairs are done.  The interesting benchmarks are JSON Schema compile time, Parser DWA build time, commit throughput, and mask throughput.  Chunk 09 should be performance-neutral because it is mostly names, file movement, and policy extraction.
