# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Risk register

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Textual include fragments accidentally duplicate a test module name. | Medium | Compile-repair pass should rename test modules or move tests into real submodules. |
| Compatibility shim triggers unused import warnings under `deny(warnings)`. | Medium | Add narrow allow attributes or remove unused re-export paths after import migration. |
| Env option extraction changes disable-flag polarity. | High | Compare each old flag against the new `from_env` implementation. |
| Runtime commit import path accidentally stays on `compiler::glr`. | Medium | Static check searches source imports. |
| `stack_can_advance_on` rename misses a profile field named `may_advance_ns`. | Low | Profile field can remain until a profiling terminology chunk; it is measurement vocabulary, not the exact predicate name. |
| Moving GLR under `parser` makes readers think GLR is a public parser backend API. | Low | Keep module crate-private and docs explicit. |
| Optimizer fragments look like true modules but share one textual namespace. | Medium | Docs and facade comments state this explicitly. |
