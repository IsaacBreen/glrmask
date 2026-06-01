# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Glossary

**Admission row**: bitset saying a top parser state has support for a terminal.  Used as a fast exact prefilter before inspecting optimized actions.

**Action row**: map from terminal to execution action.

**Advance**: consuming one completed grammar terminal from a parser GSS.

**GLR table**: optimized parser transition table over states, terminals, actions, and gotos.

**Guarded stack shift**: compact action representing a stack effect plus lower-stack predicates.

**GSS**: graph-structured stack.  A persistent representation of many parser stacks.

**Parser backend**: implementation-specific mechanism that supplies stack-effect semantics.  Here it is GLR.

**Stack applicability**: exact predicate that a stack can advance on a terminal.

**Textual fragment**: a physical source file included into another module with `include!`, preserving single-module privacy while improving file readability.
