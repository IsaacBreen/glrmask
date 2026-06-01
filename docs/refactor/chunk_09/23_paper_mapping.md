# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Mapping to the paper

The paper talks about stack-effect recognizers abstractly.  It should not require the reader to care whether the backend parser is GLR, LR, Earley-like, or something else.  In this implementation, GLR is the parser backend used to produce and execute those stack effects.

## Where GLR appears in paper terms

- Terminal stack-effect recognizers: derived from the GLR table and advance semantics.
- Parser DWA: compiled from stack-effect behavior over parser stack prefixes.
- Commit: executes completed terminal sequences against runtime parser stacks using `advance_stacks`.
- Mask: queries parser stack prefixes against the compiled Parser DWA; it depends indirectly on GLR table semantics through compilation.

## Correct prose distinction

Do not write: “the compiler's GLR parser handles runtime commit.”

Write: “the parser backend supplies a stack-advance relation used both during compilation and at runtime.”

That sentence is both more abstract and more accurate.
