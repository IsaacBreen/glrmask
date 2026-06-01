# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Mechanical operations performed

The implementation used mechanical source operations equivalent to:

```text
mv src/compiler/glr src/parser/glr
mv src/parser/glr/parser src/parser/glr/advance
create src/parser/mod.rs
create compatibility src/compiler/glr/mod.rs
replace crate::compiler::glr -> crate::parser::glr
replace glr::parser -> glr::advance
replace stack_may_advance_on -> stack_can_advance_on
split analysis.rs into analysis/* fragments
split advance/mod.rs into advance/* fragments
split table/optimize.rs into optimize/* fragments
```

These operations are deliberately simple.  The mathematical work is choosing the boundaries and names; the mechanical work is moving text.
