# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Review checklist

### Source tree

- [ ] `src/parser/mod.rs` exists.
- [ ] `src/parser/glr/mod.rs` documents analysis/table/advance.
- [ ] `src/compiler/glr/mod.rs` is only a compatibility shim.
- [ ] No source file outside the shim imports `crate::compiler::glr`.
- [ ] No source file imports `glr::parser` as the execution module.

### Naming

- [ ] `stack_can_advance_on` and `stack_can_advance_on_any` are the canonical names.
- [ ] `stack_may_advance_on` is absent.
- [ ] Comments describe these predicates as exact.

### Options

- [ ] Parser advance env reads occur only in `advance/options.rs`.
- [ ] GLR table env reads occur only in `table/options.rs`.
- [ ] Polarity of disable flags is preserved.

### Optimizer

- [ ] `table/optimize.rs` is a facade.
- [ ] Optimizer fragments are named by mathematical pass, not by temporary abbreviations.
- [ ] Guarded stack-effect fragments separate symbolic frames from materialization.

### Analysis

- [ ] Analysis fragments distinguish normalization from fixed-point set computation.
- [ ] Tests remain physically near analysis but not mixed into the main facade.

### Deferred repair

- [ ] Do not rustfmt or compile until the intended shape is accepted.
- [ ] During compile repair, preserve this source tree unless the compiler reveals a real dependency mistake.
