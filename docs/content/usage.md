GLRMask has four core types:

- `Grammar`: a grammar.
- `Module`: a grammar that can be compiled before its externs are bound, so the final compile only links the parts.
- `Constraint`: a grammar compiled for a vocabulary, ready to use.
- `ConstraintState`: the state of one generated sequence.
