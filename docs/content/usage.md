GLRMask has four core types:

- `Grammar`: a grammar, written as source text or built from an expression.
- `Module`: a grammar plus its extern bindings and any precompiled work. Can be saved and loaded.
- `Constraint`: a grammar compiled for a vocabulary, ready to use.
- `ConstraintState`: the state of one generated sequence.
