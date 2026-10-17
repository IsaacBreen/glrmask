GLRMask has four core types:

- `Grammar`: the language you want to generate.
- `PreparedGrammar`: vocabulary-specific reusable machinery, which can still have unresolved bindings.
- `Constraint`: a grammar compiled against a specific vocabulary.
- `ConstraintState`: the mutable state for a single generated sequence.
