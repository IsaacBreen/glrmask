GLRMask has four core types:

- `Grammar`: the source-language definition, without compilation settings.
- `Module`: a reusable, independently configured grammar with bindings and any retained compilation work. It may still have unresolved externs.
- `Constraint`: a fully runnable grammar compiled against a specific vocabulary.
- `ConstraintState`: the mutable state for a single generated sequence.
